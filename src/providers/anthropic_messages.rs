//! Anthropic Messages API (`/v1/messages`) JSON and SSE adapter.
//!
//! Protocol differences from chat completions, all contained here:
//! the system prompt is a top-level parameter instead of a message role;
//! tool calls and results are content blocks (`tool_use` in assistant
//! turns, `tool_result` folded into user turns); `max_tokens` is required;
//! auth uses `x-api-key` plus an `anthropic-version` header; streaming is
//! typed events (`message_start`, `content_block_delta`, `message_delta`,
//! `message_stop`) with no `[DONE]` sentinel; and thinking-enabled models
//! return `thinking` blocks that must be replayed verbatim on tool-use
//! turns — carried through the canonical contract's opaque `reasoning`
//! payload.

use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::sse::{SseDecoder, SseEvent};
use super::{parse_retry_after, retryable_status};
use crate::model::{
    AdapterError, AdapterFuture, DeltaHandler, FunctionCall, Message, ModelRequest, ModelResponse,
    ProviderAdapter, ToolCall, Usage,
};

/// The Messages API requires `max_tokens`. When neither the model nor the
/// stage sets one, allow generous room — this is a cap, not a target, and
/// soa streams by default so large values don't risk HTTP timeouts. Kept
/// within every current model's output ceiling.
const DEFAULT_MAX_TOKENS: u32 = 64_000;

const DEFAULT_API_VERSION: &str = "2023-06-01";

pub struct AnthropicMessages {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    headers: reqwest::header::HeaderMap,
}

impl AnthropicMessages {
    pub fn new(
        http: reqwest::Client,
        base_url: &str,
        api_key: Option<String>,
        headers: &std::collections::BTreeMap<String, String>,
    ) -> anyhow::Result<Self> {
        let mut header_map = reqwest::header::HeaderMap::new();
        // The provider's custom headers can override the default version
        // (e.g. to pin a newer `anthropic-version` or add beta flags).
        header_map.insert(
            "anthropic-version",
            reqwest::header::HeaderValue::from_static(DEFAULT_API_VERSION),
        );
        for (key, value) in headers {
            header_map.insert(
                key.parse::<reqwest::header::HeaderName>()
                    .map_err(|_| anyhow!("invalid header name `{key}`"))?,
                value
                    .parse::<reqwest::header::HeaderValue>()
                    .map_err(|_| anyhow!("invalid value for header `{key}`"))?,
            );
        }
        Ok(Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            headers: header_map,
        })
    }

    async fn complete_inner(
        &self,
        canonical: ModelRequest<'_>,
        on_delta: Option<DeltaHandler<'_>>,
    ) -> std::result::Result<ModelResponse, AdapterError> {
        let request = MessagesRequest::from_canonical(canonical);
        let url = format!("{}/messages", self.base_url);
        let mut builder = self.http.post(&url).json(&request).headers(self.headers.clone());
        if let Some(key) = &self.api_key
            && !key.is_empty()
        {
            builder = builder.header("x-api-key", key);
        }

        let mut response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                let retryable = !error.is_builder();
                let source = anyhow::Error::new(error).context(format!("request to {url} failed"));
                return Err(AdapterError::classified(source, retryable, None));
            }
        };

        let status = response.status();
        if !status.is_success() {
            let retry_after = parse_retry_after(&response);
            let body = response.text().await.unwrap_or_default();
            return Err(AdapterError::classified(
                anyhow!("provider returned {status} from {url}: {body}"),
                retryable_status(status),
                retry_after,
            ));
        }

        if !canonical.stream {
            let body = response.text().await.unwrap_or_default();
            let parsed: MessagesResponse = serde_json::from_str(&body).map_err(|error| {
                AdapterError::fatal(
                    anyhow::Error::new(error)
                        .context(format!("unexpected response from {url}: {body}")),
                )
            })?;
            let result = parsed.into_canonical();
            if let (Some(handler), Some(content)) = (on_delta, result.content.as_deref()) {
                handler(content);
            }
            return Ok(result);
        }

        let mut accumulator = StreamAccumulator::default();
        let mut decoder = SseDecoder::default();
        'stream: loop {
            let chunk = match response.chunk().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) => {
                    let events = decoder.finish().map_err(|error| {
                        AdapterError::fatal(
                            anyhow::Error::new(error)
                                .context(format!("invalid UTF-8 in stream from {url}")),
                        )
                    })?;
                    for event in events {
                        if apply_stream_event(event, &mut accumulator, on_delta, &url)? {
                            break 'stream;
                        }
                    }
                    if accumulator.finished {
                        break;
                    }
                    // The Messages API signals completion with message_stop,
                    // not a [DONE] sentinel; EOF before it is an interrupted
                    // generation.
                    return Err(AdapterError::transient(anyhow!(
                        "stream from {url} ended before the generation finished"
                    )));
                }
                Err(error) => {
                    return Err(AdapterError::transient(
                        anyhow::Error::new(error)
                            .context(format!("stream from {url} was interrupted")),
                    ));
                }
            };
            let events = decoder.push(&chunk).map_err(|error| {
                AdapterError::fatal(
                    anyhow::Error::new(error)
                        .context(format!("invalid UTF-8 in stream from {url}")),
                )
            })?;
            for event in events {
                if apply_stream_event(event, &mut accumulator, on_delta, &url)? {
                    break 'stream;
                }
            }
        }

        Ok(accumulator.finish())
    }
}

impl ProviderAdapter for AnthropicMessages {
    fn name(&self) -> &'static str {
        "anthropic_messages"
    }

    fn complete<'a>(
        &'a self,
        request: ModelRequest<'a>,
        on_delta: Option<DeltaHandler<'a>>,
    ) -> AdapterFuture<'a> {
        Box::pin(async move { self.complete_inner(request, on_delta).await })
    }
}

// ---------------------------------------------------------------------------
// Request mapping
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct MessagesRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<Value>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    content: Value,
}

#[derive(Serialize)]
struct WireTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a Value,
}

impl<'a> MessagesRequest<'a> {
    fn from_canonical(request: ModelRequest<'a>) -> Self {
        // System messages hoist to the top-level parameter. The loop puts
        // the system prompt first, but collect defensively from anywhere.
        let mut system_parts: Vec<&str> = Vec::new();
        let mut messages: Vec<WireMessage> = Vec::new();
        for message in request.messages {
            match message {
                Message::System { content } => system_parts.push(content),
                Message::User { content } => messages.push(WireMessage {
                    role: "user",
                    content: Value::String(content.clone()),
                }),
                Message::Assistant {
                    content,
                    tool_calls,
                    reasoning,
                } => {
                    let mut blocks: Vec<Value> = Vec::new();
                    // Thinking blocks replay first and verbatim: models
                    // with thinking enabled reject a tool-use turn whose
                    // thinking was dropped or edited.
                    if let Some(Value::Array(saved)) = reasoning {
                        blocks.extend(saved.iter().cloned());
                    }
                    // The API rejects empty text blocks — only include one
                    // when there is actual text.
                    if let Some(text) = content
                        && !text.is_empty()
                    {
                        blocks.push(json!({ "type": "text", "text": text }));
                    }
                    for call in tool_calls.iter().flatten() {
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.function.name,
                            "input": encode_tool_input(&call.function.arguments),
                        }));
                    }
                    messages.push(WireMessage {
                        role: "assistant",
                        content: Value::Array(blocks),
                    });
                }
                Message::Tool {
                    content,
                    tool_call_id,
                } => {
                    let block = json!({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                    });
                    // Results of one parallel round must arrive in a single
                    // user turn — splitting them degrades the model's
                    // parallel tool use.
                    match messages.last_mut() {
                        Some(WireMessage {
                            role: "user",
                            content: Value::Array(blocks),
                        }) => blocks.push(block),
                        _ => messages.push(WireMessage {
                            role: "user",
                            content: Value::Array(vec![block]),
                        }),
                    }
                }
            }
        }

        Self {
            model: request.model,
            max_tokens: request.sampling.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            system: (!system_parts.is_empty()).then(|| system_parts.join("\n\n")),
            messages,
            tools: request
                .tools
                .iter()
                .map(|tool| WireTool {
                    name: &tool.name,
                    description: &tool.description,
                    input_schema: &tool.parameters,
                })
                .collect(),
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            tool_choice: request.constraints.tool_choice.map(|choice| match choice {
                crate::model::ToolChoice::Any => json!({ "type": "any" }),
                crate::model::ToolChoice::Tool(name) => json!({ "type": "tool", "name": name }),
            }),
            output_config: request
                .constraints
                .output_schema
                .map(|schema| json!({ "format": { "type": "json_schema", "schema": schema } })),
            stream: request.stream,
        }
    }
}

/// `tool_use.input` must be an object. `Null` means no arguments; a raw
/// string is a malformed generation preserved by the canonical layer —
/// wrap it so the round-trip stays valid JSON.
fn encode_tool_input(arguments: &Value) -> Value {
    match arguments {
        Value::Null => json!({}),
        Value::Object(_) => arguments.clone(),
        other => json!({ "input": other }),
    }
}

// ---------------------------------------------------------------------------
// Response mapping
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct MessagesResponse {
    #[serde(default)]
    content: Vec<Value>,
    stop_reason: Option<String>,
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
}

impl WireUsage {
    /// `input_tokens` excludes cached tokens; the canonical `prompt_tokens`
    /// is the full context, so fold both cache figures in. Cache reads are
    /// reported separately for discounted pricing; cache writes bill at a
    /// premium soa does not model (it never sends `cache_control`, so they
    /// are zero for soa's own requests) — counting them at the input rate
    /// under-counts by the 25% write premium at worst.
    fn into_canonical(self) -> Usage {
        Usage {
            prompt_tokens: self.input_tokens
                + self.cache_read_input_tokens
                + self.cache_creation_input_tokens,
            completion_tokens: self.output_tokens,
            cache_read_tokens: self.cache_read_input_tokens,
            reasoning_tokens: 0,
        }
    }
}

/// A stop reason that means the response is incomplete or withheld rather
/// than a deliberate stop (`refusal` included: the content is empty or
/// partial and must not pass as an answer).
fn truncation_reason(stop_reason: Option<String>) -> Option<String> {
    stop_reason.filter(|reason| !matches!(reason.as_str(), "end_turn" | "tool_use" | "stop_sequence"))
}

impl MessagesResponse {
    fn into_canonical(self) -> ModelResponse {
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut reasoning: Vec<Value> = Vec::new();
        for block in self.content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(part) = block.get("text").and_then(Value::as_str) {
                        text.push_str(part);
                    }
                }
                Some("tool_use") => tool_calls.push(ToolCall {
                    id: block
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    function: FunctionCall {
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        arguments: block.get("input").cloned().unwrap_or(Value::Null),
                    },
                }),
                // Preserved verbatim for replay; `thinking` text is not part
                // of the answer.
                Some("thinking" | "redacted_thinking") => reasoning.push(block),
                _ => {}
            }
        }
        ModelResponse {
            content: (!text.is_empty()).then_some(text),
            tool_calls,
            reasoning: (!reasoning.is_empty()).then_some(Value::Array(reasoning)),
            usage: self.usage.map(WireUsage::into_canonical),
            truncation: truncation_reason(self.stop_reason),
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// One in-flight content block, keyed by stream index.
enum PartialBlock {
    Text,
    ToolUse {
        id: String,
        name: String,
        json: String,
    },
    Thinking {
        thinking: String,
        signature: String,
    },
    /// Arrives complete in `content_block_start`.
    Redacted(Value),
    Other,
}

#[derive(Default)]
struct StreamAccumulator {
    content: String,
    blocks: Vec<(usize, PartialBlock)>,
    tool_calls: Vec<ToolCall>,
    reasoning: Vec<Value>,
    usage: Usage,
    saw_usage: bool,
    stop_reason: Option<String>,
    finished: bool,
}

impl StreamAccumulator {
    fn block_mut(&mut self, index: usize) -> Option<&mut PartialBlock> {
        self.blocks
            .iter_mut()
            .find(|(at, _)| *at == index)
            .map(|(_, block)| block)
    }

    fn apply(
        &mut self,
        event: &Value,
        on_delta: Option<DeltaHandler<'_>>,
    ) -> std::result::Result<(), AdapterError> {
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(usage) = event
                    .pointer("/message/usage")
                    .and_then(|raw| serde_json::from_value::<WireUsage>(raw.clone()).ok())
                {
                    let usage = usage.into_canonical();
                    self.usage.prompt_tokens = usage.prompt_tokens;
                    self.usage.cache_read_tokens = usage.cache_read_tokens;
                    self.saw_usage = true;
                }
            }
            Some("content_block_start") => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let block = event.get("content_block").cloned().unwrap_or(Value::Null);
                let partial = match block.get("type").and_then(Value::as_str) {
                    Some("text") => PartialBlock::Text,
                    Some("tool_use") => PartialBlock::ToolUse {
                        id: block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        json: String::new(),
                    },
                    Some("thinking") => PartialBlock::Thinking {
                        thinking: String::new(),
                        signature: String::new(),
                    },
                    Some("redacted_thinking") => PartialBlock::Redacted(block),
                    _ => PartialBlock::Other,
                };
                self.blocks.push((index, partial));
            }
            Some("content_block_delta") => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let Some(delta) = event.get("delta") else {
                    return Ok(());
                };
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            if let Some(handler) = on_delta {
                                handler(text);
                            }
                            self.content.push_str(text);
                        }
                    }
                    Some("input_json_delta") => {
                        if let (
                            Some(PartialBlock::ToolUse { json, .. }),
                            Some(part),
                        ) = (
                            self.block_mut(index),
                            delta.get("partial_json").and_then(Value::as_str),
                        ) {
                            json.push_str(part);
                        }
                    }
                    // Reasoning is replay payload, not answer text — it is
                    // accumulated but never streamed to the caller.
                    Some("thinking_delta") => {
                        if let (
                            Some(PartialBlock::Thinking { thinking, .. }),
                            Some(part),
                        ) = (
                            self.block_mut(index),
                            delta.get("thinking").and_then(Value::as_str),
                        ) {
                            thinking.push_str(part);
                        }
                    }
                    Some("signature_delta") => {
                        if let (
                            Some(PartialBlock::Thinking { signature, .. }),
                            Some(part),
                        ) = (
                            self.block_mut(index),
                            delta.get("signature").and_then(Value::as_str),
                        ) {
                            signature.push_str(part);
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                if let Some(position) = self.blocks.iter().position(|(at, _)| *at == index) {
                    let (_, block) = self.blocks.remove(position);
                    match block {
                        PartialBlock::ToolUse { id, name, json } => {
                            let arguments = if json.trim().is_empty() {
                                Value::Null
                            } else {
                                serde_json::from_str(&json)
                                    .unwrap_or(Value::String(json))
                            };
                            self.tool_calls.push(ToolCall {
                                id,
                                function: FunctionCall { name, arguments },
                            });
                        }
                        PartialBlock::Thinking {
                            thinking,
                            signature,
                        } => self.reasoning.push(json!({
                            "type": "thinking",
                            "thinking": thinking,
                            "signature": signature,
                        })),
                        PartialBlock::Redacted(block) => self.reasoning.push(block),
                        PartialBlock::Text | PartialBlock::Other => {}
                    }
                }
            }
            Some("message_delta") => {
                if let Some(reason) = event
                    .pointer("/delta/stop_reason")
                    .and_then(Value::as_str)
                {
                    self.stop_reason = Some(reason.to_string());
                }
                // Cumulative output count for this message.
                if let Some(output) = event
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64)
                {
                    self.usage.completion_tokens = output;
                    self.saw_usage = true;
                }
            }
            Some("message_stop") => self.finished = true,
            Some("ping") | None => {}
            _ => {}
        }
        Ok(())
    }

    fn finish(mut self) -> ModelResponse {
        // A block the stream never closed (interrupted mid-block but the
        // provider still sent message_stop) is finalized best-effort.
        let indices: Vec<usize> = self.blocks.iter().map(|(at, _)| *at).collect();
        for index in indices {
            let stop = json!({ "type": "content_block_stop", "index": index });
            let _ = self.apply(&stop, None);
        }
        ModelResponse {
            content: (!self.content.is_empty()).then_some(std::mem::take(&mut self.content)),
            tool_calls: std::mem::take(&mut self.tool_calls),
            reasoning: (!self.reasoning.is_empty())
                .then(|| Value::Array(std::mem::take(&mut self.reasoning))),
            usage: self.saw_usage.then_some(self.usage),
            truncation: truncation_reason(self.stop_reason.take()),
        }
    }
}

fn apply_stream_event(
    event: SseEvent,
    accumulator: &mut StreamAccumulator,
    on_delta: Option<DeltaHandler<'_>>,
    url: &str,
) -> std::result::Result<bool, AdapterError> {
    let SseEvent::Data(data) = event else {
        // The Messages API never sends [DONE]; tolerate it anyway.
        return Ok(true);
    };
    if data.is_empty() {
        return Ok(false);
    }
    let parsed: Value = serde_json::from_str(&data).map_err(|error| {
        AdapterError::fatal(
            anyhow::Error::new(error)
                .context(format!("unexpected stream event from {url}: {data}")),
        )
    })?;
    if parsed.get("type").and_then(Value::as_str) == Some("error") {
        let error = parsed.get("error").cloned().unwrap_or(Value::Null);
        return Err(in_band_error(&error, url));
    }
    accumulator.apply(&parsed, on_delta)?;
    Ok(accumulator.finished)
}

/// Classify an Anthropic error object (`{"type": ..., "message": ...}`).
/// Overload, rate limiting, and internal errors retry; invalid requests,
/// auth, and permission errors fail fast.
fn in_band_error(error: &Value, url: &str) -> AdapterError {
    let kind = error
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| error.to_string());
    let retryable = matches!(
        kind.as_str(),
        "overloaded_error" | "rate_limit_error" | "api_error" | "timeout_error"
    );
    AdapterError::classified(
        anyhow!("provider error from {url}: {message} (type: {kind})"),
        retryable,
        None,
    )
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::model::{RequestConstraints, SamplingParams, ToolDefinition};

    fn sse(payload: &str) -> SseEvent {
        SseEvent::Data(payload.to_string())
    }

    #[test]
    fn canonical_request_maps_to_messages_wire_format() {
        let messages = vec![
            Message::System {
                content: "be careful".into(),
            },
            Message::User {
                content: "fix the bug".into(),
            },
            Message::Assistant {
                content: Some("Looking now.".into()),
                reasoning: Some(json!([
                    { "type": "thinking", "thinking": "hmm", "signature": "sig1" }
                ])),
                tool_calls: Some(vec![
                    ToolCall {
                        id: "t1".into(),
                        function: FunctionCall {
                            name: "read_file".into(),
                            arguments: json!({"path": "a.rs"}),
                        },
                    },
                    ToolCall {
                        id: "t2".into(),
                        function: FunctionCall {
                            name: "grep".into(),
                            arguments: json!({"pattern": "bug"}),
                        },
                    },
                ]),
            },
            Message::Tool {
                content: "contents of a.rs".into(),
                tool_call_id: "t1".into(),
            },
            Message::Tool {
                content: "3 matches".into(),
                tool_call_id: "t2".into(),
            },
            Message::User {
                content: "also check b.rs".into(),
            },
        ];
        let tools = vec![ToolDefinition {
            name: "read_file".into(),
            description: "Read a file".into(),
            parameters: json!({"type": "object"}),
        }];
        let request = MessagesRequest::from_canonical(ModelRequest {
            model: "claude-x",
            messages: &messages,
            tools: &tools,
            sampling: SamplingParams::default(),
            constraints: RequestConstraints::default(),
            stream: true,
        });
        let wire = serde_json::to_value(&request).unwrap();

        // System hoists to the top-level parameter; max_tokens is required.
        assert_eq!(wire["system"], "be careful");
        assert_eq!(wire["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(wire["stream"], true);
        assert_eq!(wire["tools"][0]["input_schema"]["type"], "object");

        let turns = wire["messages"].as_array().unwrap();
        assert_eq!(turns.len(), 4); // user, assistant, folded tool results, user
        assert_eq!(turns[0]["role"], "user");
        // Thinking replays first, verbatim; structured input round-trips.
        assert_eq!(turns[1]["content"][0]["type"], "thinking");
        assert_eq!(turns[1]["content"][0]["signature"], "sig1");
        assert_eq!(turns[1]["content"][1]["type"], "text");
        assert_eq!(turns[1]["content"][2]["type"], "tool_use");
        assert_eq!(turns[1]["content"][2]["input"]["path"], "a.rs");
        // Both parallel results fold into ONE user turn.
        assert_eq!(turns[2]["role"], "user");
        assert_eq!(turns[2]["content"].as_array().unwrap().len(), 2);
        assert_eq!(turns[2]["content"][0]["type"], "tool_result");
        assert_eq!(turns[2]["content"][0]["tool_use_id"], "t1");
        assert_eq!(turns[2]["content"][1]["tool_use_id"], "t2");
        // The steered user message stays its own (combinable) turn.
        assert_eq!(turns[3]["role"], "user");
        assert_eq!(turns[3]["content"], "also check b.rs");
    }

    #[test]
    fn assistant_turn_without_text_has_no_empty_text_block() {
        let messages = vec![
            Message::User {
                content: "go".into(),
            },
            Message::Assistant {
                content: None,
                reasoning: None,
                tool_calls: Some(vec![ToolCall {
                    id: "t1".into(),
                    function: FunctionCall {
                        name: "shell".into(),
                        arguments: Value::Null,
                    },
                }]),
            },
        ];
        let request = MessagesRequest::from_canonical(ModelRequest {
            model: "claude-x",
            messages: &messages,
            tools: &[],
            sampling: SamplingParams {
                max_tokens: Some(2048),
                ..Default::default()
            },
            constraints: RequestConstraints::default(),
            stream: false,
        });
        let wire = serde_json::to_value(&request).unwrap();
        assert_eq!(wire["max_tokens"], 2048);
        let blocks = wire["messages"][1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1, "no empty text block");
        assert_eq!(blocks[0]["type"], "tool_use");
        // Null arguments become an empty input object.
        assert_eq!(blocks[0]["input"], json!({}));
    }

    #[test]
    fn constraints_map_to_tool_choice_and_output_config() {
        let messages = vec![Message::User {
            content: "go".into(),
        }];
        let schema = json!({"type": "object"});
        let choice = crate::model::ToolChoice::Tool("grep".into());
        let request = MessagesRequest::from_canonical(ModelRequest {
            model: "m",
            messages: &messages,
            tools: &[],
            sampling: SamplingParams::default(),
            constraints: RequestConstraints {
                tool_choice: Some(&choice),
                output_schema: Some(&schema),
            },
            stream: false,
        });
        let wire = serde_json::to_value(&request).unwrap();
        assert_eq!(wire["tool_choice"], json!({"type": "tool", "name": "grep"}));
        assert_eq!(
            wire["output_config"],
            json!({"format": {"type": "json_schema", "schema": {"type": "object"}}})
        );

        let any = crate::model::ToolChoice::Any;
        let request = MessagesRequest::from_canonical(ModelRequest {
            model: "m",
            messages: &messages,
            tools: &[],
            sampling: SamplingParams::default(),
            constraints: RequestConstraints {
                tool_choice: Some(&any),
                output_schema: None,
            },
            stream: false,
        });
        let wire = serde_json::to_value(&request).unwrap();
        assert_eq!(wire["tool_choice"], json!({"type": "any"}));
        assert!(wire.get("output_config").is_none());
    }

    #[test]
    fn non_streaming_response_maps_back_to_canonical_types() {
        let parsed: MessagesResponse = serde_json::from_str(
            r#"{
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-x",
                "content": [
                    {"type": "thinking", "thinking": "let me look", "signature": "s"},
                    {"type": "text", "text": "Checking "},
                    {"type": "text", "text": "the file."},
                    {"type": "tool_use", "id": "t1", "name": "read_file",
                     "input": {"path": "a.rs"}}
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 100, "output_tokens": 40,
                          "cache_read_input_tokens": 60,
                          "cache_creation_input_tokens": 10}
            }"#,
        )
        .unwrap();
        let response = parsed.into_canonical();
        assert_eq!(response.content.as_deref(), Some("Checking the file."));
        assert_eq!(response.tool_calls[0].id, "t1");
        assert_eq!(
            response.tool_calls[0].function.arguments,
            json!({"path": "a.rs"})
        );
        // input_tokens excludes cache; canonical prompt_tokens is the total.
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 170);
        assert_eq!(usage.completion_tokens, 40);
        assert_eq!(usage.cache_read_tokens, 60);
        // tool_use is a deliberate stop, and thinking is preserved.
        assert!(response.truncation.is_none());
        assert_eq!(response.reasoning.unwrap()[0]["thinking"], "let me look");
    }

    #[test]
    fn streaming_events_accumulate_text_tools_thinking_and_usage() {
        let mut accumulator = StreamAccumulator::default();
        let streamed = std::sync::Mutex::new(String::new());
        let on_delta = |fragment: &str| streamed.lock().unwrap().push_str(fragment);
        let events = [
            r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":90,"output_tokens":1,"cache_read_input_tokens":30}}}"#,
            r#"{"type":"ping"}"#,
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"quiet plan"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Run"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ning."}}"#,
            r#"{"type":"content_block_stop","index":1}"#,
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t9","name":"shell","input":{}}}"#,
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"comm"}}"#,
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"and\":\"ls\"}"}}"#,
            r#"{"type":"content_block_stop","index":2}"#,
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}"#,
            r#"{"type":"message_stop"}"#,
        ];
        let mut done = false;
        for event in events {
            done = apply_stream_event(sse(event), &mut accumulator, Some(&on_delta), "http://t")
                .unwrap();
        }
        assert!(done);
        let response = accumulator.finish();
        // Only answer text streams; thinking stays out of the delta channel.
        assert_eq!(streamed.lock().unwrap().as_str(), "Running.");
        assert_eq!(response.content.as_deref(), Some("Running."));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "t9");
        assert_eq!(
            response.tool_calls[0].function.arguments,
            json!({"command": "ls"})
        );
        let reasoning = response.reasoning.unwrap();
        assert_eq!(reasoning[0]["thinking"], "quiet plan");
        assert_eq!(reasoning[0]["signature"], "sig");
        let usage = response.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 120); // 90 + 30 cached
        assert_eq!(usage.cache_read_tokens, 30);
        assert_eq!(usage.completion_tokens, 25);
        assert!(response.truncation.is_none());
    }

    #[test]
    fn abnormal_stop_reasons_and_errors_classify() {
        // max_tokens and refusal are truncation; normal stops are not.
        assert_eq!(
            truncation_reason(Some("max_tokens".into())).as_deref(),
            Some("max_tokens")
        );
        assert_eq!(
            truncation_reason(Some("refusal".into())).as_deref(),
            Some("refusal")
        );
        assert_eq!(truncation_reason(Some("end_turn".into())), None);
        assert_eq!(truncation_reason(Some("tool_use".into())), None);

        // In-band error events: overloaded retries, invalid request fails.
        let overloaded = apply_stream_event(
            sse(r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#),
            &mut StreamAccumulator::default(),
            None,
            "http://t",
        )
        .unwrap_err();
        assert!(overloaded.is_retryable());
        let invalid = apply_stream_event(
            sse(r#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#),
            &mut StreamAccumulator::default(),
            None,
            "http://t",
        )
        .unwrap_err();
        assert!(!invalid.is_retryable());
    }

    /// End-to-end over a real socket: request headers, wire body, SSE
    /// framing, and canonical mapping — the full `complete_inner` path.
    #[tokio::test]
    async fn live_streaming_request_round_trips() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let server_captured = captured.clone();
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = [0u8; 8192];
            let read = stream.read(&mut buffer).unwrap();
            *server_captured.lock().unwrap() =
                String::from_utf8_lossy(&buffer[..read]).into_owned();
            let events = concat!(
                "event: message_start\n",
                "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n",
                "event: content_block_start\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello from anthropic\"}}\n\n",
                "event: content_block_stop\n",
                "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\n",
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":6}}\n\n",
                "event: message_stop\n",
                "data: {\"type\":\"message_stop\"}\n\n",
            );
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{events}",
                events.len(),
            );
        });

        let adapter = AnthropicMessages::new(
            reqwest::Client::new(),
            &format!("http://{addr}/v1"),
            Some("sk-test".to_string()),
            &std::collections::BTreeMap::new(),
        )
        .unwrap();
        let messages = vec![
            Message::System {
                content: "sys".into(),
            },
            Message::User {
                content: "hi".into(),
            },
        ];
        let streamed = std::sync::Mutex::new(String::new());
        let on_delta = |fragment: &str| streamed.lock().unwrap().push_str(fragment);
        let response = adapter
            .complete(
                ModelRequest {
                    model: "claude-x",
                    messages: &messages,
                    tools: &[],
                    sampling: SamplingParams::default(),
                    constraints: RequestConstraints::default(),
                    stream: true,
                },
                Some(&on_delta),
            )
            .await
            .unwrap();

        assert_eq!(response.content.as_deref(), Some("hello from anthropic"));
        assert_eq!(streamed.lock().unwrap().as_str(), "hello from anthropic");
        let usage = response.usage.unwrap();
        assert_eq!((usage.prompt_tokens, usage.completion_tokens), (12, 6));
        assert!(response.truncation.is_none());

        // The request carried the protocol headers and hoisted system.
        let request = captured.lock().unwrap();
        assert!(request.contains("POST /v1/messages"), "{request}");
        assert!(request.contains("x-api-key: sk-test"), "{request}");
        assert!(request.contains("anthropic-version: 2023-06-01"), "{request}");
        assert!(request.contains(r#""system":"sys""#), "{request}");
        assert!(request.contains(r#""max_tokens":"#), "{request}");
    }

    #[test]
    fn interrupted_stream_without_message_stop_is_incomplete() {
        let mut accumulator = StreamAccumulator::default();
        for event in [
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}"#,
        ] {
            apply_stream_event(sse(event), &mut accumulator, None, "http://t").unwrap();
        }
        assert!(!accumulator.finished);
        // complete_inner turns this into a transient "ended before the
        // generation finished" error; finish() is only reached when
        // message_stop arrived.
    }
}
