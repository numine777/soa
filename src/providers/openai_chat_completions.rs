//! OpenAI-compatible `/chat/completions` JSON and SSE adapter.

use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::sse::{SseDecoder, SseEvent};
use super::{parse_retry_after, retryable_status};
use crate::model::{
    AdapterError, AdapterFuture, DeltaHandler, FunctionCall, Message, ModelRequest, ModelResponse,
    ProviderAdapter, ToolCall, ToolDefinition, Usage,
};

pub struct OpenAiChatCompletions {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    headers: reqwest::header::HeaderMap,
}

impl OpenAiChatCompletions {
    pub fn new(
        http: reqwest::Client,
        base_url: &str,
        api_key: Option<String>,
        headers: &std::collections::BTreeMap<String, String>,
    ) -> anyhow::Result<Self> {
        let mut header_map = reqwest::header::HeaderMap::new();
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
        let request = ChatRequest::from_canonical(canonical);
        let url = format!("{}/chat/completions", self.base_url);
        let mut builder = self.http.post(&url).json(&request);
        if !self.headers.is_empty() {
            builder = builder.headers(self.headers.clone());
        }
        if let Some(key) = &self.api_key
            && !key.is_empty()
        {
            builder = builder.bearer_auth(key);
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
            let parsed: ChatResponse = serde_json::from_str(&body).map_err(|error| {
                AdapterError::fatal(
                    anyhow::Error::new(error)
                        .context(format!("unexpected response from {url}: {body}")),
                )
            })?;
            if let Some(error) = parsed.error {
                return Err(in_band_error(&error, &url));
            }
            let choice = parsed.choices.into_iter().next().ok_or_else(|| {
                AdapterError::fatal(anyhow!("provider returned no choices from {url}"))
            })?;
            let result = choice
                .message
                .into_canonical(parsed.usage.map(Into::into), choice.finish_reason);
            if let (Some(handler), Some(content)) = (on_delta, result.content.as_deref()) {
                handler(content);
            }
            return Ok(result);
        }

        // SSE framing is independent of HTTP chunk boundaries: one network
        // chunk may contain many events, and one event (or UTF-8 sequence)
        // may span many chunks.
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
                        if apply_sse_event(event, &mut accumulator, on_delta, &url)? {
                            break 'stream;
                        }
                    }
                    if accumulator.finished {
                        break;
                    }
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
                if apply_sse_event(event, &mut accumulator, on_delta, &url)? {
                    break 'stream;
                }
            }
        }

        Ok(accumulator.finish())
    }
}

fn apply_sse_event(
    event: SseEvent,
    accumulator: &mut StreamAccumulator,
    on_delta: Option<DeltaHandler<'_>>,
    url: &str,
) -> std::result::Result<bool, AdapterError> {
    let SseEvent::Data(data) = event else {
        return Ok(true);
    };
    if data.is_empty() {
        return Ok(false);
    }
    let parsed: StreamChunk = serde_json::from_str(&data).map_err(|error| {
        AdapterError::fatal(
            anyhow::Error::new(error)
                .context(format!("unexpected stream event from {url}: {data}")),
        )
    })?;
    // Many OpenAI-compatible gateways report failures as an in-band
    // `{"error": ...}` event on a 200 stream. Surface it as the classified
    // failure it is instead of letting the stream "end before finished".
    if let Some(error) = parsed.error {
        return Err(in_band_error(&error, url));
    }
    accumulator.apply(parsed, on_delta);
    Ok(false)
}

/// Classify an in-band `error` object from a 200 response. Rate limiting,
/// overload, and server-side failures are worth retrying; everything else
/// (invalid request, context overflow, content policy) is fatal.
fn in_band_error(error: &Value, url: &str) -> AdapterError {
    let code = error
        .get("code")
        .map(|code| match code {
            Value::Number(number) => number.to_string(),
            Value::String(text) => text.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default();
    let kind = error
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| error.to_string());
    let retryable = matches!(code.as_str(), "408" | "429" | "500" | "502" | "503" | "504")
        || [kind, code.as_str(), message.as_str()].iter().any(|text| {
            let text = text.to_ascii_lowercase();
            ["rate_limit", "rate limit", "overloaded", "server_error", "timeout", "temporar"]
                .iter()
                .any(|marker| text.contains(marker))
        });
    let detail = if code.is_empty() && kind.is_empty() {
        message
    } else {
        format!("{message} (code: {code}, type: {kind})")
    };
    AdapterError::classified(
        anyhow!("provider error from {url}: {detail}"),
        retryable,
        None,
    )
}

/// A finish reason that means the provider cut the generation short rather
/// than the model stopping deliberately.
fn truncation_reason(finish_reason: Option<String>) -> Option<String> {
    finish_reason
        .filter(|reason| !matches!(reason.as_str(), "stop" | "tool_calls" | "function_call"))
}

impl ProviderAdapter for OpenAiChatCompletions {
    fn name(&self) -> &'static str {
        "openai_chat_completions"
    }

    fn complete<'a>(
        &'a self,
        request: ModelRequest<'a>,
        on_delta: Option<DeltaHandler<'a>>,
    ) -> AdapterFuture<'a> {
        Box::pin(async move { self.complete_inner(request, on_delta).await })
    }
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<MessageWire<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolWire<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<Value>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

impl<'a> ChatRequest<'a> {
    fn from_canonical(request: ModelRequest<'a>) -> Self {
        Self {
            model: request.model,
            messages: request
                .messages
                .iter()
                .map(MessageWire::from_canonical)
                .collect(),
            temperature: request.sampling.temperature,
            top_p: request.sampling.top_p,
            max_tokens: request.sampling.max_tokens,
            tools: request.tools.iter().map(ToolWire::from_canonical).collect(),
            tool_choice: request.constraints.tool_choice.map(|choice| match choice {
                crate::model::ToolChoice::Any => Value::String("required".to_string()),
                crate::model::ToolChoice::Tool(name) => serde_json::json!({
                    "type": "function",
                    "function": { "name": name },
                }),
            }),
            response_format: request.constraints.output_schema.map(|schema| {
                serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": schema,
                        "strict": true,
                    },
                })
            }),
            stream: request.stream,
            stream_options: request.stream.then_some(StreamOptions {
                include_usage: true,
            }),
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
enum MessageWire<'a> {
    System {
        content: &'a str,
    },
    User {
        content: &'a str,
    },
    Assistant {
        content: Option<&'a str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCallWire<'a>>>,
    },
    Tool {
        content: &'a str,
        tool_call_id: &'a str,
    },
}

impl<'a> MessageWire<'a> {
    fn from_canonical(message: &'a Message) -> Self {
        match message {
            Message::System { content } => Self::System { content },
            Message::User { content } => Self::User { content },
            Message::Assistant {
                content,
                tool_calls,
                // The chat-completions protocol has no reasoning replay;
                // provider-opaque reasoning payloads are dropped here.
                reasoning: _,
            } => Self::Assistant {
                content: content.as_deref(),
                tool_calls: tool_calls
                    .as_ref()
                    .map(|calls| calls.iter().map(ToolCallWire::from_canonical).collect()),
            },
            Message::Tool {
                content,
                tool_call_id,
            } => Self::Tool {
                content,
                tool_call_id,
            },
        }
    }
}

#[derive(Serialize)]
struct ToolCallWire<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    kind: &'static str,
    function: FunctionCallWire<'a>,
}

impl<'a> ToolCallWire<'a> {
    fn from_canonical(call: &'a ToolCall) -> Self {
        Self {
            id: &call.id,
            kind: "function",
            function: FunctionCallWire {
                name: &call.function.name,
                arguments: encode_arguments(&call.function.arguments),
            },
        }
    }
}

#[derive(Serialize)]
struct FunctionCallWire<'a> {
    name: &'a str,
    /// The chat-completions wire format carries arguments as a JSON-encoded
    /// string.
    arguments: String,
}

fn encode_arguments(arguments: &Value) -> String {
    match arguments {
        Value::Null => "{}".to_string(),
        // A malformed round-trip survivor: send the raw text back verbatim.
        Value::String(raw) => raw.clone(),
        other => other.to_string(),
    }
}

/// The inverse: parse the wire string into the canonical structured form,
/// preserving malformed output as a raw string so dispatch can report it.
fn decode_arguments(raw: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Null;
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[derive(Serialize)]
struct ToolWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: ToolDefinitionWire<'a>,
}

impl<'a> ToolWire<'a> {
    fn from_canonical(tool: &'a ToolDefinition) -> Self {
        Self {
            kind: "function",
            function: ToolDefinitionWire {
                name: &tool.name,
                description: &tool.description,
                parameters: &tool.parameters,
            },
        }
    }
}

#[derive(Serialize)]
struct ToolDefinitionWire<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a Value,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    usage: Option<UsageWire>,
    /// Some gateways return an error object with HTTP 200.
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

impl ResponseMessage {
    fn into_canonical(self, usage: Option<Usage>, finish_reason: Option<String>) -> ModelResponse {
        ModelResponse {
            content: self.content,
            tool_calls: self.tool_calls.into_iter().map(Into::into).collect(),
            reasoning: None,
            usage,
            truncation: truncation_reason(finish_reason),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ResponseToolCall {
    #[serde(default)]
    id: String,
    function: ResponseFunctionCall,
}

impl From<ResponseToolCall> for ToolCall {
    fn from(call: ResponseToolCall) -> Self {
        Self {
            id: call.id,
            function: FunctionCall {
                name: call.function.name,
                arguments: decode_arguments(&call.function.arguments),
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct ResponseFunctionCall {
    name: String,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct UsageWire {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    prompt_tokens_details: Option<PromptTokensDetails>,
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Default, Deserialize)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u64,
}

#[derive(Debug, Default, Deserialize)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u64,
}

impl From<UsageWire> for Usage {
    fn from(usage: UsageWire) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cache_read_tokens: usage
                .prompt_tokens_details
                .map(|details| details.cached_tokens)
                .unwrap_or_default(),
            reasoning_tokens: usage
                .completion_tokens_details
                .map(|details| details.reasoning_tokens)
                .unwrap_or_default(),
        }
    }
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    usage: Option<UsageWire>,
    /// Some gateways report failures as an in-band event on a 200 stream.
    error: Option<Value>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    id: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

/// Assemble fragmented OpenAI deltas into one canonical response.
#[derive(Default)]
struct StreamAccumulator {
    content: String,
    tool_calls: Vec<PartialToolCall>,
    usage: Option<Usage>,
    finished: bool,
    finish_reason: Option<String>,
}

/// A tool call still being assembled from stream deltas; arguments stay a
/// raw string until the stream ends.
#[derive(Default)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl StreamAccumulator {
    fn apply(&mut self, chunk: StreamChunk, on_delta: Option<DeltaHandler<'_>>) {
        if chunk.usage.is_some() {
            self.usage = chunk.usage.map(Into::into);
        }
        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finished = true;
                self.finish_reason = Some(reason);
            }
            if let Some(text) = choice.delta.content
                && !text.is_empty()
            {
                if let Some(handler) = on_delta {
                    handler(&text);
                }
                self.content.push_str(&text);
            }
            for delta in choice.delta.tool_calls.unwrap_or_default() {
                while self.tool_calls.len() <= delta.index {
                    self.tool_calls.push(PartialToolCall::default());
                }
                let slot = &mut self.tool_calls[delta.index];
                if let Some(id) = delta.id
                    && !id.is_empty()
                {
                    slot.id = id;
                }
                if let Some(function) = delta.function {
                    if let Some(name) = function.name {
                        slot.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments {
                        slot.arguments.push_str(&arguments);
                    }
                }
            }
        }
    }

    fn finish(self) -> ModelResponse {
        ModelResponse {
            content: (!self.content.is_empty()).then_some(self.content),
            reasoning: None,
            tool_calls: self
                .tool_calls
                .into_iter()
                .map(|call| ToolCall {
                    id: call.id,
                    function: FunctionCall {
                        name: call.name,
                        arguments: decode_arguments(&call.arguments),
                    },
                })
                .collect(),
            usage: self.usage,
            truncation: truncation_reason(self.finish_reason),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::model::{RequestConstraints, SamplingParams};

    fn chunk(json: &str) -> StreamChunk {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn canonical_request_maps_to_openai_wire_format() {
        let messages = vec![
            Message::User {
                content: "inspect".into(),
            },
            Message::Assistant {
                content: None,
                reasoning: None,
                tool_calls: Some(vec![ToolCall {
                    id: "c1".into(),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: json!({"path": "x"}),
                    },
                }]),
            },
        ];
        let tools = vec![ToolDefinition {
            name: "read_file".into(),
            description: "Read".into(),
            parameters: json!({"type": "object"}),
        }];
        let request = ChatRequest::from_canonical(ModelRequest {
            model: "coder",
            messages: &messages,
            tools: &tools,
            sampling: SamplingParams {
                temperature: Some(0.2),
                ..Default::default()
            },
            constraints: RequestConstraints::default(),
            stream: true,
        });
        let wire = serde_json::to_value(request).unwrap();
        assert_eq!(wire["model"], "coder");
        assert_eq!(wire["messages"][1]["tool_calls"][0]["type"], "function");
        // The wire format carries arguments as a JSON-encoded string even
        // though the canonical form is structured.
        assert_eq!(
            wire["messages"][1]["tool_calls"][0]["function"]["arguments"],
            r#"{"path":"x"}"#
        );
        assert_eq!(wire["tools"][0]["type"], "function");
        assert_eq!(wire["stream_options"]["include_usage"], true);

        let empty = ChatRequest::from_canonical(ModelRequest {
            model: "coder",
            messages: &messages[..1],
            tools: &[],
            sampling: SamplingParams::default(),
            constraints: RequestConstraints::default(),
            stream: false,
        });
        let empty_wire = serde_json::to_value(empty).unwrap();
        assert!(empty_wire.get("tools").is_none());
        assert!(empty_wire.get("stream").is_none());
        assert!(empty_wire.get("stream_options").is_none());
    }

    #[test]
    fn constraints_map_to_tool_choice_and_response_format() {
        let messages = vec![Message::User {
            content: "go".into(),
        }];
        let schema = json!({"type": "object", "properties": {"answer": {"type": "string"}}});
        let choice = crate::model::ToolChoice::Tool("grep".into());
        let request = ChatRequest::from_canonical(ModelRequest {
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
        assert_eq!(wire["tool_choice"]["function"]["name"], "grep");
        assert_eq!(wire["response_format"]["type"], "json_schema");
        assert_eq!(wire["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            wire["response_format"]["json_schema"]["schema"]["properties"]["answer"]["type"],
            "string"
        );

        let any = crate::model::ToolChoice::Any;
        let request = ChatRequest::from_canonical(ModelRequest {
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
        assert_eq!(wire["tool_choice"], "required");
        assert!(wire.get("response_format").is_none());
    }

    #[test]
    fn non_streaming_response_maps_back_to_canonical_types() {
        let parsed: ChatResponse = serde_json::from_str(
            r#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}}],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}"#,
        )
        .unwrap();
        let choice = parsed.choices.into_iter().next().unwrap();
        let response = choice
            .message
            .into_canonical(parsed.usage.map(Into::into), choice.finish_reason);

        assert!(response.content.is_none());
        assert_eq!(response.tool_calls[0].id, "c1");
        assert_eq!(response.tool_calls[0].function.name, "read_file");
        assert_eq!(response.usage.unwrap().context_tokens(), 11);
        assert!(response.truncation.is_none());
    }

    #[test]
    fn usage_details_map_to_cache_and_reasoning_tokens() {
        let wire: UsageWire = serde_json::from_str(
            r#"{"prompt_tokens":100,"completion_tokens":50,
                "prompt_tokens_details":{"cached_tokens":80,"audio_tokens":0},
                "completion_tokens_details":{"reasoning_tokens":30}}"#,
        )
        .unwrap();
        let usage: Usage = wire.into();
        assert_eq!(usage.cache_read_tokens, 80);
        assert_eq!(usage.reasoning_tokens, 30);

        // Providers that omit the detail objects report zeros.
        let plain: UsageWire =
            serde_json::from_str(r#"{"prompt_tokens":10,"completion_tokens":5}"#).unwrap();
        let usage: Usage = plain.into();
        assert_eq!(usage.cache_read_tokens, 0);
        assert_eq!(usage.reasoning_tokens, 0);
    }

    #[test]
    fn accumulates_content_deltas() {
        let mut accumulator = StreamAccumulator::default();
        accumulator.apply(
            chunk(r#"{"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}"#),
            None,
        );
        accumulator.apply(chunk(r#"{"choices":[{"delta":{"content":"lo"}}]}"#), None);
        assert_eq!(accumulator.content, "Hello");
        assert!(!accumulator.finished);
        accumulator.apply(
            chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#),
            None,
        );
        assert!(accumulator.finished);
        accumulator.apply(
            chunk(
                r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}}"#,
            ),
            None,
        );
        let response = accumulator.finish();
        assert_eq!(response.content.as_deref(), Some("Hello"));
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.usage.unwrap().context_tokens(), 120);
    }

    #[test]
    fn assembles_fragmented_tool_calls() {
        let mut accumulator = StreamAccumulator::default();
        accumulator.apply(
            chunk(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write_file","arguments":""}}]}}]}"#,
            ),
            None,
        );
        accumulator.apply(
            chunk(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}"#,
            ),
            None,
        );
        accumulator.apply(
            chunk(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}},{"index":1,"id":"c2","function":{"name":"read_file","arguments":"{}"}}]}}]}"#,
            ),
            None,
        );
        let response = accumulator.finish();
        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].id, "c1");
        assert_eq!(response.tool_calls[0].function.name, "write_file");
        // Fragments assemble into the structured canonical form.
        assert_eq!(response.tool_calls[0].function.arguments, json!({"path": "x"}));
        assert_eq!(response.tool_calls[1].function.name, "read_file");
        assert_eq!(response.tool_calls[1].function.arguments, json!({}));
        assert!(response.content.is_none());
    }




    #[test]
    fn in_band_error_events_are_surfaced_and_classified() {
        // A rate-limit error on a 200 stream is worth retrying...
        let rate_limited = apply_sse_event(
            SseEvent::Data(
                r#"{"error":{"message":"Rate limit reached","type":"rate_limit_error","code":429}}"#
                    .into(),
            ),
            &mut StreamAccumulator::default(),
            None,
            "http://test/v1",
        )
        .unwrap_err();
        assert!(rate_limited.is_retryable());

        // ...but an invalid request (e.g. context overflow) is fatal, not
        // "stream ended before the generation finished".
        let fatal = apply_sse_event(
            SseEvent::Data(
                r#"{"error":{"message":"maximum context length exceeded","type":"invalid_request_error"}}"#
                    .into(),
            ),
            &mut StreamAccumulator::default(),
            None,
            "http://test/v1",
        )
        .unwrap_err();
        assert!(!fatal.is_retryable());

        // Non-streaming 200 bodies with an error object classify the same way.
        let parsed: ChatResponse =
            serde_json::from_str(r#"{"error":{"message":"overloaded","code":"503"}}"#).unwrap();
        assert!(in_band_error(&parsed.error.unwrap(), "http://test/v1").is_retryable());
    }

    #[test]
    fn abnormal_finish_reasons_are_reported_as_truncation() {
        let mut accumulator = StreamAccumulator::default();
        accumulator.apply(
            chunk(r#"{"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}"#),
            None,
        );
        assert!(accumulator.finished);
        let response = accumulator.finish();
        assert_eq!(response.truncation.as_deref(), Some("length"));

        // Deliberate stops are not truncation.
        assert_eq!(truncation_reason(Some("stop".into())), None);
        assert_eq!(truncation_reason(Some("tool_calls".into())), None);
        assert_eq!(truncation_reason(None), None);
        assert_eq!(
            truncation_reason(Some("content_filter".into())).as_deref(),
            Some("content_filter")
        );
    }

    #[test]
    fn transient_statuses_are_adapter_specific() {
        use reqwest::StatusCode;
        for code in [500, 502, 503, 504, 429, 408] {
            assert!(
                retryable_status(StatusCode::from_u16(code).unwrap()),
                "{code}"
            );
        }
        for code in [400, 401, 403, 404, 422] {
            assert!(
                !retryable_status(StatusCode::from_u16(code).unwrap()),
                "{code}"
            );
        }
    }
}
