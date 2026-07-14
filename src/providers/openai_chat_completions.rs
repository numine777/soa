//! OpenAI-compatible `/chat/completions` JSON and SSE adapter.

use std::time::Duration;

use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::{
    AdapterError, AdapterFuture, DeltaHandler, FunctionCall, Message, ModelRequest, ModelResponse,
    ProviderAdapter, ToolCall, ToolDefinition, Usage,
};

pub struct OpenAiChatCompletions {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl OpenAiChatCompletions {
    pub fn new(http: reqwest::Client, base_url: &str, api_key: Option<String>) -> Self {
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
        }
    }

    async fn complete_inner(
        &self,
        canonical: ModelRequest<'_>,
        on_delta: Option<DeltaHandler<'_>>,
    ) -> std::result::Result<ModelResponse, AdapterError> {
        let request = ChatRequest::from_canonical(canonical);
        let url = format!("{}/chat/completions", self.base_url);
        let mut builder = self.http.post(&url).json(&request);
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
            let choice = parsed.choices.into_iter().next().ok_or_else(|| {
                AdapterError::fatal(anyhow!("provider returned no choices from {url}"))
            })?;
            let result = choice.message.into_canonical(parsed.usage.map(Into::into));
            if let (Some(handler), Some(content)) = (on_delta, result.content.as_deref()) {
                handler(content);
            }
            return Ok(result);
        }

        // SSE: buffer complete lines so UTF-8 split across chunks remains intact.
        let mut accumulator = StreamAccumulator::default();
        let mut buffer: Vec<u8> = Vec::new();
        'stream: loop {
            let chunk = match response.chunk().await {
                Ok(Some(chunk)) => chunk,
                Ok(None) if accumulator.finished => break,
                Ok(None) => {
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
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|&byte| byte == b'\n') {
                let line: Vec<u8> = buffer.drain(..=newline).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim_end();
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim_start();
                if data == "[DONE]" {
                    break 'stream;
                }
                let parsed: StreamChunk = serde_json::from_str(data).map_err(|error| {
                    AdapterError::fatal(
                        anyhow::Error::new(error)
                            .context(format!("unexpected stream chunk from {url}: {data}")),
                    )
                })?;
                if let Some(fragment) = accumulator.apply(parsed)
                    && let Some(handler) = on_delta
                {
                    handler(&fragment);
                }
            }
        }

        Ok(accumulator.finish())
    }
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
                arguments: &call.function.arguments,
            },
        }
    }
}

#[derive(Serialize)]
struct FunctionCallWire<'a> {
    name: &'a str,
    arguments: &'a str,
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
    choices: Vec<Choice>,
    usage: Option<UsageWire>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ResponseToolCall>,
}

impl ResponseMessage {
    fn into_canonical(self, usage: Option<Usage>) -> ModelResponse {
        ModelResponse {
            content: self.content,
            tool_calls: self.tool_calls.into_iter().map(Into::into).collect(),
            usage,
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
                arguments: call.function.arguments,
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
}

impl From<UsageWire> for Usage {
    fn from(usage: UsageWire) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
        }
    }
}

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    usage: Option<UsageWire>,
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
    tool_calls: Vec<ToolCall>,
    usage: Option<Usage>,
    finished: bool,
}

impl StreamAccumulator {
    fn apply(&mut self, chunk: StreamChunk) -> Option<String> {
        let mut fragment = None;
        if chunk.usage.is_some() {
            self.usage = chunk.usage.map(Into::into);
        }
        for choice in chunk.choices {
            if choice.finish_reason.is_some() {
                self.finished = true;
            }
            if let Some(text) = choice.delta.content
                && !text.is_empty()
            {
                self.content.push_str(&text);
                fragment.get_or_insert_with(String::new).push_str(&text);
            }
            for delta in choice.delta.tool_calls.unwrap_or_default() {
                while self.tool_calls.len() <= delta.index {
                    self.tool_calls.push(ToolCall {
                        id: String::new(),
                        function: FunctionCall {
                            name: String::new(),
                            arguments: String::new(),
                        },
                    });
                }
                let slot = &mut self.tool_calls[delta.index];
                if let Some(id) = delta.id
                    && !id.is_empty()
                {
                    slot.id = id;
                }
                if let Some(function) = delta.function {
                    if let Some(name) = function.name {
                        slot.function.name.push_str(&name);
                    }
                    if let Some(arguments) = function.arguments {
                        slot.function.arguments.push_str(&arguments);
                    }
                }
            }
        }
        fragment
    }

    fn finish(self) -> ModelResponse {
        ModelResponse {
            content: (!self.content.is_empty()).then_some(self.content),
            tool_calls: self.tool_calls,
            usage: self.usage,
        }
    }
}

fn retryable_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
}

fn parse_retry_after(response: &reqwest::Response) -> Option<Duration> {
    let value = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?;
    let seconds: u64 = value.trim().parse().ok()?;
    Some(Duration::from_secs(seconds.min(30)))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::model::SamplingParams;

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
                tool_calls: Some(vec![ToolCall {
                    id: "c1".into(),
                    function: FunctionCall {
                        name: "read_file".into(),
                        arguments: r#"{"path":"x"}"#.into(),
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
            stream: true,
        });
        let wire = serde_json::to_value(request).unwrap();
        assert_eq!(wire["model"], "coder");
        assert_eq!(wire["messages"][1]["tool_calls"][0]["type"], "function");
        assert_eq!(wire["tools"][0]["type"], "function");
        assert_eq!(wire["stream_options"]["include_usage"], true);

        let empty = ChatRequest::from_canonical(ModelRequest {
            model: "coder",
            messages: &messages[..1],
            tools: &[],
            sampling: SamplingParams::default(),
            stream: false,
        });
        let empty_wire = serde_json::to_value(empty).unwrap();
        assert!(empty_wire.get("tools").is_none());
        assert!(empty_wire.get("stream").is_none());
        assert!(empty_wire.get("stream_options").is_none());
    }

    #[test]
    fn non_streaming_response_maps_back_to_canonical_types() {
        let parsed: ChatResponse = serde_json::from_str(
            r#"{"choices":[{"message":{"content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}}],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}"#,
        )
        .unwrap();
        let response = parsed
            .choices
            .into_iter()
            .next()
            .unwrap()
            .message
            .into_canonical(parsed.usage.map(Into::into));

        assert!(response.content.is_none());
        assert_eq!(response.tool_calls[0].id, "c1");
        assert_eq!(response.tool_calls[0].function.name, "read_file");
        assert_eq!(response.usage.unwrap().context_tokens(), 11);
    }

    #[test]
    fn accumulates_content_deltas() {
        let mut accumulator = StreamAccumulator::default();
        assert_eq!(
            accumulator.apply(chunk(
                r#"{"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}"#
            )),
            Some("Hel".to_string())
        );
        assert_eq!(
            accumulator.apply(chunk(r#"{"choices":[{"delta":{"content":"lo"}}]}"#)),
            Some("lo".to_string())
        );
        assert!(!accumulator.finished);
        assert_eq!(
            accumulator.apply(chunk(
                r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#
            )),
            None
        );
        assert!(accumulator.finished);
        accumulator.apply(chunk(
            r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}}"#,
        ));
        let response = accumulator.finish();
        assert_eq!(response.content.as_deref(), Some("Hello"));
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.usage.unwrap().context_tokens(), 120);
    }

    #[test]
    fn assembles_fragmented_tool_calls() {
        let mut accumulator = StreamAccumulator::default();
        accumulator.apply(chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write_file","arguments":""}}]}}]}"#,
        ));
        accumulator.apply(chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}"#,
        ));
        accumulator.apply(chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}},{"index":1,"id":"c2","function":{"name":"read_file","arguments":"{}"}}]}}]}"#,
        ));
        let response = accumulator.finish();
        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].id, "c1");
        assert_eq!(response.tool_calls[0].function.name, "write_file");
        assert_eq!(response.tool_calls[0].function.arguments, r#"{"path":"x"}"#);
        assert_eq!(response.tool_calls[1].function.name, "read_file");
        assert!(response.content.is_none());
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
