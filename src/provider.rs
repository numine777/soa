//! Minimal OpenAI-compatible chat-completions client with tool-call support.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        #[serde(default)]
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments, as produced by the model.
    #[serde(default)]
    pub arguments: String,
}

/// A tool advertised to the model.
#[derive(Debug, Clone, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
}

#[derive(Serialize)]
struct ToolWire<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: &'a ToolFunction,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    // Some local servers reject an empty tools array, so omit it entirely.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolWire<'a>>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
    /// Asks for a final usage chunk when streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

/// Token counts reported by the provider for one request.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
}

impl Usage {
    /// The context the conversation now occupies: everything sent plus
    /// everything generated.
    pub fn context_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// What the model returned for one round-trip.
#[derive(Debug)]
pub struct AssistantTurn {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Real token counts, when the server reports them.
    pub usage: Option<Usage>,
}

/// Callback invoked with each streamed content fragment.
pub type DeltaHandler<'a> = &'a (dyn Fn(&str) + Send + Sync);

// ---------------------------------------------------------------------------
// SSE stream decoding
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
    /// Present on the final chunk when `include_usage` was requested.
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
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

/// Assembles streamed deltas into a complete [`AssistantTurn`]. Tool-call
/// ids, names, and argument JSON all arrive in fragments keyed by `index`.
#[derive(Default)]
struct StreamAccumulator {
    content: String,
    tool_calls: Vec<ToolCall>,
    usage: Option<Usage>,
}

impl StreamAccumulator {
    /// Apply one parsed chunk; returns the content fragment, if any, for
    /// the delta callback.
    fn apply(&mut self, chunk: StreamChunk) -> Option<String> {
        let mut fragment = None;
        if chunk.usage.is_some() {
            self.usage = chunk.usage;
        }
        for choice in chunk.choices {
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
                        kind: "function".to_string(),
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

    fn finish(self) -> AssistantTurn {
        AssistantTurn {
            content: (!self.content.is_empty()).then_some(self.content),
            tool_calls: self.tool_calls,
            usage: self.usage,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
}

pub struct ChatClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    params: SamplingParams,
    stream: bool,
}

impl ChatClient {
    pub fn new(
        http: reqwest::Client,
        base_url: &str,
        api_key: Option<String>,
        model: &str,
        params: SamplingParams,
        stream: bool,
    ) -> Self {
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model: model.to_string(),
            params,
            stream,
        }
    }

    /// One round-trip without delta reporting.
    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolFunction],
    ) -> Result<AssistantTurn> {
        self.chat_streamed(messages, tools, None).await
    }

    /// One round-trip. With streaming enabled (the provider default),
    /// `on_delta` is invoked with each content fragment as it arrives; with
    /// it disabled, `on_delta` fires once with the complete text.
    pub async fn chat_streamed(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolFunction],
        on_delta: Option<DeltaHandler<'_>>,
    ) -> Result<AssistantTurn> {
        let request = ChatRequest {
            model: &self.model,
            messages,
            temperature: self.params.temperature,
            top_p: self.params.top_p,
            max_tokens: self.params.max_tokens,
            tools: tools
                .iter()
                .map(|f| ToolWire { kind: "function", function: f })
                .collect(),
            stream: self.stream,
            stream_options: self.stream.then_some(StreamOptions { include_usage: true }),
        };

        let url = format!("{}/chat/completions", self.base_url);
        let mut builder = self.http.post(&url).json(&request);
        if let Some(key) = &self.api_key
            && !key.is_empty()
        {
            builder = builder.bearer_auth(key);
        }

        let mut response = builder
            .send()
            .await
            .with_context(|| format!("request to {url} failed"))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            bail!("provider returned {status} from {url}: {body}");
        }

        if !self.stream {
            let body = response.text().await.unwrap_or_default();
            let parsed: ChatResponse = serde_json::from_str(&body)
                .with_context(|| format!("unexpected response from {url}: {body}"))?;
            let choice = parsed
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("provider returned no choices from {url}"))?;
            if let (Some(handler), Some(content)) = (on_delta, choice.message.content.as_deref())
            {
                handler(content);
            }
            return Ok(AssistantTurn {
                content: choice.message.content,
                tool_calls: choice.message.tool_calls.unwrap_or_default(),
                usage: parsed.usage,
            });
        }

        // SSE: `data: {json}` lines terminated by `data: [DONE]`. Buffer
        // bytes and only decode complete lines, so multi-byte characters
        // split across network chunks can't be corrupted.
        let mut accumulator = StreamAccumulator::default();
        let mut buffer: Vec<u8> = Vec::new();
        'stream: while let Some(chunk) = response
            .chunk()
            .await
            .with_context(|| format!("stream from {url} was interrupted"))?
        {
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=newline).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim_end();
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim_start();
                if data == "[DONE]" {
                    break 'stream;
                }
                let parsed: StreamChunk = serde_json::from_str(data)
                    .with_context(|| format!("unexpected stream chunk from {url}: {data}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(json: &str) -> StreamChunk {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn accumulates_content_deltas() {
        let mut acc = StreamAccumulator::default();
        assert_eq!(
            acc.apply(chunk(r#"{"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}"#)),
            Some("Hel".to_string())
        );
        assert_eq!(
            acc.apply(chunk(r#"{"choices":[{"delta":{"content":"lo"}}]}"#)),
            Some("lo".to_string())
        );
        // Finish chunks with empty deltas produce no fragment.
        assert_eq!(acc.apply(chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#)), None);
        // Final usage chunk (empty choices) is captured.
        acc.apply(chunk(
            r#"{"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}}"#,
        ));
        let turn = acc.finish();
        assert_eq!(turn.content.as_deref(), Some("Hello"));
        assert!(turn.tool_calls.is_empty());
        let usage = turn.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.context_tokens(), 120);
    }

    #[test]
    fn assembles_fragmented_tool_calls() {
        let mut acc = StreamAccumulator::default();
        acc.apply(chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"write_file","arguments":""}}]}}]}"#,
        ));
        acc.apply(chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]}}]}"#,
        ));
        acc.apply(chunk(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}},{"index":1,"id":"c2","function":{"name":"read_file","arguments":"{}"}}]}}]}"#,
        ));
        let turn = acc.finish();
        assert_eq!(turn.tool_calls.len(), 2);
        assert_eq!(turn.tool_calls[0].id, "c1");
        assert_eq!(turn.tool_calls[0].function.name, "write_file");
        assert_eq!(turn.tool_calls[0].function.arguments, r#"{"path":"x"}"#);
        assert_eq!(turn.tool_calls[1].function.name, "read_file");
        assert!(turn.content.is_none());
    }
}
