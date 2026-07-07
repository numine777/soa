//! Minimal OpenAI-compatible chat-completions client with tool-call support.

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String,
    },
    User {
        content: String,
    },
    Assistant {
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
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
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
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

/// What the model returned for one round-trip.
#[derive(Debug)]
pub struct AssistantTurn {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
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
}

impl ChatClient {
    pub fn new(
        http: reqwest::Client,
        base_url: &str,
        api_key: Option<String>,
        model: &str,
        params: SamplingParams,
    ) -> Self {
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model: model.to_string(),
            params,
        }
    }

    pub async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolFunction],
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
        };

        let url = format!("{}/chat/completions", self.base_url);
        let mut builder = self.http.post(&url).json(&request);
        if let Some(key) = &self.api_key
            && !key.is_empty()
        {
            builder = builder.bearer_auth(key);
        }

        let response = builder
            .send()
            .await
            .with_context(|| format!("request to {url} failed"))?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("provider returned {status} from {url}: {body}");
        }

        let parsed: ChatResponse = serde_json::from_str(&body)
            .with_context(|| format!("unexpected response from {url}: {body}"))?;
        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("provider returned no choices from {url}"))?;

        Ok(AssistantTurn {
            content: choice.message.content,
            tool_calls: choice.message.tool_calls.unwrap_or_default(),
        })
    }
}
