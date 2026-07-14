//! Minimal OpenAI-compatible chat-completions client with tool-call support.

use std::time::Duration;

use anyhow::{Result, anyhow};
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

pub fn fmt_tokens(tokens: u64) -> String {
    if tokens >= 1000 {
        format!("{:.1}k tok", tokens as f64 / 1000.0)
    } else {
        format!("{tokens} tok")
    }
}

/// Cumulative per-model token accounting for this process. Every
/// [`ChatClient`] records its successful round-trips here, so stage loops,
/// subagents, compaction, and chat turns are all counted without threading
/// a collector through every call chain.
pub mod usage_stats {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use super::Usage;

    /// Totals for one model, keyed by its config name.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct ModelUsage {
        /// Successful round-trips (retries of a request are not counted).
        pub requests: u64,
        pub prompt_tokens: u64,
        pub completion_tokens: u64,
    }

    static TOTALS: Mutex<BTreeMap<String, ModelUsage>> = Mutex::new(BTreeMap::new());

    pub(super) fn record(label: &str, usage: Option<Usage>) {
        let mut totals = TOTALS.lock().unwrap();
        let entry = totals.entry(label.to_string()).or_default();
        entry.requests += 1;
        if let Some(usage) = usage {
            entry.prompt_tokens += usage.prompt_tokens;
            entry.completion_tokens += usage.completion_tokens;
        }
    }

    /// Per-model totals recorded so far, sorted by model name.
    pub fn snapshot() -> BTreeMap<String, ModelUsage> {
        TOTALS.lock().unwrap().clone()
    }

    /// One line per model, plus a total when several models were used.
    /// Empty when no requests completed.
    pub fn report_lines() -> Vec<String> {
        let totals = snapshot();
        let mut lines = Vec::new();
        let mut sum = ModelUsage::default();
        for (model, usage) in &totals {
            sum.requests += usage.requests;
            sum.prompt_tokens += usage.prompt_tokens;
            sum.completion_tokens += usage.completion_tokens;
            lines.push(format!("{model}: {}", describe(usage)));
        }
        if totals.len() > 1 {
            lines.push(format!("total: {}", describe(&sum)));
        }
        lines
    }

    fn describe(usage: &ModelUsage) -> String {
        format!(
            "{} request(s), {} in + {} out",
            usage.requests,
            super::fmt_tokens(usage.prompt_tokens),
            super::fmt_tokens(usage.completion_tokens),
        )
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

/// Assembles streamed deltas into a complete [`AssistantTurn`]. Tool-call
/// ids, names, and argument JSON all arrive in fragments keyed by `index`.
#[derive(Default)]
struct StreamAccumulator {
    content: String,
    tool_calls: Vec<ToolCall>,
    usage: Option<Usage>,
    /// The generation reached a `finish_reason`, so connection EOF is a
    /// normal end of stream rather than a mid-generation drop.
    finished: bool,
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

/// One endpoint a [`ChatClient`] can talk to: a provider URL plus the
/// model and sampling parameters to use there.
pub struct Target {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    /// Config-level model name, used to attribute [`usage_stats`] and
    /// name the endpoint in logs.
    pub label: String,
    pub params: SamplingParams,
    pub stream: bool,
    /// Requests to this target cross the configured external data boundary.
    pub external: bool,
}

pub struct ChatClient {
    http: reqwest::Client,
    /// The primary model first, then its fallback chain in order.
    targets: Vec<Target>,
    /// Additional attempts per target after a transient failure (network
    /// error, 408/429/5xx, or an interrupted stream). When a target's
    /// retries are exhausted — or it fails fatally — the next target in
    /// the chain is tried.
    retries: u32,
}

/// One failed request attempt, classified for the retry loop.
struct AttemptError {
    source: anyhow::Error,
    retryable: bool,
    /// Server-requested delay (`Retry-After`), which overrides backoff.
    retry_after: Option<Duration>,
}

impl AttemptError {
    fn fatal(source: anyhow::Error) -> Self {
        Self { source, retryable: false, retry_after: None }
    }

    fn transient(source: anyhow::Error) -> Self {
        Self { source, retryable: true, retry_after: None }
    }
}

/// Exponential backoff: 500ms doubling per attempt, capped at 10s.
fn backoff_delay(attempt: u32) -> Duration {
    Duration::from_millis((500u64 << attempt.min(5)).min(10_000))
}

fn retryable_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
}

fn parse_retry_after(response: &reqwest::Response) -> Option<Duration> {
    let value = response.headers().get(reqwest::header::RETRY_AFTER)?.to_str().ok()?;
    let seconds: u64 = value.trim().parse().ok()?;
    Some(Duration::from_secs(seconds.min(30)))
}

/// The part of `fragment` beyond the first `emitted` content bytes, given
/// that `cumulative_before` bytes of content preceded this fragment. Used
/// to avoid re-emitting deltas a caller already saw before a mid-stream
/// retry. Byte offsets are nudged to a char boundary (a retried generation
/// may diverge from the failed one, so the seam is best-effort; the
/// returned turn's full content is always authoritative).
fn novel_suffix(fragment: &str, cumulative_before: usize, emitted: usize) -> Option<&str> {
    let mut skip = emitted.saturating_sub(cumulative_before);
    if skip == 0 {
        return Some(fragment);
    }
    while skip < fragment.len() && !fragment.is_char_boundary(skip) {
        skip += 1;
    }
    (skip < fragment.len()).then(|| &fragment[skip..])
}

impl ChatClient {
    pub fn new(http: reqwest::Client, mut targets: Vec<Target>, retries: u32) -> Self {
        for target in &mut targets {
            target.base_url = target.base_url.trim_end_matches('/').to_string();
        }
        assert!(!targets.is_empty(), "a ChatClient needs at least one target");
        Self { http, targets, retries }
    }

    /// Config-level model names in chain order (primary first).
    pub fn target_labels(&self) -> Vec<&str> {
        self.targets.iter().map(|t| t.label.as_str()).collect()
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
    ///
    /// Transient failures (network errors, 408/429/5xx, interrupted
    /// streams) are retried with exponential backoff up to
    /// `settings.provider_retries` times; when a target is exhausted (or
    /// fails fatally) the next target in the fallback chain is tried.
    /// Content already delivered to `on_delta` before a mid-stream failure
    /// is not re-emitted, including across a failover.
    pub async fn chat_streamed(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolFunction],
        on_delta: Option<DeltaHandler<'_>>,
    ) -> Result<AssistantTurn> {
        let mut emitted = 0usize;
        let mut last_error = None;
        for (index, target) in self.targets.iter().enumerate() {
            let mut attempt = 0u32;
            let failure = loop {
                match self.attempt_chat(target, messages, tools, on_delta, &mut emitted).await {
                    Ok(turn) => {
                        usage_stats::record(&target.label, turn.usage);
                        if index > 0 {
                            tracing::warn!(
                                model = %target.label,
                                external = target.external,
                                "request served by fallback model"
                            );
                        }
                        return Ok(turn);
                    }
                    Err(error) if error.retryable && attempt < self.retries => {
                        let delay = error.retry_after.unwrap_or_else(|| backoff_delay(attempt));
                        attempt += 1;
                        tracing::warn!(
                            model = %target.label,
                            attempt,
                            retries = self.retries,
                            delay_ms = delay.as_millis() as u64,
                            error = format!("{:#}", error.source),
                            "provider request failed; retrying"
                        );
                        tokio::time::sleep(delay).await;
                    }
                    Err(error) => break error.source,
                }
            };
            if index + 1 < self.targets.len() {
                tracing::warn!(
                    model = %target.label,
                    next = %self.targets[index + 1].label,
                    error = format!("{failure:#}"),
                    "model endpoint failed; falling back"
                );
            }
            last_error = Some(failure);
        }
        let chain = self.target_labels().join(" -> ");
        Err(last_error
            .expect("at least one target")
            .context(format!("every model endpoint failed (tried: {chain})")))
    }

    async fn attempt_chat(
        &self,
        target: &Target,
        messages: &[ChatMessage],
        tools: &[ToolFunction],
        on_delta: Option<DeltaHandler<'_>>,
        emitted: &mut usize,
    ) -> std::result::Result<AssistantTurn, AttemptError> {
        tracing::debug!(
            model = %target.label,
            external = target.external,
            "sending provider request"
        );
        let request = ChatRequest {
            model: &target.model,
            messages,
            temperature: target.params.temperature,
            top_p: target.params.top_p,
            max_tokens: target.params.max_tokens,
            tools: tools
                .iter()
                .map(|f| ToolWire { kind: "function", function: f })
                .collect(),
            stream: target.stream,
            stream_options: target.stream.then_some(StreamOptions { include_usage: true }),
        };

        let url = format!("{}/chat/completions", target.base_url);
        let mut builder = self.http.post(&url).json(&request);
        if let Some(key) = &target.api_key
            && !key.is_empty()
        {
            builder = builder.bearer_auth(key);
        }

        let mut response = match builder.send().await {
            Ok(response) => response,
            // Connection, DNS, and timeout failures are transient; a
            // request that couldn't be built (e.g. bad header) is not.
            Err(e) => {
                let retryable = !e.is_builder();
                let source = anyhow::Error::new(e).context(format!("request to {url} failed"));
                return Err(AttemptError { source, retryable, retry_after: None });
            }
        };

        let status = response.status();
        if !status.is_success() {
            let retry_after = parse_retry_after(&response);
            let body = response.text().await.unwrap_or_default();
            return Err(AttemptError {
                source: anyhow!("provider returned {status} from {url}: {body}"),
                retryable: retryable_status(status),
                retry_after,
            });
        }

        if !target.stream {
            let body = response.text().await.unwrap_or_default();
            let parsed: ChatResponse = serde_json::from_str(&body).map_err(|e| {
                AttemptError::fatal(
                    anyhow::Error::new(e)
                        .context(format!("unexpected response from {url}: {body}")),
                )
            })?;
            let choice = parsed.choices.into_iter().next().ok_or_else(|| {
                AttemptError::fatal(anyhow!("provider returned no choices from {url}"))
            })?;
            if let (Some(handler), Some(content)) = (on_delta, choice.message.content.as_deref())
                && let Some(novel) = novel_suffix(content, 0, *emitted)
            {
                handler(novel);
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
        'stream: loop {
            let chunk = match response.chunk().await {
                Ok(Some(chunk)) => chunk,
                // EOF without `[DONE]` or a finish_reason means the server
                // died mid-generation (a clean close looks identical to a
                // finished body); treat it like an interrupted stream.
                Ok(None) if accumulator.finished => break,
                Ok(None) => {
                    return Err(AttemptError::transient(anyhow!(
                        "stream from {url} ended before the generation finished"
                    )));
                }
                Err(e) => {
                    return Err(AttemptError::transient(
                        anyhow::Error::new(e)
                            .context(format!("stream from {url} was interrupted")),
                    ));
                }
            };
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
                let parsed: StreamChunk = serde_json::from_str(data).map_err(|e| {
                    AttemptError::fatal(
                        anyhow::Error::new(e)
                            .context(format!("unexpected stream chunk from {url}: {data}")),
                    )
                })?;
                let before = accumulator.content.len();
                if let Some(fragment) = accumulator.apply(parsed) {
                    if let Some(handler) = on_delta
                        && let Some(novel) = novel_suffix(&fragment, before, *emitted)
                    {
                        handler(novel);
                    }
                    *emitted = (*emitted).max(before + fragment.len());
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
        // Finish chunks with empty deltas produce no fragment, but mark the
        // generation complete so a missing [DONE] isn't treated as a drop.
        assert!(!acc.finished);
        assert_eq!(acc.apply(chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#)), None);
        assert!(acc.finished);
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
    fn usage_stats_accumulate_per_model() {
        let usage = |p, c| Some(Usage { prompt_tokens: p, completion_tokens: c });
        usage_stats::record("stats-test-a", usage(100, 10));
        usage_stats::record("stats-test-a", usage(200, 20));
        // A server that reports no usage still counts the request.
        usage_stats::record("stats-test-a", None);
        usage_stats::record("stats-test-b", usage(5, 1));

        let totals = usage_stats::snapshot();
        let a = totals["stats-test-a"];
        assert_eq!((a.requests, a.prompt_tokens, a.completion_tokens), (3, 300, 30));
        assert_eq!(totals["stats-test-b"].requests, 1);

        let report = usage_stats::report_lines().join("\n");
        assert!(report.contains("stats-test-a: 3 request(s), 300 tok in + 30 tok out"));
        assert!(report.contains("total: "), "{report}");
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_delay(0), Duration::from_millis(500));
        assert_eq!(backoff_delay(1), Duration::from_millis(1000));
        assert_eq!(backoff_delay(2), Duration::from_millis(2000));
        // Capped at 10s, including for absurd attempt counts.
        assert_eq!(backoff_delay(5), Duration::from_secs(10));
        assert_eq!(backoff_delay(63), Duration::from_secs(10));
    }

    #[test]
    fn transient_statuses_classified() {
        use reqwest::StatusCode;
        for code in [500, 502, 503, 504, 429, 408] {
            assert!(retryable_status(StatusCode::from_u16(code).unwrap()), "{code}");
        }
        for code in [400, 401, 403, 404, 422] {
            assert!(!retryable_status(StatusCode::from_u16(code).unwrap()), "{code}");
        }
    }

    #[test]
    fn novel_suffix_skips_already_emitted_content() {
        // Nothing emitted yet: whole fragment is novel.
        assert_eq!(novel_suffix("hello", 0, 0), Some("hello"));
        // Fragment starts past the emitted point: all novel.
        assert_eq!(novel_suffix("lo", 3, 3), Some("lo"));
        // Fragment straddles the emitted point: only the tail is novel.
        assert_eq!(novel_suffix("hello", 0, 3), Some("lo"));
        // Fragment entirely within already-emitted content: nothing.
        assert_eq!(novel_suffix("hel", 0, 3), None);
        assert_eq!(novel_suffix("hel", 0, 10), None);
        // A skip landing mid-character nudges forward to a boundary.
        assert_eq!(novel_suffix("héllo", 0, 2), Some("llo"));
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
