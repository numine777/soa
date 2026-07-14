//! Canonical model API and provider-independent retry/fallback client.
//!
//! Agent loops speak only in terms of [`Message`], [`ToolDefinition`], and
//! [`ModelResponse`]. Provider adapters translate that contract to a wire
//! protocol such as OpenAI chat completions.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One canonical conversation message. Its serde representation is soa's
/// session-storage format; provider adapters own their separate wire types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
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

/// A provider-independent request by a model to invoke one tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments, as produced by the model.
    #[serde(default)]
    pub arguments: String,
}

/// A tool advertised to a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema for the arguments object.
    pub parameters: Value,
}

/// Provider-neutral sampling controls supported by soa's model contract.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SamplingParams {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
}

/// One canonical completion request passed to a provider adapter.
#[derive(Debug, Clone, Copy)]
pub struct ModelRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [Message],
    pub tools: &'a [ToolDefinition],
    pub sampling: SamplingParams,
    pub stream: bool,
}

/// Token counts reported by a provider for one request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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

/// What a model returned for one canonical round-trip.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Real token counts, when the provider reports them.
    pub usage: Option<Usage>,
}

/// Callback invoked with each streamed text fragment.
pub type DeltaHandler<'a> = &'a (dyn Fn(&str) + Send + Sync);

/// Classified provider-adapter failure used by the generic retry loop.
#[derive(Debug)]
pub struct AdapterError {
    source: anyhow::Error,
    retryable: bool,
    retry_after: Option<Duration>,
}

impl AdapterError {
    pub fn fatal(source: anyhow::Error) -> Self {
        Self {
            source,
            retryable: false,
            retry_after: None,
        }
    }

    pub fn transient(source: anyhow::Error) -> Self {
        Self {
            source,
            retryable: true,
            retry_after: None,
        }
    }

    pub fn classified(
        source: anyhow::Error,
        retryable: bool,
        retry_after: Option<Duration>,
    ) -> Self {
        Self {
            source,
            retryable,
            retry_after,
        }
    }
}

/// Boxed adapter future keeps [`ProviderAdapter`] object-safe without
/// imposing an async-trait dependency.
pub type AdapterFuture<'a> =
    Pin<Box<dyn Future<Output = std::result::Result<ModelResponse, AdapterError>> + Send + 'a>>;

/// Boundary implemented by every provider wire adapter.
pub trait ProviderAdapter: Send + Sync {
    /// Stable name used in diagnostics and tests.
    fn name(&self) -> &'static str;

    /// Translate one canonical request, execute it, and translate the
    /// provider response back into the canonical response.
    fn complete<'a>(
        &'a self,
        request: ModelRequest<'a>,
        on_delta: Option<DeltaHandler<'a>>,
    ) -> AdapterFuture<'a>;
}

pub fn fmt_tokens(tokens: u64) -> String {
    if tokens >= 1000 {
        format!("{:.1}k tok", tokens as f64 / 1000.0)
    } else {
        format!("{tokens} tok")
    }
}

/// Cumulative per-model token accounting for this process.
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

/// One model plus its already-constructed provider adapter.
pub struct ModelTarget {
    /// Config-level model name, used for usage attribution and logs.
    pub label: String,
    pub model: String,
    pub sampling: SamplingParams,
    pub stream: bool,
    /// Requests to this target cross the configured external data boundary.
    pub external: bool,
    pub adapter: Arc<dyn ProviderAdapter>,
}

/// Provider-neutral client that owns retries, fallback, usage, and streamed
/// delta de-duplication across attempts.
pub struct ModelClient {
    /// The primary model first, then its fallback chain in order.
    targets: Vec<ModelTarget>,
    /// Additional attempts per target after a transient failure.
    retries: u32,
}

/// Exponential backoff: 500ms doubling per attempt, capped at 10s.
fn backoff_delay(attempt: u32) -> Duration {
    Duration::from_millis((500u64 << attempt.min(5)).min(10_000))
}

/// The part of `fragment` beyond the first `emitted` content bytes, given
/// that `cumulative_before` bytes preceded this fragment in the current
/// attempt. This prevents duplicate UI output after retry or failover.
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

impl ModelClient {
    pub fn new(targets: Vec<ModelTarget>, retries: u32) -> Self {
        assert!(
            !targets.is_empty(),
            "a ModelClient needs at least one target"
        );
        Self { targets, retries }
    }

    /// Config-level model names in chain order (primary first).
    pub fn target_labels(&self) -> Vec<&str> {
        self.targets
            .iter()
            .map(|target| target.label.as_str())
            .collect()
    }

    /// One round-trip without delta reporting.
    pub async fn complete(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Result<ModelResponse> {
        self.complete_streamed(messages, tools, None).await
    }

    /// One canonical round-trip with retry and fallback orchestration.
    pub async fn complete_streamed(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_delta: Option<DeltaHandler<'_>>,
    ) -> Result<ModelResponse> {
        let emitted = AtomicUsize::new(0);
        let mut last_error = None;
        for (index, target) in self.targets.iter().enumerate() {
            let mut attempt = 0u32;
            let failure = loop {
                tracing::debug!(
                    model = %target.label,
                    adapter = target.adapter.name(),
                    external = target.external,
                    "sending model request"
                );
                let cumulative = AtomicUsize::new(0);
                let filtered_delta = |fragment: &str| {
                    let before = cumulative.fetch_add(fragment.len(), Ordering::Relaxed);
                    if let Some(handler) = on_delta
                        && let Some(novel) =
                            novel_suffix(fragment, before, emitted.load(Ordering::Relaxed))
                    {
                        handler(novel);
                        emitted.fetch_max(before + fragment.len(), Ordering::Relaxed);
                    }
                };
                let request = ModelRequest {
                    model: &target.model,
                    messages,
                    tools,
                    sampling: target.sampling,
                    stream: target.stream,
                };
                match target
                    .adapter
                    .complete(request, Some(&filtered_delta))
                    .await
                {
                    Ok(response) => {
                        usage_stats::record(&target.label, response.usage);
                        if index > 0 {
                            tracing::warn!(
                                model = %target.label,
                                adapter = target.adapter.name(),
                                external = target.external,
                                "request served by fallback model"
                            );
                        }
                        return Ok(response);
                    }
                    Err(error) if error.retryable && attempt < self.retries => {
                        let delay = error.retry_after.unwrap_or_else(|| backoff_delay(attempt));
                        attempt += 1;
                        tracing::warn!(
                            model = %target.label,
                            adapter = target.adapter.name(),
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
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    struct RecordingAdapter {
        seen: Mutex<Option<(String, usize, usize)>>,
    }

    impl ProviderAdapter for RecordingAdapter {
        fn name(&self) -> &'static str {
            "recording"
        }

        fn complete<'a>(
            &'a self,
            request: ModelRequest<'a>,
            on_delta: Option<DeltaHandler<'a>>,
        ) -> AdapterFuture<'a> {
            Box::pin(async move {
                *self.seen.lock().unwrap() = Some((
                    request.model.to_string(),
                    request.messages.len(),
                    request.tools.len(),
                ));
                if let Some(handler) = on_delta {
                    handler("done");
                }
                Ok(ModelResponse {
                    content: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    usage: Some(Usage {
                        prompt_tokens: 3,
                        completion_tokens: 1,
                    }),
                })
            })
        }
    }

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_delay(0), Duration::from_millis(500));
        assert_eq!(backoff_delay(1), Duration::from_millis(1000));
        assert_eq!(backoff_delay(2), Duration::from_millis(2000));
        assert_eq!(backoff_delay(5), Duration::from_secs(10));
        assert_eq!(backoff_delay(63), Duration::from_secs(10));
    }

    #[test]
    fn novel_suffix_skips_already_emitted_content() {
        assert_eq!(novel_suffix("hello", 0, 0), Some("hello"));
        assert_eq!(novel_suffix("lo", 3, 3), Some("lo"));
        assert_eq!(novel_suffix("hello", 0, 3), Some("lo"));
        assert_eq!(novel_suffix("hel", 0, 3), None);
        assert_eq!(novel_suffix("hel", 0, 10), None);
        assert_eq!(novel_suffix("héllo", 0, 2), Some("llo"));
    }

    #[test]
    fn old_openai_shaped_tool_calls_remain_session_compatible() {
        let message: Message = serde_json::from_str(
            r#"{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{}"}}]}"#,
        )
        .unwrap();
        let Message::Assistant {
            tool_calls: Some(calls),
            ..
        } = message
        else {
            panic!()
        };
        assert_eq!(calls[0].function.name, "read_file");
        assert!(
            !serde_json::to_string(&calls[0])
                .unwrap()
                .contains("\"type\"")
        );
    }

    #[tokio::test]
    async fn model_client_depends_only_on_the_adapter_contract() {
        let adapter = Arc::new(RecordingAdapter {
            seen: Mutex::new(None),
        });
        let client = ModelClient::new(
            vec![ModelTarget {
                label: "canonical-test".into(),
                model: "coder".into(),
                sampling: SamplingParams::default(),
                stream: true,
                external: false,
                adapter: adapter.clone(),
            }],
            0,
        );
        let messages = [Message::User {
            content: "work".into(),
        }];
        let streamed = Mutex::new(String::new());
        let on_delta = |fragment: &str| streamed.lock().unwrap().push_str(fragment);
        let response = client
            .complete_streamed(&messages, &[], Some(&on_delta))
            .await
            .unwrap();

        assert_eq!(response.content.as_deref(), Some("done"));
        assert_eq!(&*streamed.lock().unwrap(), "done");
        assert_eq!(
            *adapter.seen.lock().unwrap(),
            Some(("coder".to_string(), 1, 0))
        );
    }

    #[test]
    fn usage_stats_accumulate_per_model() {
        let usage = |p, c| {
            Some(Usage {
                prompt_tokens: p,
                completion_tokens: c,
            })
        };
        usage_stats::record("stats-test-a", usage(100, 10));
        usage_stats::record("stats-test-a", usage(200, 20));
        usage_stats::record("stats-test-a", None);
        usage_stats::record("stats-test-b", usage(5, 1));

        let totals = usage_stats::snapshot();
        let a = totals["stats-test-a"];
        assert_eq!(
            (a.requests, a.prompt_tokens, a.completion_tokens),
            (3, 300, 30)
        );
        assert_eq!(totals["stats-test-b"].requests, 1);

        let report = usage_stats::report_lines().join("\n");
        assert!(report.contains("stats-test-a: 3 request(s), 300 tok in + 30 tok out"));
        assert!(report.contains("total: "), "{report}");
    }
}
