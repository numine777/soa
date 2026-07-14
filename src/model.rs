//! Canonical model API and provider-independent retry/fallback client.
//!
//! Agent loops speak only in terms of [`Message`], [`ToolDefinition`], and
//! [`ModelResponse`]. Provider adapters translate that contract to a wire
//! protocol such as OpenAI chat completions.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
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

/// Optional prices for one model, in US dollars per million tokens.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ModelPricing {
    pub input_per_million: Option<f64>,
    pub output_per_million: Option<f64>,
}

impl ModelPricing {
    fn cost(self, usage: Usage) -> Option<f64> {
        Some(
            usage.prompt_tokens as f64 * self.input_per_million? / 1_000_000.0
                + usage.completion_tokens as f64 * self.output_per_million? / 1_000_000.0,
        )
    }

    fn is_complete(self) -> bool {
        self.input_per_million.is_some() && self.output_per_million.is_some()
    }
}

/// Limits shared by every model call and tool operation in one run.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RunLimits {
    pub max_tokens: Option<u64>,
    pub max_cost_usd: Option<f64>,
    pub max_time: Option<Duration>,
}

/// Rich aggregate for one config-level model name.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelUsage {
    #[serde(default)]
    pub adapter: String,
    #[serde(default)]
    pub external: bool,
    #[serde(default)]
    pub attempts: u64,
    #[serde(default)]
    pub failures: u64,
    #[serde(default)]
    pub requests: u64,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub unreported_requests: u64,
    #[serde(default)]
    pub cost_usd: f64,
    #[serde(default)]
    pub unpriced_requests: u64,
    #[serde(default)]
    pub latency_ms: u64,
}

/// Serializable run ledger. Checkpoints persist it so resume continues the
/// original token, cost, and active-time budgets.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageSnapshot {
    #[serde(default)]
    pub models: BTreeMap<String, ModelUsage>,
    #[serde(default)]
    pub elapsed_ms: u64,
}

#[derive(Default)]
struct UsageTotals {
    attempts: u64,
    failures: u64,
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    unreported_requests: u64,
    cost_usd: f64,
    unpriced_requests: u64,
    latency_ms: u64,
}

impl UsageSnapshot {
    fn totals(&self) -> UsageTotals {
        let mut total = UsageTotals::default();
        for usage in self.models.values() {
            total.attempts += usage.attempts;
            total.failures += usage.failures;
            total.requests += usage.requests;
            total.prompt_tokens += usage.prompt_tokens;
            total.completion_tokens += usage.completion_tokens;
            total.unreported_requests += usage.unreported_requests;
            total.cost_usd += usage.cost_usd;
            total.unpriced_requests += usage.unpriced_requests;
            total.latency_ms += usage.latency_ms;
        }
        total
    }
}

struct UsageTrackerInner {
    limits: RunLimits,
    prior_elapsed_ms: u64,
    started: Instant,
    state: Mutex<UsageSnapshot>,
    /// Prevent parallel subagents from racing the same remaining token/cost
    /// allowance. Unlimited and time-only runs do not acquire it.
    spend_gate: tokio::sync::Mutex<()>,
}

/// Cloneable, run-scoped usage ledger and budget gate.
#[derive(Clone)]
pub struct UsageTracker {
    inner: Arc<UsageTrackerInner>,
}

impl UsageTracker {
    pub fn new(limits: RunLimits, mut previous: UsageSnapshot) -> Self {
        let prior_elapsed_ms = previous.elapsed_ms;
        previous.elapsed_ms = 0;
        Self {
            inner: Arc::new(UsageTrackerInner {
                limits,
                prior_elapsed_ms,
                started: Instant::now(),
                state: Mutex::new(previous),
                spend_gate: tokio::sync::Mutex::new(()),
            }),
        }
    }

    pub fn unlimited() -> Self {
        Self::new(RunLimits::default(), UsageSnapshot::default())
    }

    fn elapsed_ms(&self) -> u64 {
        self.inner.prior_elapsed_ms.saturating_add(
            self.inner
                .started
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64,
        )
    }

    pub fn snapshot(&self) -> UsageSnapshot {
        let mut snapshot = self.inner.state.lock().unwrap().clone();
        snapshot.elapsed_ms = self.elapsed_ms();
        snapshot
    }

    /// Remaining provider-reported token budget, used to clamp the next
    /// request's output allowance. Input tokens are verified afterwards.
    fn remaining_tokens(&self) -> Option<u64> {
        let limit = self.inner.limits.max_tokens?;
        let totals = self.inner.state.lock().unwrap().totals();
        Some(
            limit.saturating_sub(
                totals
                    .prompt_tokens
                    .saturating_add(totals.completion_tokens),
            ),
        )
    }

    fn remaining_time(&self) -> Result<Option<Duration>> {
        let Some(limit) = self.inner.limits.max_time else {
            return Ok(None);
        };
        let elapsed = Duration::from_millis(self.elapsed_ms());
        match limit.checked_sub(elapsed) {
            Some(remaining) if !remaining.is_zero() => Ok(Some(remaining)),
            _ => bail!(
                "run time budget exhausted: {} elapsed of {}",
                fmt_duration(elapsed),
                fmt_duration(limit)
            ),
        }
    }

    fn check_before_request(&self, label: &str, pricing: ModelPricing) -> Result<()> {
        self.check_limits(false)?;
        if self.inner.limits.max_cost_usd.is_some() && !pricing.is_complete() {
            bail!(
                "run cost budget cannot be enforced for model `{label}`: configure both \
                 input_cost_per_million and output_cost_per_million"
            );
        }
        Ok(())
    }

    fn check_after_response(&self, usage: Option<Usage>) -> Result<()> {
        if usage.is_none()
            && (self.inner.limits.max_tokens.is_some() || self.inner.limits.max_cost_usd.is_some())
        {
            bail!(
                "run budget cannot be enforced because the provider omitted token usage for a \
                 successful request"
            );
        }
        self.check_limits(true)
    }

    fn check_limits(&self, after_response: bool) -> Result<()> {
        let snapshot = self.snapshot();
        let totals = snapshot.totals();
        let tokens = totals
            .prompt_tokens
            .saturating_add(totals.completion_tokens);
        if let Some(limit) = self.inner.limits.max_tokens
            && if after_response {
                tokens > limit
            } else {
                tokens >= limit
            }
        {
            bail!(
                "run token budget exhausted: {} used of {}",
                fmt_tokens(tokens),
                fmt_tokens(limit)
            );
        }
        if let Some(limit) = self.inner.limits.max_cost_usd
            && if after_response {
                totals.cost_usd > limit
            } else {
                totals.cost_usd >= limit
            }
        {
            bail!(
                "run cost budget exhausted: {} used of {}",
                fmt_cost(totals.cost_usd),
                fmt_cost(limit)
            );
        }
        if let Some(limit) = self.inner.limits.max_time {
            let elapsed = Duration::from_millis(snapshot.elapsed_ms);
            if elapsed >= limit {
                bail!(
                    "run time budget exhausted: {} elapsed of {}",
                    fmt_duration(elapsed),
                    fmt_duration(limit)
                );
            }
        }
        Ok(())
    }

    fn record_attempt(
        &self,
        label: &str,
        adapter: &str,
        external: bool,
        elapsed: Duration,
        response: Option<Option<Usage>>,
        pricing: ModelPricing,
    ) {
        let mut state = self.inner.state.lock().unwrap();
        let entry = state.models.entry(label.to_string()).or_default();
        entry.adapter = adapter.to_string();
        entry.external |= external;
        entry.attempts += 1;
        entry.latency_ms = entry
            .latency_ms
            .saturating_add(elapsed.as_millis().min(u128::from(u64::MAX)) as u64);
        match response {
            None => entry.failures += 1,
            Some(usage) => {
                entry.requests += 1;
                match usage {
                    Some(usage) => {
                        entry.prompt_tokens =
                            entry.prompt_tokens.saturating_add(usage.prompt_tokens);
                        entry.completion_tokens = entry
                            .completion_tokens
                            .saturating_add(usage.completion_tokens);
                        if let Some(cost) = pricing.cost(usage) {
                            entry.cost_usd += cost;
                        } else {
                            entry.unpriced_requests += 1;
                        }
                    }
                    None => {
                        entry.unreported_requests += 1;
                        entry.unpriced_requests += 1;
                    }
                }
            }
        }
    }

    /// Run an operation under the remaining wall-clock budget. This wraps
    /// whole stages, so long-running tools and delegated agents are covered.
    pub async fn within_time<T, F>(&self, future: F) -> Result<T>
    where
        F: Future<Output = Result<T>>,
    {
        self.check_limits(false)?;
        let Some(remaining) = self.remaining_time()? else {
            return future.await;
        };
        match tokio::time::timeout(remaining, future).await {
            Ok(result) => result,
            Err(_) => bail!(
                "run time budget exhausted after {}",
                fmt_duration(self.inner.limits.max_time.expect("remaining implies limit"))
            ),
        }
    }

    pub fn report_lines(&self) -> Vec<String> {
        let snapshot = self.snapshot();
        let totals = snapshot.totals();
        if totals.attempts == 0 {
            let mut budgets = Vec::new();
            if let Some(limit) = self.inner.limits.max_tokens {
                budgets.push(format!("tokens 0 tok/{}", fmt_tokens(limit)));
            }
            if let Some(limit) = self.inner.limits.max_cost_usd {
                budgets.push(format!("cost $0.0000/{}", fmt_cost(limit)));
            }
            if let Some(limit) = self.inner.limits.max_time {
                budgets.push(format!(
                    "time {}/{}",
                    fmt_duration(Duration::from_millis(snapshot.elapsed_ms)),
                    fmt_duration(limit)
                ));
            }
            if budgets.is_empty() {
                return Vec::new();
            }
            return vec![
                format!(
                    "total: no model attempts · {} elapsed",
                    fmt_duration(Duration::from_millis(snapshot.elapsed_ms))
                ),
                format!("budgets: {}", budgets.join(" · ")),
            ];
        }
        let mut lines = Vec::new();
        for (label, usage) in &snapshot.models {
            let mut details = vec![
                format!(
                    "{} request(s)/{} attempt(s)",
                    usage.requests, usage.attempts
                ),
                format!(
                    "{} in + {} out",
                    fmt_tokens(usage.prompt_tokens),
                    fmt_tokens(usage.completion_tokens)
                ),
                describe_cost(usage.cost_usd, usage.unpriced_requests),
                format!(
                    "{} provider time",
                    fmt_duration(Duration::from_millis(usage.latency_ms))
                ),
                format!("via {}", usage.adapter),
            ];
            if usage.failures > 0 {
                details.push(format!("{} failed", usage.failures));
            }
            if usage.unreported_requests > 0 {
                details.push(format!("{} without usage", usage.unreported_requests));
            }
            if usage.external {
                details.push("external".to_string());
            }
            lines.push(format!("{label}: {}", details.join(" · ")));
        }
        let mut total_line = format!(
            "total: {} request(s)/{} attempt(s) · {} in + {} out · {} · {} provider time · {} elapsed",
            totals.requests,
            totals.attempts,
            fmt_tokens(totals.prompt_tokens),
            fmt_tokens(totals.completion_tokens),
            describe_cost(totals.cost_usd, totals.unpriced_requests),
            fmt_duration(Duration::from_millis(totals.latency_ms)),
            fmt_duration(Duration::from_millis(snapshot.elapsed_ms)),
        );
        if totals.failures > 0 {
            total_line.push_str(&format!(" · {} failed", totals.failures));
        }
        if totals.unreported_requests > 0 {
            total_line.push_str(&format!(" · {} without usage", totals.unreported_requests));
        }
        lines.push(total_line);
        let mut budgets = Vec::new();
        if let Some(limit) = self.inner.limits.max_tokens {
            budgets.push(format!(
                "tokens {}/{}",
                fmt_tokens(
                    totals
                        .prompt_tokens
                        .saturating_add(totals.completion_tokens)
                ),
                fmt_tokens(limit)
            ));
        }
        if let Some(limit) = self.inner.limits.max_cost_usd {
            budgets.push(format!(
                "cost {}/{}",
                fmt_cost(totals.cost_usd),
                fmt_cost(limit)
            ));
        }
        if let Some(limit) = self.inner.limits.max_time {
            budgets.push(format!(
                "time {}/{}",
                fmt_duration(Duration::from_millis(snapshot.elapsed_ms)),
                fmt_duration(limit)
            ));
        }
        if !budgets.is_empty() {
            lines.push(format!("budgets: {}", budgets.join(" · ")));
        }
        lines
    }
}

fn fmt_cost(cost: f64) -> String {
    if cost >= 1.0 {
        format!("${cost:.2}")
    } else {
        format!("${cost:.4}")
    }
}

fn describe_cost(cost: f64, unpriced_requests: u64) -> String {
    if unpriced_requests == 0 {
        fmt_cost(cost)
    } else {
        format!("{} known + {unpriced_requests} unpriced", fmt_cost(cost))
    }
}

fn fmt_duration(duration: Duration) -> String {
    let millis = duration.as_millis();
    if millis < 1_000 {
        format!("{millis}ms")
    } else if millis < 60_000 {
        format!("{:.1}s", millis as f64 / 1_000.0)
    } else {
        let seconds = duration.as_secs();
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    }
}

/// One model plus its already-constructed provider adapter.
pub struct ModelTarget {
    /// Config-level model name, used for usage attribution and logs.
    pub label: String,
    pub model: String,
    pub sampling: SamplingParams,
    pub stream: bool,
    pub pricing: ModelPricing,
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
    usage: UsageTracker,
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
    pub fn new(targets: Vec<ModelTarget>, retries: u32, usage: UsageTracker) -> Self {
        assert!(
            !targets.is_empty(),
            "a ModelClient needs at least one target"
        );
        Self {
            targets,
            retries,
            usage,
        }
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
                let spend_guard = if self.usage.inner.limits.max_tokens.is_some()
                    || self.usage.inner.limits.max_cost_usd.is_some()
                {
                    Some(self.usage.inner.spend_gate.lock().await)
                } else {
                    None
                };
                self.usage
                    .check_before_request(&target.label, target.pricing)?;
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
                let mut sampling = target.sampling;
                if let Some(remaining) = self.usage.remaining_tokens() {
                    let remaining = remaining.min(u64::from(u32::MAX)) as u32;
                    sampling.max_tokens =
                        Some(sampling.max_tokens.unwrap_or(u32::MAX).min(remaining));
                }
                let request = ModelRequest {
                    model: &target.model,
                    messages,
                    tools,
                    sampling,
                    stream: target.stream,
                };
                let request_started = Instant::now();
                let adapter_call = target.adapter.complete(request, Some(&filtered_delta));
                let adapter_result = if let Some(remaining) = self.usage.remaining_time()? {
                    match tokio::time::timeout(remaining, adapter_call).await {
                        Ok(result) => result,
                        Err(_) => {
                            self.usage.record_attempt(
                                &target.label,
                                target.adapter.name(),
                                target.external,
                                request_started.elapsed(),
                                None,
                                target.pricing,
                            );
                            bail!(
                                "run time budget exhausted while waiting for model `{}`",
                                target.label
                            );
                        }
                    }
                } else {
                    adapter_call.await
                };
                match adapter_result {
                    Ok(response) => {
                        self.usage.record_attempt(
                            &target.label,
                            target.adapter.name(),
                            target.external,
                            request_started.elapsed(),
                            Some(response.usage),
                            target.pricing,
                        );
                        self.usage.check_after_response(response.usage)?;
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
                        self.usage.record_attempt(
                            &target.label,
                            target.adapter.name(),
                            target.external,
                            request_started.elapsed(),
                            None,
                            target.pricing,
                        );
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
                        drop(spend_guard);
                        self.usage
                            .within_time(async {
                                tokio::time::sleep(delay).await;
                                Ok(())
                            })
                            .await?;
                    }
                    Err(error) => {
                        self.usage.record_attempt(
                            &target.label,
                            target.adapter.name(),
                            target.external,
                            request_started.elapsed(),
                            None,
                            target.pricing,
                        );
                        break error.source;
                    }
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

    struct SlowAdapter;

    struct TransientAdapter;

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

    impl ProviderAdapter for SlowAdapter {
        fn name(&self) -> &'static str {
            "slow"
        }

        fn complete<'a>(
            &'a self,
            _request: ModelRequest<'a>,
            _on_delta: Option<DeltaHandler<'a>>,
        ) -> AdapterFuture<'a> {
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(ModelResponse {
                    content: Some("late".to_string()),
                    tool_calls: Vec::new(),
                    usage: Some(Usage::default()),
                })
            })
        }
    }

    impl ProviderAdapter for TransientAdapter {
        fn name(&self) -> &'static str {
            "transient"
        }

        fn complete<'a>(
            &'a self,
            _request: ModelRequest<'a>,
            _on_delta: Option<DeltaHandler<'a>>,
        ) -> AdapterFuture<'a> {
            Box::pin(async move { Err(AdapterError::transient(anyhow::anyhow!("retry me"))) })
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
                pricing: ModelPricing::default(),
                external: false,
                adapter: adapter.clone(),
            }],
            0,
            UsageTracker::unlimited(),
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
    fn usage_tracker_records_cost_attempts_failures_and_missing_usage() {
        let tracker = UsageTracker::unlimited();
        let pricing = ModelPricing {
            input_per_million: Some(2.0),
            output_per_million: Some(8.0),
        };
        tracker.record_attempt(
            "model-a",
            "adapter-a",
            true,
            Duration::from_millis(50),
            Some(Some(Usage {
                prompt_tokens: 1_000,
                completion_tokens: 100,
            })),
            pricing,
        );
        tracker.record_attempt(
            "model-a",
            "adapter-a",
            true,
            Duration::from_millis(20),
            None,
            pricing,
        );
        tracker.record_attempt(
            "model-b",
            "adapter-b",
            false,
            Duration::from_millis(10),
            Some(None),
            ModelPricing::default(),
        );

        let snapshot = tracker.snapshot();
        let model_a = &snapshot.models["model-a"];
        assert_eq!(
            (
                model_a.requests,
                model_a.attempts,
                model_a.failures,
                model_a.prompt_tokens,
                model_a.completion_tokens,
                model_a.latency_ms,
            ),
            (1, 2, 1, 1_000, 100, 70)
        );
        assert!((model_a.cost_usd - 0.0028).abs() < f64::EPSILON);
        assert_eq!(snapshot.models["model-b"].unreported_requests, 1);

        let report = tracker.report_lines().join("\n");
        assert!(
            report.contains("model-a: 1 request(s)/2 attempt(s)"),
            "{report}"
        );
        assert!(report.contains("1 failed"), "{report}");
        assert!(report.contains("1 without usage"), "{report}");
        assert!(report.contains("external"), "{report}");
    }

    #[tokio::test]
    async fn token_budget_stops_a_response_that_overshoots() {
        let tracker = UsageTracker::new(
            RunLimits {
                max_tokens: Some(3),
                ..Default::default()
            },
            UsageSnapshot::default(),
        );
        let adapter = Arc::new(RecordingAdapter {
            seen: Mutex::new(None),
        });
        let client = ModelClient::new(
            vec![ModelTarget {
                label: "budget-test".into(),
                model: "coder".into(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing::default(),
                external: false,
                adapter,
            }],
            0,
            tracker.clone(),
        );
        let error = client
            .complete(
                &[Message::User {
                    content: "work".into(),
                }],
                &[],
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("token budget exhausted"), "{error}");
        assert_eq!(tracker.snapshot().models["budget-test"].requests, 1);
    }

    #[tokio::test]
    async fn cost_and_time_budgets_stop_the_run() {
        let cost_tracker = UsageTracker::new(
            RunLimits {
                max_cost_usd: Some(0.000_001),
                ..Default::default()
            },
            UsageSnapshot::default(),
        );
        let priced = ModelClient::new(
            vec![ModelTarget {
                label: "priced".into(),
                model: "coder".into(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing {
                    input_per_million: Some(1.0),
                    output_per_million: Some(1.0),
                },
                external: true,
                adapter: Arc::new(RecordingAdapter {
                    seen: Mutex::new(None),
                }),
            }],
            0,
            cost_tracker.clone(),
        );
        let error = priced
            .complete(
                &[Message::User {
                    content: "work".into(),
                }],
                &[],
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("cost budget exhausted"), "{error}");
        assert!(cost_tracker.snapshot().models["priced"].cost_usd > 0.000_001);

        let time_tracker = UsageTracker::new(
            RunLimits {
                max_time: Some(Duration::from_millis(10)),
                ..Default::default()
            },
            UsageSnapshot::default(),
        );
        let slow = ModelClient::new(
            vec![ModelTarget {
                label: "slow".into(),
                model: "coder".into(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing::default(),
                external: false,
                adapter: Arc::new(SlowAdapter),
            }],
            0,
            time_tracker.clone(),
        );
        let error = slow
            .complete(
                &[Message::User {
                    content: "wait".into(),
                }],
                &[],
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("time budget exhausted"), "{error}");
        let slow_usage = &time_tracker.snapshot().models["slow"];
        assert_eq!((slow_usage.attempts, slow_usage.failures), (1, 1));

        let retry_tracker = UsageTracker::new(
            RunLimits {
                max_time: Some(Duration::from_millis(20)),
                ..Default::default()
            },
            UsageSnapshot::default(),
        );
        let retrying = ModelClient::new(
            vec![ModelTarget {
                label: "retrying".into(),
                model: "coder".into(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing::default(),
                external: false,
                adapter: Arc::new(TransientAdapter),
            }],
            1,
            retry_tracker,
        );
        let error = retrying
            .complete(
                &[Message::User {
                    content: "wait".into(),
                }],
                &[],
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("time budget exhausted"), "{error}");

        let resumed = UsageTracker::new(
            RunLimits {
                max_time: Some(Duration::from_secs(1)),
                ..Default::default()
            },
            UsageSnapshot {
                elapsed_ms: 1_000,
                ..Default::default()
            },
        );
        let error = resumed
            .within_time(async { Ok::<_, anyhow::Error>(()) })
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("time budget exhausted"), "{error}");
    }
}
