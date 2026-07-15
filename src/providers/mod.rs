//! Provider wire adapters for the canonical model API.

mod anthropic_messages;
mod openai_chat_completions;
mod sse;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use crate::config::{Provider, ProviderAdapterKind};
use crate::model::ProviderAdapter;

/// Transient HTTP statuses shared by every adapter: server errors, rate
/// limiting, and request timeouts (including Anthropic's 529 overloaded,
/// which is a 5xx).
pub(crate) fn retryable_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status == reqwest::StatusCode::REQUEST_TIMEOUT
}

pub(crate) fn parse_retry_after(response: &reqwest::Response) -> Option<Duration> {
    let value = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?;
    let seconds: u64 = value.trim().parse().ok()?;
    Some(Duration::from_secs(seconds.min(30)))
}

/// Construct the adapter selected by one `[providers.*]` entry.
pub fn build_adapter(provider: &Provider, http: reqwest::Client) -> Result<Arc<dyn ProviderAdapter>> {
    Ok(match provider.adapter {
        ProviderAdapterKind::OpenAiChatCompletions => {
            Arc::new(openai_chat_completions::OpenAiChatCompletions::new(
                http,
                &provider.base_url,
                provider.api_key.clone(),
                &provider.headers,
            )?)
        }
        ProviderAdapterKind::AnthropicMessages => {
            Arc::new(anthropic_messages::AnthropicMessages::new(
                http,
                &provider.base_url,
                provider.api_key.clone(),
                &provider.headers,
            )?)
        }
    })
}
