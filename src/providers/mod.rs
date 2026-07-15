//! Provider wire adapters for the canonical model API.

mod openai_chat_completions;
mod sse;

use std::sync::Arc;

use anyhow::Result;

use crate::config::{Provider, ProviderAdapterKind};
use crate::model::ProviderAdapter;

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
    })
}
