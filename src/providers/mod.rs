//! Provider wire adapters for the canonical model API.

mod openai_chat_completions;

use std::sync::Arc;

use crate::config::{Provider, ProviderAdapterKind};
use crate::model::ProviderAdapter;

/// Construct the adapter selected by one `[providers.*]` entry.
pub fn build_adapter(provider: &Provider, http: reqwest::Client) -> Arc<dyn ProviderAdapter> {
    match provider.adapter {
        ProviderAdapterKind::OpenAiChatCompletions => {
            Arc::new(openai_chat_completions::OpenAiChatCompletions::new(
                http,
                &provider.base_url,
                provider.api_key.clone(),
            ))
        }
    }
}
