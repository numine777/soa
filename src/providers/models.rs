//! Best-effort model discovery for the chat UI. Both protocols return
//! `data: [{id: ...}]`; Anthropic also paginates with `after_id`.

use std::collections::BTreeSet;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::config::{Provider, ProviderAdapterKind};

#[derive(Deserialize)]
struct ModelPage {
    data: Vec<ModelId>,
    #[serde(default)]
    has_more: bool,
    last_id: Option<String>,
}

#[derive(Deserialize)]
struct ModelId {
    id: String,
}

/// Bound the entire discovery operation, including pagination, separately
/// from the much longer generation timeout. Never retry optional discovery.
pub async fn list_models(provider: &Provider, http: &reqwest::Client) -> Result<Vec<String>> {
    tokio::time::timeout(Duration::from_secs(5), list_pages(provider, http))
        .await
        .context("model discovery timed out")?
}

async fn list_pages(provider: &Provider, http: &reqwest::Client) -> Result<Vec<String>> {
    let url = format!("{}/models", provider.base_url.trim_end_matches('/'));
    let anthropic = provider.adapter == ProviderAdapterKind::AnthropicMessages;
    let mut headers = reqwest::header::HeaderMap::new();
    if anthropic {
        headers.insert("anthropic-version", "2023-06-01".parse()?);
    }
    for (key, value) in &provider.headers {
        headers.insert(
            key.parse::<reqwest::header::HeaderName>()?,
            value.parse::<reqwest::header::HeaderValue>()?,
        );
    }
    let mut models = BTreeSet::new();
    let mut cursors = BTreeSet::new();
    let mut after_id = None;
    loop {
        let mut request = http.get(&url).headers(headers.clone());
        if let Some(key) = &provider.api_key
            && !key.is_empty()
        {
            request = if anthropic {
                request.header("x-api-key", key)
            } else {
                request.bearer_auth(key)
            };
        }
        if let Some(cursor) = &after_id {
            request = request.query(&[("after_id", cursor)]);
        }
        let page: ModelPage = request.send().await?.error_for_status()?.json().await?;
        for model in page.data {
            if !model.id.is_empty() && !model.id.chars().any(char::is_whitespace) {
                models.insert(model.id);
            }
        }
        if !anthropic || !page.has_more {
            return Ok(models.into_iter().collect());
        }
        let cursor = page
            .last_id
            .filter(|id| !id.is_empty())
            .context("model listing omitted its pagination cursor")?;
        if !cursors.insert(cursor.clone()) {
            bail!("model listing repeated its pagination cursor");
        }
        after_id = Some(cursor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn server(
        responses: Vec<(u16, &'static str)>,
    ) -> (String, tokio::task::JoinHandle<Vec<String>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/", listener.local_addr().unwrap());
        let handle = tokio::spawn(async move {
            let mut requests = Vec::new();
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    let byte = stream.read_u8().await.unwrap();
                    request.push(byte);
                }
                let headers = String::from_utf8(request.clone()).unwrap();
                let length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().unwrap())
                    })
                    .unwrap_or(0);
                let mut request_body = vec![0; length];
                stream.read_exact(&mut request_body).await.unwrap();
                request.extend(request_body);
                requests.push(String::from_utf8(request).unwrap());
                let response = format!(
                    "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
            requests
        });
        (url, handle)
    }

    fn provider(url: &str, adapter: ProviderAdapterKind) -> Provider {
        let mut provider: Provider = toml::from_str(&format!("base_url = {url:?}")).unwrap();
        provider.adapter = adapter;
        provider.api_key = Some("test-key".into());
        provider
            .headers
            .insert("x-custom".into(), "custom-value".into());
        provider
    }

    #[tokio::test]
    async fn openai_listing_uses_auth_headers_and_sorted_unique_ids() {
        let (url, server) = server(vec![(
            200,
            r#"{"data":[{"id":"z"},{"id":"org/model"},{"id":"z"},{"id":""},{"id":"bad id"}]}"#,
        )])
        .await;
        let provider = provider(&url, ProviderAdapterKind::OpenAiChatCompletions);
        assert_eq!(
            list_models(&provider, &reqwest::Client::new())
                .await
                .unwrap(),
            ["org/model", "z"]
        );
        let requests = server.await.unwrap();
        assert!(requests[0].starts_with("GET /v1/models HTTP/1.1"));
        assert!(requests[0].contains("authorization: Bearer test-key\r\n"));
        assert!(requests[0].contains("x-custom: custom-value\r\n"));
        assert!(!requests[0].contains("anthropic-version"));
    }

    #[tokio::test]
    async fn anthropic_listing_follows_pages_and_overrides_version() {
        let (url, server) = server(vec![
            (
                200,
                r#"{"data":[{"id":"claude-b"}],"has_more":true,"last_id":"claude-b"}"#,
            ),
            (200, r#"{"data":[{"id":"claude-a"}],"has_more":false}"#),
        ])
        .await;
        let mut provider = provider(&url, ProviderAdapterKind::AnthropicMessages);
        provider
            .headers
            .insert("Anthropic-Version".into(), "custom-version".into());
        assert_eq!(
            list_models(&provider, &reqwest::Client::new())
                .await
                .unwrap(),
            ["claude-a", "claude-b"]
        );
        let requests = server.await.unwrap();
        assert!(requests[1].starts_with("GET /v1/models?after_id=claude-b HTTP/1.1"));
        for request in requests {
            assert!(request.contains("x-api-key: test-key\r\n"));
            assert!(request.contains("anthropic-version: custom-version\r\n"));
            assert!(!request.contains("authorization:"));
        }
    }

    #[tokio::test]
    async fn failed_or_malformed_listings_return_no_partial_catalog() {
        for response in [
            (401, "{}"),
            (404, "{}"),
            (500, "{}"),
            (200, "not json"),
            (200, "{}"),
            (200, r#"{"data":[{}]}"#),
        ] {
            let (url, server) = server(vec![response]).await;
            let provider = provider(&url, ProviderAdapterKind::OpenAiChatCompletions);
            assert!(
                list_models(&provider, &reqwest::Client::new())
                    .await
                    .is_err()
            );
            server.await.unwrap();
        }
        let (url, server) = server(vec![
            (
                200,
                r#"{"data":[{"id":"a"}],"has_more":true,"last_id":"a"}"#,
            ),
            (500, "{}"),
        ])
        .await;
        let provider = provider(&url, ProviderAdapterKind::AnthropicMessages);
        assert!(
            list_models(&provider, &reqwest::Client::new())
                .await
                .is_err()
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn empty_results_and_broken_pagination_are_handled() {
        let (url, handle) = server(vec![(200, r#"{"data":[]}"#)]).await;
        assert!(
            list_models(
                &provider(&url, ProviderAdapterKind::OpenAiChatCompletions),
                &reqwest::Client::new()
            )
            .await
            .unwrap()
            .is_empty()
        );
        handle.await.unwrap();
        for responses in [
            vec![(200, r#"{"data":[],"has_more":true}"#)],
            vec![(200, r#"{"data":[],"has_more":true,"last_id":"a"}"#); 2],
        ] {
            let (url, handle) = server(responses).await;
            assert!(
                list_models(
                    &provider(&url, ProviderAdapterKind::AnthropicMessages),
                    &reqwest::Client::new()
                )
                .await
                .is_err()
            );
            handle.await.unwrap();
        }
    }

    #[tokio::test]
    async fn discovered_and_default_selections_send_the_correct_model_id() {
        let response = r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"#;
        let (url, handle) = server(vec![
            (200, r#"{"data":[{"id":"org/new"}]}"#),
            (200, response),
            (200, response),
        ])
        .await;
        let config: crate::config::Config = toml::from_str(&format!(
            r#"
            [providers.p]
            base_url = {url:?}
            stream = false
            default_model = "configured"
            [models.alias]
            provider = "p"
            model = "configured"
            max_tokens = 17
        "#
        ))
        .unwrap();
        let http = reqwest::Client::new();
        let listed = list_models(&config.providers["p"], &http).await.unwrap();
        for selection in [format!("p/{}", listed[0]), "p/default".into()] {
            let client = crate::stage::build_model_client(
                &config,
                &selection,
                None,
                None,
                &http,
                &crate::model::UsageTracker::unlimited(),
                None,
            )
            .unwrap();
            assert_eq!(
                client.complete(&[], &[]).await.unwrap().content.as_deref(),
                Some("ok")
            );
        }
        let requests = handle.await.unwrap();
        let bodies: Vec<serde_json::Value> = requests[1..]
            .iter()
            .map(|request| {
                assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
                serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
            })
            .collect();
        assert_eq!(bodies[0]["model"], "org/new");
        assert!(bodies[0].get("max_tokens").is_none());
        assert_eq!(bodies[1]["model"], "configured");
        assert_eq!(bodies[1]["max_tokens"], 17);
    }

    #[tokio::test]
    async fn discovery_times_out_even_with_a_long_generation_timeout() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let provider = provider(&url, ProviderAdapterKind::OpenAiChatCompletions);
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(7), list_models(&provider, &http))
            .await
            .unwrap();
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }
}
