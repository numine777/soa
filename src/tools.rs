//! Built-in tools. Currently: web search backed by a SearXNG instance.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::json;

use crate::provider::ToolFunction;

pub const WEB_SEARCH_TOOL: &str = "web_search";

pub fn web_search_definition() -> ToolFunction {
    ToolFunction {
        name: WEB_SEARCH_TOOL.to_string(),
        description: "Search the web. Returns a numbered list of results with title, URL, and snippet.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "The search query" }
            },
            "required": ["query"]
        }),
    }
}

#[derive(Deserialize)]
struct SearxResponse {
    #[serde(default)]
    results: Vec<SearxResult>,
}

#[derive(Deserialize)]
struct SearxResult {
    #[serde(default)]
    title: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    content: String,
}

pub async fn web_search(
    http: &reqwest::Client,
    searxng_url: &str,
    query: &str,
    max_results: usize,
) -> Result<String> {
    let url = format!("{}/search", searxng_url.trim_end_matches('/'));
    let response = http
        .get(&url)
        .query(&[("q", query), ("format", "json")])
        .send()
        .await
        .with_context(|| format!("searxng request to {url} failed"))?;

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        bail!(
            "searxng returned {status} from {url}: {} (is `formats: [html, json]` enabled in searxng settings?)",
            body.chars().take(200).collect::<String>()
        );
    }

    let parsed: SearxResponse = serde_json::from_str(&body)
        .with_context(|| format!("unexpected searxng response from {url}"))?;

    if parsed.results.is_empty() {
        return Ok(format!("No results for query: {query}"));
    }

    let formatted = parsed
        .results
        .iter()
        .take(max_results)
        .enumerate()
        .map(|(i, r)| format!("{}. {}\n   {}\n   {}", i + 1, r.title, r.url, r.content))
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(formatted)
}
