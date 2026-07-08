//! Built-in tools: web search backed by a SearXNG instance, and a
//! config-gated shell executor.

use std::time::Duration;

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

pub const SHELL_TOOL: &str = "shell";

pub fn shell_definition(timeout_secs: u64, allow: &[String]) -> ToolFunction {
    let mut description = format!(
        "Run a shell command (`sh -c`) in the project working directory and get its \
         exit code, stdout, and stderr. Commands are killed after {timeout_secs}s."
    );
    if !allow.is_empty() {
        description.push_str(&format!(
            " Only commands matching these patterns are permitted: {}.",
            allow.join(", ")
        ));
    }
    ToolFunction {
        name: SHELL_TOOL.to_string(),
        description,
        parameters: json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The shell command to run" }
            },
            "required": ["command"]
        }),
    }
}

/// `*`-wildcard match, anchored at both ends (`"cargo *"` matches
/// `"cargo test --all"` but not `"echo cargo x"`).
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return pattern == text;
    }
    let first = segments[0];
    let last = segments[segments.len() - 1];
    if !text.starts_with(first) || !text.ends_with(last) {
        return false;
    }
    let mut position = first.len();
    let Some(end_limit) = text.len().checked_sub(last.len()) else { return false };
    if position > end_limit {
        return false;
    }
    for segment in &segments[1..segments.len() - 1] {
        if segment.is_empty() {
            continue;
        }
        match text[position..end_limit].find(segment) {
            Some(found) => position += found + segment.len(),
            None => return false,
        }
    }
    position <= end_limit
}

/// An empty allowlist permits everything (the stage already opted in with
/// `shell = true`); otherwise the command must match one pattern.
pub fn command_allowed(allow: &[String], command: &str) -> bool {
    allow.is_empty() || allow.iter().any(|pattern| wildcard_match(pattern, command.trim()))
}

/// Run a command, capturing everything. Failures (bad exit, timeout, spawn
/// error) are reported in the returned text so the model can react.
pub async fn run_shell(command: &str, timeout: Duration) -> String {
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(command)
        .kill_on_drop(true)
        .output();
    let output = match tokio::time::timeout(timeout, child).await {
        Err(_) => {
            return format!("ERROR: command timed out after {}s and was killed", timeout.as_secs());
        }
        Ok(Err(e)) => return format!("ERROR: failed to run command: {e}"),
        Ok(Ok(output)) => output,
    };

    let mut report = format!(
        "exit code: {}",
        output.status.code().map_or_else(|| "killed by signal".to_string(), |c| c.to_string())
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stdout.trim().is_empty() {
        report.push_str(&format!("\nstdout:\n{}", stdout.trim_end()));
    }
    if !stderr.trim().is_empty() {
        report.push_str(&format!("\nstderr:\n{}", stderr.trim_end()));
    }
    if stdout.trim().is_empty() && stderr.trim().is_empty() {
        report.push_str("\n(no output)");
    }
    report
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wildcard_matching() {
        assert!(wildcard_match("cargo *", "cargo test --all"));
        assert!(!wildcard_match("cargo *", "echo cargo test"));
        assert!(wildcard_match("cargo test", "cargo test"));
        assert!(!wildcard_match("cargo test", "cargo test --all"));
        assert!(wildcard_match("git * --dry-run", "git push origin --dry-run"));
        assert!(!wildcard_match("git * --dry-run", "git push origin"));
        assert!(wildcard_match("*", "anything at all"));
        assert!(!wildcard_match("cargo *", "cargo")); // needs the trailing space
    }

    #[test]
    fn allowlist_semantics() {
        assert!(command_allowed(&[], "rm -rf /tmp/x")); // empty = opted-in, unrestricted
        let allow = vec!["cargo *".to_string(), "git status".to_string()];
        assert!(command_allowed(&allow, "cargo build"));
        assert!(command_allowed(&allow, "  git status  ")); // trimmed
        assert!(!command_allowed(&allow, "git push"));
    }

    #[tokio::test]
    async fn shell_execution_and_timeout() {
        let ok = run_shell("echo hello; echo oops >&2", Duration::from_secs(5)).await;
        assert!(ok.contains("exit code: 0"), "{ok}");
        assert!(ok.contains("stdout:\nhello"), "{ok}");
        assert!(ok.contains("stderr:\noops"), "{ok}");

        let failed = run_shell("exit 3", Duration::from_secs(5)).await;
        assert!(failed.contains("exit code: 3"), "{failed}");
        assert!(failed.contains("(no output)"), "{failed}");

        let timed_out = run_shell("sleep 5", Duration::from_secs(1)).await;
        assert!(timed_out.contains("timed out after 1s"), "{timed_out}");
    }
}
