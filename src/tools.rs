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
            " Only a single simple command matching one of these patterns is permitted; \
             pipes, command lists, redirections, subshells, and command substitutions are \
             rejected: {}.",
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

/// Whether `command` is one simple shell command rather than a compound
/// expression. Quoted or escaped metacharacters are literal and therefore
/// safe; executable control operators, redirections, subshells, and command
/// substitutions are not.
///
/// This is deliberately narrower than a complete POSIX-shell parser. Its
/// purpose is to keep broad patterns such as `cargo *` from also authorizing
/// `cargo test; dangerous-command` while retaining ordinary quoting.
pub fn shell_command_is_simple(command: &str) -> bool {
    #[derive(Clone, Copy)]
    enum Quote {
        Single,
        Double,
    }

    let mut quote = None;
    let mut chars = command.chars().peekable();
    while let Some(ch) = chars.next() {
        match quote {
            Some(Quote::Single) => {
                if ch == '\'' {
                    quote = None;
                }
            }
            Some(Quote::Double) => match ch {
                '"' => quote = None,
                '\\' => {
                    // An escaped character cannot terminate the quote or
                    // become active shell syntax.
                    chars.next();
                }
                '`' => return false,
                '$' if chars.peek() == Some(&'(') => return false,
                _ => {}
            },
            None => match ch {
                '\'' => quote = Some(Quote::Single),
                '"' => quote = Some(Quote::Double),
                '\\' => {
                    // Escaped metacharacters are literal arguments.
                    chars.next();
                }
                ';' | '|' | '&' | '<' | '>' | '(' | ')' | '\n' | '\r' | '`' => {
                    return false;
                }
                '$' if chars.peek() == Some(&'(') => return false,
                _ => {}
            },
        }
    }
    true
}

/// An empty allowlist permits everything (the stage already opted in with
/// `shell = true`); otherwise the command must be a single simple command
/// matching one pattern.
pub fn command_allowed(allow: &[String], command: &str) -> bool {
    allow.is_empty()
        || (shell_command_is_simple(command)
            && allow.iter().any(|pattern| wildcard_match(pattern, command.trim())))
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

        // A textual prefix match must not grant a second command, a pipe,
        // redirection, subshell, or command substitution.
        for command in [
            "cargo test; rm -rf important",
            "cargo test && curl https://example.invalid",
            "cargo test | tee result",
            "cargo test > result",
            "cargo test\nrm -rf important",
            "cargo test $(dangerous-command)",
            "cargo test `dangerous-command`",
            "cargo test (dangerous-command)",
        ] {
            assert!(!command_allowed(&allow, command), "allowed: {command}");
        }

        // Metacharacters that the shell treats literally remain ordinary
        // arguments; command substitution stays inert inside single quotes.
        assert!(command_allowed(&allow, r#"cargo test "name;with;semicolons""#));
        assert!(command_allowed(&allow, r"cargo test escaped\;semicolon"));
        assert!(command_allowed(&allow, "cargo test '$(literal)'"));
        assert!(!command_allowed(&allow, r#"cargo test "$(dangerous-command)""#));
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
