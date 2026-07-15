//! User-configured shell commands bound to tool-call events.
//!
//! `[[hooks]]` entries match tool calls by the same wildcard descriptor
//! that approvals use (`edit_file *`, `shell cargo *`), so one pattern
//! language covers both. A `pre_tool` hook that exits non-zero blocks the
//! call; a `post_tool` hook that exits non-zero has its output appended to
//! the tool result, which is how lint-after-edit feedback reaches the
//! model. Hooks receive a JSON payload on stdin and SOA_* env vars.

use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};

use crate::config::{Config, Hook, HookEvent};

/// Run the matching `pre_tool` hooks; the first failure blocks the call
/// and its message (fed back to the model) is returned. A hook that times
/// out or cannot be spawned also blocks — pre hooks fail closed, because
/// blocking policies must not be skippable by breaking the hook.
pub async fn pre_tool(config: &Config, descriptor: &str, arguments: &Value) -> Option<String> {
    for hook in matching(config, HookEvent::PreTool, descriptor) {
        let run = run_hook(config, hook, "pre_tool", descriptor, arguments, None).await;
        match run {
            HookRun { exit: Some(0), .. } => {}
            HookRun { exit, output } => {
                tracing::warn!(
                    pattern = %hook.pattern, exit = ?exit, "pre_tool hook blocked call"
                );
                let reason = if output.is_empty() {
                    String::new()
                } else {
                    format!(": {output}")
                };
                return Some(format!(
                    "BLOCKED: a pre_tool hook ({}) rejected this call{reason}\n\
                     Do not retry the same call; adjust your approach.",
                    describe_exit(exit),
                ));
            }
        }
    }
    None
}

/// Run the matching `post_tool` hooks; failures append their output to the
/// tool result so the model sees it.
pub async fn post_tool(
    config: &Config,
    descriptor: &str,
    arguments: &Value,
    mut output: String,
) -> String {
    for hook in matching(config, HookEvent::PostTool, descriptor) {
        let run =
            run_hook(config, hook, "post_tool", descriptor, arguments, Some(&output)).await;
        match run {
            HookRun { exit: Some(0), .. } => {}
            HookRun { exit, output: hook_output } => {
                tracing::warn!(
                    pattern = %hook.pattern, exit = ?exit, "post_tool hook reported a problem"
                );
                output.push_str(&format!(
                    "\n\n[post_tool hook `{}` {}]\n{hook_output}",
                    hook.pattern,
                    describe_exit(exit),
                ));
            }
        }
    }
    output
}

fn matching<'a>(
    config: &'a Config,
    event: HookEvent,
    descriptor: &'a str,
) -> impl Iterator<Item = &'a Hook> {
    config
        .hooks
        .iter()
        .filter(move |h| h.event == event && crate::tools::wildcard_match(&h.pattern, descriptor))
}

fn describe_exit(exit: Option<i32>) -> String {
    match exit {
        Some(code) => format!("exited {code}"),
        None => "timed out or failed to run".to_string(),
    }
}

struct HookRun {
    /// Exit code; None on timeout, signal death, or spawn failure.
    exit: Option<i32>,
    /// Combined stdout+stderr, trimmed and clamped.
    output: String,
}

async fn run_hook(
    config: &Config,
    hook: &Hook,
    event: &str,
    descriptor: &str,
    arguments: &Value,
    tool_output: Option<&str>,
) -> HookRun {
    let tool = descriptor.split_whitespace().next().unwrap_or("");
    let payload = json!({
        "event": event,
        "tool": tool,
        "descriptor": descriptor,
        "arguments": arguments,
        "output": tool_output,
    })
    .to_string();
    let paths = crate::diff::extract_paths(arguments).join("\n");

    let timeout = Duration::from_secs(
        hook.timeout_secs.unwrap_or(config.settings.shell_timeout_secs),
    );
    let spawned = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&hook.command)
        .env("SOA_EVENT", event)
        .env("SOA_TOOL", tool)
        .env("SOA_DESCRIPTOR", descriptor)
        .env("SOA_PATHS", paths)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn();
    let mut child = match spawned {
        Ok(child) => child,
        Err(e) => {
            return HookRun { exit: None, output: format!("hook failed to start: {e}") };
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        use tokio::io::AsyncWriteExt;
        let _ = stdin.write_all(payload.as_bytes()).await;
        // stdin drops here, closing the pipe so hooks that read it finish.
    }

    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Err(_) => HookRun {
            exit: None,
            output: format!("hook timed out after {}s", timeout.as_secs()),
        },
        Ok(Err(e)) => HookRun { exit: None, output: format!("hook failed: {e}") },
        Ok(Ok(result)) => {
            let mut text = String::from_utf8_lossy(&result.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&result.stderr);
            let stderr = stderr.trim();
            if !stderr.is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(stderr);
            }
            if text.chars().count() > 2000 {
                text = text.chars().take(2000).collect::<String>() + "…";
            }
            HookRun { exit: result.status.code(), output: text }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_hooks(hooks_toml: &str) -> Config {
        let toml_str = format!(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [[stage]]
            name = "s"
            model = "m"

            {hooks_toml}
            "#
        );
        toml::from_str(&toml_str).unwrap()
    }

    #[tokio::test]
    async fn pre_hook_blocks_on_failure_and_passes_on_success() {
        let config = config_with_hooks(
            r#"
            [[hooks]]
            event = "pre_tool"
            match = "edit_file *"
            command = "echo protected file >&2; exit 1"

            [[hooks]]
            event = "pre_tool"
            match = "shell *"
            command = "true"
            "#,
        );
        let blocked = pre_tool(&config, "edit_file src/x.rs", &serde_json::json!({})).await.unwrap();
        assert!(blocked.contains("BLOCKED"), "{blocked}");
        assert!(blocked.contains("protected file"), "{blocked}");
        // Non-matching descriptor: the failing hook is not consulted.
        assert!(pre_tool(&config, "shell cargo test", &serde_json::json!({})).await.is_none());
        assert!(pre_tool(&config, "read_file src/x.rs", &serde_json::json!({})).await.is_none());
    }

    #[tokio::test]
    async fn post_hook_appends_feedback_and_sees_payload() {
        let config = config_with_hooks(
            r#"
            [[hooks]]
            event = "post_tool"
            match = "write_file *"
            command = "echo lint failed in $SOA_PATHS; cat > /dev/null; exit 1"

            [[hooks]]
            event = "post_tool"
            match = "write_file *"
            command = "true"
            "#,
        );
        let result = post_tool(
            &config,
            "write_file src/x.rs",
            &serde_json::json!({"path": "src/x.rs", "content": "x"}),
            "created `src/x.rs` (1 bytes)".to_string(),
        )
        .await;
        assert!(result.starts_with("created `src/x.rs`"), "{result}");
        assert!(result.contains("[post_tool hook `write_file *` exited 1]"), "{result}");
        assert!(result.contains("lint failed in src/x.rs"), "{result}");
        // A passing hook appends nothing.
        let clean =
            post_tool(&config, "write_file y.rs", &serde_json::json!({"path":"y.rs"}), "ok".into())
                .await;
        assert!(clean.contains("[post_tool hook"), "failing hook still matches y.rs");
    }

    #[tokio::test]
    async fn hook_timeout_blocks_pre_calls() {
        let config = config_with_hooks(
            r#"
            [[hooks]]
            event = "pre_tool"
            command = "sleep 5"
            timeout_secs = 1
            "#,
        );
        let blocked = pre_tool(&config, "anything", &serde_json::json!({})).await.unwrap();
        assert!(blocked.contains("timed out"), "{blocked}");
    }

    #[tokio::test]
    async fn hook_reads_stdin_payload() {
        let config = config_with_hooks(
            r#"
            [[hooks]]
            event = "pre_tool"
            command = "grep -q '\"tool\":\"shell\"' && exit 1 || exit 0"
            "#,
        );
        // The stdin payload names the tool, so the hook can block on it.
        assert!(pre_tool(&config, "shell rm -rf /", &serde_json::json!({})).await.is_some());
        assert!(pre_tool(&config, "read_file x", &serde_json::json!({})).await.is_none());
    }
}
