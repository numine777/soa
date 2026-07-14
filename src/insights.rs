//! Signals mined from saved chat sessions, and their persistent store.
//!
//! A "signal" is ground truth about how a session actually went: the user
//! denied a tool call, a tool call failed, a file change was rolled back.
//! `soa reflect` turns them into lessons; the JSONL store keeps every
//! extracted signal so later tooling (search, embedding, evals) can build
//! on the same record without re-parsing sessions.
//!
//! Layout under the data directory:
//!   insights.jsonl    every extracted signal, one JSON object per line
//!   reflected.json    session id -> updated_at already reflected on

use std::collections::BTreeMap;
use std::io::Write;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::model::Message;
use crate::tui::store::{self, Session};

/// How long an excerpt is kept: enough to understand the failure, short
/// enough that hundreds of them fit in a reflect prompt.
const EXCERPT_CHARS: usize = 240;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalKind {
    /// The user declined the call at the approval prompt.
    Denied,
    /// The tool returned an `ERROR:` result (bad arguments, stale anchor,
    /// missing file, disallowed command, …).
    ToolError,
    /// A recorded file change was rolled back (diff-view restore or
    /// /rewind) — the strongest "that change was wrong" signal there is.
    Rollback,
    /// A commit explicitly reverting earlier work (mined from git).
    Revert,
    /// A commit rewriting lines a recent commit introduced — a candidate
    /// fix-up of someone's work, whichever tool produced it.
    Correction,
    /// A commit changing lines soa's diff log recorded as written by soa —
    /// direct downstream feedback on soa's own output.
    SoaChangeRevised,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Originating session, or empty for signals mined from git.
    #[serde(default)]
    pub session_id: String,
    pub cwd: String,
    /// Session `updated_at` (or commit author time) when extracted —
    /// signals carry no timestamps of their own.
    pub at: u64,
    pub kind: SignalKind,
    /// Advertised name of the tool involved (`git` for mined commits,
    /// empty when unknown).
    pub tool: String,
    /// Full hash of the commit a git-mined signal points at.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub commit: String,
    pub excerpt: String,
}

/// Pull every signal out of a saved session: DENIED/ERROR tool results
/// (matched back to the tool call that produced them) and rollback entries
/// in the diff log.
pub fn extract_signals(session: &Session) -> Vec<Signal> {
    let mut signals = Vec::new();
    let make = |kind, tool: &str, excerpt: &str| Signal {
        session_id: session.id.clone(),
        cwd: session.cwd.clone(),
        at: session.updated_at,
        kind,
        tool: tool.to_string(),
        commit: String::new(),
        excerpt: truncate(excerpt, EXCERPT_CHARS),
    };

    // Tool results carry only a call id; walk the history keeping the
    // id -> tool-name map from each assistant message's tool_calls.
    let mut call_names: BTreeMap<&str, &str> = BTreeMap::new();
    for message in &session.history {
        match message {
            Message::Assistant { tool_calls: Some(calls), .. } => {
                for call in calls {
                    call_names.insert(&call.id, &call.function.name);
                }
            }
            Message::Tool { content, tool_call_id } => {
                let kind = if content.starts_with("DENIED:") {
                    Some(SignalKind::Denied)
                } else if content.starts_with("ERROR:") {
                    Some(SignalKind::ToolError)
                } else {
                    None
                };
                if let Some(kind) = kind {
                    let tool = call_names.get(tool_call_id.as_str()).copied().unwrap_or("");
                    signals.push(make(kind, tool, content));
                }
            }
            _ => {}
        }
    }

    // Restores are recorded in the diff log as reverse entries with the
    // tool name `rewind`.
    for entry in &session.diffs {
        if entry.tool == "rewind" {
            signals.push(make(
                SignalKind::Rollback,
                "rewind",
                &format!("change to {} was rolled back", entry.path),
            ));
        }
    }

    signals
}

fn truncate(text: &str, max_chars: usize) -> String {
    let squashed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if squashed.chars().count() <= max_chars {
        squashed
    } else {
        let cut: String = squashed.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

fn insights_path() -> std::path::PathBuf {
    store::data_dir().join("insights.jsonl")
}

fn reflected_path() -> std::path::PathBuf {
    store::data_dir().join("reflected.json")
}

fn git_marks_path() -> std::path::PathBuf {
    store::data_dir().join("git_reflected.json")
}

/// Append signals to the persistent store, one JSON object per line.
pub fn append_signals(signals: &[Signal]) -> Result<()> {
    if signals.is_empty() {
        return Ok(());
    }
    let path = insights_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("cannot create {}", dir.display()))?;
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("cannot open {}", path.display()))?;
    for signal in signals {
        writeln!(file, "{}", serde_json::to_string(signal)?)
            .with_context(|| format!("cannot write {}", path.display()))?;
    }
    Ok(())
}

/// Which sessions have been reflected on, and at what `updated_at`. A
/// session that grew since is reflected again.
pub fn load_reflected() -> BTreeMap<String, u64> {
    std::fs::read_to_string(reflected_path())
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

pub fn save_reflected(reflected: &BTreeMap<String, u64>) -> Result<()> {
    write_json(&reflected_path(), reflected)
}

/// Per-repository git high-water marks: repo root -> last commit hash
/// already reflected on. Commits are mined from that mark to HEAD.
pub fn load_git_marks() -> BTreeMap<String, String> {
    std::fs::read_to_string(git_marks_path())
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

pub fn save_git_marks(marks: &BTreeMap<String, String>) -> Result<()> {
    write_json(&git_marks_path(), marks)
}

fn write_json<T: Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("cannot create {}", dir.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("cannot write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FunctionCall, ToolCall};

    fn session_with(history: Vec<Message>, diffs: Vec<crate::diff::DiffEntry>) -> Session {
        Session {
            id: "20260713-120000".to_string(),
            started_at: 1,
            updated_at: 2,
            stage: "implement".to_string(),
            title: "t".to_string(),
            cwd: "/tmp/proj".to_string(),
            history,
            transcript: Vec::new(),
            diffs,
            checkpoints: Vec::new(),
            branches: Vec::new(),
            transcript_baseline: 0,
            diff_baseline: 0,
        }
    }

    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            function: FunctionCall { name: name.to_string(), arguments: "{}".to_string() },
        }
    }

    #[test]
    fn extracts_denials_errors_and_rollbacks() {
        let history = vec![
            Message::User { content: "do it".into() },
            Message::Assistant {
                content: None,
                tool_calls: Some(vec![tool_call("1", "write_file"), tool_call("2", "grep")]),
            },
            Message::Tool {
                content: "DENIED: the user declined `write_file src/x.rs`.".into(),
                tool_call_id: "1".into(),
            },
            Message::Tool { content: "3 matches".into(), tool_call_id: "2".into() },
            Message::Assistant {
                content: None,
                tool_calls: Some(vec![tool_call("3", "edit_lines")]),
            },
            Message::Tool {
                content: "ERROR: stale anchor `4:9f3a` — re-read the file.".into(),
                tool_call_id: "3".into(),
            },
        ];
        let diffs = vec![
            crate::diff::DiffEntry {
                tool: "edit_file".into(),
                path: "src/x.rs".into(),
                unified: String::new(),
                added: 1,
                removed: 0,
                before: crate::diff::Snapshot::Absent,
            },
            crate::diff::DiffEntry {
                tool: "rewind".into(),
                path: "src/x.rs".into(),
                unified: String::new(),
                added: 0,
                removed: 1,
                before: crate::diff::Snapshot::Absent,
            },
        ];
        let signals = extract_signals(&session_with(history, diffs));
        let kinds: Vec<(SignalKind, &str)> =
            signals.iter().map(|s| (s.kind, s.tool.as_str())).collect();
        assert_eq!(
            kinds,
            vec![
                (SignalKind::Denied, "write_file"),
                (SignalKind::ToolError, "edit_lines"),
                (SignalKind::Rollback, "rewind"),
            ]
        );
        assert!(signals[0].excerpt.starts_with("DENIED:"));
        assert_eq!(signals[0].session_id, "20260713-120000");
        // Successful tool results and plain messages yield nothing.
        assert!(extract_signals(&session_with(
            vec![Message::User { content: "hi".into() }],
            Vec::new()
        ))
        .is_empty());
    }

    #[test]
    fn excerpts_are_squashed_and_capped() {
        let long = format!("ERROR: {}", "word ".repeat(200));
        let history = vec![
            Message::Assistant {
                content: None,
                tool_calls: Some(vec![tool_call("1", "shell")]),
            },
            Message::Tool { content: long, tool_call_id: "1".into() },
        ];
        let signals = extract_signals(&session_with(history, Vec::new()));
        assert_eq!(signals.len(), 1);
        assert!(signals[0].excerpt.chars().count() <= EXCERPT_CHARS + 1);
        assert!(!signals[0].excerpt.contains('\n'));
    }
}
