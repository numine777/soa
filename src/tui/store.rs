//! On-disk persistence for chat sessions and the prompt history.
//!
//! Layout under the data directory (`$XDG_DATA_HOME/soa` or
//! `~/.local/share/soa`):
//!   sessions/<id>.json     one file per chat session, rewritten as it grows
//!   prompt_history.jsonl   every submitted prompt, one JSON string per line

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::app::TranscriptItem;
use crate::diff::DiffEntry;
use crate::provider::ChatMessage;

const PROMPT_HISTORY_LIMIT: usize = 1000;

pub fn data_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
            home.join(".local").join("share")
        });
    base.join("soa")
}

fn sessions_dir() -> PathBuf {
    data_dir().join("sessions")
}

pub fn current_cwd() -> String {
    std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Epoch seconds → (year, month, day, hour, minute, second) in UTC.
/// Days-to-civil conversion per Howard Hinnant's algorithm.
fn civil(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hour, minute, second) =
        ((rem / 3600) as u32, ((rem % 3600) / 60) as u32, (rem % 60) as u32);

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day, hour, minute, second)
}

pub fn format_epoch(secs: u64) -> String {
    let (y, mo, d, h, mi, _) = civil(secs);
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02} UTC")
}

fn session_id_for(secs: u64) -> String {
    let (y, mo, d, h, mi, s) = civil(secs);
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}

/// A unique id for a session started now (suffixed on same-second collision).
pub fn new_session_id() -> String {
    let base = session_id_for(now_epoch());
    let mut id = base.clone();
    let mut n = 1;
    while sessions_dir().join(format!("{id}.json")).exists() {
        n += 1;
        id = format!("{base}-{n}");
    }
    id
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub started_at: u64,
    pub updated_at: u64,
    /// Active stage name when last saved.
    pub stage: String,
    /// First user prompt, for listings.
    pub title: String,
    /// Working directory the session was created in; the in-TUI picker only
    /// shows sessions for the current directory. Empty on legacy sessions.
    #[serde(default)]
    pub cwd: String,
    pub history: Vec<ChatMessage>,
    pub transcript: Vec<TranscriptItem>,
    pub diffs: Vec<DiffEntry>,
}

pub fn save_session(session: &Session) -> Result<()> {
    let dir = sessions_dir();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("cannot create {}", dir.display()))?;
    let path = dir.join(format!("{}.json", session.id));
    let json = serde_json::to_string(session)?;
    std::fs::write(&path, json).with_context(|| format!("cannot write {}", path.display()))
}

pub fn load_session(id: &str) -> Result<Session> {
    let path = sessions_dir().join(format!("{id}.json"));
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("no session `{id}` at {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("corrupt session file {}", path.display()))
}

/// All sessions, most recently updated first.
pub fn list_sessions() -> Result<Vec<Session>> {
    let dir = sessions_dir();
    let mut sessions = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(sessions), // no directory yet: no sessions
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json")
            && let Ok(raw) = std::fs::read_to_string(&path)
            && let Ok(session) = serde_json::from_str::<Session>(&raw)
        {
            sessions.push(session);
        }
    }
    sessions.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
    Ok(sessions)
}

pub fn load_latest_session() -> Result<Option<Session>> {
    Ok(list_sessions()?.into_iter().next())
}

/// Submitted prompts, oldest first; appended to disk as they happen.
pub struct PromptHistory {
    path: PathBuf,
    pub entries: Vec<String>,
}

impl PromptHistory {
    pub fn load() -> PromptHistory {
        let path = data_dir().join("prompt_history.jsonl");
        let mut entries: Vec<String> = std::fs::read_to_string(&path)
            .map(|raw| {
                raw.lines()
                    .filter_map(|line| serde_json::from_str::<String>(line).ok())
                    .collect()
            })
            .unwrap_or_default();
        if entries.len() > PROMPT_HISTORY_LIMIT {
            entries.drain(..entries.len() - PROMPT_HISTORY_LIMIT);
        }
        PromptHistory { path, entries }
    }

    /// Record a submitted prompt (skipping consecutive duplicates) and
    /// append it to the history file.
    pub fn push(&mut self, prompt: &str) {
        if self.entries.last().is_some_and(|last| last == prompt) {
            return;
        }
        self.entries.push(prompt.to_string());
        if let Some(dir) = self.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(line) = serde_json::to_string(prompt) {
            use std::io::Write;
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)
            {
                let _ = writeln!(file, "{line}");
            }
        }
    }
}

/// `XDG_DATA_HOME` is process-global, so tests that touch it (here and in
/// other modules) must not overlap.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    fn with_temp_data_dir<T>(tag: &str, test: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir()
            .join(format!("soa-store-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: serialized by ENV_LOCK.
        unsafe { std::env::set_var("XDG_DATA_HOME", &dir) };
        let result = test();
        let _ = std::fs::remove_dir_all(&dir);
        result
    }

    #[test]
    fn epoch_formatting() {
        // 2026-07-07 18:30:05 UTC
        assert_eq!(format_epoch(1_783_449_005), "2026-07-07 18:30 UTC");
        assert_eq!(session_id_for(1_783_449_005), "20260707-183005");
        assert_eq!(format_epoch(0), "1970-01-01 00:00 UTC");
    }

    #[test]
    fn session_roundtrip_and_listing() {
        with_temp_data_dir("sessions", || {
            let session = Session {
                id: "20260707-120000".to_string(),
                started_at: 100,
                updated_at: 200,
                stage: "review".to_string(),
                title: "fix the widget".to_string(),
                cwd: "/tmp/proj".to_string(),
                history: vec![ChatMessage::User { content: "hi".to_string() }],
                transcript: vec![TranscriptItem::User("hi".to_string())],
                diffs: vec![],
            };
            save_session(&session).unwrap();
            let loaded = load_session("20260707-120000").unwrap();
            assert_eq!(loaded.title, "fix the widget");
            assert_eq!(loaded.history.len(), 1);
            let latest = load_latest_session().unwrap().unwrap();
            assert_eq!(latest.id, session.id);
        });
    }

    #[test]
    fn prompt_history_roundtrip() {
        with_temp_data_dir("prompts", || {
            let mut history = PromptHistory::load();
            assert!(history.entries.is_empty());
            history.push("first");
            history.push("multi\nline");
            history.push("multi\nline"); // consecutive duplicate dropped
            let reloaded = PromptHistory::load();
            assert_eq!(reloaded.entries, vec!["first", "multi\nline"]);
        });
    }
}
