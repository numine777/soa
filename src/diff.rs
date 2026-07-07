//! Capture file changes made by write tools during a conversation.
//!
//! Tool calls are opaque, so this works heuristically: before dispatching a
//! non-read-only MCP tool we snapshot every file named by a path-like
//! argument; afterwards we re-read them and record a unified diff for each
//! file that changed.

use serde_json::Value;
use similar::{ChangeTag, TextDiff};

/// Argument keys that plausibly name a file the tool will touch.
const PATH_KEYS: &[&str] = &[
    "path",
    "file_path",
    "filepath",
    "filename",
    "file",
    "source",
    "destination",
    "target_path",
    "output_path",
];

/// Files larger than this are not snapshotted (diffing them is unhelpful).
const MAX_SNAPSHOT_BYTES: u64 = 1_000_000;

#[derive(Debug, Clone)]
pub struct DiffEntry {
    /// Advertised name of the tool that made the change.
    pub tool: String,
    pub path: String,
    /// Unified diff with `a/<path>` / `b/<path>` headers.
    pub unified: String,
    pub added: usize,
    pub removed: usize,
}

impl DiffEntry {
    pub fn title(&self) -> String {
        format!("{} (+{} −{})", self.path, self.added, self.removed)
    }
}

/// Pull candidate file paths out of a tool call's JSON arguments.
pub fn extract_paths(arguments_json: &str) -> Vec<String> {
    let Ok(Value::Object(args)) = serde_json::from_str::<Value>(arguments_json) else {
        return Vec::new();
    };
    let mut paths: Vec<String> = Vec::new();
    for key in PATH_KEYS {
        if let Some(path) = args.get(*key).and_then(Value::as_str)
            && !paths.iter().any(|p| p == path)
        {
            paths.push(path.to_string());
        }
    }
    paths
}

/// The content of a file, or `None` if it's absent, non-UTF-8, or too large.
pub fn read_text(path: &str) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > MAX_SNAPSHOT_BYTES {
        return None;
    }
    std::fs::read_to_string(path).ok()
}

/// Snapshot the named files' current contents.
pub fn snapshot(paths: &[String]) -> Vec<(String, Option<String>)> {
    paths.iter().map(|p| (p.clone(), read_text(p))).collect()
}

/// Re-read snapshotted files and produce a diff entry per changed file.
pub fn collect_changes(tool: &str, snapshots: Vec<(String, Option<String>)>) -> Vec<DiffEntry> {
    snapshots
        .into_iter()
        .filter_map(|(path, before)| {
            let after = read_text(&path);
            if before == after {
                return None;
            }
            Some(compute(tool, &path, before.as_deref(), after.as_deref()))
        })
        .collect()
}

fn compute(tool: &str, path: &str, before: Option<&str>, after: Option<&str>) -> DiffEntry {
    let old = before.unwrap_or("");
    let new = after.unwrap_or("");
    let text_diff = TextDiff::from_lines(old, new);

    let (mut added, mut removed) = (0, 0);
    for change in text_diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Insert => added += 1,
            ChangeTag::Delete => removed += 1,
            ChangeTag::Equal => {}
        }
    }

    let relative = path.trim_start_matches('/');
    let unified = text_diff
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{relative}"), &format!("b/{relative}"))
        .to_string();

    DiffEntry {
        tool: tool.to_string(),
        path: path.to_string(),
        unified,
        added,
        removed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_path_like_arguments() {
        let args = r#"{"path": "/a/b.rs", "content": "x", "destination": "/c/d.rs"}"#;
        assert_eq!(extract_paths(args), vec!["/a/b.rs", "/c/d.rs"]);
        assert!(extract_paths("not json").is_empty());
        assert!(extract_paths(r#"{"query": "hi"}"#).is_empty());
    }

    #[test]
    fn diff_counts_and_headers() {
        let entry = compute("fs__edit", "src/x.rs", Some("a\nb\nc\n"), Some("a\nB\nc\nd\n"));
        assert_eq!(entry.added, 2);
        assert_eq!(entry.removed, 1);
        assert!(entry.unified.contains("a/src/x.rs"));
        assert!(entry.unified.contains("+B"));
        assert!(entry.unified.contains("-b"));
    }

    #[test]
    fn file_creation_is_all_additions() {
        let entry = compute("fs__write", "new.txt", None, Some("one\ntwo\n"));
        assert_eq!(entry.added, 2);
        assert_eq!(entry.removed, 0);
    }
}
