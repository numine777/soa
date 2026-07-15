//! Capture file changes made by write tools during a conversation.
//!
//! Tool calls are opaque, so this works heuristically: before dispatching a
//! mutation-classified MCP tool we snapshot every file named by a path-like
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

/// A file's recorded pre-change state, kept so the change can be undone.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Snapshot {
    /// No restore data: an entry from before restore support existed, or a
    /// file that was unreadable (binary/too large) when captured.
    #[default]
    Unavailable,
    /// The file did not exist.
    Absent,
    /// The file's previous content.
    Content(String),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiffEntry {
    /// Advertised name of the tool that made the change.
    pub tool: String,
    pub path: String,
    /// Unified diff with `a/<path>` / `b/<path>` headers.
    pub unified: String,
    pub added: usize,
    pub removed: usize,
    /// Pre-change state, for restore. Defaults to `Unavailable` on entries
    /// saved before this field existed.
    #[serde(default)]
    pub before: Snapshot,
}

impl DiffEntry {
    pub fn title(&self) -> String {
        format!("{} (+{} −{})", self.path, self.added, self.removed)
    }

    pub fn restorable(&self) -> bool {
        self.before != Snapshot::Unavailable
    }
}

/// Put the file back into the entry's recorded pre-change state. Returns
/// the reverse [`DiffEntry`] (tool `rewind`) so the restore shows up in
/// the diff viewer and can itself be undone — or `Ok(None)` when the file
/// already matches the recorded state.
pub fn restore(entry: &DiffEntry) -> Result<Option<DiffEntry>, String> {
    let target = match &entry.before {
        Snapshot::Unavailable => {
            return Err(format!(
                "`{}` has no restore data (recorded before restore support, or unreadable)",
                entry.path
            ));
        }
        Snapshot::Absent => None,
        Snapshot::Content(text) => Some(text.as_str()),
    };

    let path = std::path::Path::new(&entry.path);
    let exists = path.exists();
    let current = read_text(&entry.path);
    let already_matches = match target {
        Some(text) => current.as_deref() == Some(text),
        None => !exists,
    };
    if already_matches {
        return Ok(None);
    }

    let reverse_before = match (&current, exists) {
        (Some(text), _) => Snapshot::Content(text.clone()),
        (None, false) => Snapshot::Absent,
        // Exists but unreadable: the restore proceeds but can't be undone.
        (None, true) => Snapshot::Unavailable,
    };
    match target {
        Some(text) => {
            if let Some(parent) = path.parent()
                && let Err(e) = std::fs::create_dir_all(parent)
            {
                return Err(format!("cannot create parent directory for `{}`: {e}", entry.path));
            }
            std::fs::write(path, text)
                .map_err(|e| format!("cannot restore `{}`: {e}", entry.path))?;
        }
        None => {
            std::fs::remove_file(path)
                .map_err(|e| format!("cannot remove `{}`: {e}", entry.path))?;
        }
    }

    let mut reverse = compute("rewind", &entry.path, current.as_deref(), target);
    reverse.before = reverse_before;
    Ok(Some(reverse))
}

/// The set of entries to restore to undo everything from `from` onward:
/// for each path first touched at or after `diffs[from]`, the earliest
/// restorable entry — whose `before` is that file's state at that moment.
pub fn earliest_restorable_since(diffs: &[DiffEntry], from: usize) -> Vec<DiffEntry> {
    let mut targets: Vec<DiffEntry> = Vec::new();
    for entry in diffs.iter().skip(from) {
        if entry.restorable() && !targets.iter().any(|t| t.path == entry.path) {
            targets.push(entry.clone());
        }
    }
    targets
}

/// Pull candidate file paths out of a tool call's arguments.
pub fn extract_paths(arguments: &Value) -> Vec<String> {
    let Value::Object(args) = arguments else {
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

/// Snapshot the named files' current state. A file that exists but can't
/// be read as text (binary, oversized) is `Unavailable`, distinct from
/// `Absent` — restoring must never delete a file that merely resisted
/// snapshotting.
pub fn snapshot(paths: &[String]) -> Vec<(String, Snapshot)> {
    paths
        .iter()
        .map(|p| {
            let state = match read_text(p) {
                Some(text) => Snapshot::Content(text),
                None if std::path::Path::new(p).exists() => Snapshot::Unavailable,
                None => Snapshot::Absent,
            };
            (p.clone(), state)
        })
        .collect()
}

/// Re-read snapshotted files and produce a diff entry per changed file.
pub fn collect_changes(tool: &str, snapshots: Vec<(String, Snapshot)>) -> Vec<DiffEntry> {
    snapshots
        .into_iter()
        .filter_map(|(path, before)| {
            let after = read_text(&path);
            let before_text = match &before {
                Snapshot::Content(text) => Some(text.as_str()),
                _ => None,
            };
            let changed = match (&before, &after) {
                (Snapshot::Content(text), Some(now)) => text != now,
                (Snapshot::Content(_), None) => true, // deleted (or became unreadable)
                (_, Some(_)) => true,                 // created or became readable
                (_, None) => false,                   // unknown on both sides
            };
            if !changed {
                return None;
            }
            let mut entry = compute(tool, &path, before_text, after.as_deref());
            entry.before = before; // keep Unavailable rather than compute's Absent
            Some(entry)
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
        before: before.map_or(Snapshot::Absent, |text| Snapshot::Content(text.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_path_like_arguments() {
        let args = r#"{"path": "/a/b.rs", "content": "x", "destination": "/c/d.rs"}"#;
        let args: Value = serde_json::from_str(args).unwrap();
        assert_eq!(extract_paths(&args), vec!["/a/b.rs", "/c/d.rs"]);
        assert!(extract_paths(&Value::String("not json".into())).is_empty());
        assert!(extract_paths(&serde_json::json!({"query": "hi"})).is_empty());
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
        assert_eq!(entry.before, Snapshot::Absent);
    }

    #[test]
    fn earliest_restorable_since_dedups_by_path() {
        let entry = |path: &str, before: Snapshot| DiffEntry {
            tool: "t".into(),
            path: path.into(),
            unified: String::new(),
            added: 0,
            removed: 0,
            before,
        };
        let diffs = vec![
            entry("a", Snapshot::Content("a0".into())),
            entry("b", Snapshot::Unavailable),
            entry("a", Snapshot::Content("a1".into())),
            entry("b", Snapshot::Content("b1".into())),
        ];
        // From the start: a's first entry wins; b's Unavailable entry is
        // skipped in favor of its later restorable one.
        let all = earliest_restorable_since(&diffs, 0);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].before, Snapshot::Content("a0".into()));
        assert_eq!(all[1].before, Snapshot::Content("b1".into()));
        // From index 2: only the later entries are considered.
        let since = earliest_restorable_since(&diffs, 2);
        assert_eq!(since[0].before, Snapshot::Content("a1".into()));
        // Past the end: nothing.
        assert!(earliest_restorable_since(&diffs, 4).is_empty());
    }

    #[test]
    fn legacy_entries_deserialize_as_unrestorable() {
        let raw = r#"{"tool":"t","path":"p","unified":"","added":0,"removed":0}"#;
        let entry: DiffEntry = serde_json::from_str(raw).unwrap();
        assert_eq!(entry.before, Snapshot::Unavailable);
        assert!(!entry.restorable());
        assert!(restore(&entry).unwrap_err().contains("no restore data"));
    }

    #[test]
    fn restore_roundtrip() {
        let root = std::env::temp_dir().join(format!("soa-diff-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let file = root.join("a.txt").to_string_lossy().into_owned();
        let created = root.join("new.txt").to_string_lossy().into_owned();

        // Simulate a turn: snapshot, mutate an existing file, create one.
        std::fs::write(&file, "original\n").unwrap();
        let snapshots = snapshot(&[file.clone(), created.clone()]);
        std::fs::write(&file, "modified\n").unwrap();
        std::fs::write(&created, "brand new\n").unwrap();
        let entries = collect_changes("edit_file", snapshots);
        assert_eq!(entries.len(), 2);

        // Restoring puts both files back and yields reverse entries.
        let reverse_edit = restore(&entries[0]).unwrap().unwrap();
        let reverse_create = restore(&entries[1]).unwrap().unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "original\n");
        assert!(!std::path::Path::new(&created).exists());
        assert_eq!(reverse_edit.tool, "rewind");
        assert_eq!(reverse_edit.before, Snapshot::Content("modified\n".to_string()));

        // Restoring again is a no-op; restoring the reverse re-applies.
        assert!(restore(&entries[0]).unwrap().is_none());
        restore(&reverse_edit).unwrap().unwrap();
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "modified\n");
        restore(&reverse_create).unwrap().unwrap();
        assert_eq!(std::fs::read_to_string(&created).unwrap(), "brand new\n");

        let _ = std::fs::remove_dir_all(&root);
    }
}
