//! Small atomic persistence primitives shared by run and session state.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

/// Replace a file atomically by writing its complete new contents beside it
/// and renaming only after the write succeeds. Readers see either the old or
/// the new document, never a partially truncated JSON file.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("cannot create {}", parent.display()))?;
    let file_name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?
        .to_string_lossy();
    let temporary = parent.join(format!(".{file_name}.tmp"));
    std::fs::write(&temporary, bytes)
        .with_context(|| format!("cannot write {}", temporary.display()))?;
    std::fs::rename(&temporary, path).with_context(|| format!("cannot replace {}", path.display()))
}

/// Append one compact JSON value with a newline using one buffered write.
pub fn append_json_line<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("cannot open {}", path.display()))?;
    file.write_all(&line)
        .with_context(|| format!("cannot append {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_replace_and_json_lines() {
        let dir = std::env::temp_dir().join(format!("soa-persistence-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let document = dir.join("state.json");
        atomic_write(&document, b"old").unwrap();
        atomic_write(&document, b"new").unwrap();
        assert_eq!(std::fs::read(&document).unwrap(), b"new");
        assert!(!dir.join(".state.json.tmp").exists());

        let log = dir.join("events.jsonl");
        append_json_line(&log, &serde_json::json!({"n": 1})).unwrap();
        append_json_line(&log, &serde_json::json!({"n": 2})).unwrap();
        assert_eq!(
            std::fs::read_to_string(log).unwrap(),
            "{\"n\":1}\n{\"n\":2}\n"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
