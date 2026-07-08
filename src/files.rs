//! Built-in file tools: read, write, edit, list, glob, and grep, rooted at
//! the working directory. A stage or agent opts in with `files = true`;
//! the write tools are only exposed in `read_write` mode.
//!
//! Compared to routing file access through an MCP filesystem server, these
//! run in-process (no Node dependency), give approvals and diff capture an
//! exact path to work with, and `edit_file`'s exact-string replacement is
//! far more reliable for small local models than whole-file rewrites.
//!
//! All failures are returned as `ERROR: …` strings so the model can react
//! without killing the stage.

use std::path::{Component, Path, PathBuf};

use serde_json::{Value, json};

use crate::provider::ToolFunction;

/// Directory names never descended into by `glob` and `grep`.
const IGNORED_DIRS: &[&str] = &[".git", "node_modules", "target"];
/// Caps that keep one tool call from producing megabytes of output.
const MAX_GLOB_RESULTS: usize = 500;
const MAX_GREP_MATCHES: usize = 200;
const MAX_GREP_FILE_BYTES: u64 = 4_000_000;
const MAX_WALK_ENTRIES: usize = 50_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileOp {
    Read,
    Write,
    Edit,
    List,
    Glob,
    Grep,
}

impl FileOp {
    pub fn tool_name(self) -> &'static str {
        match self {
            FileOp::Read => "read_file",
            FileOp::Write => "write_file",
            FileOp::Edit => "edit_file",
            FileOp::List => "list_dir",
            FileOp::Glob => "glob",
            FileOp::Grep => "grep",
        }
    }
}

/// The file tools a context exposes: read-only ones always, write ones
/// only when `read_write` is true. The bool is the read-only flag.
pub fn definitions(read_write: bool) -> Vec<(ToolFunction, FileOp, bool)> {
    let path_property = |description: &str| {
        json!({ "type": "string", "description": description })
    };
    let mut tools = vec![
        (
            ToolFunction {
                name: FileOp::Read.tool_name().to_string(),
                description: "Read a text file. Returns the full content, or a window \
                    of it when `offset`/`limit` are given."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": path_property("File path, relative to the working directory"),
                        "offset": { "type": "integer", "description": "1-based line to start from" },
                        "limit": { "type": "integer", "description": "Maximum lines to return" }
                    },
                    "required": ["path"]
                }),
            },
            FileOp::Read,
            true,
        ),
        (
            ToolFunction {
                name: FileOp::List.tool_name().to_string(),
                description: "List a directory: entries sorted name-ascending, \
                    directories marked with a trailing `/`."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": path_property("Directory path (default: the working directory)")
                    }
                }),
            },
            FileOp::List,
            true,
        ),
        (
            ToolFunction {
                name: FileOp::Glob.tool_name().to_string(),
                description: format!(
                    "Find files whose working-directory-relative path matches a `*` \
                     wildcard pattern (`*` also crosses `/`, so `*.rs` finds nested \
                     files). Skips {}.",
                    IGNORED_DIRS.join(", ")
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Pattern like src/*.rs or *config*" }
                    },
                    "required": ["pattern"]
                }),
            },
            FileOp::Glob,
            true,
        ),
        (
            ToolFunction {
                name: FileOp::Grep.tool_name().to_string(),
                description: format!(
                    "Search file contents with a regular expression (prefix `(?i)` for \
                     case-insensitive). Returns `path:line: text` matches. Skips \
                     binary files and {}.",
                    IGNORED_DIRS.join(", ")
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Regular expression to search for" },
                        "path": path_property("File or directory to search (default: the working directory)"),
                        "glob": { "type": "string", "description": "Only search files whose relative path matches this `*` pattern" }
                    },
                    "required": ["pattern"]
                }),
            },
            FileOp::Grep,
            true,
        ),
    ];
    if read_write {
        tools.push((
            ToolFunction {
                name: FileOp::Write.tool_name().to_string(),
                description: "Write a file, creating it (and parent directories) or \
                    replacing its content entirely. For small changes to an existing \
                    file prefer edit_file."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": path_property("File path, relative to the working directory"),
                        "content": { "type": "string", "description": "The complete new file content" }
                    },
                    "required": ["path", "content"]
                }),
            },
            FileOp::Write,
            false,
        ));
        tools.push((
            ToolFunction {
                name: FileOp::Edit.tool_name().to_string(),
                description: "Replace an exact string in a file. `old_string` must \
                    match exactly once (include surrounding lines to disambiguate), \
                    or pass replace_all to change every occurrence."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": path_property("File path, relative to the working directory"),
                        "old_string": { "type": "string", "description": "Exact text to replace" },
                        "new_string": { "type": "string", "description": "Replacement text" },
                        "replace_all": { "type": "boolean", "description": "Replace every occurrence (default false)" }
                    },
                    "required": ["path", "old_string", "new_string"]
                }),
            },
            FileOp::Edit,
            false,
        ));
    }
    tools
}

/// Execute a file tool call. Every failure is an `ERROR:` string.
pub fn dispatch(op: FileOp, arguments: &Value) -> String {
    let cwd = match std::env::current_dir() {
        Ok(cwd) => cwd,
        Err(e) => return format!("ERROR: cannot determine working directory: {e}"),
    };
    let str_arg = |key: &str| arguments.get(key).and_then(Value::as_str);
    match op {
        FileOp::Read => {
            let Some(path) = str_arg("path") else { return missing("path") };
            let offset = arguments.get("offset").and_then(Value::as_u64);
            let limit = arguments.get("limit").and_then(Value::as_u64);
            read_file(&cwd, path, offset, limit)
        }
        FileOp::Write => {
            let Some(path) = str_arg("path") else { return missing("path") };
            let Some(content) = str_arg("content") else { return missing("content") };
            write_file(&cwd, path, content)
        }
        FileOp::Edit => {
            let Some(path) = str_arg("path") else { return missing("path") };
            let Some(old) = str_arg("old_string") else { return missing("old_string") };
            let Some(new) = str_arg("new_string") else { return missing("new_string") };
            let all = arguments.get("replace_all").and_then(Value::as_bool).unwrap_or(false);
            edit_file(&cwd, path, old, new, all)
        }
        FileOp::List => list_dir(&cwd, str_arg("path").unwrap_or("")),
        FileOp::Glob => {
            let Some(pattern) = str_arg("pattern") else { return missing("pattern") };
            glob(&cwd, pattern)
        }
        FileOp::Grep => {
            let Some(pattern) = str_arg("pattern") else { return missing("pattern") };
            grep(&cwd, pattern, str_arg("path").unwrap_or(""), str_arg("glob"))
        }
    }
}

fn missing(key: &str) -> String {
    format!("ERROR: missing required string argument `{key}`")
}

/// Resolve a path against the working directory, rejecting anything that
/// escapes it. `..` and `.` are normalized lexically so not-yet-existing
/// targets (writes) can still be checked.
fn resolve(cwd: &Path, path: &str) -> Result<PathBuf, String> {
    if path.trim().is_empty() {
        return Err("ERROR: `path` must not be empty".to_string());
    }
    let joined = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        cwd.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in joined.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(format!("ERROR: `{path}` escapes the working directory"));
                }
            }
            other => normalized.push(other),
        }
    }
    if !normalized.starts_with(cwd) {
        return Err(format!(
            "ERROR: `{path}` is outside the working directory {}",
            cwd.display()
        ));
    }
    Ok(normalized)
}

fn read_file(cwd: &Path, path: &str, offset: Option<u64>, limit: Option<u64>) -> String {
    let resolved = match resolve(cwd, path) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let content = match std::fs::read_to_string(&resolved) {
        Ok(content) => content,
        Err(e) => return format!("ERROR: cannot read `{path}`: {e}"),
    };
    let (Some(offset), limit) = (offset.or(limit.map(|_| 1)), limit) else {
        return content;
    };
    let total = content.lines().count();
    let start = (offset.max(1) - 1) as usize;
    let take = limit.map(|l| l as usize).unwrap_or(usize::MAX);
    let window: Vec<&str> = content.lines().skip(start).take(take).collect();
    format!(
        "[lines {}-{} of {total}]\n{}",
        start + 1,
        start + window.len(),
        window.join("\n")
    )
}

fn write_file(cwd: &Path, path: &str, content: &str) -> String {
    let resolved = match resolve(cwd, path) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let existed = resolved.exists();
    if let Some(parent) = resolved.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        return format!("ERROR: cannot create parent directory for `{path}`: {e}");
    }
    match std::fs::write(&resolved, content) {
        Ok(()) => format!(
            "{} `{path}` ({} bytes)",
            if existed { "overwrote" } else { "created" },
            content.len()
        ),
        Err(e) => format!("ERROR: cannot write `{path}`: {e}"),
    }
}

fn edit_file(cwd: &Path, path: &str, old: &str, new: &str, replace_all: bool) -> String {
    let resolved = match resolve(cwd, path) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if old.is_empty() {
        return "ERROR: `old_string` must not be empty".to_string();
    }
    if old == new {
        return "ERROR: `old_string` and `new_string` are identical".to_string();
    }
    let content = match std::fs::read_to_string(&resolved) {
        Ok(content) => content,
        Err(e) => return format!("ERROR: cannot read `{path}`: {e}"),
    };
    let occurrences = content.matches(old).count();
    if occurrences == 0 {
        return format!(
            "ERROR: `old_string` was not found in `{path}` — re-read the file and \
             match the current text exactly"
        );
    }
    if occurrences > 1 && !replace_all {
        return format!(
            "ERROR: `old_string` occurs {occurrences} times in `{path}` — include more \
             surrounding context to pin down one occurrence, or pass replace_all"
        );
    }
    let updated = if replace_all {
        content.replace(old, new)
    } else {
        content.replacen(old, new, 1)
    };
    match std::fs::write(&resolved, updated) {
        Ok(()) => format!(
            "edited `{path}` ({} replacement{})",
            occurrences,
            if occurrences == 1 { "" } else { "s" }
        ),
        Err(e) => format!("ERROR: cannot write `{path}`: {e}"),
    }
}

fn list_dir(cwd: &Path, path: &str) -> String {
    let target = if path.trim().is_empty() { "." } else { path };
    let resolved = match resolve(cwd, target) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let entries = match std::fs::read_dir(&resolved) {
        Ok(entries) => entries,
        Err(e) => return format!("ERROR: cannot list `{target}`: {e}"),
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|entry| {
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                name.push('/');
            }
            name
        })
        .collect();
    if names.is_empty() {
        return format!("(`{target}` is empty)");
    }
    names.sort();
    names.join("\n")
}

/// Walk the tree under `root`, calling `visit` with each file's
/// cwd-relative path. Skips [`IGNORED_DIRS`] and hidden entries, and stops
/// after [`MAX_WALK_ENTRIES`] (returning whether it was cut short).
fn walk_files(cwd: &Path, root: &Path, visit: &mut dyn FnMut(&str, &Path)) -> bool {
    let mut pending = vec![root.to_path_buf()];
    let mut seen = 0usize;
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            seen += 1;
            if seen > MAX_WALK_ENTRIES {
                return true;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                if !name.starts_with('.') && !IGNORED_DIRS.contains(&name.as_str()) {
                    pending.push(path);
                }
            } else if !name.starts_with('.')
                && let Ok(relative) = path.strip_prefix(cwd)
            {
                visit(&relative.to_string_lossy(), &path);
            }
        }
    }
    false
}

fn glob(cwd: &Path, pattern: &str) -> String {
    if pattern.trim().is_empty() {
        return "ERROR: `pattern` must not be empty".to_string();
    }
    let mut matches: Vec<String> = Vec::new();
    let truncated_walk = walk_files(cwd, cwd, &mut |relative, _| {
        if matches.len() < MAX_GLOB_RESULTS && crate::tools::wildcard_match(pattern, relative) {
            matches.push(relative.to_string());
        }
    });
    if matches.is_empty() {
        return format!("no files match `{pattern}`");
    }
    matches.sort();
    let mut out = matches.join("\n");
    if matches.len() >= MAX_GLOB_RESULTS || truncated_walk {
        out.push_str("\n… [result truncated — use a narrower pattern]");
    }
    out
}

fn grep(cwd: &Path, pattern: &str, path: &str, filter: Option<&str>) -> String {
    let regex = match regex::Regex::new(pattern) {
        Ok(regex) => regex,
        Err(e) => return format!("ERROR: invalid regular expression: {e}"),
    };
    let target = if path.trim().is_empty() { "." } else { path };
    let root = match resolve(cwd, target) {
        Ok(p) => p,
        Err(e) => return e,
    };

    let mut lines: Vec<String> = Vec::new();
    let mut files_matched = 0usize;
    let mut search = |relative: &str, path: &Path| {
        if lines.len() >= MAX_GREP_MATCHES {
            return;
        }
        if let Some(filter) = filter
            && !crate::tools::wildcard_match(filter, relative)
        {
            return;
        }
        if std::fs::metadata(path).is_ok_and(|m| m.len() > MAX_GREP_FILE_BYTES) {
            return;
        }
        let Ok(bytes) = std::fs::read(path) else { return };
        if bytes[..bytes.len().min(8192)].contains(&0) {
            return; // binary
        }
        let content = String::from_utf8_lossy(&bytes);
        let mut matched = false;
        for (index, line) in content.lines().enumerate() {
            if lines.len() >= MAX_GREP_MATCHES {
                break;
            }
            if regex.is_match(line) {
                matched = true;
                let text: String = line.trim_end().chars().take(300).collect();
                lines.push(format!("{relative}:{}: {text}", index + 1));
            }
        }
        if matched {
            files_matched += 1;
        }
    };

    if root.is_file() {
        let relative =
            root.strip_prefix(cwd).unwrap_or(&root).to_string_lossy().into_owned();
        search(&relative, &root);
    } else {
        walk_files(cwd, &root, &mut search);
    }

    if lines.is_empty() {
        return format!("no matches for `{pattern}`");
    }
    let mut out = lines.join("\n");
    if lines.len() >= MAX_GREP_MATCHES {
        out.push_str("\n… [matches truncated — narrow the pattern or path]");
    } else {
        out.push_str(&format!("\n({} match(es) in {files_matched} file(s))", lines.len()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("soa-files-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {\n    println!(\"hi\");\n}\n")
            .unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn add() {}\npub fn add2() {}\n").unwrap();
        std::fs::write(root.join(".git/config"), "secret").unwrap();
        std::fs::write(root.join("README.md"), "# readme\n").unwrap();
        root
    }

    #[test]
    fn resolve_rejects_escapes() {
        let cwd = project("resolve");
        assert!(resolve(&cwd, "src/main.rs").is_ok());
        assert!(resolve(&cwd, "new/dir/file.txt").is_ok());
        assert!(resolve(&cwd, "src/../README.md").is_ok());
        assert!(resolve(&cwd, "../outside.txt").unwrap_err().contains("outside"));
        assert!(resolve(&cwd, "/etc/passwd").unwrap_err().contains("outside"));
        assert!(resolve(&cwd, "src/../../up.txt").unwrap_err().contains("outside"));
        let inside_absolute = cwd.join("README.md");
        assert!(resolve(&cwd, &inside_absolute.to_string_lossy()).is_ok());
        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[test]
    fn read_write_edit_roundtrip() {
        let cwd = project("rwe");
        assert_eq!(
            read_file(&cwd, "src/main.rs", None, None),
            "fn main() {\n    println!(\"hi\");\n}\n"
        );
        assert_eq!(
            read_file(&cwd, "src/main.rs", Some(2), Some(1)),
            "[lines 2-2 of 3]\n    println!(\"hi\");"
        );
        assert!(read_file(&cwd, "missing.rs", None, None).starts_with("ERROR"));

        // Write creates parents; edit requires a unique match.
        assert_eq!(write_file(&cwd, "deep/new.txt", "abc"), "created `deep/new.txt` (3 bytes)");
        assert_eq!(write_file(&cwd, "deep/new.txt", "abcd"), "overwrote `deep/new.txt` (4 bytes)");
        assert_eq!(
            edit_file(&cwd, "src/main.rs", "\"hi\"", "\"hello\"", false),
            "edited `src/main.rs` (1 replacement)"
        );
        assert!(read_file(&cwd, "src/main.rs", None, None).contains("hello"));
        assert!(edit_file(&cwd, "src/main.rs", "nope", "x", false).contains("not found"));
        let ambiguous = edit_file(&cwd, "src/lib.rs", "pub fn add", "fn add", false);
        assert!(ambiguous.contains("occurs 2 times"), "{ambiguous}");
        assert_eq!(
            edit_file(&cwd, "src/lib.rs", "pub fn add", "fn add", true),
            "edited `src/lib.rs` (2 replacements)"
        );
        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[test]
    fn list_glob_grep() {
        let cwd = project("lgg");
        let listing = list_dir(&cwd, "");
        assert!(listing.contains("src/"), "{listing}");
        assert!(listing.contains("README.md"));

        // `*` crosses `/`; ignored dirs are skipped.
        assert_eq!(glob(&cwd, "*.rs"), "src/lib.rs\nsrc/main.rs");
        assert_eq!(glob(&cwd, "*config*"), "no files match `*config*`");

        let hits = grep(&cwd, r"println!", "", None);
        assert!(hits.starts_with("src/main.rs:2:"), "{hits}");
        assert!(hits.contains("1 match(es) in 1 file(s)"), "{hits}");
        // Case-insensitive via inline flag; glob filter narrows files.
        assert!(grep(&cwd, "(?i)READ", "", Some("*.md")).contains("README.md:1"));
        assert!(grep(&cwd, "println", "", Some("*.md")).starts_with("no matches"));
        assert!(grep(&cwd, "[invalid", "", None).starts_with("ERROR: invalid regular"));
        let _ = std::fs::remove_dir_all(&cwd);
    }
}
