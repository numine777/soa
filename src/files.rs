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

use std::collections::VecDeque;
use std::path::{Component, Path, PathBuf};

use serde_json::{Value, json};

use crate::model::ToolDefinition;

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
    /// `read_file` with `LINE:HASH|` anchor prefixes, exposed instead of
    /// plain `Read` when the edit tools are available.
    ReadAnchored,
    Write,
    Edit,
    /// Anchor-addressed line edits (see [`edit_lines`]).
    EditLines,
    List,
    Glob,
    Grep,
}

impl FileOp {
    pub fn tool_name(self) -> &'static str {
        match self {
            FileOp::Read | FileOp::ReadAnchored => "read_file",
            FileOp::Write => "write_file",
            FileOp::Edit => "edit_file",
            FileOp::EditLines => "edit_lines",
            FileOp::List => "list_dir",
            FileOp::Glob => "glob",
            FileOp::Grep => "grep",
        }
    }
}

/// The file tools a context exposes: read-only ones always, write ones
/// only when `read_write` is true. The bool is the read-only flag.
pub fn definitions(read_write: bool) -> Vec<(ToolDefinition, FileOp, bool)> {
    let path_property = |description: &str| {
        json!({ "type": "string", "description": description })
    };
    let (read_op, read_description) = if read_write {
        (
            FileOp::ReadAnchored,
            "Read a text file. Each line is prefixed with an anchor `LINE:HASH|` for \
             edit_lines — the prefix is NOT part of the file content. Returns the full \
             content, or a window of it when `offset`/`limit` are given.",
        )
    } else {
        (
            FileOp::Read,
            "Read a text file. Returns the full content, or a window of it when \
             `offset`/`limit` are given.",
        )
    };
    let mut tools = vec![
        (
            ToolDefinition {
                name: read_op.tool_name().to_string(),
                description: read_description.to_string(),
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
            read_op,
            true,
        ),
        (
            ToolDefinition {
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
            ToolDefinition {
                name: FileOp::Glob.tool_name().to_string(),
                description: format!(
                    "Find files by glob pattern: `*` and `?` match within a path \
                     segment, `**` spans directories, and a pattern without `/` \
                     matches file names at any depth (`*.rs` finds nested files). \
                     Dotfiles are included; {} are skipped.",
                    IGNORED_DIRS.join(", ")
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Pattern like src/**/*.rs, *.toml, or ?ain.rs" }
                    },
                    "required": ["pattern"]
                }),
            },
            FileOp::Glob,
            true,
        ),
        (
            ToolDefinition {
                name: FileOp::Grep.tool_name().to_string(),
                description: format!(
                    "Search file contents with a regular expression. Returns \
                     `path:line: text` matches (context lines as `path-line- text`). \
                     Dotfiles are included; binary files and {} are skipped.",
                    IGNORED_DIRS.join(", ")
                ),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Regular expression to search for (literal string with fixed=true)" },
                        "path": path_property("File or directory to search (default: the working directory)"),
                        "glob": { "type": "string", "description": "Only search files whose relative path matches this glob pattern" },
                        "fixed": { "type": "boolean", "description": "Match the pattern as a literal string, not a regex (default false)" },
                        "ignore_case": { "type": "boolean", "description": "Case-insensitive matching (default false)" },
                        "context": { "type": "integer", "description": "Lines of context around each match, up to 10 (default 0)" },
                        "output": { "type": "string", "enum": ["content", "files", "count"], "description": "content = matching lines (default), files = only file paths, count = per-file match counts" }
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
            ToolDefinition {
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
            ToolDefinition {
                name: FileOp::EditLines.tool_name().to_string(),
                description: "Replace a range of lines using anchors from read_file. \
                    `first` and `last` are anchors like `42:9f3a` copied verbatim from \
                    a read_file listing (`last` defaults to `first`). The hash must \
                    match the file's current content — if the file changed since you \
                    read it, the edit is rejected and you must re-read. `new_text` is \
                    raw text WITHOUT anchor prefixes; empty `new_text` deletes the \
                    range; `insert_after` inserts without removing lines. The reply \
                    shows fresh anchors around the change."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": path_property("File path, relative to the working directory"),
                        "first": { "type": "string", "description": "Anchor of the first line, e.g. 42:9f3a" },
                        "last": { "type": "string", "description": "Anchor of the last line (default: first)" },
                        "new_text": { "type": "string", "description": "Replacement text, without anchor prefixes" },
                        "insert_after": { "type": "boolean", "description": "Insert new_text after `first` instead of replacing (default false)" }
                    },
                    "required": ["path", "first", "new_text"]
                }),
            },
            FileOp::EditLines,
            false,
        ));
        tools.push((
            ToolDefinition {
                name: FileOp::Edit.tool_name().to_string(),
                description: "Replace an exact string in a file. `old_string` must \
                    match exactly once (include surrounding lines to disambiguate), \
                    or pass replace_all to change every occurrence. Prefer edit_lines \
                    when you have anchors from read_file."
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
        FileOp::Read | FileOp::ReadAnchored => {
            let Some(path) = str_arg("path") else { return missing("path") };
            let offset = arguments.get("offset").and_then(Value::as_u64);
            let limit = arguments.get("limit").and_then(Value::as_u64);
            read_file(&cwd, path, offset, limit, op == FileOp::ReadAnchored)
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
        FileOp::EditLines => {
            let Some(path) = str_arg("path") else { return missing("path") };
            let Some(first) = str_arg("first") else { return missing("first") };
            let Some(new_text) = str_arg("new_text") else { return missing("new_text") };
            let last = str_arg("last");
            let insert_after =
                arguments.get("insert_after").and_then(Value::as_bool).unwrap_or(false);
            edit_lines(&cwd, path, first, last, new_text, insert_after)
        }
        FileOp::List => list_dir(&cwd, str_arg("path").unwrap_or("")),
        FileOp::Glob => {
            let Some(pattern) = str_arg("pattern") else { return missing("pattern") };
            glob(&cwd, pattern)
        }
        FileOp::Grep => {
            let Some(pattern) = str_arg("pattern") else { return missing("pattern") };
            let output = match str_arg("output").unwrap_or("content") {
                "content" => GrepOutput::Content,
                "files" => GrepOutput::Files,
                "count" => GrepOutput::Count,
                other => {
                    return format!(
                        "ERROR: unknown `output` mode `{other}` — use content, files, or count"
                    );
                }
            };
            let options = GrepOptions {
                filter: str_arg("glob"),
                fixed: arguments.get("fixed").and_then(Value::as_bool).unwrap_or(false),
                ignore_case: arguments
                    .get("ignore_case")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                context: arguments.get("context").and_then(Value::as_u64).unwrap_or(0) as usize,
                output,
            };
            grep(&cwd, pattern, str_arg("path").unwrap_or(""), &options)
        }
    }
}

fn missing(key: &str) -> String {
    format!("ERROR: missing required string argument `{key}`")
}

/// Resolve a path against the working directory, rejecting anything that
/// escapes it lexically or through a symlink. Existing targets are returned
/// canonicalized. For not-yet-existing write targets, the nearest existing
/// ancestor is canonicalized and checked before the lexical target is
/// returned.
fn resolve(cwd: &Path, path: &str) -> Result<PathBuf, String> {
    if path.trim().is_empty() {
        return Err("ERROR: `path` must not be empty".to_string());
    }
    let root = cwd.canonicalize().map_err(|e| {
        format!(
            "ERROR: cannot resolve working directory {}: {e}",
            cwd.display()
        )
    })?;
    let joined = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        root.join(path)
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

    // `exists()` follows symlinks and therefore misses dangling links. Use
    // symlink_metadata so a dangling final link is rejected rather than
    // treated as a safe, not-yet-created file.
    let mut ancestor = normalized.as_path();
    loop {
        match std::fs::symlink_metadata(ancestor) {
            Ok(_) => break,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                ancestor = ancestor.parent().ok_or_else(|| {
                    format!("ERROR: cannot resolve an existing ancestor of `{path}`")
                })?;
            }
            Err(e) => {
                return Err(format!(
                    "ERROR: cannot inspect `{}` while resolving `{path}`: {e}",
                    ancestor.display()
                ));
            }
        }
    }
    let canonical_ancestor = ancestor.canonicalize().map_err(|e| {
        format!(
            "ERROR: cannot safely resolve `{path}` through `{}`: {e}",
            ancestor.display()
        )
    })?;
    if !canonical_ancestor.starts_with(&root) {
        return Err(format!(
            "ERROR: `{path}` is outside the working directory {}",
            root.display()
        ));
    }

    // Canonicalizing an existing target closes over its final symlink too.
    // A new target cannot itself be a symlink, so the checked lexical path
    // is the path its writer needs in order to create it.
    if ancestor == normalized {
        Ok(canonical_ancestor)
    } else {
        Ok(normalized)
    }
}

/// A short content hash for line anchors: FNV-1a 32 folded to 16 bits,
/// rendered as 4 hex chars. It only guards staleness of a specific line
/// (the line number does the locating), so 16 bits is plenty.
fn line_hash(line: &str) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in line.bytes() {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(0x0100_0193);
    }
    format!("{:04x}", (hash ^ (hash >> 16)) & 0xffff)
}

/// Render lines with `LINE:HASH|` anchors, numbering from `start` (1-based).
fn anchored_lines(lines: &[&str], start: usize) -> String {
    lines
        .iter()
        .enumerate()
        .map(|(index, line)| format!("{}:{}|{line}", start + index, line_hash(line)))
        .collect::<Vec<_>>()
        .join("\n")
}

fn read_file(
    cwd: &Path,
    path: &str,
    offset: Option<u64>,
    limit: Option<u64>,
    anchored: bool,
) -> String {
    let resolved = match resolve(cwd, path) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let content = match std::fs::read_to_string(&resolved) {
        Ok(content) => content,
        Err(e) => return format!("ERROR: cannot read `{path}`: {e}"),
    };
    let (Some(offset), limit) = (offset.or(limit.map(|_| 1)), limit) else {
        if !anchored {
            return content;
        }
        let lines: Vec<&str> = content.lines().collect();
        return anchored_lines(&lines, 1);
    };
    let total = content.lines().count();
    let start = (offset.max(1) - 1) as usize;
    let take = limit.map(|l| l as usize).unwrap_or(usize::MAX);
    let window: Vec<&str> = content.lines().skip(start).take(take).collect();
    let body = if anchored { anchored_lines(&window, start + 1) } else { window.join("\n") };
    format!("[lines {}-{} of {total}]\n{body}", start + 1, start + window.len())
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

/// Parse an anchor like `42:9f3a` into its 1-based line number and hash.
fn parse_anchor(anchor: &str) -> Result<(usize, String), String> {
    let parsed = anchor
        .trim()
        .split_once(':')
        .and_then(|(number, hash)| Some((number.parse::<usize>().ok()?, hash.trim())))
        .filter(|(number, hash)| *number > 0 && !hash.is_empty());
    match parsed {
        Some((number, hash)) => Ok((number, hash.to_lowercase())),
        None => Err(format!(
            "ERROR: `{anchor}` is not a line anchor — copy one verbatim from a \
             read_file listing (they look like `42:9f3a`)"
        )),
    }
}

fn edit_lines(
    cwd: &Path,
    path: &str,
    first: &str,
    last: Option<&str>,
    new_text: &str,
    insert_after: bool,
) -> String {
    let resolved = match resolve(cwd, path) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let content = match std::fs::read_to_string(&resolved) {
        Ok(content) => content,
        Err(e) => return format!("ERROR: cannot read `{path}`: {e}"),
    };
    let had_trailing_newline = content.ends_with('\n');
    // `lines()` strips `\r`, so rejoining with `\n` would silently convert
    // a CRLF file to LF. Rejoin with the file's own line ending instead
    // (mixed-ending files are normalized to CRLF if any line uses it).
    let eol = if content.contains("\r\n") { "\r\n" } else { "\n" };
    let lines: Vec<&str> = content.lines().collect();

    // An anchor is only usable if the line it names still has the content
    // the model saw — the stale check that keeps edits from corrupting.
    let verify = |anchor: &str| -> Result<usize, String> {
        let (number, hash) = parse_anchor(anchor)?;
        let line = lines.get(number - 1).ok_or_else(|| {
            format!(
                "ERROR: anchor `{anchor}` is out of range — `{path}` has {} line(s)",
                lines.len()
            )
        })?;
        let current = line_hash(line);
        if current != hash {
            return Err(format!(
                "ERROR: stale anchor `{anchor}` — line {number} is now \
                 `{number}:{current}|{line}`. The file changed since you read it; \
                 re-read it and use fresh anchors."
            ));
        }
        Ok(number)
    };
    let first_line = match verify(first) {
        Ok(n) => n,
        Err(e) => return e,
    };
    let last_line = match last {
        Some(anchor) if !insert_after => match verify(anchor) {
            Ok(n) => n,
            Err(e) => return e,
        },
        _ => first_line,
    };
    if last_line < first_line {
        return format!("ERROR: `last` (line {last_line}) is before `first` (line {first_line})");
    }

    let replacement: Vec<&str> =
        if new_text.is_empty() { Vec::new() } else { new_text.lines().collect() };
    let mut updated: Vec<&str> = Vec::new();
    if insert_after {
        updated.extend(&lines[..first_line]);
        updated.extend(&replacement);
        updated.extend(&lines[first_line..]);
    } else {
        updated.extend(&lines[..first_line - 1]);
        updated.extend(&replacement);
        updated.extend(&lines[last_line..]);
    }
    let mut new_content = updated.join(eol);
    if had_trailing_newline && !new_content.is_empty() {
        new_content.push_str(eol);
    }
    if let Err(e) = std::fs::write(&resolved, new_content) {
        return format!("ERROR: cannot write `{path}`: {e}");
    }

    // Fresh anchors around the change, so nearby follow-up edits don't
    // need a re-read.
    let region_start = if insert_after { first_line } else { first_line - 1 };
    let window_start = region_start.saturating_sub(2);
    let window_end = (region_start + replacement.len() + 2).min(updated.len());
    let action = if insert_after {
        format!("inserted {} line(s) after line {first_line}", replacement.len())
    } else if replacement.is_empty() {
        format!("deleted lines {first_line}-{last_line}")
    } else {
        format!(
            "replaced lines {first_line}-{last_line} with {} line(s)",
            replacement.len()
        )
    };
    let mut response = format!("edited `{path}`: {action}");
    if window_start < window_end {
        response.push_str(&format!(
            "\n[lines {}-{} of {} — fresh anchors]\n{}",
            window_start + 1,
            window_end,
            updated.len(),
            anchored_lines(&updated[window_start..window_end], window_start + 1),
        ));
    }
    response
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

/// Dotfiles are searchable (`.github/`, `.env`, CI configs); only the
/// bookkeeping and dependency directories in `IGNORED_DIRS` are not.
fn searchable_path(relative: &Path) -> bool {
    relative.components().all(|component| match component {
        Component::Normal(name) => {
            let name = name.to_string_lossy();
            !IGNORED_DIRS.contains(&name.as_ref())
        }
        _ => true,
    })
}

/// Gitignore-style path matching for `glob` and grep's `glob` filter:
/// `*` and `?` match within one path segment, `**` spans segments, and a
/// pattern without `/` matches the file name at any depth (so `*.rs`
/// finds nested files, like a `.gitignore` line would).
fn glob_match(pattern: &str, relative: &str) -> bool {
    let pattern = pattern.trim_matches('/');
    if !pattern.contains('/') && pattern != "**" {
        let name = relative.rsplit('/').next().unwrap_or(relative);
        return segment_match(pattern, name);
    }
    let pattern_segments: Vec<&str> = pattern.split('/').collect();
    let path_segments: Vec<&str> = relative.split('/').collect();
    match_segments(&pattern_segments, &path_segments)
}

fn match_segments(pattern: &[&str], path: &[&str]) -> bool {
    match pattern.first() {
        None => path.is_empty(),
        // `**` covers zero or more whole segments.
        Some(&"**") => {
            (0..=path.len()).any(|skip| match_segments(&pattern[1..], &path[skip..]))
        }
        Some(segment) => {
            !path.is_empty()
                && segment_match(segment, path[0])
                && match_segments(&pattern[1..], &path[1..])
        }
    }
}

/// `*`/`?` wildcard match within one path segment (neither crosses `/`,
/// which segment splitting already guarantees here).
fn segment_match(pattern: &str, text: &str) -> bool {
    fn recurse(pattern: &[char], text: &[char]) -> bool {
        match pattern.first() {
            None => text.is_empty(),
            Some('*') => {
                recurse(&pattern[1..], text) || (!text.is_empty() && recurse(pattern, &text[1..]))
            }
            Some('?') => !text.is_empty() && recurse(&pattern[1..], &text[1..]),
            Some(ch) => text.first() == Some(ch) && recurse(&pattern[1..], &text[1..]),
        }
    }
    let pattern: Vec<char> = pattern.chars().collect();
    let text: Vec<char> = text.chars().collect();
    recurse(&pattern, &text)
}

/// Use Git's index and ignore engine when the workspace is a repository.
/// This avoids statting every ignored build artifact while still including
/// untracked, non-ignored files. Returns `None` outside a repository.
fn walk_git_files(
    cwd: &Path,
    root: &Path,
    visit: &mut dyn FnMut(&str, &Path) -> bool,
) -> Option<bool> {
    let pathspec = root
        .strip_prefix(cwd)
        .ok()
        .filter(|path| !path.as_os_str().is_empty());
    let mut command = std::process::Command::new("git");
    command
        .arg("-C")
        .arg(cwd)
        .args(["ls-files", "-co", "--exclude-standard", "-z", "--"])
        .arg(pathspec.unwrap_or_else(|| Path::new(".")));
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }

    let mut seen = 0usize;
    for raw in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|raw| !raw.is_empty())
    {
        seen += 1;
        if seen > MAX_WALK_ENTRIES {
            return Some(true);
        }
        let relative = PathBuf::from(String::from_utf8_lossy(raw).into_owned());
        if !searchable_path(&relative) {
            continue;
        }
        let path = cwd.join(&relative);
        let Ok(metadata) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            continue;
        }
        let relative_text = relative.to_string_lossy();
        if !visit(&relative_text, &path) {
            return Some(true);
        }
    }
    Some(false)
}

/// Walk the tree under `root`, calling `visit` with each file's
/// cwd-relative path. Git repositories use the index plus standard ignore
/// rules; other directories use a deterministic in-process walk. Returning
/// `false` from `visit` stops immediately. The return value says whether
/// traversal was cut short.
fn walk_files(cwd: &Path, root: &Path, visit: &mut dyn FnMut(&str, &Path) -> bool) -> bool {
    if let Some(truncated) = walk_git_files(cwd, root, visit) {
        return truncated;
    }

    let mut pending = VecDeque::from([root.to_path_buf()]);
    let mut seen = 0usize;
    while let Some(dir) = pending.pop_front() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            seen += 1;
            if seen > MAX_WALK_ENTRIES {
                return true;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            // Recursive traversal never follows symlinks. Direct file-tool
            // paths go through `resolve`, which can safely allow links whose
            // canonical target remains within the workspace.
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                if !IGNORED_DIRS.contains(&name.as_str()) {
                    pending.push_back(path);
                }
            } else if let Ok(relative) = path.strip_prefix(cwd)
                && !visit(&relative.to_string_lossy(), &path)
            {
                return true;
            }
        }
    }
    false
}

fn glob(cwd: &Path, pattern: &str) -> String {
    if pattern.trim().is_empty() {
        return "ERROR: `pattern` must not be empty".to_string();
    }
    let workspace = match cwd.canonicalize() {
        Ok(path) => path,
        Err(e) => return format!("ERROR: cannot resolve working directory: {e}"),
    };
    let mut matches: Vec<String> = Vec::new();
    let truncated_walk = walk_files(&workspace, &workspace, &mut |relative, _| {
        if glob_match(pattern, relative) {
            matches.push(relative.to_string());
        }
        matches.len() < MAX_GLOB_RESULTS
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

#[derive(Clone, Copy, PartialEq)]
enum GrepOutput {
    /// Matching lines (`path:line: text`, context lines as `path-line- text`).
    Content,
    /// Only the paths of files with at least one match.
    Files,
    /// Per-file match counts.
    Count,
}

/// Search options beyond the pattern and root path.
struct GrepOptions<'a> {
    /// Restrict the files searched (same semantics as the `glob` tool).
    filter: Option<&'a str>,
    /// Treat the pattern as a literal string, not a regex.
    fixed: bool,
    ignore_case: bool,
    /// Lines of context around each match (`Content` output only).
    context: usize,
    output: GrepOutput,
}

fn grep(cwd: &Path, pattern: &str, path: &str, options: &GrepOptions) -> String {
    let mut source = if options.fixed {
        regex::escape(pattern)
    } else {
        pattern.to_string()
    };
    if options.ignore_case {
        source = format!("(?i){source}");
    }
    let regex = match regex::Regex::new(&source) {
        Ok(regex) => regex,
        Err(e) => return format!("ERROR: invalid regular expression: {e}"),
    };
    let context = options.context.min(10);
    let target = if path.trim().is_empty() { "." } else { path };
    let root = match resolve(cwd, target) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let workspace = match cwd.canonicalize() {
        Ok(path) => path,
        Err(e) => return format!("ERROR: cannot resolve working directory: {e}"),
    };

    let mut lines: Vec<String> = Vec::new();
    let mut files_matched = 0usize;
    let mut total_matches = 0usize;
    let mut search = |relative: &str, path: &Path| -> bool {
        if total_matches >= MAX_GREP_MATCHES {
            return false;
        }
        if let Some(filter) = options.filter
            && !glob_match(filter, relative)
        {
            return true;
        }
        if std::fs::metadata(path).is_ok_and(|m| m.len() > MAX_GREP_FILE_BYTES) {
            return true;
        }
        let Ok(bytes) = std::fs::read(path) else {
            return true;
        };
        if bytes[..bytes.len().min(8192)].contains(&0) {
            return true; // binary
        }
        let content = String::from_utf8_lossy(&bytes);
        let file_lines: Vec<&str> = content.lines().collect();
        let matched: Vec<usize> = file_lines
            .iter()
            .enumerate()
            .filter(|(_, line)| regex.is_match(line))
            .map(|(index, _)| index)
            .take(MAX_GREP_MATCHES - total_matches)
            .collect();
        if matched.is_empty() {
            return true;
        }
        files_matched += 1;
        total_matches += matched.len();
        let excerpt = |index: usize| -> String {
            file_lines[index].trim_end().chars().take(300).collect()
        };
        match options.output {
            GrepOutput::Files => lines.push(relative.to_string()),
            GrepOutput::Count => lines.push(format!("{relative}: {}", matched.len())),
            GrepOutput::Content => {
                // Windows around nearby matches are merged; a gap between
                // windows in the same file shows as `--` (grep style).
                let is_match: std::collections::BTreeSet<usize> = matched.iter().copied().collect();
                let mut emitted_end = 0usize; // one past the last printed line
                for &index in &matched {
                    let start = index.saturating_sub(context).max(emitted_end);
                    let end = (index + context + 1).min(file_lines.len());
                    if start >= end {
                        continue; // already printed as an earlier match's context
                    }
                    if context > 0 && emitted_end > 0 && start > emitted_end {
                        lines.push("--".to_string());
                    }
                    for line_index in start..end {
                        let separator = if is_match.contains(&line_index) { ':' } else { '-' };
                        lines.push(format!(
                            "{relative}{separator}{}{separator} {}",
                            line_index + 1,
                            excerpt(line_index)
                        ));
                    }
                    emitted_end = end;
                }
            }
        }
        total_matches < MAX_GREP_MATCHES
    };

    let truncated_walk = if root.is_file() {
        let relative = root
            .strip_prefix(&workspace)
            .unwrap_or(&root)
            .to_string_lossy()
            .into_owned();
        search(&relative, &root);
        false
    } else {
        walk_files(&workspace, &root, &mut search)
    };

    if lines.is_empty() {
        return format!("no matches for `{pattern}`");
    }
    let mut out = lines.join("\n");
    if total_matches >= MAX_GREP_MATCHES || truncated_walk {
        out.push_str("\n… [matches truncated — narrow the pattern or path]");
    } else {
        out.push_str(&format!(
            "\n({total_matches} match(es) in {files_matched} file(s))"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_grep() -> GrepOptions<'static> {
        GrepOptions {
            filter: None,
            fixed: false,
            ignore_case: false,
            context: 0,
            output: GrepOutput::Content,
        }
    }

    #[test]
    fn glob_patterns_use_gitignore_semantics() {
        // Slash-free patterns match the file name at any depth.
        assert!(glob_match("*.rs", "src/deep/main.rs"));
        assert!(glob_match("?ain.rs", "src/main.rs"));
        assert!(glob_match("*config*", ".github/my-config.yml"));
        assert!(!glob_match("*.rs", "src/main.rs.bak"));

        // With a `/`, `*` and `?` stay within their segment…
        assert!(glob_match("src/*.rs", "src/main.rs"));
        assert!(!glob_match("src/*.rs", "src/deep/main.rs"));
        // …and `**` spans zero or more segments.
        assert!(glob_match("src/**/*.rs", "src/main.rs"));
        assert!(glob_match("src/**/*.rs", "src/a/b/main.rs"));
        assert!(glob_match("**/tests/*.rs", "crates/x/tests/it.rs"));
        assert!(!glob_match("src/**/*.rs", "other/main.rs"));
        assert!(glob_match("**", "any/path/at/all.txt"));
    }

    #[test]
    fn search_includes_dotfiles_but_not_ignored_dirs() {
        let cwd = project("dotfiles");
        std::fs::create_dir_all(cwd.join(".github/workflows")).unwrap();
        std::fs::write(cwd.join(".github/workflows/ci.yml"), "on: push\n").unwrap();
        std::fs::write(cwd.join(".env"), "SECRET=needle\n").unwrap();

        let found = glob(&cwd, "*.yml");
        assert_eq!(found, ".github/workflows/ci.yml");
        let hits = grep(&cwd, "needle", "", &plain_grep());
        assert!(hits.contains(".env:1"), "{hits}");
        // `.git` internals stay invisible even though dotfiles are searchable.
        let config_hits = grep(&cwd, "repositoryformatversion", "", &plain_grep());
        assert!(config_hits.starts_with("no matches"), "{config_hits}");
        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[test]
    fn grep_modes_fixed_case_context_and_output() {
        let cwd = project("grepmodes");
        std::fs::write(
            cwd.join("notes.txt"),
            "alpha\nfn call() beta\ngamma\ndelta\nepsilon\nzeta\neta\nfn call() again\n",
        )
        .unwrap();

        // `fixed` treats regex metacharacters literally.
        let fixed = GrepOptions { fixed: true, ..plain_grep() };
        let hits = grep(&cwd, "fn call()", "notes.txt", &fixed);
        assert!(hits.contains("notes.txt:2:"), "{hits}");
        assert!(hits.contains("notes.txt:8:"), "{hits}");
        // Unescaped, the same pattern is a regex where `()` matches empty.
        assert!(
            grep(&cwd, "fn call()", "notes.txt", &plain_grep()).contains("2 match(es)")
        );

        let ci = GrepOptions { ignore_case: true, ..plain_grep() };
        assert!(grep(&cwd, "ALPHA", "notes.txt", &ci).contains("notes.txt:1:"));

        // Context lines carry the `-` separator; nearby windows merge and
        // distant ones are divided by `--`.
        let ctx = GrepOptions { context: 1, ..plain_grep() };
        let hits = grep(&cwd, "gamma", "notes.txt", &ctx);
        assert!(hits.contains("notes.txt-2- fn call() beta"), "{hits}");
        assert!(hits.contains("notes.txt:3: gamma"), "{hits}");
        assert!(hits.contains("notes.txt-4- delta"), "{hits}");
        let both = grep(&cwd, "fn call", "notes.txt", &ctx);
        assert!(both.contains("\n--\n"), "{both}");

        // Output modes.
        let files = GrepOptions { output: GrepOutput::Files, ..plain_grep() };
        assert!(grep(&cwd, "fn call", "", &files).starts_with("notes.txt\n"));
        let count = GrepOptions { output: GrepOutput::Count, ..plain_grep() };
        assert!(grep(&cwd, "fn call", "", &count).starts_with("notes.txt: 2"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    fn project(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("soa-files-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {\n    println!(\"hi\");\n}\n")
            .unwrap();
        std::fs::write(root.join("src/lib.rs"), "pub fn add() {}\npub fn add2() {}\n").unwrap();
        std::fs::write(
            root.join(".git/config"),
            "[core]\nrepositoryformatversion = 0\n",
        )
        .unwrap();
        std::fs::write(root.join("README.md"), "# readme\n").unwrap();
        root
    }

    #[test]
    fn repository_walk_stops_immediately_when_consumer_is_full() {
        let cwd = project("early-stop").canonicalize().unwrap();
        let mut visited = 0usize;
        let truncated = walk_files(&cwd, &cwd, &mut |_, _| {
            visited += 1;
            false
        });
        assert!(truncated);
        assert_eq!(visited, 1);
        let _ = std::fs::remove_dir_all(cwd);
    }

    #[test]
    fn repository_search_uses_standard_git_ignores() {
        let cwd = project("git-ignore");
        let status = std::process::Command::new("git")
            .arg("init")
            .arg("--quiet")
            .arg(&cwd)
            .status()
            .unwrap();
        assert!(status.success());
        std::fs::write(cwd.join(".gitignore"), "ignored.log\n").unwrap();
        std::fs::write(cwd.join("ignored.log"), "needle\n").unwrap();
        std::fs::write(cwd.join("visible.txt"), "needle\n").unwrap();

        let files = glob(&cwd, "*");
        assert!(files.contains("visible.txt"), "{files}");
        assert!(!files.contains("ignored.log"), "{files}");
        let matches = grep(&cwd, "needle", "", &plain_grep());
        assert!(matches.contains("visible.txt:1"), "{matches}");
        assert!(!matches.contains("ignored.log"), "{matches}");
        let _ = std::fs::remove_dir_all(cwd);
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

    #[cfg(unix)]
    #[test]
    fn resolve_rejects_symlink_escapes_for_reads_and_new_writes() {
        use std::os::unix::fs::symlink;

        let cwd = project("resolve-symlinks");
        let outside = std::env::temp_dir().join(format!(
            "soa-files-test-{}-outside",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.txt"), "secret").unwrap();

        symlink(&outside, cwd.join("outside-link")).unwrap();
        assert!(
            resolve(&cwd, "outside-link/secret.txt")
                .unwrap_err()
                .contains("outside")
        );
        let write = write_file(&cwd, "outside-link/new.txt", "must stay contained");
        assert!(write.contains("outside"), "{write}");
        assert!(!outside.join("new.txt").exists());

        symlink(outside.join("secret.txt"), cwd.join("outside-file-link")).unwrap();
        let grep_result = grep(&cwd, "secret", "", &plain_grep());
        assert!(!grep_result.contains("outside-file-link"), "{grep_result}");
        assert_eq!(
            glob(&cwd, "*outside-file-link*"),
            "no files match `*outside-file-link*`"
        );

        // A dangling final symlink is an existing filesystem entry and must
        // not be mistaken for a safe new file.
        symlink(
            outside.join("dangling-target.txt"),
            cwd.join("dangling-link"),
        )
        .unwrap();
        let write = write_file(&cwd, "dangling-link", "must not follow");
        assert!(write.contains("cannot safely resolve"), "{write}");
        assert!(!outside.join("dangling-target.txt").exists());

        // Symlinks whose canonical target remains in the workspace continue
        // to work.
        symlink(cwd.join("src"), cwd.join("inside-link")).unwrap();
        assert_eq!(
            read_file(&cwd, "inside-link/main.rs", None, None, false),
            "fn main() {\n    println!(\"hi\");\n}\n"
        );

        let _ = std::fs::remove_dir_all(&cwd);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn read_write_edit_roundtrip() {
        let cwd = project("rwe");
        assert_eq!(
            read_file(&cwd, "src/main.rs", None, None, false),
            "fn main() {\n    println!(\"hi\");\n}\n"
        );
        assert_eq!(
            read_file(&cwd, "src/main.rs", Some(2), Some(1), false),
            "[lines 2-2 of 3]\n    println!(\"hi\");"
        );
        assert!(read_file(&cwd, "missing.rs", None, None, false).starts_with("ERROR"));

        // Write creates parents; edit requires a unique match.
        assert_eq!(write_file(&cwd, "deep/new.txt", "abc"), "created `deep/new.txt` (3 bytes)");
        assert_eq!(write_file(&cwd, "deep/new.txt", "abcd"), "overwrote `deep/new.txt` (4 bytes)");
        assert_eq!(
            edit_file(&cwd, "src/main.rs", "\"hi\"", "\"hello\"", false),
            "edited `src/main.rs` (1 replacement)"
        );
        assert!(read_file(&cwd, "src/main.rs", None, None, false).contains("hello"));
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
    fn anchored_reads_and_line_edits() {
        let cwd = project("anchors");
        let anchor_for = |path: &str, number: usize| -> String {
            let listing = read_file(&cwd, path, None, None, true);
            let line = listing.lines().nth(number - 1).unwrap();
            line.split('|').next().unwrap().to_string()
        };

        // Anchored reads prefix every line; windows keep absolute numbers.
        let listing = read_file(&cwd, "src/main.rs", None, None, true);
        assert!(listing.starts_with(&format!("1:{}|fn main() {{", line_hash("fn main() {"))));
        let window = read_file(&cwd, "src/main.rs", Some(2), Some(1), true);
        assert!(window.starts_with("[lines 2-2 of 3]\n2:"), "{window}");

        // Replace one line via its anchor; the reply carries fresh anchors.
        let anchor = anchor_for("src/main.rs", 2);
        let reply = edit_lines(&cwd, "src/main.rs", &anchor, None, "    println!(\"anchored\");", false);
        assert!(reply.starts_with("edited `src/main.rs`: replaced lines 2-2 with 1 line(s)"), "{reply}");
        assert!(reply.contains("fresh anchors"), "{reply}");
        assert!(read_file(&cwd, "src/main.rs", None, None, false).contains("anchored"));

        // The old anchor is now stale and rejected with the current line.
        let stale = edit_lines(&cwd, "src/main.rs", &anchor, None, "x", false);
        assert!(stale.contains("stale anchor"), "{stale}");
        assert!(stale.contains("re-read"), "{stale}");

        // Insert after line 1 without removing anything; then delete it by
        // range with an empty new_text.
        let top = anchor_for("src/main.rs", 1);
        let reply = edit_lines(&cwd, "src/main.rs", &top, None, "// header", true);
        assert!(reply.contains("inserted 1 line(s) after line 1"), "{reply}");
        let inserted = anchor_for("src/main.rs", 2);
        assert!(inserted.ends_with(&line_hash("// header")), "{inserted}");
        let reply = edit_lines(&cwd, "src/main.rs", &inserted, None, "", false);
        assert!(reply.contains("deleted lines 2-2"), "{reply}");
        assert!(!read_file(&cwd, "src/main.rs", None, None, false).contains("header"));

        // Range replacement across lines 1..3, trailing newline preserved.
        let (a1, a3) = (anchor_for("src/main.rs", 1), anchor_for("src/main.rs", 3));
        let reply = edit_lines(&cwd, "src/main.rs", &a1, Some(&a3), "fn main() {}", false);
        assert!(reply.contains("replaced lines 1-3 with 1 line(s)"), "{reply}");
        assert_eq!(read_file(&cwd, "src/main.rs", None, None, false), "fn main() {}\n");

        // Malformed and out-of-range anchors are corrected, not applied.
        assert!(edit_lines(&cwd, "src/main.rs", "42", None, "x", false)
            .contains("not a line anchor"));
        assert!(edit_lines(&cwd, "src/main.rs", "99:abcd", None, "x", false)
            .contains("out of range"));
        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[test]
    fn edit_lines_preserves_crlf_line_endings() {
        let cwd = project("crlf");
        std::fs::write(cwd.join("win.txt"), "alpha\r\nbeta\r\ngamma\r\n").unwrap();

        // Anchors hash the logical line (no \r), same as read_file shows.
        let anchor = format!("2:{}", line_hash("beta"));
        let reply = edit_lines(&cwd, "win.txt", &anchor, None, "BETA", false);
        assert!(reply.starts_with("edited `win.txt`"), "{reply}");
        assert_eq!(
            std::fs::read_to_string(cwd.join("win.txt")).unwrap(),
            "alpha\r\nBETA\r\ngamma\r\n",
            "untouched lines and the trailing newline must stay CRLF"
        );

        // Insertion adopts the file's line ending too.
        let anchor = format!("1:{}", line_hash("alpha"));
        edit_lines(&cwd, "win.txt", &anchor, None, "inserted", true);
        assert_eq!(
            std::fs::read_to_string(cwd.join("win.txt")).unwrap(),
            "alpha\r\ninserted\r\nBETA\r\ngamma\r\n"
        );
        let _ = std::fs::remove_dir_all(&cwd);
    }

    #[test]
    fn list_glob_grep() {
        let cwd = project("lgg");
        let listing = list_dir(&cwd, "");
        assert!(listing.contains("src/"), "{listing}");
        assert!(listing.contains("README.md"));

        // A slash-free pattern matches file names at any depth; `.git`
        // and friends are skipped.
        assert_eq!(glob(&cwd, "*.rs"), "src/lib.rs\nsrc/main.rs");
        assert_eq!(glob(&cwd, "src/**/*.rs"), "src/lib.rs\nsrc/main.rs");
        assert_eq!(glob(&cwd, "*config*"), "no files match `*config*`");

        let hits = grep(&cwd, r"println!", "", &plain_grep());
        assert!(hits.starts_with("src/main.rs:2:"), "{hits}");
        assert!(hits.contains("1 match(es) in 1 file(s)"), "{hits}");
        // Case-insensitive via inline flag; glob filter narrows files.
        let md_filter = GrepOptions { filter: Some("*.md"), ..plain_grep() };
        assert!(grep(&cwd, "(?i)READ", "", &md_filter).contains("README.md:1"));
        assert!(grep(&cwd, "println", "", &md_filter).starts_with("no matches"));
        assert!(grep(&cwd, "[invalid", "", &plain_grep()).starts_with("ERROR: invalid regular"));
        let _ = std::fs::remove_dir_all(&cwd);
    }
}
