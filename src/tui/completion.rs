//! Autocomplete for the chat input: slash commands and `@file` mentions.
//!
//! The engine is pure — [`compute`] maps (line, cursor, cwd) to an optional
//! popup state — so the popup logic is unit-testable without a terminal.
//! Inserts never include the `/` or `@` sigil: the replaced range starts
//! just after it, which keeps applying a completion a simple splice.

use std::path::Path;

/// Chat slash commands, shown in the popup with their descriptions.
/// (`/exit` is an undocumented alias of `/quit` and is left out.)
pub const COMMANDS: &[(&str, &str)] = &[
    ("branch", "save the conversation as a named branch"),
    ("branches", "switch between saved conversation lines"),
    ("clear", "drop all conversation context"),
    ("compact", "summarize the conversation and shrink context"),
    ("diff", "open the diff viewer"),
    ("export", "write the transcript to a markdown file"),
    ("help", "list commands and keys"),
    ("model", "override the model for this session"),
    ("quit", "exit"),
    ("reload", "re-read the config file"),
    ("rewind", "rewind conversation and files to a previous message"),
    ("sessions", "open the session picker"),
    ("stage", "switch the active stage"),
    ("usage", "cumulative token usage per model"),
];

const MAX_ITEMS: usize = 8;

pub struct Completion {
    pub items: Vec<Item>,
    pub selected: usize,
    /// Char offset in the cursor's line where the replaced token starts
    /// (just after the `/` or `@` sigil). Applying a completion replaces
    /// `line[replace_from..cursor]` with the selected item's `insert`.
    pub replace_from: usize,
}

#[derive(Clone)]
pub struct Item {
    /// What the popup shows (directories get a trailing `/`).
    pub label: String,
    /// Dim annotation: a command description, or empty.
    pub detail: String,
    /// Replacement text for the token being completed.
    pub insert: String,
}

/// Completions for the cursor position, or None when no popup applies.
/// `line` is the cursor's line, `cursor` a char offset into it, and
/// `first_line` whether the cursor is on the input's first line (slash
/// commands only trigger there).
pub fn compute(
    line: &str,
    cursor: usize,
    first_line: bool,
    cwd: &Path,
    stage_names: &[String],
    model_names: &[String],
) -> Option<Completion> {
    let chars: Vec<char> = line.chars().collect();
    let cursor = cursor.min(chars.len());
    let prefix = &chars[..cursor];

    // Slash commands: a `/token` at the very start of the input.
    if first_line && chars.first() == Some(&'/') {
        let token: String = prefix.get(1..).unwrap_or_default().iter().collect();
        if !token.contains(' ') {
            let items = COMMANDS
                .iter()
                .filter(|(name, _)| name.starts_with(&token))
                .map(|(name, description)| Item {
                    label: name.to_string(),
                    detail: description.to_string(),
                    insert: name.to_string(),
                })
                .collect();
            return build(items, 1);
        }
        // `/stage <partial>` and `/model <partial>`: complete the argument.
        let (partial, candidates): (&str, Vec<&str>) =
            if let Some(partial) = token.strip_prefix("stage ") {
                (partial, stage_names.iter().map(String::as_str).collect())
            } else if let Some(partial) = token.strip_prefix("model ") {
                let mut names: Vec<&str> = model_names.iter().map(String::as_str).collect();
                names.push("default");
                (partial, names)
            } else {
                return None;
            };
        if partial.contains(' ') {
            return None;
        }
        let items = candidates
            .into_iter()
            .filter(|name| name.starts_with(partial))
            .map(|name| Item {
                label: name.to_string(),
                detail: String::new(),
                insert: name.to_string(),
            })
            .collect();
        return build(items, cursor - partial.chars().count());
    }

    // `@path` mentions anywhere in the input.
    let at = mention_start(prefix)?;
    let partial: String = prefix[at + 1..].iter().collect();
    build(fs_items(cwd, &partial), at + 1)
}

fn build(items: Vec<Item>, replace_from: usize) -> Option<Completion> {
    (!items.is_empty()).then_some(Completion { items, selected: 0, replace_from })
}

/// Index of the `@` opening the mention token the cursor is inside, if
/// any. Matches the boundary rule in [`crate::mentions`]: the `@` must not
/// be preceded by an alphanumeric character (so emails don't trigger), and
/// the token may not contain whitespace or quotes.
fn mention_start(prefix: &[char]) -> Option<usize> {
    for i in (0..prefix.len()).rev() {
        let c = prefix[i];
        if c == '@' {
            let boundary = i == 0 || !prefix[i - 1].is_alphanumeric();
            return boundary.then_some(i);
        }
        if c.is_whitespace() || c == '"' {
            return None;
        }
    }
    None
}

/// Filesystem candidates for a partial path like `src/tu`: entries of
/// `cwd/src` whose names start with `tu` (case-insensitive). Hidden files
/// are listed only when the name part itself starts with a dot.
/// Directories sort first and complete with a trailing `/` so accepting
/// one descends into it; names with spaces insert quoted.
fn fs_items(cwd: &Path, partial: &str) -> Vec<Item> {
    let (dir_part, name_prefix) = match partial.rfind('/') {
        Some(slash) => (&partial[..=slash], &partial[slash + 1..]),
        None => ("", partial),
    };
    let base = cwd.join(dir_part);
    let Ok(entries) = std::fs::read_dir(&base) else { return Vec::new() };

    let prefix_lower = name_prefix.to_lowercase();
    let mut found: Vec<(bool, String)> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') && !name_prefix.starts_with('.') {
                return None;
            }
            if !name.to_lowercase().starts_with(&prefix_lower) {
                return None;
            }
            let is_dir = entry.file_type().is_ok_and(|t| t.is_dir());
            Some((is_dir, name))
        })
        .collect();
    found.sort_by_key(|(is_dir, name)| (!is_dir, name.to_lowercase()));
    found.truncate(MAX_ITEMS);

    found
        .into_iter()
        .map(|(is_dir, name)| {
            let path = format!("{dir_part}{name}{}", if is_dir { "/" } else { "" });
            let insert =
                if path.contains(' ') { format!("\"{path}\"") } else { path.clone() };
            Item {
                label: format!("{name}{}", if is_dir { "/" } else { "" }),
                detail: String::new(),
                insert,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn compute_at(line: &str, cwd: &Path) -> Option<Completion> {
        compute(
            line,
            line.chars().count(),
            true,
            cwd,
            &["research".into(), "review".into()],
            &["planner".into(), "coder".into()],
        )
    }

    #[test]
    fn completes_slash_commands_and_stage_names() {
        let cwd = PathBuf::from("/");
        let c = compute_at("/c", &cwd).unwrap();
        assert_eq!(
            c.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            vec!["clear", "compact"]
        );
        assert_eq!(c.replace_from, 1);

        // Full command still yields its (identical) completion, so the key
        // handler can tell "would change nothing" and submit instead.
        let c = compute_at("/usage", &cwd).unwrap();
        assert_eq!(c.items[0].insert, "usage");

        // Second token of /stage completes stage names.
        let c = compute_at("/stage re", &cwd).unwrap();
        assert_eq!(
            c.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            vec!["research", "review"]
        );
        assert_eq!(c.replace_from, "/stage ".len());

        // `/model` completes model names plus the `default` reset.
        let c = compute_at("/model ", &cwd).unwrap();
        assert_eq!(
            c.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            vec!["planner", "coder", "default"]
        );

        // Unknown prefixes and non-first lines don't pop up.
        assert!(compute_at("/zzz", &cwd).is_none());
        assert!(compute("/c", 2, false, &cwd, &[], &[]).is_none());
        // Other commands take no arguments.
        assert!(compute_at("/help me", &cwd).is_none());
    }

    #[test]
    fn completes_file_mentions() {
        let root = std::env::temp_dir().join(format!("soa-complete-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("main.rs"), "").unwrap();
        std::fs::write(root.join("Makefile"), "").unwrap();
        std::fs::write(root.join(".hidden"), "").unwrap();
        std::fs::write(root.join("src").join("main.rs"), "").unwrap();
        std::fs::write(root.join("has space.txt"), "").unwrap();

        // Directories first, hidden files skipped, case-insensitive match.
        let c = compute_at("look at @", &root).unwrap();
        assert_eq!(
            c.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            vec!["src/", "has space.txt", "main.rs", "Makefile"]
        );
        assert_eq!(c.replace_from, "look at @".chars().count());
        let c = compute_at("@ma", &root).unwrap();
        assert_eq!(
            c.items.iter().map(|i| i.label.as_str()).collect::<Vec<_>>(),
            vec!["main.rs", "Makefile"]
        );

        // Dotted prefix reveals hidden files; insert keeps the dir part.
        assert_eq!(compute_at("@.hi", &root).unwrap().items[0].label, ".hidden");
        let c = compute_at("@src/ma", &root).unwrap();
        assert_eq!(c.items[0].insert, "src/main.rs");

        // Names with spaces insert quoted.
        assert_eq!(compute_at("@has", &root).unwrap().items[0].insert, "\"has space.txt\"");

        // Emails and quoted tokens don't trigger.
        assert!(compute_at("mail me@ma", &root).is_none());
        assert!(compute_at("@\"has sp", &root).is_none());

        let _ = std::fs::remove_dir_all(&root);
    }
}
