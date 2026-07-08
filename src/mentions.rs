//! `@file` mentions: referencing `@src/main.rs` in a prompt attaches that
//! file's content to the message; `@some/dir` attaches a shallow listing.
//!
//! Mentions are recognized at word boundaries (so `user@host` is left
//! alone), resolve relative to the working directory (absolute paths work
//! too), and support quoting for paths with spaces: `@"my file.txt"`.
//! The original message text is preserved; attachments are appended after
//! it, so the model sees both the reference and the content.

use std::path::{Path, PathBuf};

const MAX_DIRECTORY_ENTRIES: usize = 200;

/// What happened to one `@` mention, for user feedback.
#[derive(Debug, PartialEq)]
pub enum MentionStatus {
    File { lines: usize, truncated: bool },
    Directory { entries: usize },
    NotFound,
    Unreadable,
}

#[derive(Debug)]
pub struct MentionReport {
    /// The path as the user wrote it.
    pub display: String,
    pub status: MentionStatus,
}

impl MentionReport {
    pub fn describe(&self) -> String {
        match &self.status {
            MentionStatus::File { lines, truncated: false } => {
                format!("@{} attached ({lines} lines)", self.display)
            }
            MentionStatus::File { lines, truncated: true } => {
                format!("@{} attached ({lines} lines, truncated)", self.display)
            }
            MentionStatus::Directory { entries } => {
                format!("@{} listed ({entries} entries)", self.display)
            }
            MentionStatus::NotFound => format!("@{} not found", self.display),
            MentionStatus::Unreadable => {
                format!("@{} could not be read as text", self.display)
            }
        }
    }
}

/// Expand `@` mentions in `text`: returns the message with attachment
/// blocks appended, plus a report per mention. `max_chars` bounds each
/// attached file (0 = unlimited).
pub fn expand_mentions(
    text: &str,
    cwd: &Path,
    max_chars: usize,
) -> (String, Vec<MentionReport>) {
    let mut reports = Vec::new();
    let mut attachments = String::new();

    for candidate in find_mentions(text) {
        // Sentence punctuation sticks to bare tokens ("see @src/main.rs.");
        // if the token doesn't resolve, retry with it trimmed.
        let resolved = resolve(&candidate, cwd).or_else(|| {
            let trimmed = candidate.trim_end_matches([',', '.', ';', ':', '!', '?', ')']);
            (trimmed != candidate && !trimmed.is_empty())
                .then(|| resolve(trimmed, cwd))
                .flatten()
        });

        let Some((display, path)) = resolved else {
            // Only flag path-looking tokens; "@here" style words pass through.
            if candidate.contains('/') || candidate.contains('.') {
                reports.push(MentionReport {
                    display: candidate.clone(),
                    status: MentionStatus::NotFound,
                });
            }
            continue;
        };

        if reports.iter().any(|r| r.display == display && r.status != MentionStatus::NotFound)
        {
            continue; // same file mentioned twice
        }

        if path.is_dir() {
            let (listing, entries) = list_directory(&path);
            attachments.push_str(&format!("\n\n[Directory listing: {display}]\n{listing}"));
            reports.push(MentionReport {
                display,
                status: MentionStatus::Directory { entries },
            });
        } else {
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    let lines = content.lines().count();
                    let (content, truncated) = clamp(&content, max_chars);
                    attachments.push_str(&format!(
                        "\n\n[Attached file: {display}]\n```\n{}\n```{}",
                        content.trim_end(),
                        if truncated { "\n(truncated)" } else { "" },
                    ));
                    reports.push(MentionReport {
                        display,
                        status: MentionStatus::File { lines, truncated },
                    });
                }
                Err(_) => {
                    reports.push(MentionReport {
                        display,
                        status: MentionStatus::Unreadable,
                    });
                }
            }
        }
    }

    (format!("{text}{attachments}"), reports)
}

/// Candidate paths from `@` tokens at word boundaries.
fn find_mentions(text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut boundary = true;
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if ch != '@' || !boundary {
            boundary = ch.is_whitespace() || matches!(ch, '(' | '[' | '{' | ',');
            continue;
        }
        let rest = &text[index + 1..];
        let candidate = if let Some(quoted) = rest.strip_prefix('"') {
            quoted.split('"').next().unwrap_or_default().to_string()
        } else {
            rest.split_whitespace().next().unwrap_or_default().to_string()
        };
        if !candidate.is_empty() {
            found.push(candidate.clone());
            // Skip past the token so `@a@b` doesn't double-trigger.
            for _ in 0..candidate.chars().count() {
                chars.next();
            }
        }
        boundary = false;
    }
    found
}

fn resolve(candidate: &str, cwd: &Path) -> Option<(String, PathBuf)> {
    let path = if Path::new(candidate).is_absolute() {
        PathBuf::from(candidate)
    } else {
        cwd.join(candidate)
    };
    path.exists().then(|| (candidate.to_string(), path))
}

fn list_directory(path: &Path) -> (String, usize) {
    let mut names: Vec<String> = std::fs::read_dir(path)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| {
                    let mut name = e.file_name().to_string_lossy().into_owned();
                    if e.path().is_dir() {
                        name.push('/');
                    }
                    name
                })
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    let total = names.len();
    names.truncate(MAX_DIRECTORY_ENTRIES);
    let mut listing =
        names.iter().map(|n| format!("- {n}")).collect::<Vec<_>>().join("\n");
    if total > MAX_DIRECTORY_ENTRIES {
        listing.push_str(&format!("\n… and {} more", total - MAX_DIRECTORY_ENTRIES));
    }
    (listing, total)
}

fn clamp(content: &str, max_chars: usize) -> (String, bool) {
    if max_chars == 0 || content.chars().count() <= max_chars {
        (content.to_string(), false)
    } else {
        (content.chars().take(max_chars).collect(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_workspace(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("soa-mentions-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("notes.txt"), "line one\nline two\n").unwrap();
        std::fs::write(dir.join("my file.txt"), "spaced\n").unwrap();
        std::fs::write(dir.join("sub/inner.rs"), "fn main() {}\n").unwrap();
        dir
    }

    #[test]
    fn finds_mentions_at_boundaries_only() {
        assert_eq!(find_mentions("see @a/b and (@c) but not user@host"), vec!["a/b", "c)"]);
        assert_eq!(find_mentions("@\"my file.txt\" end"), vec!["my file.txt"]);
        assert_eq!(find_mentions("no mentions here"), Vec::<String>::new());
    }

    #[test]
    fn expands_files_dirs_and_reports_missing() {
        let dir = temp_workspace("expand");
        let text = "check @notes.txt and @sub plus @ghost.rs";
        let (expanded, reports) = expand_mentions(text, &dir, 0);

        assert!(expanded.starts_with(text)); // original preserved
        assert!(expanded.contains("[Attached file: notes.txt]\n```\nline one\nline two\n```"));
        assert!(expanded.contains("[Directory listing: sub]\n- inner.rs"));
        assert_eq!(reports.len(), 3);
        assert_eq!(reports[0].status, MentionStatus::File { lines: 2, truncated: false });
        assert_eq!(reports[1].status, MentionStatus::Directory { entries: 1 });
        assert_eq!(reports[2].status, MentionStatus::NotFound);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quoted_paths_punctuation_and_truncation() {
        let dir = temp_workspace("quoted");

        // Quoted path with a space.
        let (expanded, reports) = expand_mentions("read @\"my file.txt\"!", &dir, 0);
        assert!(expanded.contains("[Attached file: my file.txt]"));
        assert_eq!(reports[0].status, MentionStatus::File { lines: 1, truncated: false });

        // Trailing sentence punctuation retried.
        let (_, reports) = expand_mentions("see @notes.txt.", &dir, 0);
        assert_eq!(reports[0].status, MentionStatus::File { lines: 2, truncated: false });

        // Truncation.
        let (expanded, reports) = expand_mentions("@notes.txt", &dir, 5);
        assert!(expanded.contains("(truncated)"));
        assert_eq!(reports[0].status, MentionStatus::File { lines: 2, truncated: true });

        // Bare words without path characters aren't flagged.
        let (_, reports) = expand_mentions("hey @everyone", &dir, 0);
        assert!(reports.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
