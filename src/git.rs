//! Mining git history for reflection signals.
//!
//! Sessions only record what soa itself did; git records what *every*
//! actor did — other AI harnesses, hand edits — and what survived. This
//! module shells out to `git` (no library dependency) to fetch recent
//! commits and detect, heuristically:
//!
//!  - **reverts**: explicit "that was wrong" markers;
//!  - **corrections**: a commit rewriting lines that an earlier commit in
//!    the mined window introduced — the classic fix-up shape;
//!  - **revisions of soa-written code**: a commit removing lines that
//!    soa's session diff log recorded as added, i.e. direct feedback on
//!    soa's own output from whoever edited it afterwards.
//!
//! All detections are *candidates*, not certainties (features evolve for
//! reasons other than being wrong); the reflect prompt presents them as
//! such and cites commit hashes so lessons stay auditable.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, bail};

/// Per-commit patch text kept for analysis (huge commits are truncated —
/// vendored files and lockfiles are not where lessons live).
const MAX_FETCH_DIFF_CHARS: usize = 60_000;
/// At most this many correction links are reported per commit.
const MAX_CORRECTIONS_PER_COMMIT: usize = 2;

#[derive(Debug, Clone)]
pub struct Commit {
    pub hash: String,
    pub short: String,
    pub author: String,
    /// Author timestamp, epoch seconds.
    pub at: u64,
    pub subject: String,
    pub body: String,
    /// Unified diff, `--no-color`, possibly truncated.
    pub diff: String,
    /// `Co-Authored-By:` trailer names — how AI harnesses sign their work.
    pub co_authors: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingKind {
    Revert,
    Correction,
    SoaChangeRevised,
}

/// One detected pattern, tied to `commits[commit]` of the mined window.
#[derive(Debug, Clone)]
pub struct Finding {
    pub commit: usize,
    pub kind: FindingKind,
    pub detail: String,
}

fn git(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git").arg("-C").arg(root).args(args).output()?;
    if !output.status.success() {
        bail!(
            "git {} failed: {}",
            args.first().unwrap_or(&"?"),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// The repository containing `dir`, or None when it isn't inside one.
pub fn repo_root(dir: &Path) -> Option<PathBuf> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| PathBuf::from(String::from_utf8_lossy(&output.stdout).trim()))
}

/// Up to `max` non-merge commits reachable from HEAD, newest first,
/// excluding `since` and its ancestors when that rev still exists (it can
/// vanish under rebases — then the window is just capped at `max`). An
/// empty repository yields an empty list, not an error.
pub fn commits_since(root: &Path, since: Option<&str>, max: usize) -> Result<Vec<Commit>> {
    if git(root, &["rev-parse", "--verify", "--quiet", "HEAD"]).is_err() {
        return Ok(Vec::new()); // no commits yet
    }
    let max_arg = max.to_string();
    let mut args = vec!["rev-list", "--no-merges", "-n", &max_arg, "HEAD"];
    let exclude = since.map(|s| format!("^{s}"));
    if let Some(exclude) = &exclude
        && git(root, &["rev-parse", "--verify", "--quiet", &format!("{}^{{commit}}", &exclude[1..])])
            .is_ok()
    {
        args.push(exclude);
    }
    git(root, &args)?
        .lines()
        .map(|hash| fetch_commit(root, hash.trim()))
        .collect()
}

fn fetch_commit(root: &Path, hash: &str) -> Result<Commit> {
    let raw = git(
        root,
        &["show", hash, "--no-color", "--format=%H%x00%an%x00%at%x00%s%x00%b%x00", "--patch"],
    )?;
    let mut parts = raw.splitn(6, '\0');
    let mut next = || parts.next().unwrap_or("").to_string();
    let (hash, author, at, subject, body, diff) =
        (next(), next(), next(), next(), next(), next());
    let co_authors = body
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim().eq_ignore_ascii_case("co-authored-by").then(|| {
                value.split('<').next().unwrap_or(value).trim().to_string()
            })
        })
        .filter(|name| !name.is_empty())
        .collect();
    Ok(Commit {
        short: hash.chars().take(7).collect(),
        hash,
        author,
        at: at.trim().parse().unwrap_or(0),
        subject,
        body,
        diff: truncate(diff.trim_start_matches('\n'), MAX_FETCH_DIFF_CHARS),
        co_authors,
    })
}

/// Per-file (path, added, removed) line counts, parsed from the patch.
pub fn files_touched(diff: &str) -> Vec<(String, usize, usize)> {
    let mut files: Vec<(String, usize, usize)> = Vec::new();
    for line in diff.lines() {
        if let Some(path) = line.strip_prefix("+++ b/") {
            files.push((path.to_string(), 0, 0));
        } else if let Some((_, added, _)) = files.last_mut()
            && line.starts_with('+')
            && !line.starts_with("+++")
        {
            *added += 1;
        } else if let Some((_, _, removed)) = files.last_mut()
            && line.starts_with('-')
            && !line.starts_with("---")
        {
            *removed += 1;
        }
    }
    files
}

/// Whether a line is distinctive enough that reappearing elsewhere means
/// something (filters out braces, blank lines, and bare punctuation).
pub fn substantial_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.chars().count() >= 12 && trimmed.chars().any(|c| c.is_alphanumeric())
}

/// Match evidence: long lines are unlikely to coincide by accident.
fn weight(line: &str) -> usize {
    if line.chars().count() >= 40 { 2 } else { 1 }
}

/// Findings require this much combined match weight — one long line or
/// two shorter ones. Keeps single boilerplate-line coincidences quiet.
const MIN_STRENGTH: usize = 2;

fn patch_lines(diff: &str, prefix: char, skip: &str) -> BTreeSet<String> {
    diff.lines()
        .filter(|l| l.starts_with(prefix) && !l.starts_with(skip))
        .map(|l| l[1..].trim().to_string())
        .filter(|l| substantial_line(l))
        .collect()
}

/// Detect reverts, corrections within the window, and revisions of
/// soa-recorded changes. `commits` is newest first (as returned by
/// [`commits_since`]); `soa_added` maps lines soa wrote to the file it
/// wrote them in.
pub fn analyze(commits: &[Commit], soa_added: &BTreeMap<String, String>) -> Vec<Finding> {
    let added: Vec<BTreeSet<String>> =
        commits.iter().map(|c| patch_lines(&c.diff, '+', "+++")).collect();
    let mut findings = Vec::new();

    for (index, commit) in commits.iter().enumerate() {
        if commit.subject.starts_with("Revert ") || commit.body.contains("This reverts commit") {
            findings.push(Finding {
                commit: index,
                kind: FindingKind::Revert,
                detail: format!("explicitly reverts earlier work: {}", commit.subject),
            });
        }

        let removed = patch_lines(&commit.diff, '-', "---");
        if removed.is_empty() {
            continue;
        }

        // Corrections: this commit rewrites lines an older commit in the
        // window added (newest first, so older means a larger index).
        let mut corrections = 0usize;
        for (older_index, older) in commits.iter().enumerate().skip(index + 1) {
            if corrections >= MAX_CORRECTIONS_PER_COMMIT {
                break;
            }
            let (matched, strength) = removed
                .iter()
                .filter(|l| added[older_index].contains(*l))
                .fold((0, 0), |(n, s), l| (n + 1, s + weight(l)));
            if strength >= MIN_STRENGTH {
                corrections += 1;
                findings.push(Finding {
                    commit: index,
                    kind: FindingKind::Correction,
                    detail: format!(
                        "rewrites {matched} line(s) introduced in {} — \"{}\"",
                        older.short,
                        excerpt(&older.subject, 60),
                    ),
                });
            }
        }

        // Revisions of soa's own output: removed lines that soa's diff log
        // recorded as added.
        let hits: Vec<(&String, &String)> = removed
            .iter()
            .filter_map(|l| soa_added.get_key_value(l))
            .collect();
        let strength: usize = hits.iter().map(|(l, _)| weight(l)).sum();
        if strength >= MIN_STRENGTH {
            let mut paths: Vec<&str> =
                hits.iter().map(|(_, path)| path.as_str()).collect();
            paths.sort_unstable();
            paths.dedup();
            findings.push(Finding {
                commit: index,
                kind: FindingKind::SoaChangeRevised,
                detail: format!(
                    "changes {} line(s) that soa wrote in {}",
                    hits.len(),
                    paths.join(", "),
                ),
            });
        }
    }
    findings
}

/// Render the mined window for the reflect prompt: oldest first (a
/// timeline, like the session digests), findings attached to their
/// commits, diffs truncated to fit `budget` characters overall.
pub fn digest(commits: &[Commit], findings: &[Finding], budget: usize) -> String {
    let mut out = String::new();
    for (index, commit) in commits.iter().enumerate().rev() {
        let mut section = format!(
            "\n## Commit {} — {} ({})\n",
            commit.short,
            excerpt(&commit.subject, 100),
            commit.author,
        );
        if !commit.co_authors.is_empty() {
            section.push_str(&format!("co-authored-by: {}\n", commit.co_authors.join(", ")));
        }
        let files = files_touched(&commit.diff);
        if !files.is_empty() {
            let summary: Vec<String> = files
                .iter()
                .take(6)
                .map(|(path, added, removed)| format!("{path} (+{added} −{removed})"))
                .collect();
            section.push_str(&format!(
                "files: {}{}\n",
                summary.join(", "),
                if files.len() > 6 { format!(" … {} more", files.len() - 6) } else { String::new() },
            ));
        }
        for finding in findings.iter().filter(|f| f.commit == index) {
            let label = match finding.kind {
                FindingKind::Revert => "revert",
                FindingKind::Correction => "possible correction",
                FindingKind::SoaChangeRevised => "revises soa-written code",
            };
            section.push_str(&format!("⚑ {label}: {}\n", finding.detail));
        }
        section.push_str(&format!("diff:\n{}\n", excerpt_block(&commit.diff, 1_200)));
        if out.len() + section.len() > budget {
            out.push_str(&format!(
                "\n… {} older commit(s) omitted for space\n",
                index + 1
            ));
            break;
        }
        out.push_str(&section);
    }
    out
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let cut: String = text.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

fn excerpt(text: &str, max_chars: usize) -> String {
    truncate(&text.split_whitespace().collect::<Vec<_>>().join(" "), max_chars)
}

/// Truncate a multi-line block, keeping whole lines.
fn excerpt_block(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out = String::new();
    for line in text.lines() {
        if out.len() + line.len() + 1 > max_chars {
            out.push_str("… [diff truncated]");
            break;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway repo with identity settings that don't depend on the
    /// machine's git config.
    struct TestRepo(PathBuf);

    impl TestRepo {
        fn new(tag: &str) -> TestRepo {
            let root =
                std::env::temp_dir().join(format!("soa-git-test-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            std::fs::create_dir_all(&root).unwrap();
            let repo = TestRepo(root);
            repo.git(&["init", "-q", "-b", "main"]);
            repo
        }

        fn git(&self, args: &[&str]) {
            let status = Command::new("git")
                .arg("-C")
                .arg(&self.0)
                .args(["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"])
                .args(args)
                .status()
                .unwrap();
            assert!(status.success(), "git {args:?} failed");
        }

        fn commit(&self, file: &str, content: &str, message: &str) {
            std::fs::write(self.0.join(file), content).unwrap();
            self.git(&["add", "-A"]);
            self.git(&["commit", "-q", "-m", message]);
        }
    }

    impl Drop for TestRepo {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // Long enough (≥40 chars) that a single rewrite is a finding.
    const ORIGINAL: &str = "let total = recompute_final_answer(42) + baseline_offset;";
    const FIXED: &str = "let total = recompute_final_answer(43) + baseline_offset;";

    #[test]
    fn mines_commits_and_detects_patterns() {
        let repo = TestRepo::new("mine");
        repo.commit("f.rs", &format!("{ORIGINAL}\n"), "add feature");
        repo.commit(
            "f.rs",
            &format!("{FIXED}\n"),
            "fix off by one\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
        );

        let root = repo_root(&repo.0).expect("inside a repo");
        assert_eq!(
            root.canonicalize().unwrap(),
            repo.0.canonicalize().unwrap()
        );

        let commits = commits_since(&root, None, 10).unwrap();
        assert_eq!(commits.len(), 2);
        // Newest first, fields parsed, trailer picked up.
        assert_eq!(commits[0].subject, "fix off by one");
        assert_eq!(commits[0].co_authors, vec!["Claude"]);
        assert_eq!(commits[1].subject, "add feature");
        assert!(commits[1].co_authors.is_empty());
        assert!(commits[0].at > 0);
        assert_eq!(commits[0].short.len(), 7);
        assert_eq!(files_touched(&commits[0].diff), vec![("f.rs".to_string(), 1, 1)]);

        // The fix rewrites a line the first commit introduced.
        let findings = analyze(&commits, &BTreeMap::new());
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].commit, 0);
        assert_eq!(findings[0].kind, FindingKind::Correction);
        assert!(findings[0].detail.contains(&commits[1].short), "{}", findings[0].detail);

        // A `since` mark limits the window; a stale mark falls back to the cap.
        assert_eq!(commits_since(&root, Some(&commits[1].hash), 10).unwrap().len(), 1);
        assert_eq!(commits_since(&root, Some("0000000"), 10).unwrap().len(), 2);
        assert_eq!(commits_since(&root, None, 1).unwrap().len(), 1);

        // The digest reads oldest-first with markers attached.
        let digest = digest(&commits, &findings, 10_000);
        let add_at = digest.find("add feature").unwrap();
        let fix_at = digest.find("fix off by one").unwrap();
        assert!(add_at < fix_at, "{digest}");
        assert!(digest.contains("possible correction"));
        assert!(digest.contains("co-authored-by: Claude"));
    }

    #[test]
    fn detects_reverts_and_soa_revisions() {
        let repo = TestRepo::new("revert");
        repo.commit("f.rs", &format!("{ORIGINAL}\n"), "add feature");
        repo.git(&["revert", "--no-edit", "HEAD"]);
        let root = repo_root(&repo.0).unwrap();
        let commits = commits_since(&root, None, 10).unwrap();
        assert_eq!(commits.len(), 2);

        // The revert removes a line soa (per the fake map) had written.
        let soa_added =
            BTreeMap::from([(ORIGINAL.to_string(), "f.rs".to_string())]);
        let findings = analyze(&commits, &soa_added);
        let kinds: Vec<FindingKind> = findings.iter().map(|f| f.kind).collect();
        assert!(kinds.contains(&FindingKind::Revert));
        assert!(kinds.contains(&FindingKind::SoaChangeRevised));
        let revised =
            findings.iter().find(|f| f.kind == FindingKind::SoaChangeRevised).unwrap();
        assert!(revised.detail.contains("f.rs"), "{}", revised.detail);
        // The revert also reads as a correction of the original commit; all
        // three findings sit on the newest commit.
        assert!(findings.iter().all(|f| f.commit == 0));
    }

    #[test]
    fn handles_missing_and_empty_repos() {
        // Not a repo at all.
        assert!(repo_root(Path::new("/")).is_none());
        // A repo with no commits yields an empty window, not an error.
        let repo = TestRepo::new("empty");
        assert!(commits_since(&repo.0, None, 10).unwrap().is_empty());
    }

    #[test]
    fn line_filters_and_weights() {
        assert!(substantial_line("    let value = compute();"));
        assert!(!substantial_line("}"));
        assert!(!substantial_line("// ------------"));
        assert!(!substantial_line(""));
        assert_eq!(weight("short line of code"), 1);
        assert_eq!(weight(ORIGINAL), 2);

        // Boilerplate-sized single matches stay quiet: one 12-char line is
        // below MIN_STRENGTH.
        let commits = vec![
            Commit {
                hash: "b".into(),
                short: "b".into(),
                author: "t".into(),
                at: 2,
                subject: "later".into(),
                body: String::new(),
                diff: "+++ b/f.rs\n-use std::io;\n+use std::fs;\n".into(),
                co_authors: vec![],
            },
            Commit {
                hash: "a".into(),
                short: "a".into(),
                author: "t".into(),
                at: 1,
                subject: "earlier".into(),
                body: String::new(),
                diff: "+++ b/f.rs\n+use std::io;\n".into(),
                co_authors: vec![],
            },
        ];
        assert!(analyze(&commits, &BTreeMap::new()).is_empty());
    }
}
