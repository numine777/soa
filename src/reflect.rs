//! `soa reflect`: distill saved sessions into persistent instructions.
//!
//! Reflection reads this directory's saved chat sessions (skipping ones
//! already reflected on), extracts failure signals (see [`crate::insights`])
//! and a digest of what was asked and answered, and has a model rewrite two
//! kinds of durable memory:
//!
//!  - **Lessons**: short imperative rules kept in a marker-delimited block
//!    of the project's `SOA.md`, which `settings.context_files` already
//!    appends to every stage and agent system prompt. The model returns the
//!    complete replacement list each run, so lessons get consolidated and
//!    pruned instead of accreting forever.
//!  - **Skills**: full skill files under the project skills directory for
//!    recurring multi-step procedures, loadable per-stage via
//!    `skills = [...]`. Reflect only overwrites skill files it authored
//!    (marked `generated: soa reflect` in their frontmatter).
//!
//! Everything is plain files: review the changes with `git diff`, revert
//! with `git checkout`. Extracted signals are appended to the insights
//! store so future tooling can reuse them without re-parsing sessions.

use std::fmt::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};

use crate::config::Config;
use crate::insights::{self, Signal};
use crate::model::Message;
use crate::tui::store::{self, Session};

/// Hard caps that keep the prompt bounded and the written files reviewable.
const MAX_SESSIONS_PER_RUN: usize = 12;
const MAX_DIGEST_CHARS: usize = 24_000;
const MAX_LESSONS: usize = 15;
const MAX_LESSON_CHARS: usize = 300;
const MAX_SKILLS_PER_RUN: usize = 3;
/// Git mining window and its share of the prompt.
const MAX_GIT_COMMITS: usize = 20;
const MAX_GIT_DIGEST_CHARS: usize = 12_000;
/// Sessions whose diff logs feed the "lines soa wrote" index, and its cap.
const MAX_SOA_LINE_SESSIONS: usize = 30;
const MAX_SOA_LINES: usize = 5_000;

const LESSONS_START: &str =
    "<!-- soa:lessons:start — managed by `soa reflect`; edit outside this block -->";
const LESSONS_END: &str = "<!-- soa:lessons:end -->";
const GENERATED_MARKER: &str = "generated: soa reflect";

const SYSTEM_PROMPT: &str = "\
You maintain the persistent memory of `soa`, a staged coding agent. You are \
given the current lesson list, the available skills, and digests of recent \
chat sessions including failure signals (denied tool calls, tool errors, \
rolled-back file changes).

Reply with ONLY a JSON object, no prose or code fences, in this shape:
{
  \"lessons\": [\"...\"],
  \"skills\": [{\"name\": \"kebab-case-name\", \"description\": \"one line\", \"body\": \"markdown\"}],
  \"note\": \"one short paragraph for the user summarizing what you changed and why\"
}

lessons: the COMPLETE replacement list. Start from the current lessons: keep \
the ones that still apply, merge duplicates, drop obsolete ones, and add new \
ones justified by the digests. Each lesson is one short imperative sentence \
naming the concrete situation it applies to (tool, file kind, or command). \
Never invent lessons the digests do not support; fewer good lessons beat \
many vague ones. An unchanged list is a valid answer.

skills: rarely, when the digests show the same multi-step procedure \
recurring, write it up as a skill (markdown body with concrete commands and \
checks). Otherwise return an empty array.

A \"Recent commits\" section, when present, is the project's git history \
from EVERY tool — other AI harnesses and hand edits included, not just \
soa. Its ⚑ markers (possible correction, revert, revises soa-written \
code) are heuristic candidates: code also changes because requirements \
changed, so treat a marker as evidence only when the diff supports it. A \
lesson derived from commits must cite the short hash, e.g. \
\"(commit abc1234)\".";

/// The model's parsed reply.
#[derive(Debug, serde::Deserialize)]
struct Proposal {
    #[serde(default)]
    lessons: Vec<String>,
    #[serde(default)]
    skills: Vec<SkillProposal>,
    #[serde(default)]
    note: String,
}

#[derive(Debug, serde::Deserialize)]
struct SkillProposal {
    name: String,
    #[serde(default)]
    description: String,
    body: String,
}

/// Run one reflection pass over this directory's unreflected sessions.
/// `dry_run` prints the proposal without writing anything.
pub async fn run(config: &Config, model_override: Option<&str>, dry_run: bool) -> Result<()> {
    let cwd = store::current_cwd();
    let model = match model_override {
        Some(name) => name.to_string(),
        None => config
            .settings
            .reflect_model
            .clone()
            .or_else(|| config.stages.first().map(|s| s.model.clone()))
            .context("no model to reflect with (set settings.reflect_model)")?,
    };

    // Newest-first from the store; keep the newest batch, then reflect in
    // chronological order so digests read as a timeline.
    let reflected = insights::load_reflected();
    let all_sessions: Vec<Session> = store::list_sessions()?
        .into_iter()
        .filter(|s| s.cwd == cwd)
        .collect();
    let mut sessions: Vec<Session> = all_sessions
        .iter()
        .filter(|s| reflected.get(&s.id) != Some(&s.updated_at))
        .take(MAX_SESSIONS_PER_RUN)
        .cloned()
        .collect();
    sessions.reverse();

    // Git mining: commits since the last reflected mark — the record of
    // what every tool (and hand edit) did, not just soa.
    let repo = crate::git::repo_root(std::path::Path::new(&cwd));
    let mut git_marks = insights::load_git_marks();
    let (commits, findings) = match &repo {
        Some(root) => {
            let mark = git_marks
                .get(&root.display().to_string())
                .map(String::as_str);
            match crate::git::commits_since(root, mark, MAX_GIT_COMMITS) {
                Ok(commits) => {
                    let soa_added = soa_added_lines(&all_sessions);
                    let findings = crate::git::analyze(&commits, &soa_added);
                    (commits, findings)
                }
                Err(e) => {
                    eprintln!("⚠ git mining skipped: {e:#}");
                    (Vec::new(), Vec::new())
                }
            }
        }
        None => (Vec::new(), Vec::new()),
    };

    if sessions.is_empty() && commits.is_empty() {
        println!("nothing new to reflect on (no unreflected sessions or commits for {cwd})");
        return Ok(());
    }

    let lessons_path = lessons_file(config);
    let existing_lessons = read_lessons(&lessons_path);
    let skill_summaries: Vec<String> = crate::skills::list_skills(config)
        .iter()
        .map(|s| {
            format!(
                "{} — {}",
                s.name,
                if s.description.is_empty() {
                    "(no description)"
                } else {
                    &s.description
                }
            )
        })
        .collect();

    // Digest until the prompt budget runs out; sessions that don't fit are
    // NOT marked reflected, so the next run picks them up.
    let mut all_signals: Vec<Signal> = Vec::new();
    let mut digests = String::new();
    let mut digested = 0usize;
    for session in &sessions {
        let signals = insights::extract_signals(session);
        let digest = digest_session(session, &signals);
        if digested > 0 && digests.len() + digest.len() > MAX_DIGEST_CHARS {
            eprintln!(
                "⚠ digest budget reached — {} session(s) deferred to the next run",
                sessions.len() - digested,
            );
            break;
        }
        digests.push_str(&digest);
        all_signals.extend(signals);
        digested += 1;
    }
    let sessions = &sessions[..digested];

    // Git findings become signals too: persisted alongside the session
    // ones, each carrying the commit hash it points at.
    let git_signals: Vec<Signal> = findings
        .iter()
        .map(|finding| {
            let commit = &commits[finding.commit];
            Signal {
                session_id: String::new(),
                cwd: cwd.clone(),
                at: commit.at,
                kind: match finding.kind {
                    crate::git::FindingKind::Revert => insights::SignalKind::Revert,
                    crate::git::FindingKind::Correction => insights::SignalKind::Correction,
                    crate::git::FindingKind::SoaChangeRevised => {
                        insights::SignalKind::SoaChangeRevised
                    }
                },
                tool: "git".to_string(),
                commit: commit.hash.clone(),
                excerpt: format!("commit {}: {}", commit.short, finding.detail),
            }
        })
        .collect();
    all_signals.extend(git_signals);

    let git_section = if commits.is_empty() {
        String::new()
    } else {
        format!(
            "\n\n# Recent commits (from every tool: other harnesses, hand edits, soa)\n{}",
            crate::git::digest(&commits, &findings, MAX_GIT_DIGEST_CHARS),
        )
    };
    let user_prompt = format!(
        "# Current lessons\n{}\n\n# Existing skills\n{}\n\n# Session digests\n{}{}",
        if existing_lessons.is_empty() {
            "(none yet)".to_string()
        } else {
            existing_lessons
                .iter()
                .map(|l| format!("- {l}"))
                .collect::<Vec<_>>()
                .join("\n")
        },
        if skill_summaries.is_empty() {
            "(none)".to_string()
        } else {
            skill_summaries.join("\n")
        },
        if digests.is_empty() {
            "(none new)".to_string()
        } else {
            digests
        },
        git_section,
    );

    eprintln!(
        "reflecting on {} session(s) and {} commit(s) with model `{model}` ({} signal(s))…",
        sessions.len(),
        commits.len(),
        all_signals.len(),
    );
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            config.settings.request_timeout_secs,
        ))
        .build()
        .context("failed to build HTTP client")?;
    let usage = crate::model::UsageTracker::unlimited();
    let client = crate::stage::build_model_client(config, &model, None, None, &http, &usage)?;
    let messages = vec![
        Message::System {
            content: SYSTEM_PROMPT.to_string(),
        },
        Message::User {
            content: user_prompt,
        },
    ];
    let reply = client.complete(&messages, &[]).await?;
    let text = reply.content.unwrap_or_default();
    let proposal = parse_proposal(&text)
        .with_context(|| format!("model did not return usable JSON:\n{}", excerpt(&text, 400)))?;

    let (lessons, wipe_refused) =
        guard_lesson_wipe(&existing_lessons, validate_lessons(proposal.lessons));
    if wipe_refused {
        eprintln!(
            "⚠ the model proposed deleting all {} lesson(s); keeping the existing list — \
             edit the lessons block in the file directly if you really want it cleared",
            existing_lessons.len()
        );
    }
    let skills: Vec<(String, SkillProposal)> = proposal
        .skills
        .into_iter()
        .filter_map(|skill| slug(&skill.name).map(|name| (name, skill)))
        .take(MAX_SKILLS_PER_RUN)
        .collect();

    if !proposal.note.is_empty() {
        println!("{}\n", proposal.note.trim());
    }
    print_lessons_diff(&existing_lessons, &lessons);

    if dry_run {
        for (name, skill) in &skills {
            println!("would write skill `{name}`: {}", skill.description);
        }
        println!("\n(dry run — nothing written)");
        return Ok(());
    }

    // Lessons: rewrite the managed block, leaving the rest of the file
    // untouched.
    if lessons != existing_lessons {
        let current = std::fs::read_to_string(&lessons_path).unwrap_or_default();
        let updated = replace_lessons_block(&current, &lessons);
        std::fs::write(&lessons_path, updated)
            .with_context(|| format!("cannot write {}", lessons_path.display()))?;
        println!("updated {}", lessons_path.display());
        let covered = config
            .settings
            .context_files
            .iter()
            .any(|f| f.file_name() == lessons_path.file_name());
        if !covered {
            eprintln!(
                "⚠ {} is not in settings.context_files — lessons won't reach the models \
                 until it is added",
                lessons_path.display()
            );
        }
    } else {
        println!("lessons unchanged");
    }

    // Skills: whole files in the project skills dir, overwriting only what
    // reflect itself authored.
    let skills_dir = crate::skills::skills_dirs(config)
        .into_iter()
        .next()
        .expect("skills_dirs always returns the project dir first");
    for (name, skill) in &skills {
        let path = skills_dir.join(format!("{name}.md"));
        if let Ok(current) = std::fs::read_to_string(&path)
            && !current.contains(GENERATED_MARKER)
        {
            eprintln!(
                "⚠ skipped skill `{name}`: {} exists and was not written by reflect",
                path.display()
            );
            continue;
        }
        std::fs::create_dir_all(&skills_dir)
            .with_context(|| format!("cannot create {}", skills_dir.display()))?;
        let content = format!(
            "---\nname: {name}\ndescription: {}\n{GENERATED_MARKER} ({})\n---\n\n{}\n",
            skill.description.replace('\n', " "),
            store::format_epoch(store::now_epoch()),
            skill.body.trim(),
        );
        std::fs::write(&path, content)
            .with_context(|| format!("cannot write {}", path.display()))?;
        println!(
            "wrote skill `{name}` ({}) — attach it with skills = [\"{name}\"]",
            path.display()
        );
    }

    // Bookkeeping: persist the signals, mark the sessions reflected, and
    // advance the git mark to the newest mined commit.
    insights::append_signals(&all_signals)?;
    let mut reflected = reflected;
    for session in sessions {
        reflected.insert(session.id.clone(), session.updated_at);
    }
    insights::save_reflected(&reflected)?;
    if let (Some(root), Some(newest)) = (&repo, commits.first()) {
        git_marks.insert(root.display().to_string(), newest.hash.clone());
        insights::save_git_marks(&git_marks)?;
    }
    println!(
        "reflected on {} session(s) and {} commit(s); {} signal(s) recorded in the insights store",
        sessions.len(),
        commits.len(),
        all_signals.len(),
    );
    Ok(())
}

/// Where lessons live: `SOA.md` next to the config file.
fn lessons_file(config: &Config) -> PathBuf {
    config.base_dir.join("SOA.md")
}

/// Index of lines soa itself wrote (added lines from recorded session
/// diffs) to the file it wrote them in — matched against commit diffs to
/// spot soa's output being revised downstream. `sessions` is newest first.
fn soa_added_lines(sessions: &[Session]) -> std::collections::BTreeMap<String, String> {
    let mut lines = std::collections::BTreeMap::new();
    for session in sessions.iter().take(MAX_SOA_LINE_SESSIONS) {
        for entry in &session.diffs {
            if entry.tool == "rewind" {
                continue; // a rollback is not soa authorship
            }
            for line in entry.unified.lines() {
                if lines.len() >= MAX_SOA_LINES {
                    return lines;
                }
                if let Some(added) = line.strip_prefix('+')
                    && !line.starts_with("+++")
                    && crate::git::substantial_line(added)
                {
                    lines
                        .entry(added.trim().to_string())
                        .or_insert_with(|| entry.path.clone());
                }
            }
        }
    }
    lines
}

/// One session as prompt text: what was asked, how it ended, what failed.
fn digest_session(session: &Session, signals: &[Signal]) -> String {
    let mut out = format!(
        "\n## Session {} — {}\n",
        session.id,
        excerpt(&session.title, 100)
    );
    let user_messages: Vec<&str> = session
        .history
        .iter()
        .filter_map(|m| match m {
            Message::User { content } => Some(content.as_str()),
            _ => None,
        })
        .collect();
    for message in user_messages.iter().take(8) {
        let _ = writeln!(out, "user: {}", excerpt(message, 160));
    }
    if user_messages.len() > 8 {
        let _ = writeln!(out, "… {} more user message(s)", user_messages.len() - 8);
    }
    if let Some(Message::Assistant {
        content: Some(content),
        ..
    }) = session.history.iter().rev().find(|m| {
        matches!(
            m,
            Message::Assistant {
                content: Some(_),
                ..
            }
        )
    }) {
        let _ = writeln!(out, "final reply: {}", excerpt(content, 240));
    }
    if signals.is_empty() {
        let _ = writeln!(out, "signals: none");
    } else {
        for signal in signals {
            let _ = writeln!(
                out,
                "signal [{}] {}: {}",
                match signal.kind {
                    insights::SignalKind::Denied => "denied",
                    insights::SignalKind::ToolError => "tool-error",
                    insights::SignalKind::Rollback => "rollback",
                    // Git kinds never come from a session, but stay
                    // printable if that ever changes.
                    insights::SignalKind::Revert => "revert",
                    insights::SignalKind::Correction => "correction",
                    insights::SignalKind::SoaChangeRevised => "soa-change-revised",
                },
                signal.tool,
                signal.excerpt,
            );
        }
    }
    out
}

/// Parse the model's reply, tolerating prose or code fences around the
/// JSON object.
fn parse_proposal(text: &str) -> Result<Proposal> {
    if let Ok(proposal) = serde_json::from_str::<Proposal>(text.trim()) {
        return Ok(proposal);
    }
    let start = text
        .find('{')
        .ok_or_else(|| anyhow!("no JSON object in the reply"))?;
    let end = text
        .rfind('}')
        .ok_or_else(|| anyhow!("no JSON object in the reply"))?;
    if end <= start {
        bail!("no JSON object in the reply");
    }
    Ok(serde_json::from_str(&text[start..=end])?)
}

/// Clamp the lesson list: drop empties, cap lengths and count.
fn validate_lessons(lessons: Vec<String>) -> Vec<String> {
    lessons
        .into_iter()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .filter(|l| !l.is_empty())
        .map(|l| excerpt(&l, MAX_LESSON_CHARS))
        .take(MAX_LESSONS)
        .collect()
}

/// Refuse to replace a non-empty lesson list with nothing: a lazy or
/// confused model returning `"lessons": []` must not wipe accumulated
/// memory. Deleting everything is a human decision (edit the block in the
/// file directly). Returns the list to keep and whether the guard fired.
fn guard_lesson_wipe(existing: &[String], proposed: Vec<String>) -> (Vec<String>, bool) {
    if proposed.is_empty() && !existing.is_empty() {
        (existing.to_vec(), true)
    } else {
        (proposed, false)
    }
}

/// A safe skill filename: lowercase kebab-case, or None if nothing usable
/// remains.
fn slug(name: &str) -> Option<String> {
    let cleaned: String = name
        .trim()
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let parts: Vec<&str> = cleaned.split('-').filter(|p| !p.is_empty()).collect();
    let slug = parts.join("-");
    (!slug.is_empty() && slug.len() <= 64).then_some(slug)
}

/// The lessons currently in the managed block of the given file.
fn read_lessons(path: &std::path::Path) -> Vec<String> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Some((_, block, _)) = split_lessons_block(&content) else {
        return Vec::new();
    };
    block
        .lines()
        .filter_map(|line| line.strip_prefix("- "))
        .map(|l| l.to_string())
        .collect()
}

/// Split a file into (before, block-content, after) around the managed
/// markers, if both are present in order.
fn split_lessons_block(content: &str) -> Option<(&str, &str, &str)> {
    let start = content.find(LESSONS_START)?;
    let block_start = start + LESSONS_START.len();
    let end_offset = content[block_start..].find(LESSONS_END)?;
    let block_end = block_start + end_offset;
    Some((
        &content[..start],
        &content[block_start..block_end],
        &content[block_end + LESSONS_END.len()..],
    ))
}

/// Rewrite (or append) the managed lessons block, leaving user content
/// intact. An empty lesson list writes an empty block, keeping the markers
/// so hand-placement survives.
fn replace_lessons_block(content: &str, lessons: &[String]) -> String {
    let bullets: String = lessons.iter().map(|l| format!("- {l}\n")).collect();
    let block =
        format!("{LESSONS_START}\n## Lessons (from `soa reflect`)\n\n{bullets}{LESSONS_END}");
    match split_lessons_block(content) {
        Some((before, _, after)) => format!("{before}{block}{after}"),
        None if content.trim().is_empty() => format!("{block}\n"),
        None => format!("{}\n\n{block}\n", content.trim_end()),
    }
}

fn print_lessons_diff(before: &[String], after: &[String]) {
    for lesson in after.iter().filter(|l| !before.contains(l)) {
        println!("+ {lesson}");
    }
    for lesson in before.iter().filter(|l| !after.contains(l)) {
        println!("- {lesson}");
    }
}

fn excerpt(text: &str, max_chars: usize) -> String {
    let squashed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if squashed.chars().count() <= max_chars {
        squashed
    } else {
        let cut: String = squashed.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_fenced_and_wrapped_json() {
        let bare = r#"{"lessons": ["a"], "skills": [], "note": "n"}"#;
        assert_eq!(parse_proposal(bare).unwrap().lessons, vec!["a"]);

        let fenced = format!("```json\n{bare}\n```");
        assert_eq!(parse_proposal(&fenced).unwrap().lessons, vec!["a"]);

        let chatty = format!("Here is my analysis:\n{bare}\nHope that helps!");
        let proposal = parse_proposal(&chatty).unwrap();
        assert_eq!(proposal.lessons, vec!["a"]);
        assert_eq!(proposal.note, "n");

        // Missing fields default to empty rather than erroring.
        assert!(
            parse_proposal(r#"{"lessons": []}"#)
                .unwrap()
                .skills
                .is_empty()
        );
        assert!(parse_proposal("no json here").is_err());
    }

    #[test]
    fn lessons_block_roundtrip_preserves_user_content() {
        // Appending to a file with existing content.
        let user_file = "# My project\n\nHand-written notes.\n";
        let one = replace_lessons_block(user_file, &["always run cargo check".to_string()]);
        assert!(one.starts_with("# My project\n\nHand-written notes."));
        assert!(one.contains("- always run cargo check\n"));

        // Read back and replace: user content above and below survives.
        let mut with_suffix = one.clone();
        with_suffix.push_str("\nMore notes below.\n");
        assert_eq!(
            read_lessons_str(&with_suffix),
            vec!["always run cargo check"]
        );
        let two = replace_lessons_block(
            &with_suffix,
            &[
                "prefer edit_lines".to_string(),
                "run tests before replying".to_string(),
            ],
        );
        assert!(two.contains("Hand-written notes."));
        assert!(two.contains("More notes below."));
        assert!(two.contains("- prefer edit_lines\n- run tests before replying\n"));
        assert!(!two.contains("always run cargo check"));
        // Exactly one managed block.
        assert_eq!(two.matches(LESSONS_START).count(), 1);

        // Empty file gets just the block; empty lessons keep the markers.
        let fresh = replace_lessons_block("", &["x".to_string()]);
        assert!(fresh.starts_with(LESSONS_START));
        let emptied = replace_lessons_block(&fresh, &[]);
        assert!(emptied.contains(LESSONS_START) && emptied.contains(LESSONS_END));
        assert!(!emptied.contains("- x"));
    }

    fn read_lessons_str(content: &str) -> Vec<String> {
        split_lessons_block(content)
            .map(|(_, block, _)| {
                block
                    .lines()
                    .filter_map(|line| line.strip_prefix("- "))
                    .map(|l| l.to_string())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn slugs_and_lesson_validation() {
        assert_eq!(
            slug("Careful Editing!"),
            Some("careful-editing".to_string())
        );
        assert_eq!(slug("--x--"), Some("x".to_string()));
        assert_eq!(slug("  "), None);
        assert_eq!(slug(&"y".repeat(80)), None);

        let lessons = validate_lessons(vec![
            "  spaced   out  ".to_string(),
            String::new(),
            "z".repeat(500),
        ]);
        assert_eq!(lessons[0], "spaced out");
        assert_eq!(lessons.len(), 2); // the empty one is dropped
        assert!(lessons[1].chars().count() <= MAX_LESSON_CHARS + 1);
    }

    #[test]
    fn empty_proposal_cannot_wipe_existing_lessons() {
        let existing = vec!["always run tests".to_string()];

        // An empty replacement of a non-empty list is refused...
        let (kept, refused) = guard_lesson_wipe(&existing, Vec::new());
        assert!(refused);
        assert_eq!(kept, existing);

        // ...but a real replacement, and an empty-to-empty answer, pass through.
        let (kept, refused) = guard_lesson_wipe(&existing, vec!["new rule".to_string()]);
        assert!(!refused);
        assert_eq!(kept, vec!["new rule".to_string()]);
        let (kept, refused) = guard_lesson_wipe(&[], Vec::new());
        assert!(!refused);
        assert!(kept.is_empty());
    }

    #[test]
    fn soa_added_lines_indexes_recorded_writes() {
        let entry = |tool: &str, path: &str, unified: &str| crate::diff::DiffEntry {
            tool: tool.to_string(),
            path: path.to_string(),
            unified: unified.to_string(),
            added: 0,
            removed: 0,
            before: crate::diff::Snapshot::Absent,
        };
        let mut session = crate::tui::store::Session {
            id: "s1".to_string(),
            started_at: 0,
            updated_at: 1,
            stage: "s".to_string(),
            title: "t".to_string(),
            cwd: "/p".to_string(),
            history: Vec::new(),
            transcript: Vec::new(),
            diffs: vec![
                entry(
                    "edit_file",
                    "src/a.rs",
                    "--- a/src/a.rs\n+++ b/src/a.rs\n@@\n-old_line_of_code();\n+let written_by_soa = compute_from_input(x);\n+}\n",
                ),
                entry(
                    "rewind",
                    "src/a.rs",
                    "+++ b/src/a.rs\n+let rolled_back_line = never_kept();\n",
                ),
            ],
            checkpoints: Vec::new(),
            branches: Vec::new(),
            transcript_baseline: 0,
            diff_baseline: 0,
        };
        let lines = soa_added_lines(std::slice::from_ref(&session));
        // The added substantial line is indexed to its file; the removed
        // line, the header, the brace, and the rewind entry are not.
        assert_eq!(
            lines.get("let written_by_soa = compute_from_input(x);"),
            Some(&"src/a.rs".to_string())
        );
        assert_eq!(lines.len(), 1);

        // First-writer wins when the same line appears in more sessions.
        session.diffs[0].path = "src/b.rs".to_string();
        let two = vec![
            crate::tui::store::Session {
                id: "s2".to_string(),
                ..session.clone()
            },
            session,
        ];
        let lines = soa_added_lines(&two);
        assert_eq!(
            lines
                .get("let written_by_soa = compute_from_input(x);")
                .unwrap(),
            "src/b.rs"
        );
    }

    #[test]
    fn digest_names_signals_and_caps_messages() {
        let mut history = vec![Message::User {
            content: "fix the bug".to_string(),
        }];
        for i in 0..10 {
            history.push(Message::User {
                content: format!("more {i}"),
            });
        }
        history.push(Message::Assistant {
            content: Some("done, tests pass".to_string()),
            tool_calls: None,
        });
        let session = crate::tui::store::Session {
            id: "s1".to_string(),
            started_at: 0,
            updated_at: 1,
            stage: "implement".to_string(),
            title: "fix the bug".to_string(),
            cwd: "/p".to_string(),
            history,
            transcript: Vec::new(),
            diffs: Vec::new(),
            checkpoints: Vec::new(),
            branches: Vec::new(),
            transcript_baseline: 0,
            diff_baseline: 0,
        };
        let signals = vec![Signal {
            session_id: "s1".to_string(),
            cwd: "/p".to_string(),
            at: 1,
            kind: insights::SignalKind::Denied,
            tool: "write_file".to_string(),
            commit: String::new(),
            excerpt: "DENIED: nope".to_string(),
        }];
        let digest = digest_session(&session, &signals);
        assert!(digest.contains("## Session s1"));
        assert!(digest.contains("user: fix the bug"));
        assert!(digest.contains("… 3 more user message(s)"));
        assert!(digest.contains("final reply: done, tests pass"));
        assert!(digest.contains("signal [denied] write_file: DENIED: nope"));
        // No signals reads as an explicit "none".
        assert!(digest_session(&session, &[]).contains("signals: none"));
    }
}
