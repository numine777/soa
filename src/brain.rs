//! The brain: persistent agent memory that survives across runs.
//!
//! Modeled after the SOUL.md pattern: one always-in-context file plus an
//! organized directory of reference memories, so accumulated knowledge does
//! not blow out the context window.
//!
//!  - `BRAIN.md` (in `settings.brain_dir`, default `brain/` next to the
//!    config) reaches every stage and agent system prompt. It holds
//!    hand-written core notes plus a managed index block: one line per
//!    stored memory (`name — description`), regenerated from the memory
//!    files themselves so it cannot drift.
//!  - `<name>.md` files beside it each hold one full memory behind
//!    frontmatter. Only the index line is in context; the body is fetched
//!    on demand with the `brain_read` tool.
//!
//! A stage or agent opts into the tools with `brain = true`: `brain_read`
//! retrieves a memory, `brain_write` saves or updates one (and regenerates
//! the index), `brain_forget` deletes one. Writes are confined to the brain
//! directory, carry the `filesystem_write` effect (so approvals and the
//! read-only delegation boundary see them honestly), and are plain files —
//! review what the agent learned with `git diff`, revert with
//! `git checkout`.

use std::path::PathBuf;

use serde_json::{Value, json};

use crate::config::Config;
use crate::model::ToolDefinition;

pub const INDEX_FILE: &str = "BRAIN.md";
const INDEX_START: &str = "<!-- soa:brain:index:start — managed by soa; edit outside this block -->";
const INDEX_END: &str = "<!-- soa:brain:index:end -->";

/// Bounds that keep memory files reviewable and the index compact.
const MAX_DESCRIPTION_CHARS: usize = 200;
const MAX_CONTENT_CHARS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainOp {
    Read,
    Write,
    Forget,
}

impl BrainOp {
    pub fn tool_name(self) -> &'static str {
        match self {
            BrainOp::Read => "brain_read",
            BrainOp::Write => "brain_write",
            BrainOp::Forget => "brain_forget",
        }
    }

    pub fn mutating(self) -> bool {
        matches!(self, BrainOp::Write | BrainOp::Forget)
    }
}

/// One stored memory, summarized for the index and listings.
#[derive(Debug, Clone)]
pub struct Memory {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
}

pub fn brain_dir(config: &Config) -> PathBuf {
    config.base_dir.join(
        config
            .settings
            .brain_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("brain")),
    )
}

pub fn index_path(config: &Config) -> PathBuf {
    brain_dir(config).join(INDEX_FILE)
}

/// A safe memory filename: lowercase kebab-case, or None if nothing usable
/// remains. Also the traversal guard — a slug can never leave the brain dir.
pub(crate) fn slug(name: &str) -> Option<String> {
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

/// `brain` is reserved: on case-insensitive filesystems `brain.md` would
/// collide with `BRAIN.md`.
fn memory_slug(name: &str) -> Result<String, String> {
    match slug(name) {
        Some(slug) if slug == "brain" => {
            Err("ERROR: `brain` is a reserved name (it would collide with BRAIN.md); \
                 pick a more specific one"
                .to_string())
        }
        Some(slug) => Ok(slug),
        None => Err(format!(
            "ERROR: `{}` does not reduce to a usable kebab-case name (letters, digits, \
             dashes; at most 64 characters)",
            name.trim()
        )),
    }
}

fn memory_path(config: &Config, slug: &str) -> PathBuf {
    brain_dir(config).join(format!("{slug}.md"))
}

/// All stored memories, sorted by name. A missing directory is an empty
/// brain, not an error.
pub fn list_memories(config: &Config) -> Vec<Memory> {
    let dir = brain_dir(config);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut memories: Vec<Memory> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_stem()?.to_str()?.to_string();
            if path.extension().is_none_or(|e| e != "md") || path.file_name()? == INDEX_FILE {
                return None;
            }
            let raw = std::fs::read_to_string(&path).ok()?;
            let parsed = crate::skills::parse_frontmatter(&name, &raw);
            Some(Memory {
                name: parsed.0,
                description: parsed.1,
                path,
            })
        })
        .collect();
    memories.sort_by(|a, b| a.name.cmp(&b.name));
    memories
}

/// The index block's body: one pointer line per memory.
fn index_lines(memories: &[Memory]) -> String {
    if memories.is_empty() {
        return "(no stored memories yet)\n".to_string();
    }
    memories
        .iter()
        .map(|m| {
            let description = if m.description.is_empty() {
                "(no description)"
            } else {
                &m.description
            };
            format!("- **{}** — {description}\n", m.name)
        })
        .collect()
}

/// Split a file into (before, after) around the managed index block, if
/// both markers are present in order.
fn split_index_block(content: &str) -> Option<(&str, &str)> {
    let start = content.find(INDEX_START)?;
    let end_offset = content[start..].find(INDEX_END)?;
    Some((
        &content[..start],
        &content[start + end_offset + INDEX_END.len()..],
    ))
}

/// Rewrite (or append) the managed index block, leaving content outside it
/// intact.
fn replace_index_block(content: &str, memories: &[Memory]) -> String {
    let block = format!(
        "{INDEX_START}\n## Memory index\n\n{}{INDEX_END}",
        index_lines(memories)
    );
    match split_index_block(content) {
        Some((before, after)) => format!("{before}{block}{after}"),
        None if content.trim().is_empty() => {
            format!("# Brain\n\nCore notes for this project's agents.\n\n{block}\n")
        }
        None => format!("{}\n\n{block}\n", content.trim_end()),
    }
}

/// Regenerate `BRAIN.md`'s index block from the directory's memory files.
fn regenerate_index(config: &Config) -> std::io::Result<()> {
    let path = index_path(config);
    let current = std::fs::read_to_string(&path).unwrap_or_default();
    std::fs::write(&path, replace_index_block(&current, &list_memories(config)))
}

/// The system-prompt section: BRAIN.md's non-index content plus a live
/// index generated from the directory scan — so the in-context view never
/// drifts from the files, even after hand edits or git reverts.
/// `has_tools` selects the guidance: contexts with `brain = true` are told
/// how to read and write memories; others just see what exists.
pub fn compose_section(config: &Config, has_tools: bool) -> Option<String> {
    let memories = list_memories(config);
    let index_file = std::fs::read_to_string(index_path(config)).unwrap_or_default();
    let core = match split_index_block(&index_file) {
        Some((before, after)) => format!("{}\n{}", before.trim_end(), after.trim()),
        None => index_file,
    };
    let core = core.trim();
    if memories.is_empty() && core.is_empty() && !has_tools {
        return None;
    }

    let dir = brain_dir(config);
    let mut section = String::from("# Brain (persistent memory)\n\n");
    if has_tools {
        section.push_str(
            "Lessons accumulated across this project's runs. The index below lists \
             stored memories by name — their full content is NOT in this context. \
             Before working in an area a memory covers, call `brain_read` with its \
             name. When you learn something durable and non-obvious (a correction \
             from the user, a project convention, a failure worth avoiding), save it \
             with `brain_write`: a short kebab-case name, a one-line description, and \
             concise content. Update the existing memory instead of creating a \
             near-duplicate; `brain_forget` deletes one that proved wrong.\n\n",
        );
    } else {
        let _ = {
            use std::fmt::Write as _;
            writeln!(
                section,
                "Lessons accumulated across this project's runs. The index below lists \
                 stored memories; each one's full content lives in {}/<name>.md.\n",
                dir.display()
            )
        };
    }
    if !core.is_empty() {
        section.push_str(core);
        section.push_str("\n\n");
    }
    section.push_str("## Memory index\n\n");
    section.push_str(&index_lines(&memories));
    Some(section.trim_end().to_string())
}

/// Tool definitions for a context with `brain = true`.
pub fn definitions() -> Vec<(ToolDefinition, BrainOp)> {
    let name_property = |description: &str| json!({ "type": "string", "description": description });
    vec![
        (
            ToolDefinition {
                name: BrainOp::Read.tool_name().to_string(),
                description: "Read one stored memory from the brain (persistent \
                    project memory). Use the names listed in the Memory index of \
                    your system prompt."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": name_property("The memory's name, as shown in the index")
                    },
                    "required": ["name"]
                }),
            },
            BrainOp::Read,
        ),
        (
            ToolDefinition {
                name: BrainOp::Write.tool_name().to_string(),
                description: "Save or update one memory in the brain (persistent \
                    project memory that outlives this run). Use it for durable, \
                    non-obvious lessons — corrections, conventions, failures worth \
                    avoiding — not for run-specific state. Writing an existing name \
                    replaces that memory; the index updates automatically."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": name_property("Short kebab-case name, e.g. `api-retry-policy`"),
                        "description": {
                            "type": "string",
                            "description": "One line stating the lesson — shown in every future run's index"
                        },
                        "content": {
                            "type": "string",
                            "description": "The full memory: the lesson, why it holds, and how to apply it"
                        }
                    },
                    "required": ["name", "description", "content"]
                }),
            },
            BrainOp::Write,
        ),
        (
            ToolDefinition {
                name: BrainOp::Forget.tool_name().to_string(),
                description: "Delete one stored memory from the brain. Only for a \
                    memory that is wrong or obsolete."
                    .to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "name": name_property("The memory's name, as shown in the index")
                    },
                    "required": ["name"]
                }),
            },
            BrainOp::Forget,
        ),
    ]
}

/// Files a brain call may touch, for diff capture.
pub fn affected_paths(config: &Config, op: BrainOp, arguments: &Value) -> Vec<String> {
    if !op.mutating() {
        return Vec::new();
    }
    let Some(slug) = arguments
        .get("name")
        .and_then(Value::as_str)
        .and_then(|name| memory_slug(name).ok())
    else {
        return Vec::new();
    };
    vec![
        memory_path(config, &slug).display().to_string(),
        index_path(config).display().to_string(),
    ]
}

/// A successful [`save_memory`] result.
pub(crate) struct SavedMemory {
    pub slug: String,
    pub path: PathBuf,
    /// True when an existing memory was replaced rather than created.
    pub existed: bool,
}

/// One memory file's full contents.
pub(crate) fn memory_file(name: &str, description: &str, body: &str) -> String {
    format!(
        "---\nname: {name}\ndescription: {description}\n---\n\n{}\n",
        body.trim()
    )
}

/// Validate and write one memory, regenerating the index. The single write
/// path shared by the `brain_write` tool and `soa reflect`; failures are
/// `ERROR: …` strings suitable for both models and humans.
pub(crate) fn save_memory(
    config: &Config,
    name: &str,
    description: &str,
    content: &str,
) -> Result<SavedMemory, String> {
    if name.trim().is_empty() {
        return Err("ERROR: a memory needs a non-empty `name`".to_string());
    }
    let slug = memory_slug(name)?;
    let description = one_line(description);
    let content = content.trim();
    if description.is_empty() {
        return Err("ERROR: a memory needs a one-line `description`".to_string());
    }
    if content.is_empty() {
        return Err("ERROR: a memory needs non-empty `content`".to_string());
    }
    if content.chars().count() > MAX_CONTENT_CHARS {
        return Err(format!(
            "ERROR: memory content is too large ({} characters, max \
             {MAX_CONTENT_CHARS}); distill it or split it into several memories",
            content.chars().count()
        ));
    }
    let path = memory_path(config, &slug);
    let existed = path.exists();
    std::fs::create_dir_all(brain_dir(config))
        .and_then(|()| std::fs::write(&path, memory_file(&slug, &description, content)))
        .map_err(|e| format!("ERROR: cannot write {}: {e}", path.display()))?;
    regenerate_index(config)
        .map_err(|e| format!("ERROR: memory saved but the index update failed: {e}"))?;
    Ok(SavedMemory {
        slug,
        path,
        existed,
    })
}

/// Execute a brain tool call. All failures are `ERROR: …` strings so the
/// model can react without killing the stage.
pub fn dispatch(config: &Config, op: BrainOp, arguments: &Value) -> String {
    let name = arguments
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if name.trim().is_empty() {
        return format!("ERROR: {} requires a non-empty `name` string", op.tool_name());
    }
    let slug = match memory_slug(name) {
        Ok(slug) => slug,
        Err(error) => return error,
    };
    let path = memory_path(config, &slug);

    match op {
        BrainOp::Read => match std::fs::read_to_string(&path) {
            Ok(raw) => {
                let (name, description, body) = crate::skills::parse_frontmatter(&slug, &raw);
                let header = if description.is_empty() {
                    format!("## {name}")
                } else {
                    format!("## {name} — {description}")
                };
                format!("{header}\n\n{body}")
            }
            Err(_) => format!(
                "ERROR: no memory named `{slug}`. The Memory index in your system \
                 prompt lists what exists."
            ),
        },
        BrainOp::Write => {
            let description = arguments
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let content = arguments
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default();
            match save_memory(config, name, description, content) {
                Ok(saved) => format!(
                    "{} memory `{}` ({}); the index is updated",
                    if saved.existed { "updated" } else { "saved" },
                    saved.slug,
                    saved.path.display()
                ),
                Err(error) => error,
            }
        }
        BrainOp::Forget => {
            if !path.is_file() {
                return format!("ERROR: no memory named `{slug}` to forget");
            }
            if let Err(e) = std::fs::remove_file(&path) {
                return format!("ERROR: cannot delete {}: {e}", path.display());
            }
            if let Err(e) = regenerate_index(config) {
                return format!("ERROR: memory deleted but the index update failed: {e}");
            }
            format!("forgot memory `{slug}`; the index is updated")
        }
    }
}

/// Squash to one bounded line for frontmatter and the index.
fn one_line(text: &str) -> String {
    crate::reflect::excerpt(text, MAX_DESCRIPTION_CHARS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn config_with_brain_dir(dir: &Path) -> Config {
        let toml_str = format!(
            r#"
            [settings]
            brain_dir = "{}"

            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [[stage]]
            name = "s"
            model = "m"
            "#,
            dir.display()
        );
        let mut config: Config = toml::from_str(&toml_str).unwrap();
        config.base_dir = PathBuf::from("/");
        config
    }

    fn temp_config(label: &str) -> (Config, PathBuf) {
        let dir = std::env::temp_dir().join(format!("soa-brain-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        (config_with_brain_dir(&dir), dir)
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("API Retry policy!"), Some("api-retry-policy".to_string()));
        assert_eq!(slug("--x--"), Some("x".to_string()));
        assert_eq!(slug("  "), None);
        assert_eq!(slug(&"y".repeat(80)), None);
        assert!(memory_slug("BRAIN").is_err());
        assert!(memory_slug("brain-md").is_ok());
    }

    #[test]
    fn write_read_list_forget_roundtrip() {
        let (config, dir) = temp_config("roundtrip");

        let saved = dispatch(
            &config,
            BrainOp::Write,
            &json!({
                "name": "API Retry Policy",
                "description": "429s honor Retry-After",
                "content": "The gateway returns Retry-After on 429.\nHonor it."
            }),
        );
        assert!(saved.starts_with("saved memory `api-retry-policy`"), "{saved}");

        // The memory file has frontmatter; the index block lists it.
        let index = std::fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        assert!(index.contains(INDEX_START) && index.contains(INDEX_END), "{index}");
        assert!(index.contains("- **api-retry-policy** — 429s honor Retry-After"), "{index}");

        let read = dispatch(&config, BrainOp::Read, &json!({"name": "api-retry-policy"}));
        assert!(read.starts_with("## api-retry-policy — 429s honor Retry-After"), "{read}");
        assert!(read.contains("Honor it."), "{read}");

        // Overwriting reports an update and keeps a single index line.
        let updated = dispatch(
            &config,
            BrainOp::Write,
            &json!({"name": "api-retry-policy", "description": "updated", "content": "new"}),
        );
        assert!(updated.starts_with("updated memory"), "{updated}");
        let memories = list_memories(&config);
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].description, "updated");

        let forgotten = dispatch(&config, BrainOp::Forget, &json!({"name": "api-retry-policy"}));
        assert!(forgotten.starts_with("forgot memory"), "{forgotten}");
        assert!(list_memories(&config).is_empty());
        assert!(
            std::fs::read_to_string(dir.join(INDEX_FILE))
                .unwrap()
                .contains("(no stored memories yet)")
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dispatch_validates_and_reports_errors() {
        let (config, dir) = temp_config("errors");
        for (op, args, want) in [
            (BrainOp::Read, json!({}), "non-empty `name`"),
            (BrainOp::Read, json!({"name": "ghost"}), "no memory named `ghost`"),
            (BrainOp::Write, json!({"name": "x", "content": "c"}), "`description`"),
            (BrainOp::Write, json!({"name": "x", "description": "d"}), "`content`"),
            (BrainOp::Write, json!({"name": "brain", "description": "d", "content": "c"}), "reserved"),
            (BrainOp::Write, json!({"name": "../up", "description": "d", "content": "c"}), "saved"),
            (BrainOp::Forget, json!({"name": "ghost"}), "no memory named"),
        ] {
            let out = dispatch(&config, op, &args);
            assert!(out.contains(want), "{op:?} {args} → {out}");
        }
        // The traversal attempt above was slugged into the brain dir, not
        // written outside it.
        assert!(dir.join("up.md").is_file());
        assert!(!dir.parent().unwrap().join("up.md").exists());

        let oversized = dispatch(
            &config,
            BrainOp::Write,
            &json!({"name": "big", "description": "d", "content": "z".repeat(MAX_CONTENT_CHARS + 1)}),
        );
        assert!(oversized.contains("too large"), "{oversized}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn index_block_preserves_hand_written_content() {
        let (config, dir) = temp_config("handwritten");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(INDEX_FILE),
            "# Brain\n\nCore rule: prefer small diffs.\n",
        )
        .unwrap();
        dispatch(
            &config,
            BrainOp::Write,
            &json!({"name": "one", "description": "first", "content": "body"}),
        );
        let index = std::fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        assert!(index.starts_with("# Brain\n\nCore rule: prefer small diffs."), "{index}");
        assert!(index.contains("- **one** — first"), "{index}");
        // A second write replaces the block instead of appending another.
        dispatch(
            &config,
            BrainOp::Write,
            &json!({"name": "two", "description": "second", "content": "body"}),
        );
        let index = std::fs::read_to_string(dir.join(INDEX_FILE)).unwrap();
        assert_eq!(index.matches(INDEX_START).count(), 1);
        assert!(index.contains("- **one**") && index.contains("- **two**"), "{index}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compose_section_tracks_files_and_tools() {
        let (config, dir) = temp_config("compose");

        // Nothing on disk: only tool-bearing contexts get a section (it
        // tells them how to start saving memories).
        assert!(compose_section(&config, false).is_none());
        let fresh = compose_section(&config, true).unwrap();
        assert!(fresh.contains("brain_write"), "{fresh}");
        assert!(fresh.contains("(no stored memories yet)"), "{fresh}");

        dispatch(
            &config,
            BrainOp::Write,
            &json!({"name": "one", "description": "first lesson", "content": "body"}),
        );
        std::fs::write(
            dir.join(INDEX_FILE),
            std::fs::read_to_string(dir.join(INDEX_FILE))
                .unwrap()
                .replace("Core notes for this project's agents.", "Always run cargo check."),
        )
        .unwrap();

        // With tools: guidance, core content, and the index line.
        let with_tools = compose_section(&config, true).unwrap();
        assert!(with_tools.starts_with("# Brain (persistent memory)"), "{with_tools}");
        assert!(with_tools.contains("Always run cargo check."), "{with_tools}");
        assert!(with_tools.contains("- **one** — first lesson"), "{with_tools}");
        assert!(with_tools.contains("brain_read"), "{with_tools}");

        // Without tools: the index still reaches the prompt, pointing at
        // the files instead of the tools.
        let without = compose_section(&config, false).unwrap();
        assert!(without.contains("- **one** — first lesson"), "{without}");
        assert!(!without.contains("brain_write"), "{without}");
        assert!(without.contains("<name>.md"), "{without}");

        // The live index wins over a stale file: hand-delete the memory
        // and the section reflects the directory, not the old block.
        std::fs::remove_file(dir.join("one.md")).unwrap();
        let stale = compose_section(&config, false).unwrap();
        assert!(stale.contains("(no stored memories yet)"), "{stale}");
        assert!(!stale.contains("- **one**"), "{stale}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
