//! Skills: reusable instruction packages attached to stages and agents.
//!
//! A skill is a markdown file with optional YAML-style frontmatter:
//!
//! ```markdown
//! ---
//! name: rust-review
//! description: Conventions for reviewing Rust code
//! ---
//! When reviewing Rust code, check for…
//! ```
//!
//! Lookup for `skills = ["rust-review"]` tries, in order:
//!   <skills_dir>/rust-review/SKILL.md    (directory-style, can ship assets)
//!   <skills_dir>/rust-review.md          (single file)
//! first in the project skills dir (`settings.skills_dir`, default `skills/`
//! next to the config file), then in the global dir
//! (`$XDG_DATA_HOME/soa/skills`, default `~/.local/share/soa/skills`).
//! The skill body is appended to the stage's or agent's system prompt.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use crate::config::Config;

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub body: String,
    pub path: PathBuf,
}

/// Directories searched for skills, in priority order.
pub fn skills_dirs(config: &Config) -> Vec<PathBuf> {
    let project = config.base_dir.join(
        config
            .settings
            .skills_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("skills")),
    );
    vec![project, crate::tui::store::data_dir().join("skills")]
}

/// Load one skill by name.
pub fn load_skill(config: &Config, name: &str) -> Result<Skill> {
    let dirs = skills_dirs(config);
    for dir in &dirs {
        for candidate in [dir.join(name).join("SKILL.md"), dir.join(format!("{name}.md"))] {
            if candidate.is_file() {
                let raw = std::fs::read_to_string(&candidate)
                    .with_context(|| format!("cannot read skill file {}", candidate.display()))?;
                return Ok(parse_skill(name, &raw, candidate));
            }
        }
    }
    bail!(
        "skill `{name}` not found (looked for {name}/SKILL.md and {name}.md in {})",
        dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// All discoverable skills, project dir first; project shadows global on
/// name collisions.
pub fn list_skills(config: &Config) -> Vec<Skill> {
    let mut skills: Vec<Skill> = Vec::new();
    for dir in skills_dirs(config) {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = if path.is_dir() && path.join("SKILL.md").is_file() {
                entry.file_name().to_string_lossy().into_owned()
            } else if path.extension().is_some_and(|e| e == "md") {
                path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
            } else {
                continue;
            };
            if name.is_empty() || skills.iter().any(|s| s.name == name) {
                continue;
            }
            if let Ok(skill) = load_skill(config, &name) {
                skills.push(skill);
            }
        }
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Split optional `---` frontmatter (name/description) from the body.
fn parse_skill(name: &str, raw: &str, path: PathBuf) -> Skill {
    let mut skill = Skill {
        name: name.to_string(),
        description: String::new(),
        body: raw.trim().to_string(),
        path,
    };
    let Some(rest) = raw.strip_prefix("---") else { return skill };
    let Some(end) = rest.find("\n---") else { return skill };
    let frontmatter = &rest[..end];
    skill.body = rest[end + 4..].trim().to_string();
    for line in frontmatter.lines() {
        if let Some((key, value)) = line.split_once(':') {
            match key.trim() {
                "name" => skill.name = value.trim().to_string(),
                "description" => skill.description = value.trim().to_string(),
                _ => {}
            }
        }
    }
    skill
}

/// Append the named skills to a system prompt. `owner` is used in errors.
pub fn apply_skills(
    config: &Config,
    owner: &str,
    system: Option<String>,
    skill_names: &[String],
) -> Result<Option<String>> {
    if skill_names.is_empty() {
        return Ok(system);
    }
    let mut composed = system.unwrap_or_default();
    for name in skill_names {
        let skill =
            load_skill(config, name).with_context(|| format!("`{owner}` requires skills"))?;
        if !composed.is_empty() {
            composed.push_str("\n\n");
        }
        composed.push_str(&format!("# Skill: {}\n\n{}", skill.name, skill.body));
    }
    Ok(Some(composed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_skills_dir(dir: &std::path::Path) -> Config {
        let toml_str = format!(
            r#"
            [settings]
            skills_dir = "{}"

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

    #[test]
    fn parses_frontmatter_and_body() {
        let skill = parse_skill(
            "x",
            "---\nname: renamed\ndescription: does things\n---\n\nThe body.",
            PathBuf::new(),
        );
        assert_eq!(skill.name, "renamed");
        assert_eq!(skill.description, "does things");
        assert_eq!(skill.body, "The body.");

        let bare = parse_skill("x", "Just a body.", PathBuf::new());
        assert_eq!(bare.name, "x");
        assert_eq!(bare.body, "Just a body.");
        assert!(bare.description.is_empty());
    }

    #[test]
    fn loads_and_composes_skills() {
        let dir = std::env::temp_dir().join(format!("soa-skills-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("nested")).unwrap();
        std::fs::write(dir.join("flat.md"), "---\ndescription: flat skill\n---\nFLAT BODY").unwrap();
        std::fs::write(dir.join("nested/SKILL.md"), "NESTED BODY").unwrap();
        let config = config_with_skills_dir(&dir);

        let composed = apply_skills(
            &config,
            "stage `s`",
            Some("Base prompt.".to_string()),
            &["flat".to_string(), "nested".to_string()],
        )
        .unwrap()
        .unwrap();
        assert!(composed.starts_with("Base prompt."));
        assert!(composed.contains("# Skill: flat\n\nFLAT BODY"));
        assert!(composed.contains("# Skill: nested\n\nNESTED BODY"));

        // Missing skill errors with the owner named.
        let err = apply_skills(&config, "stage `s`", None, &["ghost".to_string()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("stage `s`"), "{err}");

        let listed = list_skills(&config);
        assert_eq!(
            listed.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            vec!["flat", "nested"]
        );
        assert_eq!(listed[0].description, "flat skill");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
