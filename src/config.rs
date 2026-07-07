//! Configuration schema and validation for the `soa.toml` file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub settings: Settings,
    #[serde(default)]
    pub providers: BTreeMap<String, Provider>,
    #[serde(default)]
    pub models: BTreeMap<String, Model>,
    #[serde(default)]
    pub mcp: BTreeMap<String, McpServer>,
    #[serde(default, rename = "stage")]
    pub stages: Vec<Stage>,

    /// Directory containing the config file; relative paths (e.g.
    /// `system_prompt_file`) resolve against it. Set by [`Config::load`].
    #[serde(skip)]
    pub base_dir: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Base URL of a SearXNG instance; required if any stage sets `web_search = true`.
    pub searxng_url: Option<String>,
    /// Maximum results returned by the `web_search` tool.
    #[serde(default = "default_search_results")]
    pub searxng_max_results: usize,
    /// Default cap on model round-trips per stage; stages can override with `max_turns`.
    #[serde(default = "default_max_turns")]
    pub default_max_turns: u32,
    /// HTTP timeout for provider requests, in seconds. Local models can be slow.
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
}

// A hand-written Default keeps the "[settings] table absent" case in sync
// with the per-field serde defaults.
impl Default for Settings {
    fn default() -> Self {
        Settings {
            searxng_url: None,
            searxng_max_results: default_search_results(),
            default_max_turns: default_max_turns(),
            request_timeout_secs: default_timeout(),
        }
    }
}

fn default_search_results() -> usize {
    8
}
fn default_max_turns() -> u32 {
    16
}
fn default_timeout() -> u64 {
    600
}

/// An OpenAI-compatible chat-completions endpoint (Ollama, LM Studio,
/// llama.cpp, vLLM, or a hosted API).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provider {
    /// e.g. "http://localhost:11434/v1"
    pub base_url: String,
    /// Optional bearer token. Supports `${ENV_VAR}` expansion.
    pub api_key: Option<String>,
}

/// A named model: a provider reference plus default sampling parameters.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    /// Key into `[providers]`.
    pub provider: String,
    /// Model id as the provider knows it, e.g. "qwen3:32b".
    pub model: String,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case", deny_unknown_fields)]
pub enum McpServer {
    /// Spawn a local process speaking MCP over stdio.
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        /// Extra environment variables; values support `${ENV_VAR}` expansion.
        #[serde(default)]
        env: BTreeMap<String, String>,
        /// Tool names to treat as read-only even if the server does not
        /// annotate them with `readOnlyHint`.
        #[serde(default)]
        readonly_tools: Vec<String>,
    },
    /// Connect to a streamable-HTTP MCP endpoint.
    Http {
        url: String,
        /// Bearer token (without the `Bearer ` prefix). Supports `${ENV_VAR}`.
        auth_token: Option<String>,
        /// Extra headers sent with every request; values support `${ENV_VAR}`.
        #[serde(default)]
        headers: BTreeMap<String, String>,
        #[serde(default)]
        readonly_tools: Vec<String>,
    },
}

impl McpServer {
    pub fn readonly_tools(&self) -> &[String] {
        match self {
            McpServer::Stdio { readonly_tools, .. } => readonly_tools,
            McpServer::Http { readonly_tools, .. } => readonly_tools,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Only tools that are known read-only are exposed to the model.
    #[default]
    #[serde(alias = "ro")]
    ReadOnly,
    /// All tools from the stage's MCP servers are exposed.
    #[serde(alias = "rw")]
    ReadWrite,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stage {
    pub name: String,
    /// Key into `[models]`.
    pub model: String,
    /// Defaults to `read_only`.
    #[serde(default)]
    pub mode: Mode,
    /// Keys into `[mcp]`; these servers' tools are available to this stage.
    #[serde(default)]
    pub mcp: Vec<String>,
    /// Expose the SearXNG `web_search` tool to this stage.
    #[serde(default)]
    pub web_search: bool,
    /// Inline system prompt. Mutually exclusive with `system_prompt_file`.
    pub system_prompt: Option<String>,
    /// Path to a file containing the system prompt, relative to the config file.
    pub system_prompt_file: Option<PathBuf>,
    /// User-message template. Placeholders: `{{input}}` (the original task),
    /// `{{previous}}` (output of the previous stage), `{{stage.<name>}}`
    /// (output of any earlier stage). Defaults to `{{input}}` for the first
    /// stage; later stages get the task plus the previous stage's output.
    pub prompt: Option<String>,
    pub max_turns: Option<u32>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

impl Stage {
    /// Resolve the system prompt, reading `system_prompt_file` if set.
    pub fn resolve_system_prompt(&self, base_dir: &Path) -> Result<Option<String>> {
        match (&self.system_prompt, &self.system_prompt_file) {
            (Some(inline), None) => Ok(Some(inline.clone())),
            (None, Some(file)) => {
                let path = base_dir.join(file);
                let text = std::fs::read_to_string(&path).with_context(|| {
                    format!(
                        "stage `{}`: cannot read system_prompt_file {}",
                        self.name,
                        path.display()
                    )
                })?;
                Ok(Some(text))
            }
            (None, None) => Ok(None),
            (Some(_), Some(_)) => unreachable!("rejected during validation"),
        }
    }

    pub fn prompt_template(&self, is_first: bool) -> String {
        match &self.prompt {
            Some(p) => p.clone(),
            None if is_first => "{{input}}".to_string(),
            None => "# Task\n{{input}}\n\n# Input from previous stage\n{{previous}}".to_string(),
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Config> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read config file {}", path.display()))?;
        let mut config: Config = toml::from_str(&raw)
            .with_context(|| format!("invalid config in {}", path.display()))?;
        config.base_dir = path
            .canonicalize()
            .with_context(|| format!("cannot resolve path {}", path.display()))?
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        config.expand_env()?;
        config.validate()?;
        Ok(config)
    }

    /// Expand `${VAR}` references in fields that plausibly hold secrets or
    /// machine-specific values. Prompt text is intentionally left untouched.
    fn expand_env(&mut self) -> Result<()> {
        if let Some(url) = &self.settings.searxng_url {
            self.settings.searxng_url = Some(expand_env(url)?);
        }
        for provider in self.providers.values_mut() {
            provider.base_url = expand_env(&provider.base_url)?;
            if let Some(key) = &provider.api_key {
                provider.api_key = Some(expand_env(key)?);
            }
        }
        for server in self.mcp.values_mut() {
            match server {
                McpServer::Stdio { env, .. } => {
                    for value in env.values_mut() {
                        *value = expand_env(value)?;
                    }
                }
                McpServer::Http {
                    url,
                    auth_token,
                    headers,
                    ..
                } => {
                    *url = expand_env(url)?;
                    if let Some(token) = auth_token {
                        *auth_token = Some(expand_env(token)?);
                    }
                    for value in headers.values_mut() {
                        *value = expand_env(value)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        let mut errors: Vec<String> = Vec::new();

        for (name, model) in &self.models {
            if !self.providers.contains_key(&model.provider) {
                errors.push(format!(
                    "model `{name}` references unknown provider `{}`",
                    model.provider
                ));
            }
        }

        if self.stages.is_empty() {
            errors.push("no [[stage]] entries defined".to_string());
        }

        let mut seen_stage_names: Vec<&str> = Vec::new();
        for (index, stage) in self.stages.iter().enumerate() {
            let name = &stage.name;
            if seen_stage_names.contains(&name.as_str()) {
                errors.push(format!("duplicate stage name `{name}`"));
            }

            if !self.models.contains_key(&stage.model) {
                errors.push(format!(
                    "stage `{name}` references unknown model `{}`",
                    stage.model
                ));
            }
            for server in &stage.mcp {
                if !self.mcp.contains_key(server) {
                    errors.push(format!(
                        "stage `{name}` references unknown mcp server `{server}`"
                    ));
                }
            }
            if stage.system_prompt.is_some() && stage.system_prompt_file.is_some() {
                errors.push(format!(
                    "stage `{name}` sets both system_prompt and system_prompt_file"
                ));
            }
            if stage.web_search && self.settings.searxng_url.is_none() {
                errors.push(format!(
                    "stage `{name}` enables web_search but settings.searxng_url is not set"
                ));
            }

            // Template references must point at the task or earlier stages.
            let template = stage.prompt_template(index == 0);
            for var in template_vars(&template) {
                match var.as_str() {
                    "input" => {}
                    "previous" if index > 0 => {}
                    "previous" => errors.push(format!(
                        "stage `{name}` uses {{{{previous}}}} but is the first stage"
                    )),
                    other => match other.strip_prefix("stage.") {
                        Some(referenced) if seen_stage_names.contains(&referenced) => {}
                        Some(referenced) => errors.push(format!(
                            "stage `{name}` references {{{{stage.{referenced}}}}} which is not an earlier stage"
                        )),
                        None => errors.push(format!(
                            "stage `{name}` uses unknown template variable {{{{{other}}}}}"
                        )),
                    },
                }
            }

            seen_stage_names.push(name);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            bail!("invalid configuration:\n  - {}", errors.join("\n  - "))
        }
    }
}

/// Replace `${VAR}` with the value of the environment variable `VAR`.
/// A reference to an unset variable is an error.
pub fn expand_env(input: &str) -> Result<String> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("${") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find('}') else {
            bail!("unterminated ${{...}} reference in `{input}`");
        };
        let var = &after[..end];
        let value = std::env::var(var)
            .with_context(|| format!("environment variable `{var}` referenced in config is not set"))?;
        out.push_str(&value);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Extract `{{var}}` placeholder names from a template.
pub fn template_vars(template: &str) -> Vec<String> {
    let mut vars = Vec::new();
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else { break };
        vars.push(after[..end].trim().to_string());
        rest = &after[end + 2..];
    }
    vars
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml_str: &str) -> Result<Config> {
        let mut config: Config = toml::from_str(toml_str)?;
        config.expand_env()?;
        config.validate()?;
        Ok(config)
    }

    const MINIMAL: &str = r#"
        [providers.local]
        base_url = "http://localhost:11434/v1"

        [models.default]
        provider = "local"
        model = "qwen3:8b"

        [[stage]]
        name = "answer"
        model = "default"
    "#;

    #[test]
    fn minimal_config_parses() {
        let config = parse(MINIMAL).unwrap();
        assert_eq!(config.stages.len(), 1);
        assert_eq!(config.stages[0].mode, Mode::ReadOnly);
        assert_eq!(config.settings.default_max_turns, 16);
    }

    #[test]
    fn example_config_parses() {
        let raw = include_str!("../soa.toml");
        let config: Config = toml::from_str(raw).unwrap();
        config.validate().unwrap();
        assert!(config.stages.len() >= 2);
    }

    #[test]
    fn unknown_model_rejected() {
        let err = parse(&MINIMAL.replace("model = \"default\"\n", "model = \"nope\"\n"))
            .unwrap_err()
            .to_string();
        // Only the stage reference changes; the [models.default] table keeps its own model id.
        assert!(err.contains("unknown model"), "{err}");
    }

    #[test]
    fn web_search_requires_searxng_url() {
        let toml_str = MINIMAL.replace("name = \"answer\"", "name = \"answer\"\nweb_search = true");
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("searxng_url"), "{err}");
    }

    #[test]
    fn previous_in_first_stage_rejected() {
        let toml_str = MINIMAL.replace(
            "name = \"answer\"",
            "name = \"answer\"\nprompt = \"{{previous}}\"",
        );
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("first stage"), "{err}");
    }

    #[test]
    fn forward_stage_reference_rejected() {
        let toml_str = format!(
            "{MINIMAL}\n[[stage]]\nname = \"second\"\nmodel = \"default\"\nprompt = \"{{{{stage.third}}}}\"\n"
        );
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("stage.third"), "{err}");
    }

    #[test]
    fn env_expansion() {
        // SAFETY: test-only; no other thread reads this variable.
        unsafe { std::env::set_var("SOA_TEST_TOKEN", "sekrit") };
        assert_eq!(expand_env("Bearer ${SOA_TEST_TOKEN}!").unwrap(), "Bearer sekrit!");
        assert!(expand_env("${SOA_DOES_NOT_EXIST_XYZ}").is_err());
        assert_eq!(expand_env("plain").unwrap(), "plain");
    }

    #[test]
    fn template_var_extraction() {
        assert_eq!(
            template_vars("a {{input}} b {{ stage.plan }} c"),
            vec!["input".to_string(), "stage.plan".to_string()]
        );
    }
}
