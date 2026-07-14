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
    /// Subagents that stages (and other agents) can delegate tasks to.
    #[serde(default)]
    pub agents: BTreeMap<String, Agent>,
    /// Named pipelines: ordered lists of stage names. When present,
    /// `soa run --workflow <name>` picks one; otherwise the [[stage]]
    /// declaration order is the pipeline.
    #[serde(default)]
    pub workflows: BTreeMap<String, Workflow>,
    #[serde(default, rename = "stage")]
    pub stages: Vec<Stage>,
    /// Shell commands bound to tool-call events (see [`Hook`]).
    #[serde(default)]
    pub hooks: Vec<Hook>,

    /// Directory containing the config file; relative paths (e.g.
    /// `system_prompt_file`) resolve against it. Set by [`Config::load`].
    #[serde(skip)]
    pub base_dir: PathBuf,

    /// Project instructions (`settings.context_files`, default `SOA.md`)
    /// appended to every stage and agent system prompt, in order. Set by
    /// [`Config::load`].
    #[serde(skip)]
    pub project_contexts: Vec<ProjectContext>,
}

/// One project-instructions file discovered near the working directory.
#[derive(Debug, Clone)]
pub struct ProjectContext {
    pub path: PathBuf,
    pub content: String,
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
    /// Total stage executions allowed in one `soa run`, including re-runs
    /// caused by `reprompt_stage`. Guards against reprompt loops.
    #[serde(default = "default_max_stage_runs")]
    pub max_stage_runs: u32,
    /// Tool results longer than this many characters are truncated before
    /// they enter the conversation, so one oversized result (e.g. a
    /// recursive directory tree) cannot blow the model's context window.
    /// 0 disables truncation.
    #[serde(default = "default_max_tool_output_chars")]
    pub max_tool_output_chars: usize,
    /// How deep agent delegation may nest: at 1 stages can spawn agents but
    /// agents cannot spawn agents; at 2 agents get one more level; and so
    /// on. Guards against delegation loops.
    #[serde(default = "default_max_agent_depth")]
    pub max_agent_depth: u32,
    /// When real usage exceeds this fraction of a model's `context_tokens`,
    /// chat auto-compacts and stage loops truncate older tool results.
    /// 0 disables. Only applies to models that declare `context_tokens`.
    #[serde(default = "default_auto_compact_threshold")]
    pub auto_compact_threshold: f64,
    /// Dispatch a round's tool calls concurrently when every call in it is
    /// read-only. Rounds containing writes, approvals, or control tools
    /// always run sequentially regardless.
    #[serde(default = "default_parallel_tools")]
    pub parallel_tools: bool,
    /// How many times a failed provider request is retried (with
    /// exponential backoff) before the turn errors. Covers network
    /// failures, 408/429/5xx responses, and interrupted streams.
    /// 0 disables retries.
    #[serde(default = "default_provider_retries")]
    pub provider_retries: u32,
    /// HTTP timeout for provider requests, in seconds. Local models can be slow.
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
    /// Shell commands run by the built-in `shell` tool are killed after
    /// this many seconds.
    #[serde(default = "default_shell_timeout")]
    pub shell_timeout_secs: u64,
    /// Directory holding skills, relative to the config file (default
    /// `skills/`). The global `~/.local/share/soa/skills` is also searched.
    pub skills_dir: Option<PathBuf>,
    /// Workflow used by `soa run` when none is passed. Falls back to a
    /// workflow literally named `default`, then to the [[stage]] order.
    pub default_workflow: Option<String>,
    /// Project-instruction files appended to every stage and agent system
    /// prompt. Each entry is discovered by walking up from the working
    /// directory (absolute paths are read directly); the files that exist
    /// are sourced in the listed order, missing ones are skipped.
    #[serde(default = "default_context_files")]
    pub context_files: Vec<PathBuf>,
    /// Model used by `soa reflect` to distill sessions into lessons and
    /// skills (default: the first stage's model).
    pub reflect_model: Option<String>,
}

/// A named pipeline over the stage library.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workflow {
    #[serde(default)]
    pub description: String,
    /// Stage names in execution order (each stage may appear once).
    pub stages: Vec<String>,
}

// A hand-written Default keeps the "[settings] table absent" case in sync
// with the per-field serde defaults.
impl Default for Settings {
    fn default() -> Self {
        Settings {
            searxng_url: None,
            searxng_max_results: default_search_results(),
            default_max_turns: default_max_turns(),
            max_stage_runs: default_max_stage_runs(),
            max_tool_output_chars: default_max_tool_output_chars(),
            max_agent_depth: default_max_agent_depth(),
            auto_compact_threshold: default_auto_compact_threshold(),
            parallel_tools: default_parallel_tools(),
            provider_retries: default_provider_retries(),
            request_timeout_secs: default_timeout(),
            shell_timeout_secs: default_shell_timeout(),
            skills_dir: None,
            default_workflow: None,
            context_files: default_context_files(),
            reflect_model: None,
        }
    }
}

fn default_context_files() -> Vec<PathBuf> {
    vec![PathBuf::from("SOA.md")]
}

fn default_shell_timeout() -> u64 {
    120
}

fn default_search_results() -> usize {
    8
}
fn default_max_turns() -> u32 {
    16
}
fn default_max_stage_runs() -> u32 {
    24
}
fn default_max_tool_output_chars() -> usize {
    30_000
}
fn default_max_agent_depth() -> u32 {
    2
}
fn default_auto_compact_threshold() -> f64 {
    0.8
}
fn default_provider_retries() -> u32 {
    3
}
fn default_parallel_tools() -> bool {
    true
}
fn default_timeout() -> u64 {
    600
}

/// A user-configured shell command bound to a tool-call event.
///
/// `pre_tool` hooks run before a call is dispatched (before any approval
/// prompt); a non-zero exit blocks the call and the hook's output is fed
/// to the model. `post_tool` hooks run after; a non-zero exit appends the
/// hook's output to the tool result as feedback (e.g. lint errors).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hook {
    pub event: HookEvent,
    /// `*`-wildcard pattern matched against the same call descriptor that
    /// approvals use: `edit_file *`, `shell cargo *`, `fs__write_file`,
    /// `agent__researcher`. Default: every call.
    #[serde(rename = "match", default = "default_hook_match")]
    pub pattern: String,
    /// Command run with `sh -c` in the working directory. Receives a JSON
    /// payload on stdin and SOA_EVENT / SOA_TOOL / SOA_DESCRIPTOR /
    /// SOA_PATHS in the environment.
    pub command: String,
    /// Kill the hook after this many seconds (default:
    /// `settings.shell_timeout_secs`).
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    PreTool,
    PostTool,
}

fn default_hook_match() -> String {
    "*".to_string()
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
    /// Stream responses token-by-token over SSE. Disable for servers that
    /// don't support `"stream": true`.
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Whether requests to this endpoint leave the user's trusted local
    /// boundary. This label is used to prevent an implicit local-to-external
    /// fallback unless the model explicitly opts in.
    #[serde(default)]
    pub data_boundary: DataBoundary,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataBoundary {
    #[default]
    Local,
    External,
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
    /// The model's context window, in tokens. Enables the pressure gauge
    /// in the chat status bar, auto-compaction, and mid-stage shedding of
    /// old tool results.
    pub context_tokens: Option<u64>,
    /// Other `[models]` entries to fail over to, in order, when this
    /// model's endpoint stays down after retries (or rejects the request
    /// outright). Fallbacks may declare their own fallbacks; the chain is
    /// followed breadth-first and cycles are ignored.
    #[serde(default)]
    pub fallback: Vec<String>,
    /// Permit this model's fallback chain to cross from a local provider to
    /// an external provider. Without this explicit consent, validation
    /// rejects the cross-boundary edge before any request can be sent.
    #[serde(default)]
    pub allow_external_fallback: bool,
}

/// Observable effects a tool may have. Contexts use these labels to add
/// approval gates beyond the default mutation/process policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolEffect {
    FilesystemRead,
    FilesystemWrite,
    ProcessExecute,
    NetworkEgress,
    ExternalRead,
    ExternalMutation,
}

#[derive(Debug, Clone, Deserialize)]
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

/// A subagent: a model with its own system prompt, mode, and tools that a
/// stage (or another agent) can delegate a self-contained task to. Each
/// agent listed in a `subagents` field is exposed to that context's model
/// as a tool named `agent__<name>`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Agent {
    /// Shown to the calling model as the tool description — say what this
    /// agent is good at so the model knows when to delegate.
    #[serde(default)]
    pub description: String,
    /// Key into `[models]`.
    pub model: String,
    #[serde(default)]
    pub mode: Mode,
    #[serde(default)]
    pub mcp: Vec<String>,
    #[serde(default)]
    pub web_search: bool,
    pub system_prompt: Option<String>,
    pub system_prompt_file: Option<PathBuf>,
    pub max_turns: Option<u32>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    /// Agents this agent may itself delegate to (subject to
    /// `settings.max_agent_depth`).
    #[serde(default)]
    pub subagents: Vec<String>,
    /// Skills appended to this agent's system prompt.
    #[serde(default)]
    pub skills: Vec<String>,
    /// Offer the built-in `shell` tool (explicit grant, independent of mode).
    #[serde(default)]
    pub shell: bool,
    /// Restrict shell commands to these `*`-wildcard patterns.
    #[serde(default)]
    pub shell_allow: Vec<String>,
    /// Offer the built-in file tools, rooted at the working directory
    /// (write tools only in read_write mode).
    #[serde(default)]
    pub files: bool,
    /// Pause mutation/process effects, plus `approval_effects`, for
    /// interactive approval.
    #[serde(default)]
    pub require_approval: bool,
    /// Additional effects to gate when `require_approval = true`. Mutating
    /// and process-execution effects are always gated by default; this list
    /// is primarily useful for read-like effects such as `network_egress`.
    #[serde(default)]
    pub approval_effects: Vec<ToolEffect>,
    /// Calls matching these patterns skip the approval prompt.
    #[serde(default)]
    pub auto_approve: Vec<String>,
}

impl Agent {
    pub fn resolve_system_prompt(&self, base_dir: &Path) -> Result<Option<String>> {
        resolve_prompt_source(&self.system_prompt, &self.system_prompt_file, base_dir)
    }
}

/// Resolve an inline-or-file prompt pair (mutual exclusivity is enforced
/// during validation).
fn resolve_prompt_source(
    inline: &Option<String>,
    file: &Option<PathBuf>,
    base_dir: &Path,
) -> Result<Option<String>> {
    match (inline, file) {
        (Some(inline), None) => Ok(Some(inline.clone())),
        (None, Some(file)) => {
            let path = base_dir.join(file);
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("cannot read system_prompt_file {}", path.display()))?;
            Ok(Some(text))
        }
        (None, None) => Ok(None),
        (Some(_), Some(_)) => unreachable!("rejected during validation"),
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
    /// Stages this stage may hand control back to via the `reprompt_stage`
    /// tool (may include itself). The pipeline resumes from the target stage
    /// and continues in declared order, so a reviewer that reprompts an
    /// earlier `implement` stage will run again afterwards.
    #[serde(default)]
    pub can_reprompt: Vec<String>,
    /// Keys into `[agents]`: subagents this stage's model may delegate to.
    #[serde(default)]
    pub subagents: Vec<String>,
    /// Skills appended to this stage's system prompt (see the skills dir).
    #[serde(default)]
    pub skills: Vec<String>,
    /// Offer the built-in `shell` tool. This is an explicit grant,
    /// independent of `mode` — a read_only review stage can run tests
    /// without gaining file-write tools.
    #[serde(default)]
    pub shell: bool,
    /// Restrict shell commands to these `*`-wildcard patterns
    /// (e.g. `["cargo *", "git status"]`). Empty = unrestricted.
    #[serde(default)]
    pub shell_allow: Vec<String>,
    /// Offer the built-in file tools (read_file, list_dir, glob, grep, and
    /// — in read_write mode — write_file and edit_file), rooted at the
    /// working directory.
    #[serde(default)]
    pub files: bool,
    /// Pause mutation/process effects, plus `approval_effects`, for
    /// interactive approval (y/n/always).
    /// Without an interactive approver (piped runs), gated calls are denied.
    #[serde(default)]
    pub require_approval: bool,
    /// Additional effects to gate when `require_approval = true`. Mutating
    /// and process-execution effects remain gated regardless of this list.
    #[serde(default)]
    pub approval_effects: Vec<ToolEffect>,
    /// Calls matching these `*`-wildcard patterns skip the approval prompt.
    /// Patterns match tool names (`filesystem__edit_file`, `agent__coder`)
    /// or, for the shell tool, `shell <command>` (`shell cargo *`).
    #[serde(default)]
    pub auto_approve: Vec<String>,
}

impl Stage {
    /// Resolve the system prompt, reading `system_prompt_file` if set.
    pub fn resolve_system_prompt(&self, base_dir: &Path) -> Result<Option<String>> {
        resolve_prompt_source(&self.system_prompt, &self.system_prompt_file, base_dir)
            .with_context(|| format!("stage `{}`", self.name))
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
        let cwd = std::env::current_dir().unwrap_or_default();
        config.project_contexts = resolve_context_files(&config.settings.context_files, &cwd);
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

        for (index, hook) in self.hooks.iter().enumerate() {
            if hook.command.trim().is_empty() {
                errors.push(format!("hooks[{index}] has an empty command"));
            }
        }

        for (name, model) in &self.models {
            for fallback in &model.fallback {
                if fallback == name {
                    errors.push(format!("model `{name}` lists itself as a fallback"));
                } else if !self.models.contains_key(fallback) {
                    errors.push(format!(
                        "model `{name}` has unknown fallback model `{fallback}`"
                    ));
                } else {
                    let source_boundary = self
                        .providers
                        .get(&model.provider)
                        .map(|provider| provider.data_boundary);
                    let fallback_boundary = self
                        .models
                        .get(fallback)
                        .and_then(|fallback_model| self.providers.get(&fallback_model.provider))
                        .map(|provider| provider.data_boundary);
                    if source_boundary == Some(DataBoundary::Local)
                        && fallback_boundary == Some(DataBoundary::External)
                        && !model.allow_external_fallback
                    {
                        errors.push(format!(
                            "model `{name}` falls back from local data boundary to external \
                             model `{fallback}`; set allow_external_fallback = true on \
                             model `{name}` to consent"
                        ));
                    }
                }
            }
        }

        if !(0.0..=1.0).contains(&self.settings.auto_compact_threshold) {
            errors.push(format!(
                "settings.auto_compact_threshold must be between 0 and 1 (got {})",
                self.settings.auto_compact_threshold
            ));
        }

        for (name, agent) in &self.agents {
            if !self.models.contains_key(&agent.model) {
                errors.push(format!(
                    "agent `{name}` references unknown model `{}`",
                    agent.model
                ));
            }
            for server in &agent.mcp {
                if !self.mcp.contains_key(server) {
                    errors.push(format!(
                        "agent `{name}` references unknown mcp server `{server}`"
                    ));
                }
            }
            if agent.system_prompt.is_some() && agent.system_prompt_file.is_some() {
                errors.push(format!(
                    "agent `{name}` sets both system_prompt and system_prompt_file"
                ));
            }
            if agent.web_search && self.settings.searxng_url.is_none() {
                errors.push(format!(
                    "agent `{name}` enables web_search but settings.searxng_url is not set"
                ));
            }
            for subagent in &agent.subagents {
                if !self.agents.contains_key(subagent) {
                    errors.push(format!(
                        "agent `{name}` subagents references unknown agent `{subagent}`"
                    ));
                }
            }
            if !agent.shell_allow.is_empty() && !agent.shell {
                errors.push(format!(
                    "agent `{name}` sets shell_allow but not `shell = true`"
                ));
            }
            if !agent.auto_approve.is_empty() && !agent.require_approval {
                errors.push(format!(
                    "agent `{name}` sets auto_approve but not `require_approval = true`"
                ));
            }
            if !agent.approval_effects.is_empty() && !agent.require_approval {
                errors.push(format!(
                    "agent `{name}` sets approval_effects but not `require_approval = true`"
                ));
            }
        }

        let all_stage_names: Vec<&str> = self.stages.iter().map(|s| s.name.as_str()).collect();
        let mut seen_stage_names: Vec<&str> = Vec::new();
        for (index, stage) in self.stages.iter().enumerate() {
            let name = &stage.name;
            if seen_stage_names.contains(&name.as_str()) {
                errors.push(format!("duplicate stage name `{name}`"));
            }

            for target in &stage.can_reprompt {
                if !all_stage_names.contains(&target.as_str()) {
                    errors.push(format!(
                        "stage `{name}` can_reprompt references unknown stage `{target}`"
                    ));
                }
            }

            for subagent in &stage.subagents {
                if !self.agents.contains_key(subagent) {
                    errors.push(format!(
                        "stage `{name}` subagents references unknown agent `{subagent}`"
                    ));
                }
            }

            if !stage.shell_allow.is_empty() && !stage.shell {
                errors.push(format!(
                    "stage `{name}` sets shell_allow but not `shell = true`"
                ));
            }
            if !stage.auto_approve.is_empty() && !stage.require_approval {
                errors.push(format!(
                    "stage `{name}` sets auto_approve but not `require_approval = true`"
                ));
            }
            if !stage.approval_effects.is_empty() && !stage.require_approval {
                errors.push(format!(
                    "stage `{name}` sets approval_effects but not `require_approval = true`"
                ));
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

        for (name, workflow) in &self.workflows {
            if workflow.stages.is_empty() {
                errors.push(format!("workflow `{name}` has no stages"));
            }
            let mut seen: Vec<&str> = Vec::new();
            for stage_name in &workflow.stages {
                if !all_stage_names.contains(&stage_name.as_str()) {
                    errors.push(format!(
                        "workflow `{name}` references unknown stage `{stage_name}`"
                    ));
                }
                if seen.contains(&stage_name.as_str()) {
                    errors.push(format!(
                        "workflow `{name}` lists stage `{stage_name}` more than once"
                    ));
                }
                seen.push(stage_name);
            }
        }
        if let Some(default) = &self.settings.default_workflow
            && !self.workflows.contains_key(default)
        {
            errors.push(format!(
                "settings.default_workflow references unknown workflow `{default}`"
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            bail!("invalid configuration:\n  - {}", errors.join("\n  - "))
        }
    }

    /// Resolve which stages `soa run` executes, in order: an explicitly
    /// requested workflow, else `settings.default_workflow`, else a workflow
    /// literally named `default`, else every stage in declaration order.
    pub fn resolve_workflow(&self, requested: Option<&str>) -> Result<Vec<usize>> {
        let workflow_name = requested
            .map(str::to_string)
            .or_else(|| self.settings.default_workflow.clone())
            .or_else(|| self.workflows.contains_key("default").then(|| "default".to_string()));

        let Some(name) = workflow_name else {
            return Ok((0..self.stages.len()).collect());
        };
        let workflow = self.workflows.get(&name).with_context(|| {
            format!(
                "no workflow named `{name}` (available: {})",
                self.workflows.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        Ok(workflow
            .stages
            .iter()
            .map(|stage_name| {
                self.stages
                    .iter()
                    .position(|s| &s.name == stage_name)
                    .expect("workflow stages are validated at config load")
            })
            .collect())
    }
}

/// Resolve `settings.context_files` against a working directory: each
/// entry is read from `cwd` or the nearest ancestor that has it (so runs
/// from a subdirectory of a project still pick up its instructions);
/// absolute entries are read directly. Files that exist are returned in
/// the listed order; missing or blank ones are skipped, and two entries
/// resolving to the same file are sourced once.
fn resolve_context_files(entries: &[PathBuf], cwd: &Path) -> Vec<ProjectContext> {
    let mut seen = std::collections::BTreeSet::new();
    entries
        .iter()
        .filter_map(|entry| {
            let found = if entry.is_absolute() {
                let content = std::fs::read_to_string(entry).ok()?;
                ProjectContext { path: entry.clone(), content }
            } else {
                cwd.ancestors().find_map(|dir| {
                    let path = dir.join(entry);
                    let content = std::fs::read_to_string(&path).ok()?;
                    Some(ProjectContext { path, content })
                })?
            };
            (!found.content.trim().is_empty() && seen.insert(found.path.clone()))
                .then_some(found)
        })
        .collect()
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
    fn resolves_context_files_in_order_walking_up() {
        let root = std::env::temp_dir().join(format!("soa-ctx-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let nested = root.join("a").join("b");
        std::fs::create_dir_all(&nested).unwrap();
        let entries =
            |names: &[&str]| names.iter().map(PathBuf::from).collect::<Vec<_>>();

        // Walk-up discovery: found in an ancestor of the working directory.
        std::fs::write(root.join("SOA.md"), "top instructions").unwrap();
        let found = resolve_context_files(&entries(&["SOA.md"]), &nested);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].content, "top instructions");
        assert_eq!(found[0].path, root.join("SOA.md"));

        // The nearest file wins for each entry independently.
        std::fs::write(nested.join("SOA.md"), "near instructions").unwrap();
        let found = resolve_context_files(&entries(&["SOA.md"]), &nested);
        assert_eq!(found[0].content, "near instructions");

        // Multiple entries source in the listed order, not discovery depth;
        // missing and blank ones are skipped; duplicates source once.
        std::fs::write(root.join("AGENTS.md"), "agent instructions").unwrap();
        std::fs::write(nested.join("BLANK.md"), "  \n").unwrap();
        let found = resolve_context_files(
            &entries(&["AGENTS.md", "BLANK.md", "MISSING.md", "SOA.md", "AGENTS.md"]),
            &nested,
        );
        assert_eq!(
            found.iter().map(|c| c.content.as_str()).collect::<Vec<_>>(),
            vec!["agent instructions", "near instructions"]
        );

        let _ = std::fs::remove_dir_all(&root);
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
    fn fallback_references_validated() {
        let toml_str = MINIMAL.replace(
            "model = \"qwen3:8b\"",
            "model = \"qwen3:8b\"\nfallback = [\"ghost\", \"default\"]",
        );
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("unknown fallback model `ghost`"), "{err}");
        assert!(err.contains("model `default` lists itself as a fallback"), "{err}");

        let toml_str = format!(
            "{}\n[models.backup]\nprovider = \"local\"\nmodel = \"qwen3:4b\"\n",
            MINIMAL.replace(
                "model = \"qwen3:8b\"",
                "model = \"qwen3:8b\"\nfallback = [\"backup\"]"
            )
        );
        assert!(parse(&toml_str).is_ok());
    }

    #[test]
    fn external_fallback_requires_explicit_consent() {
        let cloud_tables = r#"
            [providers.cloud]
            base_url = "https://api.example.invalid/v1"
            data_boundary = "external"

            [models.cloud]
            provider = "cloud"
            model = "proprietary-coder"
        "#;
        let without_consent = format!(
            "{}\n{cloud_tables}",
            MINIMAL.replace(
                "model = \"qwen3:8b\"",
                "model = \"qwen3:8b\"\nfallback = [\"cloud\"]"
            )
        );
        let error = parse(&without_consent).unwrap_err().to_string();
        assert!(error.contains("local data boundary to external"), "{error}");
        assert!(error.contains("allow_external_fallback = true"), "{error}");

        let with_consent = without_consent.replace(
            "fallback = [\"cloud\"]",
            "fallback = [\"cloud\"]\nallow_external_fallback = true",
        );
        assert!(parse(&with_consent).is_ok());
    }

    #[test]
    fn additional_approval_effects_require_approval_mode() {
        let without_gate = MINIMAL.replace(
            "name = \"answer\"",
            "name = \"answer\"\napproval_effects = [\"network_egress\"]",
        );
        let error = parse(&without_gate).unwrap_err().to_string();
        assert!(error.contains("approval_effects"), "{error}");
        assert!(error.contains("require_approval = true"), "{error}");

        let with_gate = without_gate.replace(
            "approval_effects = [\"network_egress\"]",
            "require_approval = true\napproval_effects = [\"network_egress\"]",
        );
        assert!(parse(&with_gate).is_ok());
    }

    #[test]
    fn agent_references_validated() {
        // Stage referencing an unknown agent.
        let toml_str = MINIMAL.replace(
            "name = \"answer\"",
            "name = \"answer\"\nsubagents = [\"nope\"]",
        );
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("unknown agent `nope`"), "{err}");

        // Agent with a bad model and a bad subagent reference.
        let toml_str = format!(
            "{MINIMAL}\n[agents.helper]\nmodel = \"missing\"\nsubagents = [\"ghost\"]\n"
        );
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("agent `helper` references unknown model `missing`"), "{err}");
        assert!(err.contains("unknown agent `ghost`"), "{err}");

        // A valid agent wired to the stage parses.
        let toml_str = format!(
            "{}\n[agents.helper]\nmodel = \"default\"\ndescription = \"helps\"\n",
            MINIMAL.replace("name = \"answer\"", "name = \"answer\"\nsubagents = [\"helper\"]")
        );
        assert!(parse(&toml_str).is_ok());
    }

    #[test]
    fn workflow_validation_and_resolution() {
        // Unknown stage and duplicate stage rejected.
        let toml_str = format!(
            "{MINIMAL}\n[workflows.bad]\nstages = [\"answer\", \"ghost\", \"answer\"]\n"
        );
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("unknown stage `ghost`"), "{err}");
        assert!(err.contains("more than once"), "{err}");

        // Missing default_workflow rejected.
        let toml_str = format!(
            "{}\n[workflows.w]\nstages = [\"answer\"]\n",
            MINIMAL.replace(
                "[providers.local]",
                "[settings]\ndefault_workflow = \"nope\"\n\n[providers.local]"
            )
        );
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("unknown workflow `nope`"), "{err}");

        // Resolution precedence: explicit > default_workflow > "default" > all stages.
        let toml_str = format!(
            "{MINIMAL}\n[[stage]]\nname = \"second\"\nmodel = \"default\"\n\n\
             [workflows.default]\nstages = [\"second\"]\n\
             [workflows.full]\nstages = [\"answer\", \"second\"]\n"
        );
        let config = parse(&toml_str).unwrap();
        assert_eq!(config.resolve_workflow(Some("full")).unwrap(), vec![0, 1]);
        assert_eq!(config.resolve_workflow(None).unwrap(), vec![1]); // named "default"
        assert!(config.resolve_workflow(Some("nope")).is_err());

        // No workflows at all: declaration order.
        let config = parse(MINIMAL).unwrap();
        assert_eq!(config.resolve_workflow(None).unwrap(), vec![0]);
    }

    #[test]
    fn unknown_reprompt_target_rejected() {
        let toml_str = MINIMAL.replace(
            "name = \"answer\"",
            "name = \"answer\"\ncan_reprompt = [\"answer\", \"nope\"]",
        );
        let err = parse(&toml_str).unwrap_err().to_string();
        assert!(err.contains("can_reprompt references unknown stage `nope`"), "{err}");
        // Self-reference and any existing stage are fine.
        let toml_str = MINIMAL.replace(
            "name = \"answer\"",
            "name = \"answer\"\ncan_reprompt = [\"answer\"]",
        );
        assert!(parse(&toml_str).is_ok());
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
