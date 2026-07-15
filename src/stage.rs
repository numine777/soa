//! Stage execution: template rendering, effect-aware tool assembly and
//! approval policy, and the per-stage agentic tool-call loop.

use std::collections::{BTreeMap, VecDeque};
use std::future::Future;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use futures_util::stream::{FuturesUnordered, StreamExt};
use rmcp::model::JsonObject;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::approval::{Approvals, Decision};
use crate::config::{Config, DataBoundary, McpServer, Mode, Stage, ToolEffect};
use crate::mcp::McpManager;
use crate::model::{
    DeltaHandler, Message, ModelClient, ModelPricing, ModelTarget, ProviderAdapter, SamplingParams,
    ToolCall, ToolDefinition, Usage, UsageTracker,
};
use crate::tools;

/// Render `{{input}}`, `{{previous}}`, and `{{stage.<name>}}` placeholders.
pub fn render_template(
    template: &str,
    input: &str,
    previous: Option<&str>,
    stage_outputs: &BTreeMap<String, String>,
) -> Result<String> {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else {
            out.push_str("{{");
            rest = after;
            continue;
        };
        let var = after[..end].trim();
        match var {
            "input" => out.push_str(input),
            "previous" => out.push_str(
                previous.ok_or_else(|| anyhow!("{{{{previous}}}} used in the first stage"))?,
            ),
            other => {
                let name = other
                    .strip_prefix("stage.")
                    .ok_or_else(|| anyhow!("unknown template variable {{{{{other}}}}}"))?;
                let output = stage_outputs
                    .get(name)
                    .ok_or_else(|| anyhow!("no output recorded for stage `{name}`"))?;
                out.push_str(output);
            }
        }
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Where a tool call is routed when the model invokes it.
#[derive(Debug, Clone)]
pub enum ToolBinding {
    Mcp {
        server: String,
        tool: String,
    },
    WebSearch,
    /// Fetch a URL and return its readable text.
    WebFetch,
    /// Delegate the call's `task` to a configured subagent.
    Agent {
        agent: String,
    },
    /// Run the call's `command` with `sh -c`, restricted to the owning
    /// context's allowlist patterns (empty = unrestricted).
    Shell {
        allow: Vec<String>,
    },
    /// A built-in file tool, rooted at the working directory.
    File {
        op: crate::files::FileOp,
    },
}

/// Compact set of observable effects attached to a tool. Keeping this
/// separate from `read_only` lets approval, delegation, and parallelism make
/// decisions about the actual capability involved (for example, network
/// egress is read-like but may still need consent).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolEffects(u16);

impl ToolEffects {
    pub const NONE: ToolEffects = ToolEffects(0);

    const fn bit(effect: ToolEffect) -> u16 {
        match effect {
            ToolEffect::FilesystemRead => 1 << 0,
            ToolEffect::FilesystemWrite => 1 << 1,
            ToolEffect::ProcessExecute => 1 << 2,
            ToolEffect::NetworkEgress => 1 << 3,
            ToolEffect::ExternalRead => 1 << 4,
            ToolEffect::ExternalMutation => 1 << 5,
        }
    }

    pub const fn one(effect: ToolEffect) -> ToolEffects {
        ToolEffects(Self::bit(effect))
    }

    pub fn insert(&mut self, effect: ToolEffect) {
        self.0 |= Self::bit(effect);
    }

    pub fn union(self, other: ToolEffects) -> ToolEffects {
        ToolEffects(self.0 | other.0)
    }

    pub fn contains(self, effect: ToolEffect) -> bool {
        self.0 & Self::bit(effect) != 0
    }

    pub fn intersects(self, effects: &[ToolEffect]) -> bool {
        effects.iter().any(|effect| self.contains(*effect))
    }

    /// Effects that can change state or execute arbitrary local processes.
    pub fn mutating_or_process(self) -> bool {
        self.contains(ToolEffect::FilesystemWrite)
            || self.contains(ToolEffect::ProcessExecute)
            || self.contains(ToolEffect::ExternalMutation)
    }

    /// Safe to expose through a read-only delegation boundary.
    pub fn read_only_safe(self) -> bool {
        !self.mutating_or_process()
    }

    /// Safe to dispatch beside other calls in the same tool round.
    pub fn parallel_safe(self) -> bool {
        !self.mutating_or_process()
    }
}

pub struct StageTool {
    pub definition: ToolDefinition,
    pub binding: ToolBinding,
    pub effects: ToolEffects,
}

/// What a model context (stage or agent) exposes to its model: shared shape
/// for tool assembly.
pub struct ToolProfile<'a> {
    /// For log messages.
    pub owner: &'a str,
    pub mode: Mode,
    pub mcp: &'a [String],
    pub web_search: bool,
    pub web_fetch: bool,
    pub subagents: &'a [String],
    pub shell: bool,
    pub shell_allow: &'a [String],
    pub files: bool,
}

impl crate::config::Stage {
    pub fn tool_profile(&self) -> ToolProfile<'_> {
        ToolProfile {
            owner: &self.name,
            mode: self.mode,
            mcp: &self.mcp,
            web_search: self.web_search,
            web_fetch: self.web_fetch,
            subagents: &self.subagents,
            shell: self.shell,
            shell_allow: &self.shell_allow,
            files: self.files,
        }
    }
}

impl crate::config::Agent {
    pub fn tool_profile<'a>(&'a self, name: &'a str) -> ToolProfile<'a> {
        ToolProfile {
            owner: name,
            mode: self.mode,
            mcp: &self.mcp,
            web_search: self.web_search,
            web_fetch: self.web_fetch,
            subagents: &self.subagents,
            shell: self.shell,
            shell_allow: &self.shell_allow,
            files: self.files,
        }
    }
}

/// How a stage finished: with a final answer, or by handing control to
/// another stage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StageOutcome {
    Final(String),
    Reprompt {
        target: String,
        instructions: String,
    },
}

/// Append-only events emitted by the canonical model/tool loop. Pipeline
/// checkpoints persist these during an active stage and replay them on
/// resume, so completed model responses and tool calls are not repeated.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentLoopEvent {
    Started {
        system: Option<String>,
        messages: Vec<Message>,
    },
    ContextShed {
        keep_recent: usize,
    },
    Assistant {
        content: Option<String>,
        tool_calls: Vec<ToolCall>,
        usage: Option<Usage>,
        /// Opaque provider reasoning payload — see
        /// [`crate::model::Message::Assistant`]. Absent in logs written by
        /// older soa versions.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<Value>,
    },
    /// Intent record written just before a mutating or process-executing
    /// call runs. On resume, a started call with no recorded result is NOT
    /// re-executed — its effects may already have happened — and the model
    /// is asked to verify instead.
    ToolStarted {
        call_index: usize,
    },
    ToolResult {
        call_index: usize,
        content: String,
    },
    UserMessage {
        content: String,
    },
    /// One event of a subagent's own loop, recorded while an
    /// `agent__<name>` delegation (the pending call at `call_index`) runs.
    /// Resume extracts these and continues the delegation mid-run instead
    /// of re-running it from scratch; nested delegations nest the variant.
    Subagent {
        call_index: usize,
        event: Box<AgentLoopEvent>,
    },
    Finished {
        outcome: StageOutcome,
        usage: Option<Usage>,
    },
}

/// Non-durable progress notifications used by the chat UI.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentLoopObservation {
    ToolCall { name: String, args: String },
    ToolDone { preview: String },
    Notice(String),
}

pub type LoopEventSink<'a> = Option<&'a (dyn Fn(AgentLoopEvent) + Send + Sync)>;
pub type LoopObservationSink<'a> = Option<&'a (dyn Fn(AgentLoopObservation) + Send + Sync)>;

/// Per-caller behavior around the one canonical agent loop.
pub struct AgentLoopOptions<'a> {
    pub owner_kind: &'static str,
    pub owner: &'a str,
    pub model_name: &'a str,
    pub system: Option<&'a str>,
    pub max_turns: u32,
    pub depth: u32,
    pub require_approval: bool,
    pub approval_effects: &'a [ToolEffect],
    pub auto_approve: &'a [String],
    /// Forces the FIRST model request of the run to call a tool (or one
    /// specific tool); later rounds are unconstrained so the model can
    /// still produce a final answer.
    pub tool_choice: Option<&'a crate::model::ToolChoice>,
    /// JSON Schema constraining the response text; sent on every request.
    pub output_schema: Option<&'a Value>,
    pub reprompt_targets: &'a [String],
    pub on_delta: Option<DeltaHandler<'a>>,
    pub terminate_streamed_response: bool,
    pub on_diff: DiffSink<'a>,
    pub on_event: LoopEventSink<'a>,
    pub on_observation: LoopObservationSink<'a>,
    pub steer: Option<&'a Mutex<VecDeque<String>>>,
    pub tool_errors_as_results: bool,
}

pub struct AgentLoopResult {
    pub outcome: StageOutcome,
    pub messages: Vec<Message>,
    pub usage: Option<Usage>,
}

pub const REPROMPT_TOOL: &str = "reprompt_stage";

/// The routing tool offered to stages with a non-empty `can_reprompt` list.
fn reprompt_tool(targets: &[String]) -> ToolDefinition {
    ToolDefinition {
        name: REPROMPT_TOOL.to_string(),
        description: "Hand control to another stage because more work is needed. \
            The pipeline resumes from that stage and continues in order (so a stage \
            that normally runs after the target will run again). Calling this ends \
            your turn immediately. Only call it when further changes are genuinely \
            required — otherwise reply normally with your final answer."
            .to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "stage": {
                    "type": "string",
                    "enum": targets,
                    "description": "The stage to hand control to"
                },
                "instructions": {
                    "type": "string",
                    "description": "Specific, actionable instructions for that stage"
                }
            },
            "required": ["stage", "instructions"]
        }),
    }
}

/// Tool names advertised to the model must match `[a-zA-Z0-9_-]{1,64}`.
fn sanitize_tool_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    cleaned.chars().take(64).collect()
}

/// Assemble the tools visible to a context, applying mode and effect filters.
/// MCP tool names are namespaced as `<server>__<tool>` to avoid collisions;
/// subagents appear as `agent__<name>`. `depth` is how many delegation
/// levels deep this context already is: agent tools stop being offered at
/// `settings.max_agent_depth`.
pub fn assemble_tools(
    profile: &ToolProfile<'_>,
    config: &Config,
    mcp: &McpManager,
    depth: u32,
) -> Result<Vec<StageTool>> {
    let mut stage_tools = Vec::new();

    for server_name in profile.mcp {
        let connection = mcp
            .get(server_name)
            .ok_or_else(|| anyhow!("mcp server `{server_name}` is not connected"))?;
        for tool in &connection.tools {
            let read_only = connection.is_read_only(tool);
            if profile.mode == Mode::ReadOnly && !read_only {
                tracing::debug!(
                    owner = %profile.owner, server = %server_name, tool = %tool.name,
                    "hidden in read_only mode"
                );
                continue;
            }
            let mut effects = if read_only {
                ToolEffects::one(ToolEffect::ExternalRead)
            } else {
                ToolEffects::one(ToolEffect::ExternalMutation)
            };
            if matches!(config.mcp.get(server_name), Some(McpServer::Http { .. })) {
                effects.insert(ToolEffect::NetworkEgress);
            }
            stage_tools.push(StageTool {
                definition: ToolDefinition {
                    name: sanitize_tool_name(&format!("{server_name}__{}", tool.name)),
                    description: tool.description.clone().unwrap_or_default().into_owned(),
                    parameters: Value::Object((*tool.input_schema).clone()),
                },
                binding: ToolBinding::Mcp {
                    server: server_name.clone(),
                    tool: tool.name.to_string(),
                },
                effects,
            });
        }
    }

    if profile.web_search {
        stage_tools.push(StageTool {
            definition: tools::web_search_definition(),
            binding: ToolBinding::WebSearch,
            effects: ToolEffects::one(ToolEffect::NetworkEgress),
        });
    }

    if profile.web_fetch {
        stage_tools.push(StageTool {
            definition: tools::web_fetch_definition(),
            binding: ToolBinding::WebFetch,
            effects: ToolEffects::one(ToolEffect::NetworkEgress),
        });
    }

    // `shell = true` is an explicit per-context grant, offered regardless
    // of mode: requiring read_write just to run tests would force the whole
    // MCP write surface onto review-style stages.
    if profile.shell {
        stage_tools.push(StageTool {
            definition: tools::shell_definition(
                config.settings.shell_timeout_secs,
                profile.shell_allow,
            ),
            binding: ToolBinding::Shell {
                allow: profile.shell_allow.to_vec(),
            },
            effects: ToolEffects::one(ToolEffect::ProcessExecute),
        });
    }

    // `files = true` exposes the built-in file tools; the mode decides
    // whether the write tools are included.
    if profile.files {
        for (definition, op, read_only) in
            crate::files::definitions(profile.mode == Mode::ReadWrite)
        {
            stage_tools.push(StageTool {
                definition,
                binding: ToolBinding::File { op },
                effects: ToolEffects::one(if read_only {
                    ToolEffect::FilesystemRead
                } else {
                    ToolEffect::FilesystemWrite
                }),
            });
        }
    }

    if depth < config.settings.max_agent_depth {
        for agent_name in profile.subagents {
            let agent = config
                .agents
                .get(agent_name)
                .ok_or_else(|| anyhow!("unknown agent `{agent_name}`"))?;
            // Include everything the delegated agent can reach, not merely
            // its declared mode. In particular, a read-only agent with an
            // explicit shell grant is not safe to smuggle through a
            // read-only caller.
            let nested_tools =
                assemble_tools(&agent.tool_profile(agent_name), config, mcp, depth + 1)?;
            let mut agent_effects = nested_tools
                .iter()
                .fold(ToolEffects::NONE, |all, tool| all.union(tool.effects));
            if agent.mode == Mode::ReadWrite {
                // Preserve the conservative mode boundary even when the
                // agent currently has no configured write tool.
                agent_effects.insert(ToolEffect::ExternalMutation);
            }
            if model_chain_reaches_external(config, &agent.model) {
                agent_effects.insert(ToolEffect::NetworkEgress);
            }
            if profile.mode == Mode::ReadOnly && !agent_effects.read_only_safe() {
                tracing::debug!(
                    owner = %profile.owner, agent = %agent_name,
                    effects = ?agent_effects,
                    "effectful agent hidden in read_only mode"
                );
                continue;
            }
            let about = if agent.description.is_empty() {
                format!("Delegate a task to the `{agent_name}` agent.")
            } else {
                agent.description.clone()
            };
            stage_tools.push(StageTool {
                definition: ToolDefinition {
                    name: sanitize_tool_name(&format!("agent__{agent_name}")),
                    description: format!(
                        "{about} Runs as a separate agent with its own tools and no memory \
                         of this conversation: give it one complete, self-contained task \
                         and it returns its final answer."
                    ),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "The complete task, including all context the agent needs"
                            }
                        },
                        "required": ["task"]
                    }),
                },
                binding: ToolBinding::Agent { agent: agent_name.clone() },
                effects: agent_effects,
            });
        }
    } else if !profile.subagents.is_empty() {
        tracing::debug!(
            owner = %profile.owner, depth,
            "subagents hidden: settings.max_agent_depth reached"
        );
    }

    Ok(stage_tools)
}

fn model_chain_reaches_external(config: &Config, model_name: &str) -> bool {
    let mut queue = std::collections::VecDeque::from([model_name.to_string()]);
    let mut seen = std::collections::BTreeSet::new();
    while let Some(name) = queue.pop_front() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(model) = config.models.get(&name) else {
            continue;
        };
        if config
            .providers
            .get(&model.provider)
            .is_some_and(|provider| provider.data_boundary == DataBoundary::External)
        {
            return true;
        }
        queue.extend(model.fallback.iter().cloned());
    }
    false
}

/// Approval policy for one tool call, derived from its owning context.
pub struct CallPolicy<'a> {
    /// True when this call's effects require approval in its owning context.
    pub require_approval: bool,
    pub auto_approve: &'a [String],
    pub approvals: &'a Approvals,
}

impl<'a> CallPolicy<'a> {
    pub fn approval_required(
        context_requires: bool,
        additional_effects: &[ToolEffect],
        tool_effects: ToolEffects,
    ) -> bool {
        context_requires
            && (tool_effects.mutating_or_process() || tool_effects.intersects(additional_effects))
    }

    /// Policy for a context's tool: mutation and process execution use the
    /// compatibility default, while contexts can additionally gate effects
    /// such as network egress.
    pub fn for_tool(
        context_requires: bool,
        additional_effects: &'a [ToolEffect],
        auto_approve: &'a [String],
        approvals: &'a Approvals,
        tool_effects: ToolEffects,
    ) -> CallPolicy<'a> {
        CallPolicy {
            require_approval: Self::approval_required(
                context_requires,
                additional_effects,
                tool_effects,
            ),
            auto_approve,
            approvals,
        }
    }
}

/// True when a round's tool calls may dispatch concurrently: several
/// calls, every one resolving to a parallel-safe, non-approval-gated tool.
/// Mutations could conflict, approval prompts must stay sequential, and control tools
/// (`reprompt_stage`) or unknown names take the sequential path, which
/// knows how to answer them.
pub fn parallel_round(
    enabled: bool,
    calls: &[crate::model::ToolCall],
    parallel_safe_of: impl Fn(&str) -> Option<bool>,
) -> bool {
    enabled
        && calls.len() > 1
        && calls
            .iter()
            .all(|call| parallel_safe_of(&call.function.name) == Some(true))
}

/// How a call is presented to the approver and matched against patterns.
pub struct CallDescriptor {
    pub descriptor: String,
    pub detail: String,
    /// What an "always" grant would cover for the rest of the session.
    pub always_pattern: String,
    /// Broad approval patterns are safe only for simple shell commands.
    /// Compound commands must receive an explicit one-off approval even if
    /// their textual prefix matches an `auto_approve` or session grant.
    pub pattern_safe: bool,
}

/// Compact JSON for logs, transcripts, and hook/approval details.
pub fn arguments_preview(arguments: &Value) -> String {
    match arguments {
        Value::Null => "{}".to_string(),
        other => other.to_string(),
    }
}

pub fn call_descriptor(binding: &ToolBinding, arguments: &Value) -> CallDescriptor {
    let args = arguments;
    let detail_json = arguments_preview(arguments);
    match binding {
        ToolBinding::Mcp { server, tool } => {
            let name = sanitize_tool_name(&format!("{server}__{tool}"));
            CallDescriptor {
                descriptor: name.clone(),
                detail: truncate(&detail_json, 200),
                always_pattern: name,
                pattern_safe: true,
            }
        }
        ToolBinding::WebSearch => CallDescriptor {
            descriptor: tools::WEB_SEARCH_TOOL.to_string(),
            detail: truncate(&detail_json, 200),
            always_pattern: tools::WEB_SEARCH_TOOL.to_string(),
            pattern_safe: true,
        },
        ToolBinding::WebFetch => {
            let url = args.get("url").and_then(Value::as_str).unwrap_or("?");
            CallDescriptor {
                descriptor: format!("{} {}", tools::WEB_FETCH_TOOL, truncate(url, 160)),
                detail: truncate(url, 200),
                always_pattern: format!("{} *", tools::WEB_FETCH_TOOL),
                pattern_safe: true,
            }
        }
        ToolBinding::Shell { .. } => {
            let command = args
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            let first_word = command.split_whitespace().next().unwrap_or("?");
            CallDescriptor {
                descriptor: format!("shell {}", truncate(command, 160)),
                detail: command.to_string(),
                always_pattern: format!("shell {first_word} *"),
                pattern_safe: tools::shell_command_is_simple(command),
            }
        }
        ToolBinding::Agent { agent } => {
            let task = args.get("task").and_then(Value::as_str).unwrap_or_default();
            CallDescriptor {
                descriptor: format!("agent__{agent}"),
                detail: truncate(task, 200),
                always_pattern: format!("agent__{agent}"),
                pattern_safe: true,
            }
        }
        ToolBinding::File { op } => {
            let path = args.get("path").and_then(Value::as_str).unwrap_or("?");
            CallDescriptor {
                descriptor: format!("{} {}", op.tool_name(), truncate(path, 160)),
                detail: truncate(&detail_json, 200),
                always_pattern: format!("{} *", op.tool_name()),
                pattern_safe: true,
            }
        }
    }
}

/// Execute a tool call: enforce the approval policy, run it, and clamp the
/// result so a single oversized output cannot exhaust the model's context.
/// `depth` is the caller's delegation depth (0 for stages and chat).
#[allow(clippy::too_many_arguments)]
pub async fn dispatch_tool_call(
    binding: &ToolBinding,
    arguments: &Value,
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
    usage: &UsageTracker,
    depth: u32,
    policy: &CallPolicy<'_>,
    subagent: SubagentCheckpoint<'_>,
) -> Result<String> {
    let described = call_descriptor(binding, arguments);

    // pre_tool hooks run before the approval prompt: a call a hook is
    // going to block should never interrupt the user.
    if let Some(blocked) =
        crate::hooks::pre_tool(config, &described.descriptor, arguments).await
    {
        return Ok(blocked);
    }

    if policy.require_approval {
        let pre_approved = described.pattern_safe
            && (policy
                .auto_approve
                .iter()
                .any(|pattern| tools::wildcard_match(pattern, &described.descriptor))
                || policy.approvals.session_allowed(&described.descriptor));
        if !pre_approved {
            if !policy.approvals.is_interactive() {
                return Ok(format!(
                    "DENIED: `{}` requires approval, but this session has no interactive \
                     approver. Ask the user to add a matching auto_approve pattern or to \
                     run interactively.",
                    described.descriptor
                ));
            }
            tracing::info!(call = %described.descriptor, "awaiting approval");
            match policy
                .approvals
                .request(
                    described.descriptor.clone(),
                    described.detail,
                    described.always_pattern.clone(),
                )
                .await
            {
                Decision::Approve => {}
                Decision::AlwaysAllow => {
                    policy.approvals.allow_always(described.always_pattern);
                }
                Decision::Deny => {
                    return Ok(format!(
                        "DENIED: the user declined `{}`. Do not retry the same call; \
                         adjust your approach or ask for guidance.",
                        described.descriptor
                    ));
                }
            }
        }
    }

    let output = dispatch_tool_call_inner(
        binding,
        arguments,
        config,
        mcp,
        http,
        usage,
        depth,
        policy,
        subagent,
    )
    .await?;
    let output =
        crate::hooks::post_tool(config, &described.descriptor, arguments, output).await;
    Ok(clamp_tool_output(
        output,
        config.settings.max_tool_output_chars,
    ))
}

/// Truncate at a character boundary, telling the model what was cut so it
/// can re-query more narrowly.
fn clamp_tool_output(output: String, max_chars: usize) -> String {
    if max_chars == 0 || output.chars().count() <= max_chars {
        return output;
    }
    let total = output.chars().count();
    let kept: String = output.chars().take(max_chars).collect();
    format!(
        "{kept}\n… [tool output truncated: {total} characters total, showing the first \
         {max_chars}. Re-run the tool with a narrower query (e.g. a specific \
         subdirectory or file) to see the rest.]"
    )
}

#[allow(clippy::too_many_arguments)]
async fn dispatch_tool_call_inner(
    binding: &ToolBinding,
    arguments: &Value,
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
    usage: &UsageTracker,
    depth: u32,
    policy: &CallPolicy<'_>,
    subagent: SubagentCheckpoint<'_>,
) -> Result<String> {
    // The adapter already decoded the wire encoding. `Null` means "no
    // arguments"; any other non-object survived a malformed generation —
    // feed it back to the model instead of aborting the stage.
    let empty;
    let arguments = match arguments {
        Value::Object(_) => arguments,
        Value::Null => {
            empty = Value::Object(JsonObject::new());
            &empty
        }
        other => {
            return Ok(format!(
                "ERROR: tool arguments were not a JSON object: {}",
                truncate(&other.to_string(), 200)
            ));
        }
    };

    match binding {
        ToolBinding::WebSearch => {
            let query = arguments
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if query.is_empty() {
                return Ok("ERROR: web_search requires a non-empty `query` string".to_string());
            }
            let searxng_url = config
                .settings
                .searxng_url
                .as_deref()
                .context("searxng_url is not configured")?;
            match tools::web_search(
                http,
                searxng_url,
                query,
                config.settings.searxng_max_results,
            )
            .await
            {
                Ok(results) => Ok(results),
                Err(e) => Ok(format!("ERROR: web search failed: {e:#}")),
            }
        }
        ToolBinding::WebFetch => {
            let url = arguments
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            if url.is_empty() {
                return Ok("ERROR: web_fetch requires a non-empty `url` string".to_string());
            }
            match tools::web_fetch(http, url).await {
                Ok(content) => Ok(content),
                Err(e) => Ok(format!("ERROR: web fetch failed: {e:#}")),
            }
        }
        ToolBinding::Mcp { server, tool } => {
            let Value::Object(object) = arguments else {
                return Ok("ERROR: tool arguments must be a JSON object".to_string());
            };
            let connection = mcp
                .get(server)
                .ok_or_else(|| anyhow!("mcp server `{server}` is not connected"))?;
            match connection.call(tool, object.clone()).await {
                Ok(text) => Ok(text),
                // Transport-level failure: report to the model, keep the stage alive.
                Err(e) => Ok(format!("ERROR: {e:#}")),
            }
        }
        ToolBinding::Agent { agent } => {
            let task = arguments
                .get("task")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if task.trim().is_empty() {
                return Ok("ERROR: the agent needs a non-empty `task` string".to_string());
            }
            match run_agent(
                config,
                agent,
                task,
                mcp,
                http,
                usage,
                depth + 1,
                policy.approvals,
                subagent,
            )
            .await
            {
                Ok(answer) => Ok(answer),
                // The agent's failure becomes feedback, not a crashed turn.
                Err(e) => Ok(format!("ERROR: agent `{agent}` failed: {e:#}")),
            }
        }
        ToolBinding::Shell { allow } => {
            let command = arguments
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if command.trim().is_empty() {
                return Ok("ERROR: `command` must be a non-empty string".to_string());
            }
            if !allow.is_empty() && !tools::shell_command_is_simple(command) {
                return Ok(
                    "ERROR: command not permitted — shell_allow accepts one simple command; \
                     pipes, command lists, redirections, subshells, and command substitutions \
                     require an unrestricted shell grant"
                        .to_string(),
                );
            }
            if !tools::command_allowed(allow, command) {
                return Ok(format!(
                    "ERROR: command not permitted here — allowed patterns: {}",
                    allow.join(", ")
                ));
            }
            tracing::info!(command = %truncate(command, 200), "shell exec");
            Ok(tools::run_shell(
                command,
                std::time::Duration::from_secs(config.settings.shell_timeout_secs),
            )
            .await)
        }
        ToolBinding::File { op } => {
            tracing::info!(
                tool = op.tool_name(),
                args = %truncate(&arguments_preview(arguments), 200),
                "file tool"
            );
            Ok(crate::files::dispatch(*op, arguments))
        }
    }
}

#[derive(Debug)]
struct PendingToolRound {
    calls: Vec<ToolCall>,
    results: BTreeMap<usize, String>,
    /// Calls whose execution began (mutating/process calls only) — see
    /// [`AgentLoopEvent::ToolStarted`].
    started: std::collections::BTreeSet<usize>,
}

struct ReplayedAgentLoop {
    system: Option<String>,
    messages: Vec<Message>,
    pending: Option<PendingToolRound>,
    finished: Option<StageOutcome>,
    turns: u32,
    usage: Option<Usage>,
    /// Context was shed after the last usage measurement, so `usage` no
    /// longer reflects the conversation's real size — pre-flight sizing
    /// must fall back to the character estimate.
    shed_since_usage: bool,
}

/// Rebuild the conversation a cancelled (or failed) loop had accumulated,
/// so completed tool rounds survive into the caller's history — their
/// effects are already on disk, and a model that cannot see them will
/// contradict the filesystem. An incomplete final round is closed with
/// synthetic "interrupted" results so the history stays protocol-valid.
/// Returns `None` when nothing was recorded.
pub fn salvage_cancelled_loop(events: &[AgentLoopEvent]) -> Result<Option<Vec<Message>>> {
    if events.is_empty() {
        return Ok(None);
    }
    let replayed = replay_agent_loop(events)?;
    let mut messages = replayed.messages;
    if let Some(round) = replayed.pending {
        for (index, call) in round.calls.iter().enumerate() {
            let content = round.results.get(&index).cloned().unwrap_or_else(|| {
                "The user cancelled the turn before this tool call finished; \
                 it may or may not have taken effect."
                    .to_string()
            });
            messages.push(Message::Tool {
                content,
                tool_call_id: call.id.clone(),
            });
        }
    }
    Ok(Some(messages))
}

fn flush_completed_round(
    messages: &mut Vec<Message>,
    pending: &mut Option<PendingToolRound>,
) -> Result<()> {
    let Some(round) = pending.as_ref() else {
        return Ok(());
    };
    if round.results.len() != round.calls.len() {
        return Ok(());
    }
    for (index, call) in round.calls.iter().enumerate() {
        let content = round
            .results
            .get(&index)
            .with_context(|| format!("agent event log is missing tool result {index}"))?
            .clone();
        messages.push(Message::Tool {
            content,
            tool_call_id: call.id.clone(),
        });
    }
    *pending = None;
    Ok(())
}

/// Rebuild the canonical conversation from an append-only loop event log.
/// Tool results may be logged in completion order, but are materialized in
/// the assistant's original call order before the next model request.
fn replay_agent_loop(events: &[AgentLoopEvent]) -> Result<ReplayedAgentLoop> {
    let mut system = None;
    let mut messages = Vec::new();
    let mut pending: Option<PendingToolRound> = None;
    let mut finished = None;
    let mut turns = 0u32;
    let mut usage = None;
    let mut shed_since_usage = false;
    let mut started = false;

    for (event_index, event) in events.iter().enumerate() {
        if finished.is_some() {
            bail!("agent event log contains entries after its finished event");
        }
        match event {
            AgentLoopEvent::Started {
                system: saved_system,
                messages: saved_messages,
            } => {
                if started || event_index != 0 {
                    bail!("agent event log contains a misplaced started event");
                }
                started = true;
                system = saved_system.clone();
                messages = saved_messages.clone();
            }
            AgentLoopEvent::ContextShed { keep_recent } => {
                if !started {
                    bail!("agent event log does not start with a started event");
                }
                flush_completed_round(&mut messages, &mut pending)?;
                if pending.is_some() {
                    bail!("agent event log sheds context during an incomplete tool round");
                }
                shed_context(&mut messages, *keep_recent);
                shed_since_usage = true;
            }
            AgentLoopEvent::Assistant {
                content,
                tool_calls,
                usage: reported_usage,
                reasoning,
            } => {
                if !started {
                    bail!("agent event log does not start with a started event");
                }
                flush_completed_round(&mut messages, &mut pending)?;
                if pending.is_some() {
                    bail!("agent event log starts a model turn before all tools completed");
                }
                if tool_calls.is_empty() {
                    bail!("agent event log has an assistant event without tool calls");
                }
                messages.push(Message::Assistant {
                    content: content.clone(),
                    tool_calls: Some(tool_calls.clone()),
                    reasoning: reasoning.clone(),
                });
                pending = Some(PendingToolRound {
                    calls: tool_calls.clone(),
                    results: BTreeMap::new(),
                    started: std::collections::BTreeSet::new(),
                });
                turns = turns.saturating_add(1);
                usage = *reported_usage;
                shed_since_usage = false;
            }
            AgentLoopEvent::ToolStarted { call_index } => {
                let round = pending
                    .as_mut()
                    .context("agent event log has a tool intent outside a tool round")?;
                if *call_index >= round.calls.len() {
                    bail!("agent event log has out-of-range tool intent {call_index}");
                }
                // Repeated interruptions may record the same intent again;
                // the set makes that harmless.
                round.started.insert(*call_index);
            }
            AgentLoopEvent::ToolResult {
                call_index,
                content,
            } => {
                let round = pending
                    .as_mut()
                    .context("agent event log has a tool result outside a tool round")?;
                if *call_index >= round.calls.len() {
                    bail!("agent event log has out-of-range tool result {call_index}");
                }
                if round.results.insert(*call_index, content.clone()).is_some() {
                    bail!("agent event log repeats tool result {call_index}");
                }
                flush_completed_round(&mut messages, &mut pending)?;
            }
            // Subagent events shape the delegation's own log, not the
            // parent conversation — resume extraction is their consumer.
            AgentLoopEvent::Subagent { .. } => {}
            AgentLoopEvent::UserMessage { content } => {
                flush_completed_round(&mut messages, &mut pending)?;
                if pending.is_some() {
                    bail!("agent event log steers during an incomplete tool round");
                }
                messages.push(Message::User {
                    content: content.clone(),
                });
            }
            AgentLoopEvent::Finished {
                outcome,
                usage: reported_usage,
            } => {
                if matches!(outcome, StageOutcome::Final(_)) {
                    flush_completed_round(&mut messages, &mut pending)?;
                    if pending.is_some() {
                        bail!("agent event log finishes normally during an incomplete tool round");
                    }
                    let StageOutcome::Final(content) = outcome else {
                        unreachable!()
                    };
                    // Reasoning for a completed final turn is not carried
                    // (providers accept omitted thinking on past turns).
                    messages.push(Message::Assistant {
                        content: Some(content.clone()),
                        tool_calls: None,
                        reasoning: None,
                    });
                }
                finished = Some(outcome.clone());
                usage = *reported_usage;
            }
        }
    }

    if !started {
        bail!("agent event log does not start with a started event");
    }
    flush_completed_round(&mut messages, &mut pending)?;
    Ok(ReplayedAgentLoop {
        system,
        messages,
        pending,
        finished,
        turns,
        usage,
        shed_since_usage,
    })
}

fn record_loop_event(
    events: &mut Vec<AgentLoopEvent>,
    sink: LoopEventSink<'_>,
    event: AgentLoopEvent,
) {
    if let Some(sink) = sink {
        sink(event.clone());
    }
    events.push(event);
}

fn observe_loop(sink: LoopObservationSink<'_>, event: AgentLoopObservation) {
    if let Some(sink) = sink {
        sink(event);
    }
}

/// Checkpoint hooks for one `agent__<name>` delegation: the subagent
/// events already recorded for this call (so an interrupted delegation
/// resumes mid-run), and a sink that persists the delegation's new events
/// wrapped under the call's index.
pub struct SubagentCheckpoint<'a> {
    pub resume_events: Vec<AgentLoopEvent>,
    pub on_event: LoopEventSink<'a>,
}

/// The subagent events recorded for `call_index` in the current round —
/// everything since the last assistant turn, because earlier rounds reuse
/// the same call indices.
fn subagent_resume_events(events: &[AgentLoopEvent], call_index: usize) -> Vec<AgentLoopEvent> {
    let round_start = events
        .iter()
        .rposition(|event| matches!(event, AgentLoopEvent::Assistant { .. }))
        .map(|index| index + 1)
        .unwrap_or(0);
    events[round_start..]
        .iter()
        .filter_map(|event| match event {
            AgentLoopEvent::Subagent {
                call_index: at,
                event,
            } if *at == call_index => Some((**event).clone()),
            _ => None,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
async fn execute_agent_loop_tool(
    call: &ToolCall,
    call_index: usize,
    subagent_resume: Vec<AgentLoopEvent>,
    bindings: &BTreeMap<&str, &StageTool>,
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
    usage: &UsageTracker,
    approvals: &Approvals,
    options: &AgentLoopOptions<'_>,
) -> Result<String> {
    observe_loop(
        options.on_observation,
        AgentLoopObservation::ToolCall {
            name: call.function.name.clone(),
            args: arguments_preview(&call.function.arguments),
        },
    );
    tracing::info!(
        owner_kind = options.owner_kind,
        owner = options.owner,
        tool = %call.function.name,
        args = %truncate(&arguments_preview(&call.function.arguments), 200),
        "agent loop tool call"
    );

    let output = match bindings.get(call.function.name.as_str()) {
        None => format!("ERROR: unknown tool `{}`", call.function.name),
        Some(tool) => {
            let snapshots = if options.on_diff.is_some()
                && tool.effects.mutating_or_process()
                && matches!(
                    tool.binding,
                    ToolBinding::Mcp { .. } | ToolBinding::File { .. }
                ) {
                crate::diff::snapshot(&crate::diff::extract_paths(&call.function.arguments))
            } else {
                Vec::new()
            };
            let policy = CallPolicy::for_tool(
                options.require_approval,
                options.approval_effects,
                options.auto_approve,
                approvals,
                tool.effects,
            );
            // Delegations mirror their inner loop into the parent's log,
            // tagged by call index, so a crash mid-delegation resumes the
            // subagent instead of re-running it.
            let outer_sink = options.on_event;
            let wrap = move |event: AgentLoopEvent| {
                if let Some(sink) = outer_sink {
                    sink(AgentLoopEvent::Subagent {
                        call_index,
                        event: Box::new(event),
                    });
                }
            };
            let subagent = SubagentCheckpoint {
                resume_events: subagent_resume,
                on_event: outer_sink
                    .is_some()
                    .then_some(&wrap as &(dyn Fn(AgentLoopEvent) + Send + Sync)),
            };
            match dispatch_tool_call(
                &tool.binding,
                &call.function.arguments,
                config,
                mcp,
                http,
                usage,
                options.depth,
                &policy,
                subagent,
            )
            .await
            {
                Ok(output) => {
                    if let Some(on_diff) = options.on_diff {
                        for entry in crate::diff::collect_changes(&call.function.name, snapshots) {
                            on_diff(entry);
                        }
                    }
                    output
                }
                Err(error) if options.tool_errors_as_results => format!("ERROR: {error:#}"),
                Err(error) => return Err(error),
            }
        }
    };

    let preview = output
        .lines()
        .next()
        .unwrap_or("")
        .chars()
        .take(100)
        .collect();
    observe_loop(
        options.on_observation,
        AgentLoopObservation::ToolDone { preview },
    );
    tracing::debug!(
        owner_kind = options.owner_kind,
        owner = options.owner,
        tool = %call.function.name,
        output = %truncate(&output, 500),
        "agent loop tool result"
    );
    Ok(output)
}

/// Canonical model/tool state machine shared by stages, configured
/// subagents, and interactive chat turns.
#[allow(clippy::too_many_arguments)]
pub async fn run_agent_loop(
    client: &ModelClient,
    tools: &[StageTool],
    initial_messages: Vec<Message>,
    resume_events: &[AgentLoopEvent],
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
    usage: &UsageTracker,
    approvals: &Approvals,
    options: AgentLoopOptions<'_>,
) -> Result<AgentLoopResult> {
    let mut definitions: Vec<ToolDefinition> =
        tools.iter().map(|tool| tool.definition.clone()).collect();
    if !options.reprompt_targets.is_empty() {
        definitions.push(reprompt_tool(options.reprompt_targets));
    }
    let bindings: BTreeMap<&str, &StageTool> = tools
        .iter()
        .map(|tool| (tool.definition.name.as_str(), tool))
        .collect();
    let mut events = resume_events.to_vec();
    if events.is_empty() {
        record_loop_event(
            &mut events,
            options.on_event,
            AgentLoopEvent::Started {
                system: options.system.map(str::to_string),
                messages: initial_messages,
            },
        );
    }

    loop {
        let state = replay_agent_loop(&events)?;
        if let Some(outcome) = state.finished {
            // A final response checkpointed just before a crash was already
            // paid for but was not committed at the stage boundary. Replay
            // its saved text once for this new CLI invocation; freshly
            // generated responses have already streamed and are not echoed.
            if events.len() == resume_events.len()
                && let StageOutcome::Final(content) = &outcome
                && let Some(handler) = options.on_delta
                && !content.is_empty()
            {
                handler(content);
                if options.terminate_streamed_response {
                    handler("\n");
                }
            }
            return Ok(AgentLoopResult {
                outcome,
                messages: state.messages,
                usage: state.usage,
            });
        }

        if let Some(round) = state.pending {
            let missing: Vec<(usize, ToolCall)> = round
                .calls
                .iter()
                .enumerate()
                .filter(|(index, _)| !round.results.contains_key(index))
                .map(|(index, call)| (index, call.clone()))
                .collect();
            let missing_calls: Vec<ToolCall> =
                missing.iter().map(|(_, call)| call.clone()).collect();
            let run_parallel =
                parallel_round(config.settings.parallel_tools, &missing_calls, |name| {
                    bindings.get(name).map(|tool| {
                        tool.effects.parallel_safe()
                            && !CallPolicy::approval_required(
                                options.require_approval,
                                options.approval_effects,
                                tool.effects,
                            )
                    })
                });

            if run_parallel {
                tracing::info!(
                    owner_kind = options.owner_kind,
                    owner = options.owner,
                    calls = missing.len(),
                    "parallel agent loop tool round"
                );
                let mut futures = FuturesUnordered::new();
                for (call_index, call) in &missing {
                    let resume = subagent_resume_events(&events, *call_index);
                    let bindings = &bindings;
                    let options = &options;
                    futures.push(async move {
                        (
                            *call_index,
                            execute_agent_loop_tool(
                                call, *call_index, resume, bindings, config, mcp, http, usage,
                                approvals, options,
                            )
                            .await,
                        )
                    });
                }
                while let Some((call_index, output)) = futures.next().await {
                    record_loop_event(
                        &mut events,
                        options.on_event,
                        AgentLoopEvent::ToolResult {
                            call_index,
                            content: output?,
                        },
                    );
                }
            } else {
                for (call_index, call) in missing {
                    if call.function.name == REPROMPT_TOOL && !options.reprompt_targets.is_empty() {
                        match parse_reprompt(&call.function.arguments, options.reprompt_targets) {
                            Ok(outcome) => {
                                record_loop_event(
                                    &mut events,
                                    options.on_event,
                                    AgentLoopEvent::Finished {
                                        outcome,
                                        usage: state.usage,
                                    },
                                );
                                break;
                            }
                            Err(problem) => {
                                record_loop_event(
                                    &mut events,
                                    options.on_event,
                                    AgentLoopEvent::ToolResult {
                                        call_index,
                                        content: format!("ERROR: {problem}"),
                                    },
                                );
                                continue;
                            }
                        }
                    }
                    let tool_binding = bindings.get(call.function.name.as_str());
                    let mutating = tool_binding
                        .is_some_and(|tool| tool.effects.mutating_or_process());
                    let is_delegation = tool_binding
                        .is_some_and(|tool| matches!(tool.binding, ToolBinding::Agent { .. }));
                    // Delegations are exempt from the intent rule: their own
                    // checkpointed event log records exactly what ran, and a
                    // resumed inner loop applies the intent rule recursively
                    // to its own mutating calls.
                    if mutating && !is_delegation {
                        // An intent with no result means a previous, interrupted
                        // invocation began this call: its effects may already
                        // be on disk. Re-executing a non-idempotent operation
                        // blind is worse than asking the model to verify.
                        if round.started.contains(&call_index) {
                            tracing::warn!(
                                owner_kind = options.owner_kind,
                                owner = options.owner,
                                tool = %call.function.name,
                                "interrupted mutating call not re-executed on resume"
                            );
                            observe_loop(
                                options.on_observation,
                                AgentLoopObservation::Notice(format!(
                                    "`{}` was interrupted mid-execution by the previous run —                                      asking the model to verify instead of re-running it",
                                    call.function.name
                                )),
                            );
                            record_loop_event(
                                &mut events,
                                options.on_event,
                                AgentLoopEvent::ToolResult {
                                    call_index,
                                    content: format!(
                                        "INTERRUPTED: a previous run started this `{}` call but was                                          interrupted before recording its result — it may or may not                                          have taken effect. Verify the current state (re-read the                                          file, check the command's observable effects) before                                          deciding whether to repeat it.",
                                        call.function.name
                                    ),
                                },
                            );
                            continue;
                        }
                        record_loop_event(
                            &mut events,
                            options.on_event,
                            AgentLoopEvent::ToolStarted { call_index },
                        );
                    }
                    let resume = subagent_resume_events(&events, call_index);
                    let output = execute_agent_loop_tool(
                        &call, call_index, resume, &bindings, config, mcp, http, usage,
                        approvals, &options,
                    )
                    .await?;
                    record_loop_event(
                        &mut events,
                        options.on_event,
                        AgentLoopEvent::ToolResult {
                            call_index,
                            content: output,
                        },
                    );
                }
            }

            if replay_agent_loop(&events)?.finished.is_none()
                && replay_agent_loop(&events)?.pending.is_none()
                && let Some(steer) = options.steer
            {
                let steered: Vec<String> = steer.lock().unwrap().drain(..).collect();
                if !steered.is_empty() {
                    observe_loop(
                        options.on_observation,
                        AgentLoopObservation::Notice(format!(
                            "↪ delivered {} queued message(s) to the model",
                            steered.len()
                        )),
                    );
                    for content in steered {
                        record_loop_event(
                            &mut events,
                            options.on_event,
                            AgentLoopEvent::UserMessage { content },
                        );
                    }
                }
            }
            continue;
        }

        if state.turns >= options.max_turns {
            bail!(
                "{} `{}` did not produce a final answer within {} turns (raise `max_turns` on the {} or `default_max_turns` in settings)",
                options.owner_kind,
                options.owner,
                options.max_turns,
                options.owner_kind,
            );
        }

        // Pre-flight context management: size the request BEFORE sending
        // it, and shed older tool results while it would still breach the
        // model's window — instead of letting the provider reject a doomed
        // request after the fact.
        if let Some(capacity) = declared_capacity(config, options.model_name) {
            let threshold_tokens = (capacity as f64 * config.settings.auto_compact_threshold) as u64;
            let estimate = preflight_estimate(&state, &definitions);
            if estimate >= threshold_tokens {
                let mut candidate = state.messages.clone();
                let mut shed_levels: Vec<usize> = Vec::new();
                // Escalate: keep the 2 most recent tool results, then 1,
                // then none, stopping as soon as the estimate fits.
                for keep_recent in [2usize, 1, 0] {
                    if shed_context(&mut candidate, keep_recent) > 0 {
                        shed_levels.push(keep_recent);
                    }
                    if crate::model::estimate_request_tokens(
                        state.system.as_deref(),
                        &candidate,
                        &definitions,
                    ) < threshold_tokens
                    {
                        break;
                    }
                }
                if !shed_levels.is_empty() {
                    tracing::warn!(
                        owner_kind = options.owner_kind,
                        owner = options.owner,
                        estimate,
                        capacity,
                        levels = ?shed_levels,
                        "pre-flight context pressure: truncated older tool results"
                    );
                    observe_loop(
                        options.on_observation,
                        AgentLoopObservation::Notice(format!(
                            "context estimated at {} of {} — truncated older tool result(s) before sending",
                            crate::model::fmt_tokens(estimate),
                            crate::model::fmt_tokens(capacity),
                        )),
                    );
                    for keep_recent in shed_levels {
                        record_loop_event(
                            &mut events,
                            options.on_event,
                            AgentLoopEvent::ContextShed { keep_recent },
                        );
                    }
                    // Re-replay so the request is built from the shed
                    // conversation (and the estimate base resets).
                    continue;
                }
                if estimate >= capacity {
                    // Nothing left to trim. The estimate is ±25%, so the
                    // provider stays the final arbiter — send, but say why
                    // a rejection is coming if it does.
                    tracing::warn!(
                        owner_kind = options.owner_kind,
                        owner = options.owner,
                        estimate,
                        capacity,
                        "request likely exceeds the context window and nothing is left to shed"
                    );
                    observe_loop(
                        options.on_observation,
                        AgentLoopObservation::Notice(format!(
                            "context estimated at {} — likely over the model's {} window, and no \
                             older tool results are left to truncate",
                            crate::model::fmt_tokens(estimate),
                            crate::model::fmt_tokens(capacity),
                        )),
                    );
                }
            }
        }

        let mut request = Vec::with_capacity(state.messages.len() + 1);
        if let Some(system) = &state.system {
            request.push(Message::System {
                content: system.clone(),
            });
        }
        request.extend(state.messages.iter().cloned());
        let constraints = crate::model::RequestConstraints {
            // turns counts model responses so far: 0 = the opening request.
            tool_choice: (state.turns == 0)
                .then_some(options.tool_choice)
                .flatten(),
            output_schema: options.output_schema,
        };
        let reply = client
            .complete_streamed(&request, &definitions, constraints, options.on_delta)
            .await?;
        if options.terminate_streamed_response
            && let (Some(handler), Some(content)) = (options.on_delta, reply.content.as_deref())
            && !content.is_empty()
        {
            handler("\n");
        }
        if let Some(reason) = reply.truncation.as_deref() {
            tracing::warn!(
                owner_kind = options.owner_kind,
                owner = options.owner,
                reason,
                "provider cut the response short"
            );
            observe_loop(
                options.on_observation,
                AgentLoopObservation::Notice(format!(
                    "response truncated by the provider ({reason}) — it is incomplete"
                )),
            );
        }

        if reply.tool_calls.is_empty() {
            let mut answer = reply.content.unwrap_or_default();
            // A cut-off answer must not masquerade as a complete one: the
            // marker travels with it into {{previous}} and the transcript.
            if let Some(reason) = &reply.truncation {
                answer.push_str(&format!(
                    "\n\n[warning: this response was truncated by the provider ({reason}) and is incomplete]"
                ));
            }
            record_loop_event(
                &mut events,
                options.on_event,
                AgentLoopEvent::Finished {
                    outcome: StageOutcome::Final(answer),
                    usage: reply.usage,
                },
            );
            continue;
        }

        record_loop_event(
            &mut events,
            options.on_event,
            AgentLoopEvent::Assistant {
                content: reply.content,
                tool_calls: reply.tool_calls,
                usage: reply.usage,
                reasoning: reply.reasoning,
            },
        );
    }
}

/// Run a subagent to completion on a task and return its final answer.
/// Boxed because agents may recursively spawn agents (bounded by
/// `settings.max_agent_depth` via `assemble_tools`).
#[allow(clippy::too_many_arguments)]
pub fn run_agent<'a>(
    config: &'a Config,
    agent_name: &'a str,
    task: &'a str,
    mcp: &'a McpManager,
    http: &'a reqwest::Client,
    usage: &'a UsageTracker,
    depth: u32,
    approvals: &'a Approvals,
    checkpoint: SubagentCheckpoint<'a>,
) -> std::pin::Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        let agent = config
            .agents
            .get(agent_name)
            .ok_or_else(|| anyhow!("unknown agent `{agent_name}`"))?;
        let client = build_model_client(
            config,
            &agent.model,
            agent.temperature,
            agent.max_tokens,
            http,
            usage,
        )?;

        let agent_tools = assemble_tools(&agent.tool_profile(agent_name), config, mcp, depth)?;
        let system = crate::skills::compose_system(
            config,
            &format!("agent `{agent_name}`"),
            agent.resolve_system_prompt(&config.base_dir)?,
            &agent.skills,
        )?;
        let max_turns = agent.max_turns.unwrap_or(config.settings.default_max_turns);
        let tool_choice = agent.parsed_tool_choice();
        let output_schema = agent.resolve_output_schema(&config.base_dir)?;
        tracing::info!(
            agent = %agent_name, model = %agent.model, tools = agent_tools.len(), depth,
            task = %truncate(task, 200), "running agent"
        );
        if !checkpoint.resume_events.is_empty() {
            tracing::info!(
                agent = %agent_name,
                events = checkpoint.resume_events.len(),
                "resuming interrupted delegation from its checkpointed events"
            );
        }
        let result = run_agent_loop(
            &client,
            &agent_tools,
            vec![Message::User {
                content: task.to_string(),
            }],
            &checkpoint.resume_events,
            config,
            mcp,
            http,
            usage,
            approvals,
            AgentLoopOptions {
                owner_kind: "agent",
                owner: agent_name,
                model_name: &agent.model,
                system: system.as_deref(),
                max_turns,
                depth,
                require_approval: agent.require_approval,
                approval_effects: &agent.approval_effects,
                auto_approve: &agent.auto_approve,
                tool_choice: tool_choice.as_ref(),
                output_schema: output_schema.as_ref(),
                reprompt_targets: &[],
                on_delta: None,
                terminate_streamed_response: false,
                on_diff: None,
                on_event: checkpoint.on_event,
                on_observation: None,
                steer: None,
                tool_errors_as_results: false,
            },
        )
        .await?;
        tracing::info!(agent = %agent_name, "agent complete");
        match result.outcome {
            StageOutcome::Final(output) => Ok(output),
            StageOutcome::Reprompt { .. } => unreachable!("agents have no reprompt targets"),
        }
    })
}

/// State threaded through the pipeline: the original task, the previous
/// stage's output, and every completed stage's output by name.
#[derive(Debug, Default)]
pub struct PipelineContext {
    pub input: String,
    pub previous: Option<String>,
    pub outputs: BTreeMap<String, String>,
}

impl PipelineContext {
    pub fn new(input: &str) -> Self {
        PipelineContext {
            input: input.to_string(),
            ..Default::default()
        }
    }

    pub fn record(&mut self, stage_name: &str, output: String) {
        self.outputs.insert(stage_name.to_string(), output.clone());
        self.previous = Some(output);
    }
}

/// Build a canonical model client for a named model, with optional
/// caller-level sampling overrides taking precedence over each model's
/// defaults. The client's targets are the model followed by its fallback
/// chain, resolved breadth-first with cycles ignored.
pub fn build_model_client(
    config: &Config,
    model_name: &str,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    http: &reqwest::Client,
    usage: &UsageTracker,
) -> Result<ModelClient> {
    let mut targets = Vec::new();
    let mut adapters: BTreeMap<String, Arc<dyn ProviderAdapter>> = BTreeMap::new();
    let mut queue = std::collections::VecDeque::from([model_name.to_string()]);
    let mut seen = std::collections::BTreeSet::new();
    while let Some(name) = queue.pop_front() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let model = config
            .models
            .get(&name)
            .ok_or_else(|| anyhow!("unknown model `{name}`"))?;
        let provider = config
            .providers
            .get(&model.provider)
            .ok_or_else(|| anyhow!("unknown provider `{}`", model.provider))?;
        let adapter = match adapters.entry(model.provider.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.get().clone(),
            std::collections::btree_map::Entry::Vacant(entry) => entry
                .insert(crate::providers::build_adapter(provider, http.clone())?)
                .clone(),
        };
        targets.push(ModelTarget {
            model: model.model.clone(),
            label: name,
            sampling: SamplingParams {
                temperature: temperature.or(model.temperature),
                top_p: model.top_p,
                max_tokens: max_tokens.or(model.max_tokens),
            },
            stream: provider.stream,
            pricing: ModelPricing {
                input_per_million: model.input_cost_per_million,
                output_per_million: model.output_cost_per_million,
                cached_input_per_million: model.cached_input_cost_per_million,
            },
            external: provider.data_boundary == DataBoundary::External,
            adapter,
        });
        queue.extend(model.fallback.iter().cloned());
    }

    Ok(ModelClient::new(
        targets,
        config.settings.provider_retries,
        usage.clone(),
    ))
}

/// Receives a [`crate::diff::DiffEntry`] for each file change a stage's
/// write tools make, when the caller wants them (the chat TUI's diff
/// viewer does; plain CLI runs pass None).
pub type DiffSink<'a> = Option<&'a (dyn Fn(crate::diff::DiffEntry) + Send + Sync)>;

/// Run one stage to completion. `reprompt_targets` are the stages the model
/// may hand control to via `reprompt_stage` (empty = tool not offered, as in
/// chat mode and single-stage runs). `on_delta` receives streamed content
/// fragments for live display; `on_diff` receives captured file changes.
#[allow(clippy::too_many_arguments)]
pub async fn run_stage(
    config: &Config,
    stage: &Stage,
    is_first: bool,
    context: &PipelineContext,
    mcp: &McpManager,
    http: &reqwest::Client,
    usage: &UsageTracker,
    resume_events: &[AgentLoopEvent],
    on_event: LoopEventSink<'_>,
    reprompt_targets: &[String],
    on_delta: Option<crate::model::DeltaHandler<'_>>,
    on_diff: DiffSink<'_>,
    approvals: &Approvals,
) -> Result<StageOutcome> {
    let client = build_model_client(
        config,
        &stage.model,
        stage.temperature,
        stage.max_tokens,
        http,
        usage,
    )?;

    let stage_tools = assemble_tools(&stage.tool_profile(), config, mcp, 0)?;
    let user_prompt = render_template(
        &stage.prompt_template(is_first),
        &context.input,
        context.previous.as_deref(),
        &context.outputs,
    )?;

    let system = crate::skills::compose_system(
        config,
        &format!("stage `{}`", stage.name),
        stage.resolve_system_prompt(&config.base_dir)?,
        &stage.skills,
    )?;
    let max_turns = stage.max_turns.unwrap_or(config.settings.default_max_turns);
    let tool_choice = stage.parsed_tool_choice();
    let output_schema = stage.resolve_output_schema(&config.base_dir)?;
    tracing::info!(
        stage = %stage.name, model = %stage.model,
        tools = stage_tools.len() + usize::from(!reprompt_targets.is_empty()),
        mode = ?stage.mode, "running stage"
    );
    let result = run_agent_loop(
        &client,
        &stage_tools,
        vec![Message::User {
            content: user_prompt,
        }],
        resume_events,
        config,
        mcp,
        http,
        usage,
        approvals,
        AgentLoopOptions {
            owner_kind: "stage",
            owner: &stage.name,
            model_name: &stage.model,
            system: system.as_deref(),
            max_turns,
            depth: 0,
            require_approval: stage.require_approval,
            approval_effects: &stage.approval_effects,
            auto_approve: &stage.auto_approve,
            tool_choice: tool_choice.as_ref(),
            output_schema: output_schema.as_ref(),
            reprompt_targets,
            on_delta,
            terminate_streamed_response: true,
            on_diff,
            on_event,
            on_observation: None,
            steer: None,
            tool_errors_as_results: false,
        },
    )
    .await?;
    tracing::info!(stage = %stage.name, "stage complete");
    Ok(result.outcome)
}

/// If the last reply's usage puts the conversation over the auto-compact
/// threshold of the model's declared context window, return
/// `(used, capacity)`.
/// The model's declared context window, when pre-flight management is
/// enabled (`auto_compact_threshold` > 0 and `context_tokens` declared).
fn declared_capacity(config: &Config, model_name: &str) -> Option<u64> {
    if config.settings.auto_compact_threshold <= 0.0 {
        return None;
    }
    config.models.get(model_name)?.context_tokens
}

/// Size the request about to be sent. Provider-reported usage covers
/// everything up to and including the last assistant response, so only
/// the suffix appended since (tool results, steered messages) needs the
/// character estimate — unless shedding invalidated the measurement, in
/// which case the whole conversation is estimated.
fn preflight_estimate(state: &ReplayedAgentLoop, definitions: &[ToolDefinition]) -> u64 {
    if let Some(usage) = state.usage
        && !state.shed_since_usage
    {
        let suffix_start = state
            .messages
            .iter()
            .rposition(|message| matches!(message, Message::Assistant { .. }))
            .map(|index| index + 1)
            .unwrap_or(0);
        return usage.context_tokens()
            + crate::model::estimate_tokens(&state.messages[suffix_start..]);
    }
    crate::model::estimate_request_tokens(state.system.as_deref(), &state.messages, definitions)
}

/// Free context by truncating older tool results in place, keeping the
/// most recent `keep_recent` intact. Returns how many were trimmed.
/// Deterministic (no model call), so it's safe mid-stage.
pub fn shed_context(messages: &mut [Message], keep_recent: usize) -> usize {
    const MARKER: &str = "[trimmed: an earlier tool result was truncated to save context]";
    let tool_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m, Message::Tool { .. }))
        .map(|(i, _)| i)
        .collect();
    if tool_indices.len() <= keep_recent {
        return 0;
    }
    let mut trimmed = 0;
    for &index in &tool_indices[..tool_indices.len() - keep_recent] {
        if let Message::Tool { content, .. } = &mut messages[index] {
            if content.starts_with(MARKER) || content.chars().count() <= 400 {
                continue;
            }
            let head: String = content.chars().take(200).collect();
            *content = format!("{MARKER}\n{head}…");
            trimmed += 1;
        }
    }
    trimmed
}

fn parse_reprompt(arguments: &Value, targets: &[String]) -> Result<StageOutcome, String> {
    #[derive(serde::Deserialize)]
    struct RepromptArgs {
        stage: String,
        instructions: String,
    }
    let args: RepromptArgs = serde_json::from_value(arguments.clone())
        .map_err(|e| format!("reprompt_stage arguments were not valid JSON: {e}"))?;
    if !targets.contains(&args.stage) {
        return Err(format!(
            "cannot reprompt stage `{}` — allowed targets: {}",
            args.stage,
            targets.join(", ")
        ));
    }
    if args.instructions.trim().is_empty() {
        return Err("`instructions` must not be empty".to_string());
    }
    Ok(StageOutcome::Reprompt {
        target: args.stage,
        instructions: args.instructions,
    })
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let prefix: String = text.chars().take(max_chars).collect();
        format!("{prefix}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ResumeAdapter {
        requests: std::sync::Mutex<Vec<Vec<Message>>>,
    }

    impl ProviderAdapter for ResumeAdapter {
        fn name(&self) -> &'static str {
            "resume-test"
        }

        fn complete<'a>(
            &'a self,
            request: crate::model::ModelRequest<'a>,
            _on_delta: Option<crate::model::DeltaHandler<'a>>,
        ) -> crate::model::AdapterFuture<'a> {
            Box::pin(async move {
                self.requests
                    .lock()
                    .unwrap()
                    .push(request.messages.to_vec());
                Ok(crate::model::ModelResponse {
                    content: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    reasoning: None,
                    usage: Some(Usage {
                        prompt_tokens: 20,
                        completion_tokens: 1,
                        ..Usage::default()
                    }),
                    truncation: None,
                })
            })
        }
    }

    fn tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            function: crate::model::FunctionCall {
                name: name.to_string(),
                arguments: serde_json::json!({}),
            },
        }
    }

    #[test]
    fn salvage_closes_interrupted_tool_rounds() {
        // Nothing recorded (workflow cancels, pre-model cancels) — nothing
        // to salvage.
        assert!(salvage_cancelled_loop(&[]).unwrap().is_none());

        let events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![tool_call("c1", "write_file"), tool_call("c2", "shell")],
                usage: None,
                reasoning: None,
            },
            // The turn is interrupted with only the first result recorded.
            AgentLoopEvent::ToolResult {
                call_index: 0,
                content: "wrote `x`".to_string(),
            },
        ];
        let messages = salvage_cancelled_loop(&events).unwrap().unwrap();
        assert_eq!(messages.len(), 4); // user, assistant, recorded + synthetic results
        assert!(matches!(
            &messages[1],
            Message::Assistant { tool_calls: Some(calls), .. } if calls.len() == 2
        ));
        let Message::Tool {
            content,
            tool_call_id,
        } = &messages[2]
        else {
            panic!("expected the recorded tool result");
        };
        assert_eq!((content.as_str(), tool_call_id.as_str()), ("wrote `x`", "c1"));
        let Message::Tool {
            content,
            tool_call_id,
        } = &messages[3]
        else {
            panic!("expected a synthetic result for the unfinished call");
        };
        assert_eq!(tool_call_id, "c2");
        assert!(content.contains("cancelled"), "{content}");
    }

    #[tokio::test]
    async fn tool_choice_constrains_only_the_opening_request() {
        /// Returns a tool call on the first request, a final answer after,
        /// recording each request's tool_choice.
        struct ChoiceAdapter {
            choices: std::sync::Mutex<Vec<Option<crate::model::ToolChoice>>>,
        }
        impl ProviderAdapter for ChoiceAdapter {
            fn name(&self) -> &'static str {
                "choice-test"
            }
            fn complete<'a>(
                &'a self,
                request: crate::model::ModelRequest<'a>,
                _on_delta: Option<crate::model::DeltaHandler<'a>>,
            ) -> crate::model::AdapterFuture<'a> {
                Box::pin(async move {
                    let mut choices = self.choices.lock().unwrap();
                    choices.push(request.constraints.tool_choice.cloned());
                    let first = choices.len() == 1;
                    // The schema must ride along on every request.
                    assert!(request.constraints.output_schema.is_some());
                    Ok(crate::model::ModelResponse {
                        content: (!first).then(|| "done".to_string()),
                        reasoning: None,
                        tool_calls: if first {
                            vec![tool_call("c1", "missing_tool")]
                        } else {
                            Vec::new()
                        },
                        usage: None,
                        truncation: None,
                    })
                })
            }
        }

        let config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [[stage]]
            name = "s"
            model = "m"
            "#,
        )
        .unwrap();
        let adapter = Arc::new(ChoiceAdapter {
            choices: std::sync::Mutex::new(Vec::new()),
        });
        let usage = UsageTracker::unlimited();
        let client = ModelClient::new(
            vec![ModelTarget {
                label: "m".to_string(),
                model: "x".to_string(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing::default(),
                external: false,
                adapter: adapter.clone(),
            }],
            0,
            usage.clone(),
        );
        let choice = crate::model::ToolChoice::Any;
        let schema = serde_json::json!({"type": "object"});
        let mcp = McpManager::default();
        let approvals = Approvals::non_interactive();
        let result = run_agent_loop(
            &client,
            &[],
            vec![Message::User {
                content: "task".to_string(),
            }],
            &[],
            &config,
            &mcp,
            &reqwest::Client::new(),
            &usage,
            &approvals,
            AgentLoopOptions {
                owner_kind: "stage",
                owner: "s",
                model_name: "m",
                system: None,
                max_turns: 3,
                depth: 0,
                require_approval: false,
                approval_effects: &[],
                auto_approve: &[],
                tool_choice: Some(&choice),
                output_schema: Some(&schema),
                reprompt_targets: &[],
                on_delta: None,
                terminate_streamed_response: false,
                on_diff: None,
                on_event: None,
                on_observation: None,
                steer: None,
                tool_errors_as_results: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.outcome, StageOutcome::Final("done".to_string()));

        // Forced on the opening request; unconstrained afterwards so the
        // model can produce the final answer.
        let choices = adapter.choices.lock().unwrap();
        assert_eq!(
            *choices,
            vec![Some(crate::model::ToolChoice::Any), None]
        );
    }

    #[test]
    fn subagent_events_extract_scoped_to_the_current_round() {
        let sub = |call_index: usize, content: &str| AgentLoopEvent::Subagent {
            call_index,
            event: Box::new(AgentLoopEvent::UserMessage {
                content: content.to_string(),
            }),
        };
        let assistant = |name: &str| AgentLoopEvent::Assistant {
            content: None,
            tool_calls: vec![tool_call("c", name)],
            usage: None,
            reasoning: None,
        };
        let events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            assistant("agent__helper"),
            sub(0, "old round"),
            AgentLoopEvent::ToolResult {
                call_index: 0,
                content: "done".to_string(),
            },
            // A new round reuses call index 0; extraction must not leak
            // the previous round's events into it.
            assistant("agent__helper"),
            sub(0, "current round"),
            sub(1, "other call"),
        ];
        let extracted = subagent_resume_events(&events, 0);
        assert_eq!(extracted.len(), 1);
        assert!(matches!(
            &extracted[0],
            AgentLoopEvent::UserMessage { content } if content == "current round"
        ));
        assert!(subagent_resume_events(&events, 2).is_empty());
    }

    #[tokio::test]
    async fn interrupted_delegation_resumes_from_its_checkpoint() {
        let config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [agents.helper]
            model = "m"
            description = "helps"

            [[stage]]
            name = "s"
            model = "m"
            "#,
        )
        .unwrap();
        let adapter = Arc::new(ResumeAdapter {
            requests: std::sync::Mutex::new(Vec::new()),
        });
        let usage = UsageTracker::unlimited();
        let client = ModelClient::new(
            vec![ModelTarget {
                label: "m".to_string(),
                model: "x".to_string(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing::default(),
                external: false,
                adapter: adapter.clone(),
            }],
            0,
            usage.clone(),
        );
        // A write-capable delegation, so the intent rule would previously
        // have synthesized an INTERRUPTED result instead of resuming.
        let delegation_tool = StageTool {
            definition: crate::model::ToolDefinition {
                name: "agent__helper".to_string(),
                description: "delegate".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            binding: ToolBinding::Agent {
                agent: "helper".to_string(),
            },
            effects: ToolEffects::one(ToolEffect::ProcessExecute),
        };
        let delegation_call = ToolCall {
            id: "d1".to_string(),
            function: crate::model::FunctionCall {
                name: "agent__helper".to_string(),
                arguments: serde_json::json!({"task": "sub task"}),
            },
        };
        // The previous run died after the subagent FINISHED but before the
        // parent recorded the delegation's result. Its answer is on disk.
        let events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![delegation_call],
                usage: None,
                reasoning: None,
            },
            // Legacy intent record (pre-checkpointing logs have these for
            // delegations) — must not trigger the verify-first synthetic.
            AgentLoopEvent::ToolStarted { call_index: 0 },
            AgentLoopEvent::Subagent {
                call_index: 0,
                event: Box::new(AgentLoopEvent::Started {
                    system: None,
                    messages: vec![Message::User {
                        content: "sub task".to_string(),
                    }],
                }),
            },
            AgentLoopEvent::Subagent {
                call_index: 0,
                event: Box::new(AgentLoopEvent::Finished {
                    outcome: StageOutcome::Final("the sub answer".to_string()),
                    usage: None,
                }),
            },
        ];
        let recorded: std::sync::Mutex<Vec<AgentLoopEvent>> = std::sync::Mutex::new(Vec::new());
        let sink = |event: AgentLoopEvent| recorded.lock().unwrap().push(event);
        let mcp = McpManager::default();
        let approvals = Approvals::non_interactive();
        let result = run_agent_loop(
            &client,
            std::slice::from_ref(&delegation_tool),
            Vec::new(),
            &events,
            &config,
            &mcp,
            &reqwest::Client::new(),
            &usage,
            &approvals,
            AgentLoopOptions {
                owner_kind: "stage",
                owner: "s",
                model_name: "m",
                system: None,
                max_turns: 3,
                depth: 0,
                require_approval: false,
                approval_effects: &[],
                auto_approve: &[],
                tool_choice: None,
                output_schema: None,
                reprompt_targets: &[],
                on_delta: None,
                terminate_streamed_response: false,
                on_diff: None,
                on_event: Some(&sink),
                on_observation: None,
                steer: None,
                tool_errors_as_results: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.outcome, StageOutcome::Final("done".to_string()));

        // The delegation's checkpointed answer was recovered — no synthetic
        // INTERRUPTED result, no re-run of the subagent's model calls.
        let requests = adapter.requests.lock().unwrap();
        assert_eq!(requests.len(), 1, "only the parent's follow-up request");
        let tool_content = requests[0]
            .iter()
            .find_map(|message| match message {
                Message::Tool { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .unwrap();
        assert_eq!(tool_content, "the sub answer");
        // The resumed-and-already-finished delegation emitted no new
        // subagent events.
        assert!(
            !recorded
                .lock()
                .unwrap()
                .iter()
                .any(|event| matches!(event, AgentLoopEvent::Subagent { .. })),
        );
    }

    #[test]
    fn replay_tracks_context_shed_staleness() {
        let mut events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![tool_call("c1", "read")],
                usage: Some(Usage {
                    prompt_tokens: 50,
                    completion_tokens: 5,
                    ..Usage::default()
                }),
                reasoning: None,
            },
            AgentLoopEvent::ToolResult {
                call_index: 0,
                content: "x".repeat(600),
            },
            AgentLoopEvent::ContextShed { keep_recent: 0 },
        ];
        // A shed after the last measurement makes the usage stale…
        let replayed = replay_agent_loop(&events).unwrap();
        assert!(replayed.shed_since_usage);

        // …and the next measured response resets it.
        events.push(AgentLoopEvent::Assistant {
            content: None,
            tool_calls: vec![tool_call("c2", "read")],
            usage: Some(Usage {
                prompt_tokens: 60,
                completion_tokens: 5,
                ..Usage::default()
            }),
            reasoning: None,
        });
        let replayed = replay_agent_loop(&events).unwrap();
        assert!(!replayed.shed_since_usage);
    }

    #[tokio::test]
    async fn preflight_sheds_context_before_sending_a_doomed_request() {
        let config: Config = toml::from_str(
            r#"
            [settings]
            auto_compact_threshold = 0.8

            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"
            context_tokens = 1000
            "#,
        )
        .unwrap();
        let adapter = Arc::new(ResumeAdapter {
            requests: std::sync::Mutex::new(Vec::new()),
        });
        let usage = UsageTracker::unlimited();
        let client = ModelClient::new(
            vec![ModelTarget {
                label: "m".to_string(),
                model: "x".to_string(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing::default(),
                external: false,
                adapter: adapter.clone(),
            }],
            0,
            usage.clone(),
        );
        // The last measured context (900) plus a 6000-char tool result
        // estimates far beyond 80% of the 1000-token window.
        let events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![tool_call("c1", "removed_tool")],
                usage: Some(Usage {
                    prompt_tokens: 800,
                    completion_tokens: 100,
                    ..Usage::default()
                }),
                reasoning: None,
            },
            AgentLoopEvent::ToolResult {
                call_index: 0,
                content: "x".repeat(6000),
            },
        ];
        let recorded: std::sync::Mutex<Vec<AgentLoopEvent>> = std::sync::Mutex::new(Vec::new());
        let sink = |event: AgentLoopEvent| recorded.lock().unwrap().push(event);
        let mcp = McpManager::default();
        let approvals = Approvals::non_interactive();
        let result = run_agent_loop(
            &client,
            &[],
            Vec::new(),
            &events,
            &config,
            &mcp,
            &reqwest::Client::new(),
            &usage,
            &approvals,
            AgentLoopOptions {
                owner_kind: "stage",
                owner: "s",
                model_name: "m",
                system: None,
                max_turns: 3,
                depth: 0,
                require_approval: false,
                approval_effects: &[],
                auto_approve: &[],
                tool_choice: None,
                output_schema: None,
                reprompt_targets: &[],
                on_delta: None,
                terminate_streamed_response: false,
                on_diff: None,
                on_event: Some(&sink),
                on_observation: None,
                steer: None,
                tool_errors_as_results: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.outcome, StageOutcome::Final("done".to_string()));

        // The shed was recorded BEFORE the request went out…
        assert!(
            recorded
                .lock()
                .unwrap()
                .iter()
                .any(|event| matches!(event, AgentLoopEvent::ContextShed { .. })),
            "expected a pre-flight ContextShed event"
        );
        // …so the model never saw the full 6000-char result.
        let requests = adapter.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let tool_content = requests[0]
            .iter()
            .find_map(|message| match message {
                Message::Tool { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .expect("the tool result is still present");
        assert!(tool_content.starts_with("[trimmed"), "{tool_content}");
        assert!(tool_content.len() < 400, "was actually truncated");
    }

    #[test]
    fn agent_event_log_replays_partial_parallel_round_in_call_order() {
        let mut events = vec![
            AgentLoopEvent::Started {
                system: Some("system".to_string()),
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![tool_call("first", "read_a"), tool_call("second", "read_b")],
                usage: Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 2,
                    ..Usage::default()
                }),
                reasoning: None,
            },
            // Parallel completion order is intentionally reversed.
            AgentLoopEvent::ToolResult {
                call_index: 1,
                content: "b".to_string(),
            },
        ];
        let partial = replay_agent_loop(&events).unwrap();
        assert_eq!(partial.turns, 1);
        assert_eq!(partial.messages.len(), 2);
        assert_eq!(partial.pending.unwrap().results.len(), 1);

        events.push(AgentLoopEvent::ToolResult {
            call_index: 0,
            content: "a".to_string(),
        });
        let complete = replay_agent_loop(&events).unwrap();
        assert!(complete.pending.is_none());
        let ids: Vec<&str> = complete
            .messages
            .iter()
            .filter_map(|message| match message {
                Message::Tool { tool_call_id, .. } => Some(tool_call_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(ids, vec!["first", "second"]);
    }

    #[test]
    fn agent_event_log_finished_response_is_immediately_resumable() {
        let events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Finished {
                outcome: StageOutcome::Final("done".to_string()),
                usage: Some(Usage {
                    prompt_tokens: 4,
                    completion_tokens: 1,
                    ..Usage::default()
                }),
            },
        ];
        let replayed = replay_agent_loop(&events).unwrap();
        assert_eq!(replayed.finished, Some(StageOutcome::Final("done".into())));
        assert!(matches!(
            replayed.messages.last(),
            Some(Message::Assistant {
                content: Some(content),
                tool_calls: None,
                reasoning: None,
            }) if content == "done"
        ));
    }

    #[tokio::test]
    async fn canonical_loop_resumes_only_missing_tools_then_calls_model() {
        let config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [[stage]]
            name = "s"
            model = "m"
            "#,
        )
        .unwrap();
        let adapter = Arc::new(ResumeAdapter {
            requests: std::sync::Mutex::new(Vec::new()),
        });
        let usage = UsageTracker::unlimited();
        let client = ModelClient::new(
            vec![ModelTarget {
                label: "m".to_string(),
                model: "x".to_string(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing::default(),
                external: false,
                adapter: adapter.clone(),
            }],
            0,
            usage.clone(),
        );
        let events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![
                    tool_call("first", "removed_tool_a"),
                    tool_call("second", "removed_tool_b"),
                ],
                usage: Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 2,
                    ..Usage::default()
                }),
                reasoning: None,
            },
            AgentLoopEvent::ToolResult {
                call_index: 1,
                content: "saved second result".to_string(),
            },
        ];
        let mcp = McpManager::default();
        let approvals = Approvals::non_interactive();
        let result = run_agent_loop(
            &client,
            &[],
            Vec::new(),
            &events,
            &config,
            &mcp,
            &reqwest::Client::new(),
            &usage,
            &approvals,
            AgentLoopOptions {
                owner_kind: "stage",
                owner: "s",
                model_name: "m",
                system: None,
                max_turns: 2,
                depth: 0,
                require_approval: false,
                approval_effects: &[],
                auto_approve: &[],
                tool_choice: None,
                output_schema: None,
                reprompt_targets: &[],
                on_delta: None,
                terminate_streamed_response: false,
                on_diff: None,
                on_event: None,
                on_observation: None,
                steer: None,
                tool_errors_as_results: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.outcome, StageOutcome::Final("done".to_string()));
        let requests = adapter.requests.lock().unwrap();
        assert_eq!(
            requests.len(),
            1,
            "the saved assistant turn was not repeated"
        );
        let tools: Vec<(&str, &str)> = requests[0]
            .iter()
            .filter_map(|message| match message {
                Message::Tool {
                    content,
                    tool_call_id,
                } => Some((tool_call_id.as_str(), content.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(tools[0].0, "first");
        assert!(tools[0].1.contains("unknown tool"));
        assert_eq!(tools[1], ("second", "saved second result"));
    }

    #[tokio::test]
    async fn interrupted_mutating_calls_are_not_reexecuted_on_resume() {
        let config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [[stage]]
            name = "s"
            model = "m"
            "#,
        )
        .unwrap();
        let adapter = Arc::new(ResumeAdapter {
            requests: std::sync::Mutex::new(Vec::new()),
        });
        let usage = UsageTracker::unlimited();
        let client = ModelClient::new(
            vec![ModelTarget {
                label: "m".to_string(),
                model: "x".to_string(),
                sampling: SamplingParams::default(),
                stream: false,
                pricing: ModelPricing::default(),
                external: false,
                adapter: adapter.clone(),
            }],
            0,
            usage.clone(),
        );
        let shell_tool = StageTool {
            definition: tools::shell_definition(5, &[]),
            binding: ToolBinding::Shell { allow: Vec::new() },
            effects: ToolEffects::one(ToolEffect::ProcessExecute),
        };
        let shell_call = |id: &str, command: &str| ToolCall {
            id: id.to_string(),
            function: crate::model::FunctionCall {
                name: "shell".to_string(),
                arguments: serde_json::json!({ "command": command }),
            },
        };
        let events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![
                    shell_call("c1", "echo one"),
                    shell_call("c2", "echo two"),
                ],
                usage: None,
                reasoning: None,
            },
            // The previous run began c1 (intent recorded) and died before
            // its result reached disk; c2 was never started.
            AgentLoopEvent::ToolStarted { call_index: 0 },
        ];
        let recorded: std::sync::Mutex<Vec<AgentLoopEvent>> = std::sync::Mutex::new(Vec::new());
        let sink = |event: AgentLoopEvent| recorded.lock().unwrap().push(event);
        let mcp = McpManager::default();
        let approvals = Approvals::non_interactive();
        let result = run_agent_loop(
            &client,
            std::slice::from_ref(&shell_tool),
            Vec::new(),
            &events,
            &config,
            &mcp,
            &reqwest::Client::new(),
            &usage,
            &approvals,
            AgentLoopOptions {
                owner_kind: "stage",
                owner: "s",
                model_name: "m",
                system: None,
                max_turns: 2,
                depth: 0,
                require_approval: false,
                approval_effects: &[],
                auto_approve: &[],
                tool_choice: None,
                output_schema: None,
                reprompt_targets: &[],
                on_delta: None,
                terminate_streamed_response: false,
                on_diff: None,
                on_event: Some(&sink),
                on_observation: None,
                steer: None,
                tool_errors_as_results: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(result.outcome, StageOutcome::Final("done".to_string()));

        let requests = adapter.requests.lock().unwrap();
        let tools: Vec<(&str, &str)> = requests[0]
            .iter()
            .filter_map(|message| match message {
                Message::Tool {
                    content,
                    tool_call_id,
                } => Some((tool_call_id.as_str(), content.as_str())),
                _ => None,
            })
            .collect();
        // The started-but-unfinished call gets a verify-first synthetic
        // result instead of running `echo one` again…
        assert_eq!(tools[0].0, "c1");
        assert!(tools[0].1.contains("INTERRUPTED"), "{}", tools[0].1);
        assert!(!tools[0].1.contains("stdout"), "{}", tools[0].1);
        // …while the never-started call executes normally, with a fresh
        // intent recorded before it ran.
        assert_eq!(tools[1].0, "c2");
        assert!(tools[1].1.contains("two"), "{}", tools[1].1);
        let recorded = recorded.lock().unwrap();
        assert!(
            recorded
                .iter()
                .any(|event| matches!(event, AgentLoopEvent::ToolStarted { call_index: 1 })),
            "the fresh mutating call must record an intent before executing"
        );
        assert!(
            !recorded
                .iter()
                .any(|event| matches!(event, AgentLoopEvent::ToolStarted { call_index: 0 })),
            "the interrupted call must not record a second intent"
        );
    }

    #[test]
    fn renders_all_placeholder_kinds() {
        let mut outputs = BTreeMap::new();
        outputs.insert("plan".to_string(), "the plan".to_string());
        let rendered = render_template(
            "task: {{input}}; prev: {{previous}}; plan: {{ stage.plan }}",
            "do it",
            Some("prior"),
            &outputs,
        )
        .unwrap();
        assert_eq!(rendered, "task: do it; prev: prior; plan: the plan");
    }

    #[test]
    fn missing_stage_output_errors() {
        let err = render_template("{{stage.nope}}", "x", None, &BTreeMap::new()).unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn agent_tools_gated_by_mode_and_depth() {
        let config: Config = toml::from_str(
            r#"
            [settings]
            searxng_url = "http://localhost:8888"

            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [agents.reader]
            model = "m"
            description = "reads things"
            web_search = true

            [agents.writer]
            model = "m"
            mode = "read_write"

            [agents.sheller]
            model = "m"
            shell = true

            [[stage]]
            name = "s"
            model = "m"
            subagents = ["reader", "writer", "sheller"]
            "#,
        )
        .unwrap();
        let mcp = McpManager::default();

        // Network egress remains read-like, but write mode and process
        // execution cannot cross a read-only delegation boundary.
        let names: Vec<String> = assemble_tools(&config.stages[0].tool_profile(), &config, &mcp, 0)
            .unwrap()
            .into_iter()
            .map(|t| t.definition.name)
            .collect();
        assert_eq!(names, vec!["agent__reader"]);

        // At the depth cap (default max_agent_depth = 2) no agents are offered.
        let at_cap = assemble_tools(&config.stages[0].tool_profile(), &config, &mcp, 2).unwrap();
        assert!(at_cap.is_empty());
    }

    #[test]
    fn web_fetch_is_assembled_from_the_profile_flag() {
        let config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [[stage]]
            name = "s"
            model = "m"
            web_fetch = true
            "#,
        )
        .unwrap();
        let mcp = McpManager::default();
        let tools = assemble_tools(&config.stages[0].tool_profile(), &config, &mcp, 0).unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].definition.name, "web_fetch");
        // Egress is read-like: available to read-only stages, gateable via
        // approval_effects = ["network_egress"].
        assert!(!tools[0].effects.mutating_or_process());
        assert!(tools[0].effects.intersects(&[ToolEffect::NetworkEgress]));

        let descriptor = call_descriptor(
            &ToolBinding::WebFetch,
            &serde_json::json!({"url": "https://example.com/doc"}),
        );
        assert_eq!(descriptor.descriptor, "web_fetch https://example.com/doc");
        assert_eq!(descriptor.always_pattern, "web_fetch *");
    }

    #[test]
    fn fallback_chain_resolves_breadth_first_and_cuts_cycles() {
        let config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.a]
            provider = "p"
            model = "a-id"
            fallback = ["b", "c"]

            [models.b]
            provider = "p"
            model = "b-id"
            fallback = ["d", "a"]

            [models.c]
            provider = "p"
            model = "c-id"

            [models.d]
            provider = "p"
            model = "d-id"

            [[stage]]
            name = "s"
            model = "a"
            "#,
        )
        .unwrap();
        let http = reqwest::Client::new();
        let usage = UsageTracker::unlimited();
        let client = build_model_client(&config, "a", None, None, &http, &usage).unwrap();
        // a's own fallbacks first, then theirs; the cycle back to `a` is cut.
        assert_eq!(client.target_labels(), vec!["a", "b", "c", "d"]);
        // A model with no fallbacks yields a single target.
        let solo = build_model_client(&config, "c", None, None, &http, &usage).unwrap();
        assert_eq!(solo.target_labels(), vec!["c"]);
    }

    #[test]
    fn parallel_round_requires_every_call_to_be_safe() {
        let call = |name: &str| crate::model::ToolCall {
            id: "x".into(),
            function: crate::model::FunctionCall {
                name: name.into(),
                arguments: serde_json::json!({}),
            },
        };
        let parallel_safe = |name: &str| match name {
            "read" => Some(true),
            "write" => Some(false),
            _ => None,
        };
        let reads = [call("read"), call("read")];
        assert!(parallel_round(true, &reads, parallel_safe));
        // Disabled by config, single call, any unsafe call, or any unknown tool.
        assert!(!parallel_round(false, &reads, parallel_safe));
        assert!(!parallel_round(true, &reads[..1], parallel_safe));
        assert!(!parallel_round(
            true,
            &[call("read"), call("write")],
            parallel_safe
        ));
        assert!(!parallel_round(
            true,
            &[call("read"), call("reprompt_stage")],
            parallel_safe
        ));
    }

    #[test]
    fn approval_policy_can_add_read_like_effects() {
        let network = ToolEffects::one(ToolEffect::NetworkEgress);
        assert!(!CallPolicy::approval_required(true, &[], network));
        assert!(CallPolicy::approval_required(
            true,
            &[ToolEffect::NetworkEgress],
            network,
        ));
        assert!(!CallPolicy::approval_required(
            false,
            &[ToolEffect::NetworkEgress],
            network,
        ));

        // Compatibility defaults always gate mutation and process execution.
        assert!(CallPolicy::approval_required(
            true,
            &[],
            ToolEffects::one(ToolEffect::FilesystemWrite),
        ));
        assert!(CallPolicy::approval_required(
            true,
            &[],
            ToolEffects::one(ToolEffect::ProcessExecute),
        ));
    }

    #[test]
    fn compound_shell_commands_cannot_use_pattern_approvals() {
        let binding = ToolBinding::Shell { allow: Vec::new() };
        let simple =
            call_descriptor(&binding, &serde_json::json!({"command": "cargo test --all"}));
        assert!(simple.pattern_safe);
        assert!(tools::wildcard_match("shell cargo *", &simple.descriptor));

        let compound = call_descriptor(
            &binding,
            &serde_json::json!({"command": "cargo test; dangerous-command"}),
        );
        assert!(!compound.pattern_safe);
        // It still textually matches the broad pattern, proving that the
        // separate pattern_safe gate is what prevents auto-approval.
        assert!(tools::wildcard_match("shell cargo *", &compound.descriptor));
    }

    #[test]
    fn file_tools_gated_by_mode() {
        let config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [[stage]]
            name = "reader"
            model = "m"
            files = true

            [[stage]]
            name = "writer"
            model = "m"
            mode = "read_write"
            files = true

            [[stage]]
            name = "none"
            model = "m"
            "#,
        )
        .unwrap();
        let mcp = McpManager::default();
        let names = |index: usize| -> Vec<String> {
            assemble_tools(&config.stages[index].tool_profile(), &config, &mcp, 0)
                .unwrap()
                .into_iter()
                .map(|t| t.definition.name)
                .collect()
        };
        assert_eq!(names(0), vec!["read_file", "list_dir", "glob", "grep"]);
        assert_eq!(
            names(1),
            vec![
                "read_file",
                "list_dir",
                "glob",
                "grep",
                "write_file",
                "edit_lines",
                "edit_file"
            ]
        );
        assert!(names(2).is_empty());

        let writer_tools =
            assemble_tools(&config.stages[1].tool_profile(), &config, &mcp, 0).unwrap();
        let read = writer_tools
            .iter()
            .find(|tool| tool.definition.name == "read_file")
            .unwrap();
        let write = writer_tools
            .iter()
            .find(|tool| tool.definition.name == "write_file")
            .unwrap();
        assert!(read.effects.contains(ToolEffect::FilesystemRead));
        assert!(write.effects.contains(ToolEffect::FilesystemWrite));
    }

    #[test]
    fn preflight_sizing_uses_real_usage_and_gates_on_config() {
        let mut config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.gauged]
            provider = "p"
            model = "x"
            context_tokens = 1000

            [models.unbounded]
            provider = "p"
            model = "y"

            [[stage]]
            name = "s"
            model = "gauged"
            "#,
        )
        .unwrap();

        // Capacity requires a declared window, a known model, and a
        // nonzero threshold.
        assert_eq!(declared_capacity(&config, "gauged"), Some(1000));
        assert_eq!(declared_capacity(&config, "unbounded"), None);
        assert_eq!(declared_capacity(&config, "missing"), None);
        config.settings.auto_compact_threshold = 0.0;
        assert_eq!(declared_capacity(&config, "gauged"), None);

        // With fresh usage: measured base plus a char estimate of the
        // suffix appended since the last assistant response.
        let events = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: vec![Message::User {
                    content: "task".to_string(),
                }],
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![tool_call("c1", "read")],
                usage: Some(Usage {
                    prompt_tokens: 500,
                    completion_tokens: 100,
                    ..Usage::default()
                }),
                reasoning: None,
            },
            AgentLoopEvent::ToolResult {
                call_index: 0,
                content: "x".repeat(4000),
            },
        ];
        let state = replay_agent_loop(&events).unwrap();
        let estimate = preflight_estimate(&state, &[]);
        // 600 measured + ~1000 for the 4000-char result (+ envelope).
        assert!((1550..1700).contains(&estimate), "{estimate}");

        // After a shed, the measurement is stale: the whole (now smaller)
        // conversation is re-estimated from characters instead.
        let mut shed_events = events;
        shed_events.push(AgentLoopEvent::ContextShed { keep_recent: 0 });
        let state = replay_agent_loop(&shed_events).unwrap();
        assert!(state.shed_since_usage);
        let after = preflight_estimate(&state, &[]);
        assert!(after < 200, "{after}");
    }

    #[test]
    fn sheds_older_tool_results_only() {
        let big = "x".repeat(1000);
        let mut messages = vec![
            Message::System {
                content: "s".into(),
            },
            Message::User {
                content: "u".into(),
            },
            Message::Tool {
                content: big.clone(),
                tool_call_id: "1".into(),
            },
            Message::Tool {
                content: big.clone(),
                tool_call_id: "2".into(),
            },
            Message::Tool {
                content: big.clone(),
                tool_call_id: "3".into(),
            },
        ];
        assert_eq!(shed_context(&mut messages, 2), 1);
        let Message::Tool { content, .. } = &messages[2] else {
            panic!()
        };
        assert!(content.starts_with("[trimmed"));
        assert!(content.len() < 400);
        // Recent two untouched.
        let Message::Tool { content, .. } = &messages[4] else {
            panic!()
        };
        assert_eq!(content, &big);
        // Idempotent: already-trimmed entries aren't re-counted.
        assert_eq!(shed_context(&mut messages, 2), 0);
        // Small results aren't worth trimming.
        let mut small = vec![
            Message::Tool {
                content: "tiny".into(),
                tool_call_id: "1".into(),
            },
            Message::Tool {
                content: big.clone(),
                tool_call_id: "2".into(),
            },
        ];
        assert_eq!(shed_context(&mut small, 1), 0);
    }

    #[test]
    fn clamps_oversized_tool_output() {
        let big = "x".repeat(50);
        let clamped = clamp_tool_output(big.clone(), 10);
        assert!(clamped.starts_with("xxxxxxxxxx\n… [tool output truncated: 50 characters"));
        // Under the limit and limit-0 (disabled) pass through unchanged.
        assert_eq!(clamp_tool_output(big.clone(), 50), big);
        assert_eq!(clamp_tool_output(big.clone(), 0), big);
    }

    #[test]
    fn sanitizes_tool_names() {
        assert_eq!(sanitize_tool_name("fs__read.file"), "fs__read-file");
        assert_eq!(sanitize_tool_name(&"x".repeat(100)).len(), 64);
    }
}
