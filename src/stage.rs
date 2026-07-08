//! Stage execution: template rendering, tool assembly (with read-only
//! filtering), and the per-stage agentic tool-call loop.

use std::collections::BTreeMap;
use std::future::Future;

use anyhow::{Context, Result, anyhow, bail};
use rmcp::model::JsonObject;
use serde_json::Value;

use crate::approval::{Approvals, Decision};
use crate::config::{Config, Mode, Stage};
use crate::mcp::McpManager;
use crate::provider::{ChatClient, ChatMessage, SamplingParams, ToolFunction};
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
    Mcp { server: String, tool: String },
    WebSearch,
    /// Delegate the call's `task` to a configured subagent.
    Agent { agent: String },
    /// Run the call's `command` with `sh -c`, restricted to the owning
    /// context's allowlist patterns (empty = unrestricted).
    Shell { allow: Vec<String> },
    /// A built-in file tool, rooted at the working directory.
    File { op: crate::files::FileOp },
}

pub struct StageTool {
    pub definition: ToolFunction,
    pub binding: ToolBinding,
    /// Whether this tool is classified read-only (used for diff capture).
    pub read_only: bool,
}

/// What a model context (stage or agent) exposes to its model: shared shape
/// for tool assembly.
pub struct ToolProfile<'a> {
    /// For log messages.
    pub owner: &'a str,
    pub mode: Mode,
    pub mcp: &'a [String],
    pub web_search: bool,
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
            subagents: &self.subagents,
            shell: self.shell,
            shell_allow: &self.shell_allow,
            files: self.files,
        }
    }
}

/// How a stage finished: with a final answer, or by handing control to
/// another stage.
#[derive(Debug)]
pub enum StageOutcome {
    Final(String),
    Reprompt { target: String, instructions: String },
}

pub const REPROMPT_TOOL: &str = "reprompt_stage";

/// The routing tool offered to stages with a non-empty `can_reprompt` list.
fn reprompt_tool(targets: &[String]) -> ToolFunction {
    ToolFunction {
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
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '-' })
        .collect();
    cleaned.chars().take(64).collect()
}

/// Assemble the tools visible to a context, applying the read-only filter.
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
            stage_tools.push(StageTool {
                definition: ToolFunction {
                    name: sanitize_tool_name(&format!("{server_name}__{}", tool.name)),
                    description: tool.description.clone().unwrap_or_default().into_owned(),
                    parameters: Value::Object((*tool.input_schema).clone()),
                },
                binding: ToolBinding::Mcp {
                    server: server_name.clone(),
                    tool: tool.name.to_string(),
                },
                read_only,
            });
        }
    }

    if profile.web_search {
        stage_tools.push(StageTool {
            definition: tools::web_search_definition(),
            binding: ToolBinding::WebSearch,
            read_only: true,
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
            binding: ToolBinding::Shell { allow: profile.shell_allow.to_vec() },
            read_only: false,
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
                read_only,
            });
        }
    }

    if depth < config.settings.max_agent_depth {
        for agent_name in profile.subagents {
            let agent = config
                .agents
                .get(agent_name)
                .ok_or_else(|| anyhow!("unknown agent `{agent_name}`"))?;
            let agent_read_only = agent.mode == Mode::ReadOnly;
            // A read-only context must not gain write access by delegating.
            if profile.mode == Mode::ReadOnly && !agent_read_only {
                tracing::debug!(
                    owner = %profile.owner, agent = %agent_name,
                    "read_write agent hidden in read_only mode"
                );
                continue;
            }
            let about = if agent.description.is_empty() {
                format!("Delegate a task to the `{agent_name}` agent.")
            } else {
                agent.description.clone()
            };
            stage_tools.push(StageTool {
                definition: ToolFunction {
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
                read_only: agent_read_only,
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

/// Approval policy for one tool call, derived from its owning context.
pub struct CallPolicy<'a> {
    /// True when the context gates writes and this tool is not read-only.
    pub require_approval: bool,
    pub auto_approve: &'a [String],
    pub approvals: &'a Approvals,
}

impl<'a> CallPolicy<'a> {
    /// Policy for a context's tool: approval applies only to
    /// non-read-only tools of contexts that opted in.
    pub fn for_tool(
        context_requires: bool,
        auto_approve: &'a [String],
        approvals: &'a Approvals,
        tool_read_only: bool,
    ) -> CallPolicy<'a> {
        CallPolicy {
            require_approval: context_requires && !tool_read_only,
            auto_approve,
            approvals,
        }
    }
}

/// True when a round's tool calls may dispatch concurrently: several
/// calls, every one resolving to a read-only tool. Writes could conflict,
/// approval prompts must stay sequential, and control tools
/// (`reprompt_stage`) or unknown names take the sequential path, which
/// knows how to answer them.
pub fn parallel_round(
    enabled: bool,
    calls: &[crate::provider::ToolCall],
    read_only_of: impl Fn(&str) -> Option<bool>,
) -> bool {
    enabled
        && calls.len() > 1
        && calls.iter().all(|call| read_only_of(&call.function.name) == Some(true))
}

/// How a call is presented to the approver and matched against patterns.
pub struct CallDescriptor {
    pub descriptor: String,
    pub detail: String,
    /// What an "always" grant would cover for the rest of the session.
    pub always_pattern: String,
}

pub fn call_descriptor(binding: &ToolBinding, arguments_json: &str) -> CallDescriptor {
    let args: Value = serde_json::from_str(arguments_json).unwrap_or(Value::Null);
    match binding {
        ToolBinding::Mcp { server, tool } => {
            let name = sanitize_tool_name(&format!("{server}__{tool}"));
            CallDescriptor {
                descriptor: name.clone(),
                detail: truncate(arguments_json, 200),
                always_pattern: name,
            }
        }
        ToolBinding::WebSearch => CallDescriptor {
            descriptor: tools::WEB_SEARCH_TOOL.to_string(),
            detail: truncate(arguments_json, 200),
            always_pattern: tools::WEB_SEARCH_TOOL.to_string(),
        },
        ToolBinding::Shell { .. } => {
            let command =
                args.get("command").and_then(Value::as_str).unwrap_or_default().trim();
            let first_word = command.split_whitespace().next().unwrap_or("?");
            CallDescriptor {
                descriptor: format!("shell {}", truncate(command, 160)),
                detail: command.to_string(),
                always_pattern: format!("shell {first_word} *"),
            }
        }
        ToolBinding::Agent { agent } => {
            let task = args.get("task").and_then(Value::as_str).unwrap_or_default();
            CallDescriptor {
                descriptor: format!("agent__{agent}"),
                detail: truncate(task, 200),
                always_pattern: format!("agent__{agent}"),
            }
        }
        ToolBinding::File { op } => {
            let path = args.get("path").and_then(Value::as_str).unwrap_or("?");
            CallDescriptor {
                descriptor: format!("{} {}", op.tool_name(), truncate(path, 160)),
                detail: truncate(arguments_json, 200),
                always_pattern: format!("{} *", op.tool_name()),
            }
        }
    }
}

/// Execute a tool call: enforce the approval policy, run it, and clamp the
/// result so a single oversized output cannot exhaust the model's context.
/// `depth` is the caller's delegation depth (0 for stages and chat).
pub async fn dispatch_tool_call(
    binding: &ToolBinding,
    arguments_json: &str,
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
    depth: u32,
    policy: &CallPolicy<'_>,
) -> Result<String> {
    let described = call_descriptor(binding, arguments_json);

    // pre_tool hooks run before the approval prompt: a call a hook is
    // going to block should never interrupt the user.
    if let Some(blocked) =
        crate::hooks::pre_tool(config, &described.descriptor, arguments_json).await
    {
        return Ok(blocked);
    }

    if policy.require_approval {
        let pre_approved = policy
            .auto_approve
            .iter()
            .any(|pattern| tools::wildcard_match(pattern, &described.descriptor))
            || policy.approvals.session_allowed(&described.descriptor);
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

    let output =
        dispatch_tool_call_inner(binding, arguments_json, config, mcp, http, depth, policy)
            .await?;
    let output =
        crate::hooks::post_tool(config, &described.descriptor, arguments_json, output).await;
    Ok(clamp_tool_output(output, config.settings.max_tool_output_chars))
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

async fn dispatch_tool_call_inner(
    binding: &ToolBinding,
    arguments_json: &str,
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
    depth: u32,
    policy: &CallPolicy<'_>,
) -> Result<String> {
    let arguments: Value = if arguments_json.trim().is_empty() {
        Value::Object(JsonObject::new())
    } else {
        match serde_json::from_str(arguments_json) {
            Ok(value) => value,
            // Feed malformed JSON back to the model instead of aborting the stage.
            Err(e) => return Ok(format!("ERROR: tool arguments were not valid JSON: {e}")),
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
            match tools::web_search(http, searxng_url, query, config.settings.searxng_max_results)
                .await
            {
                Ok(results) => Ok(results),
                Err(e) => Ok(format!("ERROR: web search failed: {e:#}")),
            }
        }
        ToolBinding::Mcp { server, tool } => {
            let Value::Object(object) = arguments else {
                return Ok("ERROR: tool arguments must be a JSON object".to_string());
            };
            let connection = mcp
                .get(server)
                .ok_or_else(|| anyhow!("mcp server `{server}` is not connected"))?;
            match connection.call(tool, object).await {
                Ok(text) => Ok(text),
                // Transport-level failure: report to the model, keep the stage alive.
                Err(e) => Ok(format!("ERROR: {e:#}")),
            }
        }
        ToolBinding::Agent { agent } => {
            let task = arguments.get("task").and_then(Value::as_str).unwrap_or_default();
            if task.trim().is_empty() {
                return Ok("ERROR: the agent needs a non-empty `task` string".to_string());
            }
            match run_agent(config, agent, task, mcp, http, depth + 1, policy.approvals).await
            {
                Ok(answer) => Ok(answer),
                // The agent's failure becomes feedback, not a crashed turn.
                Err(e) => Ok(format!("ERROR: agent `{agent}` failed: {e:#}")),
            }
        }
        ToolBinding::Shell { allow } => {
            let command =
                arguments.get("command").and_then(Value::as_str).unwrap_or_default();
            if command.trim().is_empty() {
                return Ok("ERROR: `command` must be a non-empty string".to_string());
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
            tracing::info!(tool = op.tool_name(), args = %truncate(arguments_json, 200), "file tool");
            Ok(crate::files::dispatch(*op, &arguments))
        }
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
    depth: u32,
    approvals: &'a Approvals,
) -> std::pin::Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
    Box::pin(async move {
        let agent = config
            .agents
            .get(agent_name)
            .ok_or_else(|| anyhow!("unknown agent `{agent_name}`"))?;
        let client =
            build_client(config, &agent.model, agent.temperature, agent.max_tokens, http)?;

        let agent_tools = assemble_tools(&agent.tool_profile(agent_name), config, mcp, depth)?;
        let definitions: Vec<ToolFunction> =
            agent_tools.iter().map(|t| t.definition.clone()).collect();
        let bindings: BTreeMap<&str, &StageTool> = agent_tools
            .iter()
            .map(|t| (t.definition.name.as_str(), t))
            .collect();

        let system = crate::skills::compose_system(
            config,
            &format!("agent `{agent_name}`"),
            agent.resolve_system_prompt(&config.base_dir)?,
            &agent.skills,
        )?;
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(ChatMessage::System { content: system });
        }
        messages.push(ChatMessage::User { content: task.to_string() });

        let max_turns = agent.max_turns.unwrap_or(config.settings.default_max_turns);
        tracing::info!(
            agent = %agent_name, model = %agent.model, tools = definitions.len(), depth,
            task = %truncate(task, 200), "running agent"
        );

        for turn in 1..=max_turns {
            let reply = client.chat(&messages, &definitions).await?;

            if reply.tool_calls.is_empty() {
                tracing::info!(agent = %agent_name, turns = turn, "agent complete");
                return Ok(reply.content.unwrap_or_default());
            }

            if let Some((used, capacity)) =
                context_pressure(config, &agent.model, reply.usage.as_ref())
            {
                let trimmed = shed_context(&mut messages, 2);
                if trimmed > 0 {
                    tracing::warn!(
                        agent = %agent_name, used, capacity, trimmed,
                        "context pressure: truncated older tool results"
                    );
                }
            }

            let tool_calls = reply.tool_calls.clone();
            messages.push(ChatMessage::Assistant {
                content: reply.content,
                tool_calls: Some(reply.tool_calls),
            });

            if parallel_round(config.settings.parallel_tools, &tool_calls, |name| {
                bindings.get(name).map(|t| t.read_only)
            }) {
                tracing::info!(agent = %agent_name, calls = tool_calls.len(), "parallel tool round");
                let outputs = futures_util::future::join_all(tool_calls.iter().map(|call| {
                    let tool = bindings[call.function.name.as_str()];
                    let policy = CallPolicy::for_tool(
                        agent.require_approval,
                        &agent.auto_approve,
                        approvals,
                        tool.read_only,
                    );
                    async move {
                        tracing::info!(
                            agent = %agent_name, tool = %call.function.name,
                            args = %truncate(&call.function.arguments, 200),
                            "agent tool call"
                        );
                        dispatch_tool_call(
                            &tool.binding,
                            &call.function.arguments,
                            config,
                            mcp,
                            http,
                            depth,
                            &policy,
                        )
                        .await
                    }
                }))
                .await;
                for (call, output) in tool_calls.iter().zip(outputs) {
                    messages.push(ChatMessage::Tool {
                        content: output?,
                        tool_call_id: call.id.clone(),
                    });
                }
                continue;
            }

            for call in tool_calls {
                tracing::info!(
                    agent = %agent_name, tool = %call.function.name,
                    args = %truncate(&call.function.arguments, 200),
                    "agent tool call"
                );
                let output = match bindings.get(call.function.name.as_str()) {
                    Some(tool) => {
                        let policy = CallPolicy::for_tool(
                            agent.require_approval,
                            &agent.auto_approve,
                            approvals,
                            tool.read_only,
                        );
                        dispatch_tool_call(
                            &tool.binding,
                            &call.function.arguments,
                            config,
                            mcp,
                            http,
                            depth,
                            &policy,
                        )
                        .await?
                    }
                    None => format!("ERROR: unknown tool `{}`", call.function.name),
                };
                messages.push(ChatMessage::Tool { content: output, tool_call_id: call.id });
            }
        }

        bail!(
            "agent `{agent_name}` did not produce a final answer within {max_turns} turns \
             (raise `max_turns` on the agent or `default_max_turns` in settings)"
        )
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
        PipelineContext { input: input.to_string(), ..Default::default() }
    }

    pub fn record(&mut self, stage_name: &str, output: String) {
        self.outputs.insert(stage_name.to_string(), output.clone());
        self.previous = Some(output);
    }
}

/// Build a chat client for a named model, with optional caller-level
/// sampling overrides taking precedence over each model's defaults. The
/// client's targets are the model followed by its fallback chain,
/// resolved breadth-first with cycles ignored.
pub fn build_client(
    config: &Config,
    model_name: &str,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    http: &reqwest::Client,
) -> Result<ChatClient> {
    let mut targets = Vec::new();
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
        targets.push(crate::provider::Target {
            base_url: provider.base_url.clone(),
            api_key: provider.api_key.clone(),
            model: model.model.clone(),
            label: name,
            params: SamplingParams {
                temperature: temperature.or(model.temperature),
                top_p: model.top_p,
                max_tokens: max_tokens.or(model.max_tokens),
            },
            stream: provider.stream,
        });
        queue.extend(model.fallback.iter().cloned());
    }

    Ok(ChatClient::new(http.clone(), targets, config.settings.provider_retries))
}

/// Run one stage to completion. `reprompt_targets` are the stages the model
/// may hand control to via `reprompt_stage` (empty = tool not offered, as in
/// chat mode and single-stage runs). `on_delta` receives streamed content
/// fragments for live display.
#[allow(clippy::too_many_arguments)]
pub async fn run_stage(
    config: &Config,
    stage: &Stage,
    is_first: bool,
    context: &PipelineContext,
    mcp: &McpManager,
    http: &reqwest::Client,
    reprompt_targets: &[String],
    on_delta: Option<crate::provider::DeltaHandler<'_>>,
    approvals: &Approvals,
) -> Result<StageOutcome> {
    let client =
        build_client(config, &stage.model, stage.temperature, stage.max_tokens, http)?;

    let stage_tools = assemble_tools(&stage.tool_profile(), config, mcp, 0)?;
    let mut definitions: Vec<ToolFunction> =
        stage_tools.iter().map(|t| t.definition.clone()).collect();
    if !reprompt_targets.is_empty() {
        definitions.push(reprompt_tool(reprompt_targets));
    }
    let bindings: BTreeMap<&str, &StageTool> = stage_tools
        .iter()
        .map(|t| (t.definition.name.as_str(), t))
        .collect();

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
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(ChatMessage::System { content: system });
    }
    messages.push(ChatMessage::User { content: user_prompt });

    let max_turns = stage.max_turns.unwrap_or(config.settings.default_max_turns);
    tracing::info!(
        stage = %stage.name, model = %stage.model, tools = definitions.len(),
        mode = ?stage.mode, "running stage"
    );

    for turn in 1..=max_turns {
        let reply = client.chat_streamed(&messages, &definitions, on_delta).await?;

        // Terminate the streamed text before anything else (logs, tool
        // lines) writes to the same terminal.
        if let (Some(handler), Some(content)) = (on_delta, reply.content.as_deref())
            && !content.is_empty()
        {
            handler("\n");
        }

        if reply.tool_calls.is_empty() {
            let content = reply.content.unwrap_or_default();
            tracing::info!(stage = %stage.name, turns = turn, "stage complete");
            return Ok(StageOutcome::Final(content));
        }

        // Under context pressure, truncate older tool results before the
        // next request instead of overflowing the window.
        if let Some((used, capacity)) = context_pressure(config, &stage.model, reply.usage.as_ref())
        {
            let trimmed = shed_context(&mut messages, 2);
            if trimmed > 0 {
                tracing::warn!(
                    stage = %stage.name, used, capacity, trimmed,
                    "context pressure: truncated older tool results"
                );
            }
        }

        let tool_calls = reply.tool_calls.clone();
        messages.push(ChatMessage::Assistant {
            content: reply.content,
            tool_calls: Some(reply.tool_calls),
        });

        if parallel_round(config.settings.parallel_tools, &tool_calls, |name| {
            bindings.get(name).map(|t| t.read_only)
        }) {
            tracing::info!(stage = %stage.name, calls = tool_calls.len(), "parallel tool round");
            let outputs = futures_util::future::join_all(tool_calls.iter().map(|call| {
                let tool = bindings[call.function.name.as_str()];
                let policy = CallPolicy::for_tool(
                    stage.require_approval,
                    &stage.auto_approve,
                    approvals,
                    tool.read_only,
                );
                async move {
                    tracing::info!(
                        stage = %stage.name, tool = %call.function.name,
                        args = %truncate(&call.function.arguments, 200),
                        "tool call"
                    );
                    dispatch_tool_call(
                        &tool.binding,
                        &call.function.arguments,
                        config,
                        mcp,
                        http,
                        0,
                        &policy,
                    )
                    .await
                }
            }))
            .await;
            for (call, output) in tool_calls.iter().zip(outputs) {
                messages.push(ChatMessage::Tool {
                    content: output?,
                    tool_call_id: call.id.clone(),
                });
            }
            continue;
        }

        for call in tool_calls {
            tracing::info!(
                stage = %stage.name, tool = %call.function.name,
                args = %truncate(&call.function.arguments, 200),
                "tool call"
            );
            if call.function.name == REPROMPT_TOOL && !reprompt_targets.is_empty() {
                match parse_reprompt(&call.function.arguments, reprompt_targets) {
                    // Any tool calls batched after the handoff are dropped.
                    Ok(outcome) => return Ok(outcome),
                    Err(problem) => {
                        // Give the model the validation error and let it retry.
                        messages.push(ChatMessage::Tool {
                            content: format!("ERROR: {problem}"),
                            tool_call_id: call.id,
                        });
                        continue;
                    }
                }
            }
            let output = match bindings.get(call.function.name.as_str()) {
                Some(tool) => {
                    let policy = CallPolicy::for_tool(
                        stage.require_approval,
                        &stage.auto_approve,
                        approvals,
                        tool.read_only,
                    );
                    dispatch_tool_call(
                        &tool.binding,
                        &call.function.arguments,
                        config,
                        mcp,
                        http,
                        0,
                        &policy,
                    )
                    .await?
                }
                None => format!("ERROR: unknown tool `{}`", call.function.name),
            };
            tracing::debug!(stage = %stage.name, tool = %call.function.name, output = %truncate(&output, 500));
            messages.push(ChatMessage::Tool {
                content: output,
                tool_call_id: call.id,
            });
        }
    }

    bail!(
        "stage `{}` did not produce a final answer within {max_turns} turns \
         (raise `max_turns` on the stage or `default_max_turns` in settings)",
        stage.name
    )
}

/// If the last reply's usage puts the conversation over the auto-compact
/// threshold of the model's declared context window, return
/// `(used, capacity)`.
pub fn context_pressure(
    config: &Config,
    model_name: &str,
    usage: Option<&crate::provider::Usage>,
) -> Option<(u64, u64)> {
    let threshold = config.settings.auto_compact_threshold;
    if threshold <= 0.0 {
        return None;
    }
    let capacity = config.models.get(model_name)?.context_tokens?;
    let used = usage?.context_tokens();
    (used as f64 >= capacity as f64 * threshold).then_some((used, capacity))
}

/// Free context by truncating older tool results in place, keeping the
/// most recent `keep_recent` intact. Returns how many were trimmed.
/// Deterministic (no model call), so it's safe mid-stage.
pub fn shed_context(messages: &mut [ChatMessage], keep_recent: usize) -> usize {
    const MARKER: &str = "[trimmed: an earlier tool result was truncated to save context]";
    let tool_indices: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m, ChatMessage::Tool { .. }))
        .map(|(i, _)| i)
        .collect();
    if tool_indices.len() <= keep_recent {
        return 0;
    }
    let mut trimmed = 0;
    for &index in &tool_indices[..tool_indices.len() - keep_recent] {
        if let ChatMessage::Tool { content, .. } = &mut messages[index] {
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

fn parse_reprompt(arguments_json: &str, targets: &[String]) -> Result<StageOutcome, String> {
    #[derive(serde::Deserialize)]
    struct RepromptArgs {
        stage: String,
        instructions: String,
    }
    let args: RepromptArgs = serde_json::from_str(arguments_json)
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
    Ok(StageOutcome::Reprompt { target: args.stage, instructions: args.instructions })
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
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [agents.reader]
            model = "m"
            description = "reads things"

            [agents.writer]
            model = "m"
            mode = "read_write"

            [[stage]]
            name = "s"
            model = "m"
            subagents = ["reader", "writer"]
            "#,
        )
        .unwrap();
        let mcp = McpManager::default();

        // Read-only stage: only the read-only agent is offered.
        let names: Vec<String> =
            assemble_tools(&config.stages[0].tool_profile(), &config, &mcp, 0)
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
        let client = build_client(&config, "a", None, None, &http).unwrap();
        // a's own fallbacks first, then theirs; the cycle back to `a` is cut.
        assert_eq!(client.target_labels(), vec!["a", "b", "c", "d"]);
        // A model with no fallbacks yields a single target.
        let solo = build_client(&config, "c", None, None, &http).unwrap();
        assert_eq!(solo.target_labels(), vec!["c"]);
    }

    #[test]
    fn parallel_round_requires_all_read_only() {
        let call = |name: &str| crate::provider::ToolCall {
            id: "x".into(),
            kind: "function".into(),
            function: crate::provider::FunctionCall { name: name.into(), arguments: "{}".into() },
        };
        let read_only = |name: &str| match name {
            "read" => Some(true),
            "write" => Some(false),
            _ => None,
        };
        let reads = [call("read"), call("read")];
        assert!(parallel_round(true, &reads, read_only));
        // Disabled by config, single call, any write, or any unknown tool.
        assert!(!parallel_round(false, &reads, read_only));
        assert!(!parallel_round(true, &reads[..1], read_only));
        assert!(!parallel_round(true, &[call("read"), call("write")], read_only));
        assert!(!parallel_round(true, &[call("read"), call("reprompt_stage")], read_only));
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
            vec!["read_file", "list_dir", "glob", "grep", "write_file", "edit_lines", "edit_file"]
        );
        assert!(names(2).is_empty());
    }

    #[test]
    fn context_pressure_gated_by_threshold_and_capacity() {
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
        let usage = |prompt, completion| {
            Some(crate::provider::Usage { prompt_tokens: prompt, completion_tokens: completion })
        };

        // Default threshold 0.8 of 1000: fires at 800 (prompt + completion), not below.
        assert_eq!(
            context_pressure(&config, "gauged", usage(700, 100).as_ref()),
            Some((800, 1000))
        );
        assert_eq!(context_pressure(&config, "gauged", usage(700, 99).as_ref()), None);
        // No declared window, no reported usage, or unknown model: never fires.
        assert_eq!(context_pressure(&config, "unbounded", usage(9000, 0).as_ref()), None);
        assert_eq!(context_pressure(&config, "gauged", None), None);
        assert_eq!(context_pressure(&config, "missing", usage(9000, 0).as_ref()), None);
        // Threshold 0 disables the check entirely.
        config.settings.auto_compact_threshold = 0.0;
        assert_eq!(context_pressure(&config, "gauged", usage(9000, 0).as_ref()), None);
    }

    #[test]
    fn sheds_older_tool_results_only() {
        let big = "x".repeat(1000);
        let mut messages = vec![
            ChatMessage::System { content: "s".into() },
            ChatMessage::User { content: "u".into() },
            ChatMessage::Tool { content: big.clone(), tool_call_id: "1".into() },
            ChatMessage::Tool { content: big.clone(), tool_call_id: "2".into() },
            ChatMessage::Tool { content: big.clone(), tool_call_id: "3".into() },
        ];
        assert_eq!(shed_context(&mut messages, 2), 1);
        let ChatMessage::Tool { content, .. } = &messages[2] else { panic!() };
        assert!(content.starts_with("[trimmed"));
        assert!(content.len() < 400);
        // Recent two untouched.
        let ChatMessage::Tool { content, .. } = &messages[4] else { panic!() };
        assert_eq!(content, &big);
        // Idempotent: already-trimmed entries aren't re-counted.
        assert_eq!(shed_context(&mut messages, 2), 0);
        // Small results aren't worth trimming.
        let mut small = vec![
            ChatMessage::Tool { content: "tiny".into(), tool_call_id: "1".into() },
            ChatMessage::Tool { content: big.clone(), tool_call_id: "2".into() },
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
