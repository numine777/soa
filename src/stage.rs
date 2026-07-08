//! Stage execution: template rendering, tool assembly (with read-only
//! filtering), and the per-stage agentic tool-call loop.

use std::collections::BTreeMap;
use std::future::Future;

use anyhow::{Context, Result, anyhow, bail};
use rmcp::model::JsonObject;
use serde_json::Value;

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
}

impl crate::config::Stage {
    pub fn tool_profile(&self) -> ToolProfile<'_> {
        ToolProfile {
            owner: &self.name,
            mode: self.mode,
            mcp: &self.mcp,
            web_search: self.web_search,
            subagents: &self.subagents,
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

/// Execute a tool call and clamp the result so a single oversized output
/// (e.g. a recursive directory tree) cannot exhaust the model's context.
/// `depth` is the caller's delegation depth (0 for stages and chat).
pub async fn dispatch_tool_call(
    binding: &ToolBinding,
    arguments_json: &str,
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
    depth: u32,
) -> Result<String> {
    let output =
        dispatch_tool_call_inner(binding, arguments_json, config, mcp, http, depth).await?;
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
            match run_agent(config, agent, task, mcp, http, depth + 1).await {
                Ok(answer) => Ok(answer),
                // The agent's failure becomes feedback, not a crashed turn.
                Err(e) => Ok(format!("ERROR: agent `{agent}` failed: {e:#}")),
            }
        }
    }
}

/// Run a subagent to completion on a task and return its final answer.
/// Boxed because agents may recursively spawn agents (bounded by
/// `settings.max_agent_depth` via `assemble_tools`).
pub fn run_agent<'a>(
    config: &'a Config,
    agent_name: &'a str,
    task: &'a str,
    mcp: &'a McpManager,
    http: &'a reqwest::Client,
    depth: u32,
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
        let bindings: BTreeMap<&str, &ToolBinding> = agent_tools
            .iter()
            .map(|t| (t.definition.name.as_str(), &t.binding))
            .collect();

        let mut messages = Vec::new();
        if let Some(system) = agent.resolve_system_prompt(&config.base_dir)? {
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

            let tool_calls = reply.tool_calls.clone();
            messages.push(ChatMessage::Assistant {
                content: reply.content,
                tool_calls: Some(reply.tool_calls),
            });

            for call in tool_calls {
                tracing::info!(
                    agent = %agent_name, tool = %call.function.name,
                    args = %truncate(&call.function.arguments, 200),
                    "agent tool call"
                );
                let output = match bindings.get(call.function.name.as_str()) {
                    Some(binding) => {
                        dispatch_tool_call(
                            binding,
                            &call.function.arguments,
                            config,
                            mcp,
                            http,
                            depth,
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
/// sampling overrides taking precedence over the model's defaults.
pub fn build_client(
    config: &Config,
    model_name: &str,
    temperature: Option<f64>,
    max_tokens: Option<u32>,
    http: &reqwest::Client,
) -> Result<ChatClient> {
    let model = config
        .models
        .get(model_name)
        .ok_or_else(|| anyhow!("unknown model `{model_name}`"))?;
    let provider = config
        .providers
        .get(&model.provider)
        .ok_or_else(|| anyhow!("unknown provider `{}`", model.provider))?;

    Ok(ChatClient::new(
        http.clone(),
        &provider.base_url,
        provider.api_key.clone(),
        &model.model,
        SamplingParams {
            temperature: temperature.or(model.temperature),
            top_p: model.top_p,
            max_tokens: max_tokens.or(model.max_tokens),
        },
    ))
}

/// Run one stage to completion. `reprompt_targets` are the stages the model
/// may hand control to via `reprompt_stage` (empty = tool not offered, as in
/// chat mode and single-stage runs).
pub async fn run_stage(
    config: &Config,
    stage: &Stage,
    is_first: bool,
    context: &PipelineContext,
    mcp: &McpManager,
    http: &reqwest::Client,
    reprompt_targets: &[String],
) -> Result<StageOutcome> {
    let client =
        build_client(config, &stage.model, stage.temperature, stage.max_tokens, http)?;

    let stage_tools = assemble_tools(&stage.tool_profile(), config, mcp, 0)?;
    let mut definitions: Vec<ToolFunction> =
        stage_tools.iter().map(|t| t.definition.clone()).collect();
    if !reprompt_targets.is_empty() {
        definitions.push(reprompt_tool(reprompt_targets));
    }
    let bindings: BTreeMap<&str, &ToolBinding> = stage_tools
        .iter()
        .map(|t| (t.definition.name.as_str(), &t.binding))
        .collect();

    let user_prompt = render_template(
        &stage.prompt_template(is_first),
        &context.input,
        context.previous.as_deref(),
        &context.outputs,
    )?;

    let mut messages = Vec::new();
    if let Some(system) = stage.resolve_system_prompt(&config.base_dir)? {
        messages.push(ChatMessage::System { content: system });
    }
    messages.push(ChatMessage::User { content: user_prompt });

    let max_turns = stage.max_turns.unwrap_or(config.settings.default_max_turns);
    tracing::info!(
        stage = %stage.name, model = %stage.model, tools = definitions.len(),
        mode = ?stage.mode, "running stage"
    );

    for turn in 1..=max_turns {
        let reply = client.chat(&messages, &definitions).await?;

        if reply.tool_calls.is_empty() {
            let content = reply.content.unwrap_or_default();
            tracing::info!(stage = %stage.name, turns = turn, "stage complete");
            return Ok(StageOutcome::Final(content));
        }

        let tool_calls = reply.tool_calls.clone();
        messages.push(ChatMessage::Assistant {
            content: reply.content,
            tool_calls: Some(reply.tool_calls),
        });

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
                Some(binding) => {
                    dispatch_tool_call(binding, &call.function.arguments, config, mcp, http, 0)
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
