//! Stage execution: template rendering, tool assembly (with read-only
//! filtering), and the per-stage agentic tool-call loop.

use std::collections::BTreeMap;

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
pub enum ToolBinding {
    Mcp { server: String, tool: String },
    WebSearch,
}

pub struct StageTool {
    pub definition: ToolFunction,
    pub binding: ToolBinding,
}

/// Tool names advertised to the model must match `[a-zA-Z0-9_-]{1,64}`.
fn sanitize_tool_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '-' })
        .collect();
    cleaned.chars().take(64).collect()
}

/// Assemble the tools visible to a stage, applying the read-only filter.
/// MCP tool names are namespaced as `<server>__<tool>` to avoid collisions.
pub fn assemble_tools(stage: &Stage, mcp: &McpManager) -> Result<Vec<StageTool>> {
    let mut stage_tools = Vec::new();

    for server_name in &stage.mcp {
        let connection = mcp
            .get(server_name)
            .ok_or_else(|| anyhow!("mcp server `{server_name}` is not connected"))?;
        for tool in &connection.tools {
            if stage.mode == Mode::ReadOnly && !connection.is_read_only(tool) {
                tracing::debug!(
                    stage = %stage.name, server = %server_name, tool = %tool.name,
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
            });
        }
    }

    if stage.web_search {
        stage_tools.push(StageTool {
            definition: tools::web_search_definition(),
            binding: ToolBinding::WebSearch,
        });
    }

    Ok(stage_tools)
}

async fn dispatch_tool_call(
    binding: &ToolBinding,
    arguments_json: &str,
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
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
    }
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

/// Run one stage to completion and return its final text output.
pub async fn run_stage(
    config: &Config,
    stage: &Stage,
    is_first: bool,
    context: &PipelineContext,
    mcp: &McpManager,
    http: &reqwest::Client,
) -> Result<String> {
    let model = config
        .models
        .get(&stage.model)
        .ok_or_else(|| anyhow!("unknown model `{}`", stage.model))?;
    let provider = config
        .providers
        .get(&model.provider)
        .ok_or_else(|| anyhow!("unknown provider `{}`", model.provider))?;

    let client = ChatClient::new(
        http.clone(),
        &provider.base_url,
        provider.api_key.clone(),
        &model.model,
        SamplingParams {
            temperature: stage.temperature.or(model.temperature),
            top_p: model.top_p,
            max_tokens: stage.max_tokens.or(model.max_tokens),
        },
    );

    let stage_tools = assemble_tools(stage, mcp)?;
    let definitions: Vec<ToolFunction> =
        stage_tools.iter().map(|t| t.definition.clone()).collect();
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
        stage = %stage.name, model = %model.model, tools = definitions.len(),
        mode = ?stage.mode, "running stage"
    );

    for turn in 1..=max_turns {
        let reply = client.chat(&messages, &definitions).await?;

        if reply.tool_calls.is_empty() {
            let content = reply.content.unwrap_or_default();
            tracing::info!(stage = %stage.name, turns = turn, "stage complete");
            return Ok(content);
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
            let output = match bindings.get(call.function.name.as_str()) {
                Some(binding) => {
                    dispatch_tool_call(binding, &call.function.arguments, config, mcp, http)
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
    fn sanitizes_tool_names() {
        assert_eq!(sanitize_tool_name("fs__read.file"), "fs__read-file");
        assert_eq!(sanitize_tool_name(&"x".repeat(100)).len(), 64);
    }
}
