//! MCP server connections and tool dispatch, built on the official `rmcp` SDK.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, anyhow};
use rmcp::model::{CallToolRequestParams, JsonObject, Tool};
use rmcp::service::RunningService;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::{StreamableHttpClientTransport, TokioChildProcess};
use rmcp::{RoleClient, ServiceExt};

use crate::config::McpServer;

pub struct McpConnection {
    pub name: String,
    service: RunningService<RoleClient, ()>,
    pub tools: Vec<Tool>,
    readonly_overrides: HashSet<String>,
}

impl McpConnection {
    pub async fn connect(name: &str, config: &McpServer) -> Result<McpConnection> {
        let service = match config {
            McpServer::Stdio { command, args, env, .. } => {
                let mut cmd = tokio::process::Command::new(command);
                cmd.args(args).envs(env);
                let transport = TokioChildProcess::new(cmd)
                    .with_context(|| format!("mcp `{name}`: failed to spawn `{command}`"))?;
                ()
                    .serve(transport)
                    .await
                    .with_context(|| format!("mcp `{name}`: initialize handshake failed"))?
            }
            McpServer::Http { url, auth_token, headers, .. } => {
                let mut transport_config = StreamableHttpClientTransportConfig::with_uri(url.clone());
                if let Some(token) = auth_token {
                    transport_config = transport_config.auth_header(token.clone());
                }
                if !headers.is_empty() {
                    let mut header_map = HashMap::new();
                    for (key, value) in headers {
                        header_map.insert(
                            key.parse::<http::HeaderName>()
                                .with_context(|| format!("mcp `{name}`: invalid header name `{key}`"))?,
                            value
                                .parse::<http::HeaderValue>()
                                .with_context(|| format!("mcp `{name}`: invalid value for header `{key}`"))?,
                        );
                    }
                    transport_config = transport_config.custom_headers(header_map);
                }
                let transport = StreamableHttpClientTransport::from_config(transport_config);
                ()
                    .serve(transport)
                    .await
                    .with_context(|| format!("mcp `{name}`: initialize handshake failed for {url}"))?
            }
        };

        let tools = service
            .list_all_tools()
            .await
            .with_context(|| format!("mcp `{name}`: tools/list failed"))?;

        Ok(McpConnection {
            name: name.to_string(),
            service,
            tools,
            readonly_overrides: config.readonly_tools().iter().cloned().collect(),
        })
    }

    /// A tool is read-only if the server annotates it `readOnlyHint = true`
    /// or the config lists it in `readonly_tools`.
    pub fn is_read_only(&self, tool: &Tool) -> bool {
        tool.annotations
            .as_ref()
            .and_then(|a| a.read_only_hint)
            .unwrap_or(false)
            || self.readonly_overrides.contains(tool.name.as_ref())
    }

    /// Call a tool and flatten the result to text for the model.
    /// Tool-level failures are returned as `Ok` text so the model can react.
    pub async fn call(&self, tool_name: &str, arguments: JsonObject) -> Result<String> {
        let result = self
            .service
            .call_tool(CallToolRequestParams::new(tool_name.to_string()).with_arguments(arguments))
            .await
            .with_context(|| format!("mcp `{}`: call to `{tool_name}` failed", self.name))?;

        let mut text: String = result
            .content
            .iter()
            .filter_map(|block| block.as_text().map(|t| t.text.as_str()))
            .collect::<Vec<_>>()
            .join("\n");
        if text.is_empty()
            && let Some(structured) = &result.structured_content
        {
            text = structured.to_string();
        }
        if result.is_error == Some(true) {
            text = format!("ERROR from tool `{tool_name}`: {text}");
        }
        if text.is_empty() {
            text = "(tool returned no output)".to_string();
        }
        Ok(text)
    }

    pub async fn shutdown(self) {
        let _ = self.service.cancel().await;
    }
}

/// All MCP connections needed for a run, keyed by config name.
#[derive(Default)]
pub struct McpManager {
    connections: HashMap<String, McpConnection>,
}

impl McpManager {
    /// Connect to the named servers (typically the union of the stages' `mcp` lists).
    pub async fn connect(
        servers: impl IntoIterator<Item = String>,
        config: &crate::config::Config,
    ) -> Result<McpManager> {
        let mut manager = McpManager::default();
        for name in servers {
            if manager.connections.contains_key(&name) {
                continue;
            }
            let server_config = config
                .mcp
                .get(&name)
                .ok_or_else(|| anyhow!("unknown mcp server `{name}`"))?;
            tracing::info!(server = %name, "connecting to MCP server");
            let connection = McpConnection::connect(&name, server_config).await?;
            tracing::info!(server = %name, tools = connection.tools.len(), "connected");
            manager.connections.insert(name, connection);
        }
        Ok(manager)
    }

    pub fn get(&self, name: &str) -> Option<&McpConnection> {
        self.connections.get(name)
    }

    pub async fn shutdown(self) {
        for (_, connection) in self.connections {
            connection.shutdown().await;
        }
    }
}
