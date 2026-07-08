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
    /// Kept for respawning: a crashed server is reconnected transparently
    /// on the next tool call.
    server_config: McpServer,
    quiet: bool,
    /// RwLock so concurrent tool calls proceed in parallel (read);
    /// reconnecting swaps the service exclusively (write).
    service: tokio::sync::RwLock<RunningService<RoleClient, ()>>,
    /// Bumped on every reconnect, so concurrent failures don't each
    /// respawn the server.
    generation: std::sync::atomic::AtomicU64,
    pub tools: Vec<Tool>,
    readonly_overrides: HashSet<String>,
}

impl McpConnection {
    /// `quiet` discards spawned servers' stderr instead of inheriting it —
    /// required in TUI mode, where stray writes would corrupt the display.
    pub async fn connect(name: &str, config: &McpServer, quiet: bool) -> Result<McpConnection> {
        let service = Self::establish(name, config, quiet).await?;
        let tools = service
            .list_all_tools()
            .await
            .with_context(|| format!("mcp `{name}`: tools/list failed"))?;

        Ok(McpConnection {
            name: name.to_string(),
            server_config: config.clone(),
            quiet,
            service: tokio::sync::RwLock::new(service),
            generation: std::sync::atomic::AtomicU64::new(0),
            tools,
            readonly_overrides: config.readonly_tools().iter().cloned().collect(),
        })
    }

    async fn establish(
        name: &str,
        config: &McpServer,
        quiet: bool,
    ) -> Result<RunningService<RoleClient, ()>> {
        let service = match config {
            McpServer::Stdio { command, args, env, .. } => {
                let mut cmd = tokio::process::Command::new(command);
                cmd.args(args).envs(env);
                let stderr = if quiet {
                    std::process::Stdio::null()
                } else {
                    std::process::Stdio::inherit()
                };
                let (transport, _) = TokioChildProcess::builder(cmd)
                    .stderr(stderr)
                    .spawn()
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
        Ok(service)
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
    /// A protocol-level failure (typically a crashed stdio server) triggers
    /// one transparent reconnect-and-retry before giving up.
    pub async fn call(&self, tool_name: &str, arguments: JsonObject) -> Result<String> {
        let generation = self.generation.load(std::sync::atomic::Ordering::Acquire);
        let result = match self.try_call(tool_name, arguments.clone()).await {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(
                    server = %self.name, tool = %tool_name, error = format!("{e:#}"),
                    "mcp call failed; reconnecting and retrying"
                );
                self.reconnect(generation).await.with_context(|| {
                    format!("mcp `{}`: reconnect after failed call also failed", self.name)
                })?;
                self.try_call(tool_name, arguments).await.with_context(|| {
                    format!("mcp `{}`: `{tool_name}` failed again after reconnect", self.name)
                })?
            }
        };

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

    async fn try_call(
        &self,
        tool_name: &str,
        arguments: JsonObject,
    ) -> Result<rmcp::model::CallToolResult> {
        let service = self.service.read().await;
        service
            .call_tool(CallToolRequestParams::new(tool_name.to_string()).with_arguments(arguments))
            .await
            .with_context(|| format!("mcp `{}`: call to `{tool_name}` failed", self.name))
    }

    /// Replace the dead service with a freshly connected one (respawning
    /// the process for stdio servers). `observed` is the generation the
    /// caller saw before its call failed: when several concurrent calls
    /// fail together, only the first respawns and the rest reuse it.
    async fn reconnect(&self, observed: u64) -> Result<()> {
        use std::sync::atomic::Ordering;
        let mut service = self.service.write().await;
        if self.generation.load(Ordering::Acquire) != observed {
            return Ok(()); // another caller already reconnected
        }
        let fresh = Self::establish(&self.name, &self.server_config, self.quiet).await?;
        let dead = std::mem::replace(&mut *service, fresh);
        self.generation.fetch_add(1, Ordering::Release);
        drop(service);
        let _ = dead.cancel().await; // reap the old child, ignore its state
        tracing::info!(server = %self.name, "mcp server reconnected");
        Ok(())
    }

    pub async fn shutdown(self) {
        let _ = self.service.into_inner().cancel().await;
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
        quiet: bool,
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
            let connection = McpConnection::connect(&name, server_config, quiet).await?;
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
