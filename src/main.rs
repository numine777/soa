mod config;
mod diff;
mod mcp;
mod provider;
mod stage;
mod tools;
mod tui;

use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};

use config::{Config, Mode};
use mcp::McpManager;

/// soa — a staged, TOML-configured harness for local AI models.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Path to the configuration file.
    #[arg(short, long, default_value = "soa.toml", global = true)]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate the configuration and exit.
    Check,
    /// List the configured stages.
    Stages,
    /// Connect to all MCP servers and list their tools (ro = read-only).
    Tools,
    /// Run the stage pipeline on a task (reads stdin if no task is given).
    Run {
        /// The task to perform.
        task: Vec<String>,
        /// Run only this stage instead of the whole pipeline.
        #[arg(long)]
        stage: Option<String>,
    },
    /// Interactive chat TUI using a stage's model, tools, and mode.
    Chat {
        /// Stage to chat with (default: the first stage).
        #[arg(long)]
        stage: Option<String>,
        /// Disable mouse capture (keeps the terminal's native text selection).
        #[arg(long)]
        no_mouse: bool,
        /// Resume a saved session: `--resume` for the most recent,
        /// `--resume <id>` for a specific one (see `soa sessions`).
        #[arg(long, value_name = "ID", num_args = 0..=1, default_missing_value = "latest")]
        resume: Option<String>,
    },
    /// List saved chat sessions.
    Sessions,
}

fn env_filter() -> tracing_subscriber::EnvFilter {
    tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "soa=info".into())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // In chat mode the terminal belongs to the TUI, so logs go to a file.
    if matches!(cli.command, Command::Chat { .. }) {
        let log_path = std::env::temp_dir().join("soa-chat.log");
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .with_context(|| format!("cannot open log file {}", log_path.display()))?;
        tracing_subscriber::fmt()
            .with_env_filter(env_filter())
            .with_writer(std::sync::Arc::new(log_file))
            .with_ansi(false)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter(env_filter())
            .with_writer(std::io::stderr)
            .init();
    }

    let config = Config::load(&cli.config)?;

    match cli.command {
        Command::Check => {
            // Config::load already validated; also check prompt files resolve.
            for stage in &config.stages {
                stage.resolve_system_prompt(&config.base_dir)?;
            }
            println!(
                "OK: {} provider(s), {} model(s), {} mcp server(s), {} agent(s), {} stage(s)",
                config.providers.len(),
                config.models.len(),
                config.mcp.len(),
                config.agents.len(),
                config.stages.len()
            );
            Ok(())
        }
        Command::Stages => {
            for (index, stage) in config.stages.iter().enumerate() {
                let mode = match stage.mode {
                    Mode::ReadOnly => "read_only",
                    Mode::ReadWrite => "read_write",
                };
                println!(
                    "{}. {}  model={}  mode={}  mcp=[{}]{}",
                    index + 1,
                    stage.name,
                    stage.model,
                    mode,
                    stage.mcp.join(", "),
                    if stage.web_search { "  +web_search" } else { "" },
                );
            }
            Ok(())
        }
        Command::Tools => {
            let manager = McpManager::connect(config.mcp.keys().cloned(), &config, false).await?;
            for name in config.mcp.keys() {
                let connection = manager.get(name).expect("connected above");
                println!("[{name}]");
                for tool in &connection.tools {
                    let marker = if connection.is_read_only(tool) { "ro" } else { "rw" };
                    println!(
                        "  {} ({marker})  {}",
                        tool.name,
                        tool.description.as_deref().unwrap_or("").lines().next().unwrap_or("")
                    );
                }
            }
            manager.shutdown().await;
            Ok(())
        }
        Command::Run { task, stage } => {
            let task = if task.is_empty() {
                let mut buffer = String::new();
                std::io::stdin()
                    .read_to_string(&mut buffer)
                    .context("failed to read task from stdin")?;
                buffer.trim().to_string()
            } else {
                task.join(" ")
            };
            if task.is_empty() {
                bail!("no task given (pass it as an argument or on stdin)");
            }
            run_pipeline(&config, &task, stage.as_deref()).await
        }
        Command::Chat { stage, no_mouse, resume } => {
            tui::run(config, stage.as_deref(), !no_mouse, resume.as_deref()).await
        }
        Command::Sessions => {
            let sessions = tui::store::list_sessions()?;
            if sessions.is_empty() {
                println!("no saved sessions ({})", tui::store::data_dir().display());
                return Ok(());
            }
            for session in sessions {
                println!(
                    "{}  {}  [{}]  {}  ({})",
                    session.id,
                    tui::store::format_epoch(session.updated_at),
                    session.stage,
                    session.title,
                    if session.cwd.is_empty() { "unknown dir" } else { &session.cwd },
                );
            }
            Ok(())
        }
    }
}

async fn run_pipeline(config: &Config, task: &str, only_stage: Option<&str>) -> Result<()> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.settings.request_timeout_secs))
        .build()
        .context("failed to build HTTP client")?;

    // Single-stage mode: one run, no reprompting.
    if let Some(name) = only_stage {
        let stage = config
            .stages
            .iter()
            .find(|s| s.name == name)
            .with_context(|| format!("no stage named `{name}`"))?;
        let servers: Vec<String> = stage
            .mcp
            .iter()
            .cloned()
            .chain(config.agents.values().flat_map(|a| a.mcp.iter().cloned()))
            .collect();
        let manager = McpManager::connect(servers, config, false).await?;
        let context = stage::PipelineContext::new(task);
        eprintln!("── stage {} ──", stage.name);
        let result =
            stage::run_stage(config, stage, true, &context, &manager, &http, &[]).await;
        manager.shutdown().await;
        return match result? {
            stage::StageOutcome::Final(output) => {
                println!("{output}");
                Ok(())
            }
            stage::StageOutcome::Reprompt { .. } => unreachable!("no reprompt targets offered"),
        };
    }

    // Agents can be reached from any stage, so connect their servers too.
    let needed_servers: Vec<String> = config
        .stages
        .iter()
        .flat_map(|s| s.mcp.iter().cloned())
        .chain(config.agents.values().flat_map(|a| a.mcp.iter().cloned()))
        .collect();
    let manager = McpManager::connect(needed_servers, config, false).await?;

    let mut context = stage::PipelineContext::new(task);
    let mut current = 0;
    let mut runs = 0u32;
    let mut last_output = None;

    let mut result = Ok(());
    while current < config.stages.len() {
        let stage = &config.stages[current];
        runs += 1;
        if runs > config.settings.max_stage_runs {
            result = Err(anyhow::anyhow!(
                "stopped after {} stage runs without finishing — likely a reprompt \
                 loop; raise settings.max_stage_runs if this is intentional",
                config.settings.max_stage_runs
            ));
            break;
        }

        eprintln!("── run {runs} · stage {} ──", stage.name);
        let is_first = context.previous.is_none();
        match stage::run_stage(
            config,
            stage,
            is_first,
            &context,
            &manager,
            &http,
            &stage.can_reprompt,
        )
        .await
        {
            Ok(stage::StageOutcome::Final(output)) => {
                context.record(&stage.name, output.clone());
                current += 1;
                // Intermediate outputs go to stderr; only the pipeline's
                // final answer lands on stdout so `soa run` is pipe-friendly.
                if current < config.stages.len() {
                    eprintln!("{output}\n");
                }
                last_output = Some(output);
            }
            Ok(stage::StageOutcome::Reprompt { target, instructions }) => {
                eprintln!("↩ {} reprompts {}:", stage.name, target);
                eprintln!("{instructions}\n");
                // The handoff instructions become the sender's recorded
                // output, so the target sees them as {{previous}}.
                context.record(&stage.name, instructions);
                current = config
                    .stages
                    .iter()
                    .position(|s| s.name == target)
                    .expect("can_reprompt targets are validated at config load");
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }

    if result.is_ok()
        && let Some(output) = last_output
    {
        println!("{output}");
    }

    manager.shutdown().await;
    result
}
