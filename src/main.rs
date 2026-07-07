mod config;
mod mcp;
mod provider;
mod stage;
mod tools;

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
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "soa=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let config = Config::load(&cli.config)?;

    match cli.command {
        Command::Check => {
            // Config::load already validated; also check prompt files resolve.
            for stage in &config.stages {
                stage.resolve_system_prompt(&config.base_dir)?;
            }
            println!(
                "OK: {} provider(s), {} model(s), {} mcp server(s), {} stage(s)",
                config.providers.len(),
                config.models.len(),
                config.mcp.len(),
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
            let manager = McpManager::connect(config.mcp.keys().cloned(), &config).await?;
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
    }
}

async fn run_pipeline(config: &Config, task: &str, only_stage: Option<&str>) -> Result<()> {
    let stages: Vec<(usize, &config::Stage)> = match only_stage {
        Some(name) => {
            let found = config
                .stages
                .iter()
                .enumerate()
                .find(|(_, s)| s.name == name)
                .with_context(|| format!("no stage named `{name}`"))?;
            vec![found]
        }
        None => config.stages.iter().enumerate().collect(),
    };

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.settings.request_timeout_secs))
        .build()
        .context("failed to build HTTP client")?;

    let needed_servers: Vec<String> = stages
        .iter()
        .flat_map(|(_, s)| s.mcp.iter().cloned())
        .collect();
    let manager = McpManager::connect(needed_servers, config).await?;

    let mut context = stage::PipelineContext::new(task);
    let total = stages.len();

    let mut result = Ok(());
    for (position, (index, stage)) in stages.iter().enumerate() {
        eprintln!("── stage {}/{}: {} ──", position + 1, total, stage.name);
        match stage::run_stage(config, stage, *index == 0, &context, &manager, &http).await {
            Ok(output) => {
                // Intermediate outputs go to stderr; only the final stage's
                // answer lands on stdout so `soa run` is pipe-friendly.
                if position + 1 < total {
                    eprintln!("{output}\n");
                } else {
                    println!("{output}");
                }
                context.record(&stage.name, output);
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }

    manager.shutdown().await;
    result
}
