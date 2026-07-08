mod approval;
mod config;
mod diff;
mod mcp;
mod mentions;
mod provider;
mod skills;
mod stage;
mod tools;
mod tui;

use std::io::{IsTerminal, Read, Write};
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
        /// Named workflow to run (default: settings.default_workflow, a
        /// workflow named `default`, or every stage in declaration order).
        #[arg(short, long)]
        workflow: Option<String>,
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
    /// List discoverable skills.
    Skills,
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
            // Config::load already validated; also check that prompt files
            // and referenced skills resolve.
            for stage in &config.stages {
                let system = stage.resolve_system_prompt(&config.base_dir)?;
                skills::apply_skills(
                    &config,
                    &format!("stage `{}`", stage.name),
                    system,
                    &stage.skills,
                )?;
            }
            for (name, agent) in &config.agents {
                let system = agent.resolve_system_prompt(&config.base_dir)?;
                skills::apply_skills(&config, &format!("agent `{name}`"), system, &agent.skills)?;
            }
            println!(
                "OK: {} provider(s), {} model(s), {} mcp server(s), {} agent(s), {} stage(s), {} workflow(s)",
                config.providers.len(),
                config.models.len(),
                config.mcp.len(),
                config.agents.len(),
                config.stages.len(),
                config.workflows.len()
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
                    "{}. {}  model={}  mode={}  mcp=[{}]{}{}",
                    index + 1,
                    stage.name,
                    stage.model,
                    mode,
                    stage.mcp.join(", "),
                    if stage.web_search { "  +web_search" } else { "" },
                    if stage.skills.is_empty() {
                        String::new()
                    } else {
                        format!("  skills=[{}]", stage.skills.join(", "))
                    },
                );
            }
            if !config.workflows.is_empty() {
                println!("\nworkflows:");
                for (name, workflow) in &config.workflows {
                    let default_marker = match &config.settings.default_workflow {
                        Some(default) if default == name => "  (default)",
                        None if name == "default" => "  (default)",
                        _ => "",
                    };
                    println!(
                        "  {name}: {}{}{}",
                        workflow.stages.join(" -> "),
                        if workflow.description.is_empty() {
                            String::new()
                        } else {
                            format!("  — {}", workflow.description)
                        },
                        default_marker,
                    );
                }
            }
            Ok(())
        }
        Command::Skills => {
            let found = skills::list_skills(&config);
            if found.is_empty() {
                let dirs = skills::skills_dirs(&config);
                println!(
                    "no skills found (searched: {})",
                    dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
                );
                return Ok(());
            }
            for skill in found {
                println!(
                    "{}  {}  ({})",
                    skill.name,
                    if skill.description.is_empty() { "-" } else { &skill.description },
                    skill.path.display()
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
        Command::Run { task, stage, workflow } => {
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
            // Expand @file mentions relative to the current directory.
            let cwd = std::env::current_dir().context("cannot determine working directory")?;
            let (task, reports) =
                mentions::expand_mentions(&task, &cwd, config.settings.max_tool_output_chars);
            for report in &reports {
                eprintln!("• {}", report.describe());
            }
            run_pipeline(&config, &task, stage.as_deref(), workflow.as_deref()).await
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

/// Approvals for pipeline runs: prompt on the terminal when stdin is a
/// TTY; otherwise gated calls are denied with an explanation.
fn terminal_approvals() -> std::sync::Arc<approval::Approvals> {
    use approval::{Approvals, Decision};
    if !std::io::stdin().is_terminal() {
        return std::sync::Arc::new(Approvals::non_interactive());
    }
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<approval::ApprovalRequest>();
    tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let mut lines = tokio::io::BufReader::new(tokio::io::stdin()).lines();
        while let Some(request) = rx.recv().await {
            eprintln!("\n⚠ approval needed: {}", request.descriptor);
            if !request.detail.is_empty() && request.detail != request.descriptor {
                eprintln!("  {}", request.detail);
            }
            eprint!("  [y] once · [a] always ({}) · [N] deny > ", request.always_pattern);
            let _ = std::io::stderr().flush();
            let decision = match lines.next_line().await {
                Ok(Some(line)) => match line.trim().to_lowercase().as_str() {
                    "y" | "yes" => Decision::Approve,
                    "a" | "always" => Decision::AlwaysAllow,
                    _ => Decision::Deny,
                },
                _ => Decision::Deny,
            };
            let _ = request.responder.send(decision);
        }
    });
    std::sync::Arc::new(Approvals::new(tx))
}

/// Streams stage output to stderr as it arrives.
fn stream_to_stderr(fragment: &str) {
    let mut err = std::io::stderr();
    let _ = err.write_all(fragment.as_bytes());
    let _ = err.flush();
}

/// The final answer goes to stdout for pipes — except when both stdout and
/// stderr are the same terminal, where it just got streamed and printing it
/// again would duplicate it on screen.
fn print_final(output: &str) {
    if !(std::io::stdout().is_terminal() && std::io::stderr().is_terminal()) {
        println!("{output}");
    }
}

async fn run_pipeline(
    config: &Config,
    task: &str,
    only_stage: Option<&str>,
    workflow: Option<&str>,
) -> Result<()> {
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
        let approvals = terminal_approvals();
        let context = stage::PipelineContext::new(task);
        eprintln!("── stage {} ──", stage.name);
        let result = stage::run_stage(
            config,
            stage,
            true,
            &context,
            &manager,
            &http,
            &[],
            Some(&stream_to_stderr),
            &approvals,
        )
        .await;
        manager.shutdown().await;
        return match result? {
            stage::StageOutcome::Final(output) => {
                eprintln!();
                print_final(&output);
                Ok(())
            }
            stage::StageOutcome::Reprompt { .. } => unreachable!("no reprompt targets offered"),
        };
    }

    // The workflow gives the execution order as indexes into config.stages.
    let order = config.resolve_workflow(workflow)?;
    if order.is_empty() {
        bail!("nothing to run: the selected workflow is empty");
    }

    // Agents can be reached from any stage, so connect their servers too.
    let needed_servers: Vec<String> = order
        .iter()
        .flat_map(|&i| config.stages[i].mcp.iter().cloned())
        .chain(config.agents.values().flat_map(|a| a.mcp.iter().cloned()))
        .collect();
    let manager = McpManager::connect(needed_servers, config, false).await?;
    let approvals = terminal_approvals();

    let workflow_stage_names: Vec<&str> =
        order.iter().map(|&i| config.stages[i].name.as_str()).collect();

    let mut context = stage::PipelineContext::new(task);
    let mut position = 0;
    let mut runs = 0u32;
    let mut last_output = None;

    let mut result = Ok(());
    while position < order.len() {
        let stage = &config.stages[order[position]];
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
        // Only offer reprompt targets that are part of the active workflow.
        let reprompt_targets: Vec<String> = stage
            .can_reprompt
            .iter()
            .filter(|t| workflow_stage_names.contains(&t.as_str()))
            .cloned()
            .collect();
        match stage::run_stage(
            config,
            stage,
            is_first,
            &context,
            &manager,
            &http,
            &reprompt_targets,
            Some(&stream_to_stderr),
            &approvals,
        )
        .await
        {
            Ok(stage::StageOutcome::Final(output)) => {
                context.record(&stage.name, output.clone());
                position += 1;
                // The output already streamed to stderr; just separate stages.
                eprintln!("\n");
                last_output = Some(output);
            }
            Ok(stage::StageOutcome::Reprompt { target, instructions }) => {
                eprintln!("↩ {} reprompts {}:", stage.name, target);
                eprintln!("{instructions}\n");
                // The handoff instructions become the sender's recorded
                // output, so the target sees them as {{previous}}.
                context.record(&stage.name, instructions);
                position = workflow_stage_names
                    .iter()
                    .position(|name| *name == target)
                    .expect("reprompt targets are filtered to the active workflow");
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
        print_final(&output);
    }

    manager.shutdown().await;
    result
}
