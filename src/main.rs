mod approval;
mod config;
mod diff;
mod files;
mod git;
mod hooks;
mod insights;
mod mcp;
mod mentions;
mod model;
mod persistence;
mod providers;
mod reflect;
mod runs;
mod skills;
mod stage;
mod tools;
mod tui;

use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

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
        /// Resume an interrupted run from its last completed stage:
        /// `--resume` for this directory's most recent, `--resume <ID>`
        /// for a specific one (see `soa runs`).
        #[arg(long, value_name = "ID", num_args = 0..=1, default_missing_value = "latest")]
        resume: Option<String>,
    },
    /// List interrupted pipeline runs that can be resumed.
    Runs,
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
    /// Distill recent sessions into lessons (SOA.md) and skills: failure
    /// signals (denied calls, tool errors, rollbacks) become durable
    /// instructions that reach every stage. Review the result with git.
    Reflect {
        /// Print the proposal without writing any files.
        #[arg(long)]
        dry_run: bool,
        /// Model to reflect with (default: settings.reflect_model, then the
        /// first stage's model).
        #[arg(long)]
        model: Option<String>,
    },
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
                skills::compose_system(
                    &config,
                    &format!("stage `{}`", stage.name),
                    system,
                    &stage.skills,
                )?;
            }
            for (name, agent) in &config.agents {
                let system = agent.resolve_system_prompt(&config.base_dir)?;
                skills::compose_system(&config, &format!("agent `{name}`"), system, &agent.skills)?;
            }
            println!(
                "OK: {} provider(s), {} model(s), {} mcp server(s), {} agent(s), {} stage(s), {} workflow(s), {} hook(s)",
                config.providers.len(),
                config.models.len(),
                config.mcp.len(),
                config.agents.len(),
                config.stages.len(),
                config.workflows.len(),
                config.hooks.len()
            );
            if config.project_contexts.is_empty() {
                let candidates: Vec<String> = config
                    .settings
                    .context_files
                    .iter()
                    .map(|f| f.display().to_string())
                    .collect();
                println!(
                    "project instructions: none (searched for {})",
                    candidates.join(", ")
                );
            }
            for context in &config.project_contexts {
                println!(
                    "project instructions: {} ({} chars)",
                    context.path.display(),
                    context.content.len()
                );
            }
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
                    if stage.web_search {
                        "  +web_search"
                    } else {
                        ""
                    },
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
                    dirs.iter()
                        .map(|d| d.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                return Ok(());
            }
            for skill in found {
                println!(
                    "{}  {}  ({})",
                    skill.name,
                    if skill.description.is_empty() {
                        "-"
                    } else {
                        &skill.description
                    },
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
                    let marker = if connection.is_read_only(tool) {
                        "ro"
                    } else {
                        "rw"
                    };
                    println!(
                        "  {} ({marker})  {}",
                        tool.name,
                        tool.description
                            .as_deref()
                            .unwrap_or("")
                            .lines()
                            .next()
                            .unwrap_or("")
                    );
                }
            }
            manager.shutdown().await;
            Ok(())
        }
        Command::Run {
            task,
            stage,
            workflow,
            resume,
        } => {
            if let Some(resume) = resume {
                if !task.is_empty() || stage.is_some() || workflow.is_some() {
                    bail!(
                        "--resume continues a checkpointed run; it cannot be combined \
                         with a task, --stage, or --workflow"
                    );
                }
                let state = if resume == "latest" {
                    runs::latest_run_for(&tui::store::current_cwd())?.context(
                        "no interrupted run to resume in this directory (see `soa runs`)",
                    )?
                } else {
                    runs::load_run(&resume)?
                };
                return resume_pipeline(&config, state).await;
            }
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
        Command::Runs => {
            let states = runs::list_runs()?;
            if states.is_empty() {
                println!(
                    "no interrupted runs ({})",
                    tui::store::data_dir().join("runs").display()
                );
                return Ok(());
            }
            for state in states {
                let title: String = state
                    .task
                    .lines()
                    .next()
                    .unwrap_or("")
                    .chars()
                    .take(60)
                    .collect();
                println!(
                    "{}  {}  next stage {}/{} [{}]  {}  ({})",
                    state.id,
                    tui::store::format_epoch(state.updated_at),
                    state.position + 1,
                    state.stage_names.len(),
                    state
                        .stage_names
                        .get(state.position)
                        .map_or("?", |s| s.as_str()),
                    title,
                    if state.cwd.is_empty() {
                        "unknown dir"
                    } else {
                        &state.cwd
                    },
                );
            }
            Ok(())
        }
        Command::Chat {
            stage,
            no_mouse,
            resume,
        } => {
            tui::run(
                config,
                cli.config.clone(),
                stage.as_deref(),
                !no_mouse,
                resume.as_deref(),
            )
            .await
        }
        Command::Reflect { dry_run, model } => {
            reflect::run(&config, model.as_deref(), dry_run).await
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
                    if session.cwd.is_empty() {
                        "unknown dir"
                    } else {
                        &session.cwd
                    },
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
            eprint!(
                "  [y] once · [a] always ({}) · [N] deny > ",
                request.always_pattern
            );
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

const STREAM_FLUSH_INTERVAL: Duration = Duration::from_millis(16);
const STREAM_BUFFER_BYTES: usize = 4096;

struct StderrStream {
    state: std::sync::Mutex<StderrStreamState>,
}

struct StderrStreamState {
    pending: Vec<u8>,
    last_flush: Instant,
}

impl StderrStream {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            state: std::sync::Mutex::new(StderrStreamState {
                pending: Vec::with_capacity(STREAM_BUFFER_BYTES),
                // Let the first fragment appear immediately; bursts after
                // that coalesce to roughly one terminal write per frame.
                last_flush: now.checked_sub(STREAM_FLUSH_INTERVAL).unwrap_or(now),
            }),
        }
    }

    fn push(&self, fragment: &str) {
        let mut state = self.state.lock().expect("stderr stream lock");
        state.pending.extend_from_slice(fragment.as_bytes());
        if fragment.contains('\n')
            || state.pending.len() >= STREAM_BUFFER_BYTES
            || state.last_flush.elapsed() >= STREAM_FLUSH_INTERVAL
        {
            Self::flush_locked(&mut state);
        }
    }

    fn flush(&self) {
        Self::flush_locked(&mut self.state.lock().expect("stderr stream lock"));
    }

    fn flush_locked(state: &mut StderrStreamState) {
        if state.pending.is_empty() {
            return;
        }
        let mut err = std::io::stderr().lock();
        let _ = err.write_all(&state.pending);
        let _ = err.flush();
        state.pending.clear();
        state.last_flush = Instant::now();
    }
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
        let usage = model::UsageTracker::new(
            config.settings.run_limits(),
            model::UsageSnapshot::default(),
        );
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
        let manager = match usage
            .within_time(McpManager::connect(servers, config, false))
            .await
        {
            Ok(manager) => manager,
            Err(error) => {
                print_usage_summary(&usage);
                return Err(error);
            }
        };
        let approvals = terminal_approvals();
        let context = stage::PipelineContext::new(task);
        eprintln!("── stage {} ──", stage.name);
        let stream = StderrStream::new();
        let on_delta = |fragment: &str| stream.push(fragment);
        let result = stage::run_stage(
            config,
            stage,
            true,
            &context,
            &manager,
            &http,
            &usage,
            &[],
            None,
            &[],
            Some(&on_delta),
            None,
            &approvals,
        );
        let result = usage.within_time(result).await;
        stream.flush();
        manager.shutdown().await;
        print_usage_summary(&usage);
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
    let stage_names: Vec<String> = order
        .iter()
        .map(|&i| config.stages[i].name.clone())
        .collect();
    let state = runs::RunState::new(task, stage_names);
    run_workflow(config, &http, order, state).await
}

/// Continue a checkpointed pipeline run from its first incomplete stage.
async fn resume_pipeline(config: &Config, state: runs::RunState) -> Result<()> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.settings.request_timeout_secs))
        .build()
        .context("failed to build HTTP client")?;

    // The checkpoint stores stage names, so a reordered config can't
    // silently change what the resumed run does — but every recorded
    // stage must still exist.
    let order: Vec<usize> = state
        .stage_names
        .iter()
        .map(|name| {
            config
                .stages
                .iter()
                .position(|s| s.name == *name)
                .with_context(|| {
                    format!(
                        "run {} uses stage `{name}`, which is no longer configured",
                        state.id
                    )
                })
        })
        .collect::<Result<_>>()?;
    let next = state
        .stage_names
        .get(state.position)
        .map_or("?", |s| s.as_str());
    eprintln!(
        "resuming run {} ({} of {} stage(s) done, {} run(s) used) at stage {next}",
        state.id,
        state.position,
        state.stage_names.len(),
        state.runs,
    );
    run_workflow(config, &http, order, state).await
}

/// Execute a workflow from the point described by `state`, checkpointing
/// stage boundaries plus each durable event inside the active agent loop.
/// The checkpoint is removed when the pipeline finishes.
async fn run_workflow(
    config: &Config,
    http: &reqwest::Client,
    order: Vec<usize>,
    mut state: runs::RunState,
) -> Result<()> {
    let usage = model::UsageTracker::new(config.settings.run_limits(), state.usage.clone());
    if let Some(active) = &state.active_stage {
        let expected = order
            .get(state.position)
            .and_then(|index| config.stages.get(*index))
            .map(|stage| stage.name.as_str());
        if expected != Some(active.stage.as_str()) || active.run != state.runs {
            bail!(
                "run checkpoint has inconsistent active-stage progress: expected {:?} at run {}, found `{}` at run {}",
                expected,
                state.runs,
                active.stage,
                active.run,
            );
        }
    }
    // Agents can be reached from any stage, so connect their servers too.
    let needed_servers: Vec<String> = order
        .iter()
        .flat_map(|&i| config.stages[i].mcp.iter().cloned())
        .chain(config.agents.values().flat_map(|a| a.mcp.iter().cloned()))
        .collect();
    let manager = match usage
        .within_time(McpManager::connect(needed_servers, config, false))
        .await
    {
        Ok(manager) => manager,
        Err(error) => {
            state.usage = usage.snapshot();
            state.updated_at = tui::store::now_epoch();
            if let Err(save_error) = runs::save_run(&state) {
                eprintln!("⚠ cannot write run checkpoint: {save_error:#}");
            }
            eprintln!("✗ run interrupted — continue it with `soa run --resume`");
            print_usage_summary(&usage);
            return Err(error);
        }
    };
    let approvals = terminal_approvals();

    let workflow_stage_names: Vec<&str> = order
        .iter()
        .map(|&i| config.stages[i].name.as_str())
        .collect();

    let mut context = stage::PipelineContext {
        input: state.task.clone(),
        previous: state.previous.clone(),
        outputs: state.outputs.clone(),
    };
    let mut position = state.position;
    let mut runs = state.runs;
    let mut last_output = None;

    // Checkpoint the fresh run too, so a crash inside the first stage
    // still leaves the task resumable.
    checkpoint(&mut state, position, runs, &context, &usage);

    let mut result = Ok(());
    while position < order.len() {
        let stage = &config.stages[order[position]];
        let (resume_events, resuming_stage) = match &state.active_stage {
            Some(progress) => (progress.events.clone(), true),
            None => {
                runs += 1;
                if let Err(error) = runs::clear_stage_events(&state.id) {
                    eprintln!("⚠ cannot reset mid-stage checkpoint: {error:#}");
                }
                state.active_stage = Some(runs::StageProgress {
                    stage: stage.name.clone(),
                    run: runs,
                    events: Vec::new(),
                });
                checkpoint(&mut state, position, runs, &context, &usage);
                (Vec::new(), false)
            }
        };
        if runs > config.settings.max_stage_runs {
            result = Err(anyhow::anyhow!(
                "stopped after {} stage runs without finishing — likely a reprompt \
                 loop; raise settings.max_stage_runs if this is intentional",
                config.settings.max_stage_runs
            ));
            break;
        }

        eprintln!(
            "── {}run {runs} · stage {} ──",
            if resuming_stage { "resuming " } else { "" },
            stage.name
        );
        let is_first = context.previous.is_none();
        // Only offer reprompt targets that are part of the active workflow.
        let reprompt_targets: Vec<String> = stage
            .can_reprompt
            .iter()
            .filter(|t| workflow_stage_names.contains(&t.as_str()))
            .cloned()
            .collect();
        let run_id = state.id.clone();
        let stage_name = stage.name.clone();
        let event_state = std::sync::Mutex::new(&mut state);
        let on_event = |event: stage::AgentLoopEvent| {
            let usage = usage.snapshot();
            {
                let mut state = event_state.lock().expect("run checkpoint lock");
                state
                    .active_stage
                    .as_mut()
                    .expect("active stage exists while its loop runs")
                    .events
                    .push(event.clone());
                state.usage = usage.clone();
                state.updated_at = tui::store::now_epoch();
            }
            if let Err(error) =
                runs::append_stage_event(&run_id, &stage_name, runs, &event, usage)
            {
                eprintln!("⚠ cannot write mid-stage checkpoint: {error:#}");
            }
        };
        let stream = StderrStream::new();
        let on_delta = |fragment: &str| stream.push(fragment);
        let stage_result = usage
            .within_time(stage::run_stage(
                config,
                stage,
                is_first,
                &context,
                &manager,
                http,
                &usage,
                &resume_events,
                Some(&on_event),
                &reprompt_targets,
                Some(&on_delta),
                None,
                &approvals,
            ))
            .await;
        stream.flush();
        drop(event_state);
        match stage_result {
            Ok(stage::StageOutcome::Final(output)) => {
                context.record(&stage.name, output.clone());
                position += 1;
                state.active_stage = None;
                checkpoint(&mut state, position, runs, &context, &usage);
                // The output already streamed to stderr; just separate stages.
                eprintln!("\n");
                last_output = Some(output);
            }
            Ok(stage::StageOutcome::Reprompt {
                target,
                instructions,
            }) => {
                eprintln!("↩ {} reprompts {}:", stage.name, target);
                eprintln!("{instructions}\n");
                // The handoff instructions become the sender's recorded
                // output, so the target sees them as {{previous}}.
                context.record(&stage.name, instructions);
                position = workflow_stage_names
                    .iter()
                    .position(|name| *name == target)
                    .expect("reprompt targets are filtered to the active workflow");
                state.active_stage = None;
                checkpoint(&mut state, position, runs, &context, &usage);
            }
            Err(e) => {
                result = Err(e);
                break;
            }
        }
    }

    match &result {
        Ok(()) => {
            if let Err(e) = runs::remove_run(&state.id) {
                eprintln!("⚠ cannot remove finished run checkpoint: {e:#}");
            }
        }
        Err(_) => {
            checkpoint(&mut state, position, runs, &context, &usage);
            eprintln!("✗ run interrupted — continue it with `soa run --resume`");
        }
    }

    print_usage_summary(&usage);
    if result.is_ok()
        && let Some(output) = last_output
    {
        print_final(&output);
    }

    manager.shutdown().await;
    result
}

/// Rich cumulative usage for this run, including the portion before resume.
fn print_usage_summary(usage: &model::UsageTracker) {
    let lines = usage.report_lines();
    if lines.is_empty() {
        return;
    }
    eprintln!("── usage ──");
    for line in lines {
        eprintln!("{line}");
    }
}

/// Sync the checkpoint with the loop's progress and persist it. A failed
/// write warns rather than killing the run: the pipeline is still useful
/// without resumability.
fn checkpoint(
    state: &mut runs::RunState,
    position: usize,
    runs: u32,
    context: &stage::PipelineContext,
    usage: &model::UsageTracker,
) {
    state.position = position;
    state.runs = runs;
    state.previous = context.previous.clone();
    state.outputs = context.outputs.clone();
    state.usage = usage.snapshot();
    state.updated_at = tui::store::now_epoch();
    match runs::save_run(state) {
        Ok(()) if state.active_stage.is_none() => {
            if let Err(error) = runs::clear_stage_events(&state.id) {
                eprintln!("⚠ cannot clear completed stage event log: {error:#}");
            }
        }
        Ok(()) => {}
        Err(error) => {
            eprintln!("⚠ cannot write run checkpoint: {error:#}");
        }
    }
}
