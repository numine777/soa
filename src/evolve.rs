//! `soa evolve`: closed-loop input improvement.
//!
//! Where `soa reflect` proposes once and never learns whether it helped,
//! evolve runs the full loop from the self-improving-harness literature:
//! **mine** weaknesses from eval execution traces, **propose** one targeted
//! edit to an evolvable input (a stage's or agent's `system_prompt_file`,
//! or the SOA.md lessons block), **validate** by re-running every eval —
//! including held-out ones the proposer never sees — and **adopt** only on
//! strict improvement, reverting otherwise. Each verdict is appended to a
//! history log that future proposals read, so the loop does not retry what
//! already failed.
//!
//! Evals run non-interactively (approval-gated calls are denied, exactly
//! like piped `soa run`), so stages that mutate need `auto_approve`
//! patterns. Checks should be idempotent or clean up after themselves: the
//! same eval runs repeatedly against the same working directory.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::approval::Approvals;
use crate::config::{Config, Eval};
use crate::mcp::McpManager;
use crate::model::Message;
use crate::reflect;
use crate::stage::{AgentLoopEvent, PipelineContext, StageOutcome};
use crate::tui::store;

/// Caps that keep proposer prompts bounded.
const MAX_INPUT_CHARS: usize = 16_000;
const MAX_PROPOSAL_CHARS: usize = 24_000;
const MAX_HISTORY_SHOWN: usize = 10;
const MAX_CHECK_EXCERPT: usize = 800;
const MAX_OUTPUT_EXCERPT: usize = 600;
/// Equal pass counts still adopt when total tokens drop by this fraction.
const TOKEN_IMPROVEMENT: f64 = 0.10;

const SYSTEM_PROMPT: &str = "\
You improve the INPUTS of `soa`, a staged coding agent: stage and agent \
system prompts, and a persistent lessons list. You are shown the current \
inputs, the results of a scored eval suite (which tasks passed, which \
failed, and the failure evidence), and the history of previous proposals \
with their verdicts.

Reply with ONLY a JSON object, no prose or code fences, in this shape:
{
  \"target\": \"<one of the listed target ids>\",
  \"content\": \"<the complete replacement content for that target>\",
  \"rationale\": \"one short sentence: what weakness this addresses\"
}

Rules: change ONE target, minimally — the smallest edit that plausibly \
fixes an observed failure. Address the failure PATTERN, not the specific \
eval wording; a proposal that merely hardcodes an expected answer will be \
rejected by held-out evals you cannot see. Do not retry proposals the \
history shows were rejected. For the `lessons` target, content is plain \
lines, one lesson per line. If every eval passes, propose an edit that \
makes the inputs more economical (shorter prompts, fewer wasted turns) \
without losing meaning.";

/// One eval's scored result.
#[derive(Debug, Clone)]
pub struct EvalOutcome {
    pub name: String,
    pub holdout: bool,
    pub passed: bool,
    pub error: Option<String>,
    pub tokens: u64,
    /// The eval run's full ledger (per-model and per-stage breakdowns),
    /// for `soa eval`'s metrics report.
    pub usage: crate::model::UsageSnapshot,
    pub wall_ms: u64,
    pub check_excerpt: String,
    pub output_excerpt: String,
    pub signals: Vec<String>,
}

/// A suite run: every eval, scored.
#[derive(Debug, Clone, Default)]
pub struct SuiteScore {
    pub outcomes: Vec<EvalOutcome>,
}

impl SuiteScore {
    pub fn passed(&self) -> usize {
        self.outcomes.iter().filter(|o| o.passed).count()
    }

    pub fn tokens(&self) -> u64 {
        self.outcomes.iter().map(|o| o.tokens).sum()
    }
}

/// One or more repetitions of the whole suite. Agentic runs are noisy;
/// comparing repeated runs keeps a lucky (or unlucky) single sample from
/// deciding a verdict.
#[derive(Debug, Clone, Default)]
pub struct SuiteStats {
    pub runs: Vec<SuiteScore>,
}

impl SuiteStats {
    #[cfg(test)]
    pub fn single(score: SuiteScore) -> Self {
        SuiteStats { runs: vec![score] }
    }

    fn eval_count(&self) -> usize {
        self.runs.first().map_or(0, |run| run.outcomes.len())
    }

    /// How many of the repetitions passed the eval at `index`.
    fn passes(&self, index: usize) -> usize {
        self.runs
            .iter()
            .filter(|run| run.outcomes.get(index).is_some_and(|o| o.passed))
            .count()
    }

    fn pass_rate(&self, index: usize) -> f64 {
        self.passes(index) as f64 / self.runs.len().max(1) as f64
    }

    /// Every eval passed in every repetition.
    fn all_green(&self) -> bool {
        !self.runs.is_empty()
            && self
                .runs
                .iter()
                .all(|run| run.passed() == run.outcomes.len())
    }

    fn mean_tokens(&self) -> f64 {
        if self.runs.is_empty() {
            return 0.0;
        }
        self.runs.iter().map(|run| run.tokens() as f64).sum::<f64>() / self.runs.len() as f64
    }

    /// Sample variance of per-repetition suite tokens (0 with one run).
    fn tokens_variance(&self) -> f64 {
        if self.runs.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_tokens();
        self.runs
            .iter()
            .map(|run| (run.tokens() as f64 - mean).powi(2))
            .sum::<f64>()
            / (self.runs.len() - 1) as f64
    }

    /// A merged view for the proposer prompt: per eval, the first failing
    /// repetition's evidence when any repetition failed (flaky failures are
    /// the interesting signal), otherwise the last repetition's outcome.
    pub fn representative(&self) -> SuiteScore {
        let mut outcomes = Vec::new();
        for index in 0..self.eval_count() {
            let failing = self
                .runs
                .iter()
                .filter_map(|run| run.outcomes.get(index))
                .find(|o| !o.passed);
            let outcome =
                failing.or_else(|| self.runs.last().and_then(|run| run.outcomes.get(index)));
            if let Some(outcome) = outcome {
                outcomes.push(outcome.clone());
            }
        }
        SuiteScore { outcomes }
    }

    pub fn describe(&self) -> String {
        let evals = self.eval_count();
        let fully_passing = (0..evals)
            .filter(|&index| self.passes(index) == self.runs.len())
            .count();
        if self.runs.len() == 1 {
            format!(
                "{fully_passing}/{evals} passing · {}",
                crate::model::fmt_tokens(self.mean_tokens().round() as u64),
            )
        } else {
            format!(
                "{fully_passing}/{evals} passing in all {} run(s) · mean {}",
                self.runs.len(),
                crate::model::fmt_tokens(self.mean_tokens().round() as u64),
            )
        }
    }
}

/// Whether a candidate suite run justifies keeping the proposal.
#[derive(Debug, PartialEq)]
pub enum Verdict {
    Adopt(String),
    Reject(String),
}

/// Strict improvement: no eval's pass rate may drop (held-out evals
/// included), and the candidate must either raise a pass rate or — when the
/// suite was already green — cut mean suite tokens meaningfully. With
/// repeated runs the token cut must also clear the measured run-to-run
/// noise, so a lucky sample cannot masquerade as an economy win; with one
/// run per side this reduces to the plain 10% threshold.
pub fn compare(baseline: &SuiteStats, candidate: &SuiteStats) -> Verdict {
    let evals = baseline.eval_count().min(candidate.eval_count());
    let name = |index: usize| -> (&str, bool) {
        let outcome = &baseline.runs[0].outcomes[index];
        (outcome.name.as_str(), outcome.holdout)
    };
    for index in 0..evals {
        if candidate.pass_rate(index) < baseline.pass_rate(index) {
            let (name, holdout) = name(index);
            return Verdict::Reject(format!(
                "regressed `{name}`{} (passed {}/{} → {}/{})",
                if holdout { " (holdout)" } else { "" },
                baseline.passes(index),
                baseline.runs.len(),
                candidate.passes(index),
                candidate.runs.len(),
            ));
        }
    }
    let improved: Vec<&str> = (0..evals)
        .filter(|&index| candidate.pass_rate(index) > baseline.pass_rate(index))
        .map(|index| name(index).0)
        .collect();
    if !improved.is_empty() {
        return Verdict::Adopt(format!("newly passing: {}", improved.join(", ")));
    }
    let (base, cand) = (baseline.mean_tokens(), candidate.mean_tokens());
    if baseline.all_green() && base > 0.0 {
        // The drop must clear both the 10% floor and (with repetitions) a
        // one-sided 95% bound on the difference of means.
        let noise = 1.645
            * (baseline.tokens_variance() / baseline.runs.len().max(1) as f64
                + candidate.tokens_variance() / candidate.runs.len().max(1) as f64)
                .sqrt();
        let required = (base * TOKEN_IMPROVEMENT).max(noise);
        if cand < base - required {
            return Verdict::Adopt(format!(
                "same passes, mean tokens {} → {}",
                crate::model::fmt_tokens(base.round() as u64),
                crate::model::fmt_tokens(cand.round() as u64),
            ));
        }
    }
    Verdict::Reject("no eval newly passes and tokens did not improve".to_string())
}

/// An input the proposer may rewrite. Nothing outside this set is ever
/// touched — in particular, never the config file itself.
#[derive(Debug, Clone, PartialEq)]
pub enum Target {
    /// The managed lessons block of the project instructions file.
    Lessons,
    /// A `system_prompt_file` referenced by a stage or agent.
    PromptFile { id: String, path: PathBuf },
}

impl Target {
    pub fn id(&self) -> &str {
        match self {
            Target::Lessons => "lessons",
            Target::PromptFile { id, .. } => id,
        }
    }
}

/// The evolvable surface: every file-based system prompt in the config,
/// plus the lessons block.
pub fn evolvable_targets(config: &Config) -> Vec<Target> {
    let mut targets = vec![Target::Lessons];
    let resolve = |path: &PathBuf| {
        if path.is_absolute() {
            path.clone()
        } else {
            config.base_dir.join(path)
        }
    };
    for stage in &config.stages {
        if let Some(path) = &stage.system_prompt_file {
            targets.push(Target::PromptFile {
                id: format!("stage:{}", stage.name),
                path: resolve(path),
            });
        }
    }
    for (name, agent) in &config.agents {
        if let Some(path) = &agent.system_prompt_file {
            targets.push(Target::PromptFile {
                id: format!("agent:{name}"),
                path: resolve(path),
            });
        }
    }
    targets
}

/// What the proposer model returned.
#[derive(Debug, Deserialize)]
struct Proposal {
    target: String,
    content: String,
    #[serde(default)]
    rationale: String,
}

fn parse_proposal(text: &str) -> Result<Proposal> {
    let start = text.find('{').context("no JSON object in the reply")?;
    let end = text.rfind('}').context("no JSON object in the reply")?;
    if end <= start {
        bail!("no JSON object in the reply");
    }
    let proposal: Proposal = serde_json::from_str(&text[start..=end])?;
    if proposal.content.trim().is_empty() {
        bail!("proposal content is empty");
    }
    if proposal.content.chars().count() > MAX_PROPOSAL_CHARS {
        bail!(
            "proposal content is too long ({} chars, cap {MAX_PROPOSAL_CHARS})",
            proposal.content.chars().count()
        );
    }
    Ok(proposal)
}

/// The pre-application state of a target, for revert.
enum Backup {
    Lessons(Vec<String>),
    File { path: PathBuf, content: String },
}

fn read_target(config: &Config, target: &Target) -> Result<String> {
    match target {
        Target::Lessons => Ok(reflect::read_lessons(&reflect::lessons_file(config)).join("\n")),
        Target::PromptFile { path, .. } => std::fs::read_to_string(path)
            .with_context(|| format!("cannot read {}", path.display())),
    }
}

fn apply_target(config: &Config, target: &Target, content: &str) -> Result<Backup> {
    match target {
        Target::Lessons => {
            let path = reflect::lessons_file(config);
            let previous = reflect::read_lessons(&path);
            let lessons = reflect::validate_lessons(
                content.lines().map(|l| l.trim_start_matches("- ").to_string()).collect(),
            );
            let current = std::fs::read_to_string(&path).unwrap_or_default();
            std::fs::write(&path, reflect::replace_lessons_block(&current, &lessons))
                .with_context(|| format!("cannot write {}", path.display()))?;
            Ok(Backup::Lessons(previous))
        }
        Target::PromptFile { path, .. } => {
            let previous = std::fs::read_to_string(path)
                .with_context(|| format!("cannot read {}", path.display()))?;
            std::fs::write(path, content)
                .with_context(|| format!("cannot write {}", path.display()))?;
            Ok(Backup::File {
                path: path.clone(),
                content: previous,
            })
        }
    }
}

fn revert(config: &Config, backup: Backup) -> Result<()> {
    match backup {
        Backup::Lessons(lessons) => {
            let path = reflect::lessons_file(config);
            let current = std::fs::read_to_string(&path).unwrap_or_default();
            std::fs::write(&path, reflect::replace_lessons_block(&current, &lessons))
                .with_context(|| format!("cannot write {}", path.display()))
        }
        Backup::File { path, content } => std::fs::write(&path, content)
            .with_context(|| format!("cannot write {}", path.display())),
    }
}

/// One line per attempted proposal, appended to
/// `<data dir>/evolve_history.jsonl` and fed back to future proposals.
#[derive(Debug, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub at: u64,
    pub target: String,
    pub rationale: String,
    pub adopted: bool,
    pub reason: String,
    pub baseline: String,
    pub candidate: String,
}

fn history_path() -> PathBuf {
    store::data_dir().join("evolve_history.jsonl")
}

fn load_history() -> Vec<HistoryEntry> {
    let Ok(raw) = std::fs::read_to_string(history_path()) else {
        return Vec::new();
    };
    raw.lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Eval execution
// ---------------------------------------------------------------------------

/// Trace signals mined from one eval run's event log: tool failures,
/// denials, and reprompt bounces — the evidence the proposer reasons from.
fn mine_signals(events: &[AgentLoopEvent], reprompts: usize) -> Vec<String> {
    let mut signals = Vec::new();
    let mut round_tools: Vec<String> = Vec::new();
    for event in events {
        match event {
            AgentLoopEvent::Assistant { tool_calls, .. } => {
                round_tools = tool_calls
                    .iter()
                    .map(|call| call.function.name.clone())
                    .collect();
            }
            AgentLoopEvent::ToolResult {
                call_index,
                content,
            } => {
                if content.starts_with("ERROR") || content.starts_with("DENIED") {
                    let tool = round_tools
                        .get(*call_index)
                        .map(String::as_str)
                        .unwrap_or("?");
                    signals.push(format!("{tool}: {}", reflect::excerpt(content, 160)));
                }
            }
            _ => {}
        }
    }
    let turns = events
        .iter()
        .filter(|event| matches!(event, AgentLoopEvent::Assistant { .. }))
        .count();
    if reprompts > 0 {
        signals.push(format!("review bounced the work back {reprompts} time(s)"));
    }
    if turns > 0 {
        signals.push(format!("{turns} model turn(s) used"));
    }
    signals.truncate(12);
    signals
}

/// Run one eval's workflow in-memory (no run checkpoints — evolve runs are
/// ephemeral) and grade it with the check command.
async fn run_eval(
    config: &Config,
    eval: &Eval,
    mcp: &McpManager,
    http: &reqwest::Client,
    approvals: &Approvals,
) -> Result<EvalOutcome> {
    let started = std::time::Instant::now();
    let task = eval.resolve_task(&config.base_dir)?;
    let order = config.resolve_workflow(eval.workflow.as_deref())?;
    if order.is_empty() {
        bail!("eval `{}`: the selected workflow is empty", eval.name);
    }
    let workflow_stage_names: Vec<&str> =
        order.iter().map(|&i| config.stages[i].name.as_str()).collect();

    let usage = crate::model::UsageTracker::new(
        config.settings.run_limits(),
        crate::model::UsageSnapshot::default(),
    );
    let mut context = PipelineContext::new(&task);
    let events: std::sync::Mutex<Vec<AgentLoopEvent>> = std::sync::Mutex::new(Vec::new());
    let on_event = |event: AgentLoopEvent| events.lock().unwrap().push(event);

    let mut position = 0usize;
    let mut runs = 0u32;
    let mut reprompts = 0usize;
    let mut last_output = String::new();
    let mut error = None;
    while position < order.len() {
        let stage = &config.stages[order[position]];
        runs += 1;
        if runs > config.settings.max_stage_runs {
            error = Some(format!(
                "stopped after {} stage runs without finishing",
                config.settings.max_stage_runs
            ));
            break;
        }
        let is_first = context.previous.is_none();
        let reprompt_targets: Vec<String> = stage
            .can_reprompt
            .iter()
            .filter(|t| workflow_stage_names.contains(&t.as_str()))
            .cloned()
            .collect();
        let result = usage
            .within_time(crate::stage::run_stage(
                config,
                stage,
                is_first,
                &context,
                mcp,
                http,
                &usage,
                &[],
                Some(&on_event),
                &reprompt_targets,
                None,
                None,
                approvals,
            ))
            .await;
        match result {
            Ok(StageOutcome::Final(output)) => {
                context.record(&stage.name, output.clone());
                last_output = output;
                position += 1;
            }
            Ok(StageOutcome::Reprompt {
                target,
                instructions,
            }) => {
                reprompts += 1;
                context.record(&stage.name, instructions);
                position = workflow_stage_names
                    .iter()
                    .position(|name| *name == target)
                    .expect("reprompt targets are filtered to the active workflow");
            }
            Err(e) => {
                error = Some(format!("{e:#}"));
                break;
            }
        }
    }

    // Grade with the check command; the run's final output is in the
    // environment so answer-shaped evals can be graded without files.
    let (check_passed, check_excerpt) = if error.is_some() {
        (false, String::new())
    } else {
        run_check(config, eval, &last_output).await
    };
    let events = events.into_inner().unwrap();
    let snapshot = usage.snapshot();
    Ok(EvalOutcome {
        name: eval.name.clone(),
        holdout: eval.holdout,
        passed: error.is_none() && check_passed,
        error,
        tokens: snapshot
            .models
            .values()
            .map(|m| m.prompt_tokens + m.completion_tokens)
            .sum(),
        usage: snapshot,
        wall_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        check_excerpt,
        output_excerpt: reflect::excerpt(&last_output, MAX_OUTPUT_EXCERPT),
        signals: mine_signals(&events, reprompts),
    })
}

/// Like [`reflect::excerpt`] but keeps the END of the text — build and
/// test logs put the verdict last, so the tail is the diagnostic part.
fn tail_excerpt(text: &str, max_chars: usize) -> String {
    let squashed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let count = squashed.chars().count();
    if count <= max_chars {
        squashed
    } else {
        let kept: String = squashed.chars().skip(count - max_chars).collect();
        format!("…{kept}")
    }
}

async fn run_check(config: &Config, eval: &Eval, output: &str) -> (bool, String) {
    let timeout = std::time::Duration::from_secs(
        eval.timeout_secs.unwrap_or(config.settings.shell_timeout_secs),
    );
    let child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(&eval.check)
        .env("SOA_EVAL", &eval.name)
        .env("SOA_OUTPUT", reflect::excerpt(output, 100_000))
        .kill_on_drop(true)
        .output();
    match tokio::time::timeout(timeout, child).await {
        Err(_) => (
            false,
            format!("check timed out after {}s", timeout.as_secs()),
        ),
        Ok(Err(e)) => (false, format!("check failed to run: {e}")),
        Ok(Ok(result)) => {
            let mut text = String::from_utf8_lossy(&result.stdout).trim().to_string();
            let stderr = String::from_utf8_lossy(&result.stderr);
            if !stderr.trim().is_empty() {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(stderr.trim());
            }
            (
                result.status.success(),
                tail_excerpt(&text, MAX_CHECK_EXCERPT),
            )
        }
    }
}

/// Repeat the whole suite `repetitions` times for a noise-aware verdict.
async fn run_suite_repeated(
    config: &Config,
    mcp: &McpManager,
    http: &reqwest::Client,
    approvals: &Approvals,
    repetitions: u32,
) -> Result<SuiteStats> {
    let mut runs = Vec::new();
    for round in 1..=repetitions.max(1) {
        if repetitions > 1 {
            eprintln!("  (suite run {round}/{repetitions})");
        }
        runs.push(run_suite(config, &[], mcp, http, approvals).await?);
    }
    Ok(SuiteStats { runs })
}

/// Run the configured evals (all of them, or just the names in `filter`)
/// and score each one. Shared by `soa evolve` and `soa eval`.
pub(crate) async fn run_suite(
    config: &Config,
    filter: &[String],
    mcp: &McpManager,
    http: &reqwest::Client,
    approvals: &Approvals,
) -> Result<SuiteScore> {
    let mut outcomes = Vec::new();
    for eval in &config.evals {
        if !filter.is_empty() && !filter.contains(&eval.name) {
            continue;
        }
        eprint!("  eval `{}` … ", eval.name);
        let outcome = run_eval(config, eval, mcp, http, approvals).await?;
        eprintln!(
            "{}{}",
            if outcome.passed { "pass" } else { "FAIL" },
            outcome
                .error
                .as_deref()
                .map(|e| format!(" ({})", reflect::excerpt(e, 120)))
                .unwrap_or_default(),
        );
        // A check failure without the check's own words is undebuggable —
        // show the tail of its output right where the FAIL is reported.
        if !outcome.passed && outcome.error.is_none() && !outcome.check_excerpt.is_empty() {
            eprintln!("    check: {}", tail_excerpt(&outcome.check_excerpt, 400));
        }
        outcomes.push(outcome);
    }
    Ok(SuiteScore { outcomes })
}

// ---------------------------------------------------------------------------
// Proposal
// ---------------------------------------------------------------------------

fn build_proposer_prompt(
    config: &Config,
    targets: &[Target],
    baseline: &SuiteScore,
    history: &[HistoryEntry],
) -> Result<String> {
    let mut prompt = String::new();
    prompt.push_str("# Evolvable inputs\n\n");
    for target in targets {
        let content = read_target(config, target).unwrap_or_default();
        prompt.push_str(&format!(
            "## target: {}\n```\n{}\n```\n\n",
            target.id(),
            reflect::excerpt(&content, MAX_INPUT_CHARS),
        ));
    }

    prompt.push_str("# Eval results\n\n");
    let holdouts = baseline.outcomes.iter().filter(|o| o.holdout).count();
    if holdouts > 0 {
        prompt.push_str(&format!(
            "({holdouts} additional held-out eval(s) exist; their content is \
             hidden and your proposal must not regress them.)\n\n"
        ));
    }
    for outcome in baseline.outcomes.iter().filter(|o| !o.holdout) {
        prompt.push_str(&format!(
            "## eval `{}` — {}\n",
            outcome.name,
            if outcome.passed { "PASS" } else { "FAIL" }
        ));
        if let Some(error) = &outcome.error {
            prompt.push_str(&format!("run error: {error}\n"));
        }
        if !outcome.passed && !outcome.check_excerpt.is_empty() {
            prompt.push_str(&format!("check output:\n{}\n", outcome.check_excerpt));
        }
        if !outcome.output_excerpt.is_empty() {
            prompt.push_str(&format!("final output: {}\n", outcome.output_excerpt));
        }
        for signal in &outcome.signals {
            prompt.push_str(&format!("- {signal}\n"));
        }
        prompt.push('\n');
    }

    if !history.is_empty() {
        prompt.push_str("# Previous proposals (do not retry rejected ones)\n\n");
        for entry in history.iter().rev().take(MAX_HISTORY_SHOWN) {
            prompt.push_str(&format!(
                "- [{}] {}: {} — {}\n",
                if entry.adopted { "adopted" } else { "rejected" },
                entry.target,
                reflect::excerpt(&entry.rationale, 120),
                reflect::excerpt(&entry.reason, 120),
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str(&format!(
        "Valid targets: {}\n",
        targets
            .iter()
            .map(Target::id)
            .collect::<Vec<_>>()
            .join(", "),
    ));
    Ok(prompt)
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

pub async fn run(
    config: &Config,
    iterations: u32,
    dry_run: bool,
    model_override: Option<&str>,
    suite_runs: u32,
) -> Result<()> {
    if config.evals.is_empty() {
        bail!(
            "no [[eval]] entries configured — soa evolve needs scored tasks \
             to measure improvement against"
        );
    }
    let targets = evolvable_targets(config);
    if targets.len() == 1 {
        eprintln!(
            "⚠ only the lessons block is evolvable — move stage/agent prompts \
             to system_prompt_file to widen the surface"
        );
    }
    if !config.evals.iter().any(|e| e.holdout) {
        eprintln!(
            "⚠ no holdout evals: proposals can overfit to the suite — mark at \
             least one eval `holdout = true`"
        );
    }
    let model = match model_override {
        Some(name) => name.to_string(),
        None => config
            .settings
            .evolve_model
            .clone()
            .or_else(|| config.settings.reflect_model.clone())
            .or_else(|| config.stages.first().map(|s| s.model.clone()))
            .context("no model to propose with (set settings.evolve_model)")?,
    };

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(
            config.settings.request_timeout_secs,
        ))
        .build()
        .context("failed to build HTTP client")?;
    let servers: Vec<String> = config
        .stages
        .iter()
        .flat_map(|s| s.mcp.iter().cloned())
        .chain(config.agents.values().flat_map(|a| a.mcp.iter().cloned()))
        .collect();
    let mcp = McpManager::connect(servers, config, false).await?;
    // Evolve is autonomous: approval-gated calls are denied, as in piped
    // `soa run`. Stages that must mutate need auto_approve patterns.
    let approvals = Approvals::non_interactive();

    eprintln!("── evolve baseline ──");
    let mut baseline = run_suite_repeated(config, &mcp, &http, &approvals, suite_runs).await?;
    eprintln!("baseline: {}", baseline.describe());

    let proposer_usage = crate::model::UsageTracker::unlimited();
    let client =
        crate::stage::build_model_client(config, &model, None, None, &http, &proposer_usage, None)?;

    let mut adopted: Vec<String> = Vec::new();
    for iteration in 1..=iterations {
        eprintln!("\n── evolve iteration {iteration}/{iterations} ──");
        let history = load_history();
        let representative = baseline.representative();
        let prompt = build_proposer_prompt(config, &targets, &representative, &history)?;
        let messages = vec![
            Message::System {
                content: SYSTEM_PROMPT.to_string(),
            },
            Message::User { content: prompt },
        ];
        let reply = client.complete(&messages, &[]).await?;
        let text = reply.content.unwrap_or_default();
        let proposal = parse_proposal(&text)
            .with_context(|| format!("proposer returned unusable output:\n{}", reflect::excerpt(&text, 400)))?;
        let Some(target) = targets.iter().find(|t| t.id() == proposal.target) else {
            bail!(
                "proposer picked unknown target `{}` (valid: {})",
                proposal.target,
                targets.iter().map(Target::id).collect::<Vec<_>>().join(", "),
            );
        };
        eprintln!(
            "proposal: {} — {}",
            target.id(),
            reflect::excerpt(&proposal.rationale, 200),
        );
        if dry_run {
            eprintln!("\n{}\n\n(dry run — nothing applied)", proposal.content);
            mcp.shutdown().await;
            return Ok(());
        }

        let backup = apply_target(config, target, &proposal.content)?;
        let candidate = run_suite_repeated(config, &mcp, &http, &approvals, suite_runs).await?;
        eprintln!("candidate: {}", candidate.describe());
        let verdict = compare(&baseline, &candidate);
        let (was_adopted, reason) = match &verdict {
            Verdict::Adopt(reason) => {
                eprintln!("✔ adopted ({reason})");
                adopted.push(format!("{} — {}", target.id(), proposal.rationale));
                (true, reason.clone())
            }
            Verdict::Reject(reason) => {
                eprintln!("✗ reverted ({reason})");
                revert(config, backup)?;
                (false, reason.clone())
            }
        };
        let entry = HistoryEntry {
            at: store::now_epoch(),
            target: target.id().to_string(),
            rationale: proposal.rationale.clone(),
            adopted: was_adopted,
            reason,
            baseline: baseline.describe(),
            candidate: candidate.describe(),
        };
        if let Err(e) = crate::persistence::append_json_line(&history_path(), &entry) {
            eprintln!("⚠ cannot record evolve history: {e:#}");
        }
        if was_adopted {
            baseline = candidate;
        }
    }

    mcp.shutdown().await;
    eprintln!("\n── evolve summary ──");
    eprintln!("final: {}", baseline.describe());
    if adopted.is_empty() {
        eprintln!("no proposals adopted — inputs are unchanged");
    } else {
        for change in &adopted {
            eprintln!("adopted: {change}");
        }
        eprintln!("review the changes with `git diff`, revert with `git checkout`");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(name: &str, passed: bool, holdout: bool, tokens: u64) -> EvalOutcome {
        EvalOutcome {
            name: name.to_string(),
            holdout,
            passed,
            error: None,
            tokens,
            usage: crate::model::UsageSnapshot::default(),
            wall_ms: 0,
            check_excerpt: String::new(),
            output_excerpt: String::new(),
            signals: Vec::new(),
        }
    }

    fn suite(outcomes: Vec<EvalOutcome>) -> SuiteStats {
        SuiteStats::single(SuiteScore { outcomes })
    }

    #[test]
    fn verdicts_require_strict_improvement() {
        // A newly passing eval adopts.
        let baseline = suite(vec![outcome("a", false, false, 100)]);
        let candidate = suite(vec![outcome("a", true, false, 120)]);
        assert!(matches!(compare(&baseline, &candidate), Verdict::Adopt(_)));

        // Regressing anything — holdouts especially — rejects, even when
        // another eval improves.
        let baseline = suite(vec![
            outcome("a", false, false, 100),
            outcome("h", true, true, 100),
        ]);
        let candidate = suite(vec![
            outcome("a", true, false, 100),
            outcome("h", false, true, 100),
        ]);
        match compare(&baseline, &candidate) {
            Verdict::Reject(reason) => assert!(reason.contains("holdout"), "{reason}"),
            verdict => panic!("expected reject, got {verdict:?}"),
        }

        // All-green suites adopt on a meaningful token reduction only.
        let baseline = suite(vec![outcome("a", true, false, 1000)]);
        let candidate = suite(vec![outcome("a", true, false, 850)]);
        assert!(matches!(compare(&baseline, &candidate), Verdict::Adopt(_)));
        let candidate = suite(vec![outcome("a", true, false, 950)]);
        assert!(matches!(compare(&baseline, &candidate), Verdict::Reject(_)));

        // A failing suite with no newly-passing eval rejects even if
        // tokens drop (economy must not mask stagnation).
        let baseline = suite(vec![outcome("a", false, false, 1000)]);
        let candidate = suite(vec![outcome("a", false, false, 100)]);
        assert!(matches!(compare(&baseline, &candidate), Verdict::Reject(_)));
    }

    #[test]
    fn repeated_runs_gate_verdicts_on_pass_rate_and_noise() {
        let stats = |runs: Vec<(bool, u64)>| SuiteStats {
            runs: runs
                .into_iter()
                .map(|(passed, tokens)| SuiteScore {
                    outcomes: vec![outcome("a", passed, false, tokens)],
                })
                .collect(),
        };

        // A pass-rate drop (2/2 → 1/2) rejects even though no single run
        // pair shows a clean regression.
        let baseline = stats(vec![(true, 1000), (true, 1000)]);
        let flaky = stats(vec![(true, 900), (false, 900)]);
        match compare(&baseline, &flaky) {
            Verdict::Reject(reason) => assert!(reason.contains("2/2 → 1/2"), "{reason}"),
            verdict => panic!("expected reject, got {verdict:?}"),
        }

        // A pass-rate improvement adopts.
        let baseline = stats(vec![(false, 1000), (true, 1000)]);
        let better = stats(vec![(true, 1000), (true, 1000)]);
        assert!(matches!(compare(&baseline, &better), Verdict::Adopt(_)));

        // Noisy baseline (1000 vs 2000): a drop that beats the 10% floor
        // but sits inside the measured noise band is rejected…
        let baseline = stats(vec![(true, 1000), (true, 2000)]);
        let inside_noise = stats(vec![(true, 1300), (true, 1300)]);
        assert!(matches!(
            compare(&baseline, &inside_noise),
            Verdict::Reject(_)
        ));
        // …while a drop that clears the noise band is adopted.
        let clear_win = stats(vec![(true, 500), (true, 500)]);
        assert!(matches!(compare(&baseline, &clear_win), Verdict::Adopt(_)));
    }

    #[test]
    fn tail_excerpt_keeps_the_end_of_long_output() {
        assert_eq!(tail_excerpt("short output", 100), "short output");
        let long = format!("{} FAILED: bazel test //x:y", "noise ".repeat(200));
        let tail = tail_excerpt(&long, 40);
        assert!(tail.starts_with('…'), "{tail}");
        assert!(tail.ends_with("FAILED: bazel test //x:y"), "{tail}");
    }

    #[test]
    fn proposals_parse_and_reject_unknown_targets() {
        let proposal = parse_proposal(
            r#"Here you go: {"target": "stage:implement", "content": "Be careful.", "rationale": "less rework"}"#,
        )
        .unwrap();
        assert_eq!(proposal.target, "stage:implement");
        assert_eq!(proposal.rationale, "less rework");

        assert!(parse_proposal("no json here").is_err());
        assert!(parse_proposal(r#"{"target": "x", "content": ""}"#).is_err());
        let oversized = format!(
            r#"{{"target": "x", "content": "{}"}}"#,
            "y".repeat(MAX_PROPOSAL_CHARS + 1)
        );
        assert!(parse_proposal(&oversized).is_err());
    }

    #[test]
    fn evolvable_surface_is_prompt_files_and_lessons_only() {
        let mut config: Config = toml::from_str(
            r#"
            [providers.p]
            base_url = "http://localhost/v1"

            [models.m]
            provider = "p"
            model = "x"

            [agents.helper]
            model = "m"
            description = "helps"
            system_prompt_file = "prompts/helper.md"

            [[stage]]
            name = "plan"
            model = "m"
            system_prompt_file = "prompts/plan.md"

            [[stage]]
            name = "inline"
            model = "m"
            system_prompt = "inline prompts are not evolvable"
            "#,
        )
        .unwrap();
        config.base_dir = PathBuf::from("/cfg");
        let targets = evolvable_targets(&config);
        let ids: Vec<&str> = targets.iter().map(Target::id).collect();
        assert_eq!(ids, vec!["lessons", "stage:plan", "agent:helper"]);
        // Relative prompt paths resolve against the config directory.
        assert!(targets.iter().any(|t| matches!(
            t,
            Target::PromptFile { path, .. } if path == &PathBuf::from("/cfg/prompts/plan.md")
        )));
    }

    /// One-shot OpenAI-compatible mock: serves the given bodies in order.
    fn mock_server(responses: Vec<&'static str>) -> std::net::SocketAddr {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for body in responses {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buffer = [0u8; 65536];
                let _ = stream.read(&mut buffer);
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len(),
                );
            }
        });
        addr
    }

    #[tokio::test]
    async fn run_eval_scores_pass_and_fail_through_the_check() {
        let addr = mock_server(vec![
            r#"{"choices":[{"message":{"content":"the answer is 42"},"finish_reason":"stop"}]}"#,
            r#"{"choices":[{"message":{"content":"no idea"},"finish_reason":"stop"}]}"#,
        ]);
        let config: Config = toml::from_str(&format!(
            r#"
            [providers.mock]
            base_url = "http://{addr}/v1"
            stream = false

            [models.m]
            provider = "mock"
            model = "x"

            [[stage]]
            name = "answer"
            model = "m"

            [[eval]]
            name = "graded"
            task = "what is the answer?"
            check = "echo \"$SOA_OUTPUT\" | grep -q 42"
            "#,
        ))
        .unwrap();
        let http = reqwest::Client::new();
        let mcp = McpManager::default();
        let approvals = Approvals::non_interactive();

        // First run: the model says 42; the check greps it from SOA_OUTPUT.
        let outcome = run_eval(&config, &config.evals[0], &mcp, &http, &approvals)
            .await
            .unwrap();
        assert!(outcome.passed, "{outcome:?}");
        assert!(outcome.output_excerpt.contains("42"));

        // Second run: the model answers wrong; the same check fails.
        let outcome = run_eval(&config, &config.evals[0], &mcp, &http, &approvals)
            .await
            .unwrap();
        assert!(!outcome.passed);
        assert!(outcome.error.is_none(), "a failing check is not a run error");
    }

    #[test]
    fn apply_and_revert_round_trip_prompt_files() {
        let dir = std::env::temp_dir().join(format!("soa-evolve-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plan.md");
        std::fs::write(&path, "original prompt").unwrap();
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
        let target = Target::PromptFile {
            id: "stage:plan".to_string(),
            path: path.clone(),
        };
        let backup = apply_target(&config, &target, "improved prompt").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "improved prompt");
        revert(&config, backup).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "original prompt");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
