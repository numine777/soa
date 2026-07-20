//! `soa eval` — run the `[[eval]]` suite as a measurement instrument and
//! emit machine-readable token metrics.
//!
//! Where `soa evolve` runs the suite to *change* inputs, `soa eval` runs it
//! to *measure* them: N repetitions of every (or a filtered set of) evals,
//! with per-eval and per-suite aggregates of pass rate, tokens, cost, turns,
//! tool calls, and wall time. JSON goes to stdout so comparisons between two
//! configurations (a skill on/off, a rules edit, a new CLI tool) can be
//! scripted; progress and a human summary go to stderr.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::approval::Approvals;
use crate::config::Config;
use crate::evolve::{EvalOutcome, run_suite};
use crate::mcp::McpManager;
use crate::model::{ScopeUsage, fmt_tokens};

/// Metrics for one execution of one eval.
#[derive(Debug, Serialize)]
struct RunMetrics {
    passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_read_tokens: u64,
    reasoning_tokens: u64,
    total_tokens: u64,
    /// Known model spend; requests without configured prices are counted
    /// in `unpriced_requests` instead of silently costing $0.
    cost_usd: f64,
    unpriced_requests: u64,
    unreported_requests: u64,
    turns: u64,
    tool_calls: u64,
    wall_ms: u64,
    /// Per-stage/agent breakdown — where the tokens actually went.
    scopes: BTreeMap<String, ScopeUsage>,
}

impl RunMetrics {
    fn from_outcome(outcome: &EvalOutcome) -> Self {
        let models = outcome.usage.models.values();
        let (mut prompt, mut completion, mut cached, mut reasoning) = (0u64, 0u64, 0u64, 0u64);
        let (mut cost, mut unpriced, mut unreported) = (0.0f64, 0u64, 0u64);
        for usage in models {
            prompt += usage.prompt_tokens;
            completion += usage.completion_tokens;
            cached += usage.cache_read_tokens;
            reasoning += usage.reasoning_tokens;
            cost += usage.cost_usd;
            unpriced += usage.unpriced_requests;
            unreported += usage.unreported_requests;
        }
        RunMetrics {
            passed: outcome.passed,
            error: outcome.error.clone(),
            prompt_tokens: prompt,
            completion_tokens: completion,
            cache_read_tokens: cached,
            reasoning_tokens: reasoning,
            total_tokens: prompt + completion,
            cost_usd: cost,
            unpriced_requests: unpriced,
            unreported_requests: unreported,
            turns: outcome.usage.scopes.values().map(|s| s.turns).sum(),
            tool_calls: outcome.usage.scopes.values().map(|s| s.tool_calls).sum(),
            wall_ms: outcome.wall_ms,
            scopes: outcome.usage.scopes.clone(),
        }
    }
}

/// Mean/min/max style aggregate over a set of executions.
#[derive(Debug, Serialize)]
struct Aggregate {
    runs: usize,
    pass_rate: f64,
    mean_total_tokens: f64,
    min_total_tokens: u64,
    max_total_tokens: u64,
    mean_cost_usd: f64,
    mean_turns: f64,
    mean_tool_calls: f64,
    mean_wall_ms: f64,
}

impl Aggregate {
    fn over(metrics: &[&RunMetrics]) -> Self {
        let n = metrics.len().max(1) as f64;
        let mean = |f: &dyn Fn(&RunMetrics) -> f64| metrics.iter().map(|m| f(m)).sum::<f64>() / n;
        Aggregate {
            runs: metrics.len(),
            pass_rate: metrics.iter().filter(|m| m.passed).count() as f64 / n,
            mean_total_tokens: mean(&|m| m.total_tokens as f64),
            min_total_tokens: metrics.iter().map(|m| m.total_tokens).min().unwrap_or(0),
            max_total_tokens: metrics.iter().map(|m| m.total_tokens).max().unwrap_or(0),
            mean_cost_usd: mean(&|m| m.cost_usd),
            mean_turns: mean(&|m| m.turns as f64),
            mean_tool_calls: mean(&|m| m.tool_calls as f64),
            mean_wall_ms: mean(&|m| m.wall_ms as f64),
        }
    }
}

#[derive(Debug, Serialize)]
struct EvalReport {
    name: String,
    holdout: bool,
    runs: Vec<RunMetrics>,
    aggregate: Aggregate,
}

#[derive(Debug, Serialize)]
struct SuiteReport {
    config: String,
    suite_runs: u32,
    evals: Vec<EvalReport>,
    /// Aggregate over every (eval, run) execution in the report.
    aggregate: Aggregate,
}

pub async fn run(
    config: &Config,
    config_path: &std::path::Path,
    suite_runs: u32,
    filter: &[String],
) -> Result<()> {
    if config.evals.is_empty() {
        bail!("no [[eval]] entries configured — soa eval needs scored tasks to measure");
    }
    if suite_runs == 0 {
        bail!("--runs must be at least 1");
    }
    for name in filter {
        if !config.evals.iter().any(|e| e.name == *name) {
            bail!("no eval named `{name}` (see the [[eval]] entries in the config)");
        }
    }

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
    // Measurement runs are autonomous, like evolve: approval-gated calls
    // are denied unless auto_approve patterns cover them.
    let approvals = Approvals::non_interactive();

    // outcomes[eval name] preserves suite order via first insertion.
    let mut order: Vec<String> = Vec::new();
    let mut outcomes: BTreeMap<String, Vec<EvalOutcome>> = BTreeMap::new();
    for round in 1..=suite_runs {
        eprintln!("── suite run {round}/{suite_runs} ──");
        let score = run_suite(config, filter, &mcp, &http, &approvals).await?;
        for outcome in score.outcomes {
            if !order.contains(&outcome.name) {
                order.push(outcome.name.clone());
            }
            outcomes
                .entry(outcome.name.clone())
                .or_default()
                .push(outcome);
        }
    }
    mcp.shutdown().await;
    if order.is_empty() {
        bail!("the eval filter matched nothing");
    }

    let evals: Vec<EvalReport> = order
        .iter()
        .map(|name| {
            let runs: Vec<RunMetrics> = outcomes[name]
                .iter()
                .map(RunMetrics::from_outcome)
                .collect();
            let aggregate = Aggregate::over(&runs.iter().collect::<Vec<_>>());
            EvalReport {
                name: name.clone(),
                holdout: outcomes[name].first().is_some_and(|o| o.holdout),
                runs,
                aggregate,
            }
        })
        .collect();
    let all: Vec<&RunMetrics> = evals.iter().flat_map(|e| e.runs.iter()).collect();
    let report = SuiteReport {
        config: config_path.display().to_string(),
        suite_runs,
        aggregate: Aggregate::over(&all),
        evals,
    };

    eprintln!("\n── eval summary ──");
    for eval in &report.evals {
        eprintln!(
            "{}: {}/{} passing · {} mean ({}–{}) · {:.1} turn(s) · {:.1} tool call(s)",
            eval.name,
            eval.runs.iter().filter(|r| r.passed).count(),
            eval.runs.len(),
            fmt_tokens(eval.aggregate.mean_total_tokens.round() as u64),
            fmt_tokens(eval.aggregate.min_total_tokens),
            fmt_tokens(eval.aggregate.max_total_tokens),
            eval.aggregate.mean_turns,
            eval.aggregate.mean_tool_calls,
        );
    }
    eprintln!(
        "suite: {:.0}% passing · {} mean tokens per execution · ${:.4} mean cost",
        report.aggregate.pass_rate * 100.0,
        fmt_tokens(report.aggregate.mean_total_tokens.round() as u64),
        report.aggregate.mean_cost_usd,
    );

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ModelUsage, UsageSnapshot};

    #[test]
    fn run_metrics_sum_models_and_scopes() {
        let mut usage = UsageSnapshot::default();
        usage.models.insert(
            "a".into(),
            ModelUsage {
                prompt_tokens: 1_000,
                completion_tokens: 100,
                cache_read_tokens: 400,
                cost_usd: 0.01,
                unpriced_requests: 1,
                ..ModelUsage::default()
            },
        );
        usage.models.insert(
            "b".into(),
            ModelUsage {
                prompt_tokens: 500,
                completion_tokens: 50,
                ..ModelUsage::default()
            },
        );
        usage.scopes.insert(
            "stage:plan".into(),
            ScopeUsage {
                turns: 3,
                tool_calls: 7,
                ..ScopeUsage::default()
            },
        );
        usage.scopes.insert(
            "agent:helper".into(),
            ScopeUsage {
                turns: 2,
                tool_calls: 4,
                ..ScopeUsage::default()
            },
        );
        let outcome = EvalOutcome {
            name: "e".into(),
            holdout: false,
            passed: true,
            error: None,
            tokens: 1_650,
            usage,
            wall_ms: 1_234,
            check_excerpt: String::new(),
            output_excerpt: String::new(),
            signals: Vec::new(),
        };
        let metrics = RunMetrics::from_outcome(&outcome);
        assert_eq!(metrics.prompt_tokens, 1_500);
        assert_eq!(metrics.completion_tokens, 150);
        assert_eq!(metrics.total_tokens, 1_650);
        assert_eq!(metrics.cache_read_tokens, 400);
        assert_eq!(metrics.unpriced_requests, 1);
        assert_eq!((metrics.turns, metrics.tool_calls), (5, 11));
        assert_eq!(metrics.wall_ms, 1_234);
        assert_eq!(metrics.scopes.len(), 2);
    }

    #[test]
    fn aggregate_reports_pass_rate_and_token_spread() {
        let run = |passed: bool, tokens: u64| RunMetrics {
            passed,
            error: None,
            prompt_tokens: tokens,
            completion_tokens: 0,
            cache_read_tokens: 0,
            reasoning_tokens: 0,
            total_tokens: tokens,
            cost_usd: 0.02,
            unpriced_requests: 0,
            unreported_requests: 0,
            turns: 4,
            tool_calls: 6,
            wall_ms: 100,
            scopes: BTreeMap::new(),
        };
        let runs = [run(true, 1_000), run(false, 3_000)];
        let aggregate = Aggregate::over(&runs.iter().collect::<Vec<_>>());
        assert_eq!(aggregate.runs, 2);
        assert!((aggregate.pass_rate - 0.5).abs() < f64::EPSILON);
        assert!((aggregate.mean_total_tokens - 2_000.0).abs() < f64::EPSILON);
        assert_eq!(
            (aggregate.min_total_tokens, aggregate.max_total_tokens),
            (1_000, 3_000)
        );
        assert!((aggregate.mean_turns - 4.0).abs() < f64::EPSILON);
        assert!((aggregate.mean_cost_usd - 0.02).abs() < f64::EPSILON);
    }
}
