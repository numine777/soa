//! Checkpoints for pipeline runs, enabling `soa run --resume`.
//!
//! A run's state is atomically written to `<data_dir>/runs/<id>.json` at
//! stage boundaries, while active-stage events use an append-only JSONL
//! sidecar. Resuming merges the two and continues incomplete tool rounds;
//! finished runs remove both checkpoint files.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::model::UsageSnapshot;
use crate::stage::AgentLoopEvent;
use crate::tui::store;

fn runs_dir() -> PathBuf {
    store::data_dir().join("runs")
}

fn usage_dir() -> PathBuf {
    store::data_dir().join("usage")
}

/// Machine-readable token/cost record for one `soa run`, written when the
/// run ends (completed or failed). Unlike checkpoints — which are deleted
/// once a run finishes — these persist, so token-economics comparisons can
/// be scripted over past runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    pub id: String,
    /// First 500 chars of the task, to identify the run.
    pub task: String,
    pub cwd: String,
    pub stage_names: Vec<String>,
    /// "completed" or "failed".
    pub outcome: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub started_at: u64,
    pub finished_at: u64,
    pub usage: UsageSnapshot,
}

/// Persist a usage record under `<data dir>/usage/<id>.json` and, when
/// given, at an explicit path too. Returns the automatic path.
pub fn save_usage_record(
    record: &UsageRecord,
    explicit: Option<&std::path::Path>,
) -> Result<PathBuf> {
    let bytes = serde_json::to_vec_pretty(record)?;
    let path = usage_dir().join(format!("{}.json", record.id));
    crate::persistence::atomic_write(&path, &bytes)?;
    if let Some(explicit) = explicit {
        crate::persistence::atomic_write(explicit, &bytes)?;
    }
    Ok(path)
}

fn event_log_path(id: &str) -> PathBuf {
    runs_dir().join(format!("{id}.events.jsonl"))
}

#[derive(Serialize, Deserialize)]
struct StageEventRecord {
    stage: String,
    run: u32,
    event: AgentLoopEvent,
    usage: UsageSnapshot,
    updated_at: u64,
}

/// Durable progress inside the currently executing stage. Events are
/// append-only for this stage attempt and cleared only after its outcome is
/// committed to the pipeline checkpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageProgress {
    pub stage: String,
    pub run: u32,
    #[serde(default)]
    pub events: Vec<AgentLoopEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunState {
    pub id: String,
    pub started_at: u64,
    pub updated_at: u64,
    /// Directory the run was started from; a bare `--resume` only considers
    /// this directory's runs.
    pub cwd: String,
    /// The task, after @mention expansion (mentions are not re-expanded on
    /// resume).
    pub task: String,
    /// Resolved execution order by stage name, so a reordered config can't
    /// silently change what a resumed run does.
    pub stage_names: Vec<String>,
    /// Index into `stage_names` of the next stage to run.
    pub position: usize,
    /// Stage executions so far, still counted against
    /// `settings.max_stage_runs` after a resume.
    pub runs: u32,
    /// The `{{previous}}` value for the next stage. After a reprompt this
    /// is the handoff instructions, not the last stage's output.
    pub previous: Option<String>,
    /// Completed stage outputs, for `{{stage.<name>}}` templates.
    pub outputs: BTreeMap<String, String>,
    /// Cumulative provider usage and active elapsed time. Older checkpoints
    /// deserialize as an empty ledger.
    #[serde(default)]
    pub usage: UsageSnapshot,
    /// Mid-stage model/tool event log. Older checkpoints resume at the
    /// stage boundary because this field defaults to `None`.
    #[serde(default)]
    pub active_stage: Option<StageProgress>,
    /// File changes captured so far, so a resumed run's end-of-run change
    /// summary covers the whole run. Persisted at stage boundaries: a
    /// mid-stage interruption loses that stage's entries from the summary
    /// (the edits themselves are on disk and will not re-run).
    #[serde(default)]
    pub diffs: Vec<crate::diff::DiffEntry>,
}

impl RunState {
    pub fn new(task: &str, stage_names: Vec<String>) -> Self {
        let now = store::now_epoch();
        RunState {
            id: new_run_id(),
            started_at: now,
            updated_at: now,
            cwd: store::current_cwd(),
            task: task.to_string(),
            stage_names,
            position: 0,
            runs: 0,
            previous: None,
            outputs: BTreeMap::new(),
            usage: UsageSnapshot::default(),
            active_stage: None,
            diffs: Vec::new(),
        }
    }
}

/// A unique id for a run started now (suffixed on same-second collision).
fn new_run_id() -> String {
    let base = store::timestamp_id(store::now_epoch());
    let mut id = base.clone();
    let mut n = 1;
    while runs_dir().join(format!("{id}.json")).exists() {
        n += 1;
        id = format!("{base}-{n}");
    }
    id
}

pub fn save_run(state: &RunState) -> Result<()> {
    let path = runs_dir().join(format!("{}.json", state.id));
    crate::persistence::atomic_write(&path, &serde_json::to_vec(state)?)
}

/// Append one active-stage transition without rewriting the growing run
/// document. The latest record also carries usage/time so budgets resume at
/// the same point even if the process exits before the next stage boundary.
pub fn append_stage_event(
    id: &str,
    stage: &str,
    run: u32,
    event: &AgentLoopEvent,
    usage: UsageSnapshot,
) -> Result<()> {
    crate::persistence::append_json_line(
        &event_log_path(id),
        &StageEventRecord {
            stage: stage.to_string(),
            run,
            event: event.clone(),
            usage,
            updated_at: store::now_epoch(),
        },
    )
}

pub fn clear_stage_events(id: &str) -> Result<()> {
    match std::fs::remove_file(event_log_path(id)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("cannot remove mid-stage event log"),
    }
}

pub fn load_run(id: &str) -> Result<RunState> {
    let path = runs_dir().join(format!("{id}.json"));
    load_run_path(&path).with_context(|| format!("no interrupted run `{id}` at {}", path.display()))
}

fn load_run_path(path: &std::path::Path) -> Result<RunState> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let mut state: RunState = serde_json::from_str(&raw)
        .with_context(|| format!("corrupt run file {}", path.display()))?;
    hydrate_stage_events(&mut state)?;
    Ok(state)
}

fn hydrate_stage_events(state: &mut RunState) -> Result<()> {
    let Some(progress) = state.active_stage.as_mut() else {
        return Ok(());
    };
    let raw = match std::fs::read(event_log_path(&state.id)) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error).context("cannot read mid-stage event log"),
    };
    let lines: Vec<&[u8]> = raw
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .collect();
    let embedded_events = progress.events.len();
    for (index, line) in lines.iter().enumerate() {
        let record: StageEventRecord = match serde_json::from_slice(line) {
            Ok(record) => record,
            // A process can stop halfway through its final append. Earlier
            // corruption is not safe to skip because it could hide a tool.
            Err(error) if index + 1 == lines.len() => {
                tracing::warn!(error = %error, "ignoring incomplete final run event");
                break;
            }
            Err(error) => return Err(error).context("corrupt mid-stage event log"),
        };
        if record.stage != progress.stage || record.run != progress.run {
            bail!(
                "mid-stage event log belongs to `{}` run {}, but checkpoint expects `{}` run {}",
                record.stage,
                record.run,
                progress.stage,
                progress.run,
            );
        }
        // Boundary/error snapshots may already contain a prefix of the
        // append-only sidecar. Validate and skip that prefix so repeated
        // interruptions never duplicate model or tool events on resume.
        if index < embedded_events {
            if record.event != progress.events[index] {
                bail!(
                    "mid-stage event log diverges from checkpoint at event {}",
                    index + 1
                );
            }
            continue;
        }
        progress.events.push(record.event);
        state.usage = record.usage;
        state.updated_at = state.updated_at.max(record.updated_at);
    }
    Ok(())
}

/// All interrupted runs, most recently updated first.
pub fn list_runs() -> Result<Vec<RunState>> {
    let dir = runs_dir();
    let mut states = Vec::new();
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(_) => return Ok(states), // no directory yet: no runs
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json")
            && let Ok(state) = load_run_path(&path)
        {
            states.push(state);
        }
    }
    states.sort_by_key(|s| std::cmp::Reverse(s.updated_at));
    Ok(states)
}

/// The most recently updated interrupted run started from `cwd`.
pub fn latest_run_for(cwd: &str) -> Result<Option<RunState>> {
    Ok(list_runs()?.into_iter().find(|s| s.cwd == cwd))
}

pub fn remove_run(id: &str) -> Result<()> {
    let path = runs_dir().join(format!("{id}.json"));
    std::fs::remove_file(&path).with_context(|| format!("cannot remove {}", path.display()))?;
    clear_stage_events(id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_state_roundtrip_and_listing() {
        let _guard = store::ENV_LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join(format!("soa-runs-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // SAFETY: serialized by ENV_LOCK.
        unsafe { std::env::set_var("XDG_DATA_HOME", &dir) };

        let mut state = RunState::new("fix the widget", vec!["a".into(), "b".into()]);
        state.cwd = "/tmp/proj".to_string();
        save_run(&state).unwrap();

        let mut other = RunState::new("other task", vec!["a".into()]);
        other.cwd = "/tmp/elsewhere".to_string();
        other.updated_at = state.updated_at + 10;
        save_run(&other).unwrap();
        assert_ne!(state.id, other.id, "same-second ids must not collide");

        // Checkpoint progress and reload.
        state.position = 1;
        state.runs = 1;
        state.previous = Some("a says hi".to_string());
        state
            .outputs
            .insert("a".to_string(), "a says hi".to_string());
        state.usage.elapsed_ms = 1_234;
        state.diffs.push(crate::diff::DiffEntry {
            tool: "edit_file".to_string(),
            path: "src/widget.rs".to_string(),
            unified: String::new(),
            added: 3,
            removed: 1,
            before: crate::diff::Snapshot::Unavailable,
            via: None,
        });
        state.usage.models.insert(
            "coder".to_string(),
            crate::model::ModelUsage {
                requests: 2,
                prompt_tokens: 100,
                completion_tokens: 20,
                ..Default::default()
            },
        );
        let started = AgentLoopEvent::Started {
            system: Some("stay focused".to_string()),
            messages: vec![crate::model::Message::User {
                content: "continue".to_string(),
            }],
        };
        state.active_stage = Some(StageProgress {
            stage: "b".to_string(),
            run: 1,
            events: Vec::new(),
        });
        save_run(&state).unwrap();
        let mut event_usage = state.usage.clone();
        event_usage.elapsed_ms = 2_345;
        append_stage_event(&state.id, "b", 1, &started, state.usage.clone()).unwrap();
        append_stage_event(
            &state.id,
            "b",
            1,
            &AgentLoopEvent::UserMessage {
                content: "one more detail".to_string(),
            },
            event_usage,
        )
        .unwrap();
        let loaded = load_run(&state.id).unwrap();
        assert_eq!(loaded.position, 1);
        assert_eq!(loaded.previous.as_deref(), Some("a says hi"));
        assert_eq!(loaded.stage_names, vec!["a", "b"]);
        assert_eq!(loaded.usage.elapsed_ms, 2_345);
        assert_eq!(loaded.usage.models["coder"].requests, 2);
        assert_eq!(loaded.active_stage.as_ref().unwrap().stage, "b");
        assert_eq!(loaded.active_stage.as_ref().unwrap().events.len(), 2);
        // Captured file changes survive the checkpoint for the resumed
        // run's change summary.
        assert_eq!(loaded.diffs.len(), 1);
        assert_eq!(loaded.diffs[0].path, "src/widget.rs");
        assert_eq!((loaded.diffs[0].added, loaded.diffs[0].removed), (3, 1));
        assert!(!runs_dir().join(format!(".{}.json.tmp", state.id)).exists());

        // Saving a hydrated active stage folds its event prefix into the
        // snapshot without making the still-append-only sidecar replay it.
        save_run(&loaded).unwrap();
        assert_eq!(
            load_run(&state.id)
                .unwrap()
                .active_stage
                .unwrap()
                .events
                .len(),
            2
        );
        let mut later_usage = loaded.usage.clone();
        later_usage.elapsed_ms = 3_456;
        append_stage_event(
            &state.id,
            "b",
            1,
            &AgentLoopEvent::UserMessage {
                content: "after another interruption".to_string(),
            },
            later_usage,
        )
        .unwrap();
        let resumed_again = load_run(&state.id).unwrap();
        assert_eq!(
            resumed_again.active_stage.as_ref().unwrap().events.len(),
            3
        );
        assert_eq!(resumed_again.usage.elapsed_ms, 3_456);

        let mut legacy = serde_json::to_value(&loaded).unwrap();
        legacy.as_object_mut().unwrap().remove("usage");
        legacy.as_object_mut().unwrap().remove("active_stage");
        let legacy: RunState = serde_json::from_value(legacy).unwrap();
        assert_eq!(legacy.usage, UsageSnapshot::default());
        assert!(legacy.active_stage.is_none());

        // Listing is newest-first; latest-for-cwd filters by directory.
        let all = list_runs().unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].id, other.id);
        assert_eq!(latest_run_for("/tmp/proj").unwrap().unwrap().id, state.id);
        assert!(latest_run_for("/nowhere").unwrap().is_none());

        // Finished runs disappear.
        remove_run(&state.id).unwrap();
        assert!(load_run(&state.id).is_err());
        assert!(!event_log_path(&state.id).exists());
        assert_eq!(list_runs().unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
