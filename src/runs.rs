//! Checkpoints for pipeline runs, enabling `soa run --resume`.
//!
//! A run's state is atomically written to `<data_dir>/runs/<id>.json` at
//! stage boundaries and after every durable event inside an active stage.
//! Resuming replays that event log and continues incomplete tool rounds;
//! finished runs remove the checkpoint.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::model::UsageSnapshot;
use crate::stage::AgentLoopEvent;
use crate::tui::store;

fn runs_dir() -> PathBuf {
    store::data_dir().join("runs")
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
    let dir = runs_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let path = dir.join(format!("{}.json", state.id));
    let temporary = dir.join(format!("{}.json.tmp", state.id));
    let json = serde_json::to_string(state)?;
    std::fs::write(&temporary, json)
        .with_context(|| format!("cannot write {}", temporary.display()))?;
    std::fs::rename(&temporary, &path).with_context(|| format!("cannot replace {}", path.display()))
}

pub fn load_run(id: &str) -> Result<RunState> {
    let path = runs_dir().join(format!("{id}.json"));
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("no interrupted run `{id}` at {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("corrupt run file {}", path.display()))
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
            && let Ok(raw) = std::fs::read_to_string(&path)
            && let Ok(state) = serde_json::from_str::<RunState>(&raw)
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
    std::fs::remove_file(&path).with_context(|| format!("cannot remove {}", path.display()))
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
        state.usage.models.insert(
            "coder".to_string(),
            crate::model::ModelUsage {
                requests: 2,
                prompt_tokens: 100,
                completion_tokens: 20,
                ..Default::default()
            },
        );
        state.active_stage = Some(StageProgress {
            stage: "b".to_string(),
            run: 1,
            events: vec![AgentLoopEvent::Started {
                system: Some("stay focused".to_string()),
                messages: vec![crate::model::Message::User {
                    content: "continue".to_string(),
                }],
            }],
        });
        save_run(&state).unwrap();
        let loaded = load_run(&state.id).unwrap();
        assert_eq!(loaded.position, 1);
        assert_eq!(loaded.previous.as_deref(), Some("a says hi"));
        assert_eq!(loaded.stage_names, vec!["a", "b"]);
        assert_eq!(loaded.usage.elapsed_ms, 1_234);
        assert_eq!(loaded.usage.models["coder"].requests, 2);
        assert_eq!(loaded.active_stage.as_ref().unwrap().stage, "b");
        assert_eq!(loaded.active_stage.as_ref().unwrap().events.len(), 1);
        assert!(!runs_dir().join(format!("{}.json.tmp", state.id)).exists());

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
        assert_eq!(list_runs().unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
