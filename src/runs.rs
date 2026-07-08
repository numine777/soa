//! Checkpoints for pipeline runs, enabling `soa run --resume`.
//!
//! A run's state is written to `<data_dir>/runs/<id>.json` after every
//! completed stage and removed when the pipeline finishes, so only
//! interrupted or failed runs remain on disk. Resuming restarts at the
//! first stage that hadn't completed — mid-stage progress is not
//! checkpointed.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::tui::store;

fn runs_dir() -> PathBuf {
    store::data_dir().join("runs")
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
    let json = serde_json::to_string(state)?;
    std::fs::write(&path, json).with_context(|| format!("cannot write {}", path.display()))
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
        let dir =
            std::env::temp_dir().join(format!("soa-runs-test-{}", std::process::id()));
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
        state.outputs.insert("a".to_string(), "a says hi".to_string());
        save_run(&state).unwrap();
        let loaded = load_run(&state.id).unwrap();
        assert_eq!(loaded.position, 1);
        assert_eq!(loaded.previous.as_deref(), Some("a says hi"));
        assert_eq!(loaded.stage_names, vec!["a", "b"]);

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
