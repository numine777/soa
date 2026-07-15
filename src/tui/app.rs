//! Chat TUI state, input handling, and the background agent worker.

use std::path::PathBuf;
use std::sync::Arc;

use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tui_textarea::TextArea;

use super::completion::{self, Completion};
use super::store::{self, Branch, Checkpoint, PromptHistory, Session};
use crate::approval::{ApprovalRequest, Approvals, Decision};
use crate::config::{Config, Mode, Stage, ToolEffect};
use crate::diff::{self, DiffEntry};
use crate::mcp::McpManager;
use crate::model::{Message, ModelClient, Usage, UsageTracker, fmt_tokens};
use crate::stage::{
    AgentLoopEvent, AgentLoopObservation, AgentLoopOptions, StageTool, assemble_tools,
    build_model_client, run_agent_loop, salvage_cancelled_loop,
};

const COMPACT_INSTRUCTION: &str = "\
You are compacting this conversation to free context space. Summarize \
everything above into a briefing that preserves: the user's goals and \
constraints, decisions made, key facts (file paths, commands, URLs, code \
identifiers), work completed, and work still pending. Write terse bullet \
points. Output only the briefing.";

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum TranscriptItem {
    User(String),
    Assistant(String),
    ToolCall {
        name: String,
        args: String,
    },
    ToolDone {
        preview: String,
    },
    Info(String),
    Error(String),
    /// A `/clear` divider: everything above it is hidden from the chat and
    /// diff views (but kept in the session data).
    Cleared(String),
}

/// Events sent by the background worker back to the UI loop.
pub enum AgentEvent {
    /// A streamed fragment of the assistant's in-progress reply.
    Delta(String),
    ToolCall {
        name: String,
        args: String,
    },
    ToolDone {
        preview: String,
    },
    Diff(DiffEntry),
    /// Something the user should know mid-turn (e.g. context shedding).
    Notice(String),
    /// Turn finished: the updated history, the assistant's final text, and
    /// the last reported token usage.
    Turn {
        history: Vec<Message>,
        text: String,
        usage: Option<Usage>,
    },
    Compacted {
        summary: String,
    },
    /// A `/run` workflow entered a stage (`run` counts reprompt re-runs).
    StageStart {
        stage: String,
        run: u32,
    },
    /// A `/run` workflow finished: the last stage's output, or the error
    /// that stopped the pipeline. `task` is the mention-expanded task.
    WorkflowDone {
        workflow: String,
        task: String,
        result: Result<String, String>,
        usage: Vec<String>,
    },
    Error(String),
}

pub enum View {
    Chat,
    Diffs {
        selected: usize,
        scroll: usize,
    },
    /// Session picker over `App::session_list`.
    Sessions {
        selected: usize,
        scroll: usize,
    },
    /// Rewind picker over `App::checkpoints` (newest first, plus a
    /// session-start row at the bottom).
    Rewind {
        selected: usize,
        scroll: usize,
    },
    /// Branch picker over `App::branches`.
    Branches {
        selected: usize,
        scroll: usize,
    },
}

pub struct App {
    pub config: Arc<Config>,
    pub mcp: Arc<McpManager>,
    pub http: reqwest::Client,
    pub stage_index: usize,
    /// Provider-format conversation history (system prompt excluded; it is
    /// re-resolved from the active stage on every turn).
    pub history: Vec<Message>,
    pub transcript: Vec<TranscriptItem>,
    pub diffs: Vec<DiffEntry>,
    /// `/clear` display baselines: transcript items and diff entries before
    /// these indexes are hidden from the chat and diff views. Data is never
    /// deleted — exports and session files keep the full record.
    pub transcript_baseline: usize,
    pub diff_baseline: usize,
    /// Rewind targets: one per turn-starting user message, invalidated by
    /// compaction and `/clear` (both rewrite the history they index into).
    pub checkpoints: Vec<Checkpoint>,
    /// Stored conversation lines: `/branch` stashes, `/rewind` stashes the
    /// abandoned tail, `/branches` swaps one with the live conversation.
    pub branches: Vec<Branch>,
    pub view: View,
    pub input: TextArea<'static>,
    /// 0 = pinned to the bottom (auto-follow); larger = scrolled up.
    pub scroll_from_bottom: usize,
    /// Viewport heights recorded during draw, used for page-scroll steps.
    pub chat_viewport: usize,
    pub diff_viewport: usize,
    pub tool_count: usize,
    pub spinner: usize,
    /// The assistant reply currently streaming in, shown live at the bottom
    /// of the transcript until the turn completes.
    pub stream_buffer: String,
    /// A tool call waiting for the user's y/a/n decision; input is modal
    /// while this is set.
    pub pending_approval: Option<ApprovalRequest>,
    approvals: Arc<Approvals>,
    /// Autocomplete popup for slash commands and @file mentions,
    /// recomputed after every input keystroke.
    pub completion: Option<Completion>,
    /// Where the config was loaded from, for `/reload`.
    config_path: PathBuf,
    /// Session-level model override (`/model <name>`), applied to every
    /// stage until cleared with `/model default`.
    model_override: Option<String>,
    /// Real token usage from the most recent turn, when the server reports
    /// it; None falls back to the character estimate.
    pub last_usage: Option<Usage>,
    /// Rich usage for ordinary chat and compaction requests in this TUI
    /// session. `/run` workflows get a separate budgeted ledger.
    usage: UsageTracker,
    pub quit: bool,
    turn: Option<JoinHandle<()>>,
    /// True while the running task is a compaction rather than a chat turn.
    compacting: bool,
    /// Name of the workflow the running task is executing, if it is a
    /// `/run` pipeline rather than a chat turn.
    workflow_running: Option<String>,
    /// Messages submitted while a turn is running, shared with the worker,
    /// which injects them into the conversation between tool rounds.
    steer_queue: Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    /// Everything queued during the current turn. The worker's history clone
    /// is discarded when a turn fails or is cancelled, so these are
    /// re-appended to the surviving history — same as the turn's original
    /// message, which is pushed before the worker starts.
    steered_this_turn: Vec<String>,
    /// Event log of the running turn, shared with the worker. When a turn
    /// is cancelled or fails, the completed tool rounds recorded here are
    /// folded back into `history` so the model keeps seeing work whose
    /// effects are already on disk.
    turn_events: Arc<std::sync::Mutex<Vec<AgentLoopEvent>>>,
    tx: UnboundedSender<AgentEvent>,
    // Session persistence.
    session_id: String,
    started_at: u64,
    cwd: String,
    /// Sessions shown in the picker; loaded when it opens.
    pub session_list: Vec<Session>,
    /// Set once the user has submitted something; empty sessions aren't saved.
    has_activity: bool,
    /// A save failure is reported once, not on every subsequent persist;
    /// saving keeps being attempted and a success clears the flag.
    persist_error_shown: bool,
    // Prompt history (shell-style recall with Up/Down).
    prompt_history: PromptHistory,
    /// Position while browsing the prompt history; None = not browsing.
    recall_index: Option<usize>,
    /// The in-progress draft stashed when browsing starts.
    recall_draft: String,
}

// The TextArea is the editing engine only; ui::draw_input renders the text
// itself (soft-wrapped) and places the terminal cursor.
fn new_textarea() -> TextArea<'static> {
    let mut input = TextArea::default();
    input.set_placeholder_text("Message soa — Enter sends, Alt+Enter newline, /help for commands");
    input
}

impl App {
    pub fn new(
        config: Arc<Config>,
        config_path: PathBuf,
        mcp: Arc<McpManager>,
        stage_index: usize,
        tx: UnboundedSender<AgentEvent>,
        approvals: Arc<Approvals>,
        resumed: Option<Session>,
    ) -> App {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                config.settings.request_timeout_secs,
            ))
            .build()
            .expect("HTTP client");
        let mut app = App {
            config,
            mcp,
            http,
            stage_index,
            history: Vec::new(),
            transcript: Vec::new(),
            diffs: Vec::new(),
            transcript_baseline: 0,
            diff_baseline: 0,
            checkpoints: Vec::new(),
            branches: Vec::new(),
            view: View::Chat,
            input: new_textarea(),
            scroll_from_bottom: 0,
            chat_viewport: 20,
            diff_viewport: 20,
            tool_count: 0,
            spinner: 0,
            stream_buffer: String::new(),
            pending_approval: None,
            approvals,
            completion: None,
            config_path,
            model_override: None,
            last_usage: None,
            usage: UsageTracker::unlimited(),
            quit: false,
            turn: None,
            compacting: false,
            workflow_running: None,
            steer_queue: Arc::default(),
            steered_this_turn: Vec::new(),
            turn_events: Arc::new(std::sync::Mutex::new(Vec::new())),
            tx,
            session_id: store::new_session_id(),
            started_at: store::now_epoch(),
            cwd: store::current_cwd(),
            session_list: Vec::new(),
            has_activity: false,
            persist_error_shown: false,
            prompt_history: PromptHistory::load(),
            recall_index: None,
            recall_draft: String::new(),
        };
        app.refresh_tool_count();
        match resumed {
            // Stage restore already happened in mod.rs (respecting --stage),
            // so don't let the session override it here.
            Some(session) => {
                app.apply_session(session, false);
                app.info(format!(
                    "resumed session {} (started {}, {}, ctx ~{})",
                    app.session_id,
                    store::format_epoch(app.started_at),
                    app.stage_summary(),
                    fmt_tokens(app.token_estimate() as u64),
                ));
            }
            None => app.info(format!(
                "soa chat — {}. Type /help for commands.",
                app.stage_summary()
            )),
        }
        app
    }

    /// Replace the conversation state with a saved session's. The saved
    /// transcript already carries its original greeting.
    fn apply_session(&mut self, session: Session, restore_stage: bool) {
        if restore_stage
            && let Some(index) = self
                .config
                .stages
                .iter()
                .position(|s| s.name == session.stage)
        {
            self.stage_index = index;
            self.refresh_tool_count();
        }
        self.history = session.history;
        self.transcript = session.transcript;
        self.diffs = session.diffs;
        self.transcript_baseline = session.transcript_baseline.min(self.transcript.len());
        self.diff_baseline = session.diff_baseline.min(self.diffs.len());
        self.checkpoints = session.checkpoints;
        self.branches = session.branches;
        self.session_id = session.id;
        self.started_at = session.started_at;
        self.has_activity = true;
        self.last_usage = None;
        self.scroll_from_bottom = 0;
    }

    /// Write the session to disk. Errors are shown once in the transcript
    /// rather than crashing the conversation.
    fn persist(&mut self) {
        if !self.has_activity {
            return;
        }
        let title = self
            .transcript
            .iter()
            .find_map(|item| match item {
                TranscriptItem::User(text) => {
                    Some(text.lines().next().unwrap_or("").chars().take(80).collect())
                }
                _ => None,
            })
            .unwrap_or_else(|| "(no prompt)".to_string());
        let session = Session {
            id: self.session_id.clone(),
            started_at: self.started_at,
            updated_at: store::now_epoch(),
            stage: self.stage().name.clone(),
            title,
            cwd: self.cwd.clone(),
            history: self.history.clone(),
            transcript: self.transcript.clone(),
            diffs: self.diffs.clone(),
            checkpoints: self.checkpoints.clone(),
            branches: self.branches.clone(),
            transcript_baseline: self.transcript_baseline,
            diff_baseline: self.diff_baseline,
        };
        match store::save_session(&session) {
            Ok(()) => self.persist_error_shown = false,
            // Keep trying on later persists — a transient disk error must
            // not silently stop session saving — but report it only once.
            Err(e) if !self.persist_error_shown => {
                self.persist_error_shown = true;
                self.error(format!(
                    "failed to save session (will keep retrying): {e:#}"
                ));
            }
            Err(e) => tracing::warn!(error = format!("{e:#}"), "session save failed again"),
        }
    }

    pub fn stage(&self) -> &Stage {
        &self.config.stages[self.stage_index]
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The model requests actually use: the `/model` override when set,
    /// otherwise the active stage's model.
    pub fn active_model(&self) -> &str {
        self.model_override
            .as_deref()
            .unwrap_or(&self.stage().model)
    }

    fn stage_summary(&self) -> String {
        let stage = self.stage();
        format!(
            "stage `{}` · model `{}`{} · {} · {} tool(s)",
            stage.name,
            self.active_model(),
            if self.model_override.is_some() {
                " (override)"
            } else {
                ""
            },
            match stage.mode {
                Mode::ReadOnly => "read_only",
                Mode::ReadWrite => "read_write",
            },
            self.tool_count,
        )
    }

    fn refresh_tool_count(&mut self) {
        self.tool_count = assemble_tools(&self.stage().tool_profile(), &self.config, &self.mcp, 0)
            .map(|tools| tools.len())
            .unwrap_or(0);
    }

    pub fn is_running(&self) -> bool {
        self.turn.is_some()
    }

    pub fn spinner_tick(&mut self) {
        self.spinner = self.spinner.wrapping_add(1);
    }

    pub fn context_capacity(&self) -> Option<u64> {
        self.config
            .models
            .get(self.active_model())
            .and_then(|m| m.context_tokens)
    }

    /// Status-bar context gauge: text plus a pressure level
    /// (0 = fine, 1 = ≥70%, 2 = ≥90%). Real usage when reported, `~`
    /// estimate otherwise.
    pub fn context_status(&self) -> (String, u8) {
        let capacity = self.context_capacity();
        let (used, estimated) = match self.last_usage {
            Some(usage) => (usage.context_tokens(), false),
            None => (self.token_estimate() as u64, true),
        };
        let marker = if estimated { "~" } else { "" };
        match capacity {
            Some(capacity) => {
                let percent = used * 100 / capacity.max(1);
                let level = if percent >= 90 {
                    2
                } else if percent >= 70 {
                    1
                } else {
                    0
                };
                (
                    format!(
                        "ctx {marker}{}/{} ({percent}%)",
                        fmt_tokens(used),
                        fmt_tokens(capacity)
                    ),
                    level,
                )
            }
            None => (format!("ctx {marker}{}", fmt_tokens(used)), 0),
        }
    }

    /// Rough context-size estimate: ~4 characters per token.
    pub fn token_estimate(&self) -> usize {
        let chars: usize = self
            .history
            .iter()
            .map(|message| match message {
                Message::System { content } | Message::User { content } => content.len(),
                Message::Assistant {
                    content,
                    tool_calls,
                } => {
                    content.as_deref().map_or(0, str::len)
                        + tool_calls.as_ref().map_or(0, |calls| {
                            calls
                                .iter()
                                .map(|c| c.function.arguments.to_string().len() + 32)
                                .sum()
                        })
                }
                Message::Tool { content, .. } => content.len(),
            })
            .sum();
        chars / 4
    }

    /// Diff entries recorded since the last `/clear` — what the diff view,
    /// its keys, and the status bar operate on. Earlier entries stay in
    /// `self.diffs` (and on disk) but are hidden from the UI.
    pub fn visible_diffs(&self) -> &[DiffEntry] {
        &self.diffs[self.diff_baseline.min(self.diffs.len())..]
    }

    /// Transcript items shown in the chat pane: everything since the last
    /// `/clear` divider (inclusive, so the divider itself is visible).
    pub fn visible_transcript(&self) -> &[TranscriptItem] {
        &self.transcript[self.transcript_baseline.min(self.transcript.len())..]
    }

    fn info(&mut self, text: impl Into<String>) {
        self.transcript.push(TranscriptItem::Info(text.into()));
    }

    fn error(&mut self, text: impl Into<String>) {
        self.transcript.push(TranscriptItem::Error(text.into()));
    }

    // ------------------------------------------------------------------
    // Terminal events
    // ------------------------------------------------------------------

    pub fn on_term_event(&mut self, event: Event) {
        match event {
            Event::Key(key) if key.kind != KeyEventKind::Release => self.on_key(key),
            Event::Mouse(mouse) => self.on_mouse(mouse),
            Event::Paste(text) => {
                if matches!(self.view, View::Chat) {
                    self.input
                        .insert_str(text.replace("\r\n", "\n").replace('\r', "\n"));
                    self.refresh_completion();
                }
            }
            _ => {}
        }
    }

    /// Route an approval request from the worker into the modal prompt.
    pub fn on_approval_request(&mut self, request: ApprovalRequest) {
        self.pending_approval = Some(request);
    }

    fn resolve_approval(&mut self, decision: Decision) {
        if let Some(request) = self.pending_approval.take() {
            let note = match decision {
                Decision::Approve => format!("approved: {}", request.descriptor),
                Decision::AlwaysAllow => format!(
                    "approved: {} (always this session: {})",
                    request.descriptor, request.always_pattern
                ),
                Decision::Deny => format!("denied: {}", request.descriptor),
            };
            self.info(note);
            let _ = request.responder.send(decision);
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

        // A pending approval is modal: only y/a/n (and quit) work.
        if self.pending_approval.is_some() {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.resolve_approval(Decision::Approve);
                }
                KeyCode::Char('a') | KeyCode::Char('A') => {
                    self.resolve_approval(Decision::AlwaysAllow);
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.resolve_approval(Decision::Deny);
                }
                KeyCode::Char('q') | KeyCode::Char('c') if ctrl => self.quit = true,
                _ => {}
            }
            return;
        }

        // Global bindings.
        match key.code {
            KeyCode::Char('q') if ctrl => {
                self.quit = true;
                return;
            }
            KeyCode::Char('d') if ctrl => {
                let input_empty = self.input.lines().iter().all(|line| line.is_empty());
                if matches!(self.view, View::Chat) && !input_empty {
                    // Readline behavior: on a non-empty input Ctrl+D deletes
                    // the character under the cursor; EOF-quit needs an
                    // empty prompt.
                    self.input.input(tui_textarea::Input::from(key));
                } else {
                    self.quit = true;
                }
                return;
            }
            KeyCode::Char('g') if ctrl => {
                self.toggle_diff_view();
                return;
            }
            KeyCode::Char('c') if ctrl => {
                if self.is_running() {
                    self.cancel_turn();
                } else if self.input.lines().iter().any(|line| !line.is_empty()) {
                    self.input = new_textarea();
                    self.recall_index = None;
                    self.recall_draft.clear();
                } else {
                    self.info("use Ctrl+D or /quit to exit");
                }
                return;
            }
            _ => {}
        }

        match self.view {
            View::Diffs { .. } => self.on_diff_key(key),
            View::Sessions { .. } => self.on_sessions_key(key),
            View::Rewind { .. } => self.on_rewind_key(key),
            View::Branches { .. } => self.on_branches_key(key),
            View::Chat => self.on_chat_key(key),
        }
    }

    // ------------------------------------------------------------------
    // Session picker
    // ------------------------------------------------------------------

    /// Open the picker with this working directory's sessions (legacy
    /// sessions saved before cwd tracking are included). Row 0 is the
    /// "start new session" entry; row i+1 is `session_list[i]`.
    fn open_sessions(&mut self) {
        let all = match store::list_sessions() {
            Ok(all) => all,
            Err(e) => return self.error(format!("cannot list sessions: {e:#}")),
        };
        let list: Vec<Session> = all
            .into_iter()
            .filter(|s| s.cwd == self.cwd || s.cwd.is_empty())
            .collect();
        let selected = list
            .iter()
            .position(|s| s.id == self.session_id)
            .map_or(0, |index| index + 1);
        self.session_list = list;
        self.view = View::Sessions {
            selected,
            scroll: 0,
        };
    }

    fn on_sessions_key(&mut self, key: KeyEvent) {
        let View::Sessions {
            selected: current, ..
        } = self.view
        else {
            return;
        };
        let last = self.session_list.len(); // rows = sessions + the new-session row
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.view = View::Chat,
            KeyCode::Up | KeyCode::Char('k') => {
                self.set_picker_selection(current.saturating_sub(1));
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.set_picker_selection((current + 1).min(last));
            }
            KeyCode::Home | KeyCode::Char('g') => self.set_picker_selection(0),
            KeyCode::End | KeyCode::Char('G') => self.set_picker_selection(last),
            KeyCode::Enter if current == 0 => self.start_new_session(),
            KeyCode::Enter => self.switch_to_session(current - 1),
            KeyCode::Char('n') => self.start_new_session(),
            _ => {}
        }
    }

    fn set_picker_selection(&mut self, index: usize) {
        if let View::Sessions { selected, .. } = &mut self.view {
            *selected = index;
        }
    }

    fn switch_to_session(&mut self, index: usize) {
        self.view = View::Chat;
        let Some(session) = self.session_list.get(index).cloned() else {
            return;
        };
        if session.id == self.session_id {
            return;
        }
        if self.is_running() {
            return self.error("finish or cancel the running turn before switching sessions");
        }
        self.persist(); // save the session we're leaving
        let id = session.id.clone();
        self.apply_session(session, true);
        self.info(format!(
            "switched to session {} ({}, ctx ~{})",
            id,
            self.stage_summary(),
            fmt_tokens(self.token_estimate() as u64),
        ));
    }

    fn start_new_session(&mut self) {
        self.view = View::Chat;
        if self.is_running() {
            return self.error("finish or cancel the running turn before starting a new session");
        }
        self.persist();
        self.history.clear();
        self.transcript.clear();
        self.diffs.clear();
        self.transcript_baseline = 0;
        self.diff_baseline = 0;
        self.checkpoints.clear();
        self.branches.clear();
        self.session_id = store::new_session_id();
        self.started_at = store::now_epoch();
        self.has_activity = false;
        self.last_usage = None;
        self.scroll_from_bottom = 0;
        self.info(format!(
            "started new session {} — {}. Type /help for commands.",
            self.session_id,
            self.stage_summary(),
        ));
    }

    fn on_chat_key(&mut self, key: KeyEvent) {
        // The completion popup captures navigation keys while visible.
        if let Some(state) = &mut self.completion {
            let count = state.items.len();
            match key.code {
                KeyCode::Esc => {
                    self.completion = None;
                    return;
                }
                KeyCode::Up if key.modifiers.is_empty() => {
                    state.selected = state.selected.checked_sub(1).unwrap_or(count - 1);
                    return;
                }
                KeyCode::Down if key.modifiers.is_empty() => {
                    state.selected = (state.selected + 1) % count;
                    return;
                }
                KeyCode::Tab => {
                    self.accept_completion();
                    return;
                }
                // Enter accepts only when it would change the input; a
                // fully typed command or path submits as usual.
                KeyCode::Enter
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
                {
                    if self.completion_changes_input() {
                        self.accept_completion();
                        return;
                    }
                    self.completion = None;
                }
                _ => {}
            }
        }
        match key.code {
            KeyCode::Esc => self.cancel_turn(),
            KeyCode::Enter
                if key
                    .modifiers
                    .intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
            {
                self.input.insert_newline();
            }
            KeyCode::Enter => self.submit(),
            KeyCode::PageUp => self.scroll_up(self.chat_viewport.saturating_sub(1).max(1)),
            KeyCode::PageDown => self.scroll_down(self.chat_viewport.saturating_sub(1).max(1)),
            // Shell-style prompt recall: Up on the input's first line goes
            // back through history, Down on the last line comes forward.
            KeyCode::Up
                if key.modifiers.is_empty()
                    && self.input.cursor().0 == 0
                    && !self.prompt_history.entries.is_empty() =>
            {
                self.recall_prev();
            }
            KeyCode::Down
                if key.modifiers.is_empty()
                    && self.recall_index.is_some()
                    && self.input.cursor().0 + 1 == self.input.lines().len() =>
            {
                self.recall_next();
            }
            _ => {
                self.input.input(tui_textarea::Input::from(key));
            }
        }
        self.refresh_completion();
    }

    /// Recompute the autocomplete popup for the token under the cursor.
    fn refresh_completion(&mut self) {
        let (row, col) = self.input.cursor();
        let line = self.input.lines().get(row).cloned().unwrap_or_default();
        let stage_names: Vec<String> = self.config.stages.iter().map(|s| s.name.clone()).collect();
        let model_names: Vec<String> = self.config.models.keys().cloned().collect();
        let workflow_names: Vec<String> = self.config.workflows.keys().cloned().collect();
        self.completion = completion::compute(
            &line,
            col,
            row == 0,
            std::path::Path::new(&self.cwd),
            &stage_names,
            &model_names,
            &workflow_names,
        );
    }

    /// Whether applying the selected completion would alter the input.
    fn completion_changes_input(&self) -> bool {
        let Some(state) = &self.completion else {
            return false;
        };
        let item = &state.items[state.selected];
        let (row, col) = self.input.cursor();
        let line: Vec<char> = self
            .input
            .lines()
            .get(row)
            .map(|l| l.chars().collect())
            .unwrap_or_default();
        let from = state.replace_from.min(line.len());
        let replaced: String = line[from..col.min(line.len())].iter().collect();
        replaced != item.insert
    }

    /// Splice the selected completion into the input, then recompute (so
    /// accepting a directory descends into it).
    fn accept_completion(&mut self) {
        let Some(state) = &self.completion else {
            return;
        };
        let item = state.items[state.selected].clone();
        let (row, col) = self.input.cursor();
        let mut lines: Vec<String> = self.input.lines().to_vec();
        let Some(line) = lines.get(row).cloned() else {
            return;
        };
        let chars: Vec<char> = line.chars().collect();
        let from = state.replace_from.min(chars.len());
        let before: String = chars[..from].iter().collect();
        let after: String = chars[col.min(chars.len())..].iter().collect();
        lines[row] = format!("{before}{}{after}", item.insert);
        let new_col = from + item.insert.chars().count();

        self.set_input_text(&lines.join("\n"));
        self.input
            .move_cursor(tui_textarea::CursorMove::Jump(row as u16, new_col as u16));
        self.refresh_completion();
    }

    fn set_input_text(&mut self, text: &str) {
        self.input = new_textarea();
        self.input.insert_str(text);
    }

    fn recall_prev(&mut self) {
        let entries = &self.prompt_history.entries;
        let index = match self.recall_index {
            None => {
                self.recall_draft = self.input.lines().join("\n");
                entries.len() - 1
            }
            Some(0) => return,
            Some(current) => current - 1,
        };
        self.recall_index = Some(index);
        let text = self.prompt_history.entries[index].clone();
        self.set_input_text(&text);
    }

    fn recall_next(&mut self) {
        match self.recall_index {
            None => {}
            Some(current) if current + 1 < self.prompt_history.entries.len() => {
                self.recall_index = Some(current + 1);
                let text = self.prompt_history.entries[current + 1].clone();
                self.set_input_text(&text);
            }
            Some(_) => {
                // Past the newest entry: restore the stashed draft.
                self.recall_index = None;
                let draft = std::mem::take(&mut self.recall_draft);
                self.set_input_text(&draft);
            }
        }
    }

    fn on_diff_key(&mut self, key: KeyEvent) {
        // The view's `selected` indexes the visible (post-/clear) slice;
        // restore needs the position in the full log.
        if key.code == KeyCode::Char('r')
            && let View::Diffs { selected, .. } = &self.view
        {
            let index = self.diff_baseline + *selected;
            return self.restore_diff(index);
        }
        let visible = self.visible_diffs().len();
        let View::Diffs { selected, scroll } = &mut self.view else {
            return;
        };
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.view = View::Chat,
            KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                if *selected + 1 < visible {
                    *selected += 1;
                    *scroll = 0;
                }
            }
            KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => {
                if *selected > 0 {
                    *selected -= 1;
                    *scroll = 0;
                }
            }
            KeyCode::Up | KeyCode::Char('k') => *scroll = scroll.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => *scroll += 1,
            KeyCode::PageUp => *scroll = scroll.saturating_sub(self.diff_viewport.max(1)),
            KeyCode::PageDown => *scroll += self.diff_viewport.max(1),
            KeyCode::Home | KeyCode::Char('g') => *scroll = 0,
            KeyCode::End | KeyCode::Char('G') => *scroll = usize::MAX, // clamped at draw
            _ => {}
        }
    }

    fn on_mouse(&mut self, mouse: MouseEvent) {
        match (&mut self.view, mouse.kind) {
            (View::Chat, MouseEventKind::ScrollUp) => self.scroll_up(3),
            (View::Chat, MouseEventKind::ScrollDown) => self.scroll_down(3),
            (View::Diffs { scroll, .. }, MouseEventKind::ScrollUp) => {
                *scroll = scroll.saturating_sub(3);
            }
            (View::Diffs { scroll, .. }, MouseEventKind::ScrollDown) => *scroll += 3,
            (View::Sessions { selected, .. }, MouseEventKind::ScrollUp) => {
                *selected = selected.saturating_sub(1);
            }
            (View::Sessions { selected, .. }, MouseEventKind::ScrollDown) => {
                // rows = sessions + the new-session row at index 0
                *selected = (*selected + 1).min(self.session_list.len());
            }
            (View::Rewind { selected, .. }, MouseEventKind::ScrollUp) => {
                *selected = selected.saturating_sub(1);
            }
            (View::Rewind { selected, .. }, MouseEventKind::ScrollDown) => {
                // rows = checkpoints + the session-start row at the bottom
                *selected = (*selected + 1).min(self.checkpoints.len());
            }
            (View::Branches { selected, .. }, MouseEventKind::ScrollUp) => {
                *selected = selected.saturating_sub(1);
            }
            (View::Branches { selected, .. }, MouseEventKind::ScrollDown) => {
                *selected = (*selected + 1).min(self.branches.len().saturating_sub(1));
            }
            _ => {}
        }
    }

    fn scroll_up(&mut self, lines: usize) {
        // Clamped against the transcript height during draw.
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(lines);
    }

    fn scroll_down(&mut self, lines: usize) {
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(lines);
    }

    fn toggle_diff_view(&mut self) {
        match self.view {
            View::Diffs { .. } => self.view = View::Chat,
            View::Sessions { .. } | View::Rewind { .. } | View::Branches { .. } => {} // don't stack modal views
            View::Chat if self.visible_diffs().is_empty() => {
                self.info(if self.diff_baseline > 0 {
                    "no file changes since /clear (earlier ones are kept in the session file)"
                } else {
                    "no file changes captured yet"
                });
            }
            View::Chat => {
                self.view = View::Diffs {
                    selected: self.visible_diffs().len() - 1,
                    scroll: 0,
                };
            }
        }
    }

    // ------------------------------------------------------------------
    // Submitting input and slash commands
    // ------------------------------------------------------------------

    fn submit(&mut self) {
        let text = self.input.lines().join("\n").trim().to_string();
        if text.is_empty() {
            return;
        }
        if let Some(command) = text.strip_prefix('/') {
            self.remember_prompt(&text);
            self.input = new_textarea();
            self.run_command(command.trim());
            self.persist(); // commands like /clear change persisted state
            return;
        }
        if self.is_running() && self.compacting {
            return self.error("compacting — send again when it finishes");
        }
        self.remember_prompt(&text);
        self.has_activity = true;
        self.input = new_textarea();

        // A turn-starting message is a rewind target: record where the
        // transcript, history, and diff log stand right now. (Steered
        // messages are not checkpoints — they land mid-turn.)
        if !self.is_running() {
            self.checkpoints.push(Checkpoint {
                transcript_index: self.transcript.len(),
                history_len: self.history.len(),
                diff_len: self.diffs.len(),
            });
        }

        // @file mentions: the transcript keeps what was typed; the model
        // receives the message with file contents appended.
        let (expanded, reports) = crate::mentions::expand_mentions(
            &text,
            std::path::Path::new(&self.cwd),
            self.config.settings.max_tool_output_chars,
        );
        self.transcript.push(TranscriptItem::User(text));
        for report in &reports {
            self.info(report.describe());
        }
        self.scroll_from_bottom = 0;

        // Mid-turn steering: queue the message for the running worker, which
        // delivers it to the model after the current tool round. Anything
        // still queued when the turn ends becomes the next turn.
        if self.is_running() {
            self.info(if self.workflow_running.is_some() {
                "↪ queued — becomes a chat turn after the workflow finishes"
            } else {
                "↪ queued — delivered after the current tool round"
            });
            self.steered_this_turn.push(expanded.clone());
            self.steer_queue.lock().unwrap().push_back(expanded);
            return;
        }

        self.history.push(Message::User { content: expanded });
        self.start_turn();
        self.persist();
    }

    fn remember_prompt(&mut self, text: &str) {
        self.prompt_history.push(text);
        self.recall_index = None;
        self.recall_draft.clear();
    }

    fn run_command(&mut self, command: &str) {
        let (name, arg) = command.split_once(' ').unwrap_or((command, ""));
        match name {
            "help" => self.info(
                "commands:\n\
                 /compact        summarize the conversation and shrink context\n\
                 /clear          drop all conversation context and hide earlier\n\
                                 messages/diffs (kept in the session file)\n\
                 /usage          cumulative token usage per model since launch\n\
                 /diff           open the diff viewer (Ctrl+G)\n\
                 /rewind         pick a past message to rewind to (conversation + files)\n\
                 /branch <name>  save the conversation as a branch and keep going\n\
                 /branches       switch between saved conversation lines (d deletes)\n\
                 /run [wf] <task> run a stage pipeline on a task from here; the final\n\
                                 output joins this conversation (default workflow unless\n\
                                 the first word names one — see soa stages)\n\
                 /stage <name>   switch the active stage\n\
                 /model <name>   override the model for this session (/model default reverts)\n\
                 /reload         re-read the config file (MCP changes need a restart)\n\
                 /export [path]  write the transcript to a markdown file\n\
                 /sessions       open the session picker (switch or start new)\n\
                 /quit           exit (Ctrl+D on an empty prompt)\n\
                 mention files with @path (@src/main.rs, @\"has spaces.txt\", @somedir)\n\
                 typing / or @ pops up completions: Up/Down select · Tab/Enter accept\n\
                 typing while a turn runs queues the message: it is delivered to the\n\
                 model after the current tool round (or becomes the next turn)\n\
                 keys: Enter send · Alt+Enter newline · Up/Down recall past prompts\n\
                 PgUp/PgDn + mouse wheel scroll\n\
                 diff view: Tab/Shift+Tab switch file · j/k scroll · r restore · q close",
            ),
            "sessions" => self.open_sessions(),
            "clear" => {
                if self.is_running() {
                    self.error("cannot clear while a turn is running (Esc to cancel it first)");
                    return;
                }
                let freed = self.token_estimate();
                self.history.clear();
                self.checkpoints.clear(); // their history indexes are gone
                self.last_usage = None;
                let hidden_diffs = self.diffs.len() - self.diff_baseline;
                // A visible divider, then hide everything above it and every
                // diff so far. Nothing is deleted: the session file keeps the
                // full record, and /export still writes all of it.
                self.transcript.push(TranscriptItem::Cleared(format!(
                    "context cleared (freed ~{}){}",
                    fmt_tokens(freed as u64),
                    match hidden_diffs {
                        0 => String::new(),
                        n => format!(" · {n} earlier diff(s) hidden"),
                    },
                )));
                self.transcript_baseline = self.transcript.len() - 1;
                self.diff_baseline = self.diffs.len();
                self.scroll_from_bottom = 0;
            }
            "compact" => self.start_compact(),
            "usage" => {
                let lines = self.usage.report_lines();
                if lines.is_empty() {
                    self.info("no model requests completed yet");
                    return;
                }
                let (context, _) = self.context_status();
                self.info(format!(
                    "model usage since this TUI started:\n  {}\n  {context}",
                    lines.join("\n  "),
                ));
            }
            "diff" => self.toggle_diff_view(),
            "run" => self.start_workflow(arg),
            "rewind" => self.open_rewind(),
            "branch" => self.stash_branch(arg),
            "branches" => self.open_branches(),
            "stage" => self.switch_stage(arg),
            "model" => self.switch_model(arg),
            "reload" => self.reload_config(),
            "export" => self.export_transcript(arg),
            "quit" | "exit" => self.quit = true,
            other => self.error(format!("unknown command `/{other}` — try /help")),
        }
    }

    fn switch_model(&mut self, name: &str) {
        match name {
            "" => {
                let available: Vec<&str> = self.config.models.keys().map(String::as_str).collect();
                self.info(format!(
                    "model: `{}`{} (stage default `{}`)\n\
                     available: {}\n\
                     usage: /model <name> · /model default",
                    self.active_model(),
                    if self.model_override.is_some() {
                        " (override)"
                    } else {
                        ""
                    },
                    self.stage().model,
                    available.join(", "),
                ));
            }
            "default" => {
                self.model_override = None;
                self.info(format!(
                    "model override cleared — using the stage default `{}`",
                    self.stage().model
                ));
            }
            _ if self.config.models.contains_key(name) => {
                self.model_override = Some(name.to_string());
                self.info(format!(
                    "model set to `{name}` for every stage in this session (/model default reverts)"
                ));
            }
            _ => {
                let available: Vec<&str> = self.config.models.keys().map(String::as_str).collect();
                self.error(format!(
                    "no model named `{name}` — available: {}",
                    available.join(", ")
                ));
            }
        }
    }

    /// Re-read the config file in place: models, stages, prompts, settings,
    /// and project instructions. MCP topology stays as connected at startup.
    fn reload_config(&mut self) {
        if self.is_running() {
            return self.error("busy — wait for the current turn to finish");
        }
        let stage_name = self.stage().name.clone();
        match Config::load(&self.config_path) {
            Ok(config) => {
                self.config = Arc::new(config);
                self.stage_index = self
                    .config
                    .stages
                    .iter()
                    .position(|s| s.name == stage_name)
                    .unwrap_or_else(|| {
                        self.info(format!(
                            "stage `{stage_name}` is gone — switched to `{}`",
                            self.config.stages[0].name
                        ));
                        0
                    });
                if let Some(model) = &self.model_override
                    && !self.config.models.contains_key(model)
                {
                    self.info(format!(
                        "model override `{model}` no longer exists — cleared"
                    ));
                    self.model_override = None;
                }
                self.refresh_tool_count();
                self.info(format!(
                    "config reloaded: {} stage(s), {} model(s), {} project instruction file(s) \
                     — MCP server changes need a restart",
                    self.config.stages.len(),
                    self.config.models.len(),
                    self.config.project_contexts.len(),
                ));
            }
            Err(e) => self.error(format!("reload failed, config unchanged: {e:#}")),
        }
    }

    /// Write the transcript as markdown; refuses to overwrite.
    fn export_transcript(&mut self, arg: &str) {
        let name = if arg.is_empty() {
            format!("soa-session-{}.md", self.session_id)
        } else {
            arg.to_string()
        };
        let path = std::path::Path::new(&self.cwd).join(&name);
        if path.exists() {
            return self.error(format!("{} already exists — pass a different path", name));
        }

        let mut out = format!(
            "# soa session {}\n\n_started {} · exported {} · {}_\n",
            self.session_id,
            store::format_epoch(self.started_at),
            store::format_epoch(store::now_epoch()),
            self.stage_summary(),
        );
        for item in &self.transcript {
            match item {
                TranscriptItem::User(text) => out.push_str(&format!("\n## user\n\n{text}\n")),
                TranscriptItem::Assistant(text) => {
                    out.push_str(&format!("\n## assistant\n\n{text}\n"));
                }
                TranscriptItem::ToolCall { name, args } => {
                    let args: String = args.chars().take(200).collect();
                    out.push_str(&format!("\n> ⚒ `{name}` {args}\n"));
                }
                TranscriptItem::ToolDone { .. } => {}
                TranscriptItem::Cleared(text) => {
                    out.push_str(&format!("\n---\n\n> ⌫ {text}\n"));
                }
                TranscriptItem::Info(text) | TranscriptItem::Error(text) => {
                    for line in text.lines() {
                        out.push_str(&format!("> {line}\n"));
                    }
                }
            }
        }
        match std::fs::write(&path, out) {
            Ok(()) => self.info(format!("transcript exported to {}", path.display())),
            Err(e) => self.error(format!("cannot write {}: {e}", path.display())),
        }
    }

    fn switch_stage(&mut self, name: &str) {
        if name.is_empty() {
            let names: Vec<&str> = self.config.stages.iter().map(|s| s.name.as_str()).collect();
            self.info(format!(
                "usage: /stage <name> — available: {}",
                names.join(", ")
            ));
            return;
        }
        match self.config.stages.iter().position(|s| s.name == name) {
            Some(index) => {
                self.stage_index = index;
                self.refresh_tool_count();
                self.info(format!("switched to {}", self.stage_summary()));
            }
            None => self.error(format!("no stage named `{name}`")),
        }
    }

    // ------------------------------------------------------------------
    // Background work
    // ------------------------------------------------------------------

    fn start_turn(&mut self) {
        // Borrow the stage through the Arc, not through `self`, so the
        // fields below stay assignable.
        let config = Arc::clone(&self.config);
        let model_name = self.active_model().to_string();
        let stage = &config.stages[self.stage_index];
        let client = match build_model_client(
            &self.config,
            &model_name,
            stage.temperature,
            stage.max_tokens,
            &self.http,
            &self.usage,
        ) {
            Ok(client) => client,
            Err(e) => return self.error(format!("{e:#}")),
        };
        let system = match stage
            .resolve_system_prompt(&self.config.base_dir)
            .and_then(|system| {
                crate::skills::compose_system(
                    &config,
                    &format!("stage `{}`", stage.name),
                    system,
                    &stage.skills,
                )
            }) {
            Ok(system) => system,
            Err(e) => return self.error(format!("{e:#}")),
        };
        let stage_tools = match assemble_tools(&stage.tool_profile(), &self.config, &self.mcp, 0) {
            Ok(tools) => tools,
            Err(e) => return self.error(format!("{e:#}")),
        };
        self.tool_count = stage_tools.len();

        let max_turns = stage
            .max_turns
            .unwrap_or(self.config.settings.default_max_turns);

        self.steer_queue.lock().unwrap().clear();
        self.steered_this_turn.clear();
        self.turn_events.lock().unwrap().clear();
        let worker = turn_worker(
            client,
            stage_tools,
            Arc::clone(&self.config),
            Arc::clone(&self.mcp),
            self.http.clone(),
            self.usage.clone(),
            system,
            self.history.clone(),
            max_turns,
            self.tx.clone(),
            (
                stage.require_approval,
                stage.approval_effects.clone(),
                stage.auto_approve.clone(),
            ),
            Arc::clone(&self.approvals),
            stage.name.clone(),
            model_name,
            Arc::clone(&self.steer_queue),
            Arc::clone(&self.turn_events),
        );
        self.compacting = false;
        self.turn = Some(tokio::spawn(worker));
    }

    /// `/run [workflow] <task>`: execute a stage pipeline from the chat.
    /// Progress streams into the transcript; the final stage's output joins
    /// the conversation history, so the chat can continue from the result.
    fn start_workflow(&mut self, arg: &str) {
        if self.is_running() {
            return self.error("busy — wait for the current turn to finish (Esc cancels)");
        }
        let workflow_names: Vec<&str> = self.config.workflows.keys().map(String::as_str).collect();
        let Some((workflow, task)) = parse_run(arg, &workflow_names) else {
            return self.info(format!(
                "usage: /run [workflow] <task>{}",
                if workflow_names.is_empty() {
                    String::new()
                } else {
                    format!(" — workflows: {}", workflow_names.join(", "))
                },
            ));
        };
        let order = match self.config.resolve_workflow(workflow) {
            Ok(order) if order.is_empty() => return self.error("the selected workflow is empty"),
            Ok(order) => order,
            Err(e) => return self.error(format!("{e:#}")),
        };
        let label = workflow.unwrap_or("default").to_string();
        let banner = format!(
            "▶ workflow `{label}`: {}",
            order
                .iter()
                .map(|&i| self.config.stages[i].name.as_str())
                .collect::<Vec<_>>()
                .join(" → "),
        );

        self.has_activity = true;
        // A workflow run is a rewind target like any turn-starting message.
        self.checkpoints.push(Checkpoint {
            transcript_index: self.transcript.len(),
            history_len: self.history.len(),
            diff_len: self.diffs.len(),
        });
        let (expanded, reports) = crate::mentions::expand_mentions(
            task,
            std::path::Path::new(&self.cwd),
            self.config.settings.max_tool_output_chars,
        );
        self.transcript.push(TranscriptItem::User(task.to_string()));
        for report in &reports {
            self.info(report.describe());
        }
        self.info(banner);
        self.scroll_from_bottom = 0;

        self.steer_queue.lock().unwrap().clear();
        self.steered_this_turn.clear();
        self.turn_events.lock().unwrap().clear();
        let worker = workflow_worker(
            Arc::clone(&self.config),
            Arc::clone(&self.mcp),
            self.http.clone(),
            Arc::clone(&self.approvals),
            order,
            label.clone(),
            expanded,
            self.tx.clone(),
        );
        self.compacting = false;
        self.workflow_running = Some(label);
        self.turn = Some(tokio::spawn(worker));
        self.persist();
    }

    fn start_compact(&mut self) {
        if self.is_running() {
            return self.error("busy — wait for the current turn to finish");
        }
        if self.history.is_empty() {
            return self.info("nothing to compact");
        }
        let stage = self.stage();
        let client = match build_model_client(
            &self.config,
            self.active_model(),
            stage.temperature,
            stage.max_tokens,
            &self.http,
            &self.usage,
        ) {
            Ok(client) => client,
            Err(e) => return self.error(format!("{e:#}")),
        };
        let mut request = self.history.clone();
        request.push(Message::User {
            content: COMPACT_INSTRUCTION.to_string(),
        });
        let tx = self.tx.clone();
        self.compacting = true;
        self.turn_events.lock().unwrap().clear();
        self.turn = Some(tokio::spawn(async move {
            let event = match client.complete(&request, &[]).await {
                Ok(reply) => AgentEvent::Compacted {
                    summary: reply.content.unwrap_or_default(),
                },
                Err(e) => AgentEvent::Error(format!("compaction failed: {e:#}")),
            };
            let _ = tx.send(event);
        }));
    }

    /// Move a partially streamed reply into the transcript.
    fn flush_stream_buffer(&mut self) {
        let partial = std::mem::take(&mut self.stream_buffer);
        let partial = partial.trim_end();
        if !partial.is_empty() {
            self.transcript
                .push(TranscriptItem::Assistant(partial.to_string()));
        }
    }

    /// Kick off compaction automatically when real usage crosses the
    /// threshold of the model's declared context window.
    fn maybe_auto_compact(&mut self) {
        let threshold = self.config.settings.auto_compact_threshold;
        if threshold <= 0.0 || self.is_running() || self.history.len() <= 2 {
            return;
        }
        let (Some(capacity), Some(usage)) = (self.context_capacity(), self.last_usage) else {
            return;
        };
        let used = usage.context_tokens();
        if (used as f64) < capacity as f64 * threshold {
            return;
        }
        self.info(format!(
            "context at {}% ({} of {}) — auto-compacting",
            used * 100 / capacity.max(1),
            fmt_tokens(used),
            fmt_tokens(capacity),
        ));
        self.start_compact();
    }

    pub fn cancel_turn(&mut self) {
        if let Some(handle) = self.turn.take() {
            handle.abort();
            self.compacting = false;
            let workflow = self.workflow_running.take();
            self.pending_approval = None; // dropped responder reads as deny
            self.flush_stream_buffer();
            self.recover_interrupted_turn();
            self.info(match workflow {
                Some(name) => format!(
                    "cancelled workflow `{name}` — file changes it already made remain \
                     (Ctrl+G to review or restore)"
                ),
                None => "cancelled".to_string(),
            });
            self.persist();
        }
    }

    /// Restore one captured change (diff viewer `r`): put the file back
    /// into that entry's pre-change state. The reverse entry is recorded so
    /// the restore itself can be undone.
    fn restore_diff(&mut self, index: usize) {
        if self.is_running() {
            return self.error("cannot restore while a turn is running (Esc to cancel it first)");
        }
        let Some(entry) = self.diffs.get(index).cloned() else {
            return;
        };
        match diff::restore(&entry) {
            Ok(Some(reverse)) => {
                self.info(format!(
                    "restored {} to its state before `{}` — undo via the new rewind entry",
                    entry.path, entry.tool
                ));
                self.diffs.push(reverse);
                self.persist();
            }
            Ok(None) => self.info(format!(
                "{} already matches the state recorded before `{}`",
                entry.path, entry.tool
            )),
            Err(message) => self.error(message),
        }
    }

    /// A full snapshot of the live conversation, for a branch slot.
    fn snapshot_branch(&self, name: String) -> Branch {
        Branch {
            name,
            created_at: store::now_epoch(),
            transcript: self.transcript.clone(),
            history: self.history.clone(),
            checkpoints: self.checkpoints.clone(),
            transcript_baseline: self.transcript_baseline,
            diff_baseline: self.diff_baseline,
        }
    }

    /// A branch name that isn't taken: b1, b2, …
    fn next_branch_name(&self) -> String {
        (1..)
            .map(|n| format!("b{n}"))
            .find(|name| !self.branches.iter().any(|b| &b.name == name))
            .expect("some name is free")
    }

    /// `/branch <name>`: store a copy of the conversation as a branch and
    /// keep going. `/branch` with no name opens the picker.
    fn stash_branch(&mut self, name: &str) {
        if name.is_empty() {
            return self.open_branches();
        }
        if self.branches.iter().any(|b| b.name == name) {
            return self.error(format!("branch `{name}` already exists (see /branches)"));
        }
        if self.transcript.is_empty() {
            return self.info("nothing to branch — the conversation is empty");
        }
        self.branches.push(self.snapshot_branch(name.to_string()));
        self.has_activity = true;
        self.info(format!(
            "saved the conversation as branch `{name}` — /branches switches between lines"
        ));
        self.persist();
    }

    fn open_branches(&mut self) {
        if self.branches.is_empty() {
            return self.info(
                "no branches yet — /branch <name> saves the current line, and /rewind \
                 stores the abandoned one automatically",
            );
        }
        self.view = View::Branches {
            selected: 0,
            scroll: 0,
        };
    }

    fn on_branches_key(&mut self, key: KeyEvent) {
        let View::Branches { selected, .. } = &mut self.view else {
            return;
        };
        let current = *selected;
        let last = self.branches.len().saturating_sub(1);
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.view = View::Chat,
            KeyCode::Up | KeyCode::Char('k') => *selected = current.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => *selected = (current + 1).min(last),
            KeyCode::Home | KeyCode::Char('g') => *selected = 0,
            KeyCode::End | KeyCode::Char('G') => *selected = last,
            KeyCode::Enter => self.swap_branch(current),
            KeyCode::Char('d') => self.delete_branch(current),
            _ => {}
        }
    }

    /// Swap the live conversation with a branch slot: the branch's line
    /// becomes live, and the slot now holds what was live — so switching
    /// never loses anything and never grows the list.
    fn swap_branch(&mut self, index: usize) {
        self.view = View::Chat;
        if self.is_running() {
            return self.error("finish or cancel the running turn before switching branches");
        }
        let Some(branch) = self.branches.get_mut(index) else {
            return;
        };
        std::mem::swap(&mut branch.transcript, &mut self.transcript);
        std::mem::swap(&mut branch.history, &mut self.history);
        std::mem::swap(&mut branch.checkpoints, &mut self.checkpoints);
        // Each line keeps its own /clear baselines. The diff log is shared
        // and append-only, so a stored diff baseline stays valid; clamp
        // anyway in case the branch predates baseline tracking.
        std::mem::swap(
            &mut branch.transcript_baseline,
            &mut self.transcript_baseline,
        );
        std::mem::swap(&mut branch.diff_baseline, &mut self.diff_baseline);
        self.transcript_baseline = self.transcript_baseline.min(self.transcript.len());
        self.diff_baseline = self.diff_baseline.min(self.diffs.len());
        branch.created_at = store::now_epoch();
        let name = branch.name.clone();
        self.last_usage = None;
        self.scroll_from_bottom = 0;
        self.has_activity = true;
        self.info(format!(
            "switched to branch `{name}` — the line you left now lives in that slot"
        ));
        self.persist();
    }

    fn delete_branch(&mut self, index: usize) {
        if index >= self.branches.len() {
            return;
        }
        let branch = self.branches.remove(index);
        self.info(format!(
            "deleted branch `{}` ({})",
            branch.name,
            branch.title()
        ));
        if self.branches.is_empty() {
            self.view = View::Chat;
        }
        self.persist();
    }

    /// `/rewind`: open the checkpoint picker (newest message first, plus a
    /// session-start row).
    fn open_rewind(&mut self) {
        if self.is_running() {
            return self.error("cannot rewind while a turn is running (Esc to cancel it first)");
        }
        if self.checkpoints.is_empty() && self.diffs.iter().all(|d| !d.restorable()) {
            return self.info(
                "nothing to rewind — no checkpoints this session and no restorable file changes",
            );
        }
        self.view = View::Rewind {
            selected: 0,
            scroll: 0,
        };
    }

    fn on_rewind_key(&mut self, key: KeyEvent) {
        let View::Rewind { selected, .. } = &mut self.view else {
            return;
        };
        let current = *selected;
        let last = self.checkpoints.len(); // rows = checkpoints + session-start row
        let select = |view: &mut View, index: usize| {
            if let View::Rewind { selected, .. } = view {
                *selected = index;
            }
        };
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.view = View::Chat,
            KeyCode::Up | KeyCode::Char('k') => select(&mut self.view, current.saturating_sub(1)),
            KeyCode::Down | KeyCode::Char('j') => select(&mut self.view, (current + 1).min(last)),
            KeyCode::Home | KeyCode::Char('g') => select(&mut self.view, 0),
            KeyCode::End | KeyCode::Char('G') => select(&mut self.view, last),
            KeyCode::Enter => {
                // Row k in 0..len is the (len-1-k)-th checkpoint (newest
                // first); the last row is session start.
                let target = self.checkpoints.len().checked_sub(current + 1);
                self.rewind_to(target);
            }
            _ => {}
        }
    }

    /// Rewind to a checkpoint (`None` = session start): restore every file
    /// touched afterwards to its state at that moment, truncate the
    /// conversation, and put the rewound message back into the input.
    /// File restores are recorded as `rewind` diff entries, so a rewind
    /// that went too far can be re-applied forward from the diff viewer.
    fn rewind_to(&mut self, index: Option<usize>) {
        self.view = View::Chat;
        if self.is_running() {
            return self.error("cannot rewind while a turn is running (Esc to cancel it first)");
        }
        let checkpoint = match index {
            Some(i) => match self.checkpoints.get(i) {
                Some(checkpoint) => *checkpoint,
                None => return,
            },
            None => Checkpoint {
                transcript_index: 0,
                history_len: 0,
                diff_len: 0,
            },
        };

        // Files first: undo everything recorded at or after the checkpoint.
        let targets = diff::earliest_restorable_since(&self.diffs, checkpoint.diff_len);
        let (mut restored, mut errors) = (Vec::new(), Vec::new());
        for entry in targets {
            match diff::restore(&entry) {
                Ok(Some(reverse)) => {
                    restored.push(entry.path.clone());
                    self.diffs.push(reverse);
                }
                Ok(None) => {} // already at the checkpoint state
                Err(message) => errors.push(message),
            }
        }
        for message in errors {
            self.error(message);
        }

        // The abandoned line is stashed as a branch, not destroyed —
        // /branches returns to it.
        let stashed = if checkpoint.transcript_index < self.transcript.len() {
            let name = self.next_branch_name();
            self.branches.push(self.snapshot_branch(name.clone()));
            Some(name)
        } else {
            None
        };

        // Then the conversation: drop the rewound message and everything
        // after it, and hand the message text back for editing.
        let message = match self.transcript.get(checkpoint.transcript_index) {
            Some(TranscriptItem::User(text)) => Some(text.clone()),
            _ => None,
        };
        self.transcript.truncate(checkpoint.transcript_index);
        self.history.truncate(checkpoint.history_len);
        self.checkpoints.truncate(index.unwrap_or(0));
        // Rewinding past a /clear rewinds the clear too. Surviving
        // checkpoints are always post-clear (/clear drops them), so only
        // the session-start row (diff_len 0) actually lowers the baselines.
        self.transcript_baseline = self.transcript_baseline.min(self.transcript.len());
        self.diff_baseline = self.diff_baseline.min(checkpoint.diff_len);
        self.last_usage = None;
        self.scroll_from_bottom = 0;
        if let Some(text) = &message {
            self.set_input_text(text);
        }

        self.info(format!(
            "rewound to {}{}{}{}",
            match index {
                Some(_) => "before the selected message",
                None => "the start of the session",
            },
            match restored.len() {
                0 => String::new(),
                n => format!(" — restored {n} file(s): {}", restored.join(", ")),
            },
            match &stashed {
                Some(name) => format!(" — the abandoned line is saved as branch `{name}`"),
                None => String::new(),
            },
            if message.is_some() {
                " (the message is back in the input)"
            } else {
                ""
            },
        ));
        self.persist();
    }

    /// Recover from an interrupted (cancelled or failed) turn: fold the
    /// tool rounds the worker already completed back into `history` — their
    /// file effects are on disk, and a model that cannot see them will
    /// contradict the filesystem — then keep any undelivered steered
    /// messages. Falls back to [`Self::preserve_steered`] when the turn
    /// recorded nothing (workflows, compaction, pre-model cancels).
    fn recover_interrupted_turn(&mut self) {
        let events = std::mem::take(&mut *self.turn_events.lock().unwrap());
        match salvage_cancelled_loop(&events) {
            Ok(Some(messages)) => {
                let folded = messages.len().saturating_sub(self.history.len());
                if folded > 0 {
                    self.history = messages;
                    self.info(format!(
                        "kept {folded} message(s) from the interrupted turn in context — \
                         the model will see the tool calls that already ran"
                    ));
                }
                // Steered messages the loop already delivered are part of
                // the salvaged history; only the undelivered queue is
                // re-added.
                let leftovers: Vec<String> = self.steer_queue.lock().unwrap().drain(..).collect();
                self.steered_this_turn.clear();
                if !leftovers.is_empty() {
                    for content in leftovers {
                        self.history.push(Message::User { content });
                    }
                    self.info("queued message(s) kept in context for the next turn");
                }
            }
            result => {
                if let Err(e) = result {
                    tracing::warn!(
                        error = format!("{e:#}"),
                        "could not salvage the interrupted turn's events"
                    );
                }
                self.preserve_steered();
            }
        }
    }

    /// A failed or cancelled turn discards the worker's history clone, and
    /// any steered messages with it — keep them in the surviving history so
    /// the next turn still sees them.
    fn preserve_steered(&mut self) {
        self.steer_queue.lock().unwrap().clear();
        let queued = std::mem::take(&mut self.steered_this_turn);
        if queued.is_empty() {
            return;
        }
        for content in queued {
            self.history.push(Message::User { content });
        }
        self.info("queued message(s) kept in context for the next turn");
    }

    pub fn abort_turn(&mut self) {
        if let Some(handle) = self.turn.take() {
            handle.abort();
        }
    }

    pub fn status_word(&self) -> &'static str {
        if self.compacting {
            "compacting"
        } else if self.workflow_running.is_some() {
            "running workflow"
        } else {
            "thinking"
        }
    }

    /// Steered messages waiting to be delivered to the running turn.
    pub fn queued_count(&self) -> usize {
        self.steer_queue.lock().unwrap().len()
    }

    // ------------------------------------------------------------------
    // Agent events
    // ------------------------------------------------------------------

    pub fn on_agent_event(&mut self, event: AgentEvent) {
        let should_save = matches!(
            event,
            AgentEvent::Turn { .. }
                | AgentEvent::Compacted { .. }
                | AgentEvent::Error(_)
                | AgentEvent::Diff(_)
                | AgentEvent::WorkflowDone { .. }
        );
        match event {
            AgentEvent::Delta(fragment) => {
                self.stream_buffer.push_str(&fragment);
            }
            AgentEvent::ToolCall { name, args } => {
                // Content streamed before a tool call is commentary the
                // final answer won't repeat — keep it.
                self.flush_stream_buffer();
                self.transcript
                    .push(TranscriptItem::ToolCall { name, args });
            }
            AgentEvent::ToolDone { preview } => {
                self.transcript.push(TranscriptItem::ToolDone { preview });
            }
            AgentEvent::Diff(entry) => {
                self.info(format!("✎ {} — Ctrl+G to view", entry.title()));
                self.diffs.push(entry);
            }
            AgentEvent::Notice(text) => self.info(text),
            AgentEvent::Turn {
                history,
                text,
                usage,
            } => {
                // The buffer holds the same text that just streamed; the
                // final Turn event is authoritative.
                self.stream_buffer.clear();
                self.turn_events.lock().unwrap().clear();
                self.history = history;
                self.last_usage = usage;
                if text.trim().is_empty() {
                    self.info("(model returned an empty response)");
                } else {
                    self.transcript.push(TranscriptItem::Assistant(text));
                }
                self.turn = None;
                // Steered messages the worker never got to (the reply had no
                // further tool rounds) become the next turn immediately.
                let leftovers: Vec<String> = self.steer_queue.lock().unwrap().drain(..).collect();
                self.steered_this_turn.clear();
                if leftovers.is_empty() {
                    self.maybe_auto_compact();
                } else {
                    for content in leftovers {
                        self.history.push(Message::User { content });
                    }
                    self.info("↪ sending queued message(s)");
                    self.start_turn();
                }
            }
            AgentEvent::Compacted { summary } => {
                let before = self.token_estimate();
                // Checkpoints index into the history being replaced.
                self.checkpoints.clear();
                self.history = vec![
                    Message::User {
                        content: format!(
                            "[Summary of the conversation so far — earlier messages were compacted]\n\n{summary}"
                        ),
                    },
                    Message::Assistant {
                        content: Some("Understood — continuing with that context.".to_string()),
                        tool_calls: None,
                    },
                ];
                let after = self.token_estimate();
                self.turn = None;
                self.compacting = false;
                self.last_usage = None; // real usage returns on the next turn
                self.info(format!(
                    "context compacted: ~{} → ~{}",
                    fmt_tokens(before as u64),
                    fmt_tokens(after as u64)
                ));
            }
            AgentEvent::StageStart { stage, run } => {
                // Keep the previous stage's streamed output as transcript
                // content, then mark the boundary.
                self.flush_stream_buffer();
                self.info(format!("── stage {stage} (run {run}) ──"));
            }
            AgentEvent::WorkflowDone {
                workflow,
                task,
                result,
                usage,
            } => {
                self.turn = None;
                self.workflow_running = None;
                self.pending_approval = None;
                if !usage.is_empty() {
                    self.info(format!("workflow usage:\n  {}", usage.join("\n  ")));
                }
                match result {
                    Ok(output) => {
                        // The buffer holds the final stage's streamed text;
                        // the event's copy is authoritative.
                        self.stream_buffer.clear();
                        // The exchange joins the conversation, so the chat
                        // model can discuss what the pipeline produced.
                        self.history.push(Message::User { content: task });
                        self.history.push(Message::Assistant {
                            content: Some(output.clone()),
                            tool_calls: None,
                        });
                        if output.trim().is_empty() {
                            self.info(format!("workflow `{workflow}` finished (empty output)"));
                        } else {
                            self.transcript.push(TranscriptItem::Assistant(output));
                        }
                        self.info(format!(
                            "✔ workflow `{workflow}` finished — its result is in this \
                             conversation's context"
                        ));
                        self.last_usage = None; // per-turn usage doesn't span pipelines
                        // Messages typed during the run become the next turn.
                        let leftovers: Vec<String> =
                            self.steer_queue.lock().unwrap().drain(..).collect();
                        self.steered_this_turn.clear();
                        if !leftovers.is_empty() {
                            for content in leftovers {
                                self.history.push(Message::User { content });
                            }
                            self.info("↪ sending queued message(s)");
                            self.start_turn();
                        }
                    }
                    Err(message) => {
                        self.flush_stream_buffer();
                        self.error(format!("workflow `{workflow}` failed: {message}"));
                        self.preserve_steered();
                    }
                }
            }
            AgentEvent::Error(message) => {
                self.flush_stream_buffer();
                self.error(message);
                self.turn = None;
                self.compacting = false;
                self.recover_interrupted_turn();
            }
        }
        if should_save {
            self.persist();
        }
    }
}

/// Split a `/run` argument into an optional workflow name and the task:
/// the first word selects a workflow when it names one, otherwise the
/// whole argument is the task for the default workflow. None means there
/// is no task to run (empty, or a workflow name with nothing after it).
fn parse_run<'a>(arg: &'a str, workflows: &[&str]) -> Option<(Option<&'a str>, &'a str)> {
    let arg = arg.trim();
    let (first, rest) = arg.split_once(char::is_whitespace).unwrap_or((arg, ""));
    if workflows.contains(&first) {
        let task = rest.trim();
        return (!task.is_empty()).then_some((Some(first), task));
    }
    (!arg.is_empty()).then_some((None, arg))
}

/// A `/run` pipeline as a background task: the stage loop from `soa run`,
/// reporting through [`AgentEvent`]s instead of stderr — stage banners,
/// streamed content, captured file diffs, and a final [`AgentEvent::WorkflowDone`].
#[allow(clippy::too_many_arguments)]
async fn workflow_worker(
    config: Arc<Config>,
    mcp: Arc<McpManager>,
    http: reqwest::Client,
    approvals: Arc<Approvals>,
    order: Vec<usize>,
    workflow: String,
    task: String,
    tx: UnboundedSender<AgentEvent>,
) {
    let usage = UsageTracker::new(
        config.settings.run_limits(),
        crate::model::UsageSnapshot::default(),
    );
    let delta_tx = tx.clone();
    let on_delta = move |fragment: &str| {
        let _ = delta_tx.send(AgentEvent::Delta(fragment.to_string()));
    };
    let diff_tx = tx.clone();
    let on_diff = move |entry: crate::diff::DiffEntry| {
        let _ = diff_tx.send(AgentEvent::Diff(entry));
    };
    let done = |result: Result<String, String>| AgentEvent::WorkflowDone {
        workflow: workflow.clone(),
        task: task.clone(),
        result,
        usage: usage.report_lines(),
    };

    let stage_names: Vec<&str> = order
        .iter()
        .map(|&i| config.stages[i].name.as_str())
        .collect();
    let mut context = crate::stage::PipelineContext::new(&task);
    let mut position = 0usize;
    let mut runs = 0u32;
    let mut last_output = String::new();

    while position < order.len() {
        let stage = &config.stages[order[position]];
        runs += 1;
        if runs > config.settings.max_stage_runs {
            let _ = tx.send(done(Err(format!(
                "stopped after {} stage runs without finishing — likely a reprompt \
                 loop; raise settings.max_stage_runs if this is intentional",
                config.settings.max_stage_runs
            ))));
            return;
        }
        let _ = tx.send(AgentEvent::StageStart {
            stage: stage.name.clone(),
            run: runs,
        });

        let is_first = context.previous.is_none();
        let reprompt_targets: Vec<String> = stage
            .can_reprompt
            .iter()
            .filter(|t| stage_names.contains(&t.as_str()))
            .cloned()
            .collect();
        let stage_run = crate::stage::run_stage(
            &config,
            stage,
            is_first,
            &context,
            &mcp,
            &http,
            &usage,
            &[],
            None,
            &reprompt_targets,
            Some(&on_delta),
            Some(&on_diff),
            &approvals,
        );
        match usage.within_time(stage_run).await {
            Ok(crate::stage::StageOutcome::Final(output)) => {
                context.record(&stage.name, output.clone());
                last_output = output;
                position += 1;
            }
            Ok(crate::stage::StageOutcome::Reprompt {
                target,
                instructions,
            }) => {
                let _ = tx.send(AgentEvent::Notice(format!(
                    "↩ {} reprompts {}: {instructions}",
                    stage.name, target
                )));
                context.record(&stage.name, instructions);
                position = stage_names
                    .iter()
                    .position(|name| *name == target)
                    .expect("reprompt targets are filtered to the workflow");
            }
            Err(e) => {
                let _ = tx.send(done(Err(format!("{e:#}"))));
                return;
            }
        }
    }
    let _ = tx.send(done(Ok(last_output)));
}

/// One full agentic turn, run as a background task. Owns a clone of the
/// history; the updated history is handed back via [`AgentEvent::Turn`].
#[allow(clippy::too_many_arguments)]
async fn turn_worker(
    client: ModelClient,
    tools: Vec<StageTool>,
    config: Arc<Config>,
    mcp: Arc<McpManager>,
    http: reqwest::Client,
    usage: UsageTracker,
    system: Option<String>,
    history: Vec<Message>,
    max_turns: u32,
    tx: UnboundedSender<AgentEvent>,
    (require_approval, approval_effects, auto_approve): (bool, Vec<ToolEffect>, Vec<String>),
    approvals: Arc<Approvals>,
    stage_name: String,
    model_name: String,
    steer: Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    events: Arc<std::sync::Mutex<Vec<AgentLoopEvent>>>,
) {
    let delta_tx = tx.clone();
    let on_delta = move |fragment: &str| {
        let _ = delta_tx.send(AgentEvent::Delta(fragment.to_string()));
    };
    // Mirror every loop event into shared state the UI thread can read: if
    // this task is aborted mid-turn, the completed rounds recorded here are
    // salvaged back into the conversation history.
    let on_event = move |event: AgentLoopEvent| {
        events.lock().unwrap().push(event);
    };
    let observation_tx = tx.clone();
    let on_observation = move |event: AgentLoopObservation| match event {
        AgentLoopObservation::ToolCall { name, args } => {
            let _ = observation_tx.send(AgentEvent::ToolCall { name, args });
        }
        AgentLoopObservation::ToolDone { preview } => {
            let _ = observation_tx.send(AgentEvent::ToolDone { preview });
        }
        AgentLoopObservation::Notice(message) => {
            let _ = observation_tx.send(AgentEvent::Notice(message));
        }
    };
    let diff_tx = tx.clone();
    let on_diff = move |entry: DiffEntry| {
        let _ = diff_tx.send(AgentEvent::Diff(entry));
    };

    match run_agent_loop(
        &client,
        &tools,
        history,
        &[],
        &config,
        &mcp,
        &http,
        &usage,
        &approvals,
        AgentLoopOptions {
            owner_kind: "stage",
            owner: &stage_name,
            model_name: &model_name,
            system: system.as_deref(),
            max_turns,
            depth: 0,
            require_approval,
            approval_effects: &approval_effects,
            auto_approve: &auto_approve,
            reprompt_targets: &[],
            on_delta: Some(&on_delta),
            terminate_streamed_response: false,
            on_diff: Some(&on_diff),
            on_event: Some(&on_event),
            on_observation: Some(&on_observation),
            steer: Some(&steer),
            tool_errors_as_results: true,
        },
    )
    .await
    {
        Ok(result) => match result.outcome {
            crate::stage::StageOutcome::Final(text) => {
                let _ = tx.send(AgentEvent::Turn {
                    history: result.messages,
                    text,
                    usage: result.usage,
                });
            }
            crate::stage::StageOutcome::Reprompt { .. } => {
                unreachable!("chat turns have no reprompt targets")
            }
        },
        Err(error) => {
            let _ = tx.send(AgentEvent::Error(format!("{error:#}")));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_app() -> App {
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
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        App::new(
            Arc::new(config),
            PathBuf::from("soa.toml"),
            Arc::new(McpManager::default()),
            0,
            tx,
            Arc::new(Approvals::non_interactive()),
            None,
        )
    }

    fn entry(path: &str) -> DiffEntry {
        DiffEntry {
            tool: "edit_file".to_string(),
            path: path.to_string(),
            unified: String::new(),
            added: 1,
            removed: 0,
            // Unrestorable, so rewinds in tests never touch the filesystem.
            before: crate::diff::Snapshot::Unavailable,
        }
    }

    #[test]
    fn clear_hides_but_preserves_conversation_and_diffs() {
        let mut app = test_app();
        let greeting_len = app.transcript.len();
        app.history.push(Message::User {
            content: "hi".to_string(),
        });
        app.transcript.push(TranscriptItem::User("hi".to_string()));
        app.transcript
            .push(TranscriptItem::Assistant("hello".to_string()));
        app.diffs.push(entry("a.rs"));
        app.checkpoints.push(Checkpoint {
            transcript_index: greeting_len,
            history_len: 0,
            diff_len: 0,
        });

        app.run_command("clear");

        // Model context and rewind targets are gone; the record is not.
        assert!(app.history.is_empty());
        assert!(app.checkpoints.is_empty());
        assert_eq!(app.transcript.len(), greeting_len + 3); // + divider
        assert_eq!(app.diffs.len(), 1);
        // The UI sees only the divider, and no diffs.
        assert!(
            matches!(app.visible_transcript(), [TranscriptItem::Cleared(text)]
            if text.contains("1 earlier diff(s) hidden"))
        );
        assert!(app.visible_diffs().is_empty());

        // Post-clear activity is visible; the diff count starts fresh.
        app.diffs.push(entry("b.rs"));
        assert_eq!(app.visible_diffs().len(), 1);
        assert_eq!(app.visible_diffs()[0].path, "b.rs");

        // A second clear stacks: the new divider hides b.rs too.
        app.run_command("clear");
        assert!(app.visible_diffs().is_empty());
        assert_eq!(app.diffs.len(), 2);
    }

    #[test]
    fn rewind_to_session_start_rewinds_the_clear() {
        let mut app = test_app();
        app.transcript.push(TranscriptItem::User("hi".to_string()));
        app.diffs.push(entry("a.rs"));
        app.run_command("clear");
        assert!(app.transcript_baseline > 0);
        assert_eq!(app.diff_baseline, 1);

        app.rewind_to(None);
        assert_eq!(app.transcript_baseline, 0);
        assert_eq!(app.diff_baseline, 0);
        // Conversation truncated (only the rewind notice remains)…
        assert!(matches!(
            app.transcript.as_slice(),
            [TranscriptItem::Info(_)]
        ));
        assert_eq!(app.diffs.len(), 1); // …but the diff log survives
    }

    #[test]
    fn parse_run_splits_workflow_and_task() {
        let workflows = ["default", "quickfix"];
        // First word naming a workflow selects it; the rest is the task.
        assert_eq!(
            parse_run("quickfix fix the bug", &workflows),
            Some((Some("quickfix"), "fix the bug"))
        );
        // Otherwise the whole argument is the task for the default workflow.
        assert_eq!(
            parse_run("fix the bug", &workflows),
            Some((None, "fix the bug"))
        );
        // A task that merely starts with a workflow-like word still works.
        assert_eq!(
            parse_run("quickfixes are bad", &workflows),
            Some((None, "quickfixes are bad"))
        );
        // No task → no run.
        assert_eq!(parse_run("", &workflows), None);
        assert_eq!(parse_run("   ", &workflows), None);
        assert_eq!(parse_run("quickfix", &workflows), None);
        assert_eq!(parse_run("quickfix   ", &workflows), None);
    }

    #[test]
    fn workflow_done_joins_the_conversation() {
        let mut app = test_app();
        app.transcript
            .push(TranscriptItem::User("do the thing".to_string()));

        app.on_agent_event(AgentEvent::StageStart {
            stage: "s".to_string(),
            run: 1,
        });
        assert!(
            matches!(app.transcript.last(), Some(TranscriptItem::Info(text))
            if text.contains("stage s"))
        );

        app.workflow_running = Some("default".to_string());
        app.stream_buffer = "the final answer".to_string(); // streamed copy
        app.on_agent_event(AgentEvent::WorkflowDone {
            workflow: "default".to_string(),
            task: "do the thing (expanded)".to_string(),
            result: Ok("the final answer".to_string()),
            usage: Vec::new(),
        });

        // The exchange landed in the model history (expanded task, output),
        // the transcript shows the output once, and the run state cleared.
        assert!(app.workflow_running.is_none());
        assert!(!app.is_running());
        assert!(app.stream_buffer.is_empty());
        assert!(matches!(&app.history[..], [
            Message::User { content },
            Message::Assistant { content: Some(output), .. },
        ] if content == "do the thing (expanded)" && output == "the final answer"));
        let answers = app
            .transcript
            .iter()
            .filter(|t| matches!(t, TranscriptItem::Assistant(text) if text == "the final answer"))
            .count();
        assert_eq!(answers, 1);

        // A failed workflow reports the error and leaves history untouched.
        app.workflow_running = Some("default".to_string());
        app.on_agent_event(AgentEvent::WorkflowDone {
            workflow: "default".to_string(),
            task: "again".to_string(),
            result: Err("stage `s` blew up".to_string()),
            usage: Vec::new(),
        });
        assert_eq!(app.history.len(), 2);
        assert!(
            matches!(app.transcript.last(), Some(TranscriptItem::Error(text))
            if text.contains("blew up"))
        );
    }

    #[test]
    fn interrupted_turns_fold_completed_tool_work_into_history() {
        let mut app = test_app();
        app.history = vec![Message::User {
            content: "edit the file".to_string(),
        }];
        let call = crate::model::ToolCall {
            id: "c1".to_string(),
            function: crate::model::FunctionCall {
                name: "write_file".to_string(),
                arguments: serde_json::json!({}),
            },
        };
        *app.turn_events.lock().unwrap() = vec![
            AgentLoopEvent::Started {
                system: None,
                messages: app.history.clone(),
            },
            AgentLoopEvent::Assistant {
                content: None,
                tool_calls: vec![call],
                usage: None,
            },
            AgentLoopEvent::ToolResult {
                call_index: 0,
                content: "wrote `x`".to_string(),
            },
        ];
        // One steered message was never delivered to the worker.
        app.steer_queue
            .lock()
            .unwrap()
            .push_back("also do y".to_string());
        app.steered_this_turn.push("also do y".to_string());

        app.recover_interrupted_turn();

        // The completed tool round survives, followed by the undelivered
        // steered message; the event log and steer state are consumed.
        assert_eq!(app.history.len(), 4);
        assert!(matches!(
            &app.history[1],
            Message::Assistant {
                tool_calls: Some(_),
                ..
            }
        ));
        assert!(matches!(&app.history[2], Message::Tool { .. }));
        assert!(
            matches!(&app.history[3], Message::User { content } if content == "also do y")
        );
        assert!(app.turn_events.lock().unwrap().is_empty());
        assert!(app.steered_this_turn.is_empty());
        assert!(app.steer_queue.lock().unwrap().is_empty());

        // With no recorded events (a cancelled workflow, a compaction), the
        // conservative fallback keeps all steered messages.
        app.steered_this_turn.push("plan b".to_string());
        app.recover_interrupted_turn();
        assert!(
            matches!(app.history.last(), Some(Message::User { content }) if content == "plan b")
        );
    }

    #[test]
    fn branches_carry_their_own_clear_baselines() {
        let mut app = test_app();
        app.transcript
            .push(TranscriptItem::User("first line".to_string()));
        app.run_command("clear");
        let cleared_baseline = app.transcript_baseline;
        assert!(cleared_baseline > 0);

        // Stash the cleared line, then swap to it from a fresh one.
        app.stash_branch("stashed");
        app.transcript_baseline = 0; // pretend the live line never cleared
        app.swap_branch(0);
        assert_eq!(app.transcript_baseline, cleared_baseline);
        // Swapping back restores the uncleared view.
        app.swap_branch(0);
        assert_eq!(app.transcript_baseline, 0);
    }
}
