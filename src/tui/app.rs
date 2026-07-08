//! Chat TUI state, input handling, and the background agent worker.

use std::collections::HashMap;
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
use crate::config::{Config, Mode, Stage};
use crate::diff::{self, DiffEntry};
use crate::mcp::McpManager;
use crate::provider::{ChatClient, ChatMessage, ToolFunction, Usage, fmt_tokens, usage_stats};
use crate::stage::{
    CallPolicy, ToolBinding, assemble_tools, build_client, context_pressure,
    dispatch_tool_call, shed_context,
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
    ToolCall { name: String, args: String },
    ToolDone { preview: String },
    Info(String),
    Error(String),
}

/// Events sent by the background worker back to the UI loop.
pub enum AgentEvent {
    /// A streamed fragment of the assistant's in-progress reply.
    Delta(String),
    ToolCall { name: String, args: String },
    ToolDone { preview: String },
    Diff(DiffEntry),
    /// Something the user should know mid-turn (e.g. context shedding).
    Notice(String),
    /// Turn finished: the updated history, the assistant's final text, and
    /// the last reported token usage.
    Turn { history: Vec<ChatMessage>, text: String, usage: Option<Usage> },
    Compacted { summary: String },
    Error(String),
}

pub enum View {
    Chat,
    Diffs { selected: usize, scroll: usize },
    /// Session picker over `App::session_list`.
    Sessions { selected: usize, scroll: usize },
    /// Rewind picker over `App::checkpoints` (newest first, plus a
    /// session-start row at the bottom).
    Rewind { selected: usize, scroll: usize },
    /// Branch picker over `App::branches`.
    Branches { selected: usize, scroll: usize },
}

pub struct App {
    pub config: Arc<Config>,
    pub mcp: Arc<McpManager>,
    pub http: reqwest::Client,
    pub stage_index: usize,
    /// Provider-format conversation history (system prompt excluded; it is
    /// re-resolved from the active stage on every turn).
    pub history: Vec<ChatMessage>,
    pub transcript: Vec<TranscriptItem>,
    pub diffs: Vec<DiffEntry>,
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
    pub quit: bool,
    turn: Option<JoinHandle<()>>,
    /// True while the running task is a compaction rather than a chat turn.
    compacting: bool,
    /// Messages submitted while a turn is running, shared with the worker,
    /// which injects them into the conversation between tool rounds.
    steer_queue: Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
    /// Everything queued during the current turn. The worker's history clone
    /// is discarded when a turn fails or is cancelled, so these are
    /// re-appended to the surviving history — same as the turn's original
    /// message, which is pushed before the worker starts.
    steered_this_turn: Vec<String>,
    tx: UnboundedSender<AgentEvent>,
    // Session persistence.
    session_id: String,
    started_at: u64,
    cwd: String,
    /// Sessions shown in the picker; loaded when it opens.
    pub session_list: Vec<Session>,
    /// Set once the user has submitted something; empty sessions aren't saved.
    has_activity: bool,
    // Prompt history (shell-style recall with Up/Down).
    prompt_history: PromptHistory,
    /// Position while browsing the prompt history; None = not browsing.
    recall_index: Option<usize>,
    /// The in-progress draft stashed when browsing starts.
    recall_draft: String,
}

fn new_textarea() -> TextArea<'static> {
    let mut input = TextArea::default();
    input.set_placeholder_text("Message soa — Enter sends, Alt+Enter newline, /help for commands");
    input.set_cursor_line_style(ratatui::style::Style::default());
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
            .timeout(std::time::Duration::from_secs(config.settings.request_timeout_secs))
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
            quit: false,
            turn: None,
            compacting: false,
            steer_queue: Arc::default(),
            steered_this_turn: Vec::new(),
            tx,
            session_id: store::new_session_id(),
            started_at: store::now_epoch(),
            cwd: store::current_cwd(),
            session_list: Vec::new(),
            has_activity: false,
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
            && let Some(index) =
                self.config.stages.iter().position(|s| s.name == session.stage)
        {
            self.stage_index = index;
            self.refresh_tool_count();
        }
        self.history = session.history;
        self.transcript = session.transcript;
        self.diffs = session.diffs;
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
        };
        if let Err(e) = store::save_session(&session) {
            self.has_activity = false; // avoid an error loop
            self.error(format!("failed to save session: {e:#}"));
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
        self.model_override.as_deref().unwrap_or(&self.stage().model)
    }

    fn stage_summary(&self) -> String {
        let stage = self.stage();
        format!(
            "stage `{}` · model `{}`{} · {} · {} tool(s)",
            stage.name,
            self.active_model(),
            if self.model_override.is_some() { " (override)" } else { "" },
            match stage.mode {
                Mode::ReadOnly => "read_only",
                Mode::ReadWrite => "read_write",
            },
            self.tool_count,
        )
    }

    fn refresh_tool_count(&mut self) {
        self.tool_count =
            assemble_tools(&self.stage().tool_profile(), &self.config, &self.mcp, 0)
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
        self.config.models.get(self.active_model()).and_then(|m| m.context_tokens)
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
                ChatMessage::System { content } | ChatMessage::User { content } => content.len(),
                ChatMessage::Assistant { content, tool_calls } => {
                    content.as_deref().map_or(0, str::len)
                        + tool_calls.as_ref().map_or(0, |calls| {
                            calls.iter().map(|c| c.function.arguments.len() + 32).sum()
                        })
                }
                ChatMessage::Tool { content, .. } => content.len(),
            })
            .sum();
        chars / 4
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
                    self.input.insert_str(text.replace("\r\n", "\n").replace('\r', "\n"));
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
        self.view = View::Sessions { selected, scroll: 0 };
    }

    fn on_sessions_key(&mut self, key: KeyEvent) {
        let View::Sessions { selected: current, .. } = self.view else { return };
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
        let Some(session) = self.session_list.get(index).cloned() else { return };
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
                    if !key.modifiers.intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
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
                if key.modifiers.intersects(KeyModifiers::ALT | KeyModifiers::SHIFT) =>
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
        let stage_names: Vec<String> =
            self.config.stages.iter().map(|s| s.name.clone()).collect();
        let model_names: Vec<String> = self.config.models.keys().cloned().collect();
        self.completion = completion::compute(
            &line,
            col,
            row == 0,
            std::path::Path::new(&self.cwd),
            &stage_names,
            &model_names,
        );
    }

    /// Whether applying the selected completion would alter the input.
    fn completion_changes_input(&self) -> bool {
        let Some(state) = &self.completion else { return false };
        let item = &state.items[state.selected];
        let (row, col) = self.input.cursor();
        let line: Vec<char> =
            self.input.lines().get(row).map(|l| l.chars().collect()).unwrap_or_default();
        let from = state.replace_from.min(line.len());
        let replaced: String = line[from..col.min(line.len())].iter().collect();
        replaced != item.insert
    }

    /// Splice the selected completion into the input, then recompute (so
    /// accepting a directory descends into it).
    fn accept_completion(&mut self) {
        let Some(state) = &self.completion else { return };
        let item = state.items[state.selected].clone();
        let (row, col) = self.input.cursor();
        let mut lines: Vec<String> = self.input.lines().to_vec();
        let Some(line) = lines.get(row).cloned() else { return };
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
        if key.code == KeyCode::Char('r')
            && let View::Diffs { selected, .. } = &self.view
        {
            let index = *selected;
            return self.restore_diff(index);
        }
        let View::Diffs { selected, scroll } = &mut self.view else { return };
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => self.view = View::Chat,
            KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                if *selected + 1 < self.diffs.len() {
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
            View::Chat if self.diffs.is_empty() => {
                self.info("no file changes captured yet");
            }
            View::Chat => {
                self.view = View::Diffs { selected: self.diffs.len() - 1, scroll: 0 };
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
            self.info("↪ queued — delivered after the current tool round");
            self.steered_this_turn.push(expanded.clone());
            self.steer_queue.lock().unwrap().push_back(expanded);
            return;
        }

        self.history.push(ChatMessage::User { content: expanded });
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
                 /clear          drop all conversation context\n\
                 /usage          cumulative token usage per model since launch\n\
                 /diff           open the diff viewer (Ctrl+G)\n\
                 /rewind         pick a past message to rewind to (conversation + files)\n\
                 /branch <name>  save the conversation as a branch and keep going\n\
                 /branches       switch between saved conversation lines (d deletes)\n\
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
                self.info(format!("context cleared (freed ~{})", fmt_tokens(freed as u64)));
            }
            "compact" => self.start_compact(),
            "usage" => {
                let lines = usage_stats::report_lines();
                if lines.is_empty() {
                    self.info("no model requests completed yet");
                    return;
                }
                let (context, _) = self.context_status();
                self.info(format!(
                    "token usage since launch:\n  {}\n  {context}",
                    lines.join("\n  "),
                ));
            }
            "diff" => self.toggle_diff_view(),
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
                let available: Vec<&str> =
                    self.config.models.keys().map(String::as_str).collect();
                self.info(format!(
                    "model: `{}`{} (stage default `{}`)\n\
                     available: {}\n\
                     usage: /model <name> · /model default",
                    self.active_model(),
                    if self.model_override.is_some() { " (override)" } else { "" },
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
                let available: Vec<&str> =
                    self.config.models.keys().map(String::as_str).collect();
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
                    self.info(format!("model override `{model}` no longer exists — cleared"));
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
            let names: Vec<&str> =
                self.config.stages.iter().map(|s| s.name.as_str()).collect();
            self.info(format!("usage: /stage <name> — available: {}", names.join(", ")));
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
        let client = match build_client(
            &self.config,
            &model_name,
            stage.temperature,
            stage.max_tokens,
            &self.http,
        ) {
            Ok(client) => client,
            Err(e) => return self.error(format!("{e:#}")),
        };
        let system = match stage.resolve_system_prompt(&self.config.base_dir).and_then(
            |system| {
                crate::skills::compose_system(
                    &config,
                    &format!("stage `{}`", stage.name),
                    system,
                    &stage.skills,
                )
            },
        ) {
            Ok(system) => system,
            Err(e) => return self.error(format!("{e:#}")),
        };
        let stage_tools =
            match assemble_tools(&stage.tool_profile(), &self.config, &self.mcp, 0) {
                Ok(tools) => tools,
                Err(e) => return self.error(format!("{e:#}")),
            };
        self.tool_count = stage_tools.len();

        let definitions: Vec<ToolFunction> =
            stage_tools.iter().map(|t| t.definition.clone()).collect();
        let bindings: HashMap<String, (ToolBinding, bool)> = stage_tools
            .into_iter()
            .map(|t| (t.definition.name, (t.binding, t.read_only)))
            .collect();
        let max_turns = stage.max_turns.unwrap_or(self.config.settings.default_max_turns);

        self.steer_queue.lock().unwrap().clear();
        self.steered_this_turn.clear();
        let worker = turn_worker(
            client,
            definitions,
            bindings,
            Arc::clone(&self.config),
            Arc::clone(&self.mcp),
            self.http.clone(),
            system,
            self.history.clone(),
            max_turns,
            self.tx.clone(),
            (stage.require_approval, stage.auto_approve.clone()),
            Arc::clone(&self.approvals),
            model_name,
            Arc::clone(&self.steer_queue),
        );
        self.compacting = false;
        self.turn = Some(tokio::spawn(worker));
    }

    fn start_compact(&mut self) {
        if self.is_running() {
            return self.error("busy — wait for the current turn to finish");
        }
        if self.history.is_empty() {
            return self.info("nothing to compact");
        }
        let stage = self.stage();
        let client = match build_client(
            &self.config,
            self.active_model(),
            stage.temperature,
            stage.max_tokens,
            &self.http,
        ) {
            Ok(client) => client,
            Err(e) => return self.error(format!("{e:#}")),
        };
        let mut request = self.history.clone();
        request.push(ChatMessage::User { content: COMPACT_INSTRUCTION.to_string() });
        let tx = self.tx.clone();
        self.compacting = true;
        self.turn = Some(tokio::spawn(async move {
            let event = match client.chat(&request, &[]).await {
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
            self.transcript.push(TranscriptItem::Assistant(partial.to_string()));
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
            self.pending_approval = None; // dropped responder reads as deny
            self.flush_stream_buffer();
            self.preserve_steered();
            self.info("cancelled");
        }
    }

    /// Restore one captured change (diff viewer `r`): put the file back
    /// into that entry's pre-change state. The reverse entry is recorded so
    /// the restore itself can be undone.
    fn restore_diff(&mut self, index: usize) {
        if self.is_running() {
            return self.error("cannot restore while a turn is running (Esc to cancel it first)");
        }
        let Some(entry) = self.diffs.get(index).cloned() else { return };
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
        self.view = View::Branches { selected: 0, scroll: 0 };
    }

    fn on_branches_key(&mut self, key: KeyEvent) {
        let View::Branches { selected, .. } = &mut self.view else { return };
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
        let Some(branch) = self.branches.get_mut(index) else { return };
        std::mem::swap(&mut branch.transcript, &mut self.transcript);
        std::mem::swap(&mut branch.history, &mut self.history);
        std::mem::swap(&mut branch.checkpoints, &mut self.checkpoints);
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
        self.info(format!("deleted branch `{}` ({})", branch.name, branch.title()));
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
        self.view = View::Rewind { selected: 0, scroll: 0 };
    }

    fn on_rewind_key(&mut self, key: KeyEvent) {
        let View::Rewind { selected, .. } = &mut self.view else { return };
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
            None => Checkpoint { transcript_index: 0, history_len: 0, diff_len: 0 },
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
            if message.is_some() { " (the message is back in the input)" } else { "" },
        ));
        self.persist();
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
            self.history.push(ChatMessage::User { content });
        }
        self.info("queued message(s) kept in context for the next turn");
    }

    pub fn abort_turn(&mut self) {
        if let Some(handle) = self.turn.take() {
            handle.abort();
        }
    }

    pub fn status_word(&self) -> &'static str {
        if self.compacting { "compacting" } else { "thinking" }
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
        );
        match event {
            AgentEvent::Delta(fragment) => {
                self.stream_buffer.push_str(&fragment);
            }
            AgentEvent::ToolCall { name, args } => {
                // Content streamed before a tool call is commentary the
                // final answer won't repeat — keep it.
                self.flush_stream_buffer();
                self.transcript.push(TranscriptItem::ToolCall { name, args });
            }
            AgentEvent::ToolDone { preview } => {
                self.transcript.push(TranscriptItem::ToolDone { preview });
            }
            AgentEvent::Diff(entry) => {
                self.info(format!("✎ {} — Ctrl+G to view", entry.title()));
                self.diffs.push(entry);
            }
            AgentEvent::Notice(text) => self.info(text),
            AgentEvent::Turn { history, text, usage } => {
                // The buffer holds the same text that just streamed; the
                // final Turn event is authoritative.
                self.stream_buffer.clear();
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
                let leftovers: Vec<String> =
                    self.steer_queue.lock().unwrap().drain(..).collect();
                self.steered_this_turn.clear();
                if leftovers.is_empty() {
                    self.maybe_auto_compact();
                } else {
                    for content in leftovers {
                        self.history.push(ChatMessage::User { content });
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
                    ChatMessage::User {
                        content: format!(
                            "[Summary of the conversation so far — earlier messages were compacted]\n\n{summary}"
                        ),
                    },
                    ChatMessage::Assistant {
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
            AgentEvent::Error(message) => {
                self.flush_stream_buffer();
                self.error(message);
                self.turn = None;
                self.compacting = false;
                self.preserve_steered();
            }
        }
        if should_save {
            self.persist();
        }
    }
}

/// One full agentic turn, run as a background task. Owns a clone of the
/// history; the updated history is handed back via [`AgentEvent::Turn`].
#[allow(clippy::too_many_arguments)]
async fn turn_worker(
    client: ChatClient,
    definitions: Vec<ToolFunction>,
    bindings: HashMap<String, (ToolBinding, bool)>,
    config: Arc<Config>,
    mcp: Arc<McpManager>,
    http: reqwest::Client,
    system: Option<String>,
    mut history: Vec<ChatMessage>,
    max_turns: u32,
    tx: UnboundedSender<AgentEvent>,
    (require_approval, auto_approve): (bool, Vec<String>),
    approvals: Arc<Approvals>,
    model_name: String,
    steer: Arc<std::sync::Mutex<std::collections::VecDeque<String>>>,
) {
    let delta_tx = tx.clone();
    let on_delta = move |fragment: &str| {
        let _ = delta_tx.send(AgentEvent::Delta(fragment.to_string()));
    };

    for _ in 0..max_turns {
        let mut request = Vec::with_capacity(history.len() + 1);
        if let Some(system) = &system {
            request.push(ChatMessage::System { content: system.clone() });
        }
        request.extend(history.iter().cloned());

        let reply = match client.chat_streamed(&request, &definitions, Some(&on_delta)).await {
            Ok(reply) => reply,
            Err(e) => {
                let _ = tx.send(AgentEvent::Error(format!("{e:#}")));
                return;
            }
        };

        if reply.tool_calls.is_empty() {
            let text = reply.content.clone().unwrap_or_default();
            history.push(ChatMessage::Assistant { content: reply.content, tool_calls: None });
            let _ = tx.send(AgentEvent::Turn { history, text, usage: reply.usage });
            return;
        }

        // Mid-turn context pressure: truncate older tool results in the
        // history before the next request.
        if let Some((used, capacity)) =
            context_pressure(&config, &model_name, reply.usage.as_ref())
        {
            let trimmed = shed_context(&mut history, 2);
            if trimmed > 0 {
                let _ = tx.send(AgentEvent::Notice(format!(
                    "context at {} of {} — truncated {trimmed} older tool result(s)",
                    fmt_tokens(used),
                    fmt_tokens(capacity),
                )));
            }
        }

        let calls = reply.tool_calls.clone();
        history.push(ChatMessage::Assistant {
            content: reply.content,
            tool_calls: Some(reply.tool_calls),
        });

        // An all-read-only round dispatches concurrently; results are
        // reported and recorded in call order either way.
        if crate::stage::parallel_round(config.settings.parallel_tools, &calls, |name| {
            bindings.get(name).map(|(_, read_only)| *read_only)
        }) {
            for call in &calls {
                let _ = tx.send(AgentEvent::ToolCall {
                    name: call.function.name.clone(),
                    args: call.function.arguments.clone(),
                });
            }
            let (config, mcp, http) = (&config, &mcp, &http);
            let outputs = futures_util::future::join_all(calls.iter().map(|call| {
                let (binding, read_only) = &bindings[&call.function.name];
                let policy =
                    CallPolicy::for_tool(require_approval, &auto_approve, &approvals, *read_only);
                async move {
                    match dispatch_tool_call(
                        binding,
                        &call.function.arguments,
                        config,
                        mcp,
                        http,
                        0,
                        &policy,
                    )
                    .await
                    {
                        Ok(output) => output,
                        Err(e) => format!("ERROR: {e:#}"),
                    }
                }
            }))
            .await;
            for (call, output) in calls.iter().zip(outputs) {
                let preview: String =
                    output.lines().next().unwrap_or("").chars().take(100).collect();
                let _ = tx.send(AgentEvent::ToolDone { preview });
                history.push(ChatMessage::Tool {
                    content: output,
                    tool_call_id: call.id.clone(),
                });
            }
            // Steering: deliver messages typed during the parallel round.
            let steered: Vec<String> = steer.lock().unwrap().drain(..).collect();
            if !steered.is_empty() {
                let _ = tx.send(AgentEvent::Notice(format!(
                    "↪ delivered {} queued message(s) to the model",
                    steered.len()
                )));
                for content in steered {
                    history.push(ChatMessage::User { content });
                }
            }
            continue;
        }

        for call in calls {
            let name = call.function.name.clone();
            let _ = tx.send(AgentEvent::ToolCall {
                name: name.clone(),
                args: call.function.arguments.clone(),
            });

            let output = match bindings.get(&name) {
                None => format!("ERROR: unknown tool `{name}`"),
                Some((binding, read_only)) => {
                    // Snapshot files a write tool might touch, for the diff viewer.
                    let snapshots = if !read_only
                        && matches!(binding, ToolBinding::Mcp { .. } | ToolBinding::File { .. })
                    {
                        diff::snapshot(&diff::extract_paths(&call.function.arguments))
                    } else {
                        Vec::new()
                    };
                    let policy = CallPolicy::for_tool(
                        require_approval,
                        &auto_approve,
                        &approvals,
                        *read_only,
                    );
                    match dispatch_tool_call(
                        binding,
                        &call.function.arguments,
                        &config,
                        &mcp,
                        &http,
                        0,
                        &policy,
                    )
                    .await
                    {
                        Ok(output) => {
                            for entry in diff::collect_changes(&name, snapshots) {
                                let _ = tx.send(AgentEvent::Diff(entry));
                            }
                            output
                        }
                        Err(e) => format!("ERROR: {e:#}"),
                    }
                }
            };

            let preview: String = output
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(100)
                .collect();
            let _ = tx.send(AgentEvent::ToolDone { preview });
            history.push(ChatMessage::Tool { content: output, tool_call_id: call.id });
        }

        // Steering: deliver messages the user typed during this round before
        // the model's next request.
        let steered: Vec<String> = steer.lock().unwrap().drain(..).collect();
        if !steered.is_empty() {
            let _ = tx.send(AgentEvent::Notice(format!(
                "↪ delivered {} queued message(s) to the model",
                steered.len()
            )));
            for content in steered {
                history.push(ChatMessage::User { content });
            }
        }
    }

    let _ = tx.send(AgentEvent::Error(format!(
        "no final answer within {max_turns} tool turns — raise max_turns on the stage"
    )));
}
