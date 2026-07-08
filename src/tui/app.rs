//! Chat TUI state, input handling, and the background agent worker.

use std::collections::HashMap;
use std::sync::Arc;

use crossterm::event::{
    Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind,
};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;
use tui_textarea::TextArea;

use super::store::{self, PromptHistory, Session};
use crate::config::{Config, Mode, Stage};
use crate::diff::{self, DiffEntry};
use crate::mcp::McpManager;
use crate::provider::{ChatClient, ChatMessage, ToolFunction};
use crate::stage::{ToolBinding, assemble_tools, build_client, dispatch_tool_call};

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
    ToolCall { name: String, args: String },
    ToolDone { preview: String },
    Diff(DiffEntry),
    /// Turn finished: the updated history and the assistant's final text.
    Turn { history: Vec<ChatMessage>, text: String },
    Compacted { summary: String },
    Error(String),
}

pub enum View {
    Chat,
    Diffs { selected: usize, scroll: usize },
    /// Session picker over `App::session_list`.
    Sessions { selected: usize, scroll: usize },
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
    pub view: View,
    pub input: TextArea<'static>,
    /// 0 = pinned to the bottom (auto-follow); larger = scrolled up.
    pub scroll_from_bottom: usize,
    /// Viewport heights recorded during draw, used for page-scroll steps.
    pub chat_viewport: usize,
    pub diff_viewport: usize,
    pub tool_count: usize,
    pub spinner: usize,
    pub quit: bool,
    turn: Option<JoinHandle<()>>,
    /// True while the running task is a compaction rather than a chat turn.
    compacting: bool,
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
        mcp: Arc<McpManager>,
        stage_index: usize,
        tx: UnboundedSender<AgentEvent>,
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
            view: View::Chat,
            input: new_textarea(),
            scroll_from_bottom: 0,
            chat_viewport: 20,
            diff_viewport: 20,
            tool_count: 0,
            spinner: 0,
            quit: false,
            turn: None,
            compacting: false,
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
                    fmt_tokens(app.token_estimate()),
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
        self.session_id = session.id;
        self.started_at = session.started_at;
        self.has_activity = true;
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

    fn stage_summary(&self) -> String {
        let stage = self.stage();
        format!(
            "stage `{}` · model `{}` · {} · {} tool(s)",
            stage.name,
            stage.model,
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
                }
            }
            _ => {}
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);

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
            fmt_tokens(self.token_estimate()),
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
        self.session_id = store::new_session_id();
        self.started_at = store::now_epoch();
        self.has_activity = false;
        self.scroll_from_bottom = 0;
        self.info(format!(
            "started new session {} — {}. Type /help for commands.",
            self.session_id,
            self.stage_summary(),
        ));
    }

    fn on_chat_key(&mut self, key: KeyEvent) {
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
            View::Sessions { .. } => {} // don't stack modal views
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
        if self.is_running() {
            return; // keep the draft; a turn is already in flight
        }
        self.remember_prompt(&text);
        self.has_activity = true;
        self.input = new_textarea();
        self.transcript.push(TranscriptItem::User(text.clone()));
        self.history.push(ChatMessage::User { content: text });
        self.scroll_from_bottom = 0;
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
                 /diff           open the diff viewer (Ctrl+G)\n\
                 /stage <name>   switch the active stage\n\
                 /sessions       open the session picker (switch or start new)\n\
                 /quit           exit (Ctrl+D on an empty prompt)\n\
                 keys: Enter send · Alt+Enter newline · Up/Down recall past prompts\n\
                 PgUp/PgDn + mouse wheel scroll\n\
                 diff view: Tab/Shift+Tab switch file · j/k scroll · q close",
            ),
            "sessions" => self.open_sessions(),
            "clear" => {
                if self.is_running() {
                    self.error("cannot clear while a turn is running (Esc to cancel it first)");
                    return;
                }
                let freed = self.token_estimate();
                self.history.clear();
                self.info(format!("context cleared (freed ~{})", fmt_tokens(freed)));
            }
            "compact" => self.start_compact(),
            "diff" => self.toggle_diff_view(),
            "stage" => self.switch_stage(arg),
            "quit" | "exit" => self.quit = true,
            other => self.error(format!("unknown command `/{other}` — try /help")),
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
        let stage = &config.stages[self.stage_index];
        let client = match build_client(
            &self.config,
            &stage.model,
            stage.temperature,
            stage.max_tokens,
            &self.http,
        ) {
            Ok(client) => client,
            Err(e) => return self.error(format!("{e:#}")),
        };
        let system = match stage.resolve_system_prompt(&self.config.base_dir) {
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
            &stage.model,
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

    pub fn cancel_turn(&mut self) {
        if let Some(handle) = self.turn.take() {
            handle.abort();
            self.compacting = false;
            self.info("cancelled");
        }
    }

    pub fn abort_turn(&mut self) {
        if let Some(handle) = self.turn.take() {
            handle.abort();
        }
    }

    pub fn status_word(&self) -> &'static str {
        if self.compacting { "compacting" } else { "thinking" }
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
            AgentEvent::ToolCall { name, args } => {
                self.transcript.push(TranscriptItem::ToolCall { name, args });
            }
            AgentEvent::ToolDone { preview } => {
                self.transcript.push(TranscriptItem::ToolDone { preview });
            }
            AgentEvent::Diff(entry) => {
                self.info(format!("✎ {} — Ctrl+G to view", entry.title()));
                self.diffs.push(entry);
            }
            AgentEvent::Turn { history, text } => {
                self.history = history;
                if text.trim().is_empty() {
                    self.info("(model returned an empty response)");
                } else {
                    self.transcript.push(TranscriptItem::Assistant(text));
                }
                self.turn = None;
            }
            AgentEvent::Compacted { summary } => {
                let before = self.token_estimate();
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
                self.info(format!(
                    "context compacted: ~{} → ~{}",
                    fmt_tokens(before),
                    fmt_tokens(after)
                ));
            }
            AgentEvent::Error(message) => {
                self.error(message);
                self.turn = None;
                self.compacting = false;
            }
        }
        if should_save {
            self.persist();
        }
    }
}

pub fn fmt_tokens(tokens: usize) -> String {
    if tokens >= 1000 {
        format!("{:.1}k tok", tokens as f64 / 1000.0)
    } else {
        format!("{tokens} tok")
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
) {
    for _ in 0..max_turns {
        let mut request = Vec::with_capacity(history.len() + 1);
        if let Some(system) = &system {
            request.push(ChatMessage::System { content: system.clone() });
        }
        request.extend(history.iter().cloned());

        let reply = match client.chat(&request, &definitions).await {
            Ok(reply) => reply,
            Err(e) => {
                let _ = tx.send(AgentEvent::Error(format!("{e:#}")));
                return;
            }
        };

        if reply.tool_calls.is_empty() {
            let text = reply.content.clone().unwrap_or_default();
            history.push(ChatMessage::Assistant { content: reply.content, tool_calls: None });
            let _ = tx.send(AgentEvent::Turn { history, text });
            return;
        }

        let calls = reply.tool_calls.clone();
        history.push(ChatMessage::Assistant {
            content: reply.content,
            tool_calls: Some(reply.tool_calls),
        });

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
                    let snapshots = if !read_only && matches!(binding, ToolBinding::Mcp { .. }) {
                        diff::snapshot(&diff::extract_paths(&call.function.arguments))
                    } else {
                        Vec::new()
                    };
                    match dispatch_tool_call(
                        binding,
                        &call.function.arguments,
                        &config,
                        &mcp,
                        &http,
                        0,
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
    }

    let _ = tx.send(AgentEvent::Error(format!(
        "no final answer within {max_turns} tool turns — raise max_turns on the stage"
    )));
}
