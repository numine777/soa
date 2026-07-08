//! Interactive chat TUI (`soa chat`).
//!
//! Terminal handling notes for tmux compatibility: mouse wheel scrolling
//! works through crossterm's mouse capture (pass `--no-mouse` to keep the
//! terminal's native selection behaviour), bracketed paste keeps multi-line
//! pastes from submitting early, and all key bindings avoid the kitty
//! keyboard protocol, which tmux does not pass through.

mod app;
pub mod store;
mod ui;

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::config::Config;
use crate::mcp::McpManager;
use app::App;

pub async fn run(
    config: Config,
    stage_name: Option<&str>,
    mouse: bool,
    resume: Option<&str>,
) -> Result<()> {
    let resumed = match resume {
        None => None,
        Some("latest") => {
            let latest = store::load_latest_session()?;
            if latest.is_none() {
                eprintln!("no saved sessions found — starting fresh");
            }
            latest
        }
        Some(id) => Some(store::load_session(id)?),
    };

    // Explicit --stage wins; otherwise a resumed session restores its stage.
    let stage_index = match stage_name {
        Some(name) => config
            .stages
            .iter()
            .position(|s| s.name == name)
            .with_context(|| format!("no stage named `{name}`"))?,
        None => resumed
            .as_ref()
            .and_then(|s| config.stages.iter().position(|st| st.name == s.stage))
            .unwrap_or(0),
    };

    // Connect every server any stage references, so /stage can switch freely.
    // Done before entering the alternate screen: connection can be slow and
    // is allowed to print (child stderr itself is discarded via `quiet`).
    let servers: BTreeSet<String> = config
        .stages
        .iter()
        .flat_map(|s| s.mcp.iter().cloned())
        .chain(config.agents.values().flat_map(|a| a.mcp.iter().cloned()))
        .collect();
    if !servers.is_empty() {
        eprintln!("connecting to {} MCP server(s)…", servers.len());
    }
    let mcp = Arc::new(McpManager::connect(servers, &config, true).await?);

    let (agent_tx, mut agent_rx) = tokio::sync::mpsc::unbounded_channel();
    let mut app = App::new(Arc::new(config), Arc::clone(&mcp), stage_index, agent_tx, resumed);

    setup_terminal(mouse)?;
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal(mouse);
        original_hook(info);
    }));

    let mut terminal = Terminal::new(CrosstermBackend::new(std::io::stdout()))?;

    // Blocking crossterm reads happen on a plain thread; the async loop
    // multiplexes terminal events, agent events, and the spinner tick.
    let (term_tx, mut term_rx) = tokio::sync::mpsc::unbounded_channel();
    std::thread::spawn(move || {
        while let Ok(event) = crossterm::event::read() {
            if term_tx.send(event).is_err() {
                break;
            }
        }
    });

    let mut ticker = tokio::time::interval(Duration::from_millis(120));
    let result: Result<()> = loop {
        if app.quit {
            break Ok(());
        }
        if let Err(e) = terminal.draw(|frame| ui::draw(frame, &mut app)) {
            break Err(e.into());
        }
        tokio::select! {
            event = term_rx.recv() => match event {
                Some(event) => app.on_term_event(event),
                None => break Ok(()),
            },
            event = agent_rx.recv() => {
                if let Some(event) = event {
                    app.on_agent_event(event);
                }
            }
            _ = ticker.tick(), if app.is_running() => app.spinner_tick(),
        }
    };

    app.abort_turn();
    drop(app);
    restore_terminal(mouse);

    // Best effort: if no aborted worker still holds a reference, shut the
    // MCP servers down cleanly (otherwise they exit with our stdio pipes).
    if let Ok(manager) = Arc::try_unwrap(mcp) {
        manager.shutdown().await;
    }

    result
}

fn setup_terminal(mouse: bool) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    if mouse {
        execute!(stdout, EnableMouseCapture)?;
    }
    Ok(())
}

fn restore_terminal(mouse: bool) {
    let mut stdout = std::io::stdout();
    if mouse {
        let _ = execute!(stdout, DisableMouseCapture);
    }
    let _ = execute!(stdout, DisableBracketedPaste, LeaveAlternateScreen);
    let _ = disable_raw_mode();
}
