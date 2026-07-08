//! Rendering: transcript with wrap-aware scrolling, status bar, input box,
//! and the diff viewer.
//!
//! Transcript lines are pre-wrapped to the pane width, so scroll offsets are
//! exact — no drift between the scrollbar, PgUp/PgDn steps, and what is on
//! screen.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};

use super::app::{App, TranscriptItem, View};
use super::store;

const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    match app.view {
        View::Chat => {
            let input_height = (app.input.lines().len().clamp(1, 6) as u16) + 2;
            let approval_height = if app.pending_approval.is_some() { 2 } else { 0 };
            let [transcript_area, approval_area, status_area, input_area] = Layout::vertical([
                Constraint::Min(3),
                Constraint::Length(approval_height),
                Constraint::Length(1),
                Constraint::Length(input_height),
            ])
            .areas(area);

            draw_transcript(frame, app, transcript_area);
            if app.pending_approval.is_some() {
                draw_approval(frame, app, approval_area);
            }
            draw_status(frame, app, status_area);
            draw_input(frame, app, input_area);
            if app.pending_approval.is_none() {
                draw_completion(frame, app, input_area);
            }
        }
        View::Diffs { .. } => {
            let [diff_area, status_area] =
                Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).areas(area);
            draw_diffs(frame, app, diff_area);
            draw_status(frame, app, status_area);
        }
        View::Sessions { .. } => {
            let [list_area, status_area] =
                Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).areas(area);
            draw_sessions(frame, app, list_area);
            draw_status(frame, app, status_area);
        }
    }
}

// ---------------------------------------------------------------------------
// Session picker
// ---------------------------------------------------------------------------

fn draw_sessions(frame: &mut Frame, app: &mut App, area: Rect) {
    let current_id = app.session_id().to_string();
    let View::Sessions { selected, scroll } = &mut app.view else { return };
    // Row 0 is the "start new session" entry; sessions follow.
    let total_rows = app.session_list.len() + 1;
    *selected = (*selected).min(total_rows - 1);

    let [header_area, body_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(area);

    let header = format!(
        " sessions ({})   Enter select · j/k move · q close ",
        store::current_cwd(),
    );
    frame.render_widget(
        Paragraph::new(header).style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        header_area,
    );

    // Keep the selection inside the window.
    let viewport = body_area.height as usize;
    if *selected < *scroll {
        *scroll = *selected;
    } else if *selected >= *scroll + viewport.max(1) {
        *scroll = *selected + 1 - viewport.max(1);
    }

    let lines: Vec<Line> = (*scroll..total_rows.min(*scroll + viewport.max(1)))
        .map(|row| {
            let is_selected = row == *selected;
            let marker = if is_selected { "▶" } else { " " };
            let (text, is_current) = match row {
                0 => (format!("{marker} + start new session"), false),
                _ => {
                    let session = &app.session_list[row - 1];
                    let is_current = session.id == current_id;
                    (
                        format!(
                            "{} {}  {}  [{}]  {}{}",
                            marker,
                            session.id,
                            store::format_epoch(session.updated_at),
                            session.stage,
                            truncate_str(&session.title, 60),
                            if is_current { "  (current)" } else { "" },
                        ),
                        is_current,
                    )
                }
            };
            let style = match (is_selected, is_current) {
                (true, _) => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                (false, true) => Style::default().fg(Color::Green),
                (false, false) => Style::default(),
            };
            Line::styled(text, style)
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), body_area);
}

fn truncate_str(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let cut: String = text.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

// ---------------------------------------------------------------------------
// Transcript
// ---------------------------------------------------------------------------

fn draw_transcript(frame: &mut Frame, app: &mut App, area: Rect) {
    let inner = Rect {
        x: area.x + 1,
        y: area.y,
        width: area.width.saturating_sub(3), // left pad + scrollbar gutter
        height: area.height,
    };
    let wrap_width = inner.width.max(16) as usize;
    let lines = build_transcript_lines(app, wrap_width);

    let total = lines.len();
    let viewport = inner.height as usize;
    app.chat_viewport = viewport;

    let max_scroll = total.saturating_sub(viewport);
    app.scroll_from_bottom = app.scroll_from_bottom.min(max_scroll);

    let start = total.saturating_sub(viewport + app.scroll_from_bottom);
    let end = (start + viewport).min(total);
    let visible: Vec<Line> = lines[start..end].to_vec();
    frame.render_widget(Paragraph::new(visible), inner);

    if max_scroll > 0 {
        let mut state = ScrollbarState::new(max_scroll)
            .position(max_scroll - app.scroll_from_bottom)
            .viewport_content_length(viewport);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            area,
            &mut state,
        );
    }
}

fn build_transcript_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let dim = Style::default().fg(Color::DarkGray);
    let mut lines = Vec::new();

    for item in &app.transcript {
        match item {
            TranscriptItem::User(text) => {
                lines.push(Line::raw(""));
                push_wrapped(
                    &mut lines,
                    text,
                    width,
                    "❯ ",
                    "  ",
                    Style::default().fg(Color::Cyan),
                );
            }
            TranscriptItem::Assistant(text) => {
                lines.push(Line::raw(""));
                push_wrapped(&mut lines, text, width, "", "", Style::default());
            }
            TranscriptItem::ToolCall { name, args } => {
                let args_line = squash(args, 120);
                push_wrapped(
                    &mut lines,
                    &format!("{name} {args_line}"),
                    width,
                    "  ⚙ ",
                    "      ",
                    dim,
                );
            }
            TranscriptItem::ToolDone { preview } => {
                push_wrapped(&mut lines, &squash(preview, 120), width, "    ↳ ", "      ", dim);
            }
            TranscriptItem::Info(text) => {
                push_wrapped(
                    &mut lines,
                    text,
                    width,
                    "• ",
                    "  ",
                    Style::default().fg(Color::Yellow),
                );
            }
            TranscriptItem::Error(text) => {
                push_wrapped(
                    &mut lines,
                    text,
                    width,
                    "✗ ",
                    "  ",
                    Style::default().fg(Color::Red),
                );
            }
        }
    }

    // The in-progress streamed reply, with a cursor mark.
    if !app.stream_buffer.is_empty() {
        lines.push(Line::raw(""));
        let live = format!("{}▌", app.stream_buffer);
        push_wrapped(&mut lines, &live, width, "", "", Style::default());
    }

    lines
}

/// Wrap `text` to `width`, applying `first_prefix` to the very first line and
/// `cont_prefix` to every other physical line, all in one style.
fn push_wrapped(
    lines: &mut Vec<Line<'static>>,
    text: &str,
    width: usize,
    first_prefix: &str,
    cont_prefix: &str,
    style: Style,
) {
    let width = width.max(16);
    let mut first = true;
    for raw in text.split('\n') {
        let initial = if first { first_prefix } else { cont_prefix };
        if raw.trim().is_empty() {
            lines.push(Line::styled(initial.trim_end().to_string(), style));
        } else {
            let options = textwrap::Options::new(width)
                .initial_indent(initial)
                .subsequent_indent(cont_prefix);
            for wrapped in textwrap::wrap(raw, options) {
                lines.push(Line::styled(wrapped.into_owned(), style));
            }
        }
        first = false;
    }
}

/// Collapse to a single line and cap the length, for previews.
fn squash(text: &str, max_chars: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        collapsed
    } else {
        let cut: String = collapsed.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

// ---------------------------------------------------------------------------
// Approval modal
// ---------------------------------------------------------------------------

fn draw_approval(frame: &mut Frame, app: &App, area: Rect) {
    let Some(request) = &app.pending_approval else { return };
    let style = Style::default().fg(Color::Black).bg(Color::Yellow);
    let detail = if request.detail.is_empty() || request.detail == request.descriptor {
        String::new()
    } else {
        format!("   {}", squash(&request.detail, 100))
    };
    let lines = vec![
        Line::styled(format!(" ⚠ approve: {}{detail}", request.descriptor), style),
        Line::styled(
            format!(
                "   [y] once · [a] always this session ({}) · [n]/Esc deny",
                request.always_pattern
            ),
            style.add_modifier(Modifier::BOLD),
        ),
    ];
    frame.render_widget(Paragraph::new(lines), area);
}

// ---------------------------------------------------------------------------
// Status bar and input
// ---------------------------------------------------------------------------

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let dim = Style::default().fg(Color::DarkGray);
    let stage = app.stage();

    let mut spans: Vec<Span> = Vec::new();
    if app.is_running() {
        let queued = match app.queued_count() {
            0 => String::new(),
            n => format!(" · {n} queued"),
        };
        spans.push(Span::styled(
            format!(
                " {} {}{queued} · Esc to cancel",
                SPINNER[app.spinner % SPINNER.len()],
                app.status_word()
            ),
            Style::default().fg(Color::Yellow),
        ));
    } else {
        spans.push(Span::styled(" ●", Style::default().fg(Color::Green)));
    }
    spans.push(Span::styled(
        format!("  {} · {} · {} tool(s) · ", stage.name, app.active_model(), app.tool_count),
        dim,
    ));
    let (context_text, pressure) = app.context_status();
    spans.push(Span::styled(
        context_text,
        match pressure {
            2 => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            1 => Style::default().fg(Color::Yellow),
            _ => dim,
        },
    ));
    if !app.diffs.is_empty() {
        spans.push(Span::styled(
            format!(" · {} diff(s) ^G", app.diffs.len()),
            dim,
        ));
    }
    spans.push(Span::styled(" · PgUp/PgDn scroll · /help", dim));

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_input(frame: &mut Frame, app: &mut App, area: Rect) {
    let title = if app.is_running() {
        " drafting next message (turn in progress) "
    } else {
        " message "
    };
    app.input.set_block(
        Block::bordered()
            .border_style(Style::default().fg(Color::DarkGray))
            .title(title),
    );
    frame.render_widget(&app.input, area);
}

/// Autocomplete popup, floating just above the input box. Tab or Enter
/// accepts, Up/Down navigate, Esc closes.
fn draw_completion(frame: &mut Frame, app: &App, input_area: Rect) {
    let Some(completion) = &app.completion else { return };
    let label_width =
        completion.items.iter().map(|i| i.label.chars().count()).max().unwrap_or(0);
    let line_width = completion
        .items
        .iter()
        .map(|i| {
            label_width + if i.detail.is_empty() { 0 } else { 2 + i.detail.chars().count() }
        })
        .max()
        .unwrap_or(10);

    let height = (completion.items.len() as u16).min(input_area.y);
    let width = (line_width as u16 + 2).min(input_area.width.saturating_sub(2));
    let popup = Rect {
        x: input_area.x + 1,
        y: input_area.y.saturating_sub(height),
        width,
        height,
    };
    frame.render_widget(Clear, popup);

    let lines: Vec<Line> = completion
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let (item_style, detail_style) = if index == completion.selected {
                let selected = Style::default().bg(Color::DarkGray);
                (selected.add_modifier(Modifier::BOLD), selected.fg(Color::Gray))
            } else {
                (Style::default(), Style::default().fg(Color::DarkGray))
            };
            let mut spans =
                vec![Span::styled(format!(" {:label_width$}", item.label), item_style)];
            if !item.detail.is_empty() {
                spans.push(Span::styled(format!("  {}", item.detail), detail_style));
            }
            spans.push(Span::styled(" ", item_style));
            Line::from(spans)
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), popup);
}

// ---------------------------------------------------------------------------
// Diff viewer
// ---------------------------------------------------------------------------

fn draw_diffs(frame: &mut Frame, app: &mut App, area: Rect) {
    let View::Diffs { selected, scroll } = &mut app.view else { return };
    if app.diffs.is_empty() {
        app.view = View::Chat;
        return;
    }
    *selected = (*selected).min(app.diffs.len() - 1);
    let entry = &app.diffs[*selected];

    let [header_area, body_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(area);

    let header = format!(
        " diff {}/{} · {} · via {}   Tab/Shift-Tab file · j/k scroll · r restore · q close ",
        *selected + 1,
        app.diffs.len(),
        entry.title(),
        entry.tool,
    );
    frame.render_widget(
        Paragraph::new(header).style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        header_area,
    );

    let lines = style_unified_diff(&entry.unified);
    let total = lines.len();
    let viewport = body_area.height as usize;
    app.diff_viewport = viewport;

    let max_scroll = total.saturating_sub(viewport);
    *scroll = (*scroll).min(max_scroll);

    let inner = Rect {
        x: body_area.x + 1,
        y: body_area.y,
        width: body_area.width.saturating_sub(3),
        height: body_area.height,
    };
    // Long diff lines are clipped horizontally rather than wrapped, so line
    // numbers in hunks stay aligned; scrolling is vertical only.
    frame.render_widget(
        Paragraph::new(lines).scroll((*scroll as u16, 0)),
        inner,
    );

    if max_scroll > 0 {
        let mut state = ScrollbarState::new(max_scroll)
            .position(*scroll)
            .viewport_content_length(viewport);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            body_area,
            &mut state,
        );
    }
}

fn style_unified_diff(unified: &str) -> Vec<Line<'static>> {
    unified
        .lines()
        .map(|line| {
            let style = if line.starts_with("+++") || line.starts_with("---") {
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)
            } else if line.starts_with("@@") {
                Style::default().fg(Color::Cyan)
            } else if line.starts_with('+') {
                Style::default().fg(Color::Green)
            } else if line.starts_with('-') {
                Style::default().fg(Color::Red)
            } else {
                Style::default()
            };
            Line::styled(line.to_string(), style)
        })
        .collect()
}
