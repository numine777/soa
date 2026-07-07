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
    Block, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};

use super::app::{App, TranscriptItem, View, fmt_tokens};

const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    match app.view {
        View::Chat => {
            let input_height = (app.input.lines().len().clamp(1, 6) as u16) + 2;
            let [transcript_area, status_area, input_area] = Layout::vertical([
                Constraint::Min(3),
                Constraint::Length(1),
                Constraint::Length(input_height),
            ])
            .areas(area);

            draw_transcript(frame, app, transcript_area);
            draw_status(frame, app, status_area);
            draw_input(frame, app, input_area);
        }
        View::Diffs { .. } => {
            let [diff_area, status_area] =
                Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).areas(area);
            draw_diffs(frame, app, diff_area);
            draw_status(frame, app, status_area);
        }
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
// Status bar and input
// ---------------------------------------------------------------------------

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let dim = Style::default().fg(Color::DarkGray);
    let stage = app.stage();

    let mut spans: Vec<Span> = Vec::new();
    if app.is_running() {
        spans.push(Span::styled(
            format!(" {} {} · Esc to cancel", SPINNER[app.spinner % SPINNER.len()], app.status_word()),
            Style::default().fg(Color::Yellow),
        ));
    } else {
        spans.push(Span::styled(" ●", Style::default().fg(Color::Green)));
    }
    spans.push(Span::styled(
        format!(
            "  {} · {} · {} tool(s) · ctx ~{}",
            stage.name,
            stage.model,
            app.tool_count,
            fmt_tokens(app.token_estimate()),
        ),
        dim,
    ));
    if !app.diffs.is_empty() {
        spans.push(Span::styled(
            format!(" · {} diff(s) ^D", app.diffs.len()),
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
        " diff {}/{} · {} · via {}   Tab/Shift-Tab file · j/k scroll · q close ",
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
