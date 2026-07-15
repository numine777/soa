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
            let wrap_width = area.width.saturating_sub(2).max(1) as usize;
            let input_layout = layout_input(app.input.lines(), app.input.cursor(), wrap_width);
            let input_height = (input_layout.rows.len().clamp(1, 8) as u16) + 2;
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
            draw_input(frame, app, input_area, input_layout);
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
        View::Rewind { .. } => {
            let [list_area, status_area] =
                Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).areas(area);
            draw_rewind(frame, app, list_area);
            draw_status(frame, app, status_area);
        }
        View::Branches { .. } => {
            let [list_area, status_area] =
                Layout::vertical([Constraint::Min(3), Constraint::Length(1)]).areas(area);
            draw_branches(frame, app, list_area);
            draw_status(frame, app, status_area);
        }
    }
}

// ---------------------------------------------------------------------------
// Branch picker
// ---------------------------------------------------------------------------

fn draw_branches(frame: &mut Frame, app: &mut App, area: Rect) {
    let rows: Vec<String> = app
        .branches
        .iter()
        .map(|branch| {
            let messages = branch
                .transcript
                .iter()
                .filter(|item| matches!(item, TranscriptItem::User(_)))
                .count();
            format!(
                "{}  {} message(s)  ends at: {}  ({})",
                branch.name,
                messages,
                truncate_str(branch.title(), 56),
                store::format_epoch(branch.created_at),
            )
        })
        .collect();
    let View::Branches { selected, scroll } = &mut app.view else { return };
    *selected = (*selected).min(rows.len().saturating_sub(1));

    let [header_area, body_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(area);
    frame.render_widget(
        Paragraph::new(
            " branches — Enter swaps the live conversation with a slot · d delete · j/k move · q close ",
        )
        .style(Style::default().fg(Color::Black).bg(Color::Magenta).add_modifier(Modifier::BOLD)),
        header_area,
    );

    let viewport = body_area.height as usize;
    if *selected < *scroll {
        *scroll = *selected;
    } else if *selected >= *scroll + viewport.max(1) {
        *scroll = *selected + 1 - viewport.max(1);
    }

    let lines: Vec<Line> = rows
        .iter()
        .enumerate()
        .skip(*scroll)
        .take(viewport.max(1))
        .map(|(row, text)| {
            let is_selected = row == *selected;
            let marker = if is_selected { "▶" } else { " " };
            let style = if is_selected {
                Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::styled(format!("{marker} {text}"), style)
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), body_area);
}

// ---------------------------------------------------------------------------
// Rewind picker
// ---------------------------------------------------------------------------

fn draw_rewind(frame: &mut Frame, app: &mut App, area: Rect) {
    // Row text per checkpoint (newest first), computed before borrowing the
    // view: the message preview and how much would be undone.
    let mut rows: Vec<String> = app
        .checkpoints
        .iter()
        .enumerate()
        .rev()
        .map(|(index, checkpoint)| {
            let preview = match app.transcript.get(checkpoint.transcript_index) {
                Some(TranscriptItem::User(text)) => text.lines().next().unwrap_or(""),
                _ => "(message unavailable)",
            };
            let undone = crate::diff::earliest_restorable_since(&app.diffs, checkpoint.diff_len)
                .len();
            format!(
                "#{}  {}{}",
                index + 1,
                truncate_str(preview, 70),
                match undone {
                    0 => String::new(),
                    n => format!("  · undoes {n} file change(s)"),
                },
            )
        })
        .collect();
    let start_undone = crate::diff::earliest_restorable_since(&app.diffs, 0).len();
    rows.push(format!(
        "⏮ session start{}",
        match start_undone {
            0 => String::new(),
            n => format!("  · undoes {n} file change(s)"),
        },
    ));

    let View::Rewind { selected, scroll } = &mut app.view else { return };
    *selected = (*selected).min(rows.len() - 1);

    let [header_area, body_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(area);
    frame.render_widget(
        Paragraph::new(
            " rewind — conversation returns to before the message, touched files are restored   Enter rewind · j/k move · q close ",
        )
        .style(Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD)),
        header_area,
    );

    let viewport = body_area.height as usize;
    if *selected < *scroll {
        *scroll = *selected;
    } else if *selected >= *scroll + viewport.max(1) {
        *scroll = *selected + 1 - viewport.max(1);
    }

    let lines: Vec<Line> = rows
        .iter()
        .enumerate()
        .skip(*scroll)
        .take(viewport.max(1))
        .map(|(row, text)| {
            let is_selected = row == *selected;
            let marker = if is_selected { "▶" } else { " " };
            let style = if is_selected {
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::styled(format!("{marker} {text}"), style)
        })
        .collect();
    frame.render_widget(Paragraph::new(lines), body_area);
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

    // Only items since the last /clear are rendered; the divider itself is
    // the first visible item, so the cut is always explained on screen.
    for item in app.visible_transcript() {
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
            TranscriptItem::Cleared(text) => {
                lines.push(Line::raw(""));
                let label = format!(" {text} ");
                let bar = |n: usize| "─".repeat(n);
                let remainder = width.saturating_sub(label.chars().count());
                lines.push(Line::styled(
                    format!("{}{label}{}", bar(remainder / 2), bar(remainder - remainder / 2)),
                    dim,
                ));
            }
        }
    }

    // The in-progress streamed reply, with a cursor mark.
    if !app.stream_buffer.is_empty() {
        lines.push(Line::raw(""));
        push_wrapped(
            &mut lines,
            &app.stream_buffer,
            width,
            "",
            "",
            Style::default(),
        );
        if lines.last().is_some_and(|line| line.width() >= width.max(16)) {
            lines.push(Line::raw("▌"));
        } else if let Some(last) = lines.last_mut() {
            last.spans.push(Span::raw("▌"));
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
    if !app.visible_diffs().is_empty() {
        spans.push(Span::styled(
            format!(" · {} diff(s) ^G", app.visible_diffs().len()),
            dim,
        ));
    }
    spans.push(Span::styled(" · PgUp/PgDn scroll · /help", dim));

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The input box's soft-wrap layout: visual rows plus the cursor's visual
/// position. tui-textarea cannot soft-wrap (long lines scroll sideways),
/// so the box renders its text itself and the TextArea stays the editing
/// engine underneath.
struct InputLayout {
    rows: Vec<String>,
    /// (visual row, x column) of the cursor within `rows`.
    cursor: (usize, u16),
}

fn layout_input(lines: &[String], cursor: (usize, usize), width: usize) -> InputLayout {
    let (cursor_row, cursor_col) = cursor;
    let mut rows = Vec::new();
    let mut cursor_visual = (0, 0);
    for (index, line) in lines.iter().enumerate() {
        let start = rows.len();
        let target = (index == cursor_row).then_some(cursor_col);
        if let Some((row, x)) = wrap_line(line, width, &mut rows, target) {
            cursor_visual = (start + row, x);
        }
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    InputLayout { rows, cursor: cursor_visual }
}

/// Append one logical line to `rows`, soft-wrapped at `width` display
/// columns (character-level breaks, wide chars counted properly). When
/// `cursor` (a char index into the line, possibly one past the end) is
/// given, returns its visual position; a cursor sitting at the edge of a
/// full row spills onto a fresh row, which is materialized so the box has
/// somewhere to draw it.
fn wrap_line(
    line: &str,
    width: usize,
    rows: &mut Vec<String>,
    cursor: Option<usize>,
) -> Option<(usize, u16)> {
    use unicode_width::UnicodeWidthChar;
    let width = width.max(1);
    let start = rows.len();
    rows.push(String::new());
    let mut x = 0usize;
    let mut placed = None;
    let mut chars = 0usize;
    for (index, ch) in line.chars().enumerate() {
        chars += 1;
        let w = ch.width().unwrap_or(0);
        if x + w > width && x > 0 {
            rows.push(String::new());
            x = 0;
        }
        if cursor == Some(index) {
            placed = Some((rows.len() - 1 - start, x as u16));
        }
        rows.last_mut().expect("pushed above").push(ch);
        x += w;
    }
    if let Some(target) = cursor
        && target >= chars
        && placed.is_none()
    {
        if x >= width {
            rows.push(String::new());
            x = 0;
        }
        placed = Some((rows.len() - 1 - start, x as u16));
    }
    placed
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect, layout: InputLayout) {
    let title = if app.is_running() {
        " drafting next message (turn in progress) "
    } else {
        " message "
    };
    let block = Block::bordered()
        .border_style(Style::default().fg(Color::DarkGray))
        .title(title);

    let empty = app.input.lines().len() == 1 && app.input.lines()[0].is_empty();
    let (cursor_row, cursor_x) = layout.cursor;
    let viewport = area.height.saturating_sub(2).max(1) as usize;
    // Keep the cursor's visual row inside the box when the text is taller
    // than the (clamped) box height.
    let scroll = (cursor_row + 1).saturating_sub(viewport);

    let paragraph = if empty {
        Paragraph::new(app.input.placeholder_text())
            .style(Style::default().fg(Color::DarkGray))
    } else {
        Paragraph::new(layout.rows.iter().map(|row| Line::from(row.as_str())).collect::<Vec<_>>())
    };
    frame.render_widget(paragraph.block(block).scroll((scroll as u16, 0)), area);
    frame.set_cursor_position((
        area.x + 1 + cursor_x.min(area.width.saturating_sub(2)),
        area.y + 1 + (cursor_row - scroll) as u16,
    ));
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
    // The viewer works on the post-/clear slice; hidden entries stay in
    // the session data and are only counted in the header.
    let hidden = app.diff_baseline.min(app.diffs.len());
    let visible_len = app.diffs.len() - hidden;
    let View::Diffs { selected, scroll } = &mut app.view else { return };
    if visible_len == 0 {
        app.view = View::Chat;
        return;
    }
    *selected = (*selected).min(visible_len - 1);
    let entry = &app.diffs[hidden + *selected];

    let [header_area, body_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(area);

    let header = format!(
        " diff {}/{}{} · {} · via {}   Tab/Shift-Tab file · j/k scroll · r restore · q close ",
        *selected + 1,
        visible_len,
        match hidden {
            0 => String::new(),
            n => format!(" (+{n} before /clear)"),
        },
        entry.title(),
        entry.provenance(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(lines: &[&str], cursor: (usize, usize), width: usize) -> InputLayout {
        let lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        layout_input(&lines, cursor, width)
    }

    #[test]
    fn wraps_long_lines_and_tracks_the_cursor() {
        // 10 chars at width 4: rows of 4/4/2; cursor at char 5 is row 1, x 1.
        let l = layout(&["abcdefghij"], (0, 5), 4);
        assert_eq!(l.rows, vec!["abcd", "efgh", "ij"]);
        assert_eq!(l.cursor, (1, 1));

        // Cursor at the very end of a partial row.
        assert_eq!(layout(&["abcdefghij"], (0, 10), 4).cursor, (2, 2));
        // Cursor at the edge of a FULL row spills onto a materialized row.
        let l = layout(&["abcdefgh"], (0, 8), 4);
        assert_eq!(l.rows, vec!["abcd", "efgh", ""]);
        assert_eq!(l.cursor, (2, 0));
    }

    #[test]
    fn counts_wide_chars_by_display_width() {
        // '你' is 2 columns: three of them at width 4 wrap after two.
        let l = layout(&["你你你"], (0, 2), 4);
        assert_eq!(l.rows, vec!["你你", "你"]);
        assert_eq!(l.cursor, (1, 0));
        // Cursor mid-row lands at the char's display column, not char index.
        assert_eq!(layout(&["你a"], (0, 1), 10).cursor, (0, 2));
    }

    #[test]
    fn multiple_logical_lines_stack_their_wrapped_rows() {
        let l = layout(&["abcdef", "x", ""], (2, 0), 4);
        assert_eq!(l.rows, vec!["abcd", "ef", "x", ""]);
        assert_eq!(l.cursor, (3, 0));

        // Empty input: one empty row, cursor at the origin.
        let l = layout(&[""], (0, 0), 4);
        assert_eq!(l.rows, vec![""]);
        assert_eq!(l.cursor, (0, 0));
    }
}
