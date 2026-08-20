//! Basic TUI panels (Step 4): header, inference summary, connection and
//! error views, waiting view, and footer.
//!
//! `Frame::area()` is the single source of truth for the drawable region —
//! `AppState::terminal_size` only records the last resize event. Every
//! helper is safe for degenerate sizes (no panic, no UTF-8 mid-character
//! slicing) and never fabricates a value that the backend did not report.

use std::time::{Instant, SystemTime};

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, Paragraph, Row};
use ratatui::Frame;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::history::HistorySample;
use crate::app::log::{EventRecord, EventSeverity};
use crate::app::state::AppState;
use crate::display::Symbols;
use crate::domain::{
    BackendSnapshot, Confidence, ConnectionState, ServerState, SlotSnapshot, SystemSnapshot,
};

/// Minimum size at which the full panel layout is rendered.
pub const MIN_WIDTH: u16 = 80;
pub const MIN_HEIGHT: u16 = 20;

/// Render the full TUI frame. Safe for any size, including 0x0.
pub fn render(f: &mut Frame, state: &AppState, symbols: &Symbols) {
    let area = f.area();
    if area.width == 0 || area.height == 0 {
        return;
    }
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        render_too_small(f, area);
        return;
    }

    let snap = state.visible_snapshot();
    match snap {
        // No snapshot yet: never treat "first data pending" as a
        // disconnection — it has its own neutral view.
        None => {
            let chunks = Layout::vertical([
                Constraint::Length(6),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .split(area);
            render_waiting(f, chunks[0], &state.endpoint, symbols);
            render_footer(f, chunks[2], state, symbols);
        }
        Some(s) => {
            match (s.connection, s.server) {
                // Non-connected states get a full connection view instead of
                // the summary panels (the data would be all placeholders).
                (ConnectionState::Connected, ServerState::Loading)
                | (ConnectionState::Connected, ServerState::Sleeping) => {
                    let chunks = Layout::vertical([
                        Constraint::Length(5),
                        Constraint::Min(1),
                        Constraint::Length(1),
                    ])
                    .split(area);
                    render_header(f, chunks[0], state, s, symbols);
                    match s.server {
                        ServerState::Loading => {
                            render_lifecycle(f, chunks[1], "LOADING", "Model is loading...")
                        }
                        ServerState::Sleeping => {
                            render_lifecycle(f, chunks[1], "SLEEPING", "The server is sleeping.")
                        }
                        _ => {}
                    }
                    render_footer(f, chunks[2], state, symbols);
                }
                // Header: border + 3 content lines. Inference: border + 4 lines.
                _ => {
                    let chunks = Layout::vertical([
                        Constraint::Length(5),
                        Constraint::Min(1),
                        Constraint::Length(1),
                    ])
                    .split(area);
                    render_header(f, chunks[0], state, s, symbols);
                    match s.connection {
                        ConnectionState::Connected => {
                            // Middle area: inference (fixed), the slot table
                            // (variable), the detail panel (history, or the
                            // event log when toggled), the Resources panel
                            // (host + llama-server process, when enabled),
                            // and the status/warning area.
                            let status_count = status_lines(state, symbols).len();
                            let free = chunks[1]
                                .height
                                .saturating_sub(6)
                                .saturating_sub(status_count as u16);
                            // Resources is a compact 4-row block (2 borders +
                            // 2 content lines) with the LOWEST layout
                            // priority. At 80x20 (free=8) and 100x30 (free=18)
                            // it is hidden so the slot table keeps its full
                            // rows and the history keeps its full layout; it
                            // appears only when the terminal is tall enough
                            // (free >= 19) that the full 9-row history still
                            // fits alongside it.
                            let resources_h = if state.show_system && free >= 19 { 4 } else { 0 };
                            let detail_budget = free.saturating_sub(resources_h);
                            // The remaining space is shared by the detail
                            // panel. History tiers as space allows (block
                            // height = inner content + 2 borders):
                            // 9 = full (legend + 2-row bars + sparklines),
                            // 6 = one row per series, 4 = two series rows,
                            // 3 = one summary row, 0 = hidden.
                            let history_h = match detail_budget {
                                f if f >= 15 => 9,
                                f if f >= 12 => 6,
                                f if f >= 10 => 4,
                                f if f >= 9 => 3,
                                _ => 0,
                            };
                            // The detail panel shows either the history
                            // (default) or the event log (toggled with `l`).
                            // The event log is explicitly requested, so it
                            // takes the free space but reserves room for the
                            // slot table to keep its header + one row.
                            let detail_h = if state.show_events {
                                detail_budget.saturating_sub(4)
                            } else {
                                history_h
                            };
                            let mid = Layout::vertical([
                                Constraint::Length(6),
                                Constraint::Min(1),
                                Constraint::Length(detail_h),
                                Constraint::Length(resources_h),
                                Constraint::Length(status_count as u16),
                            ])
                            .split(chunks[1]);
                            render_inference(f, mid[0], s, symbols);
                            render_slots(f, mid[1], state, symbols);
                            if detail_h > 0 {
                                if state.show_events {
                                    render_events(f, mid[2], state, symbols);
                                } else {
                                    render_history(f, mid[2], state, symbols);
                                }
                            }
                            if resources_h > 0 {
                                render_resources(f, mid[3], state, symbols);
                            }
                            render_status(f, mid[4], state, symbols);
                        }
                        _ => render_connection_view(f, chunks[1], state, s, symbols),
                    }
                    render_footer(f, chunks[2], state, symbols);
                }
            }
        }
    }
}

/// "Terminal is too small" fallback.
fn render_too_small(f: &mut Frame, area: Rect) {
    let text = Paragraph::new(format!(
        "Terminal is too small.\n\nRequired: {} x {}\nCurrent: {} x {}",
        MIN_WIDTH, MIN_HEIGHT, area.width, area.height
    ));
    f.render_widget(text, area);
}

/// Header block: connection, backend, model, server state, phase, counts.
fn render_header(
    f: &mut Frame,
    area: Rect,
    state: &AppState,
    snap: &BackendSnapshot,
    symbols: &Symbols,
) {
    let (conn_symbol, conn_label) = connection_display(state, snap, symbols);
    let block = Block::bordered().title(" LLAMATOP ");
    let inner = block.inner(area);
    if inner.width == 0 || inner.height < 3 {
        return;
    }

    // Line 1: "<sym> <CONNECTION>   llama.cpp   <model>" — the model gets
    // whatever width is left so long names truncate instead of overflowing.
    let prefix = format!("{} {}   {}   ", conn_symbol, conn_label, backend_name());
    let model_budget = (inner.width as usize).saturating_sub(prefix.width() + 1);
    let model = snap
        .model_name
        .as_deref()
        .map(|m| trunc(m, model_budget, symbols.is_ascii()))
        .unwrap_or_else(|| placeholder(symbols).to_string());

    let line1 = format!("{}{}", prefix, model);
    let confidence = confidence_suffix(snap.workload_confidence);
    let phase = format!("{}{}", snap.workload_phase.display(), confidence);
    let line2 = format!(
        "Server: {}   Phase: {}{}",
        snap.server.as_str(),
        phase,
        if state.paused { "   PAUSED" } else { "" }
    );
    let line3 = format!(
        "Active: {}   Queued: {}   Updated: {}",
        format_opt_u64(snap.active_requests, symbols),
        format_opt_u64(snap.queued_requests, symbols),
        update_age(state, symbols),
    );

    let text = Paragraph::new(format!("{line1}\n{line2}\n{line3}"));
    f.render_widget(block, area);
    f.render_widget(text, inner);
}

fn render_inference(f: &mut Frame, area: Rect, snap: &BackendSnapshot, symbols: &Symbols) {
    let block = Block::bordered().title(" Inference ");
    let inner = block.inner(area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    f.render_widget(block, area);

    let prompt =
        rate_value(snap.prompt_tokens_per_second, snap.prompt_tokens_per_second_reported, symbols);
    let generation = rate_value(
        snap.generation_tokens_per_second,
        snap.generation_tokens_per_second_reported,
        symbols,
    );
    let active = format_opt_u64(snap.active_requests, symbols);
    let queued = format_opt_u64(snap.queued_requests, symbols);
    let context = format_context(snap.context_max_tokens, symbols);
    let slots = format_opt_u64(snap.total_slots, symbols);
    let spec = format_percent(snap.speculative.acceptance_rate(), symbols);

    let rows = [
        ("Prompt", prompt.as_str(), "Generation", generation.as_str()),
        ("Active", active.as_str(), "Queued", queued.as_str()),
        ("Context", context.as_str(), "Slots", slots.as_str()),
        ("Spec accept", spec.as_str(), "", ""),
    ];

    // Two columns per row: "label(value)   label(value)" with a 3-char gap.
    let field = (inner.width as usize).saturating_sub(3) / 2 - 13;
    let lines: Vec<String> = rows
        .iter()
        .map(|(l1, v1, l2, v2)| {
            if l2.is_empty() {
                format!("{: <13}{}", l1, v1)
            } else {
                format!("{: <13}{: >width$}   {: <13}{: >width$}", l1, v1, l2, v2, width = field)
            }
        })
        .collect();

    let text = Paragraph::new(lines.join("\n"));
    f.render_widget(text, inner);
}

/// The status/warning lines, shared by the renderer (to size the area) and
/// by `render_status` (to draw them).
fn status_lines(state: &AppState, symbols: &Symbols) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    if !state.capabilities.metrics {
        lines.push(format!("{} Metrics unavailable", symbols.warning()));
        lines.push("Throughput values cannot be displayed.".into());
        lines.push("Start llama-server with --metrics.".into());
    }
    if let Some(msg) = &state.connection_message {
        if !lines.is_empty() {
            lines.push(String::new());
        }
        lines.push(format!("{} {}", symbols.error(), trunc(msg, 200, symbols.is_ascii())));
    }
    lines
}

/// Status/warning area: capability warnings and the last error message.
/// "Slots unavailable" is shown by the slot panel itself, not here.
fn render_status(f: &mut Frame, area: Rect, state: &AppState, symbols: &Symbols) {
    let lines = status_lines(state, symbols);
    if lines.is_empty() || area.height == 0 {
        return;
    }
    let text = Paragraph::new(lines.join("\n"));
    f.render_widget(text, area);
}

/// History panel: sparkline rows for prompt/generation throughput and
/// active/queued requests.
///
/// Missing samples are gaps (blank), never zeros: a `0.0` sample renders
/// the lowest glyph of the ramp, a missing sample renders nothing. Series
/// are distinguished by their text labels (and color), never by color alone.
fn render_history(f: &mut Frame, area: Rect, state: &AppState, symbols: &Symbols) {
    let block = Block::bordered().title(" History ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let samples = state.history.samples();
    if samples.is_empty() {
        let text = Paragraph::new("No recent data");
        f.render_widget(text, inner);
        return;
    }

    // Label column: "Prompt" fits in 7 + 1 gap.
    let label_w = 8usize.min(inner.width as usize);
    let plot_w = (inner.width as usize).saturating_sub(label_w);
    if plot_w == 0 {
        return;
    }

    // Take the newest samples that fit one column each.
    let count = plot_w.min(samples.len());
    let start = samples.len() - count;
    let window: Vec<&HistorySample> = samples.iter().skip(start).collect();

    let prompt: Vec<Option<f64>> = window.iter().map(|s| s.prompt_tokens_per_second).collect();
    let generation: Vec<Option<f64>> =
        window.iter().map(|s| s.generation_tokens_per_second).collect();
    let active: Vec<Option<f64>> =
        window.iter().map(|s| s.active_requests.map(|v| v as f64)).collect();
    let queued: Vec<Option<f64>> =
        window.iter().map(|s| s.queued_requests.map(|v| v as f64)).collect();

    let ascii = symbols.is_ascii();
    let indent = " ".repeat(label_w);
    let mut lines: Vec<Line> = Vec::new();

    match inner.height {
        // Full (exactly 7 rows): legend + two 2-row bar series + two
        // 1-row sparklines. Labels sit on the first row of each series.
        h if h >= 7 => {
            let legend = format!(
                "P {}   G {}   A {}   Q {}",
                latest_rate(&prompt, symbols),
                latest_rate(&generation, symbols),
                latest_int(&active, symbols),
                latest_int(&queued, symbols)
            );
            lines.push(Line::from(legend));
            let (p_top, p_bot) = bar_rows(&prompt, ascii);
            lines.push(Line::from(format!("Prompt  {}", p_top)));
            lines.push(Line::from(format!("{indent}{}", p_bot)));
            let (g_top, g_bot) = bar_rows(&generation, ascii);
            lines.push(Line::from(format!("Gen     {}", g_top)));
            lines.push(Line::from(format!("{indent}{}", g_bot)));
            lines.push(Line::from(format!("Active  {}", sparkline(&active, ascii))));
            lines.push(Line::from(format!("Queued  {}", sparkline(&queued, ascii))));
        }
        // One row per series (no legend; labels carry the identity).
        4 => {
            lines.push(Line::from(format!("Prompt  {}", sparkline(&prompt, ascii))));
            lines.push(Line::from(format!("Gen     {}", sparkline(&generation, ascii))));
            lines.push(Line::from(format!("Active  {}", sparkline(&active, ascii))));
            lines.push(Line::from(format!("Queued  {}", sparkline(&queued, ascii))));
        }
        2 => {
            lines.push(Line::from(format!("Prompt  {}", sparkline(&prompt, ascii))));
            lines.push(Line::from(format!("Gen     {}", sparkline(&generation, ascii))));
        }
        // Single summary row: latest values only.
        _ => {
            let text = format!(
                "P {}  G {}  A {}  Q {}",
                latest_rate(&prompt, symbols),
                latest_rate(&generation, symbols),
                latest_int(&active, symbols),
                latest_int(&queued, symbols)
            );
            lines.push(Line::from(text));
        }
    }

    let text = Paragraph::new(lines);
    f.render_widget(text, inner);
}

/// Event log panel: bounded, redacted state-transition and user-action
/// events, newest at the bottom by default. Repeated identical events show
/// a `xN` suffix instead of repeating. `state.event_scroll` is the number
/// of records hidden below the viewport (0 = newest visible); the visible
/// window is derived each frame so a growing/shrinking log stays clamped.
fn render_events(f: &mut Frame, area: Rect, state: &AppState, symbols: &Symbols) {
    let title = if state.show_events {
        " Events  (l hide, c clear, PgUp/PgDn scroll) "
    } else {
        " Events "
    };
    let block = Block::bordered().title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let records = state.events.records();
    if records.is_empty() {
        let text = Paragraph::new("No events yet");
        f.render_widget(text, inner);
        return;
    }

    // Newest at the bottom. `state.event_scroll` is how many lines the view
    // has scrolled up from the newest: the bottom row aligns with the record
    // `offset` steps above the newest (offset 0 = newest at the bottom). The
    // window is derived each frame so a growing/shrinking log stays clamped.
    let len = records.len();
    let offset = state.event_scroll.min(len.saturating_sub(1));
    let end = len - offset; // exclusive end in oldest-first order
    let count = end.min(inner.height as usize);
    let start = end - count;
    let window: Vec<&EventRecord> = records.iter().skip(start).take(count).collect();

    let time_w = 9usize; // "123.4 s  " style width
    let sym_w = symbols.warning().width().max(symbols.error().width());
    let label_w = (time_w + 1 + sym_w + 1 + 4).min(inner.width as usize); // + "xNNN "
    let msg_w = (inner.width as usize).saturating_sub(label_w);

    let mut lines: Vec<Line> = Vec::with_capacity(window.len());
    // `window` is oldest-first, so the newest visible record lands on the
    // last (bottom) row of the panel.
    for rec in &window {
        let age = event_age(rec.timestamp);
        let sym = severity_symbol(rec.severity, symbols);
        let repeat =
            if rec.repeat_count > 1 { format!(" x{}", rec.repeat_count) } else { String::new() };
        let prefix = format!("{age:>9}  {sym:<w$}  {repeat} ", w = sym_w);
        let msg = trunc(&rec.message, msg_w.max(1), symbols.is_ascii());
        lines.push(Line::from(format!("{prefix}{msg}")));
    }

    let text = Paragraph::new(lines);
    f.render_widget(text, inner);
}

/// Severity symbol. The symbol is always paired with a text context (the
/// message), never color-only, so the severity is readable without color.
fn severity_symbol(severity: EventSeverity, symbols: &Symbols) -> &'static str {
    match severity {
        EventSeverity::Info => symbols.idle(),
        EventSeverity::Warning => symbols.warning(),
        EventSeverity::Error => symbols.error(),
    }
}

/// Resources panel (Phase D): host CPU/RAM and the llama-server process.
///
/// Missing values render as the placeholder, never 0. The process row is
/// only shown when exactly one candidate matched (`identity == Exact`);
/// with several candidates the endpoint association is not confirmed, so the
/// panel reports the candidate count instead of naming a process.
fn render_resources(f: &mut Frame, area: Rect, state: &AppState, symbols: &Symbols) {
    let block = Block::bordered().title(" Resources ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let sys = match &state.system {
        Some(s) => s,
        None => {
            let text = Paragraph::new(vec![
                Line::from(format!("{} System monitor unavailable", symbols.warning())),
                Line::from("Host and process metrics are not being sampled."),
            ]);
            f.render_widget(text, inner);
            return;
        }
    };

    let mut lines: Vec<Line> = Vec::new();
    // Line 1: host CPU + RAM.
    let cpu = format_percent_value(sys.cpu_usage_percent, symbols);
    let ram = ram_display(sys.ram_used_bytes, sys.ram_total_bytes, symbols);
    lines.push(Line::from(format!("CPU {cpu}   RAM {ram}")));

    // Line 2: the llama-server process, or a candidate-count note.
    let process_line = process_line(sys, symbols);
    lines.push(Line::from(process_line));

    let text = Paragraph::new(lines);
    f.render_widget(text, inner);
}

/// "RAM used/total (pct)" or placeholder when either value is missing.
fn ram_display(used: Option<u64>, total: Option<u64>, symbols: &Symbols) -> String {
    match (used, total) {
        (Some(u), Some(t)) if t > 0 => {
            let pct = (u as f64 / t as f64) * 100.0;
            format!("{}/{} ({pct:.0}%)", bytes(u), bytes(t))
        }
        _ => placeholder(symbols).to_string(),
    }
}

/// The llama-server process row. Never claims an endpoint association for a
/// process we cannot confirm is the server.
fn process_line(sys: &SystemSnapshot, symbols: &Symbols) -> String {
    match (&sys.process, sys.process_match_count) {
        (Some(p), _) => {
            let cpu = format_percent_value(p.cpu_usage_percent, symbols);
            let mem = match p.memory_bytes {
                Some(m) => bytes(m),
                None => placeholder(symbols).to_string(),
            };
            let up = match p.uptime_secs {
                Some(s) => human_secs(s),
                None => placeholder(symbols).to_string(),
            };
            format!("{} {} CPU {} Mem {} Up {}", symbols.active(), p.name, cpu, mem, up)
        }
        // Several candidates: we cannot say which one the endpoint is.
        (None, Some(n)) if n > 1 => {
            format!("{} {n} llama-server processes; endpoint not associated", symbols.warning())
        }
        // No matching process (0, or a defensive 1 with no process record):
        // HTTP monitoring continues; the process row is a neutral note.
        (None, Some(_)) => "llama-server process not found".to_string(),
        // The process list could not be read at all.
        (None, None) => "llama-server process: unavailable".to_string(),
    }
}

/// Compact byte format: 999, 1.2K, 187.8M, 1.5G.
fn bytes(v: u64) -> String {
    const UNITS: [&str; 4] = ["", "K", "M", "G"];
    let mut value = v as f64;
    let mut idx = 0;
    while value >= 1024.0 && idx < UNITS.len() - 1 {
        value /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{v}")
    } else {
        format!("{value:.1}{}", UNITS[idx])
    }
}

/// Human uptime: "12 m", "1 h 4 m", "36 s".
fn human_secs(secs: u64) -> String {
    if secs < 60 {
        format!("{secs} s")
    } else if secs < 3600 {
        format!("{} m", secs / 60)
    } else {
        format!("{} h {} m", secs / 3600, (secs % 3600) / 60)
    }
}

/// Relative age of an event, in the same style as the header's
/// "updated N ago" line (no sub-second precision: events are sparse).
fn event_age(timestamp: SystemTime) -> String {
    match SystemTime::now().duration_since(timestamp) {
        Ok(d) => {
            let secs = d.as_secs_f64();
            if secs < 1.0 {
                "<1 s".into()
            } else if secs < 60.0 {
                format!("{secs:.0} s")
            } else if secs < 3600.0 {
                format!("{:.0} min", secs / 60.0)
            } else {
                format!("{:.0} h", secs / 3600.0)
            }
        }
        Err(_) => "-".to_string(),
    }
}

/// One sparkline row: one glyph per sample. Missing -> blank (gap);
/// present (including 0) -> ramp glyph scaled to the window maximum.
fn sparkline(values: &[Option<f64>], ascii: bool) -> String {
    let ramp: Vec<char> = if ascii {
        "._:-+*#".chars().collect()
    } else {
        "▁▂▃▄▅▆▇█".chars().collect()
    };
    let max = values.iter().filter_map(|v| *v).fold(0.0_f64, f64::max);
    values
        .iter()
        .map(|v| match v {
            None => ' ',
            Some(v) => {
                let idx = if max <= 0.0 {
                    0
                } else {
                    let ratio = (*v / max).clamp(0.0, 1.0);
                    (ratio * (ramp.len() - 1) as f64) as usize
                };
                ramp[idx]
            }
        })
        .collect()
}

/// Two stacked bar rows for one rate series (top + bottom). Missing ->
/// blanks in both rows; 0 -> lowest glyph in the bottom row only.
fn bar_rows(values: &[Option<f64>], ascii: bool) -> (String, String) {
    let max = values.iter().filter_map(|v| *v).fold(0.0_f64, f64::max);
    let mut top = String::with_capacity(values.len());
    let mut bottom = String::with_capacity(values.len());
    for v in values {
        match v {
            None => {
                top.push(' ');
                bottom.push(' ');
            }
            Some(v) => {
                let ratio = if max <= 0.0 { 0.0 } else { (*v / max).clamp(0.0, 1.0) };
                let glyph = if ascii {
                    if ratio >= 0.5 {
                        '#'
                    } else {
                        '*'
                    }
                } else if ratio >= 0.5 {
                    '█'
                } else {
                    '▄'
                };
                top.push(if ratio >= 0.5 { glyph } else { ' ' });
                bottom.push(if ascii { '.' } else { '▀' });
            }
        }
    }
    (top, bottom)
}

/// Latest value of a rate series, or the placeholder (never 0).
fn latest_rate(values: &[Option<f64>], symbols: &Symbols) -> String {
    match values.iter().rev().find_map(|v| *v) {
        Some(v) => format!("{v:.1}"),
        None => placeholder(symbols).to_string(),
    }
}

/// Latest value of a count series (stored as f64 for the sparkline),
/// rendered as an integer; placeholder when missing (never 0).
fn latest_int(values: &[Option<f64>], symbols: &Symbols) -> String {
    match values.iter().rev().find_map(|v| *v) {
        Some(v) => (v as u64).to_string(),
        None => placeholder(symbols).to_string(),
    }
}

/// Slot monitoring table: stable ID order, one selected row, vertical
/// scrolling, and responsive columns. Missing values render as the
/// placeholder, never as 0. Per-slot rates do not exist in the normalized
/// model, so no rate columns are shown (no guessed values).
fn render_slots(f: &mut Frame, area: Rect, state: &AppState, symbols: &Symbols) {
    let block = Block::bordered().title(" Slots ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    if !state.capabilities.slots {
        // /slots endpoint absent — distinct from "zero slots reported".
        let text = Paragraph::new(vec![
            Line::from(format!("{} Slots unavailable", symbols.warning())),
            Line::from("Per-slot monitoring will not be available."),
        ]);
        f.render_widget(text, inner);
        return;
    }

    let slots = state.visible_slots();
    if slots.is_empty() {
        let text = Paragraph::new("No slots reported");
        f.render_widget(text, inner);
        return;
    }

    let (headers, widths) = slot_columns(inner.width as usize);
    let sel_marker = if symbols.is_ascii() { "> " } else { "▶ " };
    let rows: Vec<Row> = slots
        .iter()
        .enumerate()
        .map(|(i, slot)| {
            let marker = if i == state.selected_slot { sel_marker } else { "  " };
            let mut cells: Vec<String> = Vec::with_capacity(headers.len() + 1);
            cells.push(marker.to_string());
            cells.extend(slot_cell_values(slot, symbols, headers.len()));
            Row::new(cells)
        })
        .collect();

    let mut header_cells: Vec<&str> = Vec::with_capacity(headers.len() + 1);
    header_cells.push("");
    header_cells.extend(headers.iter().copied());
    let header = Row::new(header_cells).style(Style::default());

    // The header occupies one row of the viewport; the rest fit data rows.
    let viewport = rows.len().min(inner.height.saturating_sub(1) as usize);
    let offset = state.slot_scroll_offset(viewport);
    let visible: Vec<Row> = rows[offset..offset + viewport].to_vec();

    let table = ratatui::widgets::Table::new(visible, widths)
        .header(header)
        .row_highlight_style(Style::default().add_modifier(Modifier::BOLD));
    f.render_widget(table, inner);
}

/// Responsive slot columns: (header labels, column widths). A leading
/// marker column (the selection indicator) is prepended by the caller.
/// Fewer columns as the width shrinks; the minimum (80-wide terminal)
/// keeps ID, State, Phase, and Context.
fn slot_columns(width: usize) -> (Vec<&'static str>, Vec<Constraint>) {
    // (headers, minimum widths). The marker column is always 3 wide.
    let (headers, min): (Vec<&'static str>, Vec<usize>) = if width >= 100 {
        (vec!["ID", "State", "Phase", "Prompt", "Generated", "Context"], vec![4, 8, 11, 9, 10, 9])
    } else if width >= 88 {
        (vec!["ID", "State", "Phase", "Prompt", "Context"], vec![4, 8, 11, 9, 9])
    } else {
        (vec!["ID", "State", "Phase", "Context"], vec![4, 8, 11, 9])
    };
    let marker_width = 3usize;
    let total_min: usize = min.iter().sum::<usize>() + marker_width;
    let extra = width.saturating_sub(total_min);
    let data_total: usize = min.iter().sum();
    let mut widths: Vec<Constraint> = Vec::with_capacity(min.len() + 1);
    widths.push(Constraint::Length(marker_width as u16));
    for m in &min {
        let share = extra.saturating_mul(*m).saturating_div(data_total.max(1));
        widths.push(Constraint::Length((*m as u16).saturating_add(share as u16)));
    }
    (headers, widths)
}

/// One slot's cell values, aligned with the active column set. Missing
/// counters render as the placeholder, never as 0.
fn slot_cell_values(slot: &SlotSnapshot, symbols: &Symbols, n_data_cols: usize) -> Vec<String> {
    let state_str = if slot.is_processing { "ACTIVE" } else { "IDLE" };
    let phase = slot.phase.display();
    let prompt = format_int(slot.n_prompt_tokens, symbols);
    let generated = format_int(slot.n_decoded, symbols);
    let context = format_int(slot.n_ctx, symbols);
    // Column sets mirror `slot_columns`: wide has Generated, the others
    // drop it; compact also drops Prompt.
    match n_data_cols {
        6 => vec![slot.id.to_string(), state_str.into(), phase.into(), prompt, generated, context],
        5 => vec![slot.id.to_string(), state_str.into(), phase.into(), prompt, context],
        _ => vec![slot.id.to_string(), state_str.into(), phase.into(), context],
    }
}

/// Compact integer format: 999, 1.2K, 187.8K, 1.5M. Missing -> placeholder.
fn format_int(v: Option<u64>, symbols: &Symbols) -> String {
    match v {
        None => placeholder(symbols).to_string(),
        Some(n) => {
            if n >= 1_000_000 {
                format!("{:.1}M", n as f64 / 1_000_000.0)
            } else if n >= 1_000 {
                format!("{:.1}K", n as f64 / 1_000.0)
            } else {
                n.to_string()
            }
        }
    }
}

/// Waiting view: first snapshot not yet received.
fn render_waiting(f: &mut Frame, area: Rect, endpoint: &str, symbols: &Symbols) {
    let block = Block::bordered().title(" LLAMATOP ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width > 0 && inner.height >= 3 {
        let text = Paragraph::new(format!(
            "Waiting for data...\n\nEndpoint:\n  {}",
            trunc(endpoint, (inner.width as usize).saturating_sub(2), symbols.is_ascii())
        ));
        f.render_widget(text, inner);
    }
}

/// Lifecycle view (loading / sleeping).
fn render_lifecycle(f: &mut Frame, area: Rect, server_state: &str, detail: &str) {
    let block = Block::bordered().title(format!(" Server: {} ", server_state));
    let inner = block.inner(area);
    f.render_widget(block, area);
    if inner.width > 0 && inner.height > 0 {
        let text = Paragraph::new(detail);
        f.render_widget(text, inner);
    }
}

/// Connection view: connecting / reconnecting / disconnected / auth failure.
fn render_connection_view(
    f: &mut Frame,
    area: Rect,
    state: &AppState,
    snap: &BackendSnapshot,
    symbols: &Symbols,
) {
    let width = area.width as usize;
    let lines: Vec<String> = if state.authentication_failed {
        vec![
            format!("{} AUTHENTICATION FAILED", symbols.error()),
            String::new(),
            "Set the API key through:".into(),
            String::new(),
            format!("  $env:{} = \"...\"", state.api_key_env),
        ]
    } else {
        match snap.connection {
            ConnectionState::Connecting => vec![
                format!("{} CONNECTING", symbols.idle()),
                String::new(),
                "Connecting to llama.cpp...".into(),
            ],
            ConnectionState::Reconnecting => vec![
                format!("{} RECONNECTING", symbols.warning()),
                String::new(),
                "Connection was interrupted.".into(),
                "Retrying automatically...".into(),
            ],
            _ => {
                let mut v = vec![
                    format!("{} DISCONNECTED", symbols.error()),
                    String::new(),
                    "Could not connect to llama.cpp.".into(),
                    String::new(),
                    "Endpoint:".into(),
                    format!(
                        "  {}",
                        trunc(&state.endpoint, width.saturating_sub(2), symbols.is_ascii())
                    ),
                    String::new(),
                    "Press r to retry or q to quit.".into(),
                ];
                if let Some(msg) = &state.connection_message {
                    v.insert(
                        2,
                        format!("  {}", trunc(msg, width.saturating_sub(2), symbols.is_ascii())),
                    );
                }
                v
            }
        }
    };
    let text = Paragraph::new(lines.join("\n"));
    f.render_widget(text, area);
}

/// Footer: only keys that are actually available right now are advertised.
/// `p` appears once a snapshot exists (pause is ignored before that), and
/// reads "Resume" while paused so the label matches the action it triggers.
/// The slot-navigation keys appear only when slot selection is usable
/// (`/slots` available and at least one slot visible); they are named
/// `↑/↓` in Unicode mode and `j`/`k` in ASCII mode.
fn render_footer(f: &mut Frame, area: Rect, state: &AppState, symbols: &Symbols) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let mut footer = " q Quit   r Reconnect".to_string();
    if state.can_pause() {
        let action = if state.paused { "p Resume" } else { "p Pause" };
        footer.push_str("   ");
        footer.push_str(action);
    }
    // The event log panel (and its `l` toggle) only exists in the connected
    // view, so it is advertised only while a connected snapshot is visible.
    if state.visible_snapshot().map(|s| s.connection == ConnectionState::Connected).unwrap_or(false)
    {
        footer.push_str("   ");
        footer.push_str("l Events");
    }
    if state.can_select_slot() {
        let keys = if symbols.is_ascii() { "j/k" } else { "↑/↓" };
        footer.push_str("   ");
        footer.push_str(keys);
        footer.push_str(" Select");
    }
    let text = Paragraph::new(format!(" {} ", footer)).alignment(Alignment::Center);
    f.render_widget(text, area);
}

/// Connection symbol + label. Auth failures get a dedicated label.
fn connection_display(
    state: &AppState,
    snap: &BackendSnapshot,
    symbols: &Symbols,
) -> (&'static str, &'static str) {
    if state.authentication_failed {
        return (symbols.error(), "AUTH FAILED");
    }
    match snap.connection {
        ConnectionState::Connected => (symbols.active(), "CONNECTED"),
        ConnectionState::Connecting => (symbols.idle(), "CONNECTING"),
        ConnectionState::Reconnecting => (symbols.warning(), "RECONNECTING"),
        ConnectionState::Disconnected => (symbols.error(), "DISCONNECTED"),
        ConnectionState::Error => (symbols.error(), "ERROR"),
    }
}

/// Confidence marker: `*` for estimated, `?` for unknown. Exact/high show
/// nothing extra — the phase label itself already carries `*`/`?` when the
/// phase is estimated/unknown; this suffix covers Estimated confidence on a
/// phase whose label has no marker (e.g. MIXED with estimated evidence).
fn confidence_suffix(confidence: Confidence) -> &'static str {
    match confidence {
        Confidence::Exact | Confidence::High => "",
        Confidence::Estimated => "*",
        Confidence::Unknown => "?",
    }
}

/// Backend display name. Only llama.cpp exists; the name comes from the
/// backend trait, but the panels do not hold a backend instance, so it is
/// rendered from the fixed supported set.
fn backend_name() -> &'static str {
    "llama.cpp"
}

/// Placeholder for an unreported value (never a fabricated 0).
fn placeholder(symbols: &Symbols) -> &'static str {
    if symbols.is_ascii() {
        "-"
    } else {
        "—"
    }
}

/// Truncate a string to a display width, adding an ellipsis when cut.
/// Never slices at a UTF-8 byte boundary; ASCII mode uses `...`.
/// The ellipsis is measured with display width, not UTF-8 byte length
/// (`…` is 3 bytes but 1 column). The result always satisfies
/// `width(result) <= max_width`, including tiny budgets (width 0/1/2).
fn trunc(s: &str, max_width: usize, ascii: bool) -> String {
    if max_width == 0 {
        return String::new();
    }
    if s.width() <= max_width {
        return s.to_string();
    }
    let ellipsis = if ascii { "..." } else { "…" };
    // If even the full ellipsis does not fit, return as much of it as does
    // (e.g. ASCII width 1 -> ".", width 2 -> "..").
    if max_width < ellipsis.width() {
        return prefix_by_width(ellipsis, max_width);
    }
    let budget = max_width - ellipsis.width();
    let mut out = prefix_by_width(s, budget);
    out.push_str(ellipsis);
    out
}

/// The longest prefix of `s` whose display width is at most `max_width`.
/// Iterates by `char`, so CJK, full-width, and emoji are never split.
fn prefix_by_width(s: &str, max_width: usize) -> String {
    let mut out = String::new();
    let mut width = 0;
    for ch in s.chars() {
        let w = ch.width().unwrap_or(0);
        if width + w > max_width {
            break;
        }
        out.push(ch);
        width += w;
    }
    out
}

/// `Option<u64>` -> decimal string or placeholder.
fn format_opt_u64(v: Option<u64>, symbols: &Symbols) -> String {
    match v {
        Some(n) => n.to_string(),
        None => placeholder(symbols).to_string(),
    }
}

/// Local delta rate first, server-reported average as fallback, else "—".
fn rate_value(delta: Option<f64>, reported: Option<f64>, symbols: &Symbols) -> String {
    match delta.or(reported) {
        Some(v) => format!("{v:.1} tok/s"),
        None => placeholder(symbols).to_string(),
    }
}

/// Context size: 16384 -> 16K, 1048576 -> 1M, or placeholder.
fn format_context(v: Option<u64>, symbols: &Symbols) -> String {
    match v {
        None => placeholder(symbols).to_string(),
        Some(n) => {
            if n >= 1_000_000 {
                format!("{}M", n / 1_000_000)
            } else if n >= 1_000 {
                format!("{}K", n / 1_000)
            } else {
                n.to_string()
            }
        }
    }
}

/// Fraction -> percent string with one decimal, or placeholder.
fn format_percent(v: Option<f64>, symbols: &Symbols) -> String {
    match v {
        Some(p) => format!("{:.1}%", p * 100.0),
        None => placeholder(symbols).to_string(),
    }
}

/// Format a value that is already a 0-100 percentage (host/process CPU).
fn format_percent_value(v: Option<f64>, symbols: &Symbols) -> String {
    match v {
        Some(p) => format!("{p:.1}%"),
        None => placeholder(symbols).to_string(),
    }
}

/// "320 ms ago" / "2.1 s ago" / "3 min ago", or placeholder when never.
fn update_age(state: &AppState, symbols: &Symbols) -> String {
    match state.last_update_age_ms(Instant::now()) {
        None => placeholder(symbols).to_string(),
        Some(ms) if ms < 1000 => format!("{ms} ms ago"),
        Some(ms) if ms < 60_000 => format!("{:.1} s ago", ms as f64 / 1000.0),
        Some(ms) => format!("{} min ago", ms / 60_000),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{ProcessIdentity, ProcessSnapshot};

    #[test]
    fn trunc_keeps_short_strings_intact() {
        assert_eq!(trunc("abc", 10, false), "abc");
        assert_eq!(trunc("abc", 10, true), "abc");
    }

    #[test]
    fn trunc_unicode_uses_display_width_ellipsis() {
        // Width budget of 5 over "abcdefgh" -> "abcd…" (4 + 1 ellipsis column).
        let out = trunc("abcdefgh", 5, false);
        assert_eq!(out, "abcd…");
        assert_eq!(out.width(), 5, "must not exceed the display width budget");
        // The ellipsis is a single column, not three bytes.
        assert_eq!(out.chars().count(), 5);
    }

    #[test]
    fn trunc_ascii_mode_uses_ascii_only() {
        // ASCII ellipsis is 3 columns, so a budget of 5 keeps 2 chars + "...".
        let out = trunc("abcdefgh", 5, true);
        assert_eq!(out, "ab...");
        assert!(out.is_ascii());
        assert_eq!(out.width(), 5);
    }

    #[test]
    fn trunc_does_not_split_cjk_or_emoji_chars() {
        // CJK chars are 2 columns wide; budget 4 keeps one char + "…",
        // never a half-char.
        let out = trunc("日本語テキスト", 4, false);
        assert_eq!(out, "日…");
        assert!(out.width() <= 4);
        // Emoji are 2 columns wide and must never be split.
        let out = trunc("a🎉a🎉aaaa", 5, false);
        assert_eq!(out, "a🎉a…");
        assert_eq!(out.width(), 5);
    }

    #[test]
    fn trunc_handles_zero_and_tiny_widths_without_panic() {
        assert_eq!(trunc("abcdef", 0, false), "");
        assert_eq!(trunc("abcdef", 0, true), "");
        assert_eq!(trunc("abcdef", 1, false), "…");
        assert_eq!(trunc("あいう", 1, false), "…");
        // A string that fits is never truncated.
        assert_eq!(trunc("abc", 3, true), "abc");
    }

    #[test]
    fn trunc_ascii_tiny_widths_never_exceed_budget() {
        // ASCII ellipsis is 3 columns; smaller budgets get a shortened
        // ellipsis instead of overflowing the width.
        assert_eq!(trunc("abcdef", 1, true), ".");
        assert_eq!(trunc("abcdef", 2, true), "..");
        assert_eq!(trunc("abcdef", 3, true), "...");
        assert_eq!(trunc("abcdef", 4, true), "a...");
    }

    #[test]
    fn trunc_result_never_exceeds_max_width() {
        let samples = ["abcdef", "日本語テキスト", "a🎉b🎉c", "  spaced  "];
        for s in samples {
            for max_width in 0..=12 {
                for ascii in [false, true] {
                    let out = trunc(s, max_width, ascii);
                    assert!(
                        out.width() <= max_width,
                        "width {} > budget {} for {:?} ascii={}",
                        out.width(),
                        max_width,
                        s,
                        ascii
                    );
                    if ascii {
                        // ASCII mode must not add a Unicode ellipsis; the
                        // source characters themselves may be non-ASCII.
                        assert!(!out.contains('…'), "no Unicode ellipsis in ASCII mode: {:?}", out);
                    }
                }
            }
        }
    }

    #[test]
    fn trunc_returns_string_as_is_when_it_fits() {
        assert_eq!(trunc("…x…", 3, true), "…x…");
        assert_eq!(trunc("…x…", 10, true), "…x…");
        assert_eq!(trunc("日本語", 6, false), "日本語");
    }

    // --- Phase D: resources panel helpers ---

    #[test]
    fn bytes_compact_format() {
        assert_eq!(bytes(999), "999");
        assert_eq!(bytes(1234), "1.2K");
        assert_eq!(bytes(187_800_000), "179.1M");
        assert_eq!(bytes(1_500_000_000), "1.4G");
    }

    #[test]
    fn human_secs_format() {
        assert_eq!(human_secs(36), "36 s");
        assert_eq!(human_secs(720), "12 m");
        assert_eq!(human_secs(3720), "1 h 2 m");
    }

    #[test]
    fn ram_display_requires_both_values() {
        let sym = Symbols::new(false);
        assert_eq!(ram_display(Some(50), Some(100), &sym), "50/100 (50%)");
        assert_eq!(ram_display(None, Some(100), &sym), "—");
        assert_eq!(ram_display(Some(50), Some(0), &sym), "—");
    }

    #[test]
    fn process_line_never_fabricates_association() {
        let sym = Symbols::new(false);
        // Exactly one match: the process is shown.
        let snap = SystemSnapshot {
            cpu_usage_percent: None,
            ram_used_bytes: None,
            ram_total_bytes: None,
            process_match_count: Some(1),
            process: Some(ProcessSnapshot {
                pid: 42,
                name: "llama-server.exe".into(),
                cpu_usage_percent: Some(9.0),
                memory_bytes: Some(29_650_837_504),
                uptime_secs: Some(8372),
                identity: ProcessIdentity::Exact,
            }),
        };
        let line = process_line(&snap, &sym);
        assert!(line.contains("llama-server.exe"));
        assert!(line.contains("27.6G"), "process memory is GiB: {line}");
        assert!(!line.contains("not associated"), "a single exact match is not ambiguous");

        // Several candidates: no process is named.
        let multi = SystemSnapshot {
            process_match_count: Some(3),
            process: None,
            ..SystemSnapshot {
                cpu_usage_percent: None,
                ram_used_bytes: None,
                ram_total_bytes: None,
                process: None,
                process_match_count: None,
            }
        };
        let line = process_line(&multi, &sym);
        assert!(line.contains("3 llama-server processes"));
        assert!(line.contains("not associated"));
        assert!(!line.contains("llama-server.exe"), "no candidate is named");

        // Zero matches: neutral note.
        let none = SystemSnapshot {
            process_match_count: Some(0),
            ..SystemSnapshot {
                cpu_usage_percent: None,
                ram_used_bytes: None,
                ram_total_bytes: None,
                process: None,
                process_match_count: None,
            }
        };
        assert_eq!(process_line(&none, &sym), "llama-server process not found");
    }
}
