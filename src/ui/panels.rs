//! Basic TUI panels (Step 4): header, inference summary, connection and
//! error views, waiting view, and footer.
//!
//! `Frame::area()` is the single source of truth for the drawable region —
//! `AppState::terminal_size` only records the last resize event. Every
//! helper is safe for degenerate sizes (no panic, no UTF-8 mid-character
//! slicing) and never fabricates a value that the backend did not report.

use std::time::Instant;

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::app::state::AppState;
use crate::display::Symbols;
use crate::domain::{BackendSnapshot, Confidence, ConnectionState, ServerState};

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
            render_footer(f, chunks[2], symbols);
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
                    render_footer(f, chunks[2], symbols);
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
                            // Inference takes the top of the middle area;
                            // the status/warning area gets the rest.
                            let mid = Layout::vertical([Constraint::Length(6), Constraint::Min(0)])
                                .split(chunks[1]);
                            render_inference(f, mid[0], s, symbols);
                            render_status(f, mid[1], state, symbols);
                        }
                        _ => render_connection_view(f, chunks[1], state, s, symbols),
                    }
                    render_footer(f, chunks[2], symbols);
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

/// Status/warning area: capability warnings and the last error message.
fn render_status(f: &mut Frame, area: Rect, state: &AppState, symbols: &Symbols) {
    let mut lines: Vec<String> = Vec::new();
    if !state.capabilities.metrics {
        lines.push(format!("{} Metrics unavailable", symbols.warning()));
        lines.push("Throughput values cannot be displayed.".into());
        lines.push("Start llama-server with --metrics.".into());
    }
    if !state.capabilities.slots {
        if !lines.is_empty() {
            lines.push(String::new());
        }
        lines.push(format!("{} Slots unavailable", symbols.warning()));
        lines.push("Per-slot monitoring will not be available.".into());
    }
    if let Some(msg) = &state.connection_message {
        if !lines.is_empty() {
            lines.push(String::new());
        }
        lines.push(format!(
            "{} {}",
            symbols.error(),
            trunc(msg, (area.width as usize).saturating_sub(2), symbols.is_ascii())
        ));
    }
    if lines.is_empty() {
        return;
    }
    let text = Paragraph::new(lines.join("\n"));
    f.render_widget(text, area);
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

/// Footer: only keys that are actually implemented are advertised.
fn render_footer(f: &mut Frame, area: Rect, _symbols: &Symbols) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let text = Paragraph::new(" q Quit   r Reconnect   p Pause ").alignment(Alignment::Center);
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
/// (`…` is 3 bytes but 1 column).
fn trunc(s: &str, max_width: usize, ascii: bool) -> String {
    if max_width == 0 {
        return String::new();
    }
    if s.width() <= max_width {
        return s.to_string();
    }
    let ellipsis = if ascii { "..." } else { "…" };
    let ellipsis_width = ellipsis.width();
    if max_width <= ellipsis_width {
        return ellipsis.to_string();
    }
    let mut out = String::new();
    let mut width = 0;
    for ch in s.chars() {
        let w = ch.width().unwrap_or(0);
        if width + w > max_width - ellipsis_width {
            break;
        }
        out.push(ch);
        width += w;
    }
    out.push_str(ellipsis);
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
        assert_eq!(trunc("abc", 0, false), "");
        assert_eq!(trunc("abc", 0, true), "");
        assert_eq!(trunc("abc", 1, false), "…");
        assert_eq!(trunc("abcd", 3, true), "...");
        assert_eq!(trunc("あいう", 1, false), "…");
        // A string that fits is never truncated.
        assert_eq!(trunc("abc", 3, true), "abc");
    }
}
