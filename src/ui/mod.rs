//! Terminal user interface for `llamatop`.
//!
//! The TUI is the only module that talks to the terminal. It renders from
//! application state and never performs HTTP; data arrives through an event
//! channel from the backend collector, and keyboard input is mapped by
//! `input.rs`.
//!
//! Layering: `ui` depends on `app`, `domain`, and `config`. It does not
//! depend on `backend` raw types or perform any network I/O.

mod input;
mod panels;
mod terminal;

use ratatui::backend::CrosstermBackend;
use ratatui::Terminal as RatatuiTerminal;

use crate::app::event::AppEvent;
use crate::app::runtime;
use crate::app::state::AppState;
use crate::config::Config;
use crate::display::Symbols;
use crate::ui::input::InputReader;

pub use terminal::{PanicRestorer, TerminalGuard, TerminalModes};

/// Run the interactive TUI until the user quits.
///
/// Returns the process exit code (0 for a clean quit). `Err` is returned on
/// terminal initialization failure, a rendering failure, an event-loop
/// failure (e.g. a failed command send), or a collector task failure during
/// shutdown; the terminal is restored on every exit path.
pub async fn run_tui(config: &Config) -> anyhow::Result<i32> {
    // Only the bare `llamatop` invocation reaches here; doctor/snapshot run
    // without raw mode or the alternate screen.
    let mut guard =
        TerminalGuard::new().map_err(|e| anyhow::anyhow!("terminal initialization failed: {e}"))?;
    let restorer = PanicRestorer::install(guard.modes());

    let backend = CrosstermBackend::new(std::io::stdout());
    let mut term = RatatuiTerminal::new(backend)
        .map_err(|e| anyhow::anyhow!("terminal initialization failed: {e}"))?;

    let symbols = Symbols::new(config.ascii);
    let initial_size = term.size().map(|s| (s.width, s.height)).unwrap_or((80, 20));

    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<AppEvent>();
    let mut reader = InputReader::start(event_tx.clone());
    let config_owned = config.clone();

    let result =
        runtime::run(config_owned, event_tx, event_rx, initial_size, |state: &AppState| {
            term.draw(|f| panels::render(f, state, &symbols)).map(|_| ())
        })
        .await;

    reader.stop();
    guard.restore();
    drop(restorer);

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{BackendSnapshot, Confidence, ConnectionState, ServerState, WorkloadPhase};
    use unicode_width::UnicodeWidthChar;

    /// Render one frame and return the flattened cell text.
    fn render_content(state: &AppState, ascii: bool, width: u16, height: u16) -> String {
        let backend = ratatui::backend::TestBackend::new(width, height);
        let mut term = RatatuiTerminal::new(backend).expect("terminal");
        term.draw(|f| panels::render(f, state, &Symbols::new(ascii))).expect("draw");
        let buf = term.backend().buffer();
        (0..height)
            .flat_map(|y| (0..width).map(move |x| buf[(x, y)].symbol().to_string()))
            .collect()
    }

    fn connected_ready(overrides: impl FnOnce(&mut BackendSnapshot)) -> AppState {
        let config = Config::default();
        let mut state = AppState::new(&config);
        let mut snap = BackendSnapshot {
            connection: ConnectionState::Connected,
            server: ServerState::Ready,
            workload_phase: WorkloadPhase::Decode,
            workload_confidence: Confidence::High,
            model_name: Some("qwen3.8-27b".into()),
            active_requests: Some(1),
            queued_requests: Some(0),
            context_max_tokens: Some(262_144),
            total_slots: Some(4),
            ..Default::default()
        };
        overrides(&mut snap);
        state.apply_snapshot(snap);
        state
    }

    #[test]
    fn render_is_safe_at_zero_size() {
        let state = AppState::new(&Config::default());
        let content = render_content(&state, false, 1, 1);
        let _ = content; // must not panic; content may be empty
    }

    #[test]
    fn waiting_shows_waiting_and_endpoint_not_disconnected() {
        let state = AppState::new(&Config::default());
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Waiting for data..."));
        assert!(content.contains("http://127.0.0.1:8080"));
        assert!(!content.contains("DISCONNECTED"));
    }

    #[test]
    fn connected_ready_shows_header_and_inference() {
        let state = connected_ready(|s| {
            s.prompt_tokens_per_second = Some(1842.0);
            s.generation_tokens_per_second = Some(52.0);
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("CONNECTED"));
        assert!(content.contains("READY"));
        assert!(content.contains("DECODE"));
        assert!(content.contains("llama.cpp"));
        assert!(content.contains("qwen3.8-27b"));
        assert!(content.contains("1842.0 tok/s"));
        assert!(content.contains("52.0 tok/s"));
        assert!(content.contains("Inference"));
    }

    #[test]
    fn connecting_shows_connecting_view() {
        let mut state = connected_ready(|_| {});
        state.apply_snapshot(BackendSnapshot {
            connection: ConnectionState::Connecting,
            ..Default::default()
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("CONNECTING"));
        assert!(content.contains("Connecting to llama.cpp..."));
    }

    #[test]
    fn reconnecting_shows_reconnecting_view() {
        let mut state = connected_ready(|_| {});
        state.apply_snapshot(BackendSnapshot {
            connection: ConnectionState::Reconnecting,
            ..Default::default()
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("RECONNECTING"));
        assert!(content.contains("Retrying automatically..."));
    }

    #[test]
    fn disconnected_shows_disconnected_view_with_endpoint() {
        let mut state = connected_ready(|_| {});
        state.apply_error(crate::app::event::BackendErrorSummary::new("connection refused"));
        state.apply_snapshot(BackendSnapshot {
            connection: ConnectionState::Disconnected,
            error: Some("connection refused".into()),
            ..Default::default()
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("DISCONNECTED"));
        assert!(content.contains("Could not connect to llama.cpp."));
        assert!(content.contains("Endpoint:"));
        assert!(content.contains("http://127.0.0.1:8080"));
        assert!(content.contains("Press r to retry or q to quit."));
    }

    #[test]
    fn loading_shows_loading_view() {
        let state = connected_ready(|s| s.server = ServerState::Loading);
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("LOADING"));
        assert!(content.contains("Model is loading..."));
    }

    #[test]
    fn sleeping_shows_sleeping_view() {
        let state = connected_ready(|s| s.server = ServerState::Sleeping);
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("SLEEPING"));
    }

    #[test]
    fn idle_shows_idle_phase() {
        let state = connected_ready(|s| {
            s.workload_phase = WorkloadPhase::Idle;
            s.workload_confidence = Confidence::High;
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("IDLE"));
    }

    #[test]
    fn prefill_likely_displays_star_marker() {
        let state = connected_ready(|s| {
            s.workload_phase = WorkloadPhase::PrefillLikely;
            s.workload_confidence = Confidence::Estimated;
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("PREFILL*"));
    }

    #[test]
    fn mixed_estimated_displays_star_marker() {
        let state = connected_ready(|s| {
            s.workload_phase = WorkloadPhase::Mixed;
            s.workload_confidence = Confidence::Estimated;
        });
        let content = render_content(&state, false, 80, 20);
        // Phase label MIXED plus the estimated-confidence marker.
        assert!(content.contains("MIXED*"));
    }

    #[test]
    fn processing_unknown_displays_question_marker() {
        let state = connected_ready(|s| {
            s.workload_phase = WorkloadPhase::ProcessingUnknown;
            s.workload_confidence = Confidence::Unknown;
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("PROCESSING?"));
    }

    #[test]
    fn unavailable_metrics_display_placeholder_not_zero() {
        // Metrics unavailable: no delta and no server-reported average.
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("—"), "missing rates must show the em-dash placeholder");
        assert!(!content.contains("0.0 tok/s"), "missing rates must not be rendered as zero");
    }

    #[test]
    fn unavailable_optional_values_are_not_zero() {
        // active/queued/context/slots/spec all None.
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 80, 20);
        // The inference panel shows placeholders where nothing was reported.
        assert!(content.contains("—"));
        // Speculative acceptance is absent, not "0.0%".
        assert!(!content.contains("0.0%"));
    }

    #[test]
    fn long_model_name_does_not_break_layout() {
        let state = connected_ready(|s| {
            s.model_name = Some("Qwen3.8-27B-Q4_K_M-Long-Model-Name-That-Is-Quite-Long".into());
        });
        let content = render_content(&state, false, 80, 20);
        // The model name is truncated with an ellipsis; the rest of the
        // header line (phase/server) is still drawn.
        assert!(content.contains("Qwen3.8-27B-Q4_K_M-Long-Model"));
        assert!(content.contains("Server: READY"));
    }

    #[test]
    fn long_endpoint_does_not_break_layout() {
        let mut state = AppState::new(&Config::default());
        state.endpoint =
            "http://some-very-long-hostname.internal.example.com:8080/some/path".into();
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Waiting for data..."));
        // Truncated endpoint still starts with the scheme.
        assert!(content.contains("http://"));
    }

    #[test]
    fn ascii_mode_contains_no_unicode_status_symbols() {
        let state = connected_ready(|s| {
            s.prompt_tokens_per_second = Some(10.0);
            s.generation_tokens_per_second = Some(5.0);
        });
        let content = render_content(&state, true, 80, 20);
        // State symbols and the placeholder must all be ASCII in ASCII mode.
        assert!(!content.contains("●"));
        assert!(!content.contains("○"));
        assert!(!content.contains("▲"));
        assert!(!content.contains("✘"));
        assert!(!content.contains("…"));
        assert!(!content.contains("—"), "ASCII mode uses '-' for missing values");
        assert!(content.contains("-"), "ASCII placeholder is '-'");
    }

    #[test]
    fn rendering_at_80x20_does_not_panic() {
        let state = connected_ready(|s| {
            s.prompt_tokens_per_second = Some(1.0);
        });
        let _ = render_content(&state, false, 80, 20);
    }

    #[test]
    fn rendering_at_62x16_shows_too_small() {
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 62, 16);
        assert!(content.contains("Terminal is too small."));
        assert!(content.contains("Required: 80 x 20"));
        assert!(content.contains("Current: 62 x 16"));
    }

    #[test]
    fn rendering_at_1x1_does_not_panic() {
        let state = connected_ready(|_| {});
        let _ = render_content(&state, false, 1, 1);
        let waiting = AppState::new(&Config::default());
        let _ = render_content(&waiting, false, 1, 1);
    }

    #[test]
    fn paused_state_is_visible_in_header() {
        let mut state = connected_ready(|_| {});
        state.handle_input(crate::app::event::InputAction::TogglePause);
        assert!(state.paused);
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("PAUSED"));
    }

    #[test]
    fn footer_only_advertises_implemented_controls() {
        let state = AppState::new(&Config::default());
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("q Quit"));
        assert!(content.contains("r Reconnect"));
        // Pause is not possible before the first snapshot, so it is not
        // advertised in the waiting view.
        assert!(!content.contains("p Pause"));
        assert!(!content.contains("? Help"), "help modal is not implemented yet");
        assert!(!content.contains("l Logs"), "event log panel is not implemented yet");
    }

    #[test]
    fn footer_shows_pause_after_first_snapshot() {
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("p Pause"));
        assert!(!content.contains("p Resume"));
    }

    #[test]
    fn footer_shows_resume_label_while_paused() {
        let mut state = connected_ready(|_| {});
        state.handle_input(crate::app::event::InputAction::TogglePause);
        assert!(state.paused);
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("p Resume"));
        assert!(!content.contains("p Pause"));
    }

    #[test]
    fn rate_falls_back_to_server_reported_average() {
        let state = connected_ready(|s| {
            // No local delta; server-reported average only.
            s.prompt_tokens_per_second = None;
            s.prompt_tokens_per_second_reported = Some(999.0);
            s.generation_tokens_per_second = None;
            s.generation_tokens_per_second_reported = None;
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("999.0 tok/s"));
        // Generation has neither delta nor reported value: placeholder.
        assert!(content.contains("—"));
    }

    // --- Step 5: slot table tests ---

    /// A slot for rendering tests: only the fields the table shows.
    fn slot(id: u32, processing: bool, ctx: Option<u64>) -> crate::domain::SlotSnapshot {
        crate::domain::SlotSnapshot {
            id,
            task_id: None,
            is_processing: processing,
            n_ctx: ctx,
            n_tokens: None,
            n_prompt_tokens: None,
            n_prompt_tokens_processed: None,
            n_decoded: None,
            speculative: false,
            phase: if processing {
                crate::domain::SlotPhase::Decode
            } else {
                crate::domain::SlotPhase::Idle
            },
        }
    }

    /// Connected/ready state with the /slots capability enabled.
    fn connected_slots(overrides: impl FnOnce(&mut BackendSnapshot)) -> AppState {
        let mut state = connected_ready(overrides);
        state.apply_capabilities(crate::backend::BackendCapabilities {
            slots: true,
            metrics: true,
            ..Default::default()
        });
        state
    }

    /// Split flattened render content into rows using display width, so
    /// per-row assertions work in Unicode mode (multi-byte symbols).
    fn split_rows(content: &str, width: usize) -> Vec<String> {
        let mut rows = Vec::new();
        let mut cur = String::new();
        let mut w = 0usize;
        for ch in content.chars() {
            cur.push(ch);
            w += ch.width().unwrap_or(0);
            if w == width {
                rows.push(std::mem::take(&mut cur));
                w = 0;
            }
        }
        rows
    }

    #[test]
    fn slots_unavailable_view_when_endpoint_missing() {
        // Default capabilities: /slots not probed/available.
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Slots unavailable"));
        assert!(content.contains("Per-slot monitoring will not be available."));
        // Distinct from the zero-slots view.
        assert!(!content.contains("No slots reported"));
    }

    #[test]
    fn no_slots_reported_view_when_empty() {
        let state = connected_slots(|_| {}); // capability on, zero slots
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("No slots reported"));
        assert!(!content.contains("Slots unavailable"));
    }

    #[test]
    fn slot_table_shows_idle_slot() {
        let state = connected_slots(|s| {
            s.slots = vec![slot(0, false, Some(16_384))];
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("IDLE"));
        assert!(content.contains("16.4K"), "context must be compact-formatted");
    }

    #[test]
    fn slot_table_shows_active_decode_slot() {
        let state = connected_slots(|s| {
            s.slots = vec![slot(3, true, Some(8_192))];
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("ACTIVE"));
        assert!(content.contains("DECODE"));
        assert!(content.contains("8.2K"));
    }

    #[test]
    fn slot_rows_render_in_id_order() {
        // Deliberately out of API order; distinct context values mark rows.
        let state = connected_slots(|s| {
            s.slots = vec![
                slot(2, false, Some(300)),
                slot(0, false, Some(100)),
                slot(1, false, Some(200)),
            ];
        });
        let content = render_content(&state, false, 80, 20);
        let p100 = content.find("100").expect("slot 0 row visible");
        let p200 = content.find("200").expect("slot 1 row visible");
        let p300 = content.find("300").expect("slot 2 row visible");
        assert!(p100 < p200 && p200 < p300, "rows must be ordered by slot ID, not API order");
    }

    #[test]
    fn selected_slot_row_is_marked_and_marker_follows_selection() {
        let mut state = connected_slots(|s| {
            s.slots = vec![
                slot(0, false, Some(111)),
                slot(1, false, Some(222)),
                slot(2, false, Some(333)),
            ];
        });
        // Initially the first row (id 0, ctx 111) is selected.
        let content = render_content(&state, false, 80, 20);
        let lines = split_rows(&content, 80);
        let row111 = lines.iter().find(|l| l.contains("111")).expect("row visible");
        assert!(row111.contains('▶'), "first row must carry the selection marker");

        state.handle_input(crate::app::event::InputAction::SlotDown);
        let content = render_content(&state, false, 80, 20);
        let lines = split_rows(&content, 80);
        let row222 = lines.iter().find(|l| l.contains("222")).expect("row visible");
        assert!(row222.contains('▶'), "marker must move to the selected row");
        let row111 = lines.iter().find(|l| l.contains("111")).expect("row visible");
        assert!(!row111.contains('▶'));
    }

    #[test]
    fn selected_row_stays_visible_while_scrolling() {
        // 10 slots; at 80x20 the table fits 5 data rows.
        let mut state = connected_slots(|s| {
            s.slots = (0..10).map(|i| slot(101 + i, false, Some(16_384))).collect();
        });
        for _ in 0..9 {
            state.handle_input(crate::app::event::InputAction::SlotDown);
        }
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("110"), "selected last row must be visible");
        assert!(content.contains("106"), "first visible row");
        assert!(!content.contains("101"), "scrolled-away rows must not render");
        assert!(!content.contains("105"));
        assert!(content.contains('▶'));
    }

    #[test]
    fn long_slot_list_renders_without_panic() {
        let state = connected_slots(|s| {
            s.slots = (0..200).map(|i| slot(i, false, Some(4_096))).collect();
        });
        let _ = render_content(&state, false, 80, 20);
        let _ = render_content(&state, true, 80, 20);
    }

    #[test]
    fn wide_width_shows_generated_column() {
        let state = connected_slots(|s| {
            s.slots = vec![slot(0, false, Some(16_384))];
        });
        // Inner width 106 >= 100: wide column set.
        let wide = render_content(&state, false, 108, 30);
        assert!(wide.contains("Generated"));
        // The standard (88-99 inner) set has no Generated column.
        let standard = render_content(&state, false, 90, 20);
        assert!(!standard.contains("Generated"));
        let compact = render_content(&state, false, 80, 20);
        assert!(!compact.contains("Generated"));
    }

    #[test]
    fn ascii_slot_table_uses_ascii_markers_and_placeholders() {
        let state = connected_slots(|s| {
            s.slots = vec![slot(7, false, None)]; // missing context -> placeholder
        });
        let ascii = render_content(&state, true, 80, 20);
        assert!(!ascii.contains('▶'));
        assert!(!ascii.contains('—'));
        assert!(ascii.contains('>'), "ASCII selection marker");
        let lines = split_rows(&ascii, 80);
        let row = lines.iter().find(|l| l.contains("7")).expect("row visible");
        assert!(row.contains('-'), "missing context renders '-' in the slot row");
    }

    #[test]
    fn missing_slot_values_render_placeholder_not_zero() {
        let state = connected_slots(|s| {
            s.slots = vec![slot(77, false, None)];
        });
        let content = render_content(&state, false, 80, 20);
        let lines = split_rows(&content, 80);
        let row = lines.iter().find(|l| l.contains("77")).expect("row visible");
        assert!(row.contains('—'), "unreported slot counters must show the em-dash, not 0");
    }

    #[test]
    fn paused_slot_table_shows_frozen_slots() {
        let mut state = connected_slots(|s| {
            s.slots = vec![slot(101, false, Some(1_000))];
        });
        state.handle_input(crate::app::event::InputAction::TogglePause);
        assert!(state.paused);
        // New data arrives while paused.
        let mut snap =
            BackendSnapshot { connection: ConnectionState::Connected, ..Default::default() };
        snap.slots = vec![slot(202, true, Some(2_000))];
        state.apply_snapshot(snap);
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("101"), "frozen slot must stay visible");
        assert!(!content.contains("202"));
        assert!(content.contains("PAUSED"));
    }

    #[test]
    fn footer_shows_slot_select_keys_when_slots_available() {
        let state = connected_slots(|s| {
            s.slots = vec![slot(0, false, None)];
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("↑/↓ Select"));
        let ascii = render_content(&state, true, 80, 20);
        assert!(ascii.contains("j/k Select"));
    }

    #[test]
    fn footer_hides_slot_select_keys_when_unavailable_or_empty() {
        // /slots endpoint unavailable: no select control.
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 80, 20);
        assert!(!content.contains("Select"));
        // Endpoint available but zero slots: still no select control.
        let state = connected_slots(|_| {});
        let content = render_content(&state, false, 80, 20);
        assert!(!content.contains("Select"));
    }

    #[test]
    fn slot_table_at_80x20_does_not_panic() {
        let state = connected_slots(|s| {
            s.slots = (0..5).map(|i| slot(i, i % 2 == 1, Some(32_768))).collect();
        });
        let _ = render_content(&state, false, 80, 20);
        let _ = render_content(&state, true, 80, 20);
        // Too-small and degenerate sizes keep their existing guarantees.
        let _ = render_content(&state, false, 62, 16);
        let _ = render_content(&state, false, 1, 1);
    }

    #[test]
    fn slot_table_at_80x20_shows_header_and_row_with_metrics_warning() {
        // /slots available, /metrics unavailable: the warning lines render
        // below the table and must not hide the table header or the slot
        // row at the minimum terminal size.
        let mut state = connected_ready(|s| {
            s.slots = vec![slot(137, true, Some(8_192))];
        });
        state.apply_capabilities(crate::backend::BackendCapabilities {
            slots: true,
            metrics: false,
            ..Default::default()
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Metrics unavailable"), "warning must render");
        assert!(!content.contains("Slots unavailable"), "capability is on");
        let lines = split_rows(&content, 80);
        assert!(
            lines.iter().any(|l| l.contains("ID")
                && l.contains("State")
                && l.contains("Phase")
                && l.contains("Context")),
            "the slot table header row must be visible at 80x20"
        );
        assert!(
            lines.iter().any(|l| l.contains("137") && l.contains("DECODE")),
            "the slot's row (ID + phase) must be visible at 80x20"
        );
    }
}
