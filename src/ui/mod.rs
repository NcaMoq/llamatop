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
    fn help_modal_is_hidden_by_default() {
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 100, 30);
        // The footer advertises "? Help", but the modal itself (close hint,
        // control rows) must not be present.
        assert!(!content.contains("Press ? or Esc to close"), "no modal by default");
        assert!(!content.contains("Manual reconnect"), "no help rows by default");
        assert!(!content.contains("Toggle event log"), "no help rows by default");
    }

    #[test]
    fn help_modal_shows_only_implemented_controls() {
        let mut state = connected_ready(|_| {});
        state.handle_input(crate::app::event::InputAction::ToggleHelp);
        assert!(state.show_help);
        let content = render_content(&state, false, 100, 30);
        assert!(content.contains("Help"), "modal title");
        // Every advertised control is implemented.
        assert!(content.contains("Quit"));
        assert!(content.contains("Manual reconnect"));
        assert!(content.contains("Pause / resume"));
        assert!(content.contains("Toggle event log"));
        assert!(content.contains("Scroll event log"));
        assert!(content.contains("Slot up"));
        assert!(content.contains("Close help / event log"));
        // Unimplemented features must NOT be advertised.
        assert!(!content.contains("Focus"), "panel focus is not implemented");
        assert!(!content.contains("Tab"), "Tab/Shift+Tab are unbound");
        assert!(!content.contains("Slot detail"), "slot detail is not implemented");
    }

    #[test]
    fn help_modal_ascii_mode_has_no_status_symbols() {
        let mut state = connected_ready(|_| {});
        state.handle_input(crate::app::event::InputAction::ToggleHelp);
        let content = render_content(&state, true, 100, 30);
        // The modal rows are plain ASCII; no Unicode status symbols appear.
        assert!(content.contains("Help"));
        assert!(!content.contains("●"));
        assert!(!content.contains("○"));
        assert!(!content.contains("▲"));
        assert!(!content.contains("✘"));
        assert!(!content.contains("—"));
    }

    #[test]
    fn help_modal_does_not_panic_at_small_sizes() {
        let mut state = connected_ready(|_| {});
        state.handle_input(crate::app::event::InputAction::ToggleHelp);
        let _ = render_content(&state, false, 80, 20);
        let _ = render_content(&state, false, 62, 16);
        let _ = render_content(&state, false, 1, 1);
    }

    #[test]
    fn footer_only_advertises_implemented_controls() {
        let state = AppState::new(&Config::default());
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("q Quit"));
        assert!(content.contains("r Reconnect"));
        // The help modal is available in every view.
        assert!(content.contains("? Help"));
        // Pause is not possible before the first snapshot, so it is not
        // advertised in the waiting view.
        assert!(!content.contains("p Pause"));
        // The event log only exists in the connected view, so it is not
        // advertised in the waiting view.
        assert!(!content.contains("l Events"));
    }

    #[test]
    fn footer_shows_events_toggle_when_connected() {
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("l Events"));
        // The waiting view must not advertise it.
        let waiting = AppState::new(&Config::default());
        let waiting_content = render_content(&waiting, false, 80, 20);
        assert!(!waiting_content.contains("l Events"));
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

    /// Connected/ready state with /slots and /metrics available.
    fn connected_slots(overrides: impl FnOnce(&mut BackendSnapshot)) -> AppState {
        let mut state = connected_ready(overrides);
        state.apply_capabilities(crate::backend::BackendCapabilities {
            slots: crate::backend::EndpointAvailability::Available,
            metrics: crate::backend::EndpointAvailability::Available,
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

    /// A connected state with a given /slots observation.
    fn state_with_slots_observation(state: crate::backend::EndpointAvailability) -> AppState {
        let mut s = connected_ready(|_| {});
        s.apply_capabilities(crate::backend::BackendCapabilities {
            slots: state,
            ..Default::default()
        });
        s
    }

    #[test]
    fn slots_unsupported_view_when_endpoint_missing() {
        let state = state_with_slots_observation(crate::backend::EndpointAvailability::Unsupported);
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Slots unsupported"));
        assert!(content.contains("--no-slots"));
        // Distinct from the zero-slots view and from a temporary failure.
        assert!(!content.contains("No slots reported"));
        assert!(!content.contains("temporarily unavailable"));
    }

    #[test]
    fn slots_temporarily_unavailable_view_is_distinct() {
        let state = state_with_slots_observation(
            crate::backend::EndpointAvailability::TemporarilyUnavailable,
        );
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Slots temporarily unavailable"));
        assert!(!content.contains("Slots unsupported"));
        // A temporary failure is never "no slots reported": the data is
        // missing, not empty.
        assert!(!content.contains("No slots reported"));
    }

    #[test]
    fn slots_parse_failed_view_is_distinct() {
        let state = state_with_slots_observation(crate::backend::EndpointAvailability::ParseFailed);
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Slots response could not be parsed"));
        assert!(!content.contains("Slots unsupported"));
        assert!(!content.contains("No slots reported"));
    }

    #[test]
    fn slots_authentication_failed_view_is_distinct() {
        let state = state_with_slots_observation(
            crate::backend::EndpointAvailability::AuthenticationFailed,
        );
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Slots authentication failed"));
        assert!(content.contains("API key"));
        assert!(!content.contains("No slots reported"));
    }

    #[test]
    fn slots_unknown_view_before_any_observation() {
        // Default capabilities: /slots has never been observed.
        let state = connected_ready(|_| {});
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Slots unavailable"));
        assert!(!content.contains("Slots unsupported"));
        assert!(!content.contains("No slots reported"));
    }

    #[test]
    fn no_slots_reported_view_when_empty() {
        let state = connected_slots(|_| {}); // /slots available, zero slots
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("No slots reported"));
        assert!(!content.contains("Slots unavailable"));
        assert!(!content.contains("Slots unsupported"));
        assert!(!content.contains("could not be parsed"));
    }

    /// A connected state with a given /metrics observation.
    fn state_with_metrics_observation(state: crate::backend::EndpointAvailability) -> AppState {
        let mut s = connected_ready(|_| {});
        s.apply_capabilities(crate::backend::BackendCapabilities {
            metrics: state,
            ..Default::default()
        });
        s
    }

    #[test]
    fn metrics_warning_shows_the_specific_state() {
        // Unsupported (server started without --metrics):
        let content = render_content(
            &state_with_metrics_observation(crate::backend::EndpointAvailability::Unsupported),
            false,
            80,
            20,
        );
        assert!(content.contains("Metrics endpoint not supported by the server"));
        assert!(content.contains("--metrics"));
        // Parse failure is a different message:
        let content = render_content(
            &state_with_metrics_observation(crate::backend::EndpointAvailability::ParseFailed),
            false,
            80,
            20,
        );
        assert!(content.contains("Metrics response could not be parsed"));
        assert!(!content.contains("not supported by the server"));
        // An available /metrics renders no warning at all:
        let content = render_content(
            &state_with_metrics_observation(crate::backend::EndpointAvailability::Available),
            false,
            80,
            20,
        );
        assert!(!content.contains("Metrics temporarily unavailable"));
        assert!(!content.contains("Metrics endpoint not supported"));
        assert!(!content.contains("Metrics response could not be parsed"));
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

    // --- Phase G: responsive layout at the spec's test sizes ---
    //
    // Every size must render without panicking and without a UTF-8 split.
    // `frame.area()` is the only source of truth, so these render directly.

    /// A connected/slots state carrying history samples plus (optionally) a
    /// host + GPU monitor, so the Resources panel can be exercised.
    fn responsive_state(
        samples: &[Sample],
        sys: Option<crate::domain::SystemSnapshot>,
        gpu: Option<crate::domain::GpuMonitor>,
    ) -> AppState {
        let mut state = state_with_history(samples);
        state.system = sys;
        state.gpu = gpu;
        state
    }

    fn two_gpu_monitor() -> crate::domain::GpuMonitor {
        crate::domain::GpuMonitor {
            status: crate::domain::GpuMonitorStatus::Available,
            gpus: vec![gpu(0, "NVIDIA RTX 5090"), gpu(1, "NVIDIA RTX 4090")],
        }
    }

    #[test]
    fn responsive_1x1_and_10x5_do_not_panic() {
        // The two smallest spec sizes cannot fit the fallback text, so they
        // only need to render without panicking or a UTF-8 split.
        let state = connected_slots(|s| s.slots = vec![slot(1, true, Some(4_096))]);
        for (w, h) in [(1, 1), (10, 5)] {
            let content = render_content(&state, false, w, h);
            assert!(!content.contains('\u{fffd}'), "{w}x{h} no UTF-8 split");
        }
        // 62x16 is above the panic guard but below the full layout, so the
        // too-small fallback text fits and must be shown.
        let content = render_content(&state, false, 62, 16);
        assert!(content.contains("Terminal is too small."));
        assert!(content.contains("Required: 80 x 20"));
    }

    #[test]
    fn responsive_80x20_keeps_header_inference_slots_footer() {
        let state = connected_slots(|s| {
            s.slots = vec![slot(137, true, Some(8_192)), slot(2, false, Some(4_096))];
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("LLAMATOP"), "header title");
        assert!(content.contains("Inference"), "inference panel");
        assert!(content.contains("Slots"), "slot table");
        assert!(content.contains("137") && content.contains("DECODE"), "a slot row");
        assert!(content.contains("q Quit"), "footer");
        assert!(content.contains("? Help"), "footer help hint");
        // At 80x20 the Resources panel and history are hidden (lowest
        // priority), so neither title may appear.
        assert!(!content.contains("Resources"));
        assert!(!content.contains("History"));
    }

    #[test]
    fn responsive_100x30_full_history_no_resources() {
        let state = responsive_state(&HISTORY_SAMPLES, None, None);
        let content = render_content(&state, false, 100, 30);
        // The full 9-row history renders its legend (only the full tier
        // emits one).
        assert!(content.contains("P 80.0"), "full history legend");
        // Resources is hidden at 100x30 (free=18 < 15+height).
        assert!(!content.contains("Resources"));
    }

    #[test]
    fn responsive_100x40_resources_and_full_history() {
        let state = responsive_state(&HISTORY_SAMPLES, Some(host_sample()), None);
        let content = render_content(&state, false, 100, 40);
        assert!(content.contains("Resources"), "Resources panel visible");
        assert!(content.contains("P 80.0"), "full history legend still fits");
    }

    #[test]
    fn responsive_120x40_two_gpus_and_full_history() {
        let state =
            responsive_state(&HISTORY_SAMPLES, Some(host_sample()), Some(two_gpu_monitor()));
        let content = render_content(&state, false, 120, 40);
        assert!(content.contains("Resources"));
        assert!(content.contains("NVIDIA RTX 5090"), "first GPU row");
        assert!(content.contains("NVIDIA RTX 4090"), "second GPU row");
        assert!(content.contains("P 80.0"), "full history legend still fits");
    }

    #[test]
    fn responsive_160x50_all_panels_and_full_history() {
        let state =
            responsive_state(&HISTORY_SAMPLES, Some(host_sample()), Some(two_gpu_monitor()));
        let content = render_content(&state, false, 160, 50);
        assert!(content.contains("LLAMATOP"), "header");
        assert!(content.contains("Inference"));
        assert!(content.contains("Resources"));
        assert!(content.contains("NVIDIA RTX 5090"));
        assert!(content.contains("P 80.0"), "full history legend");
        assert!(content.contains("q Quit"), "footer");
    }

    #[test]
    fn responsive_long_gpu_name_truncates_utf8_safe() {
        // A very long (CJK + latin) GPU name must truncate by display width
        // without splitting a character or panicking.
        let mut monitor = two_gpu_monitor();
        monitor.gpus[0].name =
            Some("NVIDIA 日本語-データセンター-モデル-テスト-名前-超長".to_string());
        let state = responsive_state(&HISTORY_SAMPLES, Some(host_sample()), Some(monitor));
        let content = render_content(&state, false, 120, 40);
        // The GPU row is present and the name was truncated (ellipsis or
        // cut), never overflowing the panel or leaving a split character.
        assert!(content.contains("GPU  0"), "the GPU row renders");
        // Every rendered character must be valid (String guarantees this);
        // assert no replacement char from a bad split.
        assert!(!content.contains('\u{fffd}'), "no UTF-8 replacement char");
    }

    #[test]
    fn responsive_80x20_ascii_no_panics() {
        let state =
            responsive_state(&HISTORY_SAMPLES, Some(host_sample()), Some(two_gpu_monitor()));
        let _ = render_content(&state, true, 80, 20);
        let _ = render_content(&state, true, 100, 30);
        let _ = render_content(&state, true, 160, 50);
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

    /// One history sample for tests: (prompt, generation, active, queued).
    type Sample = (Option<f64>, Option<f64>, Option<u64>, Option<u64>);

    /// State with the given (prompt, generation, active, queued) samples
    /// already recorded in the history.
    fn state_with_history(samples: &[Sample]) -> AppState {
        let mut state = connected_slots(|_| {});
        // Drop the sample recorded by the connected-ready setup so the
        // window contains exactly the samples under test.
        state.history.clear();
        for (p, g, a, q) in samples {
            let snap = BackendSnapshot {
                connection: ConnectionState::Connected,
                prompt_tokens_per_second: *p,
                generation_tokens_per_second: *g,
                active_requests: *a,
                queued_requests: *q,
                ..Default::default()
            };
            state.history.record(&snap, std::time::Instant::now());
        }
        state
    }

    const HISTORY_SAMPLES: [Sample; 8] = [
        (Some(10.0), Some(1.0), Some(1), Some(0)),
        (Some(20.0), Some(2.0), Some(2), Some(1)),
        (Some(30.0), None, Some(1), None),
        (Some(40.0), Some(4.0), Some(4), Some(2)),
        (Some(50.0), Some(5.0), Some(3), Some(0)),
        (Some(60.0), Some(6.0), Some(2), Some(1)),
        (Some(70.0), Some(7.0), Some(1), Some(0)),
        (Some(80.0), Some(8.0), Some(1), Some(1)),
    ];

    #[test]
    fn history_panel_renders_with_samples() {
        let state = state_with_history(&HISTORY_SAMPLES);
        let content = render_content(&state, false, 100, 30);
        let lines = split_rows(&content, 100);
        assert!(
            lines.iter().any(|l| l.contains("History")),
            "panel title must render in the wide layout"
        );
        // The legend row is unique to the history panel; it must show the
        // latest values (80.0 / 8.0 / 1 / 1) with their P/G/A/Q labels.
        let legend = lines.iter().find(|l| l.contains("P 80.0")).expect("history legend row");
        assert!(legend.contains("G 8.0"));
        assert!(legend.contains("A 1"));
        assert!(legend.contains("Q 1"));
        // Series labels are text, not colors only.
        assert!(lines.iter().any(|l| l.contains("Prompt")));
        assert!(lines.iter().any(|l| l.contains("Gen     ")));
        assert!(lines.iter().any(|l| l.contains("Active  ")));
        assert!(lines.iter().any(|l| l.contains("Queued  ")));
    }

    #[test]
    fn history_panel_all_missing_values_show_placeholder_not_zero() {
        let state = state_with_history(&[(None, None, None, None), (None, None, None, None)]);
        let content = render_content(&state, false, 100, 30);
        // Legend placeholders, never "0.0".
        assert!(content.contains("P —"));
        assert!(content.contains("G —"));
        // No sparkline glyph may appear for an all-missing window.
        assert!(!content.contains('▁'));
        assert!(!content.contains('█'));
    }

    #[test]
    fn history_panel_mixed_missing_values_render_gaps() {
        let state = state_with_history(&[
            (Some(10.0), Some(1.0), Some(1), None),
            (None, None, None, None),
            (Some(20.0), Some(2.0), Some(2), Some(1)),
        ]);
        let content = render_content(&state, false, 100, 30);
        // Present values draw glyphs; the middle sample is a gap (blank).
        assert!(content.contains('▁') || content.contains('█'), "glyphs for present samples");
        let lines = split_rows(&content, 100);
        // The history Active row is the only "Active" row carrying sparkline
        // glyphs (the inference panel row has digits, not glyphs).
        let active_row = lines
            .iter()
            .find(|l| l.contains("Active") && l.contains('▄'))
            .expect("history Active row");
        // glyph (1.0), blank gap (missing sample), glyph (2.0)
        let chars: Vec<char> = active_row.chars().collect();
        let i = chars.iter().position(|&c| c == '▄').expect("first glyph");
        assert_eq!(
            &chars[i..i + 3],
            &['▄', ' ', '█'],
            "the missing middle sample must leave a blank column: {active_row}"
        );
    }

    #[test]
    fn history_panel_single_sample_renders() {
        let state = state_with_history(&[(Some(5.0), Some(1.5), Some(1), Some(0))]);
        let content = render_content(&state, false, 100, 30);
        let lines = split_rows(&content, 100);
        assert!(lines.iter().any(|l| l.contains("Prompt")));
        assert!(content.contains("P 5.0"));
        assert!(content.contains("G 1.5"));
        assert!(!content.contains("No recent data"), "a sample exists");
    }

    #[test]
    fn history_panel_empty_shows_no_recent_data() {
        // A fresh state with no recorded samples at all.
        let mut state = connected_slots(|_| {});
        state.history.clear();
        let content = render_content(&state, false, 100, 30);
        assert!(content.contains("No recent data"));
    }

    #[test]
    fn history_panel_capacity_sized_samples_do_not_panic() {
        // Capacity of the default config (120s @ 500ms) is 240; record all
        // of them — the panel must window down to the plot width.
        let samples: Vec<Sample> = (0..240)
            .map(|i| (Some(i as f64), Some(i as f64 / 10.0), Some(i % 4), Some(i % 2)))
            .collect();
        let state = state_with_history(&samples);
        let _ = render_content(&state, false, 100, 30);
        let _ = render_content(&state, false, 80, 20);
    }

    #[test]
    fn history_panel_does_not_panic_at_small_sizes() {
        let state = state_with_history(&HISTORY_SAMPLES);
        // 80x20: slots keep their full table (lowest priority is history),
        // so the history panel is hidden — without a panic.
        let content = render_content(&state, false, 80, 20);
        assert!(!content.contains("History"), "history is hidden at 80x20");
        // 1x1 and 62x16 fall back to the too-small view without panicking.
        let _ = render_content(&state, false, 1, 1);
        let _ = render_content(&state, false, 62, 16);
    }

    #[test]
    fn history_panel_ascii_mode_has_no_unicode_glyphs() {
        let state = state_with_history(&HISTORY_SAMPLES);
        let ascii = render_content(&state, true, 100, 30);
        assert!(!ascii.contains('▁'));
        assert!(!ascii.contains('█'));
        assert!(!ascii.contains('▀'));
        assert!(!ascii.contains('▄'));
        assert!(ascii.contains("Prompt"));
    }

    #[test]
    fn history_panel_large_rates_do_not_break_layout() {
        let state = state_with_history(&[
            (Some(1.0), Some(1.0), Some(1), Some(1)),
            (Some(1e12), Some(1e12), Some(1_000_000), Some(1)),
        ]);
        let content = render_content(&state, false, 100, 30);
        let lines = split_rows(&content, 100);
        assert!(
            lines.iter().any(|l| l.contains("Prompt")),
            "the panel must survive extreme values"
        );
    }

    #[test]
    fn history_zero_rate_is_distinguishable_from_missing() {
        // A true 0 sample draws the lowest ramp glyph; a missing sample
        // leaves a blank column (sparkline rows, wide layout).
        let state = state_with_history(&[
            (Some(1.0), Some(1.0), Some(0), None),
            (Some(2.0), Some(2.0), None, Some(1)),
        ]);
        let content = render_content(&state, false, 100, 30);
        let lines = split_rows(&content, 100);
        // History sparkline rows are the only "Active"/"Queued" rows with
        // ramp glyphs; the inference row shows digits.
        let active_row = lines
            .iter()
            .find(|l| l.contains("Active") && l.contains('▁'))
            .expect("history Active row");
        let chars: Vec<char> = active_row.chars().collect();
        let i = chars.iter().position(|&c| c == '▁').expect("zero glyph");
        assert_eq!(
            &chars[i..i + 2],
            &['▁', ' '],
            "zero count = lowest glyph, missing = blank: {active_row}"
        );
        let queued_row = lines
            .iter()
            .find(|l| l.contains("Queued") && l.contains('█'))
            .expect("history Queued row");
        assert!(queued_row.contains('█'), "a present count draws the highest glyph: {queued_row}");
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
            slots: crate::backend::EndpointAvailability::Available,
            metrics: crate::backend::EndpointAvailability::TemporarilyUnavailable,
            ..Default::default()
        });
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("Metrics temporarily unavailable"), "warning must render");
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

    // --- Step 7: event log panel tests ---

    use crate::app::log::{EventKind, EventSeverity};

    /// Connected/slots state with a controlled, non-empty event log and the
    /// event panel toggled visible.
    fn state_with_events(events: &[(EventSeverity, EventKind, &str)]) -> AppState {
        let mut state = connected_slots(|_| {});
        state.events.clear();
        for (sev, kind, msg) in events {
            state.events.push(*sev, *kind, *msg);
        }
        state.handle_input(crate::app::event::InputAction::ToggleEvents);
        state
    }

    #[test]
    fn event_panel_renders_when_toggled() {
        let state = state_with_events(&[
            (EventSeverity::Info, EventKind::Connected, "Connected"),
            (EventSeverity::Warning, EventKind::MetricsUnavailable, "Metrics unavailable"),
            (EventSeverity::Error, EventKind::Disconnected, "Connection lost"),
        ]);
        let content = render_content(&state, false, 100, 30);
        assert!(
            content.contains("PgUp/PgDn"),
            "the event panel title (with its scroll hint) must render"
        );
        // Newest at the bottom: all three messages are visible.
        assert!(content.contains("Connected"));
        assert!(content.contains("Metrics unavailable"));
        assert!(content.contains("Connection lost"));
        // The newest record must sit below the oldest in the panel.
        let lines = split_rows(&content, 100);
        let p_first = lines
            .iter()
            .position(|l| l.contains("Connected") && !l.contains("PgUp/PgDn"))
            .expect("oldest event row");
        let p_last =
            lines.iter().position(|l| l.contains("Connection lost")).expect("newest event row");
        assert!(p_first < p_last, "newest event must render below the oldest");
    }

    #[test]
    fn event_panel_is_hidden_by_default() {
        let mut state = connected_slots(|_| {});
        state.events.clear();
        state.events.push(EventSeverity::Info, EventKind::Connected, "Connected");
        assert!(!state.show_events, "events start hidden");
        let content = render_content(&state, false, 100, 30);
        assert!(
            !content.contains("PgUp/PgDn"),
            "the event panel must not render until toggled with `l`"
        );
        // The history panel still occupies the detail slot instead.
        assert!(content.contains("History"));
    }

    #[test]
    fn event_panel_shows_repeat_count_suffix() {
        let mut state = connected_slots(|_| {});
        state.events.clear();
        // Push the identical event three times: collapses to one record x3.
        for _ in 0..3 {
            state.events.push(
                EventSeverity::Warning,
                EventKind::MetricsUnavailable,
                "Metrics unavailable",
            );
        }
        state.handle_input(crate::app::event::InputAction::ToggleEvents);
        let content = render_content(&state, false, 100, 30);
        assert!(content.contains(" x3"), "repeated events must show xN");
        // A single (non-repeated) event shows no suffix.
        let mut one = connected_slots(|_| {});
        one.events.clear();
        one.events.push(EventSeverity::Info, EventKind::Connected, "Connected");
        one.handle_input(crate::app::event::InputAction::ToggleEvents);
        let one_content = render_content(&one, false, 100, 30);
        assert!(!one_content.contains(" x1"));
    }

    #[test]
    fn event_panel_empty_shows_no_events() {
        let mut state = connected_slots(|_| {});
        state.events.clear();
        state.handle_input(crate::app::event::InputAction::ToggleEvents);
        let content = render_content(&state, false, 100, 30);
        assert!(content.contains("No events yet"));
    }

    #[test]
    fn event_scroll_reveals_older_events() {
        // Five distinct events; the viewport at 100x30 fits ~12 rows, so all
        // are visible at offset 0. Scrolling up by 4 drops the four newest.
        let state = {
            let mut s = connected_slots(|_| {});
            s.events.clear();
            for i in 1..=5 {
                s.events.push(EventSeverity::Info, EventKind::Connected, format!("event {i}"));
            }
            s.handle_input(crate::app::event::InputAction::ToggleEvents);
            s.handle_input(crate::app::event::InputAction::LogEnd);
            s
        };
        let content = render_content(&state, false, 100, 30);
        // Scrolled to the oldest: only event 1 remains visible; the newest
        // (event 5) has scrolled off the bottom.
        assert!(content.contains("event 1"), "the oldest event is visible");
        assert!(!content.contains("event 5"), "the newest event scrolled away");
    }

    #[test]
    fn event_panel_does_not_panic_at_small_sizes() {
        let state = state_with_events(&[
            (EventSeverity::Info, EventKind::Connected, "Connected"),
            (EventSeverity::Error, EventKind::Disconnected, "Connection lost"),
        ]);
        // 80x20 keeps the slot table header + rows; the event log takes the
        // remaining free space (>= 1 row) without a panic.
        let content = render_content(&state, false, 80, 20);
        assert!(content.contains("PgUp/PgDn"), "event log still fits at 80x20");
        // Degenerate sizes fall back without panicking.
        let _ = render_content(&state, false, 1, 1);
        let _ = render_content(&state, false, 62, 16);
    }

    #[test]
    fn event_panel_ascii_mode_is_ascii_only() {
        let state = state_with_events(&[
            (EventSeverity::Info, EventKind::Connected, "Connected"),
            (EventSeverity::Warning, EventKind::MetricsUnavailable, "Metrics unavailable"),
        ]);
        let ascii = render_content(&state, true, 100, 30);
        assert!(ascii.contains("PgUp/PgDn"));
        // No Unicode severity symbols in ASCII mode.
        assert!(!ascii.contains('○'));
        assert!(!ascii.contains('▲'));
        assert!(!ascii.contains('✘'));
    }

    // --- Step 8: Resources panel tests ---

    use crate::domain::{ProcessAssociation, ProcessSnapshot};

    /// Connected/slots state with a controlled host + process sample.
    fn state_with_system(sys: crate::domain::SystemSnapshot) -> AppState {
        let mut state = connected_slots(|_| {});
        state.show_system = true;
        state.system = Some(sys);
        state
    }

    #[test]
    fn resources_panel_single_candidate_shows_unverified() {
        let state = state_with_system(crate::domain::SystemSnapshot {
            cpu_usage_percent: Some(42.0),
            ram_used_bytes: Some(16_000_000_000),
            ram_total_bytes: Some(64_000_000_000),
            process_match_count: Some(1),
            process: Some(ProcessSnapshot {
                pid: 6348,
                name: "llama-server.exe".into(),
                cpu_usage_percent: Some(9.0),
                memory_bytes: Some(29_650_837_504),
                uptime_secs: Some(8372),
            }),
            association: ProcessAssociation::SingleLocalCandidate,
        });
        // 120x40: the candidate row is long, so a wider terminal keeps it
        // whole (the 100x40 Resources panel inner width would clip it).
        let content = render_content(&state, false, 120, 40);
        assert!(content.contains("Resources"), "panel title must render");
        assert!(content.contains("CPU 42.0%"), "host CPU shown");
        assert!(content.contains("14.9G/59.6G"), "RAM used/total compact");
        assert!(content.contains("llama-server.exe"), "the candidate process is named");
        assert!(content.contains("PID 6348"), "candidate PID shown");
        assert!(content.contains("27.6G"), "process memory compact (GiB)");
        assert!(content.contains("2 h"), "uptime human-formatted");
        // A name match must not be presented as a verified association.
        assert!(content.contains("endpoint not verified"));
    }

    #[test]
    fn resources_panel_multiple_candidates_do_not_name_process() {
        let state = state_with_system(crate::domain::SystemSnapshot {
            cpu_usage_percent: Some(10.0),
            ram_used_bytes: Some(1_000),
            ram_total_bytes: Some(2_000),
            process_match_count: Some(2),
            process: None,
            association: ProcessAssociation::MultipleLocalCandidates,
        });
        let content = render_content(&state, false, 100, 40);
        assert!(content.contains("2 local llama-server candidates"));
        assert!(content.contains("not associated"));
    }

    #[test]
    fn resources_panel_remote_endpoint_notes_unavailable() {
        let state = state_with_system(crate::domain::SystemSnapshot {
            cpu_usage_percent: Some(10.0),
            ram_used_bytes: Some(1_000),
            ram_total_bytes: Some(2_000),
            process_match_count: None,
            process: None,
            association: ProcessAssociation::RemoteEndpoint,
        });
        let content = render_content(&state, false, 100, 40);
        assert!(content.contains("Remote endpoint"));
        assert!(content.contains("local process association unavailable"));
    }

    #[test]
    fn resources_panel_hidden_when_disabled() {
        let mut state = connected_slots(|_| {});
        state.show_system = false;
        state.show_gpu = false;
        state.system = Some(crate::domain::SystemSnapshot {
            cpu_usage_percent: Some(5.0),
            ram_used_bytes: None,
            ram_total_bytes: None,
            process_match_count: Some(1),
            process: None,
            association: ProcessAssociation::SingleLocalCandidate,
        });
        let content = render_content(&state, false, 100, 40);
        assert!(!content.contains("Resources"), "panel hidden when disabled");
    }

    #[test]
    fn resources_panel_unavailable_shows_placeholder_note() {
        let mut state = connected_slots(|_| {});
        state.show_system = true;
        state.system = None; // monitor not yet sampled / unavailable
        let content = render_content(&state, false, 100, 40);
        assert!(content.contains("Resources"));
        assert!(content.contains("System monitor unavailable"));
    }

    #[test]
    fn resources_panel_missing_values_use_placeholder_not_zero() {
        let state = state_with_system(crate::domain::SystemSnapshot {
            cpu_usage_percent: None,
            ram_used_bytes: Some(1_000),
            ram_total_bytes: None, // total missing -> placeholder
            process_match_count: Some(0),
            process: None,
            association: ProcessAssociation::NoneFound,
        });
        let content = render_content(&state, false, 100, 40);
        // CPU missing -> placeholder, RAM missing total -> placeholder.
        assert!(content.contains("—"), "missing values must be placeholders");
        assert!(content.contains("llama-server process not found"));
        // A placeholder must never be rendered as a fabricated 0.0%.
        assert!(!content.contains("CPU 0.0%"));
    }

    #[test]
    fn resources_panel_does_not_panic_at_small_sizes() {
        let state = state_with_system(crate::domain::SystemSnapshot {
            cpu_usage_percent: Some(1.0),
            ram_used_bytes: Some(1),
            ram_total_bytes: Some(2),
            process_match_count: Some(1),
            process: None,
            association: ProcessAssociation::SingleLocalCandidate,
        });
        let _ = render_content(&state, false, 80, 20);
        let _ = render_content(&state, false, 1, 1);
        let _ = render_content(&state, false, 62, 16);
    }

    /// A full GPU snapshot with the given index and name.
    fn gpu(index: u32, name: &str) -> crate::domain::GpuSnapshot {
        crate::domain::GpuSnapshot {
            index,
            uuid: Some(format!("GPU-uuid-{index}")),
            name: Some(name.to_string()),
            utilization_percent: Some(42),
            memory_used_bytes: Some(1 << 30),
            memory_total_bytes: Some(1 << 34),
            temperature_celsius: Some(55),
            power_watts: Some(120.0),
            power_limit_watts: Some(350.0),
            graphics_clock_mhz: Some(2500),
            memory_clock_mhz: Some(8000),
        }
    }

    /// Connected/slots state with a controlled host sample + GPU monitor.
    fn state_with_gpu(
        sys: crate::domain::SystemSnapshot,
        gpu: crate::domain::GpuMonitor,
    ) -> AppState {
        let mut state = connected_slots(|_| {});
        state.show_system = true;
        state.show_gpu = true;
        state.system = Some(sys);
        state.gpu = Some(gpu);
        state
    }

    fn host_sample() -> crate::domain::SystemSnapshot {
        crate::domain::SystemSnapshot {
            cpu_usage_percent: Some(10.0),
            ram_used_bytes: Some(10_000),
            ram_total_bytes: Some(20_000),
            process_match_count: Some(0),
            process: None,
            association: ProcessAssociation::NoneFound,
        }
    }

    #[test]
    fn gpu_section_renders_a_row_per_device() {
        let state = state_with_gpu(
            host_sample(),
            crate::domain::GpuMonitor {
                status: crate::domain::GpuMonitorStatus::Available,
                gpus: vec![gpu(0, "NVIDIA RTX 5090"), gpu(1, "NVIDIA RTX 4090")],
            },
        );
        let content = render_content(&state, false, 100, 40);
        assert!(content.contains("NVIDIA RTX 5090"), "first GPU named");
        assert!(content.contains("NVIDIA RTX 4090"), "second GPU named");
        assert!(content.contains("42%"), "utilization shown");
        assert!(content.contains("1.0G/16.0G"), "VRAM used/total compact");
        assert!(content.contains("55C"), "temperature shown");
        assert!(content.contains("120/350W"), "power/limit shown");
        // The GPU rows must never claim the device belongs to llama-server.
        assert!(!content.contains("llama-server GPU"));
        assert!(!content.contains("server GPU"));
    }

    #[test]
    fn gpu_section_unavailable_note() {
        let state = state_with_gpu(
            host_sample(),
            crate::domain::GpuMonitor {
                status: crate::domain::GpuMonitorStatus::Unavailable,
                gpus: Vec::new(),
            },
        );
        let content = render_content(&state, false, 100, 40);
        assert!(content.contains("GPU monitoring unavailable"));
    }

    #[test]
    fn gpu_section_initialization_failed_note() {
        let state = state_with_gpu(
            host_sample(),
            crate::domain::GpuMonitor {
                status: crate::domain::GpuMonitorStatus::InitializationFailed,
                gpus: Vec::new(),
            },
        );
        let content = render_content(&state, false, 100, 40);
        assert!(content.contains("failed to initialize (NVML)"));
    }

    #[test]
    fn gpu_section_sampling_failed_note() {
        let state = state_with_gpu(
            host_sample(),
            crate::domain::GpuMonitor {
                status: crate::domain::GpuMonitorStatus::SamplingFailed,
                gpus: Vec::new(),
            },
        );
        let content = render_content(&state, false, 100, 40);
        assert!(content.contains("GPU sampling failed"));
    }

    #[test]
    fn gpu_row_missing_values_use_placeholder_not_zero() {
        let mut g = gpu(0, "NVIDIA RTX 5090");
        g.utilization_percent = None;
        g.memory_total_bytes = None;
        g.temperature_celsius = None;
        g.power_watts = None;
        g.name = None;
        let state = state_with_gpu(
            host_sample(),
            crate::domain::GpuMonitor {
                status: crate::domain::GpuMonitorStatus::Available,
                gpus: vec![g],
            },
        );
        let content = render_content(&state, false, 100, 40);
        // The GPU row renders with placeholders for every missing value.
        // Scoped to the "GPU  0" prefix so the host row ("CPU 10.0%") cannot
        // satisfy or break the checks.
        assert!(content.contains("GPU  0   —   —"), "missing values use the placeholder");
        assert!(!content.contains("GPU  0   0%"), "no fabricated 0% utilization");
        assert!(!content.contains("GPU  0   0C"), "no fabricated 0C temperature");
    }

    #[test]
    fn gpu_section_hidden_when_disabled() {
        let mut state = state_with_gpu(
            host_sample(),
            crate::domain::GpuMonitor {
                status: crate::domain::GpuMonitorStatus::Available,
                gpus: vec![gpu(0, "NVIDIA RTX 5090")],
            },
        );
        state.show_gpu = false;
        let content = render_content(&state, false, 100, 40);
        assert!(!content.contains("NVIDIA RTX 5090"), "no GPU rows when disabled");
        assert!(!content.contains("GPU monitoring"), "no GPU status note when disabled");
    }
}
