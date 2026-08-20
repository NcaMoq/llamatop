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
}
