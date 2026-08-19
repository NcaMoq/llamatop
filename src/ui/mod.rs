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
mod terminal;

use ratatui::backend::CrosstermBackend;
use ratatui::Frame;
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
/// Returns the process exit code (0 for a clean quit). Terminal
/// initialization failure is the only `Err` case: the terminal state cannot
/// be trusted in that situation.
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
            term.draw(|f| render(f, state, &symbols)).map(|_| ())
        })
        .await;

    reader.stop();
    guard.restore();
    drop(restorer);

    result
}

/// Render the full TUI frame. Safe for any size, including 0x0.
fn render(f: &mut Frame, state: &AppState, symbols: &Symbols) {
    if state.terminal_size.0 == 0 || state.terminal_size.1 == 0 {
        return;
    }
    let area = f.area();
    if area.width == 0 || area.height == 0 {
        return;
    }

    let title = format!(
        " {} llamatop — {} ",
        symbols.active(),
        state
            .visible_snapshot()
            .map(|s| s.connection.as_str().to_string())
            .unwrap_or_else(|| symbols.idle().to_string())
    );
    let block = ratatui::widgets::Block::bordered().title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Minimal placeholder: will be replaced by panels in Step 4+.
    let status = state
        .visible_snapshot()
        .map(|s| {
            format!(
                "Server: {}  Phase: {}  Active: {}",
                s.server.as_str(),
                s.workload_phase.display(),
                s.active_requests.map(|v| v.to_string()).unwrap_or_else(|| "—".into())
            )
        })
        .unwrap_or_else(|| "Waiting for data...".to_string());

    let footer = " q Quit   r Reconnect   p Pause   l Logs   ? Help";
    if inner.height >= 2 {
        f.render_widget(
            ratatui::widgets::Paragraph::new(status),
            ratatui::layout::Rect::new(inner.x, inner.y, inner.width, 1),
        );
        f.render_widget(
            ratatui::widgets::Paragraph::new(footer).alignment(ratatui::layout::Alignment::Center),
            ratatui::layout::Rect::new(inner.x, inner.y + inner.height - 1, inner.width, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_is_safe_at_zero_size() {
        let backend = ratatui::backend::TestBackend::new(1, 1);
        let mut term = RatatuiTerminal::new(backend).expect("terminal");
        let state = AppState::new(&Config::default());
        term.draw(|f| render(f, &state, &Symbols::new(false))).expect("draw");
    }

    #[test]
    fn render_shows_waiting_when_no_snapshot() {
        let backend = ratatui::backend::TestBackend::new(80, 20);
        let mut term = RatatuiTerminal::new(backend).expect("terminal");
        let mut state = AppState::new(&Config::default());
        state.terminal_size = (80, 20);
        term.draw(|f| render(f, &state, &Symbols::new(false))).expect("draw");
        let buf = term.backend().buffer();
        let content: String =
            (0..20).flat_map(|y| (0..80).map(move |x| buf[(x, y)].symbol().to_string())).collect();
        assert!(content.contains("llamatop"));
        assert!(content.contains("Waiting"));
    }

    #[test]
    fn render_shows_snapshot_data() {
        let backend = ratatui::backend::TestBackend::new(80, 20);
        let mut term = RatatuiTerminal::new(backend).expect("terminal");
        let config = Config::default();
        let mut state = AppState::new(&config);
        state.terminal_size = (80, 20);
        state.apply_snapshot(crate::domain::BackendSnapshot {
            connection: crate::domain::ConnectionState::Connected,
            server: crate::domain::ServerState::Ready,
            active_requests: Some(3),
            ..Default::default()
        });
        term.draw(|f| render(f, &state, &Symbols::new(false))).expect("draw");
        let buf = term.backend().buffer();
        let content: String =
            (0..20).flat_map(|y| (0..80).map(move |x| buf[(x, y)].symbol().to_string())).collect();
        assert!(content.contains("CONNECTED"));
        assert!(content.contains("READY"));
    }
}
