//! Terminal user interface for `llamatop`.
//!
//! The TUI is the only module that talks to the terminal. It renders from
//! application state and never performs HTTP; data will arrive through an
//! event channel (see the following steps for the event loop and collector).
//!
//! Step 2 (current): minimal terminal foundation — guard, panic restoration,
//! empty frame, `q`/Ctrl+C, resize. Panels and the event loop are added in
//! the following steps.

mod terminal;

use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::Frame;
use ratatui::Terminal as RatatuiTerminal;

use crate::config::Config;
use crate::display::Symbols;

pub use terminal::{PanicRestorer, TerminalGuard, TerminalModes};

/// Bounds idle CPU use: the loop wakes at least this often to age the
/// "last update" display and at most this often when idle.
const RENDER_TICK: Duration = Duration::from_millis(100);

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
    let result = minimal_loop(&mut term, &symbols);

    // Restore the terminal before anything else; the guard's Drop is the
    // second safety net (idempotent).
    guard.restore();
    drop(restorer);

    result
}

fn minimal_loop(
    term: &mut RatatuiTerminal<CrosstermBackend<std::io::Stdout>>,
    symbols: &Symbols,
) -> anyhow::Result<i32> {
    let mut size = (0u16, 0u16);
    let mut quit = false;

    while !quit {
        if event::poll(RENDER_TICK)? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') => quit = true,
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        quit = true;
                    }
                    _ => {}
                },
                Event::Resize(width, height) => size = (width, height),
                _ => {}
            }
        }

        term.draw(|f| {
            render_minimal(f, size, symbols);
        })?;
    }

    Ok(0)
}

/// Render the minimal placeholder frame. Safe for any size, including 0x0.
fn render_minimal(f: &mut Frame, size: (u16, u16), symbols: &Symbols) {
    if size.0 == 0 || size.1 == 0 {
        return;
    }
    let area = f.area();
    if area.width == 0 || area.height == 0 {
        return;
    }

    f.render_widget(
        ratatui::widgets::Block::bordered().title(format!(" {} llamatop ", symbols.active())),
        area,
    );
    f.render_widget(
        ratatui::widgets::Paragraph::new("llamatop — monitoring TUI. Press q to quit."),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_minimal_is_safe_at_zero_size() {
        let backend = ratatui::backend::TestBackend::new(1, 1);
        let mut term = RatatuiTerminal::new(backend).expect("terminal");
        term.draw(|f| render_minimal(f, (0, 0), &Symbols::new(false))).expect("draw");
    }

    #[test]
    fn render_minimal_draws_title_on_normal_size() {
        let backend = ratatui::backend::TestBackend::new(80, 20);
        let mut term = RatatuiTerminal::new(backend).expect("terminal");
        term.draw(|f| render_minimal(f, (80, 20), &Symbols::new(false))).expect("draw");
        let buf = term.backend().buffer();
        let content: String =
            (0..20).flat_map(|y| (0..80).map(move |x| buf[(x, y)].symbol().to_string())).collect();
        assert!(content.contains("llamatop"));
    }
}
