//! Terminal state management for the TUI.
//!
//! The TUI enables raw mode, enters the alternate screen, and hides the
//! cursor. All of it must be undone on every exit path: normal quit,
//! `q`, Ctrl+C, returned errors, channel/collector failures, and panics.
//!
//! Restoration is idempotent and never panics; it is performed by
//! [`TerminalGuard::restore`] (also called from `Drop`) and, on panics, by
//! the process-wide hook installed through [`PanicRestorer`].

use std::io;
use std::sync::{Arc, Mutex};

/// The terminal mode changes the guard is responsible for restoring.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TerminalModes {
    pub raw_mode: bool,
    pub alternate_screen: bool,
    pub cursor_hidden: bool,
}

/// The crossterm-backed terminal mode restoration. Ignores errors: the
/// goal is to leave the terminal usable, and this code must never panic.
pub fn restore_modes(modes: TerminalModes) {
    if modes.cursor_hidden {
        let _ = crossterm::execute!(std::io::stdout(), crossterm::cursor::Show);
    }
    if modes.alternate_screen {
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
    }
    if modes.raw_mode {
        let _ = crossterm::terminal::disable_raw_mode();
    }
}

/// Holds the terminal mode changes made for the TUI and restores them.
///
/// Restoration is idempotent: calling [`TerminalGuard::restore`] (or dropping
/// the guard) more than once is safe, and each mode is restored at most once.
pub struct TerminalGuard {
    modes: TerminalModes,
    restored: bool,
}

impl TerminalGuard {
    /// Enable raw mode, enter the alternate screen, and hide the cursor.
    ///
    /// On failure, whatever was already enabled is restored before the error
    /// is returned, so a failed start never leaves the terminal modified.
    pub fn new() -> io::Result<Self> {
        let mut guard = Self { modes: TerminalModes::default(), restored: false };
        let result = (|| -> io::Result<()> {
            crossterm::terminal::enable_raw_mode()?;
            guard.modes.raw_mode = true;
            crossterm::execute!(
                std::io::stdout(),
                crossterm::terminal::EnterAlternateScreen,
                crossterm::cursor::Hide
            )?;
            guard.modes.alternate_screen = true;
            guard.modes.cursor_hidden = true;
            Ok(())
        })();
        if result.is_err() {
            guard.restore();
        }
        result.map(|_| guard)
    }

    /// Restore the terminal. Safe to call repeatedly; never panics.
    pub fn restore(&mut self) {
        if self.restored {
            return;
        }
        self.restored = true;
        restore_modes(self.modes);
    }

    /// The modes currently held by the guard (for tests and diagnostics).
    pub fn modes(&self) -> TerminalModes {
        self.modes
    }

    /// True after [`TerminalGuard::restore`] has run.
    pub fn is_restored(&self) -> bool {
        self.restored
    }

    /// Construct a guard from explicit modes without touching the terminal
    /// (test support).
    #[doc(hidden)]
    pub fn from_modes_for_test(modes: TerminalModes) -> Self {
        Self { modes, restored: false }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

/// Process-wide slot for the modes of the currently active TUI. Exactly one
/// TUI runs at a time, so a single static slot is enough for the panic hook
/// (which is also process-wide).
static ACTIVE_TUI_MODES: Mutex<Option<TerminalModes>> = Mutex::new(None);

/// The process-wide panic hook signature (kept as an alias to avoid a very
/// complex inline type).
type PanicHook = Arc<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>;

/// RAII installation of a process-wide panic hook that restores the TUI
/// terminal before the panic information is shown.
///
/// The previous hook is kept and called after restoration so the panic
/// message and backtrace are not lost. Dropping the restorer clears the
/// active-modes slot and re-installs the previous hook, so the registration
/// cannot outlive the TUI run.
pub struct PanicRestorer {
    previous: Option<PanicHook>,
}

impl PanicRestorer {
    /// Install the hook. `modes` are the terminal modes currently held by
    /// the active guard.
    pub fn install(modes: TerminalModes) -> Self {
        {
            let mut slot =
                ACTIVE_TUI_MODES.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            *slot = Some(modes);
        }
        let previous = Arc::new(std::panic::take_hook());
        let hook_previous = Arc::clone(&previous);
        std::panic::set_hook(Box::new(move |info| {
            restore_active_modes();
            hook_previous(info);
        }));
        Self { previous: Some(previous) }
    }
}

/// Restore the modes registered by the active TUI (no-op when none).
fn restore_active_modes() {
    let mut slot = ACTIVE_TUI_MODES.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(modes) = slot.take() {
        restore_modes(modes);
    }
}

impl Drop for PanicRestorer {
    fn drop(&mut self) {
        let mut slot = ACTIVE_TUI_MODES.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *slot = None;
        if let Some(previous) = self.previous.take() {
            std::panic::set_hook(Box::new(move |info| previous(info)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_modes() -> TerminalModes {
        TerminalModes { raw_mode: true, alternate_screen: true, cursor_hidden: true }
    }

    #[test]
    fn restore_is_idempotent() {
        let mut guard = TerminalGuard::from_modes_for_test(full_modes());
        guard.restore();
        assert!(guard.is_restored());
        guard.restore(); // second call is a no-op
        assert!(guard.is_restored());
    }

    #[test]
    fn drop_restores_only_once() {
        let guard = TerminalGuard::from_modes_for_test(full_modes());
        drop(guard); // must not panic
    }

    #[test]
    fn default_modes_restore_nothing() {
        let mut guard = TerminalGuard::from_modes_for_test(TerminalModes::default());
        guard.restore();
        assert!(guard.is_restored());
        assert_eq!(guard.modes(), TerminalModes::default());
    }

    #[test]
    fn panic_restorer_restores_modes_and_previous_hook() {
        let restorer = PanicRestorer::install(full_modes());
        let result = std::panic::catch_unwind(|| panic!("test panic"));
        assert!(result.is_err());
        // The panic hook consumed the active modes.
        let slot = ACTIVE_TUI_MODES.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(slot.is_none());
        drop(slot);
        drop(restorer);
        // After the restorer drops, the slot stays clear and the hook is the
        // pre-TUI one (a panic now must not restore anything).
        let result = std::panic::catch_unwind(|| panic!("post-tui panic"));
        assert!(result.is_err());
    }
}
