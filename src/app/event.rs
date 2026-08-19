//! Event and command types for the TUI.
//!
//! The application task (see `runtime`) owns the `AppState` and consumes
//! `AppEvent`s from two producers:
//!
//! - the **input reader thread** (key actions and terminal resizes; the
//!   key-to-action mapping lives in `crate::ui::input`)
//! - the **backend collector task** (snapshots, capabilities, errors)
//!
//! The application sends `CollectorCommand`s back to the collector. The
//! collector never writes into the state directly.

use crate::backend::BackendCapabilities;
use crate::domain::BackendSnapshot;

/// Events flowing into the application task.
#[derive(Debug)]
pub enum AppEvent {
    /// A render tick fired (bounds the draw rate to 10 FPS).
    Tick,
    /// A new stabilized snapshot from the backend collector.
    BackendSnapshot(Box<BackendSnapshot>),
    /// Updated endpoint capabilities (initial probe or reconnect).
    BackendCapabilities(BackendCapabilities),
    /// A redacted backend error (no secrets, no response bodies).
    BackendError(BackendErrorSummary),
    /// A mapped keyboard action from the input reader.
    Input(InputAction),
    /// The terminal was resized to `(width, height)`.
    Resize(u16, u16),
}

/// Keyboard actions understood by the application. The mapping from raw key
/// events to these actions lives in `crate::ui::input`, keeping key
/// decisions out of the rendering code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputAction {
    /// `q` or Ctrl+C
    Quit,
    /// `r` — manual reconnect
    Reconnect,
    /// `p` — pause or resume the displayed snapshot
    TogglePause,
    /// `?` — open or close the help modal
    ToggleHelp,
    /// `Esc` — close the help modal or other modal
    CloseModal,
    /// `Tab` — focus the next panel
    FocusNext,
    /// `Shift+Tab` — focus the previous panel
    FocusPrev,
    /// `Up` / `k` — move slot selection up
    SlotUp,
    /// `Down` / `j` — move slot selection down
    SlotDown,
    /// `Enter` — open or close the selected slot details
    ToggleSlotDetail,
    /// `l` — toggle the event log panel
    ToggleEvents,
    /// `c` — clear the history
    ClearHistory,
}

/// Commands the application sends to the backend collector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectorCommand {
    /// Re-probe capabilities, reset detection state, and fetch immediately.
    /// Commands are processed serially, so duplicate reconnects cannot run
    /// concurrently.
    Reconnect,
    /// Stop collecting and exit the collector loop.
    Stop,
}

/// Short, redacted description of a backend error.
///
/// Must never contain API keys, authorization headers, prompts, completions,
/// or raw response bodies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendErrorSummary {
    pub message: String,
}

impl BackendErrorSummary {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}
