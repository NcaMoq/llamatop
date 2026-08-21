//! The TUI application layer: state, events, history, and the runtime that
//! owns the state and consumes events.
//!
//! Layering: `app` depends on `domain`, `backend` (traits only), `config`,
//! and `detector` — never on ratatui/crossterm (those live in `ui`), and it
//! performs no HTTP itself (the collector does).

pub mod event;
pub mod history;
pub mod log;
pub mod runtime;
pub mod state;

pub use event::{AppEvent, BackendErrorSummary, CollectorCommand, InputAction};
pub use history::{History, HistorySample, MAX_HISTORY_SAMPLES};
pub use log::{EventKind, EventLog, EventRecord, EventSeverity};
pub use state::AppState;
