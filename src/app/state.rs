//! The TUI application state.
//!
//! A single task (see `crate::app::runtime`) owns this struct; no `Arc<Mutex<..>>`
//! sharing is involved. Backend data arrives through `AppEvent`s, keyboard
//! actions through `InputAction`, and the state produces the visible
//! snapshot, bounded history, and bounded event log.
//!
//! Display state (selection, focus, pause, modals) is kept strictly
//! separate from backend state (the latest snapshot).

use std::collections::VecDeque;
use std::time::Instant;

use crate::app::event::{BackendErrorSummary, InputAction};
use crate::app::history::History;
use crate::backend::BackendCapabilities;
use crate::config::Config;
use crate::domain::{BackendSnapshot, ConnectionState};

/// Maximum number of retained log events (bounded, per spec).
pub const MAX_EVENTS: usize = 200;

/// One retained log line. Messages are plain, redacted text: never prompt,
/// completion, API key, or authorization header content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppLogEvent {
    pub at: Instant,
    pub message: String,
}

/// Panels the focus cycle (Tab / Shift+Tab) moves between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FocusedPanel {
    /// The slots table (default focus).
    #[default]
    Slots,
    /// The history panel.
    History,
    /// The event log panel.
    Events,
}

impl FocusedPanel {
    const ALL: [FocusedPanel; 3] =
        [FocusedPanel::Slots, FocusedPanel::History, FocusedPanel::Events];

    pub fn next(self) -> Self {
        Self::ALL[(Self::ALL.iter().position(|p| *p == self).unwrap_or(0) + 1) % Self::ALL.len()]
    }

    pub fn prev(self) -> Self {
        let idx = Self::ALL.iter().position(|p| *p == self).unwrap_or(0);
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

/// The single owner of all TUI state.
pub struct AppState {
    // --- static configuration (display only, set once) ---
    pub endpoint: String,
    pub api_key_env: String,

    // --- backend data (updated by collector events) ---
    pub latest: Option<BackendSnapshot>,
    pub capabilities: BackendCapabilities,
    pub connection_message: Option<String>,
    /// Set while the last reported backend error was an authentication
    /// failure (the view then offers the env-var hint instead of a retry).
    pub authentication_failed: bool,

    // --- bounded logs ---
    pub events: VecDeque<AppLogEvent>,
    pub history: History,

    // --- display state ---
    pub selected_slot: usize,
    pub slot_detail_open: bool,
    pub focused_panel: FocusedPanel,
    pub paused: bool,
    pub frozen_snapshot: Option<BackendSnapshot>,
    pub show_help: bool,
    pub show_events: bool,
    pub terminal_size: (u16, u16),
    pub should_quit: bool,

    // --- internals ---
    last_update: Option<Instant>,
    reconnect_requested: bool,
    previous_connection: Option<ConnectionState>,
    previous_server_start: Option<u64>,
}

impl AppState {
    pub fn new(config: &Config) -> Self {
        let history = History::for_config(config.history_seconds, config.refresh_interval_ms);
        Self {
            endpoint: config.endpoint.clone(),
            api_key_env: config.authentication.api_key_env.clone(),
            latest: None,
            capabilities: BackendCapabilities::default(),
            connection_message: None,
            authentication_failed: false,
            events: VecDeque::with_capacity(MAX_EVENTS),
            history,
            selected_slot: 0,
            slot_detail_open: false,
            focused_panel: FocusedPanel::default(),
            paused: false,
            frozen_snapshot: None,
            show_help: false,
            show_events: false,
            terminal_size: (0, 0),
            should_quit: false,
            last_update: None,
            reconnect_requested: false,
            previous_connection: None,
            previous_server_start: None,
        }
    }

    /// The snapshot currently visible: the frozen one while paused,
    /// otherwise the latest backend snapshot.
    pub fn visible_snapshot(&self) -> Option<&BackendSnapshot> {
        if self.paused {
            self.frozen_snapshot.as_ref()
        } else {
            self.latest.as_ref()
        }
    }

    /// Seconds since the latest backend snapshot arrived (None if never).
    pub fn last_update_age(&self, now: Instant) -> Option<u64> {
        self.last_update.map(|t| now.duration_since(t).as_secs())
    }

    /// Milliseconds since the latest backend snapshot arrived (None if never).
    /// Used for the header's sub-second "updated N ms ago" display.
    pub fn last_update_age_ms(&self, now: Instant) -> Option<u64> {
        self.last_update.map(|t| now.duration_since(t).as_millis() as u64)
    }

    /// Whether the collector should reconnect after this event round.
    pub fn take_reconnect_requested(&mut self) -> bool {
        std::mem::take(&mut self.reconnect_requested)
    }

    pub fn log(&mut self, message: impl Into<String>) {
        self.events.push_back(AppLogEvent { at: Instant::now(), message: message.into() });
        while self.events.len() > MAX_EVENTS {
            self.events.pop_front();
        }
    }

    /// Handle a render tick: keeps the state current without data changes
    /// (e.g. clamping the slot selection is done on snapshot updates; the
    /// tick exists to bound the draw rate and age the UI).
    pub fn on_tick(&mut self) {}

    /// Apply a keyboard action.
    pub fn handle_input(&mut self, action: InputAction) {
        // While a modal is open, only modal-closing and quit actions apply.
        if self.show_help
            && !matches!(
                action,
                InputAction::Quit | InputAction::ToggleHelp | InputAction::CloseModal
            )
        {
            return;
        }

        match action {
            InputAction::Quit => self.should_quit = true,
            InputAction::Reconnect => {
                self.reconnect_requested = true;
                self.log("Manual reconnect requested");
            }
            InputAction::TogglePause => self.toggle_pause(),
            InputAction::ToggleHelp => self.show_help = !self.show_help,
            InputAction::CloseModal => {
                if self.show_help {
                    self.show_help = false;
                } else if self.slot_detail_open {
                    self.slot_detail_open = false;
                } else if self.show_events {
                    self.show_events = false;
                }
            }
            InputAction::FocusNext => self.focused_panel = self.focused_panel.next(),
            InputAction::FocusPrev => self.focused_panel = self.focused_panel.prev(),
            InputAction::SlotUp => {
                if self.selected_slot > 0 {
                    self.selected_slot -= 1;
                }
            }
            InputAction::SlotDown => {
                let count = self.latest.as_ref().map(|s| s.slots.len()).unwrap_or(0);
                if self.selected_slot + 1 < count {
                    self.selected_slot += 1;
                }
            }
            InputAction::ToggleSlotDetail => self.slot_detail_open = !self.slot_detail_open,
            InputAction::ToggleEvents => self.show_events = !self.show_events,
            InputAction::ClearHistory => {
                self.history.clear();
                self.log("History cleared");
            }
        }
    }

    fn toggle_pause(&mut self) {
        if self.paused {
            self.paused = false;
            self.frozen_snapshot = None;
            self.log("Resume: showing latest snapshot");
        } else if self.latest.is_none() {
            // No snapshot to freeze yet: ignoring the request keeps the
            // "Waiting for data..." view from sticking after the first
            // snapshot arrives.
            self.log("Pause ignored: no data yet");
        } else {
            self.paused = true;
            self.frozen_snapshot = self.latest.clone();
            self.log("Paused: display frozen");
        }
    }

    /// Apply a new stabilized backend snapshot.
    pub fn apply_snapshot(&mut self, snap: BackendSnapshot) {
        // Connection transitions are worth a log line.
        if let Some(prev) = self.previous_connection {
            if prev != snap.connection {
                match snap.connection {
                    ConnectionState::Connected => self.log("Connected"),
                    ConnectionState::Reconnecting => self.log("Reconnecting"),
                    ConnectionState::Disconnected => self.log("Connection lost"),
                    _ => self.log(format!("Connection: {}", snap.connection.as_str())),
                }
            }
        }
        self.previous_connection = Some(snap.connection);

        // A connected snapshot clears stale error state from previous
        // failures (auth errors, transport errors). An error carried by
        // this snapshot itself is current and is kept.
        if snap.connection == ConnectionState::Connected {
            self.authentication_failed = false;
            self.connection_message = snap.error.clone();

            let changed = self.latest.as_ref().map(|l| l.server != snap.server).unwrap_or(true);
            if changed {
                match snap.server {
                    crate::domain::ServerState::Ready => self.log("Server ready"),
                    crate::domain::ServerState::Loading => self.log("Server loading"),
                    crate::domain::ServerState::Sleeping => self.log("Server sleeping"),
                    _ => {}
                }
            }

            // Server restart: invalidate history so no invalid delta spike
            // is plotted.
            if let Some(start) = snap.server_start_unix {
                if self.previous_server_start.is_some() && self.previous_server_start != Some(start)
                {
                    self.log("Server restart detected");
                    self.history.clear();
                }
                self.previous_server_start = Some(start);
            }
        }

        // Keep the slot selection in range as the slot count changes.
        let count = snap.slots.len();
        if count == 0 {
            self.selected_slot = 0;
            self.slot_detail_open = false;
        } else if self.selected_slot >= count {
            self.selected_slot = count - 1;
        }

        // History keeps recording while paused (bounded by the ring).
        self.history.record(&snap);

        self.last_update = Some(Instant::now());
        self.latest = Some(snap);
    }

    /// Apply a capabilities update.
    pub fn apply_capabilities(&mut self, caps: BackendCapabilities) {
        if caps != self.capabilities {
            self.log("Capabilities changed");
        }
        if !caps.metrics {
            self.log("Metrics unavailable");
        }
        if !caps.slots {
            self.log("Slots unavailable");
        }
        self.capabilities = caps;
    }

    /// Apply a backend error summary.
    pub fn apply_error(&mut self, err: BackendErrorSummary) {
        // `BackendError::Authentication` renders as "authentication failed: ..."
        // (see `crate::error`); the flag drives the env-var hint view.
        if err.message.starts_with("authentication failed") {
            self.authentication_failed = true;
        }
        self.connection_message = Some(err.message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{ConnectionState, ServerState};

    fn config() -> Config {
        Config::default()
    }

    fn snapshot(connection: ConnectionState, gen_rate: Option<f64>) -> BackendSnapshot {
        BackendSnapshot {
            connection,
            server: ServerState::Ready,
            generation_tokens_per_second: gen_rate,
            active_requests: Some(1),
            slots: vec![],
            ..Default::default()
        }
    }

    #[test]
    fn snapshot_update_sets_latest() {
        let mut s = AppState::new(&config());
        let snap = snapshot(ConnectionState::Connected, Some(42.0));
        s.apply_snapshot(snap.clone());
        assert_eq!(s.latest.as_ref(), Some(&snap));
        assert_eq!(s.visible_snapshot().unwrap().generation_tokens_per_second, Some(42.0));
    }

    #[test]
    fn pause_freezes_visible_snapshot() {
        let mut s = AppState::new(&config());
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(10.0)));
        s.handle_input(InputAction::TogglePause);
        assert!(s.paused);
        // New data arrives while paused.
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(99.0)));
        // Visible snapshot stays frozen.
        assert_eq!(s.visible_snapshot().unwrap().generation_tokens_per_second, Some(10.0));
        // Internal latest still updates.
        assert_eq!(s.latest.unwrap().generation_tokens_per_second, Some(99.0));
    }

    #[test]
    fn resume_returns_to_latest_snapshot() {
        let mut s = AppState::new(&config());
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(10.0)));
        s.handle_input(InputAction::TogglePause);
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(99.0)));
        s.handle_input(InputAction::TogglePause);
        assert!(!s.paused);
        assert_eq!(s.visible_snapshot().unwrap().generation_tokens_per_second, Some(99.0));
    }

    #[test]
    fn pause_before_any_snapshot_is_ignored() {
        let mut s = AppState::new(&config());
        s.handle_input(InputAction::TogglePause);
        assert!(!s.paused, "pause must be ignored until a snapshot exists");
        // The first snapshot still becomes visible (no stuck Waiting view).
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(10.0)));
        assert_eq!(s.visible_snapshot().unwrap().generation_tokens_per_second, Some(10.0));
        // Pause works normally once data exists.
        s.handle_input(InputAction::TogglePause);
        assert!(s.paused);
    }

    #[test]
    fn authentication_error_is_cleared_by_connected_snapshot() {
        let mut s = AppState::new(&config());
        s.apply_error(BackendErrorSummary::new(
            "authentication failed: the server rejected the API key (HTTP 401)",
        ));
        assert!(s.authentication_failed);
        assert!(s.connection_message.is_some());
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(10.0)));
        assert!(!s.authentication_failed, "stale auth failure must not survive a connection");
        assert!(s.connection_message.is_none(), "stale error message must be cleared");
    }

    #[test]
    fn transport_error_is_cleared_by_connected_snapshot() {
        let mut s = AppState::new(&config());
        s.apply_error(BackendErrorSummary::new("cannot connect: connection refused"));
        assert!(s.connection_message.is_some());
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(10.0)));
        assert!(s.connection_message.is_none(), "stale transport error must be cleared");
    }

    #[test]
    fn connected_snapshot_with_active_error_keeps_it() {
        let mut s = AppState::new(&config());
        let mut snap = snapshot(ConnectionState::Connected, Some(10.0));
        snap.error = Some("props endpoint failed".into());
        s.apply_snapshot(snap);
        assert_eq!(s.connection_message.as_deref(), Some("props endpoint failed"));
        assert!(!s.authentication_failed);
    }

    #[test]
    fn history_remains_bounded() {
        let config =
            Config { history_seconds: 10, refresh_interval_ms: 1000, ..Default::default() };
        let mut s = AppState::new(&config);
        for i in 0..10_000 {
            s.apply_snapshot(snapshot(ConnectionState::Connected, Some(i as f64)));
        }
        assert!(s.history.generation_rate.len() <= s.history.generation_rate.capacity());
        assert!(s.history.generation_rate.len() <= crate::app::history::MAX_HISTORY_SAMPLES);
    }

    #[test]
    fn events_remain_bounded() {
        let mut s = AppState::new(&config());
        for i in 0..1000 {
            s.log(format!("event {i}"));
        }
        assert!(s.events.len() <= MAX_EVENTS);
        // The most recent event is retained.
        assert_eq!(s.events.back().unwrap().message, "event 999");
    }

    #[test]
    fn slot_selection_stays_in_range() {
        let mut s = AppState::new(&config());
        let mut with_slots = snapshot(ConnectionState::Connected, None);
        with_slots.slots = (0..5)
            .map(|id| crate::domain::SlotSnapshot {
                id,
                task_id: None,
                is_processing: false,
                n_ctx: None,
                n_tokens: None,
                n_prompt_tokens: None,
                n_prompt_tokens_processed: None,
                n_decoded: None,
                speculative: false,
                phase: crate::domain::SlotPhase::Idle,
            })
            .collect();
        s.apply_snapshot(with_slots.clone());
        // Move to the last slot.
        for _ in 0..4 {
            s.handle_input(InputAction::SlotDown);
        }
        assert_eq!(s.selected_slot, 4);
        // Cannot move past the end.
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 4);
        // Slots disappear: selection resets.
        s.apply_snapshot(snapshot(ConnectionState::Connected, None));
        assert_eq!(s.selected_slot, 0);
        // Shrinking slot list: selection clamps.
        let mut three = with_slots.clone();
        three.slots.truncate(3);
        s.apply_snapshot(three.clone());
        for _ in 0..2 {
            s.handle_input(InputAction::SlotDown);
        }
        assert_eq!(s.selected_slot, 2);
        three.slots.truncate(1);
        s.apply_snapshot(three);
        assert_eq!(s.selected_slot, 0);
    }

    #[test]
    fn clear_history_empties_series() {
        let mut s = AppState::new(&config());
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(1.0)));
        assert!(!s.history.generation_rate.is_empty());
        s.handle_input(InputAction::ClearHistory);
        assert!(s.history.generation_rate.is_empty());
    }

    #[test]
    fn manual_reconnect_event_sets_flag_once() {
        let mut s = AppState::new(&config());
        assert!(!s.take_reconnect_requested());
        s.handle_input(InputAction::Reconnect);
        assert!(s.take_reconnect_requested());
        // Consumed: a second take without a new keypress is false.
        assert!(!s.take_reconnect_requested());
    }

    #[test]
    fn modal_blocks_non_modal_actions() {
        let mut s = AppState::new(&config());
        s.handle_input(InputAction::ToggleHelp);
        assert!(s.show_help);
        s.handle_input(InputAction::TogglePause); // ignored while help is open
        assert!(!s.paused);
        s.handle_input(InputAction::CloseModal);
        assert!(!s.show_help);
    }
}
