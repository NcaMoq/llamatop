//! The TUI application state.
//!
//! A single task (see `crate::app::runtime`) owns this struct; no `Arc<Mutex<..>>`
//! sharing is involved. Backend data arrives through `AppEvent`s, keyboard
//! actions through `InputAction`, and the state produces the visible
//! snapshot, bounded history, and bounded event log.
//!
//! Display state (selection, focus, pause, modals) is kept strictly
//! separate from backend state (the latest snapshot).

use std::time::Instant;

use crate::app::event::{BackendErrorSummary, InputAction};
use crate::app::history::History;
use crate::app::log::{EventKind, EventLog, EventSeverity};
use crate::backend::{BackendCapabilities, EndpointAvailability};
use crate::config::Config;
use crate::domain::{
    BackendSnapshot, ConnectionState, GpuMonitor, GpuMonitorStatus, SlotSnapshot, SystemSnapshot,
    WorkloadPhase,
};

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
    /// The Resources panel (host + llama-server process).
    Resources,
}

impl FocusedPanel {
    const ALL: [FocusedPanel; 4] =
        [FocusedPanel::Slots, FocusedPanel::History, FocusedPanel::Events, FocusedPanel::Resources];

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
    /// Latest host + llama-server process sample (Phase D). `None` until the
    /// system monitor has produced a sample or when it is disabled.
    pub system: Option<SystemSnapshot>,
    /// Latest GPU monitor pass (Phase E): status plus per-GPU snapshots.
    /// `None` until the first pass arrives or when GPU monitoring is
    /// disabled.
    pub gpu: Option<GpuMonitor>,
    pub connection_message: Option<String>,
    /// Set while the last reported backend error was an authentication
    /// failure (the view then offers the env-var hint instead of a retry).
    pub authentication_failed: bool,
    /// Whether the Resources panel (host + process) is shown (config).
    pub show_system: bool,
    /// Whether the Resources panel shows a GPU section (config).
    pub show_gpu: bool,

    // --- bounded logs ---
    pub events: EventLog,
    pub history: History,

    // --- display state ---
    pub selected_slot: usize,
    /// Event log scroll distance from the NEWEST event (0 = showing the
    /// latest; larger = scrolled toward older events).
    pub event_scroll: usize,
    /// Current slot table scroll offset (rows hidden above the viewport).
    /// The renderer derives the exact offset each frame; this value keeps
    /// the position stable while the selection stays inside the viewport.
    pub slot_scroll: usize,
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
    previous_workload_phase: Option<WorkloadPhase>,
}

impl AppState {
    pub fn new(config: &Config) -> Self {
        let history = History::for_config(config.history_seconds, config.refresh_interval_ms);
        Self {
            endpoint: crate::endpoint::redact(&config.endpoint),
            api_key_env: config.authentication.api_key_env.clone(),
            latest: None,
            capabilities: BackendCapabilities::default(),
            system: None,
            gpu: None,
            connection_message: None,
            authentication_failed: false,
            show_system: config.show_system,
            show_gpu: config.show_gpu,
            events: EventLog::default(),
            history,
            selected_slot: 0,
            event_scroll: 0,
            slot_scroll: 0,
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
            previous_workload_phase: None,
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

    /// Whether pausing is possible: a snapshot must exist to freeze.
    /// Pause requests before the first snapshot are ignored, so the footer
    /// only advertises `p` once this is true.
    pub fn can_pause(&self) -> bool {
        self.latest.is_some() || self.paused
    }

    /// The visible slots in stable display order (slot ID ascending).
    /// The order is independent of the API's return order, so rows do not
    /// shuffle between snapshots.
    pub fn visible_slots(&self) -> Vec<&SlotSnapshot> {
        let mut slots: Vec<&SlotSnapshot> =
            self.visible_snapshot().map(|s| s.slots.iter().collect()).unwrap_or_default();
        slots.sort_by_key(|s| s.id);
        slots
    }

    /// Whether slot selection is available: the /slots endpoint must have an
    /// `Available` observation and at least one slot must be visible (frozen
    /// while paused). Any other observation (unsupported, temporary, parse
    /// failure, auth failure, unknown) keeps selection inert.
    pub fn can_select_slot(&self) -> bool {
        self.capabilities.slots.is_available() && !self.visible_slots().is_empty()
    }

    /// Minimum scroll offset that keeps the selected row inside a viewport
    /// of `viewport` rows. Purely derived, so resizes and slot-count
    /// changes are always handled without stored state going stale. The
    /// result is always clamped to `count.saturating_sub(viewport)`, so a
    /// stale `slot_scroll` can never point past the last row.
    pub fn slot_scroll_offset(&self, viewport: usize) -> usize {
        let count = self.visible_slots().len();
        if viewport == 0 || count <= viewport {
            return 0;
        }
        let sel = self.selected_slot.min(count - 1);
        let max_offset = count - viewport;
        if sel < self.slot_scroll {
            sel.min(max_offset)
        } else if sel >= self.slot_scroll + viewport {
            (sel + 1 - viewport).min(max_offset)
        } else {
            self.slot_scroll.min(max_offset)
        }
    }

    /// Move the selection and keep the stored scroll offset within range.
    fn set_selected_slot(&mut self, index: usize) {
        let count = self.visible_slots().len();
        if count == 0 {
            return;
        }
        self.selected_slot = index.min(count - 1);
        // Clamp the offset for the current (rough) viewport: the renderer
        // re-derives the exact offset each frame, this only bounds the
        // stored value.
        let viewport = self.terminal_size.1 as usize;
        let max_offset = count.saturating_sub(viewport).min(count - 1);
        self.slot_scroll = self.slot_scroll.min(max_offset);
    }

    /// Compute the new selection index for a slot list replacing the
    /// currently visible one. `new_slots` may arrive in any API order; it is
    /// sorted by ID (the display order) before the lookup, because the
    /// selection indexes the sorted visible list. Preserves the selected
    /// slot ID when it still exists; otherwise falls back to the row nearest
    /// the previous position. A list growing from empty selects the first
    /// row (empty input is handled by the caller).
    ///
    /// Must be called while `visible_snapshot()` still points at the OLD
    /// list (before `latest` is replaced).
    pub fn select_for_slots(&self, new_slots: &[SlotSnapshot]) -> usize {
        if new_slots.is_empty() {
            return 0;
        }
        let mut new_sorted: Vec<&SlotSnapshot> = new_slots.iter().collect();
        new_sorted.sort_by_key(|s| s.id);
        let old_slots = self.visible_slots();
        if old_slots.is_empty() {
            return 0; // empty -> populated: first row
        }
        let old_index = self.selected_slot.min(old_slots.len() - 1);
        let old_id = old_slots[old_index].id;
        if let Some(pos) = new_sorted.iter().position(|s| s.id == old_id) {
            return pos;
        }
        // Selected slot disappeared: nearest valid row to the old position.
        old_index.min(new_sorted.len() - 1)
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

    /// Record a redacted state-transition or user-action event. Never pass
    /// prompt, completion, API key, or raw response body content.
    pub fn log(&mut self, severity: EventSeverity, kind: EventKind, message: impl Into<String>) {
        self.events.push(severity, kind, message);
    }

    /// Scroll the event log by `lines` events toward older events (positive)
    /// or newer events (negative), clamped to the log length. Only applies
    /// while the event log is visible.
    fn scroll_events(&mut self, delta: isize) {
        if !self.show_events {
            return;
        }
        let max = self.events.len().saturating_sub(1);
        let cur = self.event_scroll.min(max);
        self.event_scroll = (cur as isize + delta).clamp(0, max as isize) as usize;
    }

    /// Handle a render tick: keeps the state current without data changes
    /// (e.g. clamping the slot selection is done on snapshot updates; the
    /// tick exists to bound the draw rate and age the UI).
    pub fn on_tick(&mut self) {}

    /// Apply a keyboard action.
    pub fn handle_input(&mut self, action: InputAction) {
        // While the help modal is open, background input is blocked: only the
        // actions that close or toggle the modal apply. In particular `q`
        // does NOT quit while the modal is open (close it first with `?`/Esc).
        if self.show_help && !matches!(action, InputAction::ToggleHelp | InputAction::CloseModal) {
            return;
        }

        match action {
            InputAction::Quit => self.should_quit = true,
            InputAction::Reconnect => {
                self.reconnect_requested = true;
                self.log(
                    EventSeverity::Info,
                    EventKind::ManualReconnect,
                    "Manual reconnect requested",
                );
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
                    self.event_scroll = 0;
                }
            }
            InputAction::FocusNext => self.focused_panel = self.focused_panel.next(),
            InputAction::FocusPrev => self.focused_panel = self.focused_panel.prev(),
            InputAction::SlotUp => {
                if self.can_select_slot() && self.selected_slot > 0 {
                    self.set_selected_slot(self.selected_slot - 1);
                }
            }
            InputAction::SlotDown => {
                if self.can_select_slot() && self.selected_slot + 1 < self.visible_slots().len() {
                    self.set_selected_slot(self.selected_slot + 1);
                }
            }
            InputAction::ToggleSlotDetail => self.slot_detail_open = !self.slot_detail_open,
            InputAction::ToggleEvents => {
                self.show_events = !self.show_events;
                if self.show_events {
                    self.focused_panel = FocusedPanel::Events;
                    self.event_scroll = 0;
                }
            }
            // `c` clears whatever the user is looking at: the event log when
            // it is visible, otherwise the history series.
            InputAction::ClearHistory => {
                if self.show_events {
                    self.events.clear();
                    self.event_scroll = 0;
                    self.log(EventSeverity::Info, EventKind::EventLogCleared, "Event log cleared");
                } else {
                    self.history.clear();
                    self.log(EventSeverity::Info, EventKind::HistoryCleared, "History cleared");
                }
            }
            // Event log scrolling only applies while the log is visible; the
            // state gates it so slot Up/Down (k/j) are never affected.
            // Home = newest (offset 0), End = oldest (offset = len - 1).
            InputAction::LogPageUp => self.scroll_events(10),
            InputAction::LogPageDown => self.scroll_events(-10),
            InputAction::LogHome => {
                if self.show_events {
                    self.event_scroll = 0;
                }
            }
            InputAction::LogEnd => {
                if self.show_events {
                    self.event_scroll = self.events.len().saturating_sub(1);
                }
            }
        }
    }

    fn toggle_pause(&mut self) {
        if self.paused {
            // Resume: map the selection (which points into the frozen list)
            // onto the latest list, preserving the selected slot ID.
            // `visible_slots` still resolves to the frozen list here, so
            // `select_for_slots` translates from frozen to latest.
            if let Some(latest) = self.latest.as_ref() {
                self.selected_slot = self.select_for_slots(&latest.slots);
            }
            self.paused = false;
            self.frozen_snapshot = None;
            self.log(
                EventSeverity::Info,
                EventKind::PauseChanged,
                "Resume: showing latest snapshot",
            );
        } else if self.latest.is_none() {
            // No snapshot to freeze yet: ignoring the request keeps the
            // "Waiting for data..." view from sticking after the first
            // snapshot arrives.
            self.log(EventSeverity::Info, EventKind::PauseChanged, "Pause ignored: no data yet");
        } else {
            self.paused = true;
            self.frozen_snapshot = self.latest.clone();
            self.log(EventSeverity::Info, EventKind::PauseChanged, "Paused: display frozen");
        }
    }

    /// Apply a new stabilized backend snapshot.
    pub fn apply_snapshot(&mut self, snap: BackendSnapshot) {
        // Connection transitions are worth a log line.
        if let Some(prev) = self.previous_connection {
            if prev != snap.connection {
                match snap.connection {
                    ConnectionState::Connected => {
                        self.log(EventSeverity::Info, EventKind::Connected, "Connected")
                    }
                    ConnectionState::Connecting => {
                        self.log(EventSeverity::Info, EventKind::Connecting, "Connecting")
                    }
                    ConnectionState::Reconnecting => {
                        self.log(EventSeverity::Warning, EventKind::Reconnecting, "Reconnecting")
                    }
                    ConnectionState::Disconnected => {
                        self.log(EventSeverity::Error, EventKind::Disconnected, "Connection lost")
                    }
                    ConnectionState::Error => {
                        self.log(EventSeverity::Error, EventKind::Disconnected, "Connection lost")
                    }
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
                    crate::domain::ServerState::Ready => {
                        self.log(EventSeverity::Info, EventKind::ServerStateChanged, "Server ready")
                    }
                    crate::domain::ServerState::Loading => self.log(
                        EventSeverity::Info,
                        EventKind::ServerStateChanged,
                        "Server loading",
                    ),
                    crate::domain::ServerState::Sleeping => self.log(
                        EventSeverity::Info,
                        EventKind::ServerStateChanged,
                        "Server sleeping",
                    ),
                    _ => {}
                }
            }

            // Workload phase transitions are worth a log line (only on a
            // change, so a steady DECODE phase does not fill the log).
            if let Some(prev) = self.previous_workload_phase {
                if prev != snap.workload_phase {
                    self.log(
                        EventSeverity::Info,
                        EventKind::WorkloadPhaseChanged,
                        format!("Phase: {} -> {}", prev.display(), snap.workload_phase.display()),
                    );
                }
            }
            self.previous_workload_phase = Some(snap.workload_phase);

            // Server restart: invalidate history so no invalid delta spike
            // is plotted.
            if let Some(start) = snap.server_start_unix {
                if self.previous_server_start.is_some() && self.previous_server_start != Some(start)
                {
                    self.log(
                        EventSeverity::Warning,
                        EventKind::ServerStateChanged,
                        "Server restart detected",
                    );
                    self.history.clear();
                }
                self.previous_server_start = Some(start);
            }
        }

        // Keep the slot selection valid as the slot list changes: preserve
        // the selected slot ID when it still exists, otherwise fall back to
        // the nearest row. While paused the visible (frozen) list is
        // unchanged, so the selection is restored on resume instead.
        if !self.paused {
            if snap.slots.is_empty() {
                self.selected_slot = 0;
                self.slot_scroll = 0;
                self.slot_detail_open = false;
            } else {
                self.selected_slot = self.select_for_slots(&snap.slots);
            }
        }

        // History keeps recording while paused (bounded by the ring); the
        // sample timestamp matches the one reported as "last update".
        let now = Instant::now();
        self.history.record(&snap, now);

        self.last_update = Some(now);
        self.latest = Some(snap);
    }

    /// Apply a capabilities update.
    ///
    /// An event is logged only for an endpoint whose *observation* changed
    /// (steady states do not spam the log on every poll). The health
    /// observation is intentionally not logged here: its transitions are the
    /// server-state changes already recorded as `ServerStateChanged`.
    pub fn apply_capabilities(&mut self, caps: BackendCapabilities) {
        let prev = self.capabilities;
        if caps.metrics != prev.metrics {
            self.log_capability_change(
                "metrics",
                caps.metrics,
                EventKind::MetricsAvailabilityChanged,
            );
        }
        if caps.slots != prev.slots {
            self.log_capability_change("slots", caps.slots, EventKind::SlotsAvailabilityChanged);
        }
        if caps.props != prev.props {
            self.log_capability_change("props", caps.props, EventKind::PropsAvailabilityChanged);
        }
        self.capabilities = caps;
    }

    /// Log one endpoint observation change with the state-specific message.
    fn log_capability_change(
        &mut self,
        endpoint: &str,
        state: EndpointAvailability,
        kind: EventKind,
    ) {
        match state {
            EndpointAvailability::Available => {
                self.log(EventSeverity::Info, kind, format!("{endpoint} endpoint available"));
            }
            EndpointAvailability::Unsupported => {
                self.log(
                    EventSeverity::Warning,
                    kind,
                    format!("{endpoint} endpoint is not supported by the server"),
                );
            }
            EndpointAvailability::TemporarilyUnavailable => {
                self.log(
                    EventSeverity::Warning,
                    kind,
                    format!("{endpoint} endpoint temporarily unavailable"),
                );
            }
            EndpointAvailability::AuthenticationFailed => {
                self.log(
                    EventSeverity::Error,
                    kind,
                    format!("{endpoint} endpoint authentication failed"),
                );
            }
            EndpointAvailability::ParseFailed => {
                self.log(
                    EventSeverity::Warning,
                    kind,
                    format!("{endpoint} endpoint response could not be parsed"),
                );
            }
            EndpointAvailability::Unknown => {
                self.log(EventSeverity::Info, kind, format!("{endpoint} endpoint status unknown"));
            }
        }
    }

    /// Apply a backend error summary.
    ///
    /// Each new error replaces the previous one: `authentication_failed` is
    /// re-evaluated for the incoming error, so a transport error clears a
    /// stale auth-failure state (and vice versa).
    pub fn apply_error(&mut self, err: BackendErrorSummary) {
        self.authentication_failed = is_authentication_error(&err.message);
        self.connection_message = Some(err.message.clone());
        let severity =
            if self.authentication_failed { EventSeverity::Error } else { EventSeverity::Warning };
        let kind = if self.authentication_failed {
            EventKind::AuthenticationFailed
        } else {
            EventKind::Disconnected
        };
        self.log(severity, kind, err.message);
    }

    /// Apply a host + llama-server process metrics sample (Phase D).
    pub fn apply_system(&mut self, snap: SystemSnapshot) {
        self.system = Some(snap);
    }

    /// The system monitor became unavailable (e.g. a sampling error). The
    /// panel falls back to placeholders and a single warning is logged.
    pub fn apply_system_unavailable(&mut self) {
        if self.system.is_none() {
            self.log(
                EventSeverity::Warning,
                EventKind::SystemMonitorUnavailable,
                "System monitor unavailable",
            );
        }
        self.system = None;
    }

    /// Apply a GPU monitor sampling pass (Phase E). A status change is
    /// logged once (steady states do not spam the log); a pass that has
    /// `Available` status replaces the stored pass, a failure pass keeps
    /// the last known-good data visible but records the failure status.
    pub fn apply_gpu(&mut self, monitor: GpuMonitor) {
        let prev = self.gpu.as_ref().map(|m| m.status);
        if prev != Some(monitor.status) {
            match monitor.status {
                GpuMonitorStatus::Available => {
                    let count = monitor.gpus.len();
                    self.log(
                        EventSeverity::Info,
                        EventKind::ServerStateChanged,
                        format!("GPU monitoring available ({count} device(s))"),
                    );
                }
                GpuMonitorStatus::Unavailable => {
                    self.log(
                        EventSeverity::Warning,
                        EventKind::GpuMonitorUnavailable,
                        "GPU monitoring unavailable",
                    );
                }
                GpuMonitorStatus::InitializationFailed => {
                    self.log(
                        EventSeverity::Warning,
                        EventKind::GpuMonitorUnavailable,
                        "GPU initialization failed (NVML)",
                    );
                }
                GpuMonitorStatus::SamplingFailed => {
                    self.log(
                        EventSeverity::Warning,
                        EventKind::GpuMonitorUnavailable,
                        "GPU sampling failed",
                    );
                }
                GpuMonitorStatus::Disabled => {}
            }
        }
        // A failure pass keeps the last good data on screen (the status
        // note explains it); an available pass replaces it.
        match (monitor.status, monitor.gpus.is_empty()) {
            (GpuMonitorStatus::Available, false) => self.gpu = Some(monitor),
            _ if self.gpu.is_none() => self.gpu = Some(monitor),
            _ => {
                if let Some(stored) = self.gpu.as_mut() {
                    stored.status = monitor.status;
                }
            }
        }
    }
}

/// True when an error message is the authentication failure produced by
/// `BackendError::Authentication` (see `crate::error`). Kept in one place
/// so the marker string is not duplicated.
fn is_authentication_error(message: &str) -> bool {
    message.starts_with("authentication failed")
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

    fn slot(id: u32, processing: bool) -> SlotSnapshot {
        SlotSnapshot {
            id,
            task_id: None,
            is_processing: processing,
            n_ctx: Some(16_384),
            n_tokens: Some(1_200),
            n_prompt_tokens: Some(1_100),
            n_prompt_tokens_processed: Some(1_100),
            n_decoded: Some(100),
            speculative: false,
            phase: if processing {
                crate::domain::SlotPhase::Decode
            } else {
                crate::domain::SlotPhase::Idle
            },
        }
    }

    /// A state with the /slots capability and the given slot ids visible.
    fn state_with_slots(ids: &[u32]) -> AppState {
        let mut s = AppState::new(&config());
        s.apply_capabilities(crate::backend::BackendCapabilities {
            slots: EndpointAvailability::Available,
            ..Default::default()
        });
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = ids.iter().map(|id| slot(*id, false)).collect();
        s.apply_snapshot(snap);
        s
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
    fn transport_error_replaces_stale_authentication_state() {
        let mut s = AppState::new(&config());
        s.apply_error(BackendErrorSummary::new(
            "authentication failed: the server rejected the API key (HTTP 401)",
        ));
        assert!(s.authentication_failed);
        s.apply_error(BackendErrorSummary::new("cannot connect: connection refused"));
        assert!(!s.authentication_failed, "a new transport error must clear the stale auth state");
        assert_eq!(s.connection_message.as_deref(), Some("cannot connect: connection refused"));
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
        assert!(s.history.len() <= s.history.capacity());
        assert!(s.history.len() <= crate::app::history::MAX_HISTORY_SAMPLES);
    }

    #[test]
    fn events_remain_bounded() {
        let mut s = AppState::new(&config());
        for i in 0..1000 {
            s.log(EventSeverity::Info, EventKind::Connected, format!("event {i}"));
        }
        assert!(s.events.len() <= crate::app::log::MAX_LOG_EVENTS);
        // The most recent event is retained.
        assert_eq!(s.events.records().back().unwrap().message, "event 999");
    }

    #[test]
    fn connection_transitions_create_one_event_each() {
        let mut s = AppState::new(&config());
        // The first observation sets the baseline: no event yet.
        s.apply_snapshot(snapshot(ConnectionState::Connecting, None));
        s.apply_snapshot(snapshot(ConnectionState::Connected, None));
        s.apply_snapshot(snapshot(ConnectionState::Disconnected, None));
        // One event per transition (not per snapshot, none for the first
        // observation).
        let records = s.events.records();
        let kinds: Vec<EventKind> = records.iter().map(|r| r.kind).collect();
        assert_eq!(kinds, vec![EventKind::Connected, EventKind::Disconnected]);
        // A steady state does not add a duplicate event.
        s.apply_snapshot(snapshot(ConnectionState::Disconnected, None));
        assert_eq!(s.events.len(), 2);
    }

    #[test]
    fn repeated_identical_error_collapses_repeat_count() {
        let mut s = AppState::new(&config());
        for _ in 0..5 {
            s.apply_error(BackendErrorSummary::new("cannot connect: connection refused"));
        }
        // Identical repeated errors collapse into one record.
        assert_eq!(s.events.len(), 1);
        let rec = s.events.records().back().unwrap();
        assert_eq!(rec.repeat_count, 5);
        assert_eq!(rec.kind, EventKind::Disconnected);
    }

    #[test]
    fn workload_phase_change_creates_event() {
        let mut s = AppState::new(&config());
        let mut decode = snapshot(ConnectionState::Connected, None);
        decode.workload_phase = crate::domain::WorkloadPhase::Decode;
        s.apply_snapshot(decode);
        let mut idle = snapshot(ConnectionState::Connected, None);
        idle.workload_phase = crate::domain::WorkloadPhase::Idle;
        s.apply_snapshot(idle);
        let rec = s
            .events
            .records()
            .iter()
            .find(|r| r.kind == EventKind::WorkloadPhaseChanged)
            .expect("a phase-change event exists");
        assert!(rec.message.contains("DECODE"));
        assert!(rec.message.contains("IDLE"));
    }

    #[test]
    fn steady_workload_phase_does_not_spam_events() {
        let mut s = AppState::new(&config());
        let mut decode = snapshot(ConnectionState::Connected, None);
        decode.workload_phase = crate::domain::WorkloadPhase::Decode;
        for _ in 0..10 {
            s.apply_snapshot(snapshot(ConnectionState::Connected, None));
        }
        let phase_events =
            s.events.records().iter().filter(|r| r.kind == EventKind::WorkloadPhaseChanged).count();
        assert_eq!(phase_events, 0, "a steady phase must not log repeatedly");
    }

    #[test]
    fn manual_reconnect_logs_an_event() {
        let mut s = AppState::new(&config());
        s.handle_input(InputAction::Reconnect);
        let rec = s.events.records().back().unwrap();
        assert_eq!(rec.kind, EventKind::ManualReconnect);
        assert_eq!(rec.severity, EventSeverity::Info);
    }

    #[test]
    fn pause_and_resume_log_events() {
        let mut s = AppState::new(&config());
        s.apply_snapshot(snapshot(ConnectionState::Connected, None));
        s.handle_input(InputAction::TogglePause);
        assert_eq!(s.events.records().back().unwrap().kind, EventKind::PauseChanged);
        s.handle_input(InputAction::TogglePause);
        let rec = s.events.records().back().unwrap();
        assert_eq!(rec.kind, EventKind::PauseChanged);
        assert_eq!(rec.message, "Resume: showing latest snapshot");
    }

    #[test]
    fn event_log_clear_only_when_events_visible() {
        let mut s = AppState::new(&config());
        s.apply_snapshot(snapshot(ConnectionState::Connected, None));
        // With events hidden, `c` clears the history instead (no event-log
        // clear event is recorded).
        s.handle_input(InputAction::ClearHistory);
        assert!(s.history.is_empty());
        assert!(
            s.events.records().iter().all(|r| r.kind != EventKind::EventLogCleared),
            "history clear must not log an event-log clear"
        );

        // With events visible, `c` clears the event log and logs the clear.
        s.handle_input(InputAction::ToggleEvents);
        assert!(s.show_events);
        s.handle_input(InputAction::ClearHistory);
        let rec = s.events.records().back().unwrap();
        assert_eq!(rec.kind, EventKind::EventLogCleared);
    }

    #[test]
    fn event_scroll_is_gated_to_visible_events() {
        let mut s = AppState::new(&config());
        // Distinct messages so nothing collapses.
        for i in 0..20 {
            s.log(EventSeverity::Info, EventKind::Connected, format!("e{i}"));
        }
        // Not visible: scrolling is a no-op.
        s.handle_input(InputAction::LogPageUp);
        s.handle_input(InputAction::LogEnd);
        assert_eq!(s.event_scroll, 0, "scroll must be inert while events are hidden");

        // Visible: scrolling works and is clamped. PageUp moves toward the
        // older events (larger offset from the newest).
        s.handle_input(InputAction::ToggleEvents);
        s.handle_input(InputAction::LogEnd);
        assert_eq!(s.event_scroll, 19);
        s.handle_input(InputAction::LogPageUp);
        assert_eq!(s.event_scroll, 19, "clamped at the oldest event");
        s.handle_input(InputAction::LogPageDown);
        assert_eq!(s.event_scroll, 9);
        s.handle_input(InputAction::LogHome);
        assert_eq!(s.event_scroll, 0);
        s.handle_input(InputAction::LogPageDown);
        assert_eq!(s.event_scroll, 0, "clamped at the newest event");
    }

    #[test]
    fn slot_navigation_unaffected_by_log_scroll_keys() {
        let mut s = state_with_slots(&[0, 1, 2]);
        // Log-scroll keys must not move the slot selection.
        s.handle_input(InputAction::LogPageUp);
        s.handle_input(InputAction::LogPageDown);
        assert_eq!(s.selected_slot, 0);
        // Normal slot navigation still works.
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 1);
    }

    #[test]
    fn system_sample_updates_state() {
        let mut s = AppState::new(&config());
        assert!(s.system.is_none());
        s.apply_system(crate::domain::SystemSnapshot {
            cpu_usage_percent: Some(3.0),
            ram_used_bytes: Some(100),
            ram_total_bytes: Some(200),
            process_match_count: Some(1),
            process: None,
            association: crate::domain::ProcessAssociation::SingleLocalCandidate,
        });
        assert!(s.system.is_some());
        assert_eq!(s.system.as_ref().unwrap().cpu_usage_percent, Some(3.0));
    }

    #[test]
    fn system_unavailable_logs_once_and_clears() {
        let mut s = AppState::new(&config());
        s.apply_system_unavailable();
        // First unavailable: a single warning event is logged.
        assert!(s.system.is_none());
        assert_eq!(
            s.events
                .records()
                .iter()
                .filter(|r| r.kind == EventKind::SystemMonitorUnavailable)
                .count(),
            1
        );
        // A second call (still no data) does not add another event.
        s.apply_system_unavailable();
        assert_eq!(
            s.events
                .records()
                .iter()
                .filter(|r| r.kind == EventKind::SystemMonitorUnavailable)
                .count(),
            1,
            "the unavailable warning must not repeat"
        );
    }

    /// A GPU monitor pass with the given status and device count.
    fn gpu_monitor(status: GpuMonitorStatus, count: usize) -> GpuMonitor {
        GpuMonitor {
            status,
            gpus: (0..count)
                .map(|i| crate::domain::GpuSnapshot {
                    index: i as u32,
                    uuid: Some(format!("GPU-{i}")),
                    name: Some(format!("GPU {i}")),
                    utilization_percent: Some(10),
                    memory_used_bytes: Some(1),
                    memory_total_bytes: Some(2),
                    temperature_celsius: None,
                    power_watts: None,
                    power_limit_watts: None,
                    graphics_clock_mhz: None,
                    memory_clock_mhz: None,
                })
                .collect(),
        }
    }

    #[test]
    fn gpu_sample_available_stores_gpus() {
        let mut s = AppState::new(&config());
        s.apply_gpu(gpu_monitor(GpuMonitorStatus::Available, 2));
        assert_eq!(s.gpu.as_ref().unwrap().status, GpuMonitorStatus::Available);
        assert_eq!(s.gpu.as_ref().unwrap().gpus.len(), 2);
        // A steady available pass does not log again.
        s.apply_gpu(gpu_monitor(GpuMonitorStatus::Available, 2));
        assert_eq!(
            s.events.records().iter().filter(|r| r.kind == EventKind::ServerStateChanged).count(),
            1
        );
    }

    #[test]
    fn gpu_failure_keeps_last_good_data_and_logs_once() {
        let mut s = AppState::new(&config());
        s.apply_gpu(gpu_monitor(GpuMonitorStatus::Available, 1));
        s.apply_gpu(gpu_monitor(GpuMonitorStatus::SamplingFailed, 0));
        // The failure status is recorded, but the last good device list
        // stays on screen.
        assert_eq!(s.gpu.as_ref().unwrap().status, GpuMonitorStatus::SamplingFailed);
        assert_eq!(s.gpu.as_ref().unwrap().gpus.len(), 1);
        s.apply_gpu(gpu_monitor(GpuMonitorStatus::SamplingFailed, 0));
        assert_eq!(
            s.events
                .records()
                .iter()
                .filter(|r| r.kind == EventKind::GpuMonitorUnavailable)
                .count(),
            1,
            "a steady failure must not spam the log"
        );
    }

    #[test]
    fn gpu_unavailable_before_any_data_logs_one_warning() {
        let mut s = AppState::new(&config());
        s.apply_gpu(gpu_monitor(GpuMonitorStatus::Unavailable, 0));
        assert!(s.gpu.is_some());
        assert_eq!(s.gpu.as_ref().unwrap().gpus.len(), 0);
        assert_eq!(
            s.events
                .records()
                .iter()
                .filter(|r| r.kind == EventKind::GpuMonitorUnavailable)
                .count(),
            1
        );
    }

    #[test]
    fn slot_selection_stays_in_range() {
        let mut s = AppState::new(&config());
        // Slot navigation only applies when the /slots endpoint is available.
        s.apply_capabilities(crate::backend::BackendCapabilities {
            slots: EndpointAvailability::Available,
            ..Default::default()
        });
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
    fn slot_selection_requires_the_slots_capability() {
        // Without the capability, navigation is inert even with slots.
        let mut s = AppState::new(&config());
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = vec![slot(0, false), slot(1, false)];
        s.apply_snapshot(snap);
        assert!(!s.can_select_slot());
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 0, "navigation must be ignored without /slots");

        // Enabling the capability activates selection on the same data.
        s.apply_capabilities(crate::backend::BackendCapabilities {
            slots: EndpointAvailability::Available,
            ..Default::default()
        });
        assert!(s.can_select_slot());
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 1);
    }

    #[test]
    fn capability_changes_log_per_endpoint_events() {
        let mut s = AppState::new(&config());
        // The first probe: slots unsupported, metrics temporary, props
        // unparseable. One event per changed endpoint, in a fixed order.
        let first = crate::backend::BackendCapabilities {
            slots: EndpointAvailability::Unsupported,
            metrics: EndpointAvailability::TemporarilyUnavailable,
            props: EndpointAvailability::ParseFailed,
            ..Default::default()
        };
        s.apply_capabilities(first);
        let kinds: Vec<EventKind> = s.events.records().iter().map(|r| r.kind).collect();
        assert_eq!(
            kinds,
            vec![
                EventKind::MetricsAvailabilityChanged,
                EventKind::SlotsAvailabilityChanged,
                EventKind::PropsAvailabilityChanged
            ]
        );
        // Re-applying the same observations must not add events (no per-poll
        // spam).
        s.apply_capabilities(first);
        assert_eq!(s.events.len(), 3, "steady states must not spam the log");
        // Recovery to Available logs exactly one info event.
        let recovered = crate::backend::BackendCapabilities {
            slots: EndpointAvailability::Available,
            metrics: EndpointAvailability::TemporarilyUnavailable,
            props: EndpointAvailability::ParseFailed,
            ..Default::default()
        };
        s.apply_capabilities(recovered);
        assert_eq!(s.events.len(), 4);
        let rec = s.events.records().back().unwrap();
        assert_eq!(rec.kind, EventKind::SlotsAvailabilityChanged);
        assert_eq!(rec.severity, EventSeverity::Info);
        assert_eq!(rec.message, "slots endpoint available");
    }

    #[test]
    fn capability_auth_failure_is_an_error_event() {
        let mut s = AppState::new(&config());
        s.apply_capabilities(crate::backend::BackendCapabilities {
            slots: EndpointAvailability::AuthenticationFailed,
            ..Default::default()
        });
        let rec = s.events.records().back().unwrap();
        assert_eq!(rec.kind, EventKind::SlotsAvailabilityChanged);
        assert_eq!(rec.severity, EventSeverity::Error);
        assert_eq!(rec.message, "slots endpoint authentication failed");
    }

    #[test]
    fn first_slot_is_selected_when_slots_appear() {
        let s = state_with_slots(&[0, 1, 2]);
        assert_eq!(s.selected_slot, 0);
        let visible = s.visible_slots();
        assert_eq!(visible.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn slots_display_in_id_order_regardless_of_api_order() {
        let s = state_with_slots(&[2, 0, 1]);
        let visible = s.visible_slots();
        assert_eq!(visible.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn slot_navigation_moves_down_and_up_without_wrapping() {
        let mut s = state_with_slots(&[0, 1, 2]);
        // Up on the first row is a no-op (no wrap).
        s.handle_input(InputAction::SlotUp);
        assert_eq!(s.selected_slot, 0);
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 1);
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 2);
        // Down on the last row is a no-op (no wrap).
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 2);
        s.handle_input(InputAction::SlotUp);
        assert_eq!(s.selected_slot, 1);
    }

    #[test]
    fn selection_preserves_slot_id_when_order_changes() {
        let mut s = state_with_slots(&[0, 1, 2]);
        s.handle_input(InputAction::SlotDown); // select id 1
        assert_eq!(s.selected_slot, 1);
        // The same ids arrive in a different API order; visible order is
        // still ID-ascending, so id 1 stays selected.
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = vec![slot(2, false), slot(0, false), slot(1, false)];
        s.apply_snapshot(snap);
        assert_eq!(s.selected_slot, 1);
        assert_eq!(s.visible_slots()[1].id, 1);
    }

    #[test]
    fn selection_preserves_slot_id_when_ids_shift() {
        let mut s = state_with_slots(&[0, 1, 2]);
        s.handle_input(InputAction::SlotDown);
        s.handle_input(InputAction::SlotDown); // select id 2 (last row)
        assert_eq!(s.selected_slot, 2);
        // Slot 1 disappears; id 2 moves from row 2 to row 1.
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = vec![slot(0, false), slot(2, false)];
        s.apply_snapshot(snap);
        assert_eq!(s.selected_slot, 1, "the selected slot ID must follow its new row");
        assert_eq!(s.visible_slots()[1].id, 2);
    }

    #[test]
    fn selection_falls_back_to_nearest_when_selected_slot_is_removed() {
        let mut s = state_with_slots(&[0, 1, 2]);
        s.handle_input(InputAction::SlotDown);
        s.handle_input(InputAction::SlotDown); // select id 2
        assert_eq!(s.selected_slot, 2);
        // The selected slot (id 2) is removed entirely: nearest valid row.
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = vec![slot(0, false), slot(1, false)];
        s.apply_snapshot(snap);
        assert_eq!(s.selected_slot, 1, "fall back to the last valid row");
    }

    #[test]
    fn empty_slot_list_clears_selection() {
        let mut s = state_with_slots(&[0, 1, 2]);
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 1);
        s.apply_snapshot(snapshot(ConnectionState::Connected, None)); // no slots
        assert_eq!(s.selected_slot, 0);
        assert!(!s.can_select_slot());
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 0, "no movement with zero slots");
    }

    #[test]
    fn reappearing_slots_select_first_row() {
        let mut s = state_with_slots(&[]);
        assert!(!s.can_select_slot());
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = vec![slot(5, false), slot(6, false)];
        s.apply_snapshot(snap);
        assert_eq!(s.selected_slot, 0, "a growing list selects the first row");
        assert!(s.can_select_slot());
    }

    #[test]
    fn pause_keeps_frozen_slot_list_and_selection() {
        let mut s = state_with_slots(&[0, 1, 2]);
        s.handle_input(InputAction::SlotDown); // select id 1
        s.handle_input(InputAction::TogglePause);
        assert!(s.paused);
        // New data arrives while paused: visible list stays frozen.
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = vec![slot(9, false), slot(8, false)];
        s.apply_snapshot(snap);
        assert_eq!(s.selected_slot, 1, "selection must not move while paused");
        let visible = s.visible_slots();
        assert_eq!(visible.iter().map(|s| s.id).collect::<Vec<_>>(), vec![0, 1, 2]);
        // Navigation still works inside the frozen list.
        s.handle_input(InputAction::SlotDown);
        assert_eq!(s.selected_slot, 2);
    }

    #[test]
    fn resume_remaps_selection_to_latest_by_id() {
        let mut s = state_with_slots(&[0, 1, 2]);
        s.handle_input(InputAction::SlotDown);
        s.handle_input(InputAction::SlotDown); // select id 2
        s.handle_input(InputAction::TogglePause);
        assert!(s.paused);
        // While paused the live list changes: slot 1 disappears, slot 3 joins.
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = vec![slot(0, false), slot(2, false), slot(3, false)];
        s.apply_snapshot(snap);
        assert_eq!(s.selected_slot, 2, "frozen position while paused");
        s.handle_input(InputAction::TogglePause); // resume
        assert!(!s.paused);
        assert_eq!(s.selected_slot, 1, "id 2 moved from row 2 to row 1 in the latest list");
        assert_eq!(s.visible_slots()[1].id, 2);
    }

    #[test]
    fn resume_falls_back_to_nearest_when_selected_slot_is_gone() {
        let mut s = state_with_slots(&[0, 1, 2]);
        s.handle_input(InputAction::SlotDown);
        s.handle_input(InputAction::SlotDown); // select id 2
        s.handle_input(InputAction::TogglePause);
        let mut snap = snapshot(ConnectionState::Connected, None);
        snap.slots = vec![slot(0, false), slot(1, false)]; // id 2 removed
        s.apply_snapshot(snap);
        s.handle_input(InputAction::TogglePause); // resume
        assert_eq!(s.selected_slot, 1, "nearest valid row in the latest list");
    }

    #[test]
    fn slot_scroll_offset_keeps_selection_visible() {
        let mut s = state_with_slots(&(0..10).collect::<Vec<_>>());
        // Select the last row.
        for _ in 0..9 {
            s.handle_input(InputAction::SlotDown);
        }
        assert_eq!(s.selected_slot, 9);
        // A 5-row viewport must scroll so row 9 is the last visible row.
        let offset = s.slot_scroll_offset(5);
        assert_eq!(offset, 5, "rows 5..10 are visible, including the selection");
        // Moving back above the viewport scrolls the other way; once the
        // selection is row 0 the offset returns to zero.
        s.slot_scroll = 5;
        for _ in 0..9 {
            s.handle_input(InputAction::SlotUp);
        }
        assert_eq!(s.selected_slot, 0);
        assert_eq!(s.slot_scroll_offset(5), 0);
    }

    #[test]
    fn slot_scroll_offset_never_exceeds_max() {
        let mut s = state_with_slots(&[0, 1, 2, 3, 4]);
        // A stale, oversized offset must be clamped, not panic.
        s.slot_scroll = 99;
        s.handle_input(InputAction::SlotDown);
        let offset = s.slot_scroll_offset(3);
        assert!(offset <= 5 - 3, "offset must stay within count - viewport");
        // The selected row (1) is inside the visible window.
        assert!(1 >= offset && 1 < offset + 3);
    }

    #[test]
    fn slot_scroll_offset_is_zero_when_list_fits() {
        let s = state_with_slots(&[0, 1, 2]);
        assert_eq!(s.slot_scroll_offset(5), 0);
        assert_eq!(s.slot_scroll_offset(0), 0, "a zero viewport shows nothing");
    }

    #[test]
    fn clear_history_empties_series() {
        let mut s = AppState::new(&config());
        s.apply_snapshot(snapshot(ConnectionState::Connected, Some(1.0)));
        assert!(!s.history.is_empty());
        s.handle_input(InputAction::ClearHistory);
        assert!(s.history.is_empty());
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
        s.handle_input(InputAction::SlotDown); // ignored while help is open
        s.handle_input(InputAction::Quit); // q must NOT quit while help is open
        assert!(!s.should_quit, "q is blocked while the help modal is open");
        s.handle_input(InputAction::CloseModal);
        assert!(!s.show_help);
        // After closing the modal, background actions apply again.
        s.handle_input(InputAction::Quit);
        assert!(s.should_quit, "q quits once the modal is closed");
    }

    #[test]
    fn help_toggle_is_idempotent_with_esc() {
        let mut s = AppState::new(&config());
        s.handle_input(InputAction::ToggleHelp); // `?` opens
        assert!(s.show_help);
        s.handle_input(InputAction::CloseModal); // Esc closes
        assert!(!s.show_help);
        s.handle_input(InputAction::ToggleHelp); // `?` opens again
        assert!(s.show_help);
        s.handle_input(InputAction::ToggleHelp); // `?` closes
        assert!(!s.show_help);
    }
}
