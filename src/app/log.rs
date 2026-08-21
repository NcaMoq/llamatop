//! Bounded event log for the TUI.
//!
//! Records state transitions and user actions (connection, server state,
//! workload phase, capabilities, reconnect, pause) as short, redacted
//! messages. Prompt text, completion text, API keys, and raw HTTP bodies
//! must never be passed to [`EventLog::push`]; the message is additionally
//! truncated to a display-width bound so a pathological error string cannot
//! corrupt the panel.
//!
//! Repeated identical events (same kind, severity, and message) collapse
//! into a `repeat_count` instead of adding new records, so a steady
//! condition does not scroll the log.

use std::collections::VecDeque;
use std::time::SystemTime;

use unicode_width::UnicodeWidthStr;

/// Maximum number of retained events.
pub const MAX_LOG_EVENTS: usize = 200;

/// Maximum display width of a stored event message.
pub const MAX_MESSAGE_WIDTH: usize = 160;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Connected,
    Connecting,
    Disconnected,
    Reconnecting,
    AuthenticationFailed,
    /// The llama.cpp server lifecycle state changed (ready/loading/sleeping,
    /// restart detected). GPU/system monitor transitions have their own
    /// kinds.
    ServerStateChanged,
    WorkloadPhaseChanged,
    /// An endpoint observation changed (availability, unsupported, parse
    /// failure, authentication). One variant per endpoint so the log shows
    /// which endpoint changed.
    MetricsAvailabilityChanged,
    SlotsAvailabilityChanged,
    PropsAvailabilityChanged,
    ManualReconnect,
    PauseChanged,
    HistoryCleared,
    /// The event log was cleared. The single audit record left behind keeps
    /// a cleared log distinguishable from "no events yet".
    EventLogCleared,
    /// The system monitor (host CPU/RAM + llama-server process) status
    /// changed.
    SystemMonitorStatusChanged,
    /// The GPU monitor status changed (available, unavailable, NVML
    /// initialization or sampling failure).
    GpuMonitorStatusChanged,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventRecord {
    /// First occurrence of this (possibly repeated) event.
    pub timestamp: SystemTime,
    pub severity: EventSeverity,
    pub kind: EventKind,
    pub message: String,
    pub repeat_count: u32,
}

/// A bounded, newest-at-the-back event log with repeat collapsing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EventLog {
    records: VecDeque<EventRecord>,
}

impl EventLog {
    /// Append an event. An identical event (kind, severity, message) that
    /// is the newest record increments its `repeat_count` instead of
    /// adding a new record. The message is truncated (UTF-8 safe) to
    /// [`MAX_MESSAGE_WIDTH`] display columns.
    pub fn push(&mut self, severity: EventSeverity, kind: EventKind, message: impl Into<String>) {
        let message = truncate_width(message.into(), MAX_MESSAGE_WIDTH);
        if let Some(last) = self.records.back_mut() {
            if last.kind == kind && last.severity == severity && last.message == message {
                last.repeat_count = last.repeat_count.saturating_add(1);
                return;
            }
        }
        self.records.push_back(EventRecord {
            timestamp: SystemTime::now(),
            severity,
            kind,
            message,
            repeat_count: 1,
        });
        while self.records.len() > MAX_LOG_EVENTS {
            self.records.pop_front();
        }
    }

    /// All events in oldest-first order.
    pub fn records(&self) -> &VecDeque<EventRecord> {
        &self.records
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn clear(&mut self) {
        self.records.clear();
    }
}

/// Truncate a string to `max_width` display columns without cutting a
/// character (CJK/full-width/emoji are never split).
pub fn truncate_width(s: String, max_width: usize) -> String {
    if s.width() <= max_width {
        return s;
    }
    let mut out = String::new();
    let mut width = 0usize;
    for ch in s.chars() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if width + w > max_width {
            break;
        }
        out.push(ch);
        width += w;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_identical_event_increments_repeat_count() {
        let mut log = EventLog::default();
        log.push(
            EventSeverity::Warning,
            EventKind::MetricsAvailabilityChanged,
            "metrics endpoint temporarily unavailable",
        );
        log.push(
            EventSeverity::Warning,
            EventKind::MetricsAvailabilityChanged,
            "metrics endpoint temporarily unavailable",
        );
        log.push(
            EventSeverity::Warning,
            EventKind::MetricsAvailabilityChanged,
            "metrics endpoint temporarily unavailable",
        );
        assert_eq!(log.len(), 1);
        let r = log.records().back().unwrap();
        assert_eq!(r.repeat_count, 3);
        assert_eq!(r.kind, EventKind::MetricsAvailabilityChanged);
    }

    #[test]
    fn different_message_is_not_collapsed() {
        let mut log = EventLog::default();
        log.push(EventSeverity::Info, EventKind::PauseChanged, "Paused: display frozen");
        log.push(EventSeverity::Info, EventKind::PauseChanged, "Resume: showing latest snapshot");
        assert_eq!(log.len(), 2);
        assert_eq!(log.records().front().unwrap().repeat_count, 1);
        assert_eq!(log.records().back().unwrap().repeat_count, 1);
    }

    #[test]
    fn event_capacity_removes_oldest_event() {
        let mut log = EventLog::default();
        for i in 0..(MAX_LOG_EVENTS + 50) {
            // Distinct messages so nothing collapses.
            log.push(EventSeverity::Info, EventKind::Connected, format!("event {i}"));
        }
        assert_eq!(log.len(), MAX_LOG_EVENTS);
        assert_eq!(
            log.records().front().unwrap().message,
            format!("event 50"),
            "the oldest 50 events must have been dropped"
        );
        assert_eq!(log.records().back().unwrap().message, format!("event {}", MAX_LOG_EVENTS + 49));
    }

    #[test]
    fn long_utf8_messages_truncate_safely() {
        let mut log = EventLog::default();
        // 300 CJK chars (2 columns each = 600 columns) plus a tail marker.
        let message = format!("{}TAIL", "日".repeat(300));
        log.push(EventSeverity::Error, EventKind::Disconnected, message);
        let r = log.records().back().unwrap();
        assert_eq!(r.message.width(), MAX_MESSAGE_WIDTH);
        assert!(!r.message.contains("TAIL"), "the tail must be cut off");
        // Every retained string is valid UTF-8 (String guarantees this) and
        // ends on a character boundary; the last char must be a full char.
        assert_eq!(r.message.chars().count(), MAX_MESSAGE_WIDTH / 2);
    }

    #[test]
    fn ascii_message_truncation_keeps_full_chars() {
        let mut log = EventLog::default();
        log.push(EventSeverity::Info, EventKind::Connected, "x".repeat(500));
        let r = log.records().back().unwrap();
        assert_eq!(r.message.width(), MAX_MESSAGE_WIDTH);
    }
}
