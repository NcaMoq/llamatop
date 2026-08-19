//! Hysteresis helpers so the UI does not flicker between states.
//!
//! Different transitions have different stability requirements:
//!
//! - `Loading` / `Sleeping`: apply immediately (authoritative signals)
//! - `Decode`: apply immediately when token growth is detected
//! - `PrefillLikely`: require two consecutive observations
//! - `Idle`: require two consecutive observations
//! - `Disconnected`: allow a short transient-failure window
//! - `ProcessingUnknown`: apply after uncertainty persists
//!
//! All timing uses `std::time::Instant` (monotonic), never wall-clock time.

use std::time::Instant;

/// A candidate state that must persist for N consecutive observations before
/// it is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    pub count: u8,
    pub required: u8,
}

impl Candidate {
    pub fn is_satisfied(&self) -> bool {
        self.count >= self.required
    }

    pub fn next(&self) -> Self {
        Self { count: self.count.saturating_add(1), required: self.required }
    }
}

/// Tracks a candidate value across observations.
///
/// `feed(Some(value))` accumulates while the value stays the same and resets
/// when it changes. `feed(None)` clears the candidate (signal lost).
#[derive(Debug, Clone)]
pub struct StableCandidate<T> {
    value: Option<T>,
    count: u8,
    required: u8,
}

/// Default candidate: satisfied on the first matching observation.
/// (No `T: Default` bound, so it works for enums without a default variant.)
impl<T> Default for StableCandidate<T> {
    fn default() -> Self {
        Self { value: None, count: 0, required: 1 }
    }
}

impl<T: PartialEq + Clone> StableCandidate<T> {
    pub fn new(required: u8) -> Self {
        Self { value: None, count: 0, required: required.max(1) }
    }

    /// Feed one observation; returns `Some(value)` when the candidate has
    /// persisted long enough to be applied.
    pub fn feed(&mut self, observed: Option<&T>) -> Option<T> {
        match observed {
            Some(v) => match &self.value {
                Some(current) if current == v => {
                    self.count = self.count.saturating_add(1);
                    if self.count >= self.required {
                        Some(v.clone())
                    } else {
                        None
                    }
                }
                _ => {
                    self.value = Some(v.clone());
                    self.count = 1;
                    if self.required <= 1 {
                        Some(v.clone())
                    } else {
                        None
                    }
                }
            },
            None => {
                self.value = None;
                self.count = 0;
                None
            }
        }
    }

    pub fn reset(&mut self) {
        self.value = None;
        self.count = 0;
    }
}

/// Tracks consecutive failures with a transient-tolerance window.
///
/// The caller decides the policy: e.g. treat the connection as
/// `Reconnecting` after 1-2 failures and `Disconnected` after the window
/// expires or after N consecutive failures.
#[derive(Debug, Clone, Default)]
pub struct FailureWindow {
    failures: u8,
    first_failure_at: Option<Instant>,
}

impl FailureWindow {
    pub fn record_failure(&mut self, now: Instant) {
        self.failures = self.failures.saturating_add(1);
        if self.first_failure_at.is_none() {
            self.first_failure_at = Some(now);
        }
    }

    pub fn record_success(&mut self) {
        self.failures = 0;
        self.first_failure_at = None;
    }

    pub fn failures(&self) -> u8 {
        self.failures
    }

    /// How long the current failure streak has lasted.
    pub fn duration(&self, now: Instant) -> Option<std::time::Duration> {
        self.first_failure_at.map(|t| now.duration_since(t))
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn immediate_when_required_is_one() {
        let mut c: StableCandidate<u32> = StableCandidate::new(1);
        assert_eq!(c.feed(Some(&7)), Some(7));
    }

    #[test]
    fn two_consecutive_observations_required() {
        let mut c: StableCandidate<u32> = StableCandidate::new(2);
        assert_eq!(c.feed(Some(&7)), None);
        assert_eq!(c.feed(Some(&7)), Some(7));
    }

    #[test]
    fn a_different_value_resets_the_count() {
        let mut c: StableCandidate<u32> = StableCandidate::new(2);
        c.feed(Some(&7));
        assert_eq!(c.feed(Some(&8)), None);
        assert_eq!(c.feed(Some(&8)), Some(8));
        // Switching back requires re-accumulation.
        assert_eq!(c.feed(Some(&9)), None);
    }

    #[test]
    fn none_clears_the_candidate() {
        let mut c: StableCandidate<u32> = StableCandidate::new(2);
        c.feed(Some(&7));
        assert_eq!(c.feed(None), None);
        assert_eq!(c.feed(Some(&7)), None);
        assert_eq!(c.feed(Some(&7)), Some(7));
    }

    #[test]
    fn reset_forgets_progress() {
        let mut c: StableCandidate<u32> = StableCandidate::new(2);
        c.feed(Some(&7));
        c.reset();
        assert_eq!(c.feed(Some(&7)), None);
    }

    #[test]
    fn failure_window_counts_and_times() {
        let mut w = FailureWindow::default();
        let start = Instant::now();
        w.record_failure(start);
        assert_eq!(w.failures(), 1);
        w.record_failure(start + Duration::from_millis(500));
        assert_eq!(w.failures(), 2);
        let d = w.duration(start + Duration::from_millis(1200)).unwrap();
        assert!(d >= Duration::from_millis(600));

        w.record_success();
        assert_eq!(w.failures(), 0);
        assert!(w.duration(start).is_none());
    }
}
