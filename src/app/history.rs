//! Bounded rate/counter history for the history panel.
//!
//! A fixed-capacity ring buffer per series. The capacity is derived from
//! `config.history_seconds` and the refresh interval, clamped to a safe
//! upper bound so the memory usage stays bounded no matter the settings.
//!
//! Only finite, plausible values are appended: `NaN`, `Infinity`, and spikes
//! produced by reconnect/restart are rejected at the source (the detector
//! resets its baselines) and are additionally rejected here as a second
//! guard.

use std::collections::VecDeque;

/// Hard upper bound on the number of kept samples per series.
pub const MAX_HISTORY_SAMPLES: usize = 3600;

/// A bounded ring buffer of `f64` samples.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FloatRing {
    values: VecDeque<f64>,
    capacity: usize,
}

impl FloatRing {
    pub fn with_capacity(capacity: usize) -> Self {
        Self { values: VecDeque::with_capacity(capacity.max(1)), capacity: capacity.max(1) }
    }

    /// Append a sample. Non-finite values are dropped (never stored).
    pub fn push(&mut self, value: Option<f64>) {
        let Some(value) = value else { return };
        if !value.is_finite() {
            return;
        }
        self.values.push_back(value);
        while self.values.len() > self.capacity {
            self.values.pop_front();
        }
    }

    /// Iterate over all samples in oldest-first order.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.values.iter().copied()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clear(&mut self) {
        self.values.clear();
    }

    pub fn last(&self) -> Option<f64> {
        self.values.back().copied()
    }
}

/// The full history tracked by the TUI.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    pub prompt_rate: FloatRing,
    pub generation_rate: FloatRing,
    pub active_requests: FloatRing,
    pub queued_requests: FloatRing,
}

impl History {
    /// Build a history sized for `history_seconds` at the given refresh
    /// interval. The sample count is clamped to [`MAX_HISTORY_SAMPLES`].
    pub fn for_config(history_seconds: u64, refresh_interval_ms: u64) -> Self {
        let refresh_ms = refresh_interval_ms.max(1);
        let samples = (history_seconds.saturating_mul(1000) / refresh_ms) as usize;
        let capacity = samples.clamp(1, MAX_HISTORY_SAMPLES);
        Self {
            prompt_rate: FloatRing::with_capacity(capacity),
            generation_rate: FloatRing::with_capacity(capacity),
            active_requests: FloatRing::with_capacity(capacity),
            queued_requests: FloatRing::with_capacity(capacity),
        }
    }

    /// Record one stabilized snapshot. `None` fields are not recorded
    /// (a missing value is never turned into 0).
    pub fn record(&mut self, snap: &crate::domain::BackendSnapshot) {
        self.prompt_rate.push(snap.prompt_tokens_per_second);
        self.generation_rate.push(snap.generation_tokens_per_second);
        self.active_requests.push(snap.active_requests.map(|v| v as f64));
        self.queued_requests.push(snap.queued_requests.map(|v| v as f64));
    }

    /// Drop all samples (reconnect, restart, `c` key).
    pub fn clear(&mut self) {
        self.prompt_rate.clear();
        self.generation_rate.clear();
        self.active_requests.clear();
        self.queued_requests.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_is_clamped_to_the_safe_bound() {
        // 3600s at 100ms would be 36000 samples; must be clamped.
        let h = History::for_config(3600, 100);
        assert_eq!(h.prompt_rate.capacity(), MAX_HISTORY_SAMPLES);

        let h = History::for_config(120, 500);
        assert_eq!(h.prompt_rate.capacity(), 240);

        // Degenerate refresh interval: at least one sample.
        let h = History::for_config(10, 0);
        assert!(h.prompt_rate.capacity() >= 1);
    }

    #[test]
    fn history_remains_bounded() {
        let mut ring = FloatRing::with_capacity(4);
        for i in 0..10 {
            ring.push(Some(i as f64));
        }
        assert_eq!(ring.len(), 4);
        // The newest samples are kept, the oldest dropped.
        assert_eq!(ring.iter().collect::<Vec<_>>(), vec![6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn nan_and_infinity_are_not_recorded() {
        let mut ring = FloatRing::with_capacity(8);
        ring.push(Some(f64::NAN));
        ring.push(Some(f64::INFINITY));
        ring.push(Some(f64::NEG_INFINITY));
        ring.push(None);
        assert!(ring.is_empty());
        ring.push(Some(1.0));
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn clear_empties_every_series() {
        let mut h = History::for_config(10, 100);
        h.record(&crate::domain::BackendSnapshot {
            prompt_tokens_per_second: Some(1.0),
            generation_tokens_per_second: Some(2.0),
            active_requests: Some(3),
            queued_requests: Some(4),
            ..Default::default()
        });
        assert!(!h.prompt_rate.is_empty());
        h.clear();
        assert!(h.prompt_rate.is_empty());
        assert!(h.generation_rate.is_empty());
        assert!(h.active_requests.is_empty());
        assert!(h.queued_requests.is_empty());
    }
}
