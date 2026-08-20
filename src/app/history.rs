//! Bounded, time-aligned history for the history panel.
//!
//! One `HistorySample` per observation holds *all* series at the same index,
//! so a missing value in one series can never shift the other series out of
//! time alignment (the old per-series rings each dropped their own `None`s).
//! A gap — e.g. while disconnected — is an explicit sample whose fields are
//! `None`, never a zero.
//!
//! Reconnect policy: transient outages keep recording (gap samples), so the
//! window always ends at "now". A server *restart* clears the history
//! instead, because every baseline is invalid across a restart (the detector
//! resets for the same reason).
//!
//! Only finite, non-negative values are stored: `NaN`, `Infinity`, and
//! negative rates are dropped to `None` at the source.

use std::collections::VecDeque;
use std::time::Instant;

use crate::domain::BackendSnapshot;

/// Hard upper bound on the number of kept samples.
pub const MAX_HISTORY_SAMPLES: usize = 3600;

/// One observation: every series at the same point in time.
///
/// `None` means "not reported / not measurable at this moment" and must
/// never be rendered as 0.
#[derive(Debug, Clone, PartialEq)]
pub struct HistorySample {
    pub captured_at: Instant,
    pub prompt_tokens_per_second: Option<f64>,
    pub generation_tokens_per_second: Option<f64>,
    pub active_requests: Option<u64>,
    pub queued_requests: Option<u64>,
}

/// A bounded, fixed-capacity, time-aligned sample buffer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct History {
    samples: VecDeque<HistorySample>,
    capacity: usize,
}

impl History {
    /// Build a history sized for `history_seconds` at the given refresh
    /// interval. The sample count is clamped to [`MAX_HISTORY_SAMPLES`] and
    /// to at least one (a zero capacity must not panic).
    pub fn for_config(history_seconds: u64, refresh_interval_ms: u64) -> Self {
        let refresh_ms = refresh_interval_ms.max(1);
        let samples = (history_seconds.saturating_mul(1000) / refresh_ms) as usize;
        let capacity = samples.clamp(1, MAX_HISTORY_SAMPLES);
        Self { samples: VecDeque::with_capacity(capacity), capacity }
    }

    /// Record one stabilized snapshot as a single aligned sample.
    /// `None` fields stay `None` (a missing value is never turned into 0);
    /// non-finite or negative rates are sanitized to `None`.
    pub fn record(&mut self, snap: &BackendSnapshot, captured_at: Instant) {
        let sample = HistorySample {
            captured_at,
            prompt_tokens_per_second: sanitize_rate(snap.prompt_tokens_per_second),
            generation_tokens_per_second: sanitize_rate(snap.generation_tokens_per_second),
            active_requests: snap.active_requests,
            queued_requests: snap.queued_requests,
        };
        self.samples.push_back(sample);
        while self.samples.len() > self.capacity {
            self.samples.pop_front();
        }
    }

    /// All samples in oldest-first order.
    pub fn samples(&self) -> &VecDeque<HistorySample> {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The newest sample, if any.
    pub fn last(&self) -> Option<&HistorySample> {
        self.samples.back()
    }

    /// Drop all samples (server restart, `c` key while events are shown).
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Keep a rate only when it is finite and non-negative.
fn sanitize_rate(value: Option<f64>) -> Option<f64> {
    let value = value?;
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        prompt: Option<f64>,
        generation: Option<f64>,
        active: Option<u64>,
        queued: Option<u64>,
    ) -> BackendSnapshot {
        BackendSnapshot {
            prompt_tokens_per_second: prompt,
            generation_tokens_per_second: generation,
            active_requests: active,
            queued_requests: queued,
            ..Default::default()
        }
    }

    #[test]
    fn one_observation_creates_one_aligned_sample() {
        let mut h = History::for_config(120, 500);
        h.record(&snapshot(Some(10.0), Some(2.0), Some(1), Some(0)), Instant::now());
        assert_eq!(h.len(), 1);
        let s = h.last().expect("sample");
        assert_eq!(s.prompt_tokens_per_second, Some(10.0));
        assert_eq!(s.generation_tokens_per_second, Some(2.0));
        assert_eq!(s.active_requests, Some(1));
        assert_eq!(s.queued_requests, Some(0));
    }

    #[test]
    fn missing_prompt_rate_remains_none() {
        let mut h = History::for_config(120, 500);
        h.record(&snapshot(None, Some(2.0), Some(1), Some(0)), Instant::now());
        let s = h.last().expect("sample");
        assert_eq!(s.prompt_tokens_per_second, None);
        // The other series of the same sample is untouched.
        assert_eq!(s.generation_tokens_per_second, Some(2.0));
    }

    #[test]
    fn missing_generation_rate_remains_none() {
        let mut h = History::for_config(120, 500);
        h.record(&snapshot(Some(10.0), None, None, None), Instant::now());
        let s = h.last().expect("sample");
        assert_eq!(s.generation_tokens_per_second, None);
        assert_eq!(s.active_requests, None);
        assert_eq!(s.queued_requests, None);
    }

    #[test]
    fn capacity_removes_oldest_sample() {
        let mut h = History::for_config(120, 500);
        // Override the capacity for the test without a second constructor.
        h.capacity = 4;
        h.samples = VecDeque::with_capacity(4);
        for i in 0..10 {
            h.record(&snapshot(Some(i as f64), None, None, None), Instant::now());
        }
        assert_eq!(h.len(), 4);
        let values: Vec<f64> =
            h.samples().iter().filter_map(|s| s.prompt_tokens_per_second).collect();
        assert_eq!(values, vec![6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn zero_capacity_does_not_panic() {
        let mut h = History { samples: VecDeque::new(), capacity: 0 };
        h.record(&snapshot(Some(1.0), None, None, None), Instant::now());
        h.record(&snapshot(None, None, None, None), Instant::now());
        assert_eq!(h.len(), 0, "nothing may be kept beyond the capacity");
        assert_eq!(h.capacity(), 0);
    }

    #[test]
    fn for_config_never_yields_zero_capacity() {
        let h = History::for_config(10, 0);
        assert!(h.capacity() >= 1);
        let h = History::for_config(3600, 100);
        assert_eq!(h.capacity(), MAX_HISTORY_SAMPLES);
    }

    #[test]
    fn gap_samples_keep_series_alignment() {
        let mut h = History::for_config(120, 500);
        let t0 = Instant::now();
        h.record(&snapshot(Some(10.0), Some(1.0), Some(1), None), t0);
        // A disconnected observation: everything missing, but the sample
        // still occupies its time position.
        h.record(
            &BackendSnapshot {
                connection: crate::domain::ConnectionState::Disconnected,
                ..Default::default()
            },
            t0,
        );
        h.record(&snapshot(Some(20.0), Some(2.0), None, Some(1)), t0);

        let samples = h.samples();
        assert_eq!(samples.len(), 3);
        // Same index, same observation, across all series:
        assert_eq!(samples[0].prompt_tokens_per_second, Some(10.0));
        assert!(samples[1].prompt_tokens_per_second.is_none());
        assert!(samples[1].generation_tokens_per_second.is_none());
        assert!(samples[1].active_requests.is_none());
        assert!(samples[1].queued_requests.is_none());
        assert_eq!(samples[2].prompt_tokens_per_second, Some(20.0));
        assert_eq!(samples[2].queued_requests, Some(1));
    }

    #[test]
    fn invalid_floating_point_values_are_sanitized() {
        let mut h = History::for_config(120, 500);
        h.record(&snapshot(Some(f64::NAN), None, None, None), Instant::now());
        h.record(&snapshot(None, Some(f64::INFINITY), None, None), Instant::now());
        h.record(&snapshot(None, Some(f64::NEG_INFINITY), None, None), Instant::now());
        h.record(&snapshot(Some(-5.0), None, None, None), Instant::now());
        h.record(&snapshot(Some(0.0), Some(1.0), None, None), Instant::now());

        let samples = h.samples();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples[0].prompt_tokens_per_second, None);
        assert_eq!(samples[1].generation_tokens_per_second, None);
        assert_eq!(samples[2].generation_tokens_per_second, None);
        assert_eq!(samples[3].prompt_tokens_per_second, None);
        // A genuine 0.0 is a real value, not a missing one.
        assert_eq!(samples[4].prompt_tokens_per_second, Some(0.0));
    }

    #[test]
    fn clear_empties_the_series() {
        let mut h = History::for_config(10, 100);
        h.record(&snapshot(Some(1.0), Some(2.0), Some(3), Some(4)), Instant::now());
        assert!(!h.is_empty());
        h.clear();
        assert!(h.is_empty());
    }
}
