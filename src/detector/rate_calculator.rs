//! Rate calculation from cumulative counters.
//!
//! Rules:
//! - `None` when a previous sample is missing
//! - `None` when elapsed time is zero
//! - `None` (never negative) when the counter decreased (reset/restart)
//! - `None` for NaN, Infinity, or implausibly large values
//! - `Instant`-based timing only (immune to system clock changes)

use std::collections::VecDeque;
use std::time::Instant;

/// Upper bound on a plausible token rate; values above are treated as bad data.
const MAX_PLAUSIBLE_RATE: f64 = 1_000_000.0;

/// Computes a rate from two counter samples, with a bounded smoothed view.
#[derive(Debug, Clone, Default)]
pub struct RateCalculator {
    previous: Option<CounterSample>,
    /// Recent raw rates for the smoothed value (bounded; no unbounded growth).
    recent: VecDeque<f64>,
}

#[derive(Debug, Clone, Copy)]
struct CounterSample {
    value: u64,
    at: Instant,
}

impl RateCalculator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of samples kept for the smoothed rate.
    const SMOOTH_WINDOW: usize = 8;

    /// Feed a new counter sample and return `(raw_rate, smoothed_rate)`.
    ///
    /// Both are `None` on the first sample, after a reset, or when timing is
    /// degenerate. The counter must be monotonically non-decreasing; a
    /// decrease (server restart / counter reset) discards the previous sample
    /// and yields `None` for this interval instead of a bogus spike.
    pub fn update(&mut self, current: Option<u64>, now: Instant) -> (Option<f64>, Option<f64>) {
        match (self.previous.take(), current) {
            (Some(prev), Some(curr)) => {
                let elapsed = now.duration_since(prev.at);
                let raw = if elapsed.as_secs_f64() <= 0.0 {
                    None
                } else if curr < prev.value {
                    // Counter reset or server restart: no rate for this interval.
                    None
                } else {
                    let delta = curr.saturating_sub(prev.value) as f64;
                    let rate = delta / elapsed.as_secs_f64();
                    if rate.is_finite() && rate <= MAX_PLAUSIBLE_RATE {
                        Some(rate)
                    } else {
                        None
                    }
                };

                self.previous = Some(CounterSample { value: curr, at: now });

                let smoothed = if let Some(r) = raw {
                    self.recent.push_back(r);
                    while self.recent.len() > Self::SMOOTH_WINDOW {
                        self.recent.pop_front();
                    }
                    let sum: f64 = self.recent.iter().sum();
                    Some(sum / self.recent.len() as f64)
                } else {
                    self.recent.clear();
                    None
                };

                (raw, smoothed)
            }
            _ => {
                // No usable pair: keep current as the new baseline.
                if let Some(curr) = current {
                    self.previous = Some(CounterSample { value: curr, at: now });
                }
                self.recent.clear();
                (None, None)
            }
        }
    }

    /// Discard all state (reconnect / restart).
    pub fn reset(&mut self) {
        self.previous = None;
        self.recent.clear();
    }
}

/// One-shot rate from two explicit samples (used where no persistent
/// calculator is available, e.g. per-slot deltas).
pub fn rate_between(
    prev: Option<u64>,
    curr: Option<u64>,
    elapsed: Option<std::time::Duration>,
) -> Option<f64> {
    let (p, c) = (prev?, curr?);
    let elapsed = elapsed?;
    let secs = elapsed.as_secs_f64();
    if secs <= 0.0 {
        return None;
    }
    if c < p {
        return None;
    }
    let rate = (c - p) as f64 / secs;
    if rate.is_finite() && rate <= MAX_PLAUSIBLE_RATE {
        Some(rate)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn first_sample_has_no_rate() {
        let mut rc = RateCalculator::new();
        let (raw, smoothed) = rc.update(Some(100), Instant::now());
        assert_eq!(raw, None);
        assert_eq!(smoothed, None);
    }

    #[test]
    fn rate_from_delta_over_elapsed() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(100), start);
        let later = start + Duration::from_secs(2);
        let (raw, _) = rc.update(Some(140), later);
        // 40 tokens / 2s = 20 tok/s (approx; Instant may have advanced)
        let raw = raw.expect("rate expected");
        assert!((raw - 20.0).abs() < 1.0);
    }

    #[test]
    fn counter_reset_does_not_create_a_negative_or_huge_rate() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(1000), start);
        let later = start + Duration::from_millis(500);
        let (raw, smoothed) = rc.update(Some(5), later);
        assert_eq!(raw, None);
        assert_eq!(smoothed, None);

        // Next interval works normally from the reset baseline.
        let later2 = later + Duration::from_secs(1);
        let (raw2, _) = rc.update(Some(15), later2);
        assert_eq!(raw2, Some(10.0));
    }

    #[test]
    fn missing_current_sample_yields_none() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(10), start);
        let (raw, _) = rc.update(None, start + Duration::from_secs(1));
        assert_eq!(raw, None);
    }

    #[test]
    fn zero_elapsed_yields_none() {
        let mut rc = RateCalculator::new();
        let t = Instant::now();
        rc.update(Some(10), t);
        let (raw, _) = rc.update(Some(20), t);
        assert_eq!(raw, None);
    }

    #[test]
    fn smoothed_rate_averages_recent_raw_rates() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(0), start);
        let (r1, _) = rc.update(Some(10), start + Duration::from_secs(1));
        let (r2, s2) = rc.update(Some(30), start + Duration::from_secs(2));
        assert!(r1.is_some() && r2.is_some() && s2.is_some());
        let s2 = s2.unwrap();
        let expected = (r1.unwrap() + r2.unwrap()) / 2.0;
        assert!((s2 - expected).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_state() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(100), start);
        rc.reset();
        let (raw, _) = rc.update(Some(110), start + Duration::from_secs(1));
        assert_eq!(raw, None);
    }

    #[test]
    fn rate_between_rejects_decrease() {
        assert_eq!(rate_between(Some(1000), Some(5), Some(Duration::from_secs(1))), None);
        assert_eq!(rate_between(Some(100), Some(110), Some(Duration::from_secs(2))), Some(5.0));
        assert_eq!(rate_between(None, Some(110), Some(Duration::from_secs(2))), None);
        assert_eq!(rate_between(Some(100), None, Some(Duration::from_secs(2))), None);
        assert_eq!(rate_between(Some(100), Some(110), None), None);
    }
}
