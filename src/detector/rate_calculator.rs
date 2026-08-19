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

/// One interval's rate measurement.
///
/// `raw` is the instantaneous rate for this interval (None on first sample,
/// reset, or degenerate timing). `smoothed` is a bounded moving average of
/// recent raw rates (display only — must never drive state detection).
/// `increased` is true when the counter actually grew in this interval.
/// `reset` is true when a counter decrease (server restart) was detected.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RateObservation {
    pub raw: Option<f64>,
    pub smoothed: Option<f64>,
    pub increased: bool,
    pub reset: bool,
}

impl RateObservation {
    fn none() -> Self {
        Self::default()
    }
}

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

    /// Feed a new counter sample and return the interval's `RateObservation`.
    ///
    /// `raw` is None on the first sample, after a reset, or when timing is
    /// degenerate. The counter must be monotonically non-decreasing; a
    /// decrease (server restart / counter reset) discards the previous sample
    /// and yields a reset observation instead of a bogus spike.
    pub fn update(&mut self, current: Option<u64>, now: Instant) -> RateObservation {
        match (self.previous.take(), current) {
            (Some(prev), Some(curr)) => {
                let elapsed = now.duration_since(prev.at);
                if elapsed.as_secs_f64() <= 0.0 {
                    self.previous = Some(CounterSample { value: curr, at: now });
                    return RateObservation::none();
                }
                if curr < prev.value {
                    // Counter reset or server restart.
                    self.previous = Some(CounterSample { value: curr, at: now });
                    self.recent.clear();
                    return RateObservation { reset: true, ..Default::default() };
                }

                let delta = curr.saturating_sub(prev.value) as f64;
                let rate = delta / elapsed.as_secs_f64();
                let raw =
                    if rate.is_finite() && rate <= MAX_PLAUSIBLE_RATE { Some(rate) } else { None };

                self.previous = Some(CounterSample { value: curr, at: now });

                let (smoothed, increased) = match raw {
                    Some(r) => {
                        self.recent.push_back(r);
                        while self.recent.len() > Self::SMOOTH_WINDOW {
                            self.recent.pop_front();
                        }
                        let sum: f64 = self.recent.iter().sum();
                        (Some(sum / self.recent.len() as f64), r > 0.0)
                    }
                    None => {
                        self.recent.clear();
                        (None, false)
                    }
                };

                RateObservation { raw, smoothed, increased, reset: false }
            }
            _ => {
                // No usable pair: keep current as the new baseline.
                if let Some(curr) = current {
                    self.previous = Some(CounterSample { value: curr, at: now });
                }
                self.recent.clear();
                RateObservation::none()
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
        let obs = rc.update(Some(100), Instant::now());
        assert_eq!(obs.raw, None);
        assert_eq!(obs.smoothed, None);
        assert!(!obs.increased);
        assert!(!obs.reset);
    }

    #[test]
    fn rate_from_delta_over_elapsed() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(100), start);
        let later = start + Duration::from_secs(2);
        let obs = rc.update(Some(140), later);
        // 40 tokens / 2s = 20 tok/s (approx; Instant may have advanced)
        let raw = obs.raw.expect("rate expected");
        assert!((raw - 20.0).abs() < 1.0);
        assert!(obs.increased);
    }

    #[test]
    fn counter_reset_does_not_create_a_negative_or_huge_rate() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(1000), start);
        let later = start + Duration::from_millis(500);
        let obs = rc.update(Some(5), later);
        assert_eq!(obs.raw, None);
        assert_eq!(obs.smoothed, None);
        assert!(obs.reset);
        assert!(!obs.increased);

        // Next interval works normally from the reset baseline.
        let later2 = later + Duration::from_secs(1);
        let obs2 = rc.update(Some(15), later2);
        assert_eq!(obs2.raw, Some(10.0));
        assert!(obs2.increased);
    }

    #[test]
    fn missing_current_sample_yields_none() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(10), start);
        let obs = rc.update(None, start + Duration::from_secs(1));
        assert_eq!(obs.raw, None);
    }

    #[test]
    fn zero_elapsed_yields_none() {
        let mut rc = RateCalculator::new();
        let t = Instant::now();
        rc.update(Some(10), t);
        let obs = rc.update(Some(20), t);
        assert_eq!(obs.raw, None);
    }

    #[test]
    fn smoothed_rate_averages_recent_raw_rates() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(0), start);
        let r1 = rc.update(Some(10), start + Duration::from_secs(1));
        let obs2 = rc.update(Some(30), start + Duration::from_secs(2));
        let r2 = obs2.raw.expect("raw rate");
        assert!(r1.raw.is_some() && obs2.raw.is_some());
        let s2 = obs2.smoothed.expect("smoothed");
        let expected = (r1.raw.unwrap() + r2) / 2.0;
        assert!((s2 - expected).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_state() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(100), start);
        rc.reset();
        let obs = rc.update(Some(110), start + Duration::from_secs(1));
        assert_eq!(obs.raw, None);
    }

    #[test]
    fn rate_between_rejects_decrease() {
        assert_eq!(rate_between(Some(1000), Some(5), Some(Duration::from_secs(1))), None);
        assert_eq!(rate_between(Some(100), Some(110), Some(Duration::from_secs(2))), Some(5.0));
        assert_eq!(rate_between(None, Some(110), Some(Duration::from_secs(2))), None);
        assert_eq!(rate_between(Some(100), None, Some(Duration::from_secs(2))), None);
        assert_eq!(rate_between(Some(100), Some(110), None), None);
    }

    #[test]
    fn no_increase_when_counter_unchanged() {
        let mut rc = RateCalculator::new();
        let start = Instant::now();
        rc.update(Some(100), start);
        let obs = rc.update(Some(100), start + Duration::from_secs(1));
        assert_eq!(obs.raw, Some(0.0));
        assert!(!obs.increased);
    }
}
