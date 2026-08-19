//! State detection and rate calculation over normalized snapshots.
//!
//! Pure logic: no I/O, no terminal, no HTTP. Operates on `BackendSnapshot`
//! values and returns stabilized snapshots with phases, confidence, and rates.

pub mod rate_calculator;
pub mod stability;
pub mod state_detector;

pub use rate_calculator::RateCalculator;
pub use state_detector::StateDetector;
