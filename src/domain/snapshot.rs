//! The normalized backend snapshot.
//!
//! This is the single structure the detector, history, and UI operate on.
//! It is produced by normalizing raw API responses (see `backend::llamacpp::normalize`)
//! and enriched by the detector (phases, rates). Raw HTTP/Prometheus types
//! never leak past the backend layer.

use serde::{Deserialize, Serialize};

use super::connection::ConnectionState;
use super::gpu::GpuSnapshot;
use super::inference::{Confidence, WorkloadPhase};
use super::server::ServerState;
use super::slot::SlotSnapshot;
use super::system::SystemSnapshot;

/// One normalized observation of the inference backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendSnapshot {
    /// Monotonic wall-clock (not used for deltas; deltas use `sample_index`
    /// spacing plus the detector's own `Instant`-based timing).
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Sample counter; increments by one per observation.
    pub sample_index: u64,
    pub connection: ConnectionState,
    pub server: ServerState,
    /// Detected server-wide workload phase (filled in by the detector).
    pub workload_phase: WorkloadPhase,
    pub workload_confidence: Confidence,
    pub active_requests: Option<u64>,
    pub queued_requests: Option<u64>,
    pub slots: Vec<SlotSnapshot>,
    pub prompt_tokens_total: Option<u64>,
    pub generation_tokens_total: Option<u64>,
    /// Delta-based rates, filled in by the detector (None until a delta
    /// exists). Distinct from the server-reported averages below.
    pub prompt_tokens_per_second: Option<f64>,
    pub generation_tokens_per_second: Option<f64>,
    /// Average throughput reported by the server itself (`/metrics` gauges
    /// `llamacpp:prompt_tokens_seconds` / `llamacpp:predicted_tokens_seconds`,
    /// cumulative since process start). Used as a fallback when no delta is
    /// available (e.g. one-shot snapshot).
    pub prompt_tokens_per_second_reported: Option<f64>,
    pub generation_tokens_per_second_reported: Option<f64>,
    pub context_max_tokens: Option<u64>,
    pub model_name: Option<String>,
    pub model_path: Option<String>,
    pub total_slots: Option<u64>,
    pub build_info: Option<String>,
    pub is_sleeping: Option<bool>,
    pub speculative: SpeculativeStats,
    /// Observed server process start time (epoch secs) when the backend
    /// reports one; used for restart detection.
    pub server_start_unix: Option<u64>,
    #[serde(default)]
    pub gpu: Vec<GpuSnapshot>,
    #[serde(default)]
    pub system: Option<SystemSnapshot>,
    /// Short, redacted reason string when the connection/server is in an error
    /// state (never contains secrets or full response bodies).
    pub error: Option<String>,
}

/// Speculative decoding aggregates from the metrics endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeculativeStats {
    pub draft_tokens_total: Option<u64>,
    pub accepted_tokens_total: Option<u64>,
    pub drafts_total: Option<u64>,
}

impl SpeculativeStats {
    /// Fraction of draft tokens accepted by the target model (0.0..=1.0),
    /// when both counters are known and drafts > 0.
    pub fn acceptance_rate(&self) -> Option<f64> {
        let draft = self.draft_tokens_total?;
        let accepted = self.accepted_tokens_total?;
        if draft == 0 {
            return None;
        }
        Some(accepted as f64 / draft as f64)
    }
}

impl Default for BackendSnapshot {
    fn default() -> Self {
        Self {
            timestamp: chrono::DateTime::from_timestamp(0, 0).unwrap(),
            sample_index: 0,
            connection: ConnectionState::Disconnected,
            server: ServerState::Unknown,
            workload_phase: WorkloadPhase::ProcessingUnknown,
            workload_confidence: Confidence::Unknown,
            active_requests: None,
            queued_requests: None,
            slots: Vec::new(),
            prompt_tokens_total: None,
            generation_tokens_total: None,
            prompt_tokens_per_second: None,
            generation_tokens_per_second: None,
            prompt_tokens_per_second_reported: None,
            generation_tokens_per_second_reported: None,
            context_max_tokens: None,
            model_name: None,
            model_path: None,
            total_slots: None,
            build_info: None,
            is_sleeping: None,
            speculative: SpeculativeStats::default(),
            server_start_unix: None,
            gpu: Vec::new(),
            system: None,
            error: None,
        }
    }
}

impl BackendSnapshot {
    /// True when any slot is currently processing a task.
    pub fn any_slot_processing(&self) -> bool {
        self.slots.iter().any(|s| s.is_processing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_round_trips_through_json() {
        let s = BackendSnapshot {
            model_name: Some("Qwen".into()),
            active_requests: Some(1),
            ..Default::default()
        };
        let raw = serde_json::to_string(&s).unwrap();
        let back: BackendSnapshot = serde_json::from_str(&raw).unwrap();
        assert_eq!(back.model_name.as_deref(), Some("Qwen"));
        assert_eq!(back.active_requests, Some(1));
    }

    #[test]
    fn speculative_acceptance_rate_is_accepted_over_draft_tokens() {
        let stats = SpeculativeStats {
            draft_tokens_total: Some(10),
            accepted_tokens_total: Some(6),
            drafts_total: Some(0),
        };
        assert!((stats.acceptance_rate().unwrap() - 0.6).abs() < 1e-9);

        let zero = SpeculativeStats { draft_tokens_total: Some(0), ..stats };
        assert!(zero.acceptance_rate().is_none());
    }
}
