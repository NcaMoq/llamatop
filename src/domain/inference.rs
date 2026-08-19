//! Workload phase and detection confidence.

use serde::{Deserialize, Serialize};

/// What the server is doing with the current request(s).
///
/// Note: queue state is deliberately NOT part of this enum. "Phase: DECODE,
/// Queued: 3" are independent facts and are reported separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadPhase {
    Idle,
    /// Inferred (estimated) prefill: processing is active and the prompt
    /// counter moved, but no direct prefill signal is available.
    PrefillLikely,
    Decode,
    /// Both prefill and decode evidence observed in the same window.
    Mixed,
    /// Work is active, but the exact phase cannot be determined.
    ProcessingUnknown,
}

impl WorkloadPhase {
    /// Display label. Estimated/uncertain states carry a marker so the UI never
    /// presents an inference as a fact.
    pub fn display(&self) -> &'static str {
        match self {
            WorkloadPhase::Idle => "IDLE",
            WorkloadPhase::PrefillLikely => "PREFILL*",
            WorkloadPhase::Decode => "DECODE",
            WorkloadPhase::Mixed => "MIXED",
            WorkloadPhase::ProcessingUnknown => "PROCESSING?",
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            WorkloadPhase::Idle => "idle",
            WorkloadPhase::PrefillLikely => "prefill_likely",
            WorkloadPhase::Decode => "decode",
            WorkloadPhase::Mixed => "mixed",
            WorkloadPhase::ProcessingUnknown => "processing_unknown",
        }
    }
}

/// How confident the detector is in a reported phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Confidence {
    /// Direct evidence (e.g. token counter growth).
    Exact,
    /// Strong indirect evidence.
    High,
    /// Inferred from available metrics.
    Estimated,
    /// No usable signal.
    Unknown,
}

impl Confidence {
    pub fn as_str(&self) -> &'static str {
        match self {
            Confidence::Exact => "exact",
            Confidence::High => "high",
            Confidence::Estimated => "estimated",
            Confidence::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimated_phases_are_marked() {
        assert_eq!(WorkloadPhase::PrefillLikely.display(), "PREFILL*");
        assert_eq!(WorkloadPhase::ProcessingUnknown.display(), "PROCESSING?");
        assert_eq!(WorkloadPhase::Decode.display(), "DECODE");
    }

    #[test]
    fn phases_serialize_lowercase() {
        assert_eq!(
            serde_json::to_string(&WorkloadPhase::PrefillLikely).unwrap(),
            "\"prefill_likely\""
        );
        assert_eq!(serde_json::to_string(&Confidence::Exact).unwrap(), "\"exact\"");
    }
}
