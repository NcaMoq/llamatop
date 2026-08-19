//! Per-slot state.

use serde::{Deserialize, Serialize};

use super::inference::WorkloadPhase;

/// Per-slot workload phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotPhase {
    Idle,
    PrefillLikely,
    Decode,
    ProcessingUnknown,
}

impl SlotPhase {
    pub fn display(&self) -> &'static str {
        match self {
            SlotPhase::Idle => "IDLE",
            SlotPhase::PrefillLikely => "PREFILL*",
            SlotPhase::Decode => "DECODE",
            SlotPhase::ProcessingUnknown => "PROCESSING?",
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            SlotPhase::Idle => "idle",
            SlotPhase::PrefillLikely => "prefill_likely",
            SlotPhase::Decode => "decode",
            SlotPhase::ProcessingUnknown => "processing_unknown",
        }
    }

    pub fn to_workload_phase(self) -> WorkloadPhase {
        match self {
            SlotPhase::Idle => WorkloadPhase::Idle,
            SlotPhase::PrefillLikely => WorkloadPhase::PrefillLikely,
            SlotPhase::Decode => WorkloadPhase::Decode,
            SlotPhase::ProcessingUnknown => WorkloadPhase::ProcessingUnknown,
        }
    }
}

/// A snapshot of one server slot.
///
/// All counters are the *current cumulative* values reported by the server;
/// rates are computed by the detector from deltas (never here).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SlotSnapshot {
    pub id: u32,
    /// Task id currently occupying the slot, if any.
    pub task_id: Option<u64>,
    pub is_processing: bool,
    /// Context window size for this slot.
    pub n_ctx: Option<u64>,
    /// Tokens currently in the context (prompt + generated), if known.
    pub n_tokens: Option<u64>,
    /// Prompt tokens for the current task, if any.
    pub n_prompt_tokens: Option<u64>,
    /// Prompt tokens processed so far for the current task, if any.
    pub n_prompt_tokens_processed: Option<u64>,
    /// Generated (decoded) tokens for the current task, if any.
    pub n_decoded: Option<u64>,
    /// Whether speculative decoding is enabled for this slot.
    pub speculative: bool,
    /// Detected phase for this slot (filled in by the detector).
    pub phase: SlotPhase,
}
