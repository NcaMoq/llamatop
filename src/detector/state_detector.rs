//! Workload phase and connection state detection.
//!
//! Phase detection uses deltas between consecutive observations, never a
//! single sample:
//!
//! - `Decode`: a slot's `n_decoded` grew (Exact), or the server-wide
//!   generation counter grew (High, server-wide only).
//! - `PrefillLikely`: processing is active, the decoded count did not grow,
//!   and the prompt counter did (Estimated; two consecutive observations).
//! - `Mixed`: prefill and decode evidence in the same observation window
//!   (Estimated when derived from server-wide counters).
//! - `ProcessingUnknown`: work is active but no counter moved (two
//!   consecutive observations before it is shown).
//! - `Idle`: no active requests and no slot processing (two consecutive
//!   observations).
//!
//! The server-wide generation counter can never mark an *individual* slot as
//! Decode; only that slot's own counter growth does.
//!
//! Timing uses `std::time::Instant` (monotonic), so system clock changes
//! cannot corrupt rates.

use std::collections::HashMap;
use std::time::Instant;

use crate::domain::{
    BackendSnapshot, Confidence, ConnectionState, ServerState, SlotPhase, WorkloadPhase,
};

use super::rate_calculator::{RateCalculator, RateObservation};
use super::stability::{FailureWindow, StableCandidate};

/// Consecutive failures (or streak duration) before a transient error is
/// escalated from `Reconnecting` to `Disconnected`.
const MAX_TRANSIENT_FAILURES: u8 = 3;
const MAX_TRANSIENT_DURATION: std::time::Duration = std::time::Duration::from_secs(10);

/// Baseline counters for one slot, used to compute per-slot deltas.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct SlotBaseline {
    task_id: Option<u64>,
    n_decoded: Option<u64>,
    n_prompt_processed: Option<u64>,
}

/// The state detector. Feed it normalized snapshots; it returns stabilized
/// snapshots with phases, confidence, and rates filled in.
#[derive(Debug, Default)]
pub struct StateDetector {
    previous: Option<BackendSnapshot>,
    prompt_rate: RateCalculator,
    generation_rate: RateCalculator,
    slot_baselines: HashMap<u32, SlotBaseline>,
    /// Phases that require two consecutive observations before being shown.
    pending: StableCandidate<WorkloadPhase>,
    connection_failures: FailureWindow,
}

impl StateDetector {
    pub fn new() -> Self {
        Self { pending: StableCandidate::new(2), ..Default::default() }
    }

    /// Drop all learned state (reconnect, restart, manual reset).
    pub fn reset(&mut self) {
        self.previous = None;
        self.prompt_rate.reset();
        self.generation_rate.reset();
        self.slot_baselines.clear();
        self.pending.reset();
        self.connection_failures.reset();
    }

    /// Process one normalized observation and return the stabilized snapshot.
    pub fn update(&mut self, raw: BackendSnapshot, now: Instant) -> BackendSnapshot {
        let mut out = raw;
        let prev = self.previous.clone();

        let prev_connection = prev.as_ref().map(|p| p.connection);
        let reconnected = out.connection == ConnectionState::Connected
            && matches!(
                prev_connection,
                Some(ConnectionState::Reconnecting | ConnectionState::Disconnected)
            );
        if reconnected {
            self.reset();
        }

        // Server restart detection: the process start time changed.
        if let (Some(previous), Some(current)) =
            (prev.as_ref().and_then(|p| p.server_start_unix), out.server_start_unix)
        {
            if current != previous {
                self.reset();
            }
        }

        out.connection = self.update_connection_state(out.connection, now);

        if out.connection == ConnectionState::Connected && out.server == ServerState::Ready {
            let (prompt_obs, gen_obs) = self.update_server_rates(&mut out, now);
            let slot_phases = self.update_slot_phases(&mut out);
            let candidate = self.observe_workload(&out, slot_phases, &prompt_obs, &gen_obs);
            let (phase, confidence) = self.apply_hysteresis(candidate, now);
            out.workload_phase = phase;
            out.workload_confidence = confidence;
        } else if out.connection == ConnectionState::Connected {
            // Loading / sleeping / unavailable: no workload is running, but we
            // keep counter baselines moving so the first post-load interval is
            // not mistaken for a huge rate.
            let _ = self.update_server_rates(&mut out, now);
            self.update_slot_baselines(&out);
            out.workload_phase = WorkloadPhase::Idle;
            out.workload_confidence = Confidence::Unknown;
            self.pending.reset();
        } else {
            // Reconnecting / Disconnected / Error: nothing to observe and no
            // trustworthy server state either.
            out.server = ServerState::Unavailable;
            out.workload_phase = WorkloadPhase::Idle;
            out.workload_confidence = Confidence::Unknown;
        }

        self.previous = Some(out.clone());
        out
    }

    fn update_connection_state(
        &mut self,
        observed: ConnectionState,
        now: Instant,
    ) -> ConnectionState {
        if observed == ConnectionState::Connected {
            self.connection_failures.record_success();
            return ConnectionState::Connected;
        }
        if observed == ConnectionState::Disconnected {
            return ConnectionState::Disconnected;
        }
        // A transport failure: give it a short transient window before
        // declaring the connection lost.
        self.connection_failures.record_failure(now);
        let duration = self.connection_failures.duration(now).unwrap_or_default();
        if self.connection_failures.failures() >= MAX_TRANSIENT_FAILURES
            || duration >= MAX_TRANSIENT_DURATION
        {
            ConnectionState::Disconnected
        } else {
            ConnectionState::Reconnecting
        }
    }

    /// Update server-wide rates; returns the raw observations for this
    /// interval. Display values use the smoothed rate (falling back to raw);
    /// phase evidence uses only the raw observation.
    fn update_server_rates(
        &mut self,
        out: &mut BackendSnapshot,
        now: Instant,
    ) -> (RateObservation, RateObservation) {
        let prompt_obs = self.prompt_rate.update(out.prompt_tokens_total, now);
        let gen_obs = self.generation_rate.update(out.generation_tokens_total, now);

        // Display value: smoothed when available, else raw. Raw and smoothed
        // are kept separately; only raw drives phase detection.
        out.prompt_tokens_per_second = prompt_obs.smoothed.or(prompt_obs.raw);
        out.generation_tokens_per_second = gen_obs.smoothed.or(gen_obs.raw);

        (prompt_obs, gen_obs)
    }

    /// Detect each slot's phase from its own counter deltas.
    /// Returns the set of distinct processing phases observed this interval.
    fn update_slot_phases(&mut self, out: &mut BackendSnapshot) -> Vec<SlotPhase> {
        let mut phases = Vec::new();
        for slot in &mut out.slots {
            let baseline = self
                .slot_baselines
                .get(&slot.id)
                .filter(|b| b.task_id == slot.task_id && slot.task_id.is_some())
                .copied();

            let phase = Self::slot_phase(baseline, slot);
            slot.phase = phase;
            if slot.is_processing && !phases.contains(&phase) {
                phases.push(phase);
            }
        }
        self.update_slot_baselines(out);
        phases
    }

    fn update_slot_baselines(&mut self, out: &BackendSnapshot) {
        let current_ids: std::collections::HashSet<u32> = out.slots.iter().map(|s| s.id).collect();
        self.slot_baselines.retain(|id, _| current_ids.contains(id));
        for slot in &out.slots {
            if slot.is_processing {
                self.slot_baselines.insert(
                    slot.id,
                    SlotBaseline {
                        task_id: slot.task_id,
                        n_decoded: slot.n_decoded,
                        n_prompt_processed: slot.n_prompt_tokens_processed,
                    },
                );
            } else if self.slot_baselines.contains_key(&slot.id) {
                // Task finished: clear the baseline so the next task starts fresh.
                self.slot_baselines.remove(&slot.id);
            }
        }
    }

    fn slot_phase(baseline: Option<SlotBaseline>, slot: &crate::domain::SlotSnapshot) -> SlotPhase {
        if !slot.is_processing {
            return SlotPhase::Idle;
        }
        match baseline {
            Some(b) => {
                let decoded_up = match (b.n_decoded, slot.n_decoded) {
                    (Some(prev), Some(curr)) => curr > prev,
                    _ => false,
                };
                let prompt_up = match (b.n_prompt_processed, slot.n_prompt_tokens_processed) {
                    (Some(prev), Some(curr)) => curr > prev,
                    _ => false,
                };
                if decoded_up {
                    SlotPhase::Decode
                } else if prompt_up {
                    SlotPhase::PrefillLikely
                } else {
                    SlotPhase::ProcessingUnknown
                }
            }
            // First observation of this task: no delta is available yet.
            None => SlotPhase::ProcessingUnknown,
        }
    }

    /// Combine per-slot evidence with server-wide counter evidence into a
    /// single candidate phase + confidence for this observation.
    ///
    /// Phase evidence uses only the *raw* interval observations: a smoothed
    /// rate from earlier intervals must never keep Decode active after the
    /// generation counter stops growing.
    fn observe_workload(
        &self,
        out: &BackendSnapshot,
        slot_phases: Vec<SlotPhase>,
        prompt_obs: &RateObservation,
        gen_obs: &RateObservation,
    ) -> (WorkloadPhase, Confidence) {
        let has_decode_slot = slot_phases.contains(&SlotPhase::Decode);
        let has_prefill_slot = slot_phases.contains(&SlotPhase::PrefillLikely);
        let has_unknown_slot = slot_phases.contains(&SlotPhase::ProcessingUnknown);

        // Direct slot evidence has priority.
        if has_decode_slot && has_prefill_slot {
            return (WorkloadPhase::Mixed, Confidence::High);
        }
        if has_decode_slot {
            return (WorkloadPhase::Decode, Confidence::Exact);
        }
        if has_prefill_slot {
            return (WorkloadPhase::PrefillLikely, Confidence::Estimated);
        }

        let prompt_up = prompt_obs.increased;
        let gen_up = gen_obs.increased;
        let any_processing = has_unknown_slot || out.any_slot_processing();
        let active = out.active_requests.unwrap_or(0);

        if any_processing || active > 0 {
            if prompt_up && gen_up {
                // Both counters moved in the same interval: mixed workload.
                return (WorkloadPhase::Mixed, Confidence::Estimated);
            }
            if gen_up {
                // Server-wide generation growth: decode candidate for the
                // server as a whole (not for any individual slot).
                return (WorkloadPhase::Decode, Confidence::High);
            }
            if prompt_up {
                return (WorkloadPhase::PrefillLikely, Confidence::Estimated);
            }
            return (WorkloadPhase::ProcessingUnknown, Confidence::Unknown);
        }

        // No activity at all (idle candidate; hysteresis applied by caller).
        (WorkloadPhase::Idle, Confidence::High)
    }

    /// Apply hysteresis rules to a candidate phase:
    /// - Decode, Mixed: apply immediately (strong simultaneous evidence)
    /// - Leaving Decode/Mixed: apply the new candidate immediately, so a
    ///   stopped counter never keeps Decode alive (smoothing must not extend
    ///   a phase past its raw evidence)
    /// - PrefillLikely, Idle, ProcessingUnknown: two consecutive observations
    fn apply_hysteresis(
        &mut self,
        candidate: (WorkloadPhase, Confidence),
        _now: Instant,
    ) -> (WorkloadPhase, Confidence) {
        let previous = self
            .previous
            .as_ref()
            .map(|p| (p.workload_phase, p.workload_confidence))
            .unwrap_or((WorkloadPhase::ProcessingUnknown, Confidence::Unknown));

        match candidate.0 {
            WorkloadPhase::Decode | WorkloadPhase::Mixed => {
                self.pending.reset();
                candidate
            }
            WorkloadPhase::PrefillLikely
            | WorkloadPhase::Idle
            | WorkloadPhase::ProcessingUnknown => {
                if matches!(previous.0, WorkloadPhase::Decode | WorkloadPhase::Mixed) {
                    // The decode evidence has stopped; exit without waiting
                    // for hysteresis so the phase tracks the raw counters.
                    self.pending.reset();
                    return candidate;
                }
                match self.pending.feed(Some(&candidate.0)) {
                    Some(applied) => (applied, candidate.1),
                    None => {
                        // Not stable yet: keep showing the previous state.
                        // With no previous state, do not guess: report that
                        // the phase is not yet determined.
                        previous
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::SlotSnapshot;
    use std::time::Duration;

    fn slot(
        id: u32,
        processing: bool,
        task: Option<u64>,
        decoded: Option<u64>,
        prompt: Option<u64>,
    ) -> SlotSnapshot {
        SlotSnapshot {
            id,
            task_id: task,
            is_processing: processing,
            n_ctx: Some(4096),
            n_tokens: None,
            n_prompt_tokens: prompt,
            n_prompt_tokens_processed: prompt,
            n_decoded: decoded,
            speculative: false,
            phase: SlotPhase::Idle,
        }
    }

    fn ready(
        active: Option<u64>,
        queued: Option<u64>,
        prompt_total: Option<u64>,
        gen_total: Option<u64>,
        slots: Vec<SlotSnapshot>,
    ) -> BackendSnapshot {
        BackendSnapshot {
            connection: ConnectionState::Connected,
            server: ServerState::Ready,
            active_requests: active,
            queued_requests: queued,
            prompt_tokens_total: prompt_total,
            generation_tokens_total: gen_total,
            slots,
            ..Default::default()
        }
    }

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    // --- Disconnected / transient ---

    #[test]
    fn first_transport_failure_is_reconnecting_not_disconnected() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        let mut snap = ready(None, None, None, None, vec![]);
        snap.connection = ConnectionState::Error;
        let out = d.update(snap, base);
        assert_eq!(out.connection, ConnectionState::Reconnecting);
    }

    #[test]
    fn repeated_failures_escalate_to_disconnected() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        for i in 0..MAX_TRANSIENT_FAILURES {
            let mut snap = ready(None, None, None, None, vec![]);
            snap.connection = ConnectionState::Error;
            let out = d.update(snap, at(base, i as u64 * 100));
            if i < MAX_TRANSIENT_FAILURES - 1 {
                assert_eq!(out.connection, ConnectionState::Reconnecting);
            } else {
                assert_eq!(out.connection, ConnectionState::Disconnected);
            }
        }
    }

    #[test]
    fn recovery_after_transient_failure_resets_state() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        let mut snap = ready(None, None, None, None, vec![]);
        snap.connection = ConnectionState::Error;
        d.update(snap, base);

        // Back online: counters restart from scratch; no rate on first sample.
        let out = d.update(ready(Some(0), Some(0), Some(100), Some(50), vec![]), at(base, 5000));
        assert_eq!(out.connection, ConnectionState::Connected);
        assert_eq!(out.prompt_tokens_per_second, None);
        assert_eq!(out.generation_tokens_per_second, None);
    }

    // --- Loading / sleeping ---

    #[test]
    fn loading_state_applies_immediately() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        let mut snap = ready(Some(1), Some(0), Some(10), Some(10), vec![]);
        snap.server = ServerState::Loading;
        let out = d.update(snap, base);
        assert_eq!(out.server, ServerState::Loading);
        assert_eq!(out.workload_phase, WorkloadPhase::Idle);
        assert_eq!(out.workload_confidence, Confidence::Unknown);
    }

    #[test]
    fn sleeping_state_applies_immediately() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        let mut snap = ready(None, None, None, None, vec![]);
        snap.server = ServerState::Sleeping;
        let out = d.update(snap, base);
        assert_eq!(out.server, ServerState::Sleeping);
    }

    // --- Idle ---

    #[test]
    fn idle_requires_two_consecutive_observations() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        let out1 = d.update(ready(Some(0), Some(0), Some(0), Some(0), vec![]), at(base, 100));
        assert_ne!(out1.workload_phase, WorkloadPhase::Idle, "first observation is not idle yet");
        let out2 = d.update(ready(Some(0), Some(0), Some(0), Some(0), vec![]), at(base, 600));
        assert_eq!(out2.workload_phase, WorkloadPhase::Idle);
        assert_eq!(out2.workload_confidence, Confidence::High);
    }

    // --- Decode ---

    #[test]
    fn incrementing_decoded_tokens_is_decode() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        // First observation establishes the baseline (task just started).
        let out1 = d.update(
            ready(
                Some(1),
                Some(0),
                Some(100),
                Some(10),
                vec![slot(0, true, Some(1), Some(10), Some(100))],
            ),
            at(base, 100),
        );
        assert_eq!(out1.slots[0].phase, SlotPhase::ProcessingUnknown);

        // Decoded tokens grew: exact Decode, applied immediately.
        let out2 = d.update(
            ready(
                Some(1),
                Some(0),
                Some(100),
                Some(11),
                vec![slot(0, true, Some(1), Some(11), Some(100))],
            ),
            at(base, 600),
        );
        assert_eq!(out2.slots[0].phase, SlotPhase::Decode);
        assert_eq!(out2.workload_phase, WorkloadPhase::Decode);
        assert_eq!(out2.workload_confidence, Confidence::Exact);
    }

    #[test]
    fn smoothed_generation_rate_does_not_keep_decode_active_after_raw_delta_stops() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        // Sample 1: establish the baseline (one active request).
        d.update(ready(Some(1), Some(0), Some(0), Some(0), vec![]), at(base, 0));
        // Sample 2: generation counter increases -> Decode (server-wide, High).
        let out2 = d.update(ready(Some(1), Some(0), Some(0), Some(500), vec![]), at(base, 500));
        assert_eq!(out2.workload_phase, WorkloadPhase::Decode);
        // Sample 3: generation counter does not increase. The smoothed rate
        // may still be positive, but the phase must not remain Decode solely
        // because of smoothing.
        let out3 = d.update(ready(Some(1), Some(0), Some(0), Some(500), vec![]), at(base, 1000));
        assert_ne!(out3.workload_phase, WorkloadPhase::Decode);
        // The smoothed display rate can still be positive while the phase has moved on.
        assert!(out3.generation_tokens_per_second.is_some_and(|r| r > 0.0));
    }

    #[test]
    fn queued_requests_do_not_replace_decode_phase() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(
            ready(
                Some(1),
                Some(2),
                Some(100),
                Some(10),
                vec![slot(0, true, Some(1), Some(10), Some(100))],
            ),
            at(base, 100),
        );
        let out = d.update(
            ready(
                Some(1),
                Some(2),
                Some(100),
                Some(15),
                vec![slot(0, true, Some(1), Some(15), Some(100))],
            ),
            at(base, 600),
        );
        assert_eq!(out.workload_phase, WorkloadPhase::Decode);
        assert_eq!(out.queued_requests, Some(2));
    }

    #[test]
    fn server_wide_generation_growth_is_decode_candidate_high_confidence() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        // A processing slot whose own counters do not move (no task id), but
        // the server-wide generation counter grows.
        d.update(
            ready(Some(1), Some(0), Some(100), Some(10), vec![slot(0, true, None, None, None)]),
            at(base, 100),
        );
        let out2 = d.update(
            ready(Some(1), Some(0), Some(100), Some(60), vec![slot(0, true, None, None, None)]),
            at(base, 600),
        );
        assert_eq!(out2.workload_phase, WorkloadPhase::Decode);
        assert_eq!(out2.workload_confidence, Confidence::High);
        // The individual slot must NOT be marked Decode without its own evidence.
        assert_eq!(out2.slots[0].phase, SlotPhase::ProcessingUnknown);
    }

    // --- Prefill ---

    #[test]
    fn processing_without_decode_and_with_prompt_delta_is_prefill_likely() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        let out1 = d.update(
            ready(
                Some(1),
                Some(0),
                Some(100),
                Some(0),
                vec![slot(0, true, Some(1), Some(0), Some(100))],
            ),
            at(base, 100),
        );
        // Prompt counter moves, decoded does not.
        let out2 = d.update(
            ready(
                Some(1),
                Some(0),
                Some(500),
                Some(0),
                vec![slot(0, true, Some(1), Some(0), Some(500))],
            ),
            at(base, 600),
        );
        // Prefill requires two consecutive observations.
        assert_ne!(out2.workload_phase, WorkloadPhase::PrefillLikely);
        let out3 = d.update(
            ready(
                Some(1),
                Some(0),
                Some(900),
                Some(0),
                vec![slot(0, true, Some(1), Some(0), Some(900))],
            ),
            at(base, 1100),
        );
        assert_eq!(out3.workload_phase, WorkloadPhase::PrefillLikely);
        assert_eq!(out3.workload_confidence, Confidence::Estimated);
        assert_eq!(out3.slots[0].phase, SlotPhase::PrefillLikely);
        let _ = out1;
    }

    #[test]
    fn prefill_with_queued_requests_keeps_queue_count() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(
            ready(
                Some(1),
                Some(3),
                Some(100),
                Some(0),
                vec![slot(0, true, Some(1), Some(0), Some(100))],
            ),
            at(base, 100),
        );
        d.update(
            ready(
                Some(1),
                Some(3),
                Some(500),
                Some(0),
                vec![slot(0, true, Some(1), Some(0), Some(500))],
            ),
            at(base, 600),
        );
        let out = d.update(
            ready(
                Some(1),
                Some(3),
                Some(900),
                Some(0),
                vec![slot(0, true, Some(1), Some(0), Some(900))],
            ),
            at(base, 1100),
        );
        assert_eq!(out.workload_phase, WorkloadPhase::PrefillLikely);
        assert_eq!(out.queued_requests, Some(3));
    }

    // --- Mixed ---

    #[test]
    fn mixed_when_one_slot_prefills_and_another_decodes() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(
            ready(
                Some(2),
                Some(0),
                Some(100),
                Some(10),
                vec![
                    slot(0, true, Some(1), Some(0), Some(100)),
                    slot(1, true, Some(2), Some(10), Some(50)),
                ],
            ),
            at(base, 100),
        );
        // Slot 0 prompt grows (prefill), slot 1 decoded grows (decode).
        let out = d.update(
            ready(
                Some(2),
                Some(0),
                Some(900),
                Some(60),
                vec![
                    slot(0, true, Some(1), Some(0), Some(900)),
                    slot(1, true, Some(2), Some(60), Some(50)),
                ],
            ),
            at(base, 600),
        );
        assert_eq!(out.workload_phase, WorkloadPhase::Mixed);
        assert_eq!(out.workload_confidence, Confidence::High);
        assert_eq!(out.slots[0].phase, SlotPhase::PrefillLikely);
        assert_eq!(out.slots[1].phase, SlotPhase::Decode);
    }

    #[test]
    fn mixed_from_server_wide_counters_is_estimated() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(
            ready(Some(1), Some(0), Some(100), Some(10), vec![slot(0, true, None, None, None)]),
            at(base, 100),
        );
        let out = d.update(
            ready(Some(1), Some(0), Some(900), Some(60), vec![slot(0, true, None, None, None)]),
            at(base, 600),
        );
        assert_eq!(out.workload_phase, WorkloadPhase::Mixed);
        assert_eq!(out.workload_confidence, Confidence::Estimated);
    }

    // --- ProcessingUnknown ---

    #[test]
    fn processing_without_counter_evidence_is_processing_unknown() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(
            ready(Some(1), Some(0), Some(100), Some(10), vec![slot(0, true, None, None, None)]),
            at(base, 100),
        );
        let out = d.update(
            ready(Some(1), Some(0), Some(100), Some(10), vec![slot(0, true, None, None, None)]),
            at(base, 600),
        );
        assert_eq!(out.workload_phase, WorkloadPhase::ProcessingUnknown);
        assert_eq!(out.workload_confidence, Confidence::Unknown);
    }

    // --- Restart / counter reset ---

    #[test]
    fn counter_reset_does_not_create_a_huge_rate() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(ready(Some(0), Some(0), Some(1000), Some(500), vec![]), at(base, 100));
        let out = d.update(ready(Some(0), Some(0), Some(5), Some(2), vec![]), at(base, 600));
        assert_eq!(out.prompt_tokens_per_second, None);
        assert_eq!(out.generation_tokens_per_second, None);
    }

    #[test]
    fn server_restart_resets_detector() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        let mut s1 = ready(
            Some(1),
            Some(0),
            Some(100),
            Some(10),
            vec![slot(0, true, Some(1), Some(10), Some(100))],
        );
        s1.server_start_unix = Some(1000);
        d.update(s1, at(base, 100));
        let mut s2 = ready(
            Some(1),
            Some(0),
            Some(3),
            Some(1),
            vec![slot(0, true, Some(1), Some(1), Some(3))],
        );
        s2.server_start_unix = Some(2000); // new process
        let out = d.update(s2, at(base, 600));
        assert_eq!(out.prompt_tokens_per_second, None);
        assert_eq!(out.slots[0].phase, SlotPhase::ProcessingUnknown);
    }

    // --- Multiple slots ---

    #[test]
    fn multiple_slots_are_tracked_independently() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(
            ready(
                Some(2),
                Some(0),
                Some(100),
                Some(10),
                vec![
                    slot(0, true, Some(1), Some(5), Some(50)),
                    slot(1, true, Some(2), Some(5), Some(50)),
                ],
            ),
            at(base, 100),
        );
        // Only slot 0 decodes; slot 1 stays unknown.
        let out = d.update(
            ready(
                Some(2),
                Some(0),
                Some(100),
                Some(40),
                vec![
                    slot(0, true, Some(1), Some(35), Some(50)),
                    slot(1, true, Some(2), Some(5), Some(50)),
                ],
            ),
            at(base, 600),
        );
        assert_eq!(out.slots[0].phase, SlotPhase::Decode);
        assert_eq!(out.slots[1].phase, SlotPhase::ProcessingUnknown);
        assert_eq!(out.workload_phase, WorkloadPhase::Decode);
    }

    // --- Rate sanity ---

    #[test]
    fn rates_are_computed_from_counter_deltas() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(ready(Some(0), Some(0), Some(0), Some(0), vec![]), at(base, 100));
        let out = d.update(ready(Some(0), Some(0), Some(1000), Some(50), vec![]), at(base, 1100));
        let prompt = out.prompt_tokens_per_second.expect("prompt rate");
        let gen = out.generation_tokens_per_second.expect("gen rate");
        assert!((prompt - 1000.0).abs() < 50.0, "prompt ~1000 tok/s, got {prompt}");
        assert!((gen - 50.0).abs() < 5.0, "gen ~50 tok/s, got {gen}");
    }

    #[test]
    fn transient_timeout_does_not_disconnect_immediately() {
        // A single timeout is a transient failure: the connection is reported
        // as Reconnecting, not Disconnected.
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(ready(Some(0), Some(0), Some(0), Some(0), vec![]), at(base, 100));
        let mut snap = ready(None, None, None, None, vec![]);
        snap.connection = ConnectionState::Error;
        let out = d.update(snap, at(base, 600));
        assert_eq!(out.connection, ConnectionState::Reconnecting);
        assert_eq!(out.server, ServerState::Unavailable);
    }

    #[test]
    fn new_task_on_a_slot_resets_its_baseline() {
        let mut d = StateDetector::new();
        let base = Instant::now();
        d.update(
            ready(
                Some(1),
                Some(0),
                Some(100),
                Some(10),
                vec![slot(0, true, Some(1), Some(10), Some(100))],
            ),
            at(base, 100),
        );
        // Same slot, new task: counters restart; must not be Decode from the
        // old baseline (10 > 2 would be a false positive without task check).
        let out = d.update(
            ready(
                Some(1),
                Some(0),
                Some(100),
                Some(2),
                vec![slot(0, true, Some(2), Some(2), Some(100))],
            ),
            at(base, 600),
        );
        assert_eq!(out.slots[0].phase, SlotPhase::ProcessingUnknown);
    }
}
