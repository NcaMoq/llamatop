//! Normalization from raw llama.cpp responses to the domain snapshot.
//!
//! This is the only place where raw wire types and domain types meet.
//! - `None` means "the server did not report this value" and is never replaced
//!   with a guessed 0.
//! - Rates are not computed here (the detector owns deltas and timing).

use chrono::Utc;

use crate::domain::{BackendSnapshot, ConnectionState, ServerState, SlotPhase, SlotSnapshot};

use super::health::HealthOutcome;
use super::metrics::LlamaCppRawMetrics;
use super::props::RawProps;
use super::slots::{context_used, decoded_tokens, RawSlot};

#[derive(Default)]
pub struct RawObservation {
    pub health: Option<HealthOutcome>,
    pub slots: Option<Vec<RawSlot>>,
    pub metrics: Option<LlamaCppRawMetrics>,
    pub props: Option<RawProps>,
    /// Server process start time from the `Process-Start-Time-Unix` header,
    /// when the metrics endpoint provided it.
    pub server_start_unix: Option<u64>,
    /// True when the HTTP request layer could not reach the server at all.
    pub unreachable: bool,
    /// Short redacted reason when unreachable.
    pub error: Option<String>,
}

/// Build a domain snapshot from the raw observation.
pub fn normalize(obs: &RawObservation) -> BackendSnapshot {
    let mut snap = BackendSnapshot {
        timestamp: Utc::now(),
        connection: ConnectionState::Disconnected,
        server: ServerState::Unknown,
        ..Default::default()
    };

    if obs.unreachable {
        snap.connection = ConnectionState::Error;
        snap.server = ServerState::Unavailable;
        snap.error = obs.error.clone();
        return snap;
    }

    snap.connection = ConnectionState::Connected;

    if let Some(health) = &obs.health {
        snap.server = health.server;
        if health.server == ServerState::Unavailable {
            snap.error = health.detail.clone();
        }
    }

    // Slots
    if let Some(raw_slots) = &obs.slots {
        snap.slots = raw_slots.iter().map(normalize_slot).collect();
    }

    // Metrics
    if let Some(m) = &obs.metrics {
        snap.prompt_tokens_total = m.counter(m.prompt_tokens_total);
        snap.generation_tokens_total = m.counter(m.tokens_predicted_total);
        snap.active_requests = m.gauge_u64(m.requests_processing);
        snap.queued_requests = m.gauge_u64(m.requests_deferred);
        snap.context_max_tokens = m.counter(m.n_tokens_max);
        snap.speculative = m.into_speculative_stats();
        // Server-reported average throughput (cumulative since start).
        snap.prompt_tokens_per_second_reported = m.finite(m.prompt_tokens_seconds);
        snap.generation_tokens_per_second_reported = m.finite(m.predicted_tokens_seconds);
    }

    if let Some(start) = obs.server_start_unix {
        snap.server_start_unix = Some(start);
    }

    // Props (lowest priority for state; fills identity/sleep info)
    if let Some(props) = &obs.props {
        snap.model_name =
            props.model_alias.clone().or_else(|| model_name_from_path(props.model_path.as_deref()));
        snap.model_path = props.model_path.clone();
        snap.total_slots = props.total_slots.or(snap.total_slots);
        snap.build_info = props.build_info.clone();
        snap.is_sleeping = props.is_sleeping;
        if props.is_sleeping == Some(true) {
            // Sleeping is authoritative over a stale "ok" health.
            snap.server = ServerState::Sleeping;
        }
    }

    snap
}

/// Derive a display model name from a model path (file name without the
/// `.gguf` extension) when the server does not report an alias.
fn model_name_from_path(path: Option<&str>) -> Option<String> {
    let path = path?;
    let file = path.rsplit(['/', '\\']).next()?;
    let stem = file.strip_suffix(".gguf")?;
    if stem.is_empty() {
        None
    } else {
        Some(stem.to_string())
    }
}

fn normalize_slot(raw: &RawSlot) -> SlotSnapshot {
    let phase = if raw.is_processing { SlotPhase::ProcessingUnknown } else { SlotPhase::Idle };
    SlotSnapshot {
        id: raw.id,
        task_id: raw.id_task,
        is_processing: raw.is_processing,
        n_ctx: raw.n_ctx,
        n_tokens: context_used(raw),
        n_prompt_tokens: raw.n_prompt_tokens,
        n_prompt_tokens_processed: raw.n_prompt_tokens_processed,
        n_decoded: decoded_tokens(raw),
        speculative: raw.speculative,
        phase,
    }
}

/// Extract the `Process-Start-Time-Unix` header value (epoch seconds) from a
/// raw header string. Non-numeric values are ignored.
pub fn parse_start_unix(header_value: Option<&str>) -> Option<u64> {
    header_value?.trim().parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::llamacpp::health::parse_health;
    use crate::backend::llamacpp::metrics::parse_metrics;
    use crate::backend::llamacpp::props::parse_props;
    use crate::backend::llamacpp::slots::parse_slots;

    #[test]
    fn unreachable_produces_error_snapshot() {
        let obs = RawObservation {
            unreachable: true,
            error: Some("connection refused".into()),
            ..Default::default()
        };
        let snap = normalize(&obs);
        assert_eq!(snap.connection, ConnectionState::Error);
        assert_eq!(snap.server, ServerState::Unavailable);
        assert_eq!(snap.error.as_deref(), Some("connection refused"));
    }

    #[test]
    fn healthy_observation_maps_fields() {
        let obs = RawObservation {
            health: Some(HealthOutcome { server: ServerState::Ready, detail: None }),
            metrics: Some(LlamaCppRawMetrics {
                prompt_tokens_total: Some(100.0),
                tokens_predicted_total: Some(42.0),
                requests_processing: Some(2.0),
                requests_deferred: Some(1.0),
                n_tokens_max: Some(8192.0),
                ..Default::default()
            }),
            props: Some(RawProps {
                model_alias: Some("Qwen".into()),
                total_slots: Some(2),
                is_sleeping: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        };
        let snap = normalize(&obs);
        assert_eq!(snap.connection, ConnectionState::Connected);
        assert_eq!(snap.server, ServerState::Ready);
        assert_eq!(snap.prompt_tokens_total, Some(100));
        assert_eq!(snap.generation_tokens_total, Some(42));
        assert_eq!(snap.active_requests, Some(2));
        assert_eq!(snap.queued_requests, Some(1));
        assert_eq!(snap.context_max_tokens, Some(8192));
        assert_eq!(snap.model_name.as_deref(), Some("Qwen"));
        assert_eq!(snap.is_sleeping, Some(false));
    }

    #[test]
    fn sleeping_overrides_ready_health() {
        let obs = RawObservation {
            health: Some(HealthOutcome { server: ServerState::Ready, detail: None }),
            props: Some(RawProps { is_sleeping: Some(true), ..Default::default() }),
            ..Default::default()
        };
        let snap = normalize(&obs);
        assert_eq!(snap.server, ServerState::Sleeping);
    }

    #[test]
    fn slots_normalize_idle_and_processing() {
        let nt = |n_decoded: Option<u64>| {
            crate::backend::llamacpp::slots::OneOrMany::One(
                crate::backend::llamacpp::slots::RawNextToken { n_decoded, ..Default::default() },
            )
        };
        let raw = vec![
            RawSlot {
                id: 0,
                is_processing: true,
                n_ctx: Some(4096),
                n_prompt_tokens: Some(10),
                next_token: Some(nt(Some(3))),
                ..Default::default()
            },
            RawSlot { id: 1, is_processing: false, n_ctx: None, ..Default::default() },
        ];
        let obs = RawObservation { slots: Some(raw), ..Default::default() };
        let snap = normalize(&obs);
        assert_eq!(snap.slots.len(), 2);
        assert_eq!(snap.slots[0].phase, SlotPhase::ProcessingUnknown);
        // Context occupancy is the prompt buffer; the decoded counter is
        // tracked separately (phase detection) and not added on top.
        assert_eq!(snap.slots[0].n_tokens, Some(10));
        assert_eq!(snap.slots[0].n_decoded, Some(3));
        assert_eq!(snap.slots[1].phase, SlotPhase::Idle);
        assert_eq!(snap.slots[1].n_tokens, None);
    }

    #[test]
    fn live_schema_maps_prompt_buffer_to_context_usage() {
        // The current llama.cpp shape: no n_tokens, prompt buffer 58029 in a
        // 237568 window, idle (no task counters beyond the prompt).
        let raw = vec![RawSlot {
            id: 0,
            n_ctx: Some(237568),
            speculative: true,
            is_processing: false,
            id_task: Some(1000),
            n_prompt_tokens: Some(58029),
            n_prompt_tokens_processed: Some(0),
            n_prompt_tokens_cache: Some(0),
            next_token: Some(crate::backend::llamacpp::slots::OneOrMany::Many(vec![
                crate::backend::llamacpp::slots::RawNextToken {
                    n_decoded: Some(0),
                    ..Default::default()
                },
            ])),
            ..Default::default()
        }];
        let obs = RawObservation { slots: Some(raw), ..Default::default() };
        let snap = normalize(&obs);
        assert_eq!(snap.slots[0].phase, SlotPhase::Idle);
        assert_eq!(snap.slots[0].n_ctx, Some(237568));
        assert_eq!(snap.slots[0].n_tokens, Some(58029));
        assert_eq!(snap.slots[0].n_decoded, Some(0));
    }

    #[test]
    fn prompt_sub_counts_are_not_double_counted() {
        // processed and cache describe how the prompt buffer got there; they
        // must not be summed into the occupancy.
        let raw = vec![RawSlot {
            id: 0,
            is_processing: true,
            n_ctx: Some(65536),
            n_prompt_tokens: Some(1000),
            n_prompt_tokens_processed: Some(350),
            n_prompt_tokens_cache: Some(200),
            ..Default::default()
        }];
        let obs = RawObservation { slots: Some(raw), ..Default::default() };
        let snap = normalize(&obs);
        assert_eq!(snap.slots[0].n_tokens, Some(1000));
    }

    #[test]
    fn old_schema_n_tokens_precedes_prompt_buffer() {
        let raw = vec![RawSlot {
            id: 0,
            is_processing: true,
            n_ctx: Some(4096),
            n_tokens: Some(16384),
            n_prompt_tokens: Some(100),
            ..Default::default()
        }];
        let obs = RawObservation { slots: Some(raw), ..Default::default() };
        let snap = normalize(&obs);
        assert_eq!(snap.slots[0].n_tokens, Some(16384));
        // An occupancy above the window is preserved as reported: clamping
        // is a display decision, not a mapping one.
        assert_eq!(snap.slots[0].n_ctx, Some(4096));
    }

    #[test]
    fn parse_start_unix_rejects_garbage() {
        assert_eq!(parse_start_unix(Some("1700000000")), Some(1700000000));
        assert_eq!(parse_start_unix(Some("not-a-number")), None);
        assert_eq!(parse_start_unix(None), None);
    }

    #[test]
    fn fixture_full_pipeline_maps_all_endpoints() {
        let health_body = include_str!("../../../fixtures/health_ready.json");
        let slots_body = include_str!("../../../fixtures/slots_processing.json");
        let metrics_body = include_str!("../../../fixtures/metrics.txt");
        let props_body = include_str!("../../../fixtures/props.json");

        let obs = RawObservation {
            health: Some(parse_health(200, health_body)),
            slots: Some(parse_slots(slots_body).unwrap()),
            metrics: Some(parse_metrics(metrics_body)),
            props: Some(parse_props(props_body).unwrap()),
            ..Default::default()
        };
        let snap = normalize(&obs);
        assert_eq!(snap.connection, ConnectionState::Connected);
        assert_eq!(snap.server, ServerState::Ready);
        assert_eq!(snap.prompt_tokens_total, Some(12345));
        assert_eq!(snap.generation_tokens_total, Some(678));
        assert_eq!(snap.active_requests, Some(2));
        assert_eq!(snap.queued_requests, Some(1));
        assert_eq!(snap.context_max_tokens, Some(24576));
        assert_eq!(snap.model_name.as_deref(), Some("qwen3.8-27b"));
        assert_eq!(snap.total_slots, Some(2));
        // Server-reported average throughput is carried through.
        assert!(snap.prompt_tokens_per_second_reported.is_some());
        assert!(snap.generation_tokens_per_second_reported.is_some());
        // The processing slot starts as ProcessingUnknown (no delta yet).
        assert_eq!(snap.slots[0].phase, SlotPhase::ProcessingUnknown);
        assert!(snap.error.is_none());
    }

    #[test]
    fn fixture_loading_pipeline_marks_loading() {
        let health_body = include_str!("../../../fixtures/health_loading.json");
        let obs =
            RawObservation { health: Some(parse_health(503, health_body)), ..Default::default() };
        let snap = normalize(&obs);
        assert_eq!(snap.server, ServerState::Loading);
    }

    #[test]
    fn model_name_falls_back_to_path_stem_when_no_alias() {
        let obs = RawObservation {
            props: Some(RawProps {
                model_alias: None,
                model_path: Some("models/Meta-Llama-3.1-8B-Q4_K_M.gguf".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let snap = normalize(&obs);
        assert_eq!(snap.model_name.as_deref(), Some("Meta-Llama-3.1-8B-Q4_K_M"));
    }

    #[test]
    fn model_name_none_when_path_has_no_gguf_ext() {
        let obs = RawObservation {
            props: Some(RawProps {
                model_alias: None,
                model_path: Some("models/unknown.bin".into()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let snap = normalize(&obs);
        assert_eq!(snap.model_name, None);
    }

    #[test]
    fn missing_metrics_fields_stay_none_not_zero() {
        let obs = RawObservation {
            health: Some(HealthOutcome { server: ServerState::Ready, detail: None }),
            metrics: Some(LlamaCppRawMetrics::default()),
            ..Default::default()
        };
        let snap = normalize(&obs);
        assert_eq!(snap.prompt_tokens_total, None);
        assert_eq!(snap.active_requests, None);
        assert_eq!(snap.queued_requests, None);
    }
}
