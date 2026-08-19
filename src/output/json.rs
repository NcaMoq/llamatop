//! Pure JSON output for `snapshot --json`.
//!
//! Field names are part of the public schema and must not change casually;
//! add new fields, do not rename or repurpose existing ones. `None` values
//! are omitted (skipped) so that a missing metric is distinguishable from 0.

use serde::Serialize;

use crate::domain::{GpuSnapshot, SlotPhase, SlotSnapshot};
use crate::snapshot::Snapshot;

/// The public JSON schema version.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Serialize)]
pub struct JsonSnapshot {
    pub schema_version: u32,
    /// RFC 3339 UTC timestamp of the observation.
    pub timestamp: String,
    pub backend: String,
    pub endpoint: String,
    pub connection: ConnectionOut,
    pub server: ServerOut,
    pub throughput: ThroughputOut,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gpu: Vec<GpuOut>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub slots: Vec<SlotOut>,
}

#[derive(Debug, Serialize)]
pub struct ConnectionOut {
    pub state: String,
}

#[derive(Debug, Serialize)]
pub struct ServerOut {
    pub state: String,
    pub workload_phase: String,
    pub workload_confidence: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_requests: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queued_requests: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build_info: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ThroughputOut {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_tokens_per_second: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct GpuOut {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utilization_percent: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_used_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_total_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature_celsius: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_watts: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct SlotOut {
    pub id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_id: Option<u64>,
    pub phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_ctx: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_decoded: Option<u64>,
}

/// Build the JSON document from a snapshot.
pub fn to_json(snap: &Snapshot) -> anyhow::Result<JsonSnapshot> {
    let s = &snap.snapshot;
    let prompt_rate = s.prompt_tokens_per_second.or(s.prompt_tokens_per_second_reported);
    let gen_rate = s.generation_tokens_per_second.or(s.generation_tokens_per_second_reported);

    let gpu: Vec<GpuOut> = s
        .gpu
        .iter()
        .map(|g: &GpuSnapshot| GpuOut {
            index: g.index,
            name: g.name.clone(),
            utilization_percent: g.utilization_percent,
            memory_used_bytes: g.memory_used_bytes,
            memory_total_bytes: g.memory_total_bytes,
            temperature_celsius: g.temperature_celsius,
            power_watts: g.power_watts,
        })
        .collect();

    let slots: Vec<SlotOut> = s
        .slots
        .iter()
        .map(|sl: &SlotSnapshot| SlotOut {
            id: sl.id,
            task_id: sl.task_id,
            phase: slot_phase_json(sl.phase),
            n_tokens: sl.n_tokens,
            n_ctx: sl.n_ctx,
            n_decoded: sl.n_decoded,
        })
        .collect();

    Ok(JsonSnapshot {
        schema_version: SCHEMA_VERSION,
        timestamp: s.timestamp.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        backend: snap.backend.clone(),
        endpoint: snap.endpoint.clone(),
        connection: ConnectionOut { state: s.connection.as_str().to_lowercase() },
        server: ServerOut {
            state: s.server.as_str().to_lowercase(),
            workload_phase: s.workload_phase.as_str().to_string(),
            workload_confidence: s.workload_confidence.as_str().to_string(),
            active_requests: s.active_requests,
            queued_requests: s.queued_requests,
            model_name: s.model_name.clone(),
            context_max_tokens: s.context_max_tokens,
            build_info: s.build_info.clone(),
        },
        throughput: ThroughputOut {
            prompt_tokens_per_second: prompt_rate,
            generation_tokens_per_second: gen_rate,
        },
        gpu,
        slots,
    })
}

fn slot_phase_json(phase: SlotPhase) -> String {
    phase.as_str().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendCapabilities;
    use crate::domain::{BackendSnapshot, Confidence, ConnectionState, ServerState, WorkloadPhase};

    fn snap() -> Snapshot {
        let s = BackendSnapshot {
            connection: ConnectionState::Connected,
            server: ServerState::Ready,
            workload_phase: WorkloadPhase::Decode,
            workload_confidence: Confidence::Exact,
            active_requests: Some(1),
            queued_requests: Some(0),
            model_name: Some("Qwen".into()),
            prompt_tokens_per_second: Some(1832.4),
            generation_tokens_per_second: Some(48.6),
            ..Default::default()
        };
        Snapshot {
            backend: "llama.cpp".into(),
            endpoint: "http://127.0.0.1:8080/".into(),
            snapshot: s,
            capabilities: BackendCapabilities::default(),
        }
    }

    #[test]
    fn schema_version_is_one() {
        let doc = to_json(&snap()).unwrap();
        assert_eq!(doc.schema_version, 1);
    }

    #[test]
    fn none_fields_are_omitted_not_zero() {
        let doc = to_json(&snap()).unwrap();
        let raw = serde_json::to_string(&doc).unwrap();
        // context_max_tokens is None -> omitted entirely.
        assert!(!raw.contains("context_max_tokens"));
        // active_requests is Some(1) -> present.
        assert!(raw.contains("\"active_requests\":1"));
    }

    #[test]
    fn gpu_absent_when_no_data() {
        let doc = to_json(&snap()).unwrap();
        assert!(doc.gpu.is_empty());
        let raw = serde_json::to_string(&doc).unwrap();
        assert!(!raw.contains("\"gpu\""));
    }

    #[test]
    fn rate_falls_back_to_server_reported_value() {
        let mut s = snap();
        s.snapshot.prompt_tokens_per_second = None;
        s.snapshot.prompt_tokens_per_second_reported = Some(900.0);
        let doc = to_json(&s).unwrap();
        assert_eq!(doc.throughput.prompt_tokens_per_second, Some(900.0));
    }
}
