//! Rendering of a captured snapshot for the one-shot `snapshot` command.
//!
//! Two renderers:
//! - `pretty`: human-readable, labels + values, no color required for state.
//! - `json`: a pure JSON document on stdout (schema versioned). No logo, no
//!   rules, no progress, no color. Warnings/errors go to stderr by the caller.
//!
//! This module does no HTTP and no terminal TUI; it only formats a snapshot.

mod json;
mod pretty;

use crate::snapshot::Snapshot;

/// Render a snapshot to stdout.
///
/// When `json` is true, stdout receives only the JSON document. Otherwise a
/// human-readable table is written. `ascii` selects the symbol set for the
/// pretty output (JSON output is always plain ASCII/UTF-8 text).
pub fn render(snap: &Snapshot, json: bool, ascii: bool) -> anyhow::Result<()> {
    if json {
        let doc = json::to_json(snap)?;
        let rendered = serde_json::to_string_pretty(&doc)?;
        println!("{rendered}");
    } else {
        pretty::print(snap, ascii)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendCapabilities;
    use crate::domain::{ConnectionState, ServerState, WorkloadPhase};

    fn sample() -> Snapshot {
        use crate::domain::BackendSnapshot;
        let snap = BackendSnapshot {
            connection: ConnectionState::Connected,
            server: ServerState::Ready,
            workload_phase: WorkloadPhase::Decode,
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
            snapshot: snap,
            capabilities: BackendCapabilities::default(),
        }
    }

    #[test]
    fn json_render_is_valid_json() {
        let snap = sample();
        let doc = json::to_json(&snap).expect("json doc");
        let raw = serde_json::to_string(&doc).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&raw).expect("parse back");
        assert_eq!(parsed["schema_version"], 1);
        assert_eq!(parsed["connection"]["state"], "connected");
        assert_eq!(parsed["server"]["workload_phase"], "decode");
    }
}
