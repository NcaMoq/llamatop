//! Parsing of the `/health` endpoint.
//!
//! Observed current behavior (llama.cpp master):
//! - 200 with `{"status":"ok"}` when ready (also while sleeping).
//! - 503 with `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}`
//!   while the model is loading.
//!
//! We classify by HTTP status first, then fall back to inspecting the body so
//! that wording changes do not break detection.

use serde_json::Value;

use crate::domain::ServerState;

pub struct HealthOutcome {
    pub server: ServerState,
    /// Short reason when not ready.
    pub detail: Option<String>,
}

/// Interpret a raw `/health` (status, body) pair.
pub fn parse_health(status: u16, body: &str) -> HealthOutcome {
    let value: Value = serde_json::from_str(body).unwrap_or(Value::Null);

    // Ready: 200 and (optionally) an "ok"-ish status field.
    if status == 200 {
        let status_field = value.get("status").and_then(|s| s.as_str()).unwrap_or("");
        if status_field.is_empty() || status_field.eq_ignore_ascii_case("ok") {
            return HealthOutcome { server: ServerState::Ready, detail: None };
        }
        // 200 but an unexpected status token: treat as ready but keep detail.
        return HealthOutcome {
            server: ServerState::Ready,
            detail: Some(status_field.to_string()),
        };
    }

    // Unavailable/loading: 503 (or any 5xx) with a message.
    let message = value
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .map(str::to_string);

    if (500..600).contains(&status) {
        let detail = message.clone().unwrap_or_else(|| format!("HTTP {status}"));
        // A 503 while loading is the canonical "loading" signal. We keep the
        // message for display but do NOT string-match on "Loading model" to
        // decide the state; the status is authoritative.
        return HealthOutcome { server: ServerState::Loading, detail: Some(detail) };
    }

    // 4xx or other: server is reachable but not in a state we can use.
    HealthOutcome {
        server: ServerState::Unavailable,
        detail: message.or_else(|| Some(format!("HTTP {status}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_200_is_ready() {
        let out = parse_health(200, r#"{"status":"ok"}"#);
        assert_eq!(out.server, ServerState::Ready);
        assert!(out.detail.is_none());
    }

    #[test]
    fn ok_200_without_status_field_is_ready() {
        let out = parse_health(200, "{}");
        assert_eq!(out.server, ServerState::Ready);
    }

    #[test]
    fn loading_503_is_loading() {
        let body = r#"{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}"#;
        let out = parse_health(503, body);
        assert_eq!(out.server, ServerState::Loading);
        assert_eq!(out.detail.as_deref(), Some("Loading model"));
    }

    #[test]
    fn other_5xx_is_loading() {
        let out = parse_health(500, r#"{"error":{"message":"boom"}}"#);
        assert_eq!(out.server, ServerState::Loading);
        assert_eq!(out.detail.as_deref(), Some("boom"));
    }

    #[test]
    fn client_error_is_unavailable() {
        let out = parse_health(404, r#"{"error":{"message":"nope"}}"#);
        assert_eq!(out.server, ServerState::Unavailable);
        assert_eq!(out.detail.as_deref(), Some("nope"));
    }

    #[test]
    fn fixture_health_ready() {
        let body = include_str!("../../../fixtures/health_ready.json");
        let out = parse_health(200, body);
        assert_eq!(out.server, ServerState::Ready);
        assert!(out.detail.is_none());
    }

    #[test]
    fn fixture_health_loading() {
        let body = include_str!("../../../fixtures/health_loading.json");
        let out = parse_health(503, body);
        assert_eq!(out.server, ServerState::Loading);
        assert_eq!(out.detail.as_deref(), Some("Loading model"));
    }

    #[test]
    fn non_json_body_does_not_panic() {
        let out = parse_health(503, "gateway said no");
        assert_eq!(out.server, ServerState::Loading);
        assert_eq!(out.detail.as_deref(), Some("HTTP 503"));
    }
}
