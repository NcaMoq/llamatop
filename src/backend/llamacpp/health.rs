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

#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Classify a raw `/health` response as an endpoint observation.
///
/// 200 is the only "usable" answer (`Available`); everything else is a
/// temporary state that the next snapshot re-observes: 503 is the canonical
/// "model loading" signal, 401/403 are credential rejections, and other
/// 4xx/5xx (e.g. a proxy 404) are transient from the endpoint's point of
/// view.
pub fn classify_health(
    status: u16,
    body: &str,
) -> (crate::backend::EndpointAvailability, HealthOutcome) {
    let outcome = parse_health(status, body);
    let availability = match status {
        200 => crate::backend::EndpointAvailability::Available,
        401 | 403 => crate::backend::EndpointAvailability::AuthenticationFailed,
        _ => crate::backend::EndpointAvailability::TemporarilyUnavailable,
    };
    (availability, outcome)
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

    // --- classify_health ---

    use crate::backend::EndpointAvailability;

    #[test]
    fn health_200_is_available() {
        let (avail, out) = classify_health(200, r#"{"status":"ok"}"#);
        assert_eq!(avail, EndpointAvailability::Available);
        assert_eq!(out.server, ServerState::Ready);
    }

    #[test]
    fn health_503_is_temporarily_unavailable_while_loading() {
        let body = r#"{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}"#;
        let (avail, out) = classify_health(503, body);
        assert_eq!(avail, EndpointAvailability::TemporarilyUnavailable);
        assert_eq!(out.server, ServerState::Loading);
    }

    #[test]
    fn health_401_is_authentication_failed() {
        let (avail, out) = classify_health(401, r#"{"error":"unauthorized"}"#);
        assert_eq!(avail, EndpointAvailability::AuthenticationFailed);
        assert_eq!(out.server, ServerState::Unavailable);
    }

    #[test]
    fn health_403_is_authentication_failed() {
        let (avail, _) = classify_health(403, r#"{"error":"forbidden"}"#);
        assert_eq!(avail, EndpointAvailability::AuthenticationFailed);
    }

    #[test]
    fn health_other_4xx_and_5xx_are_temporarily_unavailable() {
        // A proxy 404 on /health is not evidence the endpoint is disabled;
        // it is retried on the next observation.
        let (avail404, _) = classify_health(404, r#"{"error":"not found"}"#);
        assert_eq!(avail404, EndpointAvailability::TemporarilyUnavailable);
        let (avail502, _) = classify_health(502, r#"{"error":"bad gateway"}"#);
        assert_eq!(avail502, EndpointAvailability::TemporarilyUnavailable);
    }
}
