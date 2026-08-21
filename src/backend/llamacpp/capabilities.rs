//! Capability observation: determine the availability of each endpoint.
//!
//! Every probed endpoint produces an [`EndpointAvailability`] *observation*,
//! never a `bool`. The mapping is aligned with what llama.cpp actually
//! returns:
//!
//! | Observation | Meaning |
//! |---|---|
//! | 2xx + expected payload | `Available` |
//! | 2xx + unexpected payload | `ParseFailed` |
//! | 404 / 405 | `Unsupported` (the endpoint does not exist on this server) |
//! | 501 with a "not supported" body | `Unsupported` (llama.cpp reports a
//!   disabled endpoint this way, e.g. `--no-metrics`) |
//! | 501 without that body | `TemporarilyUnavailable` (a 501 from a proxy or
//!   gateway is not evidence the endpoint is disabled) |
//! | 401 / 403 | `AuthenticationFailed` |
//! | any other 5xx (incl. 503) | `TemporarilyUnavailable` |
//! | transport failure | `TemporarilyUnavailable` |
//! | never probed | `Unknown` (the probe default) |
//!
//! `Unknown` is the right state for a failed *initial* probe: we must not
//! conclude "unsupported" from "the server was unreachable".
//!
//! These observations are re-derived on every snapshot, so temporary states
//! recover automatically without a manual reconnect.

use super::client::LlamaCppClient;
use super::{props, slots};
use crate::backend::{BackendCapabilities, EndpointAvailability};
use crate::error::BackendError;

/// The success status for every llama.cpp monitoring endpoint.
pub const EXPECTED_SUCCESS: u16 = 200;

/// Map one observed response to an endpoint availability state.
///
/// When the status is [`EXPECTED_SUCCESS`], the body must have been accepted
/// by the endpoint's parser (`body_is_valid`); a 2xx body that fails
/// validation is `ParseFailed`, never `Available`.
pub fn classify_response(status: u16, body: &str, body_is_valid: bool) -> EndpointAvailability {
    if status == EXPECTED_SUCCESS {
        return if body_is_valid {
            EndpointAvailability::Available
        } else {
            EndpointAvailability::ParseFailed
        };
    }
    match status {
        401 | 403 => EndpointAvailability::AuthenticationFailed,
        404 | 405 => EndpointAvailability::Unsupported,
        // llama.cpp signals "endpoint disabled by configuration" with 501 and
        // a `not_supported` error type. A bare 501 (e.g. from a proxy) is not
        // evidence of that, so it stays temporarily unavailable.
        501 => {
            if body.to_ascii_lowercase().contains("not_supported") {
                EndpointAvailability::Unsupported
            } else {
                EndpointAvailability::TemporarilyUnavailable
            }
        }
        // All other statuses — including 503 (server busy / model loading)
        // and 5xx in general — are transient.
        _ => EndpointAvailability::TemporarilyUnavailable,
    }
}

/// Probe one endpoint by issuing a GET and classifying the response.
///
/// Transport failures (connection refused, timeout, DNS) are *classified* as
/// `TemporarilyUnavailable`, not propagated: an endpoint that cannot be
/// reached now is a temporary state of that endpoint, not a probe failure.
pub async fn probe_endpoint(
    client: &LlamaCppClient,
    path: &str,
    body_is_valid: impl FnOnce(&str) -> bool,
) -> EndpointAvailability {
    match client.get_raw(path).await {
        Ok((status, body, _)) => classify_response(status, &body, body_is_valid(&body)),
        // An oversized 2xx body answered the request but was not the expected
        // payload: a parse failure, not a transport failure.
        Err(BackendError::BodyTooLarge { .. }) => EndpointAvailability::ParseFailed,
        Err(err) => {
            tracing::warn!(path, "endpoint probe failed: {err}");
            EndpointAvailability::TemporarilyUnavailable
        }
    }
}

/// Probe all monitoring endpoints and assemble `BackendCapabilities`.
///
/// The probe never fails on an individual endpoint: each endpoint's outcome
/// is an observation in the returned `BackendCapabilities`. It returns `Err`
/// only when the client itself is misconfigured (a path cannot be joined to
/// the base URL), in which case no observation is trustworthy and the caller
/// keeps the default (`Unknown`) capabilities.
pub async fn probe_capabilities(
    client: &LlamaCppClient,
) -> Result<BackendCapabilities, BackendError> {
    // /health: 200 is ready; 503 is the canonical "model loading" state and
    // is classified transiently (re-observed on the next cycle).
    let health = probe_endpoint(client, "health", |_| true).await;

    // /slots: 200 with a parseable JSON payload. An empty array is valid:
    // the endpoint works and simply reports zero slots.
    let slots = probe_endpoint(client, "slots", |body| slots::parse_slots(body).is_ok()).await;

    // /metrics: 200 with any parseable text payload; the parser is total
    // (it never fails), so a 2xx body is always accepted here.
    let metrics = probe_endpoint(client, "metrics", |_| true).await;

    // /props: 200 with a parseable JSON object.
    let props = probe_endpoint(client, "props", |body| props::parse_props(body).is_ok()).await;

    Ok(BackendCapabilities {
        health,
        slots,
        metrics,
        props,
        // Speculative metrics live on the same endpoint; their presence is
        // determined per-snapshot from the actual metric lines.
        speculative_metrics: metrics.is_available(),
        model_info: props.is_available(),
        // llama.cpp has no explicit prefill/decode state field; decode is
        // detected from per-slot decoded-token growth (exact), prefill is
        // estimated.
        exact_prefill_state: false,
        exact_decode_state: slots.is_available(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_200_valid_is_available() {
        assert_eq!(classify_response(200, r#"{"ok":true}"#, true), EndpointAvailability::Available);
    }

    #[test]
    fn classify_200_invalid_body_is_parse_failed() {
        assert_eq!(
            classify_response(200, "not json at all", false),
            EndpointAvailability::ParseFailed
        );
    }

    #[test]
    fn classify_404_and_405_are_unsupported() {
        assert_eq!(classify_response(404, "", true), EndpointAvailability::Unsupported);
        assert_eq!(classify_response(405, "", true), EndpointAvailability::Unsupported);
    }

    #[test]
    fn classify_501_with_not_supported_body_is_unsupported() {
        let body = r#"{"error":{"code":501,"message":"Metrics endpoint not supported","type":"not_supported_error"}}"#;
        assert_eq!(classify_response(501, body, true), EndpointAvailability::Unsupported);
    }

    #[test]
    fn classify_bare_501_is_temporarily_unavailable() {
        // A 501 that does not carry llama.cpp's "not supported" signal is
        // not evidence the endpoint is disabled (e.g. a proxy 501).
        assert_eq!(
            classify_response(501, r#"{"error":"not implemented"}"#, true),
            EndpointAvailability::TemporarilyUnavailable
        );
    }

    #[test]
    fn classify_401_and_403_are_authentication_failed() {
        assert_eq!(classify_response(401, "", true), EndpointAvailability::AuthenticationFailed);
        assert_eq!(classify_response(403, "", true), EndpointAvailability::AuthenticationFailed);
    }

    #[test]
    fn classify_503_and_5xx_are_temporarily_unavailable() {
        assert_eq!(classify_response(503, "", true), EndpointAvailability::TemporarilyUnavailable);
        assert_eq!(classify_response(500, "", true), EndpointAvailability::TemporarilyUnavailable);
    }

    #[test]
    fn classify_503_is_never_available() {
        // A 503 is transient regardless of the body: the body cannot be
        // trusted while the server is busy or loading a model.
        assert_ne!(
            classify_response(503, r#"{"status":"loading"}"#, true),
            EndpointAvailability::Available
        );
    }

    #[test]
    fn default_capabilities_are_all_unknown() {
        // A probe that could not run at all leaves every endpoint Unknown —
        // never Unsupported: "unreachable" is not "does not exist".
        let caps = BackendCapabilities::default();
        assert_eq!(caps.health, EndpointAvailability::Unknown);
        assert_eq!(caps.slots, EndpointAvailability::Unknown);
        assert_eq!(caps.metrics, EndpointAvailability::Unknown);
        assert_eq!(caps.props, EndpointAvailability::Unknown);
    }

    // --- Probe-level tests against a real (mock) server ---

    async fn mount(server: &wiremock::MockServer, path: &str, status: u16, body: &str) {
        use wiremock::matchers::{method, path as mpath};
        use wiremock::{Mock, ResponseTemplate};
        Mock::given(method("GET"))
            .and(mpath(path))
            .respond_with(
                ResponseTemplate::new(status).set_body_raw(body.to_string(), "application/json"),
            )
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn probe_reports_per_endpoint_states() {
        use std::time::Duration;
        let server = wiremock::MockServer::start().await;
        // Health ready; slots supported (empty); metrics disabled (501 with
        // the llama.cpp signal); props requires auth.
        mount(&server, "health", 200, r#"{"status":"ok"}"#).await;
        mount(&server, "slots", 200, "[]").await;
        mount(
            &server,
            "metrics",
            501,
            r#"{"error":{"code":501,"message":"Metrics disabled","type":"not_supported_error"}}"#,
        )
        .await;
        mount(&server, "props", 401, r#"{"error":"unauthorized"}"#).await;

        let client =
            LlamaCppClient::new(server.uri().as_str(), Duration::from_secs(2), None).unwrap();
        let caps = probe_capabilities(&client).await.expect("probe succeeds");
        assert_eq!(caps.health, EndpointAvailability::Available);
        // An empty slot list is a successful observation with zero slots.
        assert_eq!(caps.slots, EndpointAvailability::Available);
        assert!(caps.exact_decode_state, "decode detection works when /slots works");
        assert_eq!(caps.metrics, EndpointAvailability::Unsupported);
        assert!(!caps.speculative_metrics, "metrics are disabled");
        assert_eq!(caps.props, EndpointAvailability::AuthenticationFailed);
        assert!(!caps.model_info);
    }

    #[tokio::test]
    async fn probe_transport_failure_is_temporary_not_unsupported() {
        use std::time::Duration;
        // A dead port: every endpoint is TemporarilyUnavailable, and none is
        // concluded to be Unsupported.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let client = LlamaCppClient::new(
            &format!("http://127.0.0.1:{port}"),
            Duration::from_millis(200),
            None,
        )
        .expect("valid");
        let caps = probe_capabilities(&client).await.expect("probe classifies, does not fail");
        assert_eq!(caps.health, EndpointAvailability::TemporarilyUnavailable);
        assert_eq!(caps.slots, EndpointAvailability::TemporarilyUnavailable);
        assert_eq!(caps.metrics, EndpointAvailability::TemporarilyUnavailable);
        assert_eq!(caps.props, EndpointAvailability::TemporarilyUnavailable);
    }

    #[tokio::test]
    async fn probe_invalid_json_is_parse_failed() {
        use std::time::Duration;
        let server = wiremock::MockServer::start().await;
        mount(&server, "health", 200, r#"{"status":"ok"}"#).await;
        // 200 but a body /slots cannot parse.
        mount(&server, "slots", 200, "this is not json").await;
        mount(&server, "metrics", 200, "# TYPE x counter").await;
        mount(&server, "props", 200, "[]").await; // not an object

        let client =
            LlamaCppClient::new(server.uri().as_str(), Duration::from_secs(2), None).unwrap();
        let caps = probe_capabilities(&client).await.expect("probe succeeds");
        assert_eq!(caps.health, EndpointAvailability::Available);
        // 200 with a body the parser rejects: ParseFailed, never Available.
        assert_eq!(caps.slots, EndpointAvailability::ParseFailed);
        // The /metrics parser is total (Prometheus text never "fails"), so a
        // 200 body is always accepted at the probe level.
        assert_eq!(caps.metrics, EndpointAvailability::Available);
        assert_eq!(caps.props, EndpointAvailability::ParseFailed);
    }
}
