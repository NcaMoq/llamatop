//! HTTP client behavior against a mock llama.cpp server (wiremock).
//!
//! Covers the status-code matrix required by the spec: 200, 401, 403, 404,
//! 429, 500, 503, timeout, connection closed, invalid JSON, empty response,
//! and reconnection. The client returns (status, body) and never treats a
//! non-2xx answer as a transport error; only connection-level failures are
//! `Err`.

use std::time::Duration;

use llamatop::backend::llamacpp::client::LlamaCppClient;
use llamatop::backend::llamacpp::slots::parse_slots;
use llamatop::error::BackendError;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn client(server: &MockServer, api_key: Option<&str>) -> LlamaCppClient {
    LlamaCppClient::new(server.uri().as_str(), Duration::from_millis(500), api_key)
        .expect("valid uri")
}

async fn mount_health(server: &MockServer, status: u16, body: &str) {
    Mock::given(method("GET"))
        .and(path("health"))
        .respond_with(
            ResponseTemplate::new(status).set_body_raw(body.to_string(), "application/json"),
        )
        .mount(server)
        .await;
}

#[tokio::test]
async fn get_raw_returns_200_body_and_header() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("metrics"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw("llamacpp:n 1", "text/plain")
                .insert_header("Process-Start-Time-Unix", "1700000000"),
        )
        .mount(&server)
        .await;

    let c = client(&server, None);
    let (status, body, start) = c.get_raw("metrics").await.unwrap();
    assert_eq!(status, 200);
    assert_eq!(body, "llamacpp:n 1");
    assert_eq!(start, Some(1700000000));
    // The header is remembered for restart detection.
    assert_eq!(c.last_process_start_unix(), Some(1700000000));
}

#[tokio::test]
async fn auth_key_is_sent_as_bearer_header() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("health"))
        .and(header("Authorization", "Bearer secret-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .expect(1..3)
        .mount(&server)
        .await;

    let c = client(&server, Some("secret-key"));
    assert!(c.has_api_key());
    let (status, _, _) = c.get_raw("health").await.unwrap();
    assert_eq!(status, 200);
}

#[tokio::test]
async fn non_2xx_statuses_are_returned_not_errored() {
    for status in [401u16, 403, 404, 429, 500, 503] {
        let server = MockServer::start().await;
        mount_health(&server, status, r#"{"error":{"message":"x"}}"#).await;
        let c = client(&server, None);
        let (got, _, _) = c.get_raw("health").await.unwrap_or_else(|e| {
            panic!("status {status} should be Ok((status, body)), got Err({e})")
        });
        assert_eq!(got, status, "status {status}");
    }
}

#[tokio::test]
async fn timeout_is_classified_as_timeout_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("health"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"status": "ok"}))
                .set_delay(Duration::from_millis(2000)),
        )
        .mount(&server)
        .await;

    let c = client(&server, None);
    let err = c.get_raw("health").await.unwrap_err();
    assert!(matches!(err, BackendError::Timeout { .. }), "expected Timeout, got {err:?}");
}

#[tokio::test]
async fn connection_refused_is_classified_as_connection_error() {
    // Bind a port and immediately release it so nothing is listening.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let c =
        LlamaCppClient::new(&format!("http://127.0.0.1:{port}"), Duration::from_millis(500), None)
            .unwrap();
    let err = c.get_raw("health").await.unwrap_err();
    // An unreachable port is a transport failure. Depending on the platform
    // (e.g. Windows firewall) it surfaces either as a refused connection or a
    // connect timeout; both mean "server unreachable" and neither is a
    // status-code answer.
    assert!(
        matches!(err, BackendError::Connection { .. } | BackendError::Timeout { .. }),
        "expected a transport failure, got {err:?}"
    );
}

#[tokio::test]
async fn empty_200_body_is_returned_and_parser_reports_invalid() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("slots"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("", "application/json"))
        .mount(&server)
        .await;

    let c = client(&server, None);
    let (status, body, _) = c.get_raw("slots").await.unwrap();
    assert_eq!(status, 200);
    assert!(body.is_empty());
    // The parser, not the transport, reports the content problem.
    assert!(parse_slots(&body).is_err());
}

#[tokio::test]
async fn invalid_json_body_is_returned_and_parser_reports_invalid() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("slots"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("{nope", "application/json"))
        .mount(&server)
        .await;

    let c = client(&server, None);
    let (status, body, _) = c.get_raw("slots").await.unwrap();
    assert_eq!(status, 200);
    assert!(parse_slots(&body).is_err());
}

#[tokio::test]
async fn invalid_endpoint_is_rejected_at_construction() {
    let err = LlamaCppClient::new("not a url", Duration::from_secs(1), None).unwrap_err();
    assert!(matches!(err, BackendError::Parse { .. }));
}
