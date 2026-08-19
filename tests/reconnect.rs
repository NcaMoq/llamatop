//! Reconnection behavior: the detector must not fabricate rates or phases
//! across a disconnect, and recovery must reset learned state.

use std::time::{Duration, Instant};

use llamatop::backend::llamacpp::LlamaCppBackend;
use llamatop::backend::{BackendCapabilities, InferenceBackend};
use llamatop::detector::StateDetector;
use llamatop::domain::{ConnectionState, ServerState, WorkloadPhase};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Build a backend pointed at a dead port (nothing listening).
fn dead_backend() -> LlamaCppBackend {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    LlamaCppBackend::new(&format!("http://127.0.0.1:{port}"), Duration::from_millis(300), None)
        .expect("valid url")
}

async fn ready_server() -> MockServer {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("slots"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("props"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"total_slots": 1})),
        )
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn unreachable_backend_yields_error_snapshot() {
    let backend = dead_backend();
    let caps = BackendCapabilities::default();
    let snap = backend.snapshot(&caps).await.unwrap();
    assert_eq!(snap.connection, ConnectionState::Error);
    assert_eq!(snap.server, ServerState::Unavailable);
    assert!(snap.error.is_some(), "transport error should be recorded");
}

#[tokio::test]
async fn single_failure_is_reconnecting_not_disconnected() {
    let mut detector = StateDetector::new();
    let backend = dead_backend();
    let caps = BackendCapabilities::default();
    let snap = backend.snapshot(&caps).await.unwrap();
    let out = detector.update(snap, Instant::now());
    assert_eq!(out.connection, ConnectionState::Reconnecting);
    assert_eq!(out.workload_phase, WorkloadPhase::Idle);
}

#[tokio::test]
async fn recovery_resets_counters_and_reports_connected() {
    let mut detector = StateDetector::new();

    // 1. A healthy observation with counters already advanced.
    let server = ready_server().await;
    let backend =
        LlamaCppBackend::new(server.uri().as_str(), Duration::from_secs(2), None).unwrap();
    let caps = BackendCapabilities { health: true, slots: true, props: true, ..Default::default() };
    let first = backend.snapshot(&caps).await.unwrap();
    let out1 = detector.update(first, Instant::now());
    assert_eq!(out1.connection, ConnectionState::Connected);
    assert_eq!(out1.server, ServerState::Ready);

    // 2. The server goes away.
    let dead = dead_backend();
    let dead_snap = dead.snapshot(&caps).await.unwrap();
    let out2 = detector.update(dead_snap, Instant::now());
    assert_eq!(out2.connection, ConnectionState::Reconnecting);

    // 3. The server comes back. Counters were discarded on reconnect, so the
    //    first interval after recovery must not produce a rate. The phase is
    //    not guessed either: with a reset detector and one observation it
    //    stays ProcessingUnknown (Idle needs two consecutive observations).
    let third = backend.snapshot(&caps).await.unwrap();
    let out3 = detector.update(third, Instant::now());
    assert_eq!(out3.connection, ConnectionState::Connected);
    assert_eq!(out3.prompt_tokens_per_second, None);
    assert_eq!(out3.generation_tokens_per_second, None);
    assert_eq!(out3.workload_phase, WorkloadPhase::ProcessingUnknown);

    // 4. A second consecutive idle observation stabilizes the phase to Idle.
    let fourth = backend.snapshot(&caps).await.unwrap();
    let out4 = detector.update(fourth, Instant::now());
    assert_eq!(out4.workload_phase, WorkloadPhase::Idle);
}

#[tokio::test]
async fn server_restart_discards_baseline() {
    // Simulate a restart via the process-start-time header: the detector must
    // reset when it changes, even if counters look continuous.
    let mut detector = StateDetector::new();

    let make = |start: u64| llamatop::domain::BackendSnapshot {
        connection: ConnectionState::Connected,
        server: ServerState::Ready,
        prompt_tokens_total: Some(1000),
        generation_tokens_total: Some(500),
        server_start_unix: Some(start),
        ..Default::default()
    };

    let base = Instant::now();
    detector.update(make(1000), base);
    let out = detector.update(make(2000), base + Duration::from_millis(500));
    // After a restart the previous baseline is discarded: no rate this interval.
    assert_eq!(out.prompt_tokens_per_second, None);
    assert_eq!(out.generation_tokens_per_second, None);
}
