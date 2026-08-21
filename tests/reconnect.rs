//! Reconnection behavior: the detector must not fabricate rates or phases
//! across a disconnect, and recovery must reset learned state.

use std::time::{Duration, Instant};

use llamatop::backend::llamacpp::LlamaCppBackend;
use llamatop::backend::{BackendCapabilities, EndpointAvailability, EndpointDue, InferenceBackend};
use llamatop::detector::StateDetector;
use llamatop::domain::{ConnectionState, ServerState, SlotPhase, SlotSnapshot, WorkloadPhase};
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
    Mock::given(method("GET"))
        .and(path("metrics"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(
                "llamacpp:prompt_tokens_total 5\nllamacpp:tokens_predicted_total 3\n",
            ),
        )
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn unreachable_backend_yields_error_snapshot() {
    let mut backend = dead_backend();
    let mut caps = BackendCapabilities::default();
    let snap = backend.snapshot(&mut caps).await.unwrap();
    assert_eq!(snap.connection, ConnectionState::Error);
    assert_eq!(snap.server, ServerState::Unavailable);
    assert!(snap.error.is_some(), "transport error should be recorded");
    // A transport failure is a temporary observation, never "unsupported".
    assert_eq!(caps.health, EndpointAvailability::TemporarilyUnavailable);
}

#[tokio::test]
async fn single_failure_is_reconnecting_not_disconnected() {
    let mut detector = StateDetector::new();
    let mut backend = dead_backend();
    let mut caps = BackendCapabilities::default();
    let snap = backend.snapshot(&mut caps).await.unwrap();
    let out = detector.update(snap, Instant::now());
    assert_eq!(out.connection, ConnectionState::Reconnecting);
    assert_eq!(out.workload_phase, WorkloadPhase::Idle);
}

#[tokio::test]
async fn recovery_resets_counters_and_reports_connected() {
    let mut detector = StateDetector::new();

    // 1. A healthy observation with counters already advanced.
    let server = ready_server().await;
    let mut backend =
        LlamaCppBackend::new(server.uri().as_str(), Duration::from_secs(2), None).unwrap();
    let mut caps = BackendCapabilities {
        health: EndpointAvailability::Available,
        slots: EndpointAvailability::Available,
        props: EndpointAvailability::Available,
        ..Default::default()
    };
    let first = backend.snapshot(&mut caps).await.unwrap();
    let out1 = detector.update(first, Instant::now());
    assert_eq!(out1.connection, ConnectionState::Connected);
    assert_eq!(out1.server, ServerState::Ready);

    // 2. The server goes away.
    let mut dead = dead_backend();
    let dead_snap = dead.snapshot(&mut caps).await.unwrap();
    let out2 = detector.update(dead_snap, Instant::now());
    assert_eq!(out2.connection, ConnectionState::Reconnecting);
    // Only the health observation changes on a transport failure; the other
    // endpoints keep their last known state (they were not re-fetched).
    assert_eq!(caps.health, EndpointAvailability::TemporarilyUnavailable);
    assert_eq!(caps.slots, EndpointAvailability::Available);

    // 3. The server comes back. Counters were discarded on reconnect, so the
    //    first interval after recovery must not produce a rate. The phase is
    //    not guessed either: with a reset detector and one observation it
    //    stays ProcessingUnknown (Idle needs two consecutive observations).
    let third = backend.snapshot(&mut caps).await.unwrap();
    let out3 = detector.update(third, Instant::now());
    assert_eq!(out3.connection, ConnectionState::Connected);
    assert_eq!(out3.prompt_tokens_per_second, None);
    assert_eq!(out3.generation_tokens_per_second, None);
    assert_eq!(out3.workload_phase, WorkloadPhase::ProcessingUnknown);
    // The health observation recovers automatically — no manual reconnect.
    assert_eq!(caps.health, EndpointAvailability::Available);

    // 4. A second consecutive idle observation stabilizes the phase to Idle.
    let fourth = backend.snapshot(&mut caps).await.unwrap();
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

#[tokio::test]
async fn snapshot_recovers_temporarily_unavailable_endpoints() {
    // The probe (or a previous fetch) observed a transient failure; the
    // server is now healthy. A single snapshot must re-observe each
    // endpoint and recover it without a manual reconnect.
    let server = ready_server().await;
    let mut backend =
        LlamaCppBackend::new(server.uri().as_str(), Duration::from_secs(2), None).unwrap();
    let mut caps = BackendCapabilities {
        slots: EndpointAvailability::TemporarilyUnavailable,
        metrics: EndpointAvailability::ParseFailed,
        props: EndpointAvailability::Unknown,
        ..Default::default()
    };
    let snap = backend.snapshot(&mut caps).await.unwrap();
    assert_eq!(snap.connection, ConnectionState::Connected);
    assert_eq!(caps.slots, EndpointAvailability::Available);
    assert_eq!(caps.metrics, EndpointAvailability::Available);
    assert_eq!(caps.props, EndpointAvailability::Available);
    // A successful /slots observation with zero slots is Available: the
    // snapshot reports exactly zero slots, not a failure.
    assert!(snap.slots.is_empty());
}

#[tokio::test]
async fn temporary_metrics_failure_is_not_metrics_disabled() {
    // A 503 on /metrics is a temporary observation, never "unsupported"
    // (which would mean the server was started without --metrics).
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("metrics"))
        .respond_with(
            ResponseTemplate::new(503).set_body_json(serde_json::json!({"error": "busy"})),
        )
        .mount(&server)
        .await;
    let mut backend =
        LlamaCppBackend::new(server.uri().as_str(), Duration::from_secs(2), None).unwrap();
    // Previous good observation; the 503 must downgrade it to temporary.
    let mut caps =
        BackendCapabilities { metrics: EndpointAvailability::Available, ..Default::default() };
    let snap = backend.snapshot(&mut caps).await.unwrap();
    assert_eq!(caps.metrics, EndpointAvailability::TemporarilyUnavailable);
    assert_ne!(caps.metrics, EndpointAvailability::Unsupported);
    // The metrics data is missing, not an empty-but-valid reading.
    assert_eq!(snap.prompt_tokens_total, None);
}

#[tokio::test]
async fn props_failure_does_not_become_an_empty_valid_model() {
    // A failed /props must leave the model fields missing (None), never
    // fabricate an empty model, and must record the failure observation.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("props"))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(serde_json::json!({"error": "boom"})),
        )
        .mount(&server)
        .await;
    let mut backend =
        LlamaCppBackend::new(server.uri().as_str(), Duration::from_secs(2), None).unwrap();
    // Previous good observation; the 500 must downgrade it to temporary.
    let mut caps =
        BackendCapabilities { props: EndpointAvailability::Available, ..Default::default() };
    let snap = backend.snapshot(&mut caps).await.unwrap();
    assert_eq!(caps.props, EndpointAvailability::TemporarilyUnavailable);
    assert_eq!(snap.model_name, None, "a failed /props must not fabricate a model");
    assert_eq!(snap.model_path, None);
}

#[tokio::test]
async fn not_due_endpoint_keeps_its_last_successful_observation() {
    // The scheduling decision (which endpoints are due) lives in the collector;
    // the backend serves not-due endpoints from its value cache. A not-due
    // endpoint must keep its last successful observation: not the new server
    // data, and not a fake `None`.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("props"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"total_slots": 1})),
        )
        .mount(&server)
        .await;
    // Scoped: dropping the guard unmounts the mock (the server's /slots
    // "changes" between the two cycles).
    let slots_mock = Mock::given(method("GET"))
        .and(path("slots"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
            {
                "id": 1,
                "n_ctx": 1024,
                "is_processing": true,
                "id_task": 100,
                "n_prompt_tokens": 8,
                "n_prompt_tokens_processed": 8,
                "next_token": {"has_next_token": true, "has_new_line": false, "n_remain": -1, "n_decoded": 42}
            }
        ])))
        .mount_as_scoped(&server)
        .await;

    let mut backend =
        LlamaCppBackend::new(server.uri().as_str(), Duration::from_secs(2), None).unwrap();
    let mut caps = BackendCapabilities::default();

    // Cycle 1: everything is due -> /slots is fetched (42 decoded).
    let first = backend.snapshot_due(&mut caps, EndpointDue::ALL).await.unwrap();
    assert_eq!(first.slots.len(), 1);
    assert_eq!(first.slots[0].n_decoded, Some(42));
    assert_eq!(caps.slots, EndpointAvailability::Available);

    // The server now reports a different task on /slots.
    drop(slots_mock);
    let _ = Mock::given(method("GET"))
        .and(path("slots"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([
            {"id": 9, "is_processing": false}
        ])))
        .mount(&server)
        .await;

    // Cycle 2: only health is due -> the cached /slots observation is
    // retained (the old task), and the slots observation is left unchanged.
    let second = backend
        .snapshot_due(&mut caps, EndpointDue { health: true, ..Default::default() })
        .await
        .unwrap();
    assert_eq!(second.slots.len(), 1);
    assert_eq!(second.slots[0].id, 1, "not-due slots keep the previous observation");
    assert_eq!(second.slots[0].n_decoded, Some(42));
    assert_ne!(second.slots[0].id, 9, "not-due endpoints are not re-fetched");
    assert_eq!(caps.slots, EndpointAvailability::Available, "not-due observation unchanged");
}

#[tokio::test]
async fn not_due_observation_does_not_alter_rate_baselines() {
    // Re-feeding the cached (unchanged) slots observation must not shift the
    // detector's baselines or phase: identical values yield a zero rate (not
    // a fabricated delta) and a stable phase.
    let mut detector = StateDetector::new();
    let slot = SlotSnapshot {
        id: 1,
        task_id: Some(100),
        is_processing: true,
        n_ctx: Some(1024),
        n_tokens: Some(50),
        n_prompt_tokens: Some(8),
        n_prompt_tokens_processed: Some(8),
        n_decoded: Some(42),
        speculative: false,
        phase: SlotPhase::Idle,
    };
    let snap = llamatop::domain::BackendSnapshot {
        connection: ConnectionState::Connected,
        server: ServerState::Ready,
        prompt_tokens_total: Some(100),
        generation_tokens_total: Some(50),
        slots: vec![slot.clone()],
        ..Default::default()
    };

    let base = Instant::now();
    detector.update(snap.clone(), base);
    let out2 = detector.update(snap.clone(), base + Duration::from_secs(1));
    let out3 = detector.update(snap, base + Duration::from_secs(2));

    assert_eq!(
        out2.workload_phase, out3.workload_phase,
        "a repeated unchanged observation must not move the phase"
    );
    assert_eq!(
        out2.generation_tokens_per_second,
        Some(0.0),
        "unchanged counters yield a zero rate, not a fabricated delta"
    );
    assert_eq!(out3.generation_tokens_per_second, Some(0.0));
}
