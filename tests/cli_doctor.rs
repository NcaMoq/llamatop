//! CLI integration tests for `doctor`.
//!
//! doctor performs network checks, so the tests point it at a wiremock server
//! (healthy case) or a dead port (unreachable case) and assert on the report
//! structure, symbols, and exit code.

use std::sync::OnceLock;
use std::time::Duration;

use assert_cmd::Command;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

static CONFIG_PATH: OnceLock<String> = OnceLock::new();

fn isolated_config_path() -> &'static String {
    CONFIG_PATH.get_or_init(|| {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml").to_string_lossy().into_owned();
        std::mem::forget(dir); // keep the tempdir alive for the whole process
        path
    })
}

fn bin() -> Command {
    let mut cmd = Command::cargo_bin("llamatop").expect("build the binary");
    cmd.env("LLAMATOP_CONFIG_PATH", isolated_config_path());
    cmd
}

#[tokio::test]
async fn doctor_reports_all_ok_against_healthy_server() {
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
        .and(path("metrics"))
        .respond_with(ResponseTemplate::new(200).set_body_raw("", "text/plain"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("props"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "total_slots": 1,
            "model_alias": "test-model"
        })))
        .mount(&server)
        .await;

    let out = bin()
        .args(["doctor", "--no-gpu", "--ascii"])
        .arg("--endpoint")
        .arg(server.uri())
        .timeout(Duration::from_secs(30))
        .output()
        .expect("run the binary");

    assert!(
        out.status.success(),
        "doctor should exit 0 for a healthy server; stderr: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Checking llama.cpp server..."));
    assert!(stdout.contains("[OK] Server reachable"));
    assert!(stdout.contains("[OK] Model ready"));
    assert!(stdout.contains("[OK] Slots endpoint available"));
    assert!(stdout.contains("[OK] Metrics endpoint available"));
    assert!(stdout.contains("[OK] Props endpoint available"));
    assert!(stdout.contains("Ready to monitor."));
    // ASCII mode must not emit Unicode marks.
    assert!(!stdout.contains('\u{2713}'), "ASCII mode must not use the checkmark");
}

#[tokio::test]
async fn doctor_flags_unavailable_metrics_as_warning() {
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
    // /metrics disabled: 501 not_supported (llama.cpp behavior without --metrics).
    Mock::given(method("GET"))
        .and(path("metrics"))
        .respond_with(
            ResponseTemplate::new(501)
                .set_body_json(serde_json::json!({"error": {"message": "not supported", "type": "not_supported_error"}})),
        )
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("props"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({"total_slots": 1})),
        )
        .mount(&server)
        .await;

    let out = bin()
        .args(["doctor", "--no-gpu", "--ascii"])
        .arg("--endpoint")
        .arg(server.uri())
        .timeout(Duration::from_secs(30))
        .output()
        .expect("run the binary");

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[WARN] Metrics endpoint unavailable (HTTP 501)"));
    assert!(stdout.contains("llamatop can start, but some metrics will be hidden."));
}

#[tokio::test]
async fn doctor_reports_unreachable_server() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let out = bin()
        .args(["doctor", "--no-gpu", "--ascii"])
        .arg("--endpoint")
        .arg(format!("http://127.0.0.1:{port}"))
        .args(["--refresh-ms", "200"])
        .timeout(Duration::from_secs(30))
        .output()
        .expect("run the binary");

    // An unreachable server is a blocking error for monitoring.
    assert_eq!(out.status.code(), Some(1));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[ERR] Server unreachable"));
    assert!(stdout.contains("Fix the issues above before monitoring."));
}

#[tokio::test]
async fn doctor_reports_loading_server() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("health"))
        .respond_with(ResponseTemplate::new(503).set_body_json(serde_json::json!({
            "error": {"message": "Loading model", "type": "unavailable_error", "code": 503}
        })))
        .mount(&server)
        .await;
    // While loading, the llama.cpp server answers every endpoint with 503.
    for p in ["slots", "metrics", "props"] {
        Mock::given(method("GET"))
            .and(path(p))
            .respond_with(
                ResponseTemplate::new(503)
                    .set_body_json(serde_json::json!({"error": {"message": "Loading model"}})),
            )
            .mount(&server)
            .await;
    }

    let out = bin()
        .args(["doctor", "--no-gpu", "--ascii"])
        .arg("--endpoint")
        .arg(server.uri())
        .timeout(Duration::from_secs(30))
        .output()
        .expect("run the binary");

    // Loading is not a hard failure: the server is reachable, the model is
    // just not ready yet.
    assert!(out.status.success(), "stderr: {:?}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[OK] Server reachable"));
    assert!(stdout.contains("[WARN] Model still loading"));
}
