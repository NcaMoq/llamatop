//! CLI integration tests for `snapshot` (pretty and JSON).
//!
//! The key guarantee: `snapshot --json` writes only a valid JSON document to
//! stdout. All diagnostics go to stderr. The tests point the binary at a
//! wiremock server and isolate the config path so the real user config is
//! never read.

use std::sync::OnceLock;
use std::time::Duration;

use assert_cmd::Command;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Isolate the config file per test binary so the user's real config is never
/// read or written.
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

async fn mount_ready(server: &MockServer) {
    Mock::given(method("GET"))
        .and(path("health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"status": "ok"})))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("slots"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!([])))
        .mount(server)
        .await;
    Mock::given(method("GET"))
        .and(path("props"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "total_slots": 1,
            "model_alias": "test-model",
            "build_info": "b1-test"
        })))
        .mount(server)
        .await;
}

#[tokio::test]
async fn json_stdout_is_pure_json_when_connected() {
    let server = MockServer::start().await;
    mount_ready(&server).await;

    let out = bin()
        .args(["snapshot", "--json", "--no-gpu"])
        .arg("--endpoint")
        .arg(server.uri())
        .args(["--refresh-ms", "300"])
        .timeout(Duration::from_secs(20))
        .output()
        .expect("run the binary");

    assert!(
        out.status.success(),
        "snapshot --json should exit 0 when connected; stderr: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    // stdout must be exactly one JSON document and nothing else.
    let parsed: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("stdout must be pure JSON: {stdout}");

    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["connection"]["state"], "connected");
    assert_eq!(parsed["server"]["state"], "ready");
    assert_eq!(parsed["server"]["model_name"], "test-model");
    assert_eq!(parsed["backend"], "llama.cpp");
}

#[tokio::test]
async fn json_stdout_is_pure_json_when_disconnected() {
    // Point at a dead port; the snapshot still emits a JSON document (with a
    // non-connected state) and exits 3.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let out = bin()
        .args(["snapshot", "--json", "--no-gpu"])
        .arg("--endpoint")
        .arg(format!("http://127.0.0.1:{port}"))
        .args(["--refresh-ms", "300"])
        .timeout(Duration::from_secs(20))
        .output()
        .expect("run the binary");

    assert_eq!(out.status.code(), Some(3), "snapshot --json should exit 3 when unreachable");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("stdout must be pure JSON: {stdout}");
    assert_eq!(parsed["schema_version"], 1);
    assert_ne!(parsed["connection"]["state"], "connected");
}

#[tokio::test]
async fn pretty_snapshot_prints_labeled_lines() {
    let server = MockServer::start().await;
    mount_ready(&server).await;

    let out = bin()
        .args(["snapshot", "--no-gpu", "--ascii"])
        .arg("--endpoint")
        .arg(server.uri())
        .args(["--refresh-ms", "300"])
        .timeout(Duration::from_secs(20))
        .output()
        .expect("run the binary");

    assert!(
        out.status.success(),
        "snapshot should exit 0 when connected; stderr: {:?}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("BACKEND"));
    assert!(stdout.contains("llama.cpp"));
    assert!(stdout.contains("CONNECTED"));
    assert!(stdout.contains("READY"));
    assert!(stdout.contains("test-model"));
    // No ANSI color codes in the pretty output.
    assert!(!stdout.contains("\u{1b}["), "pretty output must not contain ANSI escapes");
}

#[tokio::test]
async fn invalid_refresh_interval_exits_with_code_2() {
    let out = bin()
        .args(["snapshot", "--refresh-ms", "10", "--no-gpu"])
        .timeout(Duration::from_secs(20))
        .output()
        .expect("run the binary");

    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("refresh_interval_ms must be at least 100"), "stderr: {stderr}");
}

#[tokio::test]
async fn invalid_endpoint_exits_with_code_2() {
    let out = bin()
        .args(["snapshot", "--no-gpu"])
        .arg("--endpoint")
        .arg("not a url")
        .timeout(Duration::from_secs(20))
        .output()
        .expect("run the binary");

    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("valid http(s) URL"), "stderr: {stderr}");
}

#[tokio::test]
async fn help_and_version_are_stable() {
    let help =
        bin().arg("--help").timeout(Duration::from_secs(10)).output().expect("run the binary");
    assert!(help.status.success());
    let help_stdout = String::from_utf8_lossy(&help.stdout);
    assert!(help_stdout.contains("doctor"));
    assert!(help_stdout.contains("snapshot"));
    assert!(help_stdout.contains("--endpoint"));
    assert!(help_stdout.contains("--ascii"));
    assert!(help_stdout.contains("--no-gpu"));

    let version =
        bin().arg("--version").timeout(Duration::from_secs(10)).output().expect("run the binary");
    assert!(version.status.success());
    let version_stdout = String::from_utf8_lossy(&version.stdout);
    assert!(version_stdout.contains("llamatop"), "version line: {version_stdout}");
    assert!(version_stdout.contains(env!("CARGO_PKG_VERSION")), "version line: {version_stdout}");
}
