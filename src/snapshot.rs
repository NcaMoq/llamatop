//! One-shot snapshot capture (the `snapshot` command).
//!
//! Captures two observations of the backend with a short interval between
//! them so that delta-based rates and per-slot phases can be computed by the
//! state detector. The second, stabilized snapshot is what gets rendered.
//!
//! The interval is bounded (300..=800 ms) so that a single `snapshot` call
//! stays fast while still producing a usable rate.

use std::time::{Duration, Instant};

use crate::backend::llamacpp::LlamaCppBackend;
use crate::backend::{BackendCapabilities, InferenceBackend};
use crate::config::Config;
use crate::detector::StateDetector;
use crate::domain::BackendSnapshot;

/// How long to wait between the two observations.
fn sample_interval(config: &Config) -> Duration {
    let ms = config.refresh_interval_ms.clamp(300, 800);
    Duration::from_millis(ms)
}

/// The result of a one-shot capture.
pub struct Snapshot {
    pub backend: String,
    pub endpoint: String,
    pub snapshot: BackendSnapshot,
    pub capabilities: BackendCapabilities,
}

impl Snapshot {
    /// Exit code: 0 when connected, 3 when the server could not be reached.
    pub fn exit_code(&self) -> i32 {
        use crate::domain::ConnectionState;
        match self.snapshot.connection {
            ConnectionState::Connected => 0,
            _ => 3,
        }
    }
}

/// Capture a one-shot snapshot of the backend.
///
/// Transport failures do not produce an `Err`; they are encoded in the
/// snapshot (connection = Disconnected/Error) so the renderer can report
/// them. Only client-construction failures return `Err`.
pub async fn capture(config: &Config) -> anyhow::Result<Snapshot> {
    let backend = LlamaCppBackend::new(
        &config.endpoint,
        config.request_timeout(),
        config.api_key().as_deref(),
    )
    .map_err(|e| {
        anyhow::anyhow!("cannot use endpoint {}: {e}", crate::endpoint::redact(&config.endpoint))
    })?;

    let endpoint = crate::endpoint::redact(backend.endpoint());

    // Probe capabilities; if the server is unreachable the probe fails and we
    // fall through with default (all-unknown) capabilities. The snapshot
    // call below reports the transport failure in the domain snapshot.
    let capabilities = backend.probe_capabilities().await.unwrap_or_default();

    let mut detector = StateDetector::new();

    let first = backend.snapshot(&capabilities).await.unwrap_or_default();
    detector.update(first, Instant::now());

    // A short, bounded wait so the second observation yields a real delta.
    tokio::time::sleep(sample_interval(config)).await;

    let second = backend.snapshot(&capabilities).await.unwrap_or_default();
    let stabilized = detector.update(second, Instant::now());

    Ok(Snapshot {
        backend: backend.name().to_string(),
        endpoint,
        snapshot: stabilized,
        capabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::ConnectionState;

    #[test]
    fn sample_interval_is_clamped() {
        let config = Config { refresh_interval_ms: 100, ..Default::default() };
        assert_eq!(sample_interval(&config), Duration::from_millis(300));

        let config = Config { refresh_interval_ms: 2000, ..Default::default() };
        assert_eq!(sample_interval(&config), Duration::from_millis(800));

        let config = Config { refresh_interval_ms: 500, ..Default::default() };
        assert_eq!(sample_interval(&config), Duration::from_millis(500));
    }

    #[test]
    fn exit_code_reflects_connection_state() {
        let mut snap = Snapshot {
            backend: "llama.cpp".into(),
            endpoint: "http://127.0.0.1:8080/".into(),
            snapshot: BackendSnapshot::default(),
            capabilities: BackendCapabilities::default(),
        };
        snap.snapshot.connection = ConnectionState::Disconnected;
        assert_eq!(snap.exit_code(), 3);

        snap.snapshot.connection = ConnectionState::Connected;
        assert_eq!(snap.exit_code(), 0);
    }
}
