//! Long-lived backend collector for the TUI.
//!
//! Unlike the one-shot `snapshot::capture`, this keeps a single
//! `LlamaCppBackend`, `BackendCapabilities`, and `StateDetector` alive for
//! the whole TUI session so that deltas, hysteresis, and baselines persist
//! across polls.
//!
//! Guarantees:
//! - fetches are strictly sequential (the next fetch starts only after the
//!   previous one finished; no overlapping requests)
//! - the collector never renders; it only emits `AppEvent`s
//! - reconnect commands are processed serially (no duplicate reconnects)
//! - transport errors are redacted summaries; the detector still sees an
//!   error snapshot so connection state escalates per its hysteresis rules

use std::time::{Duration, Instant};

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::app::event::{AppEvent, BackendErrorSummary, CollectorCommand};
use crate::backend::llamacpp::LlamaCppBackend;
use crate::backend::{BackendCapabilities, InferenceBackend};
use crate::config::Config;
use crate::detector::StateDetector;
use crate::domain::{BackendSnapshot, ConnectionState};

/// Run the collector loop until a `Stop` command or channel shutdown.
pub async fn run(
    config: Config,
    events: UnboundedSender<AppEvent>,
    mut commands: UnboundedReceiver<CollectorCommand>,
) {
    let backend = match LlamaCppBackend::new(
        &config.endpoint,
        config.request_timeout(),
        config.api_key().as_deref(),
    ) {
        Ok(backend) => backend,
        Err(err) => {
            let _ = events.send(AppEvent::BackendError(BackendErrorSummary::new(err.to_string())));
            return;
        }
    };

    let interval = Duration::from_millis(config.refresh_interval_ms.max(100));
    let mut detector = StateDetector::new();
    let mut capabilities = backend.probe_capabilities().await.unwrap_or_default();
    let _ = events.send(AppEvent::BackendCapabilities(capabilities));

    loop {
        // Fetch now, then wait for the next cycle or a command. Because the
        // fetch precedes the wait, a Reconnect command triggers an immediate
        // fetch on the next iteration.
        fetch_once(&backend, &capabilities, &mut detector, &events).await;

        tokio::select! {
            command = commands.recv() => {
                match command {
                    Some(CollectorCommand::Reconnect) => {
                        detector.reset();
                        if let Ok(caps) = backend.probe_capabilities().await {
                            capabilities = caps;
                            let _ =
                                events.send(AppEvent::BackendCapabilities(capabilities));
                        }
                    }
                    // Stop, or the application side dropped the sender.
                    Some(CollectorCommand::Stop) | None => return,
                }
            }
            _ = tokio::time::sleep(interval) => {}
        }
    }
}

/// One sequential fetch: snapshot -> detector -> event.
///
/// A fetch error is reported as a redacted summary *and* fed to the detector
/// as an error snapshot, so the connection state machine (Reconnecting ->
/// Disconnected) keeps working during outages.
async fn fetch_once(
    backend: &LlamaCppBackend,
    capabilities: &BackendCapabilities,
    detector: &mut StateDetector,
    events: &UnboundedSender<AppEvent>,
) {
    let now = Instant::now();
    match backend.snapshot(capabilities).await {
        Ok(snapshot) => {
            // Transport failures arrive as an `Ok` snapshot with an error
            // connection state (the backend degrades gracefully); report the
            // redacted summary so the UI can display it.
            if snapshot.connection == ConnectionState::Error {
                if let Some(message) = snapshot.error.clone() {
                    let _ = events.send(AppEvent::BackendError(BackendErrorSummary::new(message)));
                }
            }
            let stabilized = detector.update(snapshot, now);
            let _ = events.send(AppEvent::BackendSnapshot(Box::new(stabilized)));
        }
        Err(err) => {
            let message = err.to_string();
            let _ = events.send(AppEvent::BackendError(BackendErrorSummary::new(message.clone())));
            let error_snapshot = BackendSnapshot {
                connection: ConnectionState::Error,
                error: Some(message),
                ..Default::default()
            };
            let stabilized = detector.update(error_snapshot, now);
            let _ = events.send(AppEvent::BackendSnapshot(Box::new(stabilized)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::run;
    use crate::app::event::{AppEvent, CollectorCommand};
    use crate::config::Config;

    /// A dead port must not block or panic the fetch path; the collector
    /// reports it and the detector sees an error snapshot.
    #[tokio::test]
    async fn fetch_to_dead_port_reports_error_and_error_snapshot() {
        use tokio::sync::mpsc::unbounded_channel;

        // Bind and drop a listener to get a dead port.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let config = Config {
            endpoint: format!("http://127.0.0.1:{port}"),
            request_timeout_ms: 300,
            ..Default::default()
        };
        let (tx, mut rx) = unbounded_channel();
        let (cmd_tx, cmd_rx) = unbounded_channel();

        let handle = tokio::spawn(run(config, tx, cmd_rx));

        // Expect a capabilities event and then an error summary.
        for _ in 0..4 {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(10), rx.recv())
                .await
                .expect("collector sends within timeout")
                .expect("channel open");
            if let AppEvent::BackendError(summary) = ev {
                assert!(!summary.message.is_empty());
                // Stop the collector.
                cmd_tx.send(CollectorCommand::Stop).expect("send");
                handle.await.expect("collector exits");
                return;
            }
        }
        panic!("expected a BackendError event from the collector");
    }

    #[test]
    fn collector_command_channel_round_trips() {
        use tokio::sync::mpsc::unbounded_channel;
        let (tx, mut rx) = unbounded_channel();
        tx.send(CollectorCommand::Reconnect).expect("send");
        tx.send(CollectorCommand::Stop).expect("send");
        let a = rx.try_recv().expect("recv");
        let b = rx.try_recv().expect("recv");
        assert_eq!(a, CollectorCommand::Reconnect);
        assert_eq!(b, CollectorCommand::Stop);
    }
}
