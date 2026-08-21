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
//! - each endpoint is fetched on its own interval
//!   (`health/slot/metrics/props_interval_ms`); a cycle runs at the earliest
//!   deadline and only fetches what is due, so a slow endpoint never blocks
//!   the others
//! - endpoints that are not due keep their last successful observation
//!   (cached by the backend); they are never fed as fake `None` values
//! - the collector never renders; it only emits `AppEvent`s
//! - reconnect commands are processed serially (no duplicate reconnects) and
//!   make every endpoint due immediately
//! - transport errors are redacted summaries; the detector still sees an
//!   error snapshot so connection state escalates per its hysteresis rules

use std::time::{Duration, Instant};

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::app::event::{AppEvent, BackendErrorSummary, CollectorCommand};
use crate::backend::llamacpp::LlamaCppBackend;
use crate::backend::{EndpointDue, InferenceBackend};
use crate::config::{Config, MIN_INTERVAL_MS};
use crate::detector::StateDetector;
use crate::domain::{BackendSnapshot, ConnectionState};

/// Upper bound applied to any per-endpoint interval: values beyond it are
/// clamped so deadline arithmetic can never overflow and every endpoint is
/// still re-observed on a bounded cycle.
const MAX_ENDPOINT_INTERVAL: Duration = Duration::from_secs(3600);

/// Per-endpoint schedule for the collector loop.
///
/// Each endpoint has its own interval and next-due instant. A cycle runs at
/// the earliest deadline, fetches everything that is due, and advances only
/// the deadlines of the endpoints that were fetched. This is the only place
/// a scheduling decision is made: the backend caches last successful values
/// but holds no time state.
#[derive(Debug, Clone)]
struct EndpointSchedule {
    next_health: Instant,
    next_slots: Instant,
    next_metrics: Instant,
    next_props: Instant,
    interval_health: Duration,
    interval_slots: Duration,
    interval_metrics: Duration,
    interval_props: Duration,
}

impl EndpointSchedule {
    fn new(config: &Config) -> Self {
        let now = Instant::now();
        // Config validation enforces MIN_INTERVAL_MS; the clamps are defense
        // in depth for callers that bypass validation (0 must never produce
        // a zero-length busy loop).
        let interval =
            |ms: u64| Duration::from_millis(ms.max(MIN_INTERVAL_MS)).min(MAX_ENDPOINT_INTERVAL);
        Self {
            next_health: now,
            next_slots: now,
            next_metrics: now,
            next_props: now,
            interval_health: interval(config.health_interval_ms),
            interval_slots: interval(config.slot_interval_ms),
            interval_metrics: interval(config.metrics_interval_ms),
            interval_props: interval(config.props_interval_ms),
        }
    }

    /// Which endpoints are due at `now` (initial state: all due).
    fn due(&self, now: Instant) -> EndpointDue {
        EndpointDue {
            health: now >= self.next_health,
            slots: now >= self.next_slots,
            metrics: now >= self.next_metrics,
            props: now >= self.next_props,
        }
    }

    /// After a cycle: the endpoints that were fetched get their next
    /// deadline set to `at` + their interval. Endpoints that were not due
    /// keep their existing deadline.
    fn advance(&mut self, due: EndpointDue, at: Instant) {
        if due.health {
            self.next_health = at + self.interval_health;
        }
        if due.slots {
            self.next_slots = at + self.interval_slots;
        }
        if due.metrics {
            self.next_metrics = at + self.interval_metrics;
        }
        if due.props {
            self.next_props = at + self.interval_props;
        }
    }

    /// How long until the next endpoint becomes due (zero when one is due
    /// now, e.g. right after a reconnect).
    fn wait(&self, now: Instant) -> Duration {
        let mut w = self.next_health.saturating_duration_since(now);
        w = w.min(self.next_slots.saturating_duration_since(now));
        w = w.min(self.next_metrics.saturating_duration_since(now));
        w.min(self.next_props.saturating_duration_since(now))
    }

    /// Manual reconnect: every endpoint is due immediately.
    fn mark_all_due(&mut self, now: Instant) {
        self.next_health = now;
        self.next_slots = now;
        self.next_metrics = now;
        self.next_props = now;
    }
}

/// Run the collector loop until a `Stop` command or channel shutdown.
pub async fn run(
    config: Config,
    events: UnboundedSender<AppEvent>,
    mut commands: UnboundedReceiver<CollectorCommand>,
) {
    let mut backend = match LlamaCppBackend::new(
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

    let mut schedule = EndpointSchedule::new(&config);
    let mut detector = StateDetector::new();
    let mut capabilities = backend.probe_capabilities().await.unwrap_or_default();
    let _ = events.send(AppEvent::BackendCapabilities(capabilities));

    loop {
        // Fetch what is due now, then wait for the next deadline or a
        // command. Because the fetch precedes the wait, a Reconnect command
        // triggers an immediate fetch on the next iteration.
        let now = Instant::now();
        let due = schedule.due(now);
        let before = capabilities;
        let snapshot = match backend.snapshot_due(&mut capabilities, due).await {
            Ok(snapshot) => Some(snapshot),
            // Authentication failures are terminal for the cycle: report them
            // and let the schedule decide when to try again.
            Err(err) => {
                let _ =
                    events.send(AppEvent::BackendError(BackendErrorSummary::new(err.to_string())));
                None
            }
        };
        if capabilities != before {
            let _ = events.send(AppEvent::BackendCapabilities(capabilities));
        }
        if let Some(snapshot) = snapshot {
            emit_snapshot(snapshot, &mut detector, &events);
        }
        schedule.advance(due, Instant::now());

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
                        schedule.mark_all_due(Instant::now());
                    }
                    // Stop, or the application side dropped the sender.
                    Some(CollectorCommand::Stop) | None => return,
                }
            }
            _ = tokio::time::sleep(schedule.wait(Instant::now())) => {}
        }
    }
}

/// Feed one snapshot to the detector and emit the resulting events.
///
/// A snapshot with an error connection state (transport failure) is reported
/// as a redacted summary so the UI can display it, and the detector still
/// sees the error snapshot so its connection state machine (Reconnecting ->
/// Disconnected) keeps working during outages.
fn emit_snapshot(
    snapshot: BackendSnapshot,
    detector: &mut StateDetector,
    events: &UnboundedSender<AppEvent>,
) {
    if snapshot.connection == ConnectionState::Error {
        if let Some(message) = snapshot.error.clone() {
            let _ = events.send(AppEvent::BackendError(BackendErrorSummary::new(message)));
        }
    }
    let stabilized = detector.update(snapshot, Instant::now());
    let _ = events.send(AppEvent::BackendSnapshot(Box::new(stabilized)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::event::{AppEvent, CollectorCommand};

    fn config_with(health_ms: u64, slots_ms: u64) -> Config {
        Config { health_interval_ms: health_ms, slot_interval_ms: slots_ms, ..Default::default() }
    }

    // --- Schedule unit tests (deterministic, no real sleeps) ---

    #[test]
    fn every_endpoint_is_due_initially() {
        let s = EndpointSchedule::new(&Config::default());
        assert_eq!(s.due(Instant::now()), EndpointDue::ALL);
    }

    #[test]
    fn health_and_slots_follow_their_own_intervals() {
        // Health every 100ms, slots every 300ms.
        let mut s = EndpointSchedule::new(&config_with(100, 300));
        let t0 = Instant::now();
        s.advance(EndpointDue::ALL, t0);
        assert_eq!(
            (s.interval_health, s.interval_slots),
            (Duration::from_millis(100), Duration::from_millis(300))
        );

        // At t0+150: health (100ms) is due, slots (300ms) is not.
        let t1 = t0 + Duration::from_millis(150);
        let due1 = s.due(t1);
        assert!(due1.health, "health is due at 150ms");
        assert!(!due1.slots, "slots is not due until 300ms");
        s.advance(due1, t1);

        // At t0+250: neither is due (next health t0+250 not reached yet
        // strictly, slots t0+300).
        let t2 = t0 + Duration::from_millis(240);
        let due2 = s.due(t2);
        assert!(!due2.health && !due2.slots, "nothing due before the next deadline");

        // At t0+260: health is due again (t0+250), slots still is not.
        let t3 = t0 + Duration::from_millis(260);
        let due3 = s.due(t3);
        assert!(due3.health && !due3.slots, "health re-due on its own interval");
        s.advance(due3, t3);

        // At t0+400: slots is due (t0+300), health is not (next t0+360+... =
        // t3+100 = t0+360 -> due at t0+360, so also due here).
        let t4 = t0 + Duration::from_millis(400);
        let due4 = s.due(t4);
        assert!(due4.slots, "slots is due after 300ms");
        assert!(due4.health, "health is due after its 100ms from t3");
    }

    #[test]
    fn metrics_and_props_follow_their_own_intervals() {
        // The same per-endpoint pattern extends to metrics and props: they are
        // scheduled independently of health/slots and of each other.
        let config = Config {
            health_interval_ms: 100,
            slot_interval_ms: 100,
            metrics_interval_ms: 300,
            props_interval_ms: 500,
            ..Default::default()
        };
        let mut s = EndpointSchedule::new(&config);
        let t0 = Instant::now();
        s.advance(EndpointDue::ALL, t0);
        assert_eq!(
            (s.interval_metrics, s.interval_props),
            (Duration::from_millis(300), Duration::from_millis(500))
        );

        // At t0+90: nothing is due yet (earliest deadline is t0+100).
        let t1 = t0 + Duration::from_millis(90);
        assert_eq!(s.due(t1), EndpointDue::NONE, "nothing due before t0+100");

        // At t0+150: health and slots re-due (100ms); metrics/props not yet.
        let t2 = t0 + Duration::from_millis(150);
        let due2 = s.due(t2);
        assert!(due2.health && due2.slots, "health/slots due on their 100ms");
        assert!(!due2.metrics, "metrics not due until 300ms");
        assert!(!due2.props, "props not due until 500ms");
        s.advance(due2, t2);

        // At t0+320: metrics is due (300ms); props still is not (500ms).
        let t3 = t0 + Duration::from_millis(320);
        let due3 = s.due(t3);
        assert!(due3.metrics, "metrics due after 300ms");
        assert!(!due3.props, "props not due until 500ms");
        s.advance(due3, t3);

        // At t0+520: props due (500ms).
        let t4 = t0 + Duration::from_millis(520);
        assert!(s.due(t4).props, "props due after 500ms");
    }

    #[test]
    fn wait_is_the_earliest_deadline() {
        let mut s = EndpointSchedule::new(&config_with(100, 300));
        let t0 = Instant::now();
        s.advance(EndpointDue::ALL, t0);
        // The next cycle must wait for health (100ms), not slots (300ms).
        let w = s.wait(t0 + Duration::from_millis(10));
        assert!(
            (w >= Duration::from_millis(80) && w <= Duration::from_millis(95)),
            "wait should be ~90ms (next health deadline), got {w:?}"
        );
    }

    #[test]
    fn reconnect_makes_every_endpoint_immediately_due() {
        let mut s = EndpointSchedule::new(&config_with(100, 300));
        let t0 = Instant::now();
        s.advance(EndpointDue::ALL, t0);
        let t1 = t0 + Duration::from_millis(50);
        assert!(!s.due(t1).slots, "slots not yet due");
        s.mark_all_due(t1);
        assert_eq!(s.due(t1), EndpointDue::ALL, "reconnect: everything due now");
        assert_eq!(s.wait(t1), Duration::ZERO, "no wait after reconnect");
    }

    #[test]
    fn intervals_are_clamped_to_the_minimum() {
        // A zero interval must not create a zero-length busy loop.
        let mut s = EndpointSchedule::new(&config_with(0, 0));
        let t0 = Instant::now();
        s.advance(EndpointDue::ALL, t0);
        assert!(
            s.wait(t0) >= Duration::from_millis(100),
            "a clamped interval must keep the cycle >= 100ms"
        );
    }

    #[test]
    fn huge_intervals_do_not_overflow_deadline_arithmetic() {
        let config = Config {
            health_interval_ms: u64::MAX,
            slot_interval_ms: u64::MAX,
            ..Default::default()
        };
        let mut s = EndpointSchedule::new(&config);
        let t0 = Instant::now();
        s.advance(EndpointDue::ALL, t0); // must not panic
        assert!(s.wait(t0) <= MAX_ENDPOINT_INTERVAL);
        // And the deadline is still reachable: it is bounded.
        assert!(s.next_health >= t0);
    }

    // --- Collector integration tests ---

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

    /// Stop must interrupt the scheduler's wait promptly, not just between
    /// fetches: the collector is sleeping when the command arrives.
    #[tokio::test]
    async fn stop_interrupts_the_scheduler_wait() {
        use tokio::sync::mpsc::unbounded_channel;

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        // Long intervals: after the first (failing) fetch the collector is
        // guaranteed to be inside its sleep when Stop arrives.
        let config = Config {
            endpoint: format!("http://127.0.0.1:{port}"),
            request_timeout_ms: 200,
            refresh_interval_ms: 1000,
            slot_interval_ms: 1000,
            metrics_interval_ms: 1000,
            props_interval_ms: 1000,
            ..Default::default()
        };
        let (tx, _rx) = unbounded_channel();
        let (cmd_tx, cmd_rx) = unbounded_channel();

        let handle = tokio::spawn(run(config, tx, cmd_rx));
        // Give the collector time to complete its first fetch and enter the
        // wait (the first fetch to a dead port fails fast).
        tokio::time::sleep(Duration::from_millis(300)).await;
        cmd_tx.send(CollectorCommand::Stop).expect("send");
        let result = tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("collector must exit promptly on Stop");
        result.expect("collector exits cleanly");
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
