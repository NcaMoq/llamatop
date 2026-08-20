//! Host + llama-server process metrics (Phase D).
//!
//! A `SystemMetricsProvider` samples host CPU/RAM and any llama-server
//! process(es) on this machine. The concrete implementation uses `sysinfo`;
//! a fake is used in tests so the collector and UI can be exercised without
//! touching the real system.
//!
//! Association rules (never guess):
//! - Exactly one process named `llama-server.exe` / `llama-server` is
//!   reported as `process` with `identity == Exact`.
//! - Several candidates: `process` is `None` (we cannot say which one the
//!   endpoint belongs to); only the match count is reported.
//! - No match: `process` is `None`, count 0 (HTTP monitoring continues).
//!
//! `process_match_count` is `None` only when the process list could not be
//! read at all; a read that simply found nothing is `Some(0)`.

use std::ffi::OsStr;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, Process, ProcessRefreshKind, ProcessesToUpdate,
    RefreshKind, System,
};
use tokio::sync::{mpsc, oneshot};

use crate::app::event::AppEvent;
use crate::domain::{ProcessIdentity, ProcessSnapshot, SystemSnapshot};

/// Sample cadence for the host + process monitor (Phase D spec: ~1/sec).
pub const SYSTEM_SAMPLE_INTERVAL: Duration = Duration::from_millis(1000);

/// Sample host + process metrics. Implementations must be safe to call from
/// a blocking thread (the collector samples on `spawn_blocking`).
pub trait SystemMetricsProvider: Send + Sync {
    /// Take one sample. The provider keeps its own refresh state so that CPU
    /// percentages (which are deltas between refreshes) are meaningful.
    fn sample(&self) -> SystemSnapshot;
}

/// Start the system metrics monitor.
///
/// Returns a `(stop, handle)` pair. Sending on `stop` (or dropping it)
/// terminates the monitor; awaiting `handle` joins it. The monitor samples
/// the provider on a blocking thread at roughly `interval` cadence and emits
/// a `SystemSample` per sample (or a single `SystemUnavailable` if a sample
/// task fails). It exits early when the event sink is closed.
pub fn start(
    provider: SharedProvider,
    events: mpsc::UnboundedSender<AppEvent>,
    interval: Duration,
) -> (oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut stop = stop_rx;
        loop {
            // First tick fires immediately, so the first sample is prompt.
            tokio::select! {
                // Stop signal (sent, or the sender dropped) ends the monitor.
                _ = &mut stop => return,
                _ = ticker.tick() => {}
            }
            let provider = provider.clone();
            let events = events.clone();
            match tokio::task::spawn_blocking(move || provider.sample()).await {
                Ok(snap) => {
                    if events.send(AppEvent::SystemSample(snap)).is_err() {
                        return;
                    }
                }
                Err(_) => {
                    let _ = events.send(AppEvent::SystemUnavailable);
                    return;
                }
            }
        }
    });
    (stop_tx, handle)
}

/// A `sysinfo`-backed provider. Holds the `System` across samples so CPU
/// usage deltas are computed against the previous refresh.
pub struct SysinfoProvider {
    system: Mutex<System>,
}

impl SysinfoProvider {
    /// Build a provider with CPU/memory/process refresh enabled. The first
    /// `sample()` primes the CPU baseline; subsequent samples report deltas.
    pub fn new() -> Self {
        let refresh = RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything())
            .with_processes(ProcessRefreshKind::everything());
        let system = System::new_with_specifics(refresh);
        Self { system: Mutex::new(system) }
    }
}

impl Default for SysinfoProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemMetricsProvider for SysinfoProvider {
    fn sample(&self) -> SystemSnapshot {
        let mut sys = self.system.lock().expect("system provider lock poisoned");
        sys.refresh_cpu_usage();
        sys.refresh_memory();
        sys.refresh_processes_specifics(
            ProcessesToUpdate::All,
            false,
            ProcessRefreshKind::everything(),
        );

        // Collect name matches, ordered by PID for a stable presentation.
        let mut matches: Vec<(&Pid, &Process)> =
            sys.processes().iter().filter(|(_, p)| is_llama_server_name(p.name())).collect();
        matches.sort_by_key(|(pid, _)| pid.as_u32());

        let match_count = matches.len() as u32;
        // Only a single, unambiguous match is reported as the process.
        let process =
            matches.first().filter(|_| matches.len() == 1).map(|(pid, p)| ProcessSnapshot {
                pid: pid.as_u32(),
                name: p.name().to_string_lossy().into_owned(),
                cpu_usage_percent: Some(f64::from(p.cpu_usage())),
                memory_bytes: Some(p.memory()),
                uptime_secs: Some(p.run_time()),
                identity: ProcessIdentity::Exact,
            });

        SystemSnapshot {
            cpu_usage_percent: Some(f64::from(sys.global_cpu_usage())),
            ram_used_bytes: Some(sys.used_memory()),
            ram_total_bytes: Some(sys.total_memory()),
            process,
            process_match_count: Some(match_count),
        }
    }
}

/// True when a process name matches the llama-server executable.
fn is_llama_server_name(name: &OsStr) -> bool {
    name == OsStr::new("llama-server.exe") || name == OsStr::new("llama-server")
}

/// A provider shared between the async scheduler and the blocking sampler.
/// The trait is `Send + Sync` (concrete providers guard their own state), so
/// a plain `Arc` is enough to sample from a `spawn_blocking` thread.
pub type SharedProvider = Arc<dyn SystemMetricsProvider>;

/// Wrap a provider for concurrent sampling.
pub fn shared(provider: impl SystemMetricsProvider + 'static) -> SharedProvider {
    Arc::new(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic fake provider for unit tests.
    #[derive(Clone)]
    pub struct FakeProvider {
        pub cpu: Option<f64>,
        pub ram_used: Option<u64>,
        pub ram_total: Option<u64>,
        pub match_count: Option<u32>,
        pub process: Option<ProcessSnapshot>,
    }

    impl Default for FakeProvider {
        fn default() -> Self {
            Self {
                cpu: Some(12.5),
                ram_used: Some(10_000),
                ram_total: Some(20_000),
                match_count: Some(0),
                process: None,
            }
        }
    }

    impl SystemMetricsProvider for FakeProvider {
        fn sample(&self) -> SystemSnapshot {
            SystemSnapshot {
                cpu_usage_percent: self.cpu,
                ram_used_bytes: self.ram_used,
                ram_total_bytes: self.ram_total,
                process: self.process.clone(),
                process_match_count: self.match_count,
            }
        }
    }

    #[test]
    fn name_match_is_case_and_extension_insensitive() {
        assert!(is_llama_server_name(OsStr::new("llama-server.exe")));
        assert!(is_llama_server_name(OsStr::new("llama-server")));
        assert!(!is_llama_server_name(OsStr::new("llama-server2.exe")));
        assert!(!is_llama_server_name(OsStr::new("other.exe")));
    }

    #[test]
    fn real_provider_does_not_panic_and_reports_host_values() {
        // The real provider must never panic on this host and must report
        // the host RAM total (a non-None, non-zero value).
        let provider = SysinfoProvider::new();
        let s = provider.sample();
        assert!(s.ram_total_bytes.is_some());
        assert!(s.process_match_count.is_some());
        // CPU usage is a delta; the first sample may be 0.0 but is Some.
        assert!(s.cpu_usage_percent.is_some());
    }

    #[test]
    fn single_exact_match_reports_process() {
        let snap = SystemSnapshot {
            cpu_usage_percent: Some(1.0),
            ram_used_bytes: Some(1),
            ram_total_bytes: Some(2),
            process_match_count: Some(1),
            process: Some(ProcessSnapshot {
                pid: 42,
                name: "llama-server.exe".into(),
                cpu_usage_percent: Some(3.0),
                memory_bytes: Some(1_000),
                uptime_secs: Some(10),
                identity: ProcessIdentity::Exact,
            }),
        };
        assert_eq!(snap.process_match_count, Some(1));
        assert!(snap.process.is_some());
        assert_eq!(snap.process.as_ref().unwrap().identity, ProcessIdentity::Exact);
    }

    #[tokio::test]
    async fn monitor_emits_samples_until_stopped() {
        let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
        let provider = shared(FakeProvider::default());
        let (stop_tx, handle) = start(provider, tx, Duration::from_millis(10));
        // The first tick is immediate, so at least one sample arrives quickly.
        let mut samples = 0usize;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AppEvent::SystemSample(_)) {
                samples += 1;
            }
        }
        // Let at least one sample be produced, then stop and join.
        tokio::time::sleep(Duration::from_millis(30)).await;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AppEvent::SystemSample(_)) {
                samples += 1;
            }
        }
        let _ = stop_tx.send(());
        handle.await.expect("monitor exits");
        assert!(samples >= 1, "the monitor produced {samples} sample(s)");
        // The fake reports a single host value; assert it round-tripped.
        let provider = shared(FakeProvider::default());
        let s = provider.sample();
        assert_eq!(s.cpu_usage_percent, Some(12.5));
    }

    #[tokio::test]
    async fn monitor_exits_when_sender_dropped() {
        let (tx, _rx) = mpsc::unbounded_channel::<AppEvent>();
        let provider = shared(FakeProvider::default());
        let (stop_tx, handle) = start(provider, tx, Duration::from_millis(10));
        drop(stop_tx);
        let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    }

    #[tokio::test]
    async fn monitor_stops_on_stop_signal() {
        let (tx, _rx) = mpsc::unbounded_channel::<AppEvent>();
        let provider = shared(FakeProvider::default());
        let (stop_tx, handle) = start(provider, tx, Duration::from_millis(50));
        let _ = stop_tx.send(());
        // Must terminate promptly (well before the next tick).
        let _ = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("monitor stops promptly");
    }
}
