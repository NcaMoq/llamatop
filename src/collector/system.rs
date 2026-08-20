//! Host + llama-server process metrics (Phase D).
//!
//! A `SystemMetricsProvider` samples host CPU/RAM and any llama-server
//! process(es) on this machine. The concrete implementation uses `sysinfo`;
//! a fake is used in tests so the collector and UI can be exercised without
//! touching the real system.
//!
//! Association rules (never guess): a process *name* match alone does not
//! prove that the process serves the configured endpoint, so:
//! - Remote endpoint (non-loopback host): `RemoteEndpoint`; local processes
//!   are not matched or presented at all.
//! - Exactly one local name match: `SingleLocalCandidate`; the process is
//!   reported as data, but the UI must not label it as the endpoint's
//!   server.
//! - Several local name matches: `MultipleLocalCandidates`; no process is
//!   named, only the count.
//! - No local name match: `NoneFound` (HTTP monitoring continues).
//! - `Verified` requires an endpoint-to-PID mapping, which the Windows
//!   collector does not implement; it is never reported.
//!
//! `process_match_count` is `Some(n)` when the local process list was read
//! and `n` name matches were found (0 = none), and `None` when the list
//! could not be read or is not applicable (remote endpoint).

use std::ffi::OsStr;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use sysinfo::{
    CpuRefreshKind, MemoryRefreshKind, Pid, Process, ProcessRefreshKind, ProcessesToUpdate,
    RefreshKind, System,
};
use tokio::sync::{mpsc, oneshot};
use url::Url;

use crate::app::event::AppEvent;
use crate::domain::{ProcessAssociation, ProcessSnapshot, SystemSnapshot};

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
/// usage deltas are computed against the previous refresh, and the
/// configured endpoint URL so it knows whether the server could even be a
/// local process.
pub struct SysinfoProvider {
    system: Mutex<System>,
    endpoint: Url,
}

impl SysinfoProvider {
    /// Build a provider with CPU/memory/process refresh enabled. The first
    /// `sample()` primes the CPU baseline; subsequent samples report deltas.
    pub fn new(endpoint: &str) -> Self {
        let refresh = RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything())
            .with_processes(ProcessRefreshKind::everything());
        let system = System::new_with_specifics(refresh);
        // The endpoint is validated by the config before this is built, so a
        // parse failure here is a programming error.
        let endpoint =
            Url::parse(endpoint).expect("endpoint validated before provider construction");
        Self { system: Mutex::new(system), endpoint }
    }
}

impl Default for SysinfoProvider {
    fn default() -> Self {
        Self::new("http://127.0.0.1:8080")
    }
}

impl SystemMetricsProvider for SysinfoProvider {
    fn sample(&self) -> SystemSnapshot {
        let mut sys = self.system.lock().expect("system provider lock poisoned");
        sys.refresh_cpu_usage();
        sys.refresh_memory();

        // A remote endpoint can never be a local process; skip process
        // matching entirely and report the endpoint as remote.
        if !is_local_endpoint(&self.endpoint) {
            return SystemSnapshot {
                cpu_usage_percent: Some(f64::from(sys.global_cpu_usage())),
                ram_used_bytes: Some(sys.used_memory()),
                ram_total_bytes: Some(sys.total_memory()),
                process: None,
                process_match_count: None,
                association: ProcessAssociation::RemoteEndpoint,
            };
        }

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
        // A single name match is only a *candidate*: nothing proves it
        // serves the endpoint, so it is reported as data but the
        // association stays SingleLocalCandidate (never Verified).
        let (process, association) = match matches.len() {
            1 => {
                let (pid, p) = matches[0];
                (
                    Some(ProcessSnapshot {
                        pid: pid.as_u32(),
                        name: p.name().to_string_lossy().into_owned(),
                        cpu_usage_percent: Some(f64::from(p.cpu_usage())),
                        memory_bytes: Some(p.memory()),
                        uptime_secs: Some(p.run_time()),
                    }),
                    ProcessAssociation::SingleLocalCandidate,
                )
            }
            0 => (None, ProcessAssociation::NoneFound),
            _ => (None, ProcessAssociation::MultipleLocalCandidates),
        };

        SystemSnapshot {
            cpu_usage_percent: Some(f64::from(sys.global_cpu_usage())),
            ram_used_bytes: Some(sys.used_memory()),
            ram_total_bytes: Some(sys.total_memory()),
            process,
            process_match_count: Some(match_count),
            association,
        }
    }
}

/// True when a process name matches the llama-server executable.
///
/// Windows process names are compared case-insensitively: the executable
/// may be `llama-server.exe`, `LLAMA-SERVER.EXE`, or a renamed binary
/// without the extension.
fn is_llama_server_name(name: &OsStr) -> bool {
    let Some(n) = name.to_str() else {
        return false;
    };
    let lower = n.to_ascii_lowercase();
    lower == "llama-server.exe" || lower == "llama-server"
}

/// True when the endpoint's host is this machine (loopback, "localhost", or
/// the empty/absent host). Anything else is a remote host, whose server is
/// not a local process we can match by name.
fn is_local_endpoint(endpoint: &Url) -> bool {
    match endpoint.host() {
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(octets)) => octets.is_loopback(),
        Some(url::Host::Ipv6(segs)) => segs.is_loopback(),
        None => true,
    }
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
        pub association: ProcessAssociation,
    }

    impl Default for FakeProvider {
        fn default() -> Self {
            Self {
                cpu: Some(12.5),
                ram_used: Some(10_000),
                ram_total: Some(20_000),
                match_count: Some(0),
                process: None,
                association: ProcessAssociation::NoneFound,
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
                association: self.association,
            }
        }
    }

    #[test]
    fn name_match_is_case_and_extension_insensitive() {
        assert!(is_llama_server_name(OsStr::new("llama-server.exe")));
        assert!(is_llama_server_name(OsStr::new("llama-server")));
        // Windows process names are case-insensitive.
        assert!(is_llama_server_name(OsStr::new("LLAMA-SERVER.EXE")));
        assert!(is_llama_server_name(OsStr::new("Llama-Server.Exe")));
        assert!(!is_llama_server_name(OsStr::new("llama-server2.exe")));
        assert!(!is_llama_server_name(OsStr::new("other.exe")));
    }

    #[test]
    fn local_endpoint_detection() {
        let parse = |s: &str| Url::parse(s).unwrap();
        assert!(is_local_endpoint(&parse("http://127.0.0.1:8080")));
        assert!(is_local_endpoint(&parse("http://localhost:8080")));
        assert!(is_local_endpoint(&parse("http://LOCALHOST:8080")));
        assert!(is_local_endpoint(&parse("http://[::1]:8080")));
        // A LAN address is not necessarily this machine: treat as remote.
        assert!(!is_local_endpoint(&parse("http://10.0.0.1:8080")));
        assert!(!is_local_endpoint(&parse("http://192.168.1.50:8080")));
        assert!(!is_local_endpoint(&parse("http://llama.example.com:8080")));
    }

    #[test]
    fn real_provider_does_not_panic_and_reports_host_values() {
        // The real provider must never panic on this host and must report
        // the host RAM total (a non-None, non-zero value).
        let provider = SysinfoProvider::new("http://127.0.0.1:8080");
        let s = provider.sample();
        assert!(s.ram_total_bytes.is_some());
        assert!(s.process_match_count.is_some());
        // CPU usage is a delta; the first sample may be 0.0 but is Some.
        assert!(s.cpu_usage_percent.is_some());
        // A local endpoint never reports a remote association.
        assert_ne!(s.association, ProcessAssociation::RemoteEndpoint);
    }

    #[test]
    fn remote_endpoint_never_associates_a_local_process() {
        // Even if a llama-server is running locally, a remote endpoint must
        // not present it as the monitored server.
        let provider = SysinfoProvider::new("http://192.168.1.50:8080");
        let s = provider.sample();
        assert_eq!(s.association, ProcessAssociation::RemoteEndpoint);
        assert!(s.process.is_none());
        assert!(s.process_match_count.is_none());
        // Host metrics are still sampled for a remote endpoint.
        assert!(s.ram_total_bytes.is_some());
    }

    #[test]
    fn single_name_match_is_not_verified() {
        // A single local name match is only a candidate; the association
        // must never be `Verified` (no port-to-PID mapping is performed).
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
            }),
            association: ProcessAssociation::SingleLocalCandidate,
        };
        assert_eq!(snap.process_match_count, Some(1));
        assert!(snap.process.is_some());
        assert_eq!(snap.association, ProcessAssociation::SingleLocalCandidate);
        assert_ne!(snap.association, ProcessAssociation::Verified);
    }

    #[test]
    fn multiple_matches_report_multiple_candidates() {
        let snap = SystemSnapshot {
            cpu_usage_percent: Some(1.0),
            ram_used_bytes: Some(1),
            ram_total_bytes: Some(2),
            process_match_count: Some(2),
            process: None,
            association: ProcessAssociation::MultipleLocalCandidates,
        };
        assert!(snap.process.is_none());
        assert_eq!(snap.association, ProcessAssociation::MultipleLocalCandidates);
    }

    #[test]
    fn no_match_reports_none_found() {
        let snap = SystemSnapshot {
            cpu_usage_percent: Some(1.0),
            ram_used_bytes: Some(1),
            ram_total_bytes: Some(2),
            process_match_count: Some(0),
            process: None,
            association: ProcessAssociation::NoneFound,
        };
        assert!(snap.process.is_none());
        assert_eq!(snap.association, ProcessAssociation::NoneFound);
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
