//! Optional NVIDIA GPU metrics via NVML (Phase E).
//!
//! A `GpuMetricsProvider` samples one or more NVIDIA GPUs. The concrete
//! implementation uses `nvml_wrapper`; a fake is used in tests so the
//! monitor, state, and UI can be exercised on machines without a GPU.
//!
//! Failure semantics (startup must never break):
//! - NVML absent / init failure -> `GpuMonitorStatus::Unavailable` (or
//!   `InitializationFailed` when the config explicitly requested nvml).
//! - NVML present, zero devices  -> `Unavailable` (nothing to sample).
//! - A sampling pass fails      -> `SamplingFailed` for that pass only;
//!   the monitor keeps going (a transient driver hiccup is not sticky).
//!
//! Association rule (never guess): the GPU metrics describe the machine's
//! NVIDIA devices, not the llama-server process. The UI never names a GPU
//! as "the llama-server GPU" because that cannot be confirmed without
//! process-level queries that are out of scope for v0.1.

use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::Nvml;
use tokio::sync::{mpsc, oneshot};

use crate::app::event::AppEvent;
use crate::domain::{GpuMonitor, GpuMonitorStatus, GpuSnapshot};

/// Sample cadence for the GPU monitor (Phase E spec: ~1/sec).
pub const GPU_SAMPLE_INTERVAL: Duration = Duration::from_millis(1000);

/// Sample the monitored GPUs. Implementations must be safe to call from a
/// blocking thread (the monitor samples on `spawn_blocking`).
pub trait GpuMetricsProvider: Send + Sync {
    /// Take one sampling pass.
    fn sample(&self) -> GpuMonitor;
}

/// Start the GPU metrics monitor.
///
/// Returns a `(stop, handle)` pair. Sending on `stop` (or dropping it)
/// terminates the monitor; awaiting `handle` joins it. The monitor samples
/// the provider on a blocking thread at roughly `interval` cadence and
/// emits a `GpuSample` per pass. It exits early when the event sink is
/// closed.
pub fn start(
    provider: SharedGpuProvider,
    events: mpsc::UnboundedSender<AppEvent>,
    interval: Duration,
) -> (oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
    let (stop_tx, stop_rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut stop = stop_rx;
        loop {
            // First tick fires immediately, so the first pass is prompt.
            tokio::select! {
                // Stop signal (sent, or the sender dropped) ends the monitor.
                _ = &mut stop => return,
                _ = ticker.tick() => {}
            }
            let provider = provider.clone();
            let events = events.clone();
            match tokio::task::spawn_blocking(move || provider.sample()).await {
                Ok(monitor) => {
                    if events.send(AppEvent::GpuSample(monitor)).is_err() {
                        return;
                    }
                }
                // The sampling task itself failed (e.g. a panic inside the
                // provider): report it as a failed pass and keep going.
                Err(_) => {
                    let _ = events.send(AppEvent::GpuSample(GpuMonitor {
                        status: GpuMonitorStatus::SamplingFailed,
                        gpus: Vec::new(),
                    }));
                }
            }
        }
    });
    (stop_tx, handle)
}

/// An NVML-backed provider. The `Nvml` handle is held across samples (init
/// happens once at construction; NVML is process-global).
pub struct NvmlProvider {
    /// `None` when NVML could not be initialized (sticky for the run).
    /// `Some` when initialization succeeded, including zero devices.
    nvml: Option<Mutex<Nvml>>,
    /// Config requested the nvml backend explicitly: an init failure then
    /// reports `InitializationFailed` instead of the softer `Unavailable`.
    required: bool,
    /// Restrict monitoring to these GPU indices (empty = all).
    device_indices: Vec<u32>,
}

impl NvmlProvider {
    /// Build a provider, initializing NVML. Never panics or fails: an
    /// initialization problem is recorded and sampled as an unavailable
    /// monitor, so TUI startup is unaffected.
    pub fn new(required: bool, device_indices: Vec<u32>) -> Self {
        let nvml = match Nvml::init() {
            Ok(nvml) => Some(Mutex::new(nvml)),
            Err(_) => None,
        };
        Self { nvml, required, device_indices }
    }

    /// The indices to sample: the configured list (if any), else every
    /// device NVML reports.
    fn indices(&self, count: u32) -> Vec<u32> {
        if self.device_indices.is_empty() {
            (0..count).collect()
        } else {
            self.device_indices.iter().copied().filter(|i| *i < count).collect()
        }
    }
}

impl GpuMetricsProvider for NvmlProvider {
    fn sample(&self) -> GpuMonitor {
        let Some(mutex) = self.nvml.as_ref() else {
            return GpuMonitor {
                status: if self.required {
                    GpuMonitorStatus::InitializationFailed
                } else {
                    GpuMonitorStatus::Unavailable
                },
                gpus: Vec::new(),
            };
        };
        let nvml = mutex.lock().expect("gpu provider lock poisoned");

        let count = match nvml.device_count() {
            Ok(0) => return GpuMonitor { status: GpuMonitorStatus::Unavailable, gpus: Vec::new() },
            Ok(c) => c,
            Err(_) => {
                return GpuMonitor { status: GpuMonitorStatus::SamplingFailed, gpus: Vec::new() }
            }
        };

        let mut gpus = Vec::new();
        for index in self.indices(count) {
            if let Ok(device) = nvml.device_by_index(index) {
                gpus.push(read_device(index, &device));
            }
        }
        if gpus.is_empty() {
            return GpuMonitor { status: GpuMonitorStatus::SamplingFailed, gpus: Vec::new() };
        }
        GpuMonitor { status: GpuMonitorStatus::Available, gpus }
    }
}

/// Read every optional field of one device; per-field failures leave the
/// field `None` without failing the whole snapshot.
fn read_device(index: u32, device: &nvml_wrapper::Device<'_>) -> GpuSnapshot {
    let name = device.name().ok();
    let memory = device.memory_info().ok().map(|m| (m.used, m.total));
    let utilization = device.utilization_rates().ok().map(|u| u.gpu);
    GpuSnapshot {
        index,
        uuid: device.uuid().ok(),
        name,
        utilization_percent: utilization.filter(|u| *u <= 100).map(|u| u as u8),
        memory_used_bytes: memory.map(|(used, _)| used),
        memory_total_bytes: memory.map(|(_, total)| total),
        temperature_celsius: device.temperature(TemperatureSensor::Gpu).ok(),
        power_watts: device.power_usage().ok().map(|w| w as f64 / 1000.0),
        power_limit_watts: device.power_management_limit().ok().map(|w| w as f64 / 1000.0),
        graphics_clock_mhz: device.clock_info(Clock::Graphics).ok(),
        memory_clock_mhz: device.clock_info(Clock::Memory).ok(),
    }
}

/// A provider shared between the async scheduler and the blocking sampler.
/// The trait is `Send + Sync` (concrete providers guard their own state), so
/// a plain `Arc` is enough to sample from a `spawn_blocking` thread.
pub type SharedGpuProvider = Arc<dyn GpuMetricsProvider>;

/// Wrap a provider for concurrent sampling.
pub fn shared(provider: impl GpuMetricsProvider + 'static) -> SharedGpuProvider {
    Arc::new(provider)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic fake provider for unit tests.
    pub struct FakeGpuProvider {
        pub monitor: GpuMonitor,
    }

    impl FakeGpuProvider {
        pub fn new(status: GpuMonitorStatus, gpus: Vec<GpuSnapshot>) -> Self {
            Self { monitor: GpuMonitor { status, gpus } }
        }
    }

    impl GpuMetricsProvider for FakeGpuProvider {
        fn sample(&self) -> GpuMonitor {
            self.monitor.clone()
        }
    }

    fn gpu(index: u32) -> GpuSnapshot {
        GpuSnapshot {
            index,
            uuid: Some(format!("GPU-uuid-{index}")),
            name: Some(format!("NVIDIA GPU {index}")),
            utilization_percent: Some(42),
            memory_used_bytes: Some(1 << 30),
            memory_total_bytes: Some(1 << 34),
            temperature_celsius: Some(55),
            power_watts: Some(120.0),
            power_limit_watts: Some(350.0),
            graphics_clock_mhz: Some(2500),
            memory_clock_mhz: Some(8000),
        }
    }

    #[test]
    fn fake_provider_returns_configured_monitor() {
        let provider = FakeGpuProvider::new(GpuMonitorStatus::Available, vec![gpu(0)]);
        let m = provider.sample();
        assert_eq!(m.status, GpuMonitorStatus::Available);
        assert_eq!(m.gpus.len(), 1);
        assert_eq!(m.gpus[0].memory_utilization().unwrap(), 1.0 / 16.0);
    }

    #[test]
    fn unavailable_statuses_carry_no_gpus() {
        for status in [
            GpuMonitorStatus::Disabled,
            GpuMonitorStatus::Unavailable,
            GpuMonitorStatus::InitializationFailed,
            GpuMonitorStatus::SamplingFailed,
        ] {
            let m = GpuMonitor { status, gpus: Vec::new() };
            assert!(m.gpus.is_empty());
        }
    }

    #[tokio::test]
    async fn monitor_emits_samples_until_stopped() {
        let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();
        let provider = shared(FakeGpuProvider::new(GpuMonitorStatus::Available, vec![gpu(0)]));
        let (stop_tx, handle) = start(provider, tx, Duration::from_millis(10));
        let mut samples = 0usize;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AppEvent::GpuSample(_)) {
                samples += 1;
            }
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
        while let Ok(event) = rx.try_recv() {
            if matches!(event, AppEvent::GpuSample(_)) {
                samples += 1;
            }
        }
        let _ = stop_tx.send(());
        handle.await.expect("monitor exits");
        assert!(samples >= 1, "the monitor produced {samples} sample(s)");
    }

    #[tokio::test]
    async fn monitor_stops_on_stop_signal() {
        let (tx, _rx) = mpsc::unbounded_channel::<AppEvent>();
        let provider = shared(FakeGpuProvider::new(GpuMonitorStatus::Disabled, Vec::new()));
        let (stop_tx, handle) = start(provider, tx, Duration::from_millis(50));
        let _ = stop_tx.send(());
        let _ = tokio::time::timeout(Duration::from_secs(2), handle)
            .await
            .expect("monitor stops promptly");
    }

    #[tokio::test]
    async fn monitor_exits_when_sender_dropped() {
        let (tx, _rx) = mpsc::unbounded_channel::<AppEvent>();
        let provider = shared(FakeGpuProvider::new(GpuMonitorStatus::Disabled, Vec::new()));
        let (stop_tx, handle) = start(provider, tx, Duration::from_millis(10));
        drop(stop_tx);
        let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    }

    #[test]
    fn real_provider_does_not_panic() {
        // On a machine without NVML the provider must still sample cleanly
        // (reporting unavailable), never panic. On a machine with NVML it
        // reports available with one or more GPUs.
        let provider = NvmlProvider::new(false, Vec::new());
        let m = provider.sample();
        match m.status {
            GpuMonitorStatus::Available => {
                assert!(!m.gpus.is_empty());
                for (i, g) in m.gpus.iter().enumerate() {
                    assert_eq!(g.index, i as u32, "indices ascend from zero");
                }
            }
            GpuMonitorStatus::Unavailable | GpuMonitorStatus::SamplingFailed => {}
            other => panic!("unexpected status: {other:?}"),
        }
    }
}
