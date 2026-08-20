//! The TUI runtime: owns the `AppState`, consumes `AppEvent`s, and drives
//! the renderer callback.
//!
//! The state is owned by exactly one task (no `Arc<Mutex<..>>` sharing).
//! Producers:
//! - the input reader thread sends `Input`/`Resize` events
//! - the backend collector task sends `BackendSnapshot`/`BackendCapabilities`
//!   /`BackendError` events
//! - a render-tick timer (10 FPS cap) sends `Tick`
//!
//! The renderer is passed in as a callback so this module stays free of
//! ratatui/crossterm types.

use std::time::Duration;

use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::app::event::{AppEvent, CollectorCommand};
use crate::app::state::AppState;
use crate::collector::backend;
use crate::collector::gpu as gpu_monitor;
use crate::collector::system as system_monitor;
use crate::config::Config;

/// Draw cadence: at most 10 FPS.
const TICK: Duration = Duration::from_millis(100);

/// Run the application event loop until the user quits.
///
/// `events_tx` is handed to the collector; the input reader (started by the
/// caller) holds its own clone. Returns the exit code (0 for a clean quit);
/// the collector is stopped and joined before returning.
pub async fn run(
    config: Config,
    events_tx: UnboundedSender<AppEvent>,
    mut events_rx: UnboundedReceiver<AppEvent>,
    initial_size: (u16, u16),
    mut draw: impl FnMut(&AppState) -> std::io::Result<()>,
) -> anyhow::Result<i32> {
    let mut state = AppState::new(&config);
    state.terminal_size = initial_size;

    // The optional host + process monitor (Phase D). It is a separate sender
    // on the same event channel, so it must be stopped before the loop's
    // receiver can see "all senders dropped". Disabled by config.
    let system_monitor = if config.show_system {
        let provider =
            system_monitor::shared(system_monitor::SysinfoProvider::new(&config.endpoint));
        Some(system_monitor::start(
            provider,
            events_tx.clone(),
            system_monitor::SYSTEM_SAMPLE_INTERVAL,
        ))
    } else {
        None
    };

    // The optional NVML GPU monitor (Phase E). NVML absence must not break
    // startup: the provider records the init failure and samples it as an
    // unavailable monitor. Disabled by config.
    let gpu_monitor = if config.show_gpu {
        let required = config.gpu.backend == "nvml";
        let provider = gpu_monitor::shared(gpu_monitor::NvmlProvider::new(
            required,
            config.gpu.device_indices.clone(),
        ));
        Some(gpu_monitor::start(provider, events_tx.clone(), gpu_monitor::GPU_SAMPLE_INTERVAL))
    } else {
        None
    };

    let (commands_tx, commands_rx) = unbounded_channel::<CollectorCommand>();
    let collector = tokio::spawn(backend::run(config, events_tx, commands_rx));

    let run_result = run_event_loop(&mut state, &mut events_rx, &commands_tx, &mut draw).await;

    // Mandatory cleanup, independent of how the loop ended: stop and join
    // both monitors and the collector so no detached task survives, even
    // after a draw/command failure.
    let system_joined = match system_monitor {
        Some((stop_tx, handle)) => {
            let _ = stop_tx.send(());
            handle.await
        }
        None => Ok(()),
    };
    let gpu_joined = match gpu_monitor {
        Some((stop_tx, handle)) => {
            let _ = stop_tx.send(());
            handle.await
        }
        None => Ok(()),
    };
    let _ = commands_tx.send(CollectorCommand::Stop);
    let joined = collector.await;

    finish_run(run_result, joined, system_joined, gpu_joined)
}

/// Combine the event-loop result with the collector and monitor joins.
///
/// - The loop's own error always wins: cleanup failures never mask it.
/// - A collector or monitor panic (join error) on an otherwise clean exit is
///   reported as a runtime error instead of silently returning success.
fn finish_run(
    run_result: anyhow::Result<i32>,
    joined: Result<(), tokio::task::JoinError>,
    system_joined: Result<(), tokio::task::JoinError>,
    gpu_joined: Result<(), tokio::task::JoinError>,
) -> anyhow::Result<i32> {
    match (run_result, joined, system_joined, gpu_joined) {
        (Ok(code), Ok(()), Ok(()), Ok(())) => Ok(code),
        (Ok(_), Err(err), Ok(()), Ok(())) => {
            Err(anyhow::anyhow!("collector task failed during shutdown: {err}"))
        }
        (Ok(_), Ok(()), Err(err), Ok(())) => {
            Err(anyhow::anyhow!("system monitor failed during shutdown: {err}"))
        }
        (Ok(_), Ok(()), Ok(()), Err(err)) => {
            Err(anyhow::anyhow!("gpu monitor failed during shutdown: {err}"))
        }
        (Ok(_), Err(err), Err(_), Ok(())) | (Ok(_), Err(err), Ok(()), Err(_)) => {
            Err(anyhow::anyhow!("collector task failed during shutdown: {err}"))
        }
        (Ok(_), Ok(()), Err(sys_err), Err(gpu_err)) => Err(anyhow::anyhow!(
            "system monitor failed during shutdown: {sys_err}; gpu monitor failed: {gpu_err}"
        )),
        (Ok(_), Err(err), Err(sys_err), Err(gpu_err)) => Err(anyhow::anyhow!(
            "collector task failed during shutdown: {err}; system monitor: {sys_err}; gpu monitor: {gpu_err}"
        )),
        (Err(err), _, _, _) => Err(err),
    }
}

/// The event loop body. Returns the exit code, or an error if a command
/// send or a draw failed. The caller owns collector cleanup.
async fn run_event_loop(
    state: &mut AppState,
    events_rx: &mut UnboundedReceiver<AppEvent>,
    commands_tx: &UnboundedSender<CollectorCommand>,
    draw: &mut impl FnMut(&AppState) -> std::io::Result<()>,
) -> anyhow::Result<i32> {
    let mut tick = tokio::time::interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        // Drain all immediately-available events before drawing, so the UI
        // catches up after slow collector responses without queuing input.
        while let Ok(event) = events_rx.try_recv() {
            handle(state, event);
        }

        if !state.should_quit {
            tokio::select! {
                event = events_rx.recv() => match event {
                    Some(e) => handle(state, e),
                    // All senders dropped (collector + input reader gone):
                    // unrecoverable channel failure; exit cleanly.
                    None => return Ok(0),
                },
                _ = tick.tick() => state.on_tick(),
            }
        }

        if state.take_reconnect_requested() {
            commands_tx.send(CollectorCommand::Reconnect)?;
        }

        if state.should_quit {
            return Ok(0);
        }

        draw(state)?;
    }
}

/// Apply one event to the state.
fn handle(state: &mut AppState, event: AppEvent) {
    match event {
        AppEvent::Tick => state.on_tick(),
        AppEvent::BackendSnapshot(snap) => state.apply_snapshot(*snap),
        AppEvent::BackendCapabilities(caps) => state.apply_capabilities(caps),
        AppEvent::BackendError(summary) => state.apply_error(summary),
        AppEvent::Input(action) => state.handle_input(action),
        AppEvent::Resize(width, height) => state.terminal_size = (width, height),
        AppEvent::SystemSample(snap) => state.apply_system(snap),
        AppEvent::SystemUnavailable => state.apply_system_unavailable(),
        AppEvent::GpuSample(monitor) => state.apply_gpu(monitor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::event::{BackendErrorSummary, InputAction};
    use crate::domain::{BackendSnapshot, ConnectionState};

    #[tokio::test]
    async fn quit_event_stops_the_collector_and_returns_zero() {
        let config = Config {
            endpoint: "http://127.0.0.1:1".to_string(), // dead port; no traffic
            request_timeout_ms: 100,
            refresh_interval_ms: 100,
            ..Default::default()
        };
        let (events_tx, events_rx) = unbounded_channel();
        let quit_tx = events_tx.clone();
        let handle = tokio::spawn(run(config, events_tx, events_rx, (80, 20), |_| Ok(())));
        // A second sender stands in for the input reader thread: send quit.
        quit_tx.send(AppEvent::Input(InputAction::Quit)).expect("send quit");
        drop(quit_tx);
        let code = tokio::time::timeout(std::time::Duration::from_secs(10), handle)
            .await
            .expect("runtime exits within timeout")
            .expect("runtime task does not panic");
        assert_eq!(code.expect("runtime returns Ok"), 0);
    }

    #[tokio::test]
    async fn draw_failure_returns_the_error_and_exits() {
        let config = Config {
            endpoint: "http://127.0.0.1:1".to_string(), // dead port; no traffic
            request_timeout_ms: 100,
            refresh_interval_ms: 100,
            ..Default::default()
        };
        let (events_tx, events_rx) = unbounded_channel();
        // The draw fails on the first call. The runtime must surface that
        // error (not mask it with a cleanup error) and terminate promptly
        // (the collector is stopped and joined, so nothing hangs).
        let handle = tokio::spawn(run(config, events_tx, events_rx, (80, 20), |_| {
            Err(std::io::Error::other("draw failed"))
        }));
        let result = tokio::time::timeout(std::time::Duration::from_secs(10), handle)
            .await
            .expect("runtime exits within timeout")
            .expect("runtime task does not panic");
        let err = result.expect_err("draw error propagates");
        assert!(err.to_string().contains("draw failed"));
    }

    /// A real `JoinError` (from a panicking task) for the join-failure cases.
    async fn panicking_join_error() -> tokio::task::JoinError {
        tokio::spawn(async { panic!("collector boom") }).await.unwrap_err()
    }

    #[test]
    fn finish_run_returns_clean_exit_on_success() {
        assert_eq!(finish_run(Ok(0), Ok(()), Ok(()), Ok(())).expect("clean exit"), 0);
    }

    #[tokio::test]
    async fn finish_run_reports_collector_panic_on_clean_exit() {
        let err = finish_run(Ok(0), Err(panicking_join_error().await), Ok(()), Ok(()))
            .expect_err("a collector panic must not be silently ignored");
        assert!(err.to_string().contains("collector task failed"));
    }

    #[tokio::test]
    async fn finish_run_reports_system_monitor_panic_on_clean_exit() {
        let err = finish_run(Ok(0), Ok(()), Err(panicking_join_error().await), Ok(()))
            .expect_err("a monitor panic must not be silently ignored");
        assert!(err.to_string().contains("system monitor failed"));
    }

    #[tokio::test]
    async fn finish_run_reports_gpu_monitor_panic_on_clean_exit() {
        let err = finish_run(Ok(0), Ok(()), Ok(()), Err(panicking_join_error().await))
            .expect_err("a monitor panic must not be silently ignored");
        assert!(err.to_string().contains("gpu monitor failed"));
    }

    #[tokio::test]
    async fn finish_run_keeps_loop_error_over_cleanup_error() {
        let loop_err = anyhow::anyhow!("draw failed");
        let err = finish_run(
            Err(loop_err),
            Err(panicking_join_error().await),
            Err(panicking_join_error().await),
            Err(panicking_join_error().await),
        )
        .expect_err("has error");
        assert!(
            err.to_string().contains("draw failed"),
            "the loop error must not be masked by the join error: {err}"
        );
    }

    #[tokio::test]
    async fn finish_run_keeps_loop_error_over_clean_join() {
        let loop_err = anyhow::anyhow!("draw failed");
        let err = finish_run(Err(loop_err), Ok(()), Ok(()), Ok(())).expect_err("has error");
        assert!(err.to_string().contains("draw failed"));
    }

    #[test]
    fn handle_maps_each_event_to_state() {
        let config = Config::default();
        let mut state = AppState::new(&config);

        handle(&mut state, AppEvent::Resize(100, 30));
        assert_eq!(state.terminal_size, (100, 30));

        handle(
            &mut state,
            AppEvent::BackendSnapshot(Box::new(BackendSnapshot {
                connection: ConnectionState::Connected,
                ..Default::default()
            })),
        );
        assert!(state.latest.is_some());

        handle(&mut state, AppEvent::BackendError(BackendErrorSummary::new("boom")));
        assert_eq!(state.connection_message.as_deref(), Some("boom"));

        handle(&mut state, AppEvent::Input(InputAction::ToggleHelp));
        assert!(state.show_help);

        handle(
            &mut state,
            AppEvent::GpuSample(crate::domain::GpuMonitor {
                status: crate::domain::GpuMonitorStatus::Unavailable,
                gpus: Vec::new(),
            }),
        );
        assert!(state.gpu.is_some());

        handle(&mut state, AppEvent::Tick);
    }
}
