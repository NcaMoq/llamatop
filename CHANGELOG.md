# Changelog

All notable changes to llamatop are documented here.

## [0.1.0] - Unreleased

### Added

- Interactive TUI (Rust, ratatui + crossterm) for monitoring a running
  `llama-server`:
  - Header panel: connection state, backend, model name, server state,
    workload phase, active/queued requests, update age.
  - Inference panel: prompt/generation throughput (local delta with
    server-reported average fallback), active/queued, context size, slot
    count, speculative acceptance rate.
  - Slot table: stable ID order, slot selection, vertical scrolling, and
    responsive columns (wide/standard/compact by terminal width).
  - History panel: time-series sparklines/bars for prompt, generation,
    active, and queued; window bounded by `history_seconds`.
  - Event log panel: connection, server-state, phase, capability, reconnect,
    and pause events; bounded with repeat collapsing; toggle with `l`, clear
    with `c`, scroll with PageUp/PageDown/Home/End.
  - Resources panel (lowest layout priority): host CPU/RAM, the
    `llama-server` process when it can be identified, and one row per
    NVIDIA GPU.
- TUI controls: `q` quit, `r` reconnect, `p` pause/resume, `l`/`c` for the
  event log, `↑`/`↓` or `j`/`k` slot selection, `?` help modal (Esc closes;
  `q` does not quit while open).
- Host CPU/RAM and `llama-server` process monitoring (sysinfo), with
  endpoint-association checks that never guess: `exact`,
  `multiple_candidates`, or `none_found`.
- Optional NVIDIA GPU monitoring via NVML: per-device utilization, VRAM
  used/total, temperature, power/limit, and clocks; degrades to a warning
  when NVML or a GPU is unavailable (never a crash or startup failure);
  disabled with `--no-gpu` or `gpu.backend = "none"`.
- `--no-system` flag and `show_system` config to disable host/process
  monitoring.
- Non-interactive commands:
  - `llamatop doctor` — environment and server connectivity check.
  - `llamatop snapshot` — one-shot capture, human-readable or pure JSON
    (`--json`, `schema_version` 1, `None` values omitted).
- Configuration: `%APPDATA%\llamatop\config.toml`, environment variables
  (`LLAMATOP_ENDPOINT`, `LLAMATOP_CONFIG_PATH`), and CLI flags, in that
  precedence order below the CLI.
- Daily-rolling log file (`llamatop.log`, 7-day retention) with
  `--verbose`/`--debug` levels.
- Privacy guarantees: prompt/completion text is never requested, stored, or
  displayed; API keys are never displayed, logged, or written to disk;
  transport errors are redacted.
