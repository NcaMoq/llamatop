# llamatop

A Windows-native terminal monitor for [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server`. It observes server state, per-slot workload phase, throughput,
and (when available) NVIDIA GPU metrics — without ever requesting, storing, or
displaying prompt or completion text, and without ever displaying or logging
API keys.

## Status

This build provides the monitoring core, the non-interactive commands, and a
basic interactive TUI:

- `llamatop doctor` — environment and server connectivity check
- `llamatop snapshot` — one-shot capture (human-readable or pure JSON)
- `llamatop` (no subcommand) — interactive TUI with a header panel
  (connection, backend, model, server state, workload phase, active/queued
  requests, update age) and an inference panel (prompt/generation throughput,
  active/queued, context size, slot count, speculative acceptance rate)

TUI controls: `q` quit, `r` reconnect, `p` pause/resume. The terminal must be
at least 80x20; smaller windows show a "too small" notice. Missing values are
rendered as a placeholder, never as 0. `--ascii` forces an ASCII-only
rendering. Slot tables, history graphs, the event log, GPU/system monitoring,
and the help modal are not yet implemented.

## Requirements

- Windows (x86_64), Rust stable (MSVC toolchain) for building
- A running `llama-server` (any recent llama.cpp build)

No Python, Node.js, Docker, or WSL is required. GPU monitoring uses NVIDIA NVML
when the driver is present; it degrades to a warning (never a crash) when NVML
or a GPU is unavailable.

## Building

```sh
cargo build --release
# executable: target\release\llamatop.exe
```

## Usage

```
llamatop [OPTIONS] [COMMAND]

Commands:
  doctor    Check the environment and server connectivity
  snapshot  Capture a single snapshot and exit

Options:
      --endpoint <URL>    llama-server URL (default: http://127.0.0.1:8080)
      --ascii             ASCII-only output (no Unicode symbols)
      --no-gpu            Disable GPU monitoring
      --refresh-ms <MS>   Snapshot refresh interval in milliseconds (minimum 100)
      --verbose           Increase log verbosity (info level)
      --debug             Debug logging (details go to the log file)
  -h, --help              Help
  -V, --version           Version
```

Examples:

```sh
llamatop doctor
llamatop snapshot
llamatop snapshot --json
llamatop snapshot --endpoint http://10.0.0.5:8080
```

`snapshot --json` writes only a versioned JSON document to stdout (diagnostics
go to stderr), so it is safe to pipe into other tools.

### Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | Success (connected; doctor found no blocking errors) |
| 1 | General failure (or doctor found blocking errors) |
| 2 | Invalid configuration |
| 3 | Server unreachable (snapshot) |

## Configuration

Configuration file location (highest precedence first):
CLI arguments > environment variables > `%APPDATA%\llamatop\config.toml` >
built-in defaults.

| Setting | Default | Notes |
| ------- | ------- | ----- |
| `endpoint` | `http://127.0.0.1:8080` | llama-server URL |
| `refresh_interval_ms` | `500` | Minimum 100 |
| `ascii` | `false` | Force ASCII output |
| `show_gpu` | `true` | GPU monitoring |
| `history_seconds` | `120` | 10..=3600 |
| `authentication.api_key_env` | `LLAMATOP_API_KEY` | Name of the env var holding the API key |
| `gpu.backend` | `auto` | `auto`, `nvml`, or `none` |

Environment variables: `LLAMATOP_ENDPOINT`, `LLAMATOP_CONFIG_PATH`, and the
variable named by `authentication.api_key_env`. The API key itself is only ever
read from that environment variable — it is never taken from the command line
(visible in the process list) or stored in the config file.

## JSON schema (`snapshot --json`)

`schema_version` is `1`. `None` values are omitted, so a missing metric is
distinguishable from `0`. Example:

```json
{
  "schema_version": 1,
  "timestamp": "2026-08-19T12:46:52Z",
  "backend": "llama.cpp",
  "endpoint": "http://127.0.0.1:8080/",
  "connection": { "state": "connected" },
  "server": {
    "state": "ready",
    "workload_phase": "idle",
    "workload_confidence": "high",
    "model_name": "qwen3.8-27b",
    "build_info": "b10488-9d77fa172"
  },
  "throughput": {}
}
```

Fields present when the server reports them: `active_requests`,
`queued_requests`, `context_max_tokens`, `prompt_tokens_per_second`,
`generation_tokens_per_second`, `gpu[]`, `slots[]`. Throughput falls back to
the server-reported cumulative average when no local delta is available.

## What it monitors (and does not)

llamatop reads only monitoring endpoints: `/health`, `/slots`, `/metrics`,
`/props`. It never requests `/completion` or `/chat/completions`. Slot
`prompt`/`generated` fields are not retained (they only appear in llama.cpp
debug builds anyway).

- `/metrics` is disabled by default in llama.cpp (501 without `--metrics`);
  llamatop tolerates this and hides the affected metrics.
- `--no-slots` servers are tolerated; slot-dependent views degrade gracefully.
- While the model is loading, llama.cpp returns 503 from all endpoints;
  llamatop reports `loading` rather than "disconnected".

### Workload phase detection

Phases are inferred from counter deltas, never guessed:

- `decode` — per-slot decoded-token growth (high confidence)
- `prefill_likely*` — prompt-token growth without decode evidence (estimated)
- `processing_unknown?` — slot busy but no counter evidence (unknown)
- `idle` — no active processing (stable over two observations)
- `loading` / `sleeping` — authoritative server states (applied immediately)

`*` / `?` mark estimated or unknown values in the display. Queued requests are
reported separately; a queue is not a phase.

## Security and privacy

- Prompts and completions are never fetched, logged, or displayed.
- API keys are never displayed, logged, or written to disk.
- Transport errors shown in output are short, redacted descriptions.

## Development

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release --target x86_64-pc-windows-msvc
```

Logs are written to the platform data directory (`llamatop.log`, daily rolling,
7-day retention) when available; `--verbose`/`--debug` raise the level.
