# llamatop

A Windows-native terminal monitor for [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server`. It observes server state, per-slot workload phase, and
throughput, without ever requesting, storing, or displaying prompt or
completion text, and without ever displaying or logging API keys. It also
monitors host CPU/RAM, the `llama-server` process, and (optionally) NVIDIA
GPUs via NVML.

## Status

v0.1 release candidate. This build provides the monitoring core, the
non-interactive commands, and a full interactive TUI:

- `llamatop doctor` — environment and server connectivity check
- `llamatop snapshot` — one-shot capture (human-readable or pure JSON)
- `llamatop` (no subcommand) — interactive TUI:
  - a **header** panel (connection, backend, model, server state, workload
    phase, active/queued requests, update age),
  - an **inference** panel (prompt/generation throughput, active/queued,
    context size, slot count, speculative acceptance rate),
  - a **slot** table (stable ID order, slot selection, vertical scrolling,
    and responsive columns: wide/standard/compact by terminal width),
  - a **history** panel (time-series sparklines/bars for prompt, generation,
    active, and queued; bounded by `history_seconds`),
  - an **event log** panel (connection, server-state, phase, capability,
    reconnect, and pause events; bounded, with repeat collapsing), and
  - a **Resources** panel (host CPU/RAM, the `llama-server` process when it
    can be identified, and one row per NVIDIA GPU).

TUI controls: `q` quit, `r` reconnect, `p` pause/resume, `l` toggle the event
log, `c` clear the event log (or the history when it is hidden),
PageUp/PageDown/Home/End to scroll the event log, and `↑`/`↓` (or `j`/`k`) to
move the slot selection. `?` opens a help modal (Esc closes it); `q` does not
quit while the modal is open. The slot keys are advertised in the footer only
when the `/slots` endpoint is available and at least one slot is reported;
"Slots unavailable" (endpoint absent) and "No slots reported" (zero slots)
are distinct views. The terminal must be at least 80x20; smaller windows show
a "too small" notice. Missing values are rendered as a placeholder, never as
0. `--ascii` forces an ASCII-only rendering.

The Resources panel has the lowest layout priority, so it appears only when
the terminal is tall enough (for example, roughly 100x31 or larger) and the
full history still fits; at 80x20 and 100x30 it is hidden so the slot table
and history keep their rows. A per-GPU row shows utilization, VRAM
used/total, temperature, and power — it never claims that a GPU belongs to
the `llama-server` process. The panel does not claim that a local process
belongs to the configured endpoint unless the association is verified: a
single matching local process is shown as a candidate labeled "endpoint not
verified", several matches are counted without naming one, and a non-local
endpoint (e.g. `http://192.168.1.50:8080`) reports that local process
association is unavailable. The slot detail view is not yet implemented.

## Requirements

- Windows (x86_64), Rust stable (MSVC toolchain) for building
- A running `llama-server` (any recent llama.cpp build)
- (Optional) an NVIDIA GPU with a current driver for GPU monitoring

No Python, Node.js, Docker, or WSL is required. GPU monitoring uses NVIDIA
NVML when the driver is present and degrades to a warning (never a crash or
a startup failure) when NVML or a GPU is unavailable. `--no-gpu` disables GPU
monitoring and `--no-system` disables host + process monitoring.

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
      --endpoint <URL>   Endpoint URL of the llama-server to monitor (default: http://127.0.0.1:8080)
      --ascii            Use ASCII-only characters in output (no Unicode symbols)
      --no-gpu           Disable GPU monitoring
      --no-system        Disable host + llama-server process monitoring
      --refresh-ms <MS>  Snapshot refresh interval in milliseconds (minimum: 100)
      --verbose          Increase log verbosity (info level)
      --debug            Enable debug logging (writes full details to the log file)
  -h, --help             Print help
  -V, --version          Print version
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
| `show_system` | `true` | Host CPU/RAM and `llama-server` process monitoring |
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
`generation_tokens_per_second`, `slots[]`. The `gpu[]` field is part of the
schema but is always omitted by `snapshot --json`: GPU metrics are sampled
live by the TUI's NVML monitor and are never fetched from the server.
Throughput falls back to the server-reported cumulative average when no local
delta is available.

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

### Polling

Each endpoint is fetched on its own interval (defaults: `/health` 1000 ms,
`/slots` 1000 ms, `/metrics` 500 ms, `/props` 2000 ms; `refresh_interval_ms`
controls the render cycle and does not fetch). When `/slots` answers but the
body cannot be parsed, the next `/slots` fetch is delayed by at least 5 s
until a response parses again — polling an unparseable endpoint on the
normal interval only adds log noise on the server side. A manual reconnect
(`r`) bypasses the delay and fetches immediately.

## Security and privacy

- Prompts and completions are never fetched, logged, or displayed.
- API keys are never displayed, logged, or written to disk.
- Endpoint URLs that contain a username or password (e.g.
  `http://user:pass@host:8080`) are rejected at configuration time, as are
  query strings and fragments. The API key is only ever read from the
  environment variable named by `authentication.api_key_env`. As defense in
  depth, every display, error, Debug, and JSON surface redacts the endpoint
  (userinfo, query, and fragment stripped).
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
