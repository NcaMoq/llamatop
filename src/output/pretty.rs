//! Human-readable snapshot output for the `snapshot` command.
//!
//! States are never conveyed by color alone; each value is a plain token.
//! Missing values render as a placeholder (`—` in Unicode, `-` in ASCII),
//! never as a fabricated 0.

use crate::display::Symbols;
use crate::domain::ConnectionState;
use crate::snapshot::Snapshot;

/// Placeholder for a value that was not reported by the server.
fn placeholder(ascii: bool) -> &'static str {
    if ascii {
        "-"
    } else {
        "—"
    }
}

/// Format a single label/value line, left-padding the label to the column width.
fn line(label: &str, value: &str) -> String {
    format!("{: <14}{}", label, value)
}

fn format_rate(value: Option<f64>, ascii: bool) -> String {
    match value {
        Some(v) => format!("{v:.1} tok/s"),
        None => placeholder(ascii).to_string(),
    }
}

fn format_bytes_gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Render the snapshot into a string. Kept separate from stdout so it is
/// testable without touching the terminal.
pub fn render(snap: &Snapshot, ascii: bool) -> String {
    let symbols = Symbols::new(ascii);
    let s = &snap.snapshot;
    let mut out = String::new();

    out.push_str(&line("BACKEND", &snap.backend));
    out.push('\n');
    out.push_str(&line("CONNECTION", s.connection.as_str()));
    out.push('\n');

    if let Some(model) = &s.model_name {
        out.push_str(&line("MODEL", model));
        out.push('\n');
    }

    out.push_str(&line("SERVER", s.server.as_str()));
    out.push('\n');
    out.push_str(&line("PHASE", s.workload_phase.display()));
    out.push('\n');

    let active =
        s.active_requests.map(|v| v.to_string()).unwrap_or_else(|| placeholder(ascii).to_string());
    out.push_str(&line("ACTIVE", &active));
    out.push('\n');
    let queued =
        s.queued_requests.map(|v| v.to_string()).unwrap_or_else(|| placeholder(ascii).to_string());
    out.push_str(&line("QUEUED", &queued));
    out.push('\n');

    let prompt = s.prompt_tokens_per_second.or(s.prompt_tokens_per_second_reported);
    let gen = s.generation_tokens_per_second.or(s.generation_tokens_per_second_reported);
    out.push_str(&line("PROMPT", &format_rate(prompt, ascii)));
    out.push('\n');
    out.push_str(&line("GENERATION", &format_rate(gen, ascii)));
    out.push('\n');

    // GPU / VRAM lines only when a GPU snapshot is present.
    if let Some(gpu) = s.gpu.first() {
        let util = match gpu.utilization_percent {
            Some(u) => format!("{u}%"),
            None => placeholder(ascii).to_string(),
        };
        out.push_str(&line("GPU", &util));
        out.push('\n');
        if let (Some(used), Some(total)) = (gpu.memory_used_bytes, gpu.memory_total_bytes) {
            let vram = format!("{}/{}", format_bytes_gib(used), format_bytes_gib(total));
            out.push_str(&line("VRAM", &vram));
            out.push('\n');
        }
    }

    // A short, secret-free note when the server could not be reached.
    if s.connection != ConnectionState::Connected {
        let reason = s.error.as_deref().unwrap_or("unknown");
        out.push('\n');
        out.push_str(&format!("{} DISCONNECTED: {reason}", symbols.error()));
        out.push('\n');
    }

    out
}

/// Write the pretty snapshot to stdout.
pub fn print(snap: &Snapshot, ascii: bool) -> anyhow::Result<()> {
    let rendered = render(snap, ascii);
    use std::io::Write;
    let mut out = std::io::stdout().lock();
    out.write_all(rendered.as_bytes())?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendCapabilities;
    use crate::domain::{BackendSnapshot, ServerState, WorkloadPhase};

    fn connected() -> Snapshot {
        let s = BackendSnapshot {
            connection: ConnectionState::Connected,
            server: ServerState::Ready,
            workload_phase: WorkloadPhase::Decode,
            active_requests: Some(1),
            queued_requests: Some(0),
            model_name: Some("Qwen".into()),
            prompt_tokens_per_second: Some(1832.4),
            generation_tokens_per_second: Some(48.6),
            ..Default::default()
        };
        Snapshot {
            backend: "llama.cpp".into(),
            endpoint: "http://127.0.0.1:8080/".into(),
            snapshot: s,
            capabilities: BackendCapabilities::default(),
        }
    }

    #[test]
    fn rate_is_formatted_to_one_decimal() {
        assert_eq!(format_rate(Some(1832.4), false), "1832.4 tok/s");
        assert_eq!(format_rate(Some(0.0), false), "0.0 tok/s");
        assert_eq!(format_rate(None, false), "—");
        assert_eq!(format_rate(None, true), "-");
    }

    #[test]
    fn placeholder_switches_with_ascii() {
        assert_eq!(placeholder(true), "-");
        assert_eq!(placeholder(false), "—");
    }

    #[test]
    fn renders_core_fields() {
        let rendered = render(&connected(), false);
        assert!(rendered.contains("BACKEND"));
        assert!(rendered.contains("llama.cpp"));
        assert!(rendered.contains("CONNECTED"));
        assert!(rendered.contains("READY"));
        assert!(rendered.contains("DECODE"));
        assert!(rendered.contains("1832.4 tok/s"));
        assert!(rendered.contains("48.6 tok/s"));
        assert!(rendered.contains("Qwen"));
    }

    #[test]
    fn missing_values_use_placeholder_not_zero() {
        let mut snap = connected();
        snap.snapshot.active_requests = None;
        snap.snapshot.queued_requests = None;
        snap.snapshot.prompt_tokens_per_second = None;
        snap.snapshot.prompt_tokens_per_second_reported = None;
        snap.snapshot.generation_tokens_per_second = None;
        snap.snapshot.generation_tokens_per_second_reported = None;
        let rendered = render(&snap, true);
        let line = |label: &str| -> String {
            rendered
                .lines()
                .find(|l| l.trim_start().starts_with(label))
                .unwrap_or_default()
                .to_string()
        };
        assert!(line("ACTIVE").ends_with("-"), "active line: {:?}", line("ACTIVE"));
        assert!(line("QUEUED").ends_with("-"), "queued line: {:?}", line("QUEUED"));
        assert!(line("PROMPT").ends_with("-"), "prompt line: {:?}", line("PROMPT"));
    }

    #[test]
    fn disconnected_snapshot_is_marked() {
        let mut s = connected();
        s.snapshot.connection = ConnectionState::Disconnected;
        s.snapshot.error = Some("connection refused".into());
        let rendered = render(&s, false);
        assert!(rendered.contains("✘ DISCONNECTED: connection refused"));
    }

    #[test]
    fn ascii_disconnected_uses_err_symbol() {
        let mut s = connected();
        s.snapshot.connection = ConnectionState::Disconnected;
        s.snapshot.error = Some("connection refused".into());
        let rendered = render(&s, true);
        assert!(rendered.contains("[ERR] DISCONNECTED: connection refused"));
    }
}
