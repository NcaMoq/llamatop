//! The `doctor` command: verify the environment and server connectivity.
//!
//! Each check is independent and degrades to a warning rather than aborting,
//! so a user can see exactly what works and what is missing. The report is
//! rendered with the same Unicode/ASCII symbol set as the rest of the CLI.
//!
//! No secrets (API key, prompt, completion) are ever printed or logged.

use crate::backend::llamacpp::client::LlamaCppClient;
use crate::backend::llamacpp::health::parse_health;
use crate::config::Config;
use crate::display::Symbols;
use crate::domain::ServerState;

/// Severity of a single check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// Works as expected.
    Ok,
    /// Feature unavailable; the app can still start.
    Warning,
    /// Something is wrong that the user should fix.
    Error,
}

/// One line of the doctor report.
#[derive(Debug, Clone)]
pub struct Check {
    pub status: CheckStatus,
    pub label: String,
}

impl Check {
    fn ok(label: impl Into<String>) -> Self {
        Self { status: CheckStatus::Ok, label: label.into() }
    }

    fn warning(label: impl Into<String>) -> Self {
        Self { status: CheckStatus::Warning, label: label.into() }
    }

    fn error(label: impl Into<String>) -> Self {
        Self { status: CheckStatus::Error, label: label.into() }
    }
}

/// The full doctor report.
pub struct Report {
    pub checks: Vec<Check>,
}

impl Report {
    pub fn has_errors(&self) -> bool {
        self.checks.iter().any(|c| c.status == CheckStatus::Error)
    }

    pub fn has_warnings(&self) -> bool {
        self.checks.iter().any(|c| c.status == CheckStatus::Warning)
    }

    /// Exit code: 0 when the app can start, 1 when a blocking error was found.
    pub fn exit_code(&self) -> i32 {
        if self.has_errors() {
            1
        } else {
            0
        }
    }

    /// Render the report to stdout.
    pub fn print(&self, ascii: bool) {
        let symbols = Symbols::new(ascii);
        let logo =
            if ascii { String::from("llamatop ") } else { String::from("\u{2601} llamatop ") };
        println!("{logo}{}", env!("CARGO_PKG_VERSION"));
        println!("{}", symbols.separator(28));
        println!();
        println!("Checking llama.cpp server...");
        println!();

        for check in &self.checks {
            let mark = match check.status {
                CheckStatus::Ok => symbols.success(),
                CheckStatus::Warning => symbols.warning(),
                CheckStatus::Error => symbols.error(),
            };
            println!("  {mark} {}", check.label);
        }

        println!();
        if self.has_errors() {
            println!("{} Fix the issues above before monitoring.", symbols.error());
        } else if self.has_warnings() {
            println!("llamatop can start, but some metrics will be hidden.");
        } else {
            println!("Ready to monitor.");
        }
    }
}

/// Runs the doctor checks against a loaded, validated config.
pub struct Doctor {
    config: Config,
}

impl Doctor {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Run all checks from a synchronous context (e.g. tests). The main
    /// binary already runs on a tokio runtime and calls [`Self::run_async`]
    /// instead, because a runtime cannot be started from within a runtime.
    pub fn run(&self) -> anyhow::Result<Report> {
        if tokio::runtime::Handle::try_current().is_ok() {
            anyhow::bail!("doctor checks must run via run_async inside an async runtime");
        }
        let rt = tokio::runtime::Runtime::new()?;
        let checks = rt.block_on(self.run_checks());
        Ok(Report { checks })
    }

    /// Run all checks on the current tokio runtime.
    pub async fn run_async(&self) -> Report {
        Report { checks: self.run_checks().await }
    }

    async fn run_checks(&self) -> Vec<Check> {
        let mut checks = Vec::new();

        // Config + endpoint were already validated by Config::load, so they
        // are guaranteed good by the time we get here.
        checks.push(Check::ok("Configuration file readable"));
        checks.push(Check::ok(format!(
            "Endpoint URL valid ({})",
            crate::endpoint::redact(&self.config.endpoint)
        )));

        // Build the client; a malformed endpoint cannot reach this point.
        let client = match LlamaCppClient::new(
            &self.config.endpoint,
            self.config.request_timeout(),
            self.config.api_key().as_deref(),
        ) {
            Ok(c) => c,
            Err(e) => {
                checks.push(Check::error(format!("Endpoint unusable: {e}")));
                return checks;
            }
        };

        // Reachability + /health.
        match client.get_raw("health").await {
            Ok((status, body, _)) => {
                checks.push(Check::ok("Server reachable"));
                let outcome = parse_health(status, &body);
                match outcome.server {
                    ServerState::Ready => checks.push(Check::ok("Model ready")),
                    ServerState::Loading => {
                        checks.push(Check::warning("Model still loading"));
                    }
                    ServerState::Sleeping => checks.push(Check::warning("Server is sleeping")),
                    _ => {
                        let detail = outcome.detail.unwrap_or_else(|| format!("HTTP {status}"));
                        checks.push(Check::warning(format!("Model not ready ({detail})")));
                    }
                }
            }
            Err(e) => {
                checks.push(Check::error(format!("Server unreachable: {e}")));
                // Remaining endpoint checks are meaningless without a server.
                checks.push(Check::warning("Slots endpoint unavailable (server down)"));
                checks.push(Check::warning("Metrics endpoint unavailable (server down)"));
                checks.push(Check::warning("Props endpoint unavailable (server down)"));
                return checks;
            }
        }

        // /slots (also used for the auth check).
        let slots_status = match client.get_raw("slots").await {
            Ok((status, _, _)) => Some(status),
            Err(_) => None,
        };
        match slots_status {
            Some(200) => checks.push(Check::ok("Slots endpoint available")),
            Some(s) => {
                checks.push(Check::warning(format!("Slots endpoint unavailable (HTTP {s})")));
            }
            None => checks.push(Check::warning("Slots endpoint unavailable (request failed)")),
        }

        // /metrics.
        match client.get_raw("metrics").await {
            Ok((200, _, _)) => {
                checks.push(Check::ok("Metrics endpoint available"));
            }
            Ok((status, _, _)) => {
                checks
                    .push(Check::warning(format!("Metrics endpoint unavailable (HTTP {status})")));
            }
            Err(_) => checks.push(Check::warning("Metrics endpoint unavailable (request failed)")),
        }

        // /props.
        match client.get_raw("props").await {
            Ok((200, _, _)) => {
                checks.push(Check::ok("Props endpoint available"));
            }
            Ok((status, _, _)) => {
                checks.push(Check::warning(format!("Props endpoint unavailable (HTTP {status})")));
            }
            Err(_) => checks.push(Check::warning("Props endpoint unavailable (request failed)")),
        }

        // Authentication (only meaningful when an API key is configured or
        // the server requires one). /health is public, so we probe /slots.
        let api_key = self.config.api_key();
        match (&api_key, slots_status) {
            (Some(_), Some(401)) => {
                checks.push(Check::error("Authentication failed (API key rejected)"));
            }
            (Some(_), Some(_)) => {
                checks.push(Check::ok("Authentication succeeded"));
            }
            (None, Some(401)) => {
                checks.push(Check::warning(
                    "Server requires an API key; set the variable named in [authentication] api_key_env",
                ));
            }
            (Some(_), None) => {
                checks.push(Check::warning(
                    "Authentication check skipped (slots endpoint unavailable)",
                ));
            }
            (None, _) => checks.push(Check::ok("Authentication not required")),
        }

        // GPU: NVML initialization + device enumeration.
        match self.gpu_checks() {
            Ok(gpu_checks) => checks.extend(gpu_checks),
            Err(_) => {
                // gpu_checks returns Err only if NVML init itself failed.
                checks.push(Check::warning("NVIDIA NVML unavailable"));
            }
        }

        // Terminal size. When output is not a TTY there is no interactive
        // terminal to size, so this is not a concern.
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            match crossterm::terminal::size() {
                Ok((w, h)) => checks.push(Check::ok(format!("Terminal size {w}x{h}"))),
                Err(_) => checks.push(Check::warning("Terminal size unavailable")),
            }
        } else {
            checks.push(Check::ok("Terminal size not checked (output is not a TTY)"));
        }

        // Unicode display heuristic.
        checks.push(self.unicode_check());

        checks
    }

    /// Probe NVML for NVIDIA GPUs. Returns `Ok(vec)` when NVML initialized
    /// (possibly with zero GPUs), or `Err` when NVML could not be loaded.
    fn gpu_checks(&self) -> Result<Vec<Check>, ()> {
        use nvml_wrapper::Nvml;

        let nvml = Nvml::init().map_err(|_| ())?;
        let mut out = Vec::new();

        match nvml.device_count() {
            Ok(0) => {
                out.push(Check::ok("NVIDIA NVML available"));
                out.push(Check::warning("No NVIDIA GPU detected"));
            }
            Ok(count) => {
                out.push(Check::ok("NVIDIA NVML available"));
                for i in 0..count {
                    match nvml.device_by_index(i).and_then(|d| d.name()) {
                        Ok(name) => out.push(Check::ok(format!("GPU {i}: {name}"))),
                        Err(_) => out.push(Check::warning(format!("GPU {i}: name unavailable"))),
                    }
                }
            }
            Err(_) => {
                out.push(Check::ok("NVIDIA NVML available"));
                out.push(Check::warning("GPU enumeration failed"));
            }
        }

        Ok(out)
    }

    fn unicode_check(&self) -> Check {
        use std::io::IsTerminal;
        if std::io::stdout().is_terminal() {
            Check::ok("Unicode display available")
        } else {
            // Output is piped/redirected: nothing is rendered interactively,
            // so the display capability is not a concern.
            Check::ok("Unicode display not checked (output is not a TTY)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_code_reflects_errors() {
        let ok = Report { checks: vec![Check::ok("a"), Check::warning("b")] };
        assert_eq!(ok.exit_code(), 0);
        assert!(!ok.has_errors());
        assert!(ok.has_warnings());

        let bad = Report { checks: vec![Check::ok("a"), Check::error("b")] };
        assert_eq!(bad.exit_code(), 1);
        assert!(bad.has_errors());
    }

    #[test]
    fn status_marks_are_distinct() {
        let s = Symbols::new(false);
        assert_eq!(s.success(), "✓");
        assert_eq!(s.warning(), "▲");
        assert_eq!(s.error(), "✘");
    }
}
