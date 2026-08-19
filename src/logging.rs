//! File logging with daily rotation and a bounded retention window.
//!
//! Logs never contain API keys, authorization headers, prompts, completions,
//! or full HTTP response bodies. Callers are responsible for passing only
//! non-sensitive fields.

use std::path::Path;
use std::time::Duration;

use tracing::Level;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Number of log files to keep (current day plus this many previous days).
const RETENTION_DAYS: u64 = 7;

/// Keeps the process alive until the log layer can be dropped. The global
/// subscriber owns the file appender; this struct only documents the
/// lifetime so `main` can hold logging for the whole run.
pub struct Logging;

impl Logging {
    /// Initialize file logging into `dir` with the given maximum level.
    ///
    /// `dir` is created if missing. If the directory cannot be created (for
    /// example on a read-only system), the error is returned so the caller
    /// can warn on stderr and run without file logging.
    pub fn init(level: Level, dir: &Path) -> Result<Self, std::io::Error> {
        std::fs::create_dir_all(dir)?;
        let file_appender = tracing_appender::rolling::daily(dir, "llamatop.log");

        Self::cleanup_old_files(dir)?;

        // `level` is a fixed Level whose Display form (e.g. "WARN") is always a
        // valid directive, so this parse cannot fail in practice.
        // `from_env_lossy` still lets `RUST_LOG` refine it when set.
        let filter = EnvFilter::builder()
            .with_default_directive(
                format!("{level}").parse().expect("fixed level is a valid directive"),
            )
            .from_env_lossy();

        let formatter = fmt::layer()
            .with_writer(file_appender)
            .with_ansi(false)
            .with_timer(fmt::time::UtcTime::rfc_3339())
            .with_target(false);

        tracing_subscriber::registry().with(filter).with(formatter).init();

        Ok(Self)
    }

    /// Remove rotated log files older than the retention window.
    /// Files are named `llamatop.log-YYYY-MM-DD` by the daily rolling policy.
    fn cleanup_old_files(dir: &Path) -> std::io::Result<()> {
        let cutoff = std::time::SystemTime::now()
            .checked_sub(Duration::from_secs(RETENTION_DAYS * 24 * 3600))
            .unwrap_or(std::time::UNIX_EPOCH);

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("llamatop.log-") {
                continue;
            }
            let modified = entry.metadata()?.modified().ok();
            if let Some(mtime) = modified {
                if mtime < cutoff {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        Ok(())
    }
}

/// Choose the log level from CLI flags: debug > verbose > default (warn).
pub fn level_from_flags(verbose: bool, debug: bool) -> Level {
    if debug {
        Level::DEBUG
    } else if verbose {
        Level::INFO
    } else {
        Level::WARN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_from_flags_priority() {
        assert_eq!(level_from_flags(false, false), Level::WARN);
        assert_eq!(level_from_flags(true, false), Level::INFO);
        assert_eq!(level_from_flags(true, true), Level::DEBUG);
        assert_eq!(level_from_flags(false, true), Level::DEBUG);
    }
}
