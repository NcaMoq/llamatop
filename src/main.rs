//! llamatop binary entry point.
//!
//! This layer adds context with `anyhow`, maps failures to exit codes, and
//! keeps human-facing output free of stack traces. JSON mode (`snapshot
//! --json`) writes only JSON to stdout; all diagnostics go to stderr.

use std::process::ExitCode;

use clap::Parser;

use llamatop::cli::{Cli, Command};
use llamatop::logging;

pub const EXIT_FAILURE: i32 = 1;
pub const EXIT_INVALID_CONFIG: i32 = 2;
pub const EXIT_SERVER_UNREACHABLE: i32 = 3;

fn main() -> ExitCode {
    let cli = Cli::parse();
    let code = run(cli);
    ExitCode::from(code as u8)
}

fn run(cli: Cli) -> i32 {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("✘ cannot start the async runtime: {err}");
            return EXIT_FAILURE;
        }
    };

    if let Some(dir) = llamatop::config::log_dir() {
        if let Err(err) =
            logging::Logging::init(logging::level_from_flags(cli.verbose, cli.debug), &dir)
        {
            eprintln!("warning: file logging unavailable: {err}");
        }
    }

    rt.block_on(dispatch(cli))
}

async fn dispatch(cli: Cli) -> i32 {
    let result = match cli.command {
        Some(Command::Doctor) => run_doctor(&cli).await,
        Some(Command::Snapshot { json }) => run_snapshot(&cli, json).await,
        None => run_tui(&cli).await,
    };
    match result {
        Ok(code) => code,
        Err(err) => {
            eprintln!("✘ {}", render_anyhow(&err));
            EXIT_FAILURE
        }
    }
}

async fn run_doctor(cli: &Cli) -> anyhow::Result<i32> {
    let config = match load_config(cli) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("✘ {err}");
            return Ok(EXIT_INVALID_CONFIG);
        }
    };
    let doctor = llamatop::doctor::Doctor::new(config.clone());
    let report = doctor.run_async().await;
    report.print(config.ascii);
    Ok(report.exit_code())
}

async fn run_snapshot(cli: &Cli, json: bool) -> anyhow::Result<i32> {
    let config = match load_config(cli) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("✘ {err}");
            return Ok(EXIT_INVALID_CONFIG);
        }
    };
    let snapshot = llamatop::snapshot::capture(&config)
        .await
        .map_err(|e| anyhow::anyhow!("failed to capture snapshot: {e}"))?;
    llamatop::output::render(&snapshot, json, config.ascii)?;
    Ok(snapshot.exit_code())
}

async fn run_tui(cli: &Cli) -> anyhow::Result<i32> {
    let config = match load_config(cli) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("✘ {err}");
            return Ok(EXIT_INVALID_CONFIG);
        }
    };
    let _ = config;
    anyhow::bail!(
        "the interactive TUI is not available in this build phase; \
         use `llamatop doctor` or `llamatop snapshot`"
    );
}

/// Load and validate configuration.
fn load_config(cli: &Cli) -> Result<llamatop::config::Config, llamatop::error::ConfigError> {
    llamatop::config::Config::load(cli.endpoint.as_deref(), cli.ascii, cli.no_gpu, cli.refresh_ms)
}

/// Format an anyhow error for human-facing output: outer context plus the
/// root cause, without a stack trace.
fn render_anyhow(err: &anyhow::Error) -> String {
    let mut out = err.to_string();
    let cause_msg = err.root_cause().to_string();
    if !cause_msg.is_empty() && cause_msg != out {
        out.push_str("\n  caused by: ");
        out.push_str(&cause_msg);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_anyhow_includes_root_cause() {
        let err = anyhow::anyhow!("outer failure: {}", anyhow::anyhow!("inner failure"));
        let rendered = render_anyhow(&err);
        assert!(rendered.contains("outer failure"));
        assert!(rendered.contains("inner failure"));
    }

    #[test]
    fn exit_codes_are_distinct() {
        assert_eq!(EXIT_FAILURE, 1);
        assert_eq!(EXIT_INVALID_CONFIG, 2);
        assert_eq!(EXIT_SERVER_UNREACHABLE, 3);
    }
}
