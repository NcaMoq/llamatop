//! Command-line interface definitions (clap, derive API).

use clap::{Parser, Subcommand};

/// llamatop: a terminal monitor for llama.cpp llama-server.
///
/// Monitors server state, per-slot workload phase, throughput, GPU and system
/// usage of a running llama-server instance.
///
/// Authentication: set the API key in the environment variable named by
/// `authentication.api_key_env` in the config file (default: LLAMATOP_API_KEY).
/// The API key is intentionally not accepted as a command-line argument because
/// process command lines are visible to other users on the system.
#[derive(Debug, Parser)]
#[command(name = "llamatop", version, about, long_about = None)]
pub struct Cli {
    /// Endpoint URL of the llama-server to monitor (default: http://127.0.0.1:8080)
    #[arg(long, global = true, value_name = "URL")]
    pub endpoint: Option<String>,

    /// Use ASCII-only characters in output (no Unicode symbols)
    #[arg(long, global = true)]
    pub ascii: bool,

    /// Disable GPU monitoring
    #[arg(long, global = true)]
    pub no_gpu: bool,

    /// Snapshot refresh interval in milliseconds (minimum: 100)
    #[arg(long, global = true, value_name = "MS")]
    pub refresh_ms: Option<u64>,

    /// Increase log verbosity (info level)
    #[arg(long, global = true)]
    pub verbose: bool,

    /// Enable debug logging (writes full details to the log file)
    #[arg(long, global = true)]
    pub debug: bool,

    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Check the environment and server connectivity
    Doctor,

    /// Capture a single snapshot and exit
    Snapshot {
        /// Output the snapshot as JSON (stdout contains only the JSON document)
        #[arg(long)]
        json: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_bare_invocation() {
        let cli = Cli::parse_from(["llamatop"]);
        assert!(cli.command.is_none());
        assert!(!cli.ascii);
        assert!(!cli.no_gpu);
        assert!(cli.endpoint.is_none());
    }

    #[test]
    fn parses_global_flags_with_subcommand() {
        let cli = Cli::parse_from([
            "llamatop",
            "--endpoint",
            "http://10.0.0.5:8080",
            "--ascii",
            "--no-gpu",
            "--refresh-ms",
            "750",
            "snapshot",
            "--json",
        ]);
        let cmd = cli.command.expect("subcommand");
        match cmd {
            Command::Snapshot { json } => assert!(json),
            Command::Doctor => panic!("expected snapshot"),
        }
        assert_eq!(cli.endpoint.as_deref(), Some("http://10.0.0.5:8080"));
        assert!(cli.ascii);
        assert!(cli.no_gpu);
        assert_eq!(cli.refresh_ms, Some(750));
    }

    #[test]
    fn parses_doctor() {
        let cli = Cli::parse_from(["llamatop", "doctor", "--verbose"]);
        assert!(matches!(cli.command, Some(Command::Doctor)));
        assert!(cli.verbose);
    }

    #[test]
    fn unknown_subcommand_is_rejected() {
        assert!(Cli::try_parse_from(["llamatop", "bogus"]).is_err());
    }
}
