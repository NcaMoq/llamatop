//! Configuration loading, validation, and platform paths.
//!
//! Precedence (highest to lowest): CLI arguments > environment variables >
//! configuration file > built-in defaults.
//!
//! The API key itself is never stored in the configuration file; only the
//! name of the environment variable that holds it (`authentication.api_key_env`).

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use url::Url;

use crate::error::{ConfigError, ConfigResult};

pub const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8080";
pub const DEFAULT_REFRESH_INTERVAL_MS: u64 = 500;
pub const DEFAULT_SLOT_INTERVAL_MS: u64 = 1000;
pub const DEFAULT_METRICS_INTERVAL_MS: u64 = 500;
pub const DEFAULT_HEALTH_INTERVAL_MS: u64 = 1000;
pub const DEFAULT_PROPS_INTERVAL_MS: u64 = 2000;
pub const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 1500;
pub const DEFAULT_HISTORY_SECONDS: u64 = 120;
pub const MIN_INTERVAL_MS: u64 = 100;
pub const MIN_HISTORY_SECONDS: u64 = 10;
pub const MAX_HISTORY_SECONDS: u64 = 3600;
pub const API_KEY_ENV_DEFAULT: &str = "LLAMATOP_API_KEY";
pub const ENV_ENDPOINT: &str = "LLAMATOP_ENDPOINT";
pub const ENV_CONFIG_PATH: &str = "LLAMATOP_CONFIG_PATH";

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    pub endpoint: String,
    pub refresh_interval_ms: u64,
    pub slot_interval_ms: u64,
    pub metrics_interval_ms: u64,
    pub health_interval_ms: u64,
    pub props_interval_ms: u64,
    pub request_timeout_ms: u64,
    pub theme: String,
    pub ascii: bool,
    pub show_gpu: bool,
    pub show_system: bool,
    pub history_seconds: u64,
    pub authentication: AuthenticationConfig,
    pub gpu: GpuConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct AuthenticationConfig {
    /// Name of the environment variable that holds the API key.
    pub api_key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct GpuConfig {
    /// GPU metrics backend: "auto", "nvml", or "none".
    pub backend: String,
    /// Restrict monitoring to these GPU indices (empty = all).
    pub device_indices: Vec<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            endpoint: DEFAULT_ENDPOINT.to_string(),
            refresh_interval_ms: DEFAULT_REFRESH_INTERVAL_MS,
            slot_interval_ms: DEFAULT_SLOT_INTERVAL_MS,
            metrics_interval_ms: DEFAULT_METRICS_INTERVAL_MS,
            health_interval_ms: DEFAULT_HEALTH_INTERVAL_MS,
            props_interval_ms: DEFAULT_PROPS_INTERVAL_MS,
            request_timeout_ms: DEFAULT_REQUEST_TIMEOUT_MS,
            theme: "cloudflare".to_string(),
            ascii: false,
            show_gpu: true,
            show_system: true,
            history_seconds: DEFAULT_HISTORY_SECONDS,
            authentication: AuthenticationConfig::default(),
            gpu: GpuConfig::default(),
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self { api_key_env: API_KEY_ENV_DEFAULT.to_string() }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self { backend: "auto".to_string(), device_indices: Vec::new() }
    }
}

impl Config {
    /// Load configuration applying precedence: CLI > env > file > defaults.
    pub fn load(
        cli_endpoint: Option<&str>,
        cli_ascii: bool,
        cli_no_gpu: bool,
        cli_no_system: bool,
        cli_refresh_ms: Option<u64>,
    ) -> ConfigResult<Config> {
        let mut config = Self::default();

        if let Some(file_config) = Self::read_file(&config_path())? {
            config = file_config;
        }

        if let Ok(env_endpoint) = std::env::var(ENV_ENDPOINT) {
            if !env_endpoint.is_empty() {
                config.endpoint = env_endpoint;
            }
        }

        if let Some(endpoint) = cli_endpoint {
            config.endpoint = endpoint.to_string();
        }
        if cli_ascii {
            config.ascii = true;
        }
        if cli_no_gpu {
            config.show_gpu = false;
        }
        if cli_no_system {
            config.show_system = false;
        }
        if let Some(refresh_ms) = cli_refresh_ms {
            config.refresh_interval_ms = refresh_ms;
        }

        config.validate()?;
        Ok(config)
    }

    /// Read and parse the configuration file if it exists.
    /// A missing file is not an error; a malformed one is.
    fn read_file(path: &PathBuf) -> ConfigResult<Option<Config>> {
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read_to_string(path)
            .map_err(|source| ConfigError::Io { path: path.display().to_string(), source })?;
        let config: Config = toml::from_str(&raw).map_err(|e| ConfigError::Parse {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        Ok(Some(config))
    }

    /// Validate all values; returns a precise, user-actionable error.
    pub fn validate(&self) -> ConfigResult<()> {
        let url = Url::parse(&self.endpoint).map_err(|e| {
            ConfigError::Invalid(format!(
                "endpoint must be a valid http(s) URL. Current value: {}. Detail: {e}",
                crate::endpoint::redact(&self.endpoint)
            ))
        })?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(ConfigError::Invalid(format!(
                "endpoint must use the http or https scheme. Current value: {}",
                crate::endpoint::redact(&self.endpoint)
            )));
        }
        // Credentials in the URL are rejected. The API key comes only from the
        // environment variable; any userinfo, query, or fragment could leak a
        // secret into display, logs, or the JSON snapshot. The error text
        // deliberately does not echo the endpoint (it may contain the secret).
        if !url.username().is_empty() || url.password().is_some() {
            return Err(ConfigError::Invalid(
                "endpoint must not include a username or password; provide the API key via the environment variable instead".to_string(),
            ));
        }
        if url.query().is_some() || url.fragment().is_some() {
            return Err(ConfigError::Invalid(
                "endpoint must not include a query string or fragment".to_string(),
            ));
        }

        let intervals: [(&str, u64); 5] = [
            ("refresh_interval_ms", self.refresh_interval_ms),
            ("slot_interval_ms", self.slot_interval_ms),
            ("metrics_interval_ms", self.metrics_interval_ms),
            ("health_interval_ms", self.health_interval_ms),
            ("props_interval_ms", self.props_interval_ms),
        ];
        for (name, value) in intervals {
            if value < MIN_INTERVAL_MS {
                return Err(ConfigError::Invalid(format!(
                    "{name} must be at least {MIN_INTERVAL_MS}. Current value: {value}"
                )));
            }
        }

        if self.request_timeout_ms < MIN_INTERVAL_MS {
            return Err(ConfigError::Invalid(format!(
                "request_timeout_ms must be at least {MIN_INTERVAL_MS}. Current value: {}",
                self.request_timeout_ms
            )));
        }

        if self.history_seconds < MIN_HISTORY_SECONDS || self.history_seconds > MAX_HISTORY_SECONDS
        {
            return Err(ConfigError::Invalid(format!(
                "history_seconds must be between {MIN_HISTORY_SECONDS} and {MAX_HISTORY_SECONDS}. Current value: {}",
                self.history_seconds
            )));
        }

        if self.gpu.backend != "auto" && self.gpu.backend != "nvml" && self.gpu.backend != "none" {
            return Err(ConfigError::Invalid(format!(
                "gpu.backend must be \"auto\", \"nvml\", or \"none\". Current value: \"{}\"",
                self.gpu.backend
            )));
        }

        Ok(())
    }

    /// Resolve the API key from the environment variable named in config.
    /// Returns `None` when the variable is unset or empty.
    /// The key value must never be logged or rendered.
    pub fn api_key(&self) -> Option<String> {
        let name = self.authentication.api_key_env.trim();
        if name.is_empty() {
            return None;
        }
        let value = std::env::var(name).ok()?;
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
    }

    pub fn refresh_interval(&self) -> Duration {
        Duration::from_millis(self.refresh_interval_ms)
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }
}

/// Path to the configuration file: `%APPDATA%\llamatop\config.toml` on Windows.
/// Overridable via `LLAMATOP_CONFIG_PATH` (useful for tests).
pub fn config_path() -> PathBuf {
    if let Ok(p) = std::env::var(ENV_CONFIG_PATH) {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    if let Some(dirs) = directories::ProjectDirs::from("", "", "llamatop") {
        return dirs.config_dir().join("config.toml");
    }
    PathBuf::from("config.toml")
}

/// Directory for log files: `%LOCALAPPDATA%\llamatop\logs` on Windows.
pub fn log_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "llamatop").map(|d| d.data_local_dir().join("logs"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_pass_validation() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.endpoint, DEFAULT_ENDPOINT);
        assert_eq!(config.refresh_interval_ms, 500);
    }

    #[test]
    fn invalid_endpoint_scheme_rejected() {
        let config = Config { endpoint: "ftp://example.com".to_string(), ..Default::default() };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("http or https scheme"));
    }

    #[test]
    fn invalid_endpoint_url_rejected() {
        let config = Config { endpoint: "not a url".to_string(), ..Default::default() };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("valid http(s) URL"));
    }

    #[test]
    fn endpoint_with_username_rejected() {
        let config =
            Config { endpoint: "http://user@example.com:8080".to_string(), ..Default::default() };
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("must not include a username or password"));
        // The error must not echo the credentials or the raw endpoint.
        assert!(!msg.contains("user@example.com"));
        assert!(!msg.contains("example.com"));
        assert!(!msg.contains("http://"));
    }

    #[test]
    fn endpoint_with_password_rejected() {
        let config = Config {
            endpoint: "http://user:topsecret@example.com:8080".to_string(),
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("must not include a username or password"));
        assert!(!msg.contains("topsecret"), "password must not leak into the error");
        assert!(!msg.contains("user:"));
    }

    #[test]
    fn endpoint_with_query_rejected() {
        let config = Config {
            endpoint: "http://127.0.0.1:8080/?token=secret".to_string(),
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("must not include a query string or fragment"));
        assert!(!msg.contains("token=secret"));
    }

    #[test]
    fn endpoint_with_fragment_rejected() {
        let config =
            Config { endpoint: "http://127.0.0.1:8080/#secret".to_string(), ..Default::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn clean_endpoints_still_pass() {
        for endpoint in [
            "http://127.0.0.1:8080",
            "http://127.0.0.1:8080/",
            "https://llama.example.com:9091/some/path",
            "http://[::1]:8080",
        ] {
            let config = Config { endpoint: endpoint.to_string(), ..Default::default() };
            assert!(config.validate().is_ok(), "{endpoint} should be accepted");
        }
    }

    #[test]
    fn refresh_interval_below_minimum_rejected() {
        let config = Config { refresh_interval_ms: 10, ..Default::default() };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("refresh_interval_ms must be at least 100"));
        assert!(err.to_string().contains("Current value: 10"));
    }

    #[test]
    fn history_seconds_out_of_range_rejected() {
        let config = Config { history_seconds: 1, ..Default::default() };
        assert!(config.validate().is_err());
        let config = Config { history_seconds: 999_999, ..Default::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn gpu_backend_must_be_known() {
        let mut config = Config::default();
        config.gpu.backend = "amdgpu".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn file_config_is_parsed_with_unknown_fields_tolerated() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "endpoint = \"http://127.0.0.1:9999\"\nfuture_option = true\n")
            .unwrap();
        let loaded = Config::read_file(&path).unwrap().unwrap();
        assert_eq!(loaded.endpoint, "http://127.0.0.1:9999");
        assert_eq!(loaded.refresh_interval_ms, DEFAULT_REFRESH_INTERVAL_MS);
    }

    #[test]
    fn malformed_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "endpoint = ").unwrap();
        let err = Config::read_file(&path).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn missing_file_is_not_an_error() {
        let missing = PathBuf::from("does/not/exist/config.toml");
        assert!(Config::read_file(&missing).unwrap().is_none());
    }

    #[test]
    fn cli_overrides_env_and_file() {
        // Simulate env endpoint
        std::env::set_var(ENV_ENDPOINT, "http://127.0.0.1:7777");
        let config =
            Config::load(Some("http://127.0.0.1:8888"), false, false, false, None).unwrap();
        assert_eq!(config.endpoint, "http://127.0.0.1:8888");

        let config = Config::load(None, false, false, false, None).unwrap();
        assert_eq!(config.endpoint, "http://127.0.0.1:7777");

        std::env::remove_var(ENV_ENDPOINT);
        let config = Config::load(None, false, false, false, Some(200)).unwrap();
        assert_eq!(config.refresh_interval_ms, 200);
    }

    #[test]
    fn cli_no_system_disables_system_monitor() {
        let config = Config::load(None, false, false, false, None).unwrap();
        assert!(config.show_system, "system monitoring is on by default");
        let config = Config::load(None, false, false, true, None).unwrap();
        assert!(!config.show_system, "--no-system must disable the monitor");
    }
}
