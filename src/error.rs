//! Typed error types for domain and library boundaries.
//!
//! These errors are used inside the crate and across module boundaries.
//! The binary boundary (`main.rs`) wraps them with `anyhow` for context.

/// Errors produced by the backend layer (HTTP transport, API status, parsing).
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("cannot connect to {endpoint}: {source}")]
    Connection {
        endpoint: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("request to {path} timed out")]
    Timeout { path: String },

    #[error("{path}: unexpected HTTP status {status}")]
    HttpStatus { path: String, status: u16 },

    #[error("{path}: HTTP {status}: {message}")]
    HttpStatusWithMessage { path: String, status: u16, message: String },

    #[error("{path}: response is not valid JSON: {detail}")]
    InvalidJson { path: String, detail: String },

    #[error("{path}: could not parse response: {detail}")]
    Parse { path: String, detail: String },

    #[error("authentication failed: the server rejected the API key (HTTP 401)")]
    Authentication,

    #[error("endpoint {path} is not supported by this server (HTTP {status}); {hint}")]
    NotSupported { path: String, status: u16, hint: String },
}

/// Errors produced while loading or validating configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read configuration file at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("cannot parse configuration file at {path}: {detail}")]
    Parse { path: String, detail: String },

    #[error("invalid configuration: {0}")]
    Invalid(String),
}

/// Result alias for configuration operations.
pub type ConfigResult<T> = Result<T, ConfigError>;

/// Human-readable classification used by CLI output (never used for control flow).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// A problem the user can fix (bad config, unreachable server).
    Error,
    /// A feature is unavailable but the application can continue.
    Warning,
}

impl From<&BackendError> for ErrorSeverity {
    fn from(err: &BackendError) -> Self {
        match err {
            BackendError::Authentication
            | BackendError::Connection { .. }
            | BackendError::Timeout { .. }
            | BackendError::HttpStatus { .. }
            | BackendError::HttpStatusWithMessage { .. } => ErrorSeverity::Error,
            BackendError::NotSupported { .. }
            | BackendError::InvalidJson { .. }
            | BackendError::Parse { .. } => ErrorSeverity::Warning,
        }
    }
}

impl From<&ConfigError> for ErrorSeverity {
    fn from(_err: &ConfigError) -> Self {
        ErrorSeverity::Error
    }
}

/// Format an error chain for human-facing CLI output (no stack traces, actionable).
pub fn render_error(err: &dyn std::error::Error) -> String {
    let mut out = String::new();
    let mut current: Option<&dyn std::error::Error> = Some(err);
    let mut depth = 0;
    while let Some(e) = current {
        if depth > 0 {
            out.push_str("\n  caused by: ");
        }
        out.push_str(e.to_string().as_str());
        current = e.source();
        depth += 1;
        if depth > 5 {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_error_includes_cause_chain() {
        #[derive(Debug, thiserror::Error)]
        enum TestError {
            #[error("outer failure")]
            Outer {
                #[source]
                source: std::io::Error,
            },
        }
        let err = TestError::Outer { source: std::io::Error::other("inner failure") };
        let rendered = render_error(&err);
        assert!(rendered.contains("outer failure"));
        assert!(rendered.contains("inner failure"));
        assert!(rendered.contains("caused by"));
    }

    #[test]
    fn severity_classification() {
        assert_eq!(ErrorSeverity::from(&BackendError::Authentication), ErrorSeverity::Error);
        assert_eq!(
            ErrorSeverity::from(&BackendError::NotSupported {
                path: "/metrics".into(),
                status: 501,
                hint: "start the server with --metrics".into(),
            }),
            ErrorSeverity::Warning
        );
        assert_eq!(
            ErrorSeverity::from(&ConfigError::Invalid("bad value".into())),
            ErrorSeverity::Error
        );
    }
}
