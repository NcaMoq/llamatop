//! HTTP client for the llama.cpp server.
//!
//! All requests are GETs to read-only monitoring endpoints. Prompts and
//! completions are never requested. The API key (when configured) is added
//! per request as a sensitive `Authorization: Bearer` header; the key value
//! is never stored in this struct, never logged, and never appears in Debug
//! output or errors.

use std::time::Duration;

use url::Url;

use crate::error::BackendError;

/// The llama.cpp server endpoint and its HTTP client.
pub struct LlamaCppClient {
    base: Url,
    http: reqwest::Client,
    /// Present when the user configured an API key; the key itself is read
    /// from the environment variable at request time and never stored here.
    api_key: Option<String>,
    /// Last observed `Process-Start-Time-Unix` header value (epoch seconds).
    /// Updated after each `/metrics` fetch; used to detect server restarts.
    last_process_start_unix: std::sync::Mutex<Option<u64>>,
}

impl std::fmt::Debug for LlamaCppClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppClient")
            .field("endpoint", &self.base.as_str())
            .field("has_api_key", &self.api_key.is_some())
            .finish_non_exhaustive()
    }
}

impl LlamaCppClient {
    pub fn new(
        endpoint: &str,
        timeout: Duration,
        api_key: Option<&str>,
    ) -> Result<Self, BackendError> {
        let mut base = Url::parse(endpoint).map_err(|_| BackendError::Parse {
            path: "endpoint".to_string(),
            detail: "not a valid URL".to_string(),
        })?;
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path().trim_end_matches('/')));
        }

        let http = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(timeout)
            .user_agent(format!("llamatop/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|source| BackendError::Connection {
                endpoint: endpoint.to_string(),
                source,
            })?;

        let api_key = api_key.filter(|k| !k.is_empty()).map(str::to_string);

        Ok(Self { base, http, api_key, last_process_start_unix: std::sync::Mutex::new(None) })
    }

    /// The configured endpoint as a plain string (safe to display/log).
    pub fn endpoint(&self) -> &str {
        self.base.as_str()
    }

    /// Whether an API key will be sent (used by doctor, never the value).
    pub fn has_api_key(&self) -> bool {
        self.api_key.is_some()
    }

    /// The last observed server process start time (epoch seconds), if the
    /// server reported it. Changes across server restarts.
    pub fn last_process_start_unix(&self) -> Option<u64> {
        self.last_process_start_unix.lock().ok().and_then(|g| *g)
    }

    fn remember_process_start_unix(&self, value: Option<u64>) {
        if let Some(v) = value {
            if let Ok(mut guard) = self.last_process_start_unix.lock() {
                *guard = Some(v);
            }
        }
    }

    /// Build the full URL for a path relative to the endpoint. Failures are
    /// reported, never silently swallowed: a path that cannot be joined, or
    /// that resolves to a different scheme/host/port (e.g. an absolute URL),
    /// is rejected so the request can never reach the wrong endpoint.
    fn url_for(&self, path: &str) -> Result<Url, BackendError> {
        let joined = self.base.join(path).map_err(|e| BackendError::Parse {
            path: path.to_string(),
            detail: format!("cannot join path onto base URL: {e}"),
        })?;
        if joined.scheme() != self.base.scheme()
            || joined.host_str() != self.base.host_str()
            || joined.port_or_known_default() != self.base.port_or_known_default()
        {
            return Err(BackendError::Parse {
                path: path.to_string(),
                detail: "path must be relative to the endpoint".to_string(),
            });
        }
        Ok(joined)
    }

    /// Perform a GET, returning `(status, body, process_start_unix)` without
    /// treating non-2xx as an error. Callers interpret the status per
    /// endpoint. The `Process-Start-Time-Unix` header (epoch seconds) is
    /// remembered on the client for restart detection.
    pub async fn get_raw(&self, path: &str) -> Result<(u16, String, Option<u64>), BackendError> {
        let url = self.url_for(path)?;
        let mut req = self.http.get(url);
        if let Some(key) = &self.api_key {
            let mut header_value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|e| BackendError::Parse {
                    path: path.to_string(),
                    detail: format!("invalid header value: {e}"),
                })?;
            header_value.set_sensitive(true);
            req = req.header(reqwest::header::AUTHORIZATION, header_value);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| classify_transport_error(self.endpoint().to_string(), path, e))?;

        let status = resp.status().as_u16();
        let start_unix = resp
            .headers()
            .get("Process-Start-Time-Unix")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok());
        self.remember_process_start_unix(start_unix);

        let body = resp.text().await.map_err(|e| BackendError::InvalidJson {
            path: path.to_string(),
            detail: format!("cannot read response body: {e}"),
        })?;
        Ok((status, body, start_unix))
    }
}

/// Map a reqwest transport error to a typed backend error.
fn classify_transport_error(endpoint: String, path: &str, e: reqwest::Error) -> BackendError {
    if e.is_timeout() {
        BackendError::Timeout { path: path.to_string() }
    } else {
        BackendError::Connection { endpoint, source: e }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_is_normalized_with_trailing_slash() {
        let client = LlamaCppClient::new("http://127.0.0.1:8080", Duration::from_secs(1), None)
            .expect("valid url");
        assert_eq!(client.url_for("health").unwrap().as_str(), "http://127.0.0.1:8080/health");
    }

    #[test]
    fn base_url_without_trailing_slash_is_normalized() {
        let client = LlamaCppClient::new("http://127.0.0.1:8080", Duration::from_secs(1), None)
            .expect("valid url");
        let joined = client.url_for("slots").unwrap();
        assert!(joined.as_str().starts_with("http://127.0.0.1:8080/"));
        assert_eq!(joined.as_str(), "http://127.0.0.1:8080/slots");
    }

    #[test]
    fn invalid_endpoint_is_rejected() {
        let err = LlamaCppClient::new("not a url", Duration::from_secs(1), None).unwrap_err();
        assert!(matches!(err, BackendError::Parse { .. }));
    }

    #[test]
    fn api_key_sets_auth_flag_only() {
        let client =
            LlamaCppClient::new("http://127.0.0.1:8080", Duration::from_secs(1), Some("secret"))
                .expect("valid url");
        assert!(client.has_api_key());
        // The endpoint string never contains the key.
        assert!(!client.endpoint().contains("secret"));
    }

    #[test]
    fn debug_output_never_contains_the_api_key() {
        let client =
            LlamaCppClient::new("http://127.0.0.1:8080", Duration::from_secs(1), Some("secret"))
                .expect("valid url");
        let debug = format!("{client:?}");
        assert!(!debug.contains("secret"), "Debug output leaks the key: {debug}");
        assert!(debug.contains("has_api_key"));
        assert!(debug.contains("127.0.0.1:8080"));
    }

    #[test]
    fn empty_api_key_is_treated_as_absent() {
        let client = LlamaCppClient::new("http://127.0.0.1:8080", Duration::from_secs(1), Some(""))
            .expect("valid url");
        assert!(!client.has_api_key());
        assert!(!format!("{client:?}").contains("secret"));
    }

    #[test]
    fn url_join_failure_is_reported_not_silently_fallen_back() {
        // A path that cannot be joined onto the base must produce a typed
        // error instead of silently requesting the base URL itself.
        let client = LlamaCppClient::new("http://127.0.0.1:8080", Duration::from_secs(1), None)
            .expect("valid url");
        let err = client.url_for("http://attacker.example/steal").unwrap_err();
        assert!(matches!(err, BackendError::Parse { .. }));
    }
}
