//! HTTP client for the llama.cpp server.
//!
//! All requests are GETs to read-only monitoring endpoints. Prompts and
//! completions are never requested. The API key (when configured) is added
//! only as an `Authorization: Bearer` header and is never logged or rendered.

use std::time::Duration;

use url::Url;

use crate::error::BackendError;

/// The llama.cpp server endpoint and its HTTP client.
#[derive(Debug)]
pub struct LlamaCppClient {
    base: Url,
    http: reqwest::Client,
    /// Present when the user configured an API key; applied to every request.
    auth_header: Option<String>,
    /// Last observed `Process-Start-Time-Unix` header value (epoch seconds).
    /// Updated after each `/metrics` fetch; used to detect server restarts.
    last_process_start_unix: std::sync::Mutex<Option<u64>>,
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

        let mut http = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(timeout)
            .user_agent(format!("llamatop/{}", env!("CARGO_PKG_VERSION")));

        let auth_header = match api_key {
            Some(key) if !key.is_empty() => Some(format!("Bearer {key}")),
            _ => None,
        };
        if auth_header.is_some() {
            // Note: the key value itself is not stored in a form that would be
            // printed; it only ever travels in the header below.
            http = http.default_headers({
                let mut h = reqwest::header::HeaderMap::new();
                h.insert(
                    reqwest::header::AUTHORIZATION,
                    auth_header.clone().unwrap().parse().map_err(|e| BackendError::Parse {
                        path: "endpoint".to_string(),
                        detail: format!("invalid header value: {e}"),
                    })?,
                );
                h
            });
        }

        let http = http.build().map_err(|source| BackendError::Connection {
            endpoint: endpoint.to_string(),
            source,
        })?;

        Ok(Self { base, http, auth_header, last_process_start_unix: std::sync::Mutex::new(None) })
    }

    /// The configured endpoint as a plain string (safe to display/log).
    pub fn endpoint(&self) -> &str {
        self.base.as_str()
    }

    /// Whether an API key will be sent (used by doctor, never the value).
    pub fn has_api_key(&self) -> bool {
        self.auth_header.is_some()
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

    /// Build the full URL for a path relative to the endpoint.
    fn url_for(&self, path: &str) -> Url {
        self.base.join(path).unwrap_or_else(|_| self.base.clone())
    }

    /// Perform a GET, returning `(status, body, process_start_unix)` without
    /// treating non-2xx as an error. Callers interpret the status per
    /// endpoint. The `Process-Start-Time-Unix` header (epoch seconds) is
    /// remembered on the client for restart detection.
    pub async fn get_raw(&self, path: &str) -> Result<(u16, String, Option<u64>), BackendError> {
        let url = self.url_for(path);
        let resp = self
            .http
            .get(url.clone())
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
        assert_eq!(client.url_for("health").as_str(), "http://127.0.0.1:8080/health");
    }

    #[test]
    fn base_url_without_trailing_slash_is_normalized() {
        let client = LlamaCppClient::new("http://127.0.0.1:8080", Duration::from_secs(1), None)
            .expect("valid url");
        assert!(client.url_for("slots").as_str().starts_with("http://127.0.0.1:8080/"));
        assert_eq!(client.url_for("slots").as_str(), "http://127.0.0.1:8080/slots");
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
}
