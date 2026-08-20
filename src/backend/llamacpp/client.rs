//! HTTP client for the llama.cpp server.
//!
//! All requests are GETs to read-only monitoring endpoints. Prompts and
//! completions are never requested. The API key (when configured) is added
//! per request as a sensitive `Authorization: Bearer` header. The key value
//! is retained only in process memory (this struct) and is never persisted,
//! displayed, logged, or included in Debug output or errors. The endpoint
//! is redacted in Debug output and errors so a credential-bearing URL can
//! never leak.

use std::time::Duration;

use url::Url;

use crate::error::BackendError;

/// The llama.cpp server endpoint and its HTTP client.
pub struct LlamaCppClient {
    base: Url,
    http: reqwest::Client,
    /// Present when the user configured an API key. The value is retained in
    /// process memory only; it is never persisted, displayed, logged, or
    /// included in Debug output or errors.
    api_key: Option<String>,
    /// Last observed `Process-Start-Time-Unix` header value (epoch seconds).
    /// Updated after each `/metrics` fetch; used to detect server restarts.
    last_process_start_unix: std::sync::Mutex<Option<u64>>,
}

impl std::fmt::Debug for LlamaCppClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaCppClient")
            .field("endpoint", &crate::endpoint::redact(self.base.as_str()))
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
        // Defense in depth: the client never retains userinfo, query, or
        // fragment, so a credential-bearing URL cannot leak through the
        // endpoint, Debug output, or connection errors. The config already
        // rejects these; this strips them even if a caller bypasses it.
        if !base.username().is_empty() || base.password().is_some() {
            let _ = base.set_username("");
            let _ = base.set_password(None);
        }
        base.set_query(None);
        base.set_fragment(None);
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path().trim_end_matches('/')));
        }

        let http = reqwest::Client::builder()
            .timeout(timeout)
            .connect_timeout(timeout)
            .user_agent(format!("llamatop/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|source| BackendError::Connection {
                endpoint: base.as_str().to_string(),
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
    /// remembered on the client for restart detection. The body is read with
    /// a size bound (see [`read_body_bounded`]) so a hostile or misbehaving
    /// server cannot force an unbounded buffer.
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

        let mut resp = req
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

        let body = read_body_bounded(&mut resp, MAX_BODY_BYTES, path).await?;
        Ok((status, body, start_unix))
    }
}

/// Maximum response body size the client will buffer. Monitoring endpoints
/// return small JSON/text payloads; 16 MiB is far above any legitimate
/// response while bounding a hostile server's ability to exhaust memory.
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Human-readable form of [`MAX_BODY_BYTES`] for error messages.
const MAX_BODY_LABEL: &str = "16 MiB";

/// Read a response body as UTF-8 text, enforcing a size limit.
///
/// The limit is applied twice: first, when the server declares a
/// `Content-Length` larger than the limit, the body is rejected without
/// being read; second, while streaming (covers chunked bodies and lying
/// `Content-Length`), the accumulated bytes are checked before each chunk is
/// buffered. The check happens before UTF-8 conversion. The error carries
/// only the path and the limit, never any body content.
async fn read_body_bounded(
    resp: &mut reqwest::Response,
    limit: usize,
    path: &str,
) -> Result<String, BackendError> {
    if let Some(len) = resp.content_length() {
        if len > limit as u64 {
            return Err(BackendError::BodyTooLarge {
                path: path.to_string(),
                limit: MAX_BODY_LABEL,
            });
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    loop {
        let chunk = resp.chunk().await.map_err(|e| BackendError::InvalidJson {
            path: path.to_string(),
            detail: format!("cannot read response body: {e}"),
        })?;
        match chunk {
            Some(chunk) => {
                if buf.len() + chunk.len() > limit {
                    return Err(BackendError::BodyTooLarge {
                        path: path.to_string(),
                        limit: MAX_BODY_LABEL,
                    });
                }
                buf.extend_from_slice(&chunk);
            }
            None => break,
        }
    }

    // Validate UTF-8 only after the size bound has been applied, so an
    // oversized non-UTF-8 body is reported as too large, not as a UTF-8 error.
    String::from_utf8(buf).map_err(|e| BackendError::InvalidJson {
        path: path.to_string(),
        detail: format!("response body is not valid UTF-8: {e}"),
    })
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
    fn client_strips_credentials_from_endpoint() {
        // Defense in depth: even if a credential-bearing endpoint reaches the
        // client (bypassing config validation), none of its surfaces expose
        // the userinfo, query, or fragment.
        let client = LlamaCppClient::new(
            "http://user:topsecret@example.com:8080/?token=q#frag",
            Duration::from_secs(1),
            None,
        )
        .expect("parses");
        let endpoint = client.endpoint();
        assert_eq!(endpoint, "http://example.com:8080/");
        assert!(!endpoint.contains("topsecret"));
        assert!(!endpoint.contains("token"));
        let debug = format!("{client:?}");
        assert!(!debug.contains("topsecret"), "Debug leaks credentials: {debug}");
        assert!(!debug.contains("user@"));
        // The joined request URL also never carries the secret.
        let joined = client.url_for("health").unwrap();
        assert!(!joined.as_str().contains("topsecret"));
        assert!(!joined.as_str().contains("user@"));
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

    // --- Response body size bound ---

    async fn mount_body(server: &wiremock::MockServer, body: String) {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, ResponseTemplate};
        Mock::given(method("GET"))
            .and(path("slots"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(body, "text/plain"))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn body_at_limit_succeeds() {
        let server = wiremock::MockServer::start().await;
        mount_body(&server, "x".repeat(100)).await;
        let c = LlamaCppClient::new(server.uri().as_str(), Duration::from_secs(1), None)
            .expect("valid");
        let url = format!("{}/slots", server.uri());
        let mut resp = c.http.get(&url).send().await.unwrap();
        let out = super::read_body_bounded(&mut resp, 100, "slots").await.unwrap();
        assert_eq!(out.len(), 100, "a body exactly at the limit is accepted");
    }

    #[tokio::test]
    async fn body_above_limit_is_rejected() {
        let server = wiremock::MockServer::start().await;
        // A distinctive payload so we can prove the error carries no body.
        mount_body(&server, "ZZZZ".repeat(25)).await;
        let c = LlamaCppClient::new(server.uri().as_str(), Duration::from_secs(1), None)
            .expect("valid");
        let url = format!("{}/slots", server.uri());
        let mut resp = c.http.get(&url).send().await.unwrap();
        let err = super::read_body_bounded(&mut resp, 99, "slots").await.unwrap_err();
        assert!(matches!(err, BackendError::BodyTooLarge { .. }), "got {err:?}");
        // The error message carries no body content.
        assert!(!err.to_string().contains("ZZZZ"), "body leaked: {}", err);
    }

    #[tokio::test]
    async fn chunked_body_above_limit_fails_via_streamed_read() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        // A raw HTTP/1.1 server that replies with a chunked body (no
        // Content-Length) of 1024 bytes. The streamed-read bound must reject
        // it against a 100-byte limit even though no Content-Length exists.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut req = [0u8; 8192];
            let _ = sock.read(&mut req).await;
            let half = "x".repeat(512);
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n200\r\n{half}\r\n200\r\n{half}\r\n0\r\n\r\n"
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.flush().await;
        });

        let url = format!("http://{addr}/slots");
        let c = LlamaCppClient::new(&url, Duration::from_secs(1), None).unwrap();
        let mut resp = c.http.get(&url).send().await.unwrap();
        assert!(resp.content_length().is_none(), "chunked body has no Content-Length");
        let err = super::read_body_bounded(&mut resp, 100, "slots").await.unwrap_err();
        assert!(matches!(err, BackendError::BodyTooLarge { .. }), "got {err:?}");
        let _ = server.await;
    }

    #[tokio::test]
    async fn small_body_through_get_raw_succeeds() {
        // The default (16 MiB) limit accepts a normal small response end to
        // end, and the body is returned intact.
        let server = wiremock::MockServer::start().await;
        mount_body(&server, r#"{"ok":true}"#.to_string()).await;
        let c = LlamaCppClient::new(server.uri().as_str(), Duration::from_secs(1), None)
            .expect("valid");
        let (status, body, _) = c.get_raw("slots").await.unwrap();
        assert_eq!(status, 200);
        assert_eq!(body, r#"{"ok":true}"#);
    }
}
