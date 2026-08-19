//! The llama.cpp `llama-server` backend.
//!
//! Reads only the monitoring endpoints (`/health`, `/slots`, `/metrics`,
//! `/props`). It never requests `/completion`, `/chat/completions`, or any
//! endpoint that would return prompt or completion text.

use std::time::Duration;

use async_trait::async_trait;
use tracing::warn;

use super::{BackendCapabilities, BackendHealth, InferenceBackend};
use crate::domain::BackendSnapshot;
use crate::error::BackendError;

pub mod capabilities;
pub mod client;
pub mod health;
pub mod metrics;
pub mod normalize;
pub mod props;
pub mod slots;

use client::LlamaCppClient;
use normalize::{normalize, RawObservation};

/// The llama.cpp inference backend.
pub struct LlamaCppBackend {
    client: LlamaCppClient,
}

impl LlamaCppBackend {
    pub fn new(
        endpoint: &str,
        timeout: Duration,
        api_key: Option<&str>,
    ) -> Result<Self, BackendError> {
        let client = LlamaCppClient::new(endpoint, timeout, api_key)?;
        Ok(Self { client })
    }

    pub fn endpoint(&self) -> &str {
        self.client.endpoint()
    }

    /// The underlying client (exposed for doctor checks that need finer
    /// control, e.g. per-endpoint status).
    pub fn client(&self) -> &LlamaCppClient {
        &self.client
    }

    /// Fetch and parse one endpoint, returning `None` when the endpoint is
    /// disabled (501/404) or the body cannot be parsed (logged, not fatal).
    async fn fetch_json<T>(&self, path: &str, parse: fn(&str) -> Result<T, String>) -> Option<T> {
        match self.client.get_raw(path).await {
            Ok((200, body, _)) => match parse(&body) {
                Ok(value) => Some(value),
                Err(detail) => {
                    warn!(path, "failed to parse {path} response: {detail}");
                    None
                }
            },
            Ok((status, _, _)) => {
                // Endpoint exists but is disabled (501) or in an error state.
                warn!(status, "endpoint {path} not usable");
                None
            }
            Err(err) => {
                warn!(path, "failed to fetch {path}: {err}");
                None
            }
        }
    }
}

#[async_trait]
impl InferenceBackend for LlamaCppBackend {
    fn name(&self) -> &'static str {
        "llama.cpp"
    }

    async fn probe_capabilities(&self) -> Result<BackendCapabilities, BackendError> {
        capabilities::probe_capabilities(&self.client).await
    }

    async fn health(&self) -> Result<BackendHealth, BackendError> {
        let (status, body, _) = self.client.get_raw("health").await?;
        if status == 401 {
            return Err(BackendError::Authentication);
        }
        let outcome = health::parse_health(status, &body);
        Ok(BackendHealth { server: outcome.server, detail: outcome.detail })
    }

    async fn snapshot(
        &self,
        capabilities: &BackendCapabilities,
    ) -> Result<BackendSnapshot, BackendError> {
        // Health is the authoritative reachability check.
        let health = match self.health().await {
            Ok(h) => Some(health::HealthOutcome { server: h.server, detail: h.detail }),
            Err(BackendError::Authentication) => return Err(BackendError::Authentication),
            Err(err) => {
                let detail = describe_transport_error(&err);
                let obs =
                    RawObservation { unreachable: true, error: Some(detail), ..Default::default() };
                return Ok(normalize(&obs));
            }
        };

        let mut obs = RawObservation { health, ..Default::default() };

        // The remaining endpoints are independent; each degrades to None.
        if capabilities.slots {
            obs.slots = self.fetch_json("slots", slots::parse_slots).await;
        }

        if capabilities.metrics {
            match self.client.get_raw("metrics").await {
                Ok((200, body, start)) => {
                    obs.metrics = Some(metrics::parse_metrics(&body));
                    obs.server_start_unix = start.or_else(|| self.client.last_process_start_unix());
                }
                Ok((status, _, _)) => {
                    warn!(status, "/metrics returned a non-200 status; skipping metrics");
                }
                Err(err) => {
                    warn!(error = %err, "failed to fetch /metrics");
                }
            }
        }

        if capabilities.props {
            obs.props = self.fetch_json("props", props::parse_props).await;
        }

        Ok(normalize(&obs))
    }
}

/// Short, secret-free description of a transport error for display.
fn describe_transport_error(err: &BackendError) -> String {
    match err {
        BackendError::Timeout { path } => format!("request to {path} timed out"),
        BackendError::Connection { source, .. } => describe_io(source),
        _ => err.to_string(),
    }
}

fn describe_io(err: &reqwest::Error) -> String {
    if err.is_timeout() {
        "request timed out".to_string()
    } else if err.is_connect() {
        "connection refused or endpoint unreachable".to_string()
    } else {
        "network error".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_exposes_endpoint_without_secret() {
        let backend =
            LlamaCppBackend::new("http://127.0.0.1:8080", Duration::from_secs(1), Some("k"))
                .expect("valid");
        assert_eq!(backend.endpoint(), "http://127.0.0.1:8080/");
        assert_eq!(backend.name(), "llama.cpp");
    }
}
