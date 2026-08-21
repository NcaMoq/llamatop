//! The llama.cpp `llama-server` backend.
//!
//! Reads only the monitoring endpoints (`/health`, `/slots`, `/metrics`,
//! `/props`). It never requests `/completion`, `/chat/completions`, or any
//! endpoint that would return prompt or completion text.

use std::time::Duration;

use async_trait::async_trait;
use tracing::warn;

use super::{BackendCapabilities, BackendHealth, EndpointAvailability, InferenceBackend};
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

    /// Fetch, classify, and parse one endpoint.
    ///
    /// Returns `(observation, parsed)` — `parsed` is `Some` only when the
    /// observation is `Available`. The observation is an
    /// [`EndpointAvailability`] so the caller can record *why* an endpoint
    /// is missing data (disabled vs. busy vs. unparseable vs. auth), not
    /// just that it is.
    async fn fetch_endpoint<T>(
        &self,
        path: &str,
        parse: fn(&str) -> Result<T, String>,
    ) -> (EndpointAvailability, Option<T>) {
        match self.client.get_raw(path).await {
            Ok((status, body, _)) => {
                let parsed = parse(&body);
                let observation = capabilities::classify_response(status, &body, parsed.is_ok());
                match (observation, parsed) {
                    (EndpointAvailability::Available, Ok(value)) => (observation, Some(value)),
                    (EndpointAvailability::ParseFailed, Err(detail)) => {
                        warn!(path, "failed to parse {path} response: {detail}");
                        (observation, None)
                    }
                    (observation, _) => (observation, None),
                }
            }
            Err(BackendError::BodyTooLarge { .. }) => {
                warn!(path, "{path} response exceeded the body limit");
                (EndpointAvailability::ParseFailed, None)
            }
            Err(err) => {
                warn!(path, "failed to fetch {path}: {err}");
                (EndpointAvailability::TemporarilyUnavailable, None)
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
        capabilities: &mut BackendCapabilities,
    ) -> Result<BackendSnapshot, BackendError> {
        // Health is the authoritative reachability check and is re-observed
        // every snapshot regardless of its previous state: it doubles as the
        // connection heartbeat.
        let (health_availability, outcome) = match self.client.get_raw("health").await {
            Ok((status, body, _)) => health::classify_health(status, &body),
            Err(BackendError::Authentication) => {
                capabilities.health = EndpointAvailability::AuthenticationFailed;
                return Err(BackendError::Authentication);
            }
            Err(err) => {
                capabilities.health = EndpointAvailability::TemporarilyUnavailable;
                let detail = describe_transport_error(&err);
                let obs =
                    RawObservation { unreachable: true, error: Some(detail), ..Default::default() };
                return Ok(normalize(&obs));
            }
        };
        capabilities.health = health_availability;

        let mut obs = RawObservation {
            health: Some(health::HealthOutcome { server: outcome.server, detail: outcome.detail }),
            ..Default::default()
        };

        // The remaining endpoints are independent; each degrades to None and
        // re-observes its availability so temporary states recover without a
        // manual reconnect. Unsupported / AuthenticationFailed endpoints are
        // skipped on the regular cycle (re-probed on reconnect).
        if capabilities.slots.needs_observation() {
            let (observation, parsed) = self.fetch_endpoint("slots", slots::parse_slots).await;
            capabilities.slots = observation;
            obs.slots = parsed;
        }

        if capabilities.metrics.needs_observation() {
            match self.client.get_raw("metrics").await {
                Ok((status, body, _)) => {
                    let parsed = metrics::parse_metrics(&body);
                    let observation = capabilities::classify_response(status, &body, true);
                    capabilities.metrics = observation;
                    if observation.is_available() {
                        obs.metrics = Some(parsed);
                        // The `Process-Start-Time-Unix` header is remembered
                        // on the client for restart detection.
                        obs.server_start_unix = self.client.last_process_start_unix();
                    } else {
                        warn!(status, "/metrics returned a non-usable status; skipping metrics");
                    }
                }
                Err(BackendError::BodyTooLarge { .. }) => {
                    capabilities.metrics = EndpointAvailability::ParseFailed;
                }
                Err(err) => {
                    capabilities.metrics = EndpointAvailability::TemporarilyUnavailable;
                    warn!(error = %err, "failed to fetch /metrics");
                }
            }
        }

        if capabilities.props.needs_observation() {
            let (observation, parsed) = self.fetch_endpoint("props", props::parse_props).await;
            capabilities.props = observation;
            obs.props = parsed;
        }

        // Keep the derived capability flags in sync with the observations so
        // a stale probe and a recovered fetch agree.
        capabilities.model_info = capabilities.props.is_available();
        capabilities.speculative_metrics = capabilities.metrics.is_available();
        capabilities.exact_decode_state = capabilities.slots.is_available();

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
