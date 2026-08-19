//! Capability probing: determine which endpoints this server supports.
//!
//! A disabled endpoint (HTTP 501 with a not_supported error) is *not* a
//! failure of the probe — it is recorded as unavailable. Only transport-level
//! failures (unreachable) make the whole probe fail.

use super::client::LlamaCppClient;
use crate::backend::BackendCapabilities;
use crate::error::BackendError;

/// Outcome of probing a single endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointStatus {
    /// The endpoint answered and is usable.
    Available,
    /// The server answered but explicitly disabled this endpoint (501).
    Disabled,
    /// The endpoint answered with an unexpected status (treated as unavailable).
    Unavailable,
}

/// Probe one endpoint by issuing a GET and interpreting the status.
///
/// `expect`: statuses that count as available for this endpoint.
///
/// Transport failures (connection refused, timeout, DNS) are propagated as
/// `Err` because they mean the server is unreachable, not that one endpoint
/// is disabled.
pub async fn probe_endpoint(
    client: &LlamaCppClient,
    path: &str,
    expect_available_statuses: &[u16],
) -> Result<EndpointStatus, BackendError> {
    let (status, _body, _start) = client.get_raw(path).await?;
    Ok(if expect_available_statuses.contains(&status) {
        EndpointStatus::Available
    } else if status == 501 || status == 404 {
        EndpointStatus::Disabled
    } else {
        EndpointStatus::Unavailable
    })
}

/// Probe all monitoring endpoints and assemble `BackendCapabilities`.
///
/// The probe fails (returns Err) only when the server cannot be reached at all.
pub async fn probe_capabilities(
    client: &LlamaCppClient,
) -> Result<BackendCapabilities, BackendError> {
    let mut caps = BackendCapabilities::default();

    let health = probe_endpoint(client, "health", &[200, 503]).await?;
    caps.health = matches!(health, EndpointStatus::Available);

    let slots = probe_endpoint(client, "slots", &[200]).await;
    caps.slots = matches!(slots, Ok(EndpointStatus::Available));

    let metrics = probe_endpoint(client, "metrics", &[200]).await;
    caps.metrics = matches!(metrics, Ok(EndpointStatus::Available));
    if caps.metrics {
        // Speculative metrics are part of the same endpoint; their presence is
        // determined per-snapshot from the actual metric lines.
        caps.speculative_metrics = true;
    }

    let props = probe_endpoint(client, "props", &[200]).await;
    caps.props = matches!(props, Ok(EndpointStatus::Available));

    caps.model_info = caps.props;

    // llama.cpp has no explicit prefill/decode state field; decode is detected
    // from per-slot decoded-token growth (exact), prefill is estimated.
    caps.exact_prefill_state = false;
    caps.exact_decode_state = caps.slots;

    Ok(caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_status_is_comparable() {
        assert_eq!(EndpointStatus::Available, EndpointStatus::Available);
        assert_ne!(EndpointStatus::Disabled, EndpointStatus::Unavailable);
    }
}
