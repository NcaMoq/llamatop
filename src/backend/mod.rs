//! Inference backend abstraction.
//!
//! A backend knows how to talk to one specific inference server implementation
//! and turns its responses into normalized domain snapshots. The UI and the
//! detector depend only on this trait and the domain types — never on raw
//! HTTP responses or llama.cpp-specific JSON.

use async_trait::async_trait;

use crate::domain::BackendSnapshot;
use crate::error::BackendError;

/// Which API endpoints a backend instance supports.
///
/// Probed at connect time, re-probed after a reconnect, and re-observed on
/// every snapshot so that temporary failures are visible and recover
/// automatically. Endpoints may be individually disabled by the server
/// (e.g. `--no-slots`, no `--metrics`); a missing endpoint must never
/// terminate the application.
///
/// Each optional endpoint carries an [`EndpointAvailability`] observation,
/// not a `bool`: "the server answered 501" (unsupported) is a different fact
/// from "the server timed out" (temporarily unavailable) and from "never
/// probed" (unknown).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub health: EndpointAvailability,
    pub slots: EndpointAvailability,
    pub metrics: EndpointAvailability,
    pub props: EndpointAvailability,
    pub model_info: bool,
    pub speculative_metrics: bool,
    /// The backend exposes a direct prefill/processing signal (exact phase).
    pub exact_prefill_state: bool,
    /// The backend exposes a direct decode signal (e.g. per-slot decoded growth).
    pub exact_decode_state: bool,
}

/// The observed availability of one endpoint.
///
/// These are *observations*, not capabilities: `Available` means "the last
/// observation showed a usable answer", `Unknown` means "we have no
/// observation yet". The collector re-observes so states recover without a
/// manual reconnect.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EndpointAvailability {
    /// No observation yet (never probed, or the probe could not run because
    /// the server was unreachable).
    #[default]
    Unknown,
    /// The endpoint answered with a usable, expected response.
    Available,
    /// The server answered and this endpoint does not exist on it (404/405,
    /// or 501 where the server reports "disabled by configuration").
    /// Re-validated only slowly; a manual reconnect re-probes it.
    Unsupported,
    /// A transport-level or server error (timeout, connection refused, DNS,
    /// 5xx). Retried automatically on the next observation.
    TemporarilyUnavailable,
    /// The server rejected the credentials (401/403). Retried only slowly;
    /// a manual reconnect re-probes it.
    AuthenticationFailed,
    /// The endpoint answered 2xx but the body was not the expected payload.
    /// Retried on the next observation.
    ParseFailed,
}

impl EndpointAvailability {
    /// True only for `Available`: the only state in which the endpoint's
    /// data can be trusted as current.
    pub fn is_available(&self) -> bool {
        matches!(self, EndpointAvailability::Available)
    }

    /// True when the endpoint should be (re-)fetched now: it is known
    /// available, or there is no usable observation yet (unknown, temporary
    /// failure, or a stale parse failure). `Unsupported` and
    /// `AuthenticationFailed` are not fetched on the regular cycle.
    pub fn needs_observation(&self) -> bool {
        !matches!(
            self,
            EndpointAvailability::Unsupported | EndpointAvailability::AuthenticationFailed
        )
    }

    /// Short, stable label for UI display (no color-only distinction).
    pub fn as_str(&self) -> &'static str {
        match self {
            EndpointAvailability::Unknown => "unknown",
            EndpointAvailability::Available => "available",
            EndpointAvailability::Unsupported => "unsupported",
            EndpointAvailability::TemporarilyUnavailable => "temporarily unavailable",
            EndpointAvailability::AuthenticationFailed => "authentication failed",
            EndpointAvailability::ParseFailed => "parse failed",
        }
    }
}

/// Which endpoints are due for a fetch in this cycle.
///
/// The scheduling decision lives in the collector (per-endpoint intervals);
/// the backend only fetches what is marked due here and keeps its last
/// successful observation for the endpoints that are not.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EndpointDue {
    pub health: bool,
    pub slots: bool,
    pub metrics: bool,
    pub props: bool,
}

impl EndpointDue {
    /// Every endpoint due (initial cycle, manual reconnect).
    pub const ALL: EndpointDue =
        EndpointDue { health: true, slots: true, metrics: true, props: true };

    /// Nothing due (defensive; a collector cycle always has at least one).
    pub const NONE: EndpointDue =
        EndpointDue { health: false, slots: false, metrics: false, props: false };
}

/// Health probe result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendHealth {
    pub server: crate::domain::ServerState,
    /// Short reason when the server is not ready (never a full body).
    pub detail: Option<String>,
}

/// A single inference backend.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Human-readable backend name (e.g. "llama.cpp").
    fn name(&self) -> &'static str;

    /// Probe which endpoints are available. Must not fail hard on a missing
    /// endpoint: encode availability in `BackendCapabilities` instead.
    async fn probe_capabilities(&self) -> Result<BackendCapabilities, BackendError>;

    /// Check server reachability and lifecycle state.
    async fn health(&self) -> Result<BackendHealth, BackendError>;

    /// Capture a full normalized snapshot of the backend.
    ///
    /// Missing optional endpoints degrade the snapshot (fields stay `None`);
    /// they do not produce an error for the whole snapshot.
    ///
    /// The capability observations are *updated* by this call: every
    /// endpoint that is fetched (see [`EndpointAvailability::needs_observation`])
    /// is re-observed, so temporary failures and parse errors recover on a
    /// later snapshot without a manual reconnect. The caller keeps the same
    /// `BackendCapabilities` value across calls and passes it in mutably.
    ///
    /// Takes `&mut self` because the backend caches the last successful
    /// observation per endpoint (shared with [`InferenceBackend::snapshot_due`]).
    async fn snapshot(
        &mut self,
        capabilities: &mut BackendCapabilities,
    ) -> Result<BackendSnapshot, BackendError>;

    /// Capture a normalized snapshot fetching only the endpoints marked in
    /// `due`.
    ///
    /// Contract:
    /// - a due endpoint is fetched and its capability observation updated,
    ///   exactly as in `snapshot`; a fetch failure degrades that endpoint's
    ///   data to missing for this snapshot (never to a guessed value);
    /// - an endpoint that is not due keeps its last successful observation
    ///   (cached by the backend); its fields in the returned snapshot carry
    ///   the previous values and its observation is left unchanged.
    ///
    /// The caller (the TUI collector) knows which endpoints were due and
    /// uses that to decide which fields are fresh.
    async fn snapshot_due(
        &mut self,
        capabilities: &mut BackendCapabilities,
        due: EndpointDue,
    ) -> Result<BackendSnapshot, BackendError>;
}

pub mod llamacpp;
