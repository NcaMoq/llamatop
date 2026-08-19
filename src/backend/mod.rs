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
/// Probed at connect time and re-probed after a reconnect. Endpoints may be
/// individually disabled by the server (e.g. `--no-slots`, no `--metrics`);
/// a missing endpoint must never terminate the application.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub health: bool,
    pub slots: bool,
    pub metrics: bool,
    pub props: bool,
    pub model_info: bool,
    pub speculative_metrics: bool,
    /// The backend exposes a direct prefill/processing signal (exact phase).
    pub exact_prefill_state: bool,
    /// The backend exposes a direct decode signal (e.g. per-slot decoded growth).
    pub exact_decode_state: bool,
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
    async fn snapshot(
        &self,
        capabilities: &BackendCapabilities,
    ) -> Result<BackendSnapshot, BackendError>;
}

pub mod llamacpp;
