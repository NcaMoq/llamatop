//! Server-level lifecycle state (loading, ready, sleeping).

use serde::{Deserialize, Serialize};

/// Lifecycle state of the inference server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerState {
    Unknown,
    Loading,
    Ready,
    Sleeping,
    Unavailable,
    Error,
}

impl ServerState {
    pub fn as_str(&self) -> &'static str {
        match self {
            ServerState::Unknown => "UNKNOWN",
            ServerState::Loading => "LOADING",
            ServerState::Ready => "READY",
            ServerState::Sleeping => "SLEEPING",
            ServerState::Unavailable => "UNAVAILABLE",
            ServerState::Error => "ERROR",
        }
    }
}
