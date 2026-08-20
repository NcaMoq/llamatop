//! Backend-agnostic monitoring state.
//!
//! This module must not depend on ratatui, reqwest, or NVML. It only describes
//! what was observed; it does not perform I/O.

pub mod connection;
pub mod gpu;
pub mod inference;
pub mod server;
pub mod slot;
pub mod snapshot;
pub mod system;

pub use connection::ConnectionState;
pub use gpu::GpuSnapshot;
pub use inference::{Confidence, WorkloadPhase};
pub use server::ServerState;
pub use slot::{SlotPhase, SlotSnapshot};
pub use snapshot::{BackendSnapshot, SpeculativeStats};
pub use system::{ProcessIdentity, ProcessSnapshot, SystemSnapshot};
