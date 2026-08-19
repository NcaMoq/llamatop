//! llamatop library root.
//!
//! Module boundaries:
//! - `domain`: backend-agnostic monitoring state. Must not depend on ratatui,
//!   reqwest, or NVML.
//! - `backend`: concrete inference server implementations (HTTP + parsing).
//! - `collector`: gathers data (backend, GPU, system) without touching the UI.
//! - `detector`: pure state detection and rate calculation over snapshots.
//! - `output`: human/JSON rendering of snapshots (no HTTP, no terminal TUI).
//! - `ui`: terminal rendering (added with the TUI phase).

pub mod backend;
pub mod cli;
pub mod config;
pub mod detector;
pub mod display;
pub mod doctor;
pub mod domain;
pub mod error;
pub mod logging;
pub mod output;
pub mod snapshot;
