//! llamatop library root.
//!
//! Module boundaries:
//! - `domain`: backend-agnostic monitoring state. Must not depend on ratatui,
//!   reqwest, or NVML.
//! - `backend`: concrete inference server implementations (HTTP + parsing).
//! - `detector`: pure state detection and rate calculation over snapshots.
//! - `collector`: gathers data (backend, GPU, system) without touching the UI.
//! - `output`: human/JSON rendering of snapshots (no HTTP, no terminal TUI).
//! - `app`: TUI application state, events, and the runtime glue.
//! - `ui`: terminal rendering and input (no HTTP, no raw llama.cpp types).

pub mod app;
pub mod backend;
pub mod cli;
pub mod collector;
pub mod config;
pub mod detector;
pub mod display;
pub mod doctor;
pub mod domain;
pub mod endpoint;
pub mod error;
pub mod logging;
pub mod output;
pub mod snapshot;
pub mod ui;
