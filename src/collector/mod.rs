//! Data collectors for the TUI.
//!
//! A collector owns its data source (the inference backend) and produces
//! `AppEvent`s. Collectors never render, never depend on the UI, and never
//! share mutable state with it.

pub mod backend;
