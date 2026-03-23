#![deny(unsafe_code)]
//! BIDS variable system for representing experimental data at multiple levels.
//!
//! This crate implements the variable hierarchy used in BIDS statistical modeling,
//! corresponding to PyBIDS' `bids.variables` module. Variables represent columns
//! from BIDS tabular files (events, participants, sessions, scans, physio, etc.)
//! and can exist at different levels of the BIDS hierarchy.
//!
//! # Variable Types
//!
//! - [`SimpleVariable`] — A variable with string/numeric values and an entity
//!   index. Used for participants.tsv, sessions.tsv, and scans.tsv data.
//!   Supports filtering, cloning with replacement, and tabular export.
//!
//! - [`SparseRunVariable`] — An event-based variable with onset, duration, and
//!   amplitude vectors (from `_events.tsv`). Can be converted to dense
//!   representation via [`SparseRunVariable::to_dense()`] using GCD-based
//!   bin sizing and linear interpolation.
//!
//! - [`DenseRunVariable`] — A uniformly-sampled time series (from physio, stim,
//!   or regressors files). Supports resampling to different rates and TR-based
//!   downsampling.
//!
//! # Node Hierarchy
//!
//! Variables are organized into a [`NodeIndex`] that mirrors the BIDS hierarchy:
//!
//! - **Dataset level** — Participant-level variables (age, sex, group)
//! - **Subject level** — Session-level variables
//! - **Session level** — Scan-level variables (acquisition time, etc.)
//! - **Run level** — Event and continuous variables for individual runs
//!
//! # Loading Variables
//!
//! Use [`load_variables()`] to automatically extract all variables from a
//! `BidsLayout`:
//!
//! ```no_run
//! # use bids_layout::BidsLayout;
//! # let layout = BidsLayout::new("/path").unwrap();
//! let index = bids_variables::load_variables(&layout, None, None).unwrap();
//! let run_collections = index.get_run_collections(
//!     &bids_core::entities::StringEntities::new()
//! );
//! ```
//!
//! # Collections
//!
//! [`VariableCollection`] groups multiple `SimpleVariable`s by name, while
//! [`RunVariableCollection`] groups sparse and dense run variables. Collections
//! can be merged across runs using [`merge_collections()`].

pub mod collections;
pub mod io;
pub mod node;
pub mod variables;

pub use collections::{RunVariableCollection, VariableCollection, merge_collections};
pub use io::load_variables;
pub use node::{Node, NodeIndex, RunInfo, RunNode};
pub use variables::{DenseRunVariable, SimpleVariable, SparseRunVariable};
