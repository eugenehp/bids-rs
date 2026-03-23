#![deny(unsafe_code)]
//! BIDS dataset layout indexing and querying.
//!
//! This is the central crate for interacting with BIDS datasets on disk. It
//! provides [`BidsLayout`], the main entry point that indexes a dataset directory
//! into a SQLite database and exposes a fluent query API for finding files.
//!
//! Corresponds to PyBIDS' `BIDSLayout` class.
//!
//! # Indexing
//!
//! When a `BidsLayout` is created, the dataset directory is walked recursively.
//! Each file's path is matched against entity regex patterns from the configuration
//! to extract BIDS entities (subject, session, task, run, suffix, extension, etc.).
//! Files and their entity tags are stored in a SQLite database — either in-memory
//! for fast one-off use, or on disk for persistent caching of large datasets.
//!
//! JSON sidecar metadata is also indexed following the BIDS inheritance principle,
//! and file associations (IntendedFor, events↔bold, bvec/bval↔DWI) are recorded.
//!
//! # Querying
//!
//! The [`GetBuilder`] provides a fluent API for filtering files:
//!
//! ```no_run
//! # use bids_layout::BidsLayout;
//! # let layout = BidsLayout::new("/path/to/dataset").unwrap();
//! let files = layout.get()
//!     .subject("01")
//!     .task("rest")
//!     .suffix("bold")
//!     .extension(".nii.gz")
//!     .collect()
//!     .unwrap();
//! ```
//!
//! Queries support exact matching, multi-value matching (`.filter_any()`),
//! regex matching (`.filter_regex()`), existence checks (`.query_any()`,
//! `.query_none()`), and scope-aware searching across derivatives.
//!
//! # Derivatives
//!
//! Derivative datasets (e.g., fMRIPrep output) can be added via
//! `layout.add_derivatives()`. Queries can then be scoped to raw data only,
//! derivatives only, a specific pipeline, or all datasets.
//!
//! # Persistent Database
//!
//! For large datasets, create a persistent index to avoid re-scanning:
//!
//! ```no_run
//! # use bids_layout::BidsLayout;
//! let layout = BidsLayout::builder("/path/to/dataset")
//!     .database_path("/path/to/index.sqlite")
//!     .build()
//!     .unwrap();
//! ```
//!
//! Subsequent calls with the same database path will load the existing index
//! instead of re-walking the filesystem.

pub mod db;
pub mod get_builder;
pub mod indexer;
pub mod layout;
pub mod query;

pub use get_builder::{GetBuilder, InvalidFilters};
pub use layout::BidsLayout;
pub use query::{Query, QueryFilter, Scope};
