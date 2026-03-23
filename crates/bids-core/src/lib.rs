#![deny(unsafe_code)]
//! Core types and abstractions for working with BIDS datasets.
//!
//! This crate provides the foundational building blocks used by all other `bids-*` crates:
//!
//! - [`Entity`] and [`EntityValue`] — BIDS entity definitions (subject, session, task, run, …)
//!   with regex-based extraction from file paths and typed value coercion.
//! - [`BidsFile`] — Representation of a single file in a BIDS dataset, with automatic
//!   file type detection, entity extraction, suffix/extension parsing, and companion
//!   file lookup.
//! - [`BidsMetadata`] — Ordered key-value metadata dictionary backed by `IndexMap`,
//!   supporting typed access (`get_f64`, `get_str`, `get_array`, …) and deserialization
//!   into arbitrary structs.
//! - [`DatasetDescription`] — Typed representation of `dataset_description.json` with
//!   validation, derivative detection, and legacy field migration.
//! - [`Config`] — Layout configuration defining entity patterns and path templates,
//!   loadable from built-in configs (`bids`, `derivatives`) or custom JSON files.
//! - [`PaddedInt`] — Zero-padded integer type that preserves formatting (e.g., `"02"`)
//!   while comparing numerically.
//! - [`BidsError`] — Comprehensive error enum covering I/O, JSON, validation, entity,
//!   filter, database, and path-building errors.
//!
//! # BIDS Entities
//!
//! BIDS filenames encode metadata as key-value pairs separated by underscores:
//!
//! ```text
//! sub-01_ses-02_task-rest_run-01_bold.nii.gz
//! ^^^^^^ ^^^^^^ ^^^^^^^^^ ^^^^^^ ^^^^
//! subject session  task     run   suffix
//! ```
//!
//! The [`mod@entities`] module defines the canonical entity ordering and provides
//! functions to parse entities from paths, sort them, and coerce values to
//! the correct types (string, padded integer, float, boolean).
//!
//! # Example
//!
//! ```
//! use bids_core::{BidsFile, Config, DatasetDescription};
//!
//! // Parse entities from a filename
//! let config = Config::bids();
//! let entities = bids_core::entities::parse_file_entities(
//!     "sub-01/eeg/sub-01_task-rest_eeg.edf",
//!     &config.entities,
//! );
//! assert_eq!(entities.get("subject").unwrap().as_str_lossy(), "01");
//! assert_eq!(entities.get("task").unwrap().as_str_lossy(), "rest");
//! ```

pub mod entities;
pub mod error;
pub mod file;
pub mod genetic;
pub mod hed;
pub mod metadata;
pub mod config;
pub mod dataset_description;
pub mod padded_int;
pub mod timeseries;
pub mod utils;

pub use entities::{Entity, EntityValue, Entities, StringEntities};
pub use error::{BidsError, Result};
pub use file::{BidsFile, FileType, CopyMode};
pub use metadata::BidsMetadata;
pub use config::Config;
pub use dataset_description::DatasetDescription;
pub use padded_int::PaddedInt;
pub use utils::{matches_entities, collect_associated_files, convert_json_keys};

/// Helper for modality crates: try to read from a companion file path.
///
/// Returns `Ok(None)` if the file doesn't exist, `Ok(Some(T))` if it does
/// and is readable, or `Err(...)` on I/O errors.
pub fn try_read_companion<T, F>(path: &std::path::Path, reader: F) -> Result<Option<T>>
where
    F: FnOnce(&std::path::Path) -> Result<T>,
{
    if path.exists() { Ok(Some(reader(path)?)) } else { Ok(None) }
}

/// Convenience macro for building an [`Entities`] map inline.
///
/// # Example
///
/// ```
/// use bids_core::entities;
///
/// let ents = entities! {
///     "subject" => "01",
///     "task" => "rest",
///     "suffix" => "eeg",
///     "extension" => ".edf",
/// };
/// assert_eq!(ents.get("subject").unwrap().as_str_lossy(), "01");
/// assert_eq!(ents.len(), 4);
/// ```
#[macro_export]
macro_rules! entities {
    ($($key:expr => $val:expr),* $(,)?) => {{
        let mut map = $crate::Entities::new();
        $(
            map.insert($key.to_string(), $crate::EntityValue::from($val));
        )*
        map
    }};
}
