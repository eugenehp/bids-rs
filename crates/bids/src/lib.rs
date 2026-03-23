#![deny(unsafe_code)]
//! Comprehensive Rust tools for BIDS (Brain Imaging Data Structure) datasets.
//!
//! This is the umbrella crate that re-exports all `bids-*` sub-crates, providing
//! a single dependency for full BIDS dataset support. It is the Rust equivalent
//! of the [PyBIDS](https://github.com/bids-standard/pybids) Python package.
//!
//! # Quick Start
//!
//! ```no_run
//! use bids::{BidsLayout, BidsFile};
//!
//! let layout = BidsLayout::new("/path/to/bids/dataset").unwrap();
//! let files = layout.get()
//!     .suffix("bold")
//!     .extension(".nii.gz")
//!     .subject("01")
//!     .collect()
//!     .unwrap();
//!
//! for f in &files {
//!     println!("{}: {:?}", f.filename, f.entities);
//! }
//! ```
//!
//! # Crate Organization
//!
//! - **Core** ([`core`]) — Fundamental types: files, entities, metadata, config, errors
//! - **I/O** ([`io`]) — TSV/JSON reading, path building, file writing
//! - **Layout** ([`layout`]) — Dataset indexing and fluent query API
//! - **Variables** ([`variables`]) — BIDS variable system for statistical modeling
//! - **Modeling** ([`modeling`]) — BIDS-StatsModels, HRF functions, transformations
//! - **Reports** ([`reports`]) — Auto-generated methods sections
//! - **Domain crates** — Modality-specific support (EEG, MEG, PET, etc.)
//! - **Infrastructure** — NIfTI parsing, signal filtering, formula parsing, schema validation
//!
//! # Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `ndarray` | Enable `ndarray` integration for EEG (`Array2`) and NIfTI (`ArrayD`) |
//! | `safetensors` | Enable safetensors export for ML frameworks |
//! | `mmap` | Enable memory-mapped NIfTI access via `memmap2` |
//! | `arrow` | Enable Apache Arrow / Parquet export for dataset manifests |

// Core
pub use bids_core as core;
pub use bids_core::error::Result;
pub use bids_core::utils::{collect_associated_files, convert_json_keys, matches_entities};
pub use bids_core::{
    BidsError, BidsFile, BidsMetadata, Config, CopyMode, DatasetDescription, Entities, Entity,
    EntityValue, PaddedInt, StringEntities,
};

// IO
pub use bids_io as io;
pub use bids_io::gradient::GradientTable;

// Layout
pub use bids_layout as layout;
pub use bids_layout::{BidsLayout, InvalidFilters, Query, QueryFilter, Scope};

// Variables
pub use bids_variables as variables;

// Validation
pub use bids_validate as validate;

// Derivatives
pub use bids_derive as derive;

// Modeling
pub use bids_modeling as modeling;
pub use bids_modeling::{ContrastInfo, HrfModel, StatsModelsGraph};

// Reports
pub use bids_reports as reports;
pub use bids_reports::BidsReport;

// Domain crates
pub use bids_beh as beh;
pub use bids_eeg as eeg;
pub use bids_ieeg as ieeg;
pub use bids_meg as meg;
pub use bids_micr as micr;
pub use bids_motion as motion;
pub use bids_mrs as mrs;
pub use bids_nirs as nirs;
pub use bids_perf as perf;
pub use bids_pet as pet;

// Dataset downloading and aggregation
pub use bids_dataset as dataset;

// Common trait for time-series data
pub use bids_core::timeseries::TimeSeries;

// Infrastructure crates
pub use bids_filter as filter;
pub use bids_formula as formula;
pub use bids_inflect as inflect;
pub use bids_nifti as nifti;
pub use bids_schema as schema;

// Version tracking
pub use bids_schema::version::{
    BidsVersion, Compatibility, MIN_COMPATIBLE_VERSION, SUPPORTED_BIDS_VERSION,
};

/// Convenience prelude importing the most commonly used types.
///
/// ```
/// use bids::prelude::*;
/// ```
pub mod prelude {
    pub use bids_core::error::Result;
    pub use bids_core::timeseries::TimeSeries;
    pub use bids_core::{BidsError, BidsFile, BidsMetadata, Config, DatasetDescription};
    pub use bids_core::{Entities, Entity, EntityValue};
    pub use bids_dataset::{Aggregator, DatasetFilter, OpenNeuro, Split};
    pub use bids_eeg::{EegData, EegLayout, ReadOptions};
    pub use bids_layout::{BidsLayout, InvalidFilters, Query, QueryFilter, Scope};
    pub use bids_nifti::NiftiImage;
    pub use bids_reports::BidsReport;
}
