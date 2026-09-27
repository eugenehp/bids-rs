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
/// Brings in the everyday surface of the whole workspace: layout and querying,
/// files and metadata, variables, statistical modeling, reports, signal
/// filtering, validation, and the modality readers.
///
/// ```
/// use bids::prelude::*;
/// ```
///
/// Each sub-crate also has its own narrower prelude — `bids::layout::prelude`,
/// `bids::modeling::prelude`, and so on — when the umbrella import is more
/// than you need.
pub mod prelude {
    // Core: files, entities, metadata, errors
    pub use bids_core::error::{BidsError, Result};
    pub use bids_core::options;
    pub use bids_core::timeseries::TimeSeries;
    pub use bids_core::utils::{collect_associated_files, convert_json_keys, matches_entities};
    pub use bids_core::{
        BidsFile, BidsMetadata, Config, CopyMode, DatasetDescription, Entities, Entity,
        EntityValue, FileType, PaddedInt, StringEntities,
    };

    // I/O
    pub use bids_io::gradient::GradientTable;
    pub use bids_io::json::{merge_json_sidecars, read_json, read_json_sidecar};
    pub use bids_io::path_builder::build_path;
    pub use bids_io::tsv::{TsvRow, read_tsv, read_tsv_gz};
    pub use bids_io::writer::{ConflictStrategy, write_to_file};

    // Layout and querying
    pub use bids_layout::{
        BidsLayout, GetBuilder, InvalidFilters, LayoutBuilder, Query, QueryFilter, Scope,
    };

    // Variables
    pub use bids_variables::{
        DenseRunVariable, LayoutVariables, NodeIndex, RunInfo, RunVariableCollection,
        SimpleVariable, SparseRunVariable, VariableCollection, VariableTiming, VariableType,
        load_variables,
    };

    // Statistical modeling
    pub use bids_modeling::{
        ContrastInfo, GlmSpec, GlmmSpec, HrfModel, NodeReport, Prior, StatsModelsGraph,
        StatsModelsNode, StatsModelsNodeOutput, Term, TransformError, TransformSpec, VarComp,
        apply_transformations, auto_model, compute_regressor, compute_vif, generate_report,
        glover_hrf, node_report, spm_hrf,
    };

    // Formulae
    pub use bids_formula::{Formula, build_design_matrix, parse_formula};

    // Reports
    pub use bids_reports::BidsReport;

    // Validation and schema
    pub use bids_schema::version::{
        BidsVersion, Compatibility, MIN_COMPATIBLE_VERSION, SUPPORTED_BIDS_VERSION,
    };
    pub use bids_validate::{ValidationIssue, ValidationResult, validate_dataset, validate_root};

    // Signal processing
    pub use bids_filter::{
        butter_bandpass, butter_highpass, butter_lowpass, filtfilt, notch_filter, resample,
    };

    // Imaging and modality readers
    pub use bids_eeg::{EegData, EegLayout, EegMetadata, ReadOptions};
    pub use bids_nifti::{GeometrySource, NiftiHeader, NiftiImage};

    // Datasets
    pub use bids_dataset::{Aggregator, DatasetFilter, OpenNeuro, Split};
}

#[cfg(test)]
mod tests {
    /// The prelude must actually import — a broken re-export here breaks every
    /// downstream `use bids::prelude::*`.
    #[test]
    fn prelude_imports() {
        use crate::prelude::*;

        let _: Option<BidsLayout> = None;
        let _: Option<BidsFile> = None;
        let _: Option<VariableCollection> = None;
        let _: Option<StatsModelsGraph> = None;
        let _: Option<GlmmSpec> = None;
        let _: Option<BidsReport> = None;
        let _: Option<ValidationResult> = None;
        let _: Option<NiftiHeader> = None;
        let _: Query = Query::Any;
        let _: Scope = Scope::All;
        // Use the version rather than assert a constant, so the import is
        // exercised without a tautology.
        assert_eq!(
            SUPPORTED_BIDS_VERSION.to_string(),
            format!(
                "{}.{}.{}",
                SUPPORTED_BIDS_VERSION.major,
                SUPPORTED_BIDS_VERSION.minor,
                SUPPORTED_BIDS_VERSION.patch
            )
        );
    }
}
