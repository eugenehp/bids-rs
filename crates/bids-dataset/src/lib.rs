#![deny(unsafe_code)]
//! Download, filter, match, and aggregate BIDS datasets for ML pipelines.
//!
//! This crate provides a high-level interface for:
//!
//! 1. **Searching** the OpenNeuro catalog by modality, species, and keywords
//! 2. **Downloading** datasets (or subsets) from the OpenNeuro S3 bucket with
//!    resume support and file-size verification
//! 3. **Filtering** local BIDS files by modality, subject, session, task,
//!    extension, and custom predicates
//! 4. **Aggregating** multiple datasets into a unified file collection with
//!    harmonized entity namespaces
//! 5. **Exporting** ML-ready manifests (CSV file lists with metadata) and
//!    train/val/test splits that data loaders can consume directly
//!
//! # Example
//!
//! ```no_run
//! use bids_dataset::{OpenNeuro, DatasetFilter, Aggregator, Split};
//!
//! let on = OpenNeuro::new();
//!
//! // Search for EEG datasets about resting state
//! let hits = on.search().modality("eeg").keyword("rest").limit(5).execute().unwrap();
//! println!("Found {} datasets", hits.len());
//!
//! // Download only metadata + EEG files from the first hit
//! let cache = std::path::Path::new("/tmp/bids-cache");
//! on.download_dataset(&hits[0].id, cache, Some(|f: &bids_dataset::RemoteFile| {
//!     f.path.ends_with(".json") || f.path.ends_with(".tsv")
//!         || f.path.ends_with(".edf") || f.path.ends_with(".bdf")
//! })).unwrap();
//!
//! // Build an aggregated manifest from local datasets
//! let mut agg = Aggregator::new();
//! agg.add_dataset(&cache.join(&hits[0].id),
//!     DatasetFilter::new().modality("eeg").extension(".edf")
//! ).unwrap();
//! agg.export_manifest("/tmp/manifest.csv").unwrap();
//! agg.export_split("/tmp/splits", Split::ratio(0.8, 0.1, 0.1)).unwrap();
//! ```

pub mod cache;
pub mod http;
pub mod openneuro;
pub mod ratelimit;
pub mod filter;
pub mod aggregate;
pub mod split;
pub mod ml;
pub mod paradigm;
pub mod evaluation;
pub mod benchmark;

pub use cache::Cache;
pub use openneuro::{OpenNeuro, DatasetInfo, RemoteFile, SearchBuilder, DownloadReport};
pub use ratelimit::RateLimitConfig;
pub use filter::DatasetFilter;
pub use aggregate::{Aggregator, FileEntry};
pub use split::Split;
pub use ml::{EpochSpec, Sample, KFold, StratifiedSplit, DatasetIter};
pub use paradigm::{Paradigm, ParadigmType, EvalStrategy};
pub use evaluation::{SampleMeta, SplitIndices, within_session_splits, cross_session_splits, cross_subject_splits, cross_subject_kfold_splits};
pub use benchmark::{BenchmarkResult, BenchmarkResults};

/// Error type for this crate — re-exports [`bids_core::BidsError`].
pub use bids_core::BidsError as Error;

/// Result type for this crate.
pub type Result<T> = std::result::Result<T, Error>;
