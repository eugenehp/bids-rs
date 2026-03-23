#![deny(unsafe_code)]
//! File I/O utilities for BIDS datasets.
//!
//! This crate handles reading and writing the file formats used in BIDS:
//!
//! - **TSV** — Tab-separated value files (`.tsv`, `.tsv.gz`) used for events,
//!   channels, electrodes, participants, sessions, and scans tables. See [`tsv`].
//! - **JSON sidecars** — Metadata files that accompany data files, following the
//!   BIDS inheritance principle where more-specific sidecars override less-specific
//!   ones. See [`json`].
//! - **Path building** — Construct BIDS-compliant file paths from entity key-value
//!   pairs using configurable patterns with optional sections (`[/ses-{session}]`),
//!   value constraints (`{suffix<T1w|T2w>}`), and defaults (`{datatype|anat}`).
//!   See [`path_builder`].
//! - **File writing** — Write data to files with configurable conflict resolution
//!   strategies (fail, skip, overwrite, append) and support for symlinks.
//!   See [`writer`].
//!
//! # BIDS Inheritance Principle
//!
//! JSON sidecar files apply to all data files in the same directory and below.
//! When multiple sidecars match a data file, they are merged with the most
//! specific file (closest to the data) taking precedence:
//!
//! ```text
//! dataset/
//!   task-rest_eeg.json          ← least specific (applies to all rest EEG)
//!   sub-01/
//!     sub-01_task-rest_eeg.json ← most specific (applies only to sub-01)
//!     eeg/
//!       sub-01_task-rest_eeg.edf
//! ```

pub mod gradient;
pub mod json;
pub mod path_builder;
pub mod tsv;
pub mod writer;

pub use gradient::{GradientTable, read_bvals, read_bvecs};
pub use json::{find_sidecars, merge_json_sidecars, read_json, read_json_sidecar};
pub use path_builder::build_path;
pub use tsv::{TsvRow, read_tsv, read_tsv_gz};
pub use writer::{ConflictStrategy, write_to_file};
