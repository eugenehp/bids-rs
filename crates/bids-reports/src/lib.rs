#![deny(unsafe_code)]
//! Auto-generation of publication-quality methods sections from BIDS datasets.
//!
//! This crate analyzes a BIDS dataset's structure and metadata to produce
//! human-readable text describing the data acquisition parameters, suitable
//! for inclusion in a research paper's methods section. Corresponds to
//! PyBIDS' `bids.reports` module.
//!
//! # Supported Datatypes
//!
//! Currently generates descriptions for:
//! - Functional MRI (task-based and resting-state)
//! - Structural MRI (T1w, T2w, FLAIR)
//! - Diffusion MRI (DWI)
//! - Field maps (phase-difference, two-phase, EPI)
//! - EEG
//!
//! # Example
//!
//! ```no_run
//! # use bids_layout::BidsLayout;
//! use bids_reports::BidsReport;
//!
//! # let layout = BidsLayout::new("/path").unwrap();
//! let report = BidsReport::new(&layout);
//! println!("{}", report.generate().unwrap());
//! ```
//!
//! # Warning
//!
//! Automatic report generation is experimental. Always verify generated text
//! before including it in a publication.

pub mod report;
pub mod parameters;
pub mod parsing;

pub use report::BidsReport;
