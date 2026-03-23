#![deny(unsafe_code)]
//! EEG (Electroencephalography) support for BIDS datasets.
//!
//! This crate provides typed access to all EEG-specific BIDS files and metadata,
//! corresponding to the [BIDS-EEG specification](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html).
//!
//! # Components
//!
//! - [`EegLayout`] — High-level interface for querying EEG files, channels,
//!   electrodes, events, metadata, coordinate systems, and physio data from
//!   a `BidsLayout`.
//! - [`Channel`] / [`ChannelType`] — Typed representation of `_channels.tsv`
//!   entries with support for all BIDS channel types (EEG, EOG, ECG, EMG,
//!   TRIG, MISC, MEGMAG, MEGGRAD, ECOG, SEEG, DBS, etc.).
//! - [`Electrode`] — Electrode positions from `_electrodes.tsv` with optional
//!   3D coordinates, material, and impedance.
//! - [`EegEvent`] — Events from `_events.tsv` with onset, duration, trial type,
//!   value, sample number, and response time.
//! - [`EegMetadata`] — Typed EEG JSON sidecar metadata including sampling
//!   frequency, channel counts, placement scheme, reference, power line
//!   frequency, recording duration, and hardware/software filter descriptions.
//! - [`CoordinateSystem`] — Coordinate system information from `_coordsystem.json`.
//! - [`EdfHeader`] — Minimal EDF/BDF header parser for extracting channel count,
//!   sampling rates, channel labels, and recording duration directly from
//!   EEG data files.
//! - [`EegData`] / [`ReadOptions`] — Read actual signal data from EDF, BDF, and
//!   BrainVision files, with support for channel inclusion/exclusion, time-range
//!   slicing, stim channel detection, and unit conversion. Use [`read_eeg_data`]
//!   for automatic format detection, or [`read_edf`] / [`read_brainvision`] directly.
//! - [`Annotation`] — Time-stamped annotation parsed from EDF+ TAL channels,
//!   BDF status channels, or BrainVision `.vmrk` marker files.
//!
//! # Example
//!
//! ```no_run
//! # use bids_layout::BidsLayout;
//! use bids_eeg::EegLayout;
//!
//! # let layout = BidsLayout::new("/path").unwrap();
//! let eeg = EegLayout::new(&layout);
//! let summary = eeg.summary().unwrap();
//! println!("{}", summary);
//!
//! for f in &eeg.get_eeg_files().unwrap() {
//!     if let Some(channels) = eeg.get_channels(f).unwrap() {
//!         println!("{}: {} channels", f.filename, channels.len());
//!     }
//! }
//! ```

pub mod channels;
pub mod coordsystem;
pub mod csp;
pub mod data;
pub mod eeg_layout;
pub mod electrodes;
pub mod events;
pub mod harmonize;
pub mod metadata;
pub mod pipeline;

pub use channels::{Channel, ChannelType, read_channels_tsv};
pub use coordsystem::CoordinateSystem;
pub use csp::CSP;
pub use data::{
    Annotation, EegData, ReadOptions, read_brainvision, read_brainvision_markers, read_edf,
    read_eeg_data,
};
pub use eeg_layout::{EdfHeader, EegDatasetSummary, EegLayout, PhysioData};
pub use electrodes::{Electrode, read_electrodes_tsv};
pub use events::{EegEvent, read_events_tsv};
pub use harmonize::{ChannelStrategy, HarmonizationPlan, apply_harmonization, plan_harmonization};
pub use metadata::EegMetadata;
pub use pipeline::{Pipeline as EegPipeline, PipelineResult};
