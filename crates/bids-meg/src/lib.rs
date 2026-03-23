#![deny(unsafe_code)]
//! Magnetoencephalography (MEG) support for BIDS datasets.
//!
//! Provides typed access to MEG-specific files, channels, events, headshape data,
//! coordinate systems, and MEG metadata (sampling frequency, channel counts,
//! dewar position, digitized landmarks).
//!
//! ## Feature flags
//!
//! - **`fiff`** — Enable FIFF data reading for Elekta/Neuromag MEG files (`.fif`).
//!   Adds the `data` module with `MegData` and `read_fiff`.
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetoencephalography.html>

pub mod ctf;
#[cfg(feature = "fiff")]
pub mod data;
pub mod headshape;
pub mod metadata;

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_layout::BidsLayout;
pub use ctf::{CtfHeader, read_ctf_header};
#[cfg(feature = "fiff")]
pub use data::{MegData, read_fiff};
pub use headshape::{DigPoint, PointKind, read_headshape_pos};
pub use metadata::MegMetadata;

pub struct MegLayout<'a> {
    layout: &'a BidsLayout,
}
impl<'a> MegLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self {
        Self { layout }
    }
    pub fn get_meg_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("meg").collect()
    }
    pub fn get_meg_files_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("meg").subject(s).collect()
    }
    pub fn get_meg_files_for_task(&self, t: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("meg").task(t).collect()
    }
    pub fn get_channels(&self, f: &BidsFile) -> Result<Option<Vec<bids_eeg::Channel>>> {
        bids_core::try_read_companion(&f.companion("channels", "tsv"), bids_eeg::read_channels_tsv)
    }
    pub fn get_events(&self, f: &BidsFile) -> Result<Option<Vec<bids_eeg::EegEvent>>> {
        bids_core::try_read_companion(&f.companion("events", "tsv"), bids_eeg::read_events_tsv)
    }
    /// Get the headshape file path for a MEG recording.
    pub fn get_headshape_path(&self, f: &BidsFile) -> Option<std::path::PathBuf> {
        let p = f.companion("headshape", "pos");
        if p.exists() { Some(p) } else { None }
    }

    /// Read and parse headshape digitization points from the companion `.pos` file.
    pub fn get_headshape(&self, f: &BidsFile) -> Result<Option<Vec<DigPoint>>> {
        let p = f.companion("headshape", "pos");
        if p.exists() {
            Ok(Some(read_headshape_pos(&p)?))
        } else {
            Ok(None)
        }
    }
    pub fn get_coordsystem(&self, f: &BidsFile) -> Result<Option<bids_eeg::CoordinateSystem>> {
        let p = f.companion("coordsystem", "json");
        if p.exists() {
            Ok(Some(bids_eeg::CoordinateSystem::from_file(&p)?))
        } else {
            Ok(None)
        }
    }
    pub fn get_metadata(&self, f: &BidsFile) -> Result<Option<MegMetadata>> {
        Ok(self.layout.get_metadata(&f.path)?.deserialize_as())
    }

    /// Read raw MEG signal data from a FIFF file.
    ///
    /// Requires the `fiff` feature flag.
    #[cfg(feature = "fiff")]
    pub fn read_data(&self, f: &BidsFile) -> Result<MegData> {
        read_fiff(&f.path)
    }
    pub fn get_all_channels_files(&self) -> Result<Vec<BidsFile>> {
        self.layout
            .get()
            .suffix("channels")
            .datatype("meg")
            .extension("tsv")
            .collect()
    }
    pub fn get_all_events_files(&self) -> Result<Vec<BidsFile>> {
        self.layout
            .get()
            .suffix("events")
            .datatype("meg")
            .extension("tsv")
            .collect()
    }
    pub fn get_meg_subjects(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("meg").return_unique("subject")
    }
    pub fn get_meg_tasks(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("meg").return_unique("task")
    }
    pub fn summary(&self) -> Result<MegSummary> {
        let files = self.get_meg_files()?;
        let subjects = self.get_meg_subjects()?;
        let tasks = self.get_meg_tasks()?;
        let sf = files
            .first()
            .and_then(|f| self.get_metadata(f).ok().flatten())
            .map(|m| m.sampling_frequency);
        let ch = files
            .first()
            .and_then(|f| self.get_channels(f).ok().flatten())
            .map(|c| c.len());
        Ok(MegSummary {
            n_subjects: subjects.len(),
            n_recordings: files.len(),
            subjects,
            tasks,
            sampling_frequency: sf,
            channel_count: ch,
        })
    }
}
/// Summary statistics for MEG data in a BIDS dataset.
#[derive(Debug, Clone)]
pub struct MegSummary {
    pub n_subjects: usize,
    pub n_recordings: usize,
    pub subjects: Vec<String>,
    pub tasks: Vec<String>,
    pub sampling_frequency: Option<f64>,
    pub channel_count: Option<usize>,
}
impl std::fmt::Display for MegSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "MEG Summary: {} subjects, {} recordings, tasks: {:?}",
            self.n_subjects, self.n_recordings, self.tasks
        )?;
        if let Some(sf) = self.sampling_frequency {
            writeln!(f, "  Sampling: {sf} Hz")?;
        }
        if let Some(ch) = self.channel_count {
            writeln!(f, "  Channels: {ch}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_meg_metadata() {
        let json = r#"{"SamplingFrequency":2400,"MEGChannelCount":274,"MEGREFChannelCount":26,"DewarPosition":"Upright"}"#;
        let md: MegMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.sampling_frequency, 2400.0);
        assert_eq!(md.meg_channel_count, Some(274));
        assert_eq!(md.dewar_position.as_deref(), Some("Upright"));
    }
}
