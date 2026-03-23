#![deny(unsafe_code)]
//! Functional Near-Infrared Spectroscopy (fNIRS) support for BIDS datasets.
//!
//! Provides access to NIRS data files, channels, optode positions (source and
//! detector), events, coordinate systems, and NIRS-specific metadata.
//!
//! ## Feature flags
//!
//! - **`snirf`** — Enable SNIRF (HDF5) data reading. Requires `libhdf5` on the system.
//!   Adds the `data` module with `NirsData` and `read_snirf`.
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/near-infrared-spectroscopy.html>

#[cfg(feature = "snirf")]
pub mod data;

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_core::metadata::BidsMetadata;
use bids_io::tsv::read_tsv;
use bids_layout::BidsLayout;
use serde::{Deserialize, Serialize};

#[cfg(feature = "snirf")]
pub use data::{NirsData, read_snirf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct NirsMetadata {
    #[serde(default)]
    pub sampling_frequency: Option<f64>,
    #[serde(rename = "NIRSChannelCount", default)]
    pub nirs_channel_count: Option<u32>,
    #[serde(default)]
    pub task_name: Option<String>,
    #[serde(default)]
    pub manufacturer: Option<String>,
    #[serde(rename = "NIRSSourceOptodeCount", default)]
    pub source_optode_count: Option<u32>,
    #[serde(rename = "NIRSDetectorOptodeCount", default)]
    pub detector_optode_count: Option<u32>,
}
impl NirsMetadata {
    pub fn from_metadata(md: &BidsMetadata) -> Option<Self> {
        md.deserialize_as()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Optode {
    pub name: String,
    pub optode_type: String,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
}
impl Optode {
    pub fn has_position(&self) -> bool {
        self.x.is_some() && self.y.is_some() && self.z.is_some()
    }
}

pub fn read_optodes_tsv(path: &std::path::Path) -> Result<Vec<Optode>> {
    let rows = read_tsv(path)?;
    Ok(rows
        .iter()
        .map(|r| Optode {
            name: r.get("name").cloned().unwrap_or_default(),
            optode_type: r.get("type").cloned().unwrap_or_default(),
            x: r.get("x").and_then(|v| v.parse().ok()),
            y: r.get("y").and_then(|v| v.parse().ok()),
            z: r.get("z").and_then(|v| v.parse().ok()),
        })
        .collect())
}

pub struct NirsLayout<'a> {
    layout: &'a BidsLayout,
}
impl<'a> NirsLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self {
        Self { layout }
    }
    pub fn get_nirs_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("nirs").collect()
    }
    pub fn get_nirs_files_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("nirs").subject(s).collect()
    }
    pub fn get_nirs_files_for_task(&self, t: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("nirs").task(t).collect()
    }
    pub fn get_channels(&self, f: &BidsFile) -> Result<Option<Vec<bids_eeg::Channel>>> {
        let p = f.companion("channels", "tsv");
        if p.exists() {
            Ok(Some(bids_eeg::read_channels_tsv(&p)?))
        } else {
            Ok(None)
        }
    }
    pub fn get_optodes(&self, f: &BidsFile) -> Result<Option<Vec<Optode>>> {
        let p = f.companion("optodes", "tsv");
        if p.exists() {
            Ok(Some(read_optodes_tsv(&p)?))
        } else {
            Ok(None)
        }
    }
    pub fn get_events(&self, f: &BidsFile) -> Result<Option<Vec<bids_eeg::EegEvent>>> {
        let p = f.companion("events", "tsv");
        if p.exists() {
            Ok(Some(bids_eeg::read_events_tsv(&p)?))
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
    pub fn get_metadata(&self, f: &BidsFile) -> Result<Option<NirsMetadata>> {
        Ok(self.layout.get_metadata(&f.path)?.deserialize_as())
    }

    /// Read NIRS signal data from a SNIRF (.snirf) file.
    ///
    /// Requires the `snirf` feature flag.
    #[cfg(feature = "snirf")]
    pub fn read_data(&self, f: &BidsFile) -> Result<NirsData> {
        read_snirf(&f.path)
    }
    pub fn get_all_channels_files(&self) -> Result<Vec<BidsFile>> {
        self.layout
            .get()
            .suffix("channels")
            .datatype("nirs")
            .extension("tsv")
            .collect()
    }
    pub fn get_all_optodes_files(&self) -> Result<Vec<BidsFile>> {
        self.layout
            .get()
            .suffix("optodes")
            .datatype("nirs")
            .extension("tsv")
            .collect()
    }
    pub fn get_nirs_subjects(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("nirs").return_unique("subject")
    }
    pub fn get_nirs_tasks(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("nirs").return_unique("task")
    }
    pub fn summary(&self) -> Result<NirsSummary> {
        let files = self.get_nirs_files()?;
        let subjects = self.get_nirs_subjects()?;
        let tasks = self.get_nirs_tasks()?;
        let md = files
            .first()
            .and_then(|f| self.get_metadata(f).ok().flatten());
        let sf = md.as_ref().and_then(|m| m.sampling_frequency);
        let n_ch = md.as_ref().and_then(|m| m.nirs_channel_count);
        Ok(NirsSummary {
            n_subjects: subjects.len(),
            n_recordings: files.len(),
            subjects,
            tasks,
            sampling_frequency: sf,
            channel_count: n_ch.map(|c| c as usize),
        })
    }
}

#[derive(Debug)]
pub struct NirsSummary {
    pub n_subjects: usize,
    pub n_recordings: usize,
    pub subjects: Vec<String>,
    pub tasks: Vec<String>,
    pub sampling_frequency: Option<f64>,
    pub channel_count: Option<usize>,
}
impl std::fmt::Display for NirsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "NIRS Summary: {} subjects, {} recordings, tasks: {:?}",
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
    fn test_nirs_metadata() {
        let json = r#"{"SamplingFrequency":10.0,"NIRSChannelCount":36,"NIRSSourceOptodeCount":8,"NIRSDetectorOptodeCount":8}"#;
        let md: NirsMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.sampling_frequency, Some(10.0));
        assert_eq!(md.nirs_channel_count, Some(36));
        assert_eq!(md.source_optode_count, Some(8));
    }
}
