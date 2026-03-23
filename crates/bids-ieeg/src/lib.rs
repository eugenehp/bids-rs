#![deny(unsafe_code)]
//! Intracranial EEG (iEEG) support for BIDS datasets.
//!
//! Provides typed access to iEEG-specific files including electrode positions
//! (with size, hemisphere, group, and type), channels, events, coordinate
//! systems, and iEEG metadata. Supports both ECoG (electrocorticography) and
//! SEEG (stereoelectroencephalography) data.
//!
//! Signal data reading is supported for all BIDS-iEEG formats (EDF, BDF,
//! BrainVision) via [`IeegLayout::read_data`], which delegates to the
//! `bids-eeg` crate's high-performance readers. Types like [`EegData`],
//! [`ReadOptions`], and [`Annotation`] are re-exported for convenience.
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/intracranial-electroencephalography.html>

pub mod layout;
pub mod metadata;

use bids_core::error::Result;
use bids_io::tsv::read_tsv;
use serde::{Deserialize, Serialize};

pub use layout::IeegLayout;
pub use metadata::IeegMetadata;

// Re-export data reading types from bids-eeg (iEEG uses the same file formats)
pub use bids_eeg::{Annotation, EegData, ReadOptions, read_brainvision, read_edf, read_eeg_data};

/// An iEEG electrode with size, hemisphere, group, and type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IeegElectrode {
    pub name: String,
    pub x: Option<f64>,
    pub y: Option<f64>,
    pub z: Option<f64>,
    pub size: Option<f64>,
    pub hemisphere: Option<String>,
    pub group: Option<String>,
    pub electrode_type: Option<String>,
    pub manufacturer: Option<String>,
}

impl IeegElectrode {
    pub fn has_position(&self) -> bool {
        self.x.is_some() && self.y.is_some() && self.z.is_some()
    }
    pub fn position(&self) -> Option<(f64, f64, f64)> {
        match (self.x, self.y, self.z) {
            (Some(x), Some(y), Some(z)) => Some((x, y, z)),
            _ => None,
        }
    }
}

pub fn read_ieeg_electrodes(path: &std::path::Path) -> Result<Vec<IeegElectrode>> {
    let rows = read_tsv(path)?;
    Ok(rows
        .iter()
        .map(|r| {
            let get = |k: &str| {
                r.get(k)
                    .filter(|s| !s.is_empty() && s.as_str() != "n/a")
                    .cloned()
            };
            let getf = |k: &str| r.get(k).and_then(|v| v.parse().ok());
            IeegElectrode {
                name: r.get("name").cloned().unwrap_or_default(),
                x: getf("x"),
                y: getf("y"),
                z: getf("z"),
                size: getf("size"),
                hemisphere: get("hemisphere"),
                group: get("group"),
                electrode_type: get("type"),
                manufacturer: get("manufacturer"),
            }
        })
        .collect())
}

/// iEEG coordinate system from _coordsystem.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct IeegCoordSystem {
    #[serde(rename = "iEEGCoordinateSystem")]
    pub coordinate_system: String,
    #[serde(rename = "iEEGCoordinateUnits")]
    pub coordinate_units: String,
    #[serde(rename = "iEEGCoordinateSystemDescription", default)]
    pub coordinate_system_description: Option<String>,
    #[serde(rename = "iEEGCoordinateProcessingDescription", default)]
    pub processing_description: Option<String>,
    #[serde(rename = "IntendedFor", default)]
    pub intended_for: Option<serde_json::Value>,
}

impl IeegCoordSystem {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&contents)?)
    }
}

#[derive(Debug)]
pub struct IeegSummary {
    pub n_subjects: usize,
    pub n_recordings: usize,
    pub subjects: Vec<String>,
    pub tasks: Vec<String>,
    pub sampling_frequency: Option<f64>,
    pub channel_count: Option<usize>,
}
impl std::fmt::Display for IeegSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "iEEG Dataset Summary:")?;
        writeln!(f, "  Subjects: {}", self.n_subjects)?;
        writeln!(f, "  Recordings: {}", self.n_recordings)?;
        writeln!(f, "  Tasks: {:?}", self.tasks)?;
        if let Some(sf) = self.sampling_frequency {
            writeln!(f, "  Sampling Frequency: {sf} Hz")?;
        }
        if let Some(cc) = self.channel_count {
            writeln!(f, "  Channels: {cc}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ieeg_metadata() {
        let json = r#"{"SamplingFrequency":512,"SEEGChannelCount":124,"EEGChannelCount":2,"ECGChannelCount":2}"#;
        let md: IeegMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.sampling_frequency, 512.0);
        assert_eq!(md.seeg_channel_count, Some(124));
        assert_eq!(md.total_channel_count(), 128);
    }

    #[test]
    fn test_ieeg_coordsystem() {
        let json = r#"{"iEEGCoordinateSystem":"ACPC","iEEGCoordinateUnits":"mm"}"#;
        let cs: IeegCoordSystem = serde_json::from_str(json).unwrap();
        assert_eq!(cs.coordinate_system, "ACPC");
        assert_eq!(cs.coordinate_units, "mm");
    }
}
