#![deny(unsafe_code)]
//! Motion capture support for BIDS datasets.
//!
//! Provides access to motion tracking data, channels, and metadata including
//! tracking system name, sampling frequency, and recording type.
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/motion.html>

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_core::metadata::BidsMetadata;
use bids_layout::BidsLayout;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct MotionMetadata {
    #[serde(default)]
    pub sampling_frequency: Option<f64>,
    #[serde(default)]
    pub task_name: Option<String>,
    #[serde(default)]
    pub tracking_system_name: Option<String>,
    #[serde(default)]
    pub manufacturer: Option<String>,
    #[serde(default)]
    pub manufacturers_model_name: Option<String>,
    #[serde(default)]
    pub recording_type: Option<String>,
}
impl MotionMetadata {
    pub fn from_metadata(md: &BidsMetadata) -> Option<Self> {
        md.deserialize_as()
    }
}

pub struct MotionLayout<'a> {
    layout: &'a BidsLayout,
}
impl<'a> MotionLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self {
        Self { layout }
    }
    pub fn get_motion_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("motion").collect()
    }
    pub fn get_motion_files_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("motion").subject(s).collect()
    }
    pub fn get_motion_files_for_task(&self, t: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("motion").task(t).collect()
    }
    pub fn get_channels(&self, f: &BidsFile) -> Result<Option<Vec<bids_eeg::Channel>>> {
        let p = f.companion("channels", "tsv");
        if p.exists() {
            Ok(Some(bids_eeg::read_channels_tsv(&p)?))
        } else {
            Ok(None)
        }
    }
    pub fn get_metadata(&self, f: &BidsFile) -> Result<Option<MotionMetadata>> {
        Ok(self.layout.get_metadata(&f.path)?.deserialize_as())
    }
    pub fn get_motion_subjects(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("motion").return_unique("subject")
    }
    pub fn get_motion_tasks(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("motion").return_unique("task")
    }
    pub fn summary(&self) -> Result<MotionSummary> {
        let files = self.get_motion_files()?;
        let subjects = self.get_motion_subjects()?;
        let tasks = self.get_motion_tasks()?;
        let sf = files
            .first()
            .and_then(|f| self.get_metadata(f).ok().flatten())
            .and_then(|m| m.sampling_frequency);
        let ch = files
            .first()
            .and_then(|f| self.get_channels(f).ok().flatten())
            .map(|c| c.len());
        Ok(MotionSummary {
            n_subjects: subjects.len(),
            n_recordings: files.len(),
            subjects,
            tasks,
            sampling_frequency: sf,
            channel_count: ch,
        })
    }
}

#[derive(Debug)]
pub struct MotionSummary {
    pub n_subjects: usize,
    pub n_recordings: usize,
    pub subjects: Vec<String>,
    pub tasks: Vec<String>,
    pub sampling_frequency: Option<f64>,
    pub channel_count: Option<usize>,
}
impl std::fmt::Display for MotionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Motion Summary: {} subjects, {} recordings, tasks: {:?}",
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
    fn test_motion_metadata() {
        let json = r#"{"SamplingFrequency":120.0,"TrackingSystemName":"PhaseSpace","RecordingType":"continuous"}"#;
        let md: MotionMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.sampling_frequency, Some(120.0));
        assert_eq!(md.tracking_system_name.as_deref(), Some("PhaseSpace"));
    }
}
