//! Typed EEG metadata from JSON sidecars.
//!
//! [`EegMetadata`] provides typed access to all EEG-specific sidecar fields
//! including sampling frequency, channel counts, reference, placement scheme,
//! power line frequency, recording duration, and filter descriptions.

use bids_core::metadata::BidsMetadata;
use serde::{Deserialize, Serialize};

/// EEG-specific metadata from the JSON sidecar.
///
/// See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct EegMetadata {
    /// Name of the task.
    #[serde(default)]
    pub task_name: Option<String>,
    /// Description of the task.
    #[serde(default)]
    pub task_description: Option<String>,
    /// Sampling frequency of the EEG recording in Hz.
    pub sampling_frequency: f64,
    /// Number of EEG channels.
    #[serde(rename = "EEGChannelCount", default)]
    pub eeg_channel_count: Option<u32>,
    /// Number of EOG channels.
    #[serde(rename = "EOGChannelCount", default)]
    pub eog_channel_count: Option<u32>,
    /// Number of ECG channels.
    #[serde(rename = "ECGChannelCount", default)]
    pub ecg_channel_count: Option<u32>,
    /// Number of EMG channels.
    #[serde(rename = "EMGChannelCount", default)]
    pub emg_channel_count: Option<u32>,
    /// Number of miscellaneous channels.
    #[serde(default)]
    pub misc_channel_count: Option<u32>,
    /// Number of trigger channels.
    #[serde(default)]
    pub trigger_channel_count: Option<u32>,
    /// EEG placement scheme (e.g., "10-20", "10-10", "10-5").
    #[serde(rename = "EEGPlacementScheme", default)]
    pub eeg_placement_scheme: Option<String>,
    /// EEG reference electrode(s).
    #[serde(rename = "EEGReference", default)]
    pub eeg_reference: Option<String>,
    /// EEG ground electrode.
    #[serde(rename = "EEGGround", default)]
    pub eeg_ground: Option<String>,
    /// Duration of the recording in seconds.
    #[serde(default)]
    pub recording_duration: Option<f64>,
    /// Type of recording: "continuous", "epoched", "discontinuous".
    #[serde(default)]
    pub recording_type: Option<String>,
    /// Power line frequency in Hz (50 or 60).
    #[serde(default)]
    pub power_line_frequency: Option<f64>,
    /// Software filters applied.
    #[serde(default)]
    pub software_filters: Option<serde_json::Value>,
    /// Hardware filters applied.
    #[serde(default)]
    pub hardware_filters: Option<serde_json::Value>,
    /// Manufacturer of the EEG system.
    #[serde(default)]
    pub manufacturer: Option<String>,
    /// Manufacturer's model name.
    #[serde(default)]
    pub manufacturers_model_name: Option<String>,
    /// Name of the cap (if applicable).
    #[serde(default)]
    pub cap_manufacturer: Option<String>,
    /// Cap model.
    #[serde(default)]
    pub cap_manufacturers_model_name: Option<String>,
    /// Institutional department name.
    #[serde(default)]
    pub institution_name: Option<String>,
    /// Institutional department name.
    #[serde(default)]
    pub institutional_department_name: Option<String>,
    /// Address of the institution.
    #[serde(default)]
    pub institution_address: Option<String>,
    /// Subject artifact description.
    #[serde(default)]
    pub subject_artifact_description: Option<String>,
}

impl EegMetadata {
    /// Try to extract EEG metadata from a generic BidsMetadata.
    pub fn from_metadata(md: &BidsMetadata) -> Option<Self> {
        md.deserialize_as()
    }

    /// Get the total channel count.
    pub fn total_channel_count(&self) -> u32 {
        self.eeg_channel_count.unwrap_or(0)
            + self.eog_channel_count.unwrap_or(0)
            + self.ecg_channel_count.unwrap_or(0)
            + self.emg_channel_count.unwrap_or(0)
            + self.misc_channel_count.unwrap_or(0)
            + self.trigger_channel_count.unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eeg_metadata_parse() {
        let json = r#"{
            "TaskName": "rest",
            "SamplingFrequency": 256,
            "EEGChannelCount": 64,
            "EOGChannelCount": 2,
            "ECGChannelCount": 1,
            "EMGChannelCount": 0,
            "EEGPlacementScheme": "10-20",
            "EEGReference": "Cz",
            "RecordingDuration": 300.0,
            "RecordingType": "continuous",
            "PowerLineFrequency": 50
        }"#;

        let md: EegMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.sampling_frequency, 256.0);
        assert_eq!(md.eeg_channel_count, Some(64));
        assert_eq!(md.eeg_reference.as_deref(), Some("Cz"));
        assert_eq!(md.total_channel_count(), 67);
    }
}
