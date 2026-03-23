//! Typed iEEG metadata from JSON sidecars.

use bids_core::metadata::BidsMetadata;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct IeegMetadata {
    pub sampling_frequency: f64,
    #[serde(rename = "ECOGChannelCount", default)]
    pub ecog_channel_count: Option<u32>,
    #[serde(rename = "SEEGChannelCount", default)]
    pub seeg_channel_count: Option<u32>,
    #[serde(rename = "EEGChannelCount", default)]
    pub eeg_channel_count: Option<u32>,
    #[serde(rename = "EOGChannelCount", default)]
    pub eog_channel_count: Option<u32>,
    #[serde(rename = "ECGChannelCount", default)]
    pub ecg_channel_count: Option<u32>,
    #[serde(rename = "EMGChannelCount", default)]
    pub emg_channel_count: Option<u32>,
    #[serde(default)]
    pub misc_channel_count: Option<u32>,
    #[serde(default)]
    pub trigger_channel_count: Option<u32>,
    #[serde(default)]
    pub power_line_frequency: Option<f64>,
    #[serde(default)]
    pub recording_duration: Option<f64>,
    #[serde(default)]
    pub recording_type: Option<String>,
    #[serde(default)]
    pub task_name: Option<String>,
    #[serde(default)]
    pub manufacturer: Option<String>,
    #[serde(rename = "iEEGReference", default)]
    pub ieeg_reference: Option<String>,
    #[serde(rename = "iEEGGround", default)]
    pub ieeg_ground: Option<String>,
    #[serde(rename = "iEEGPlacementScheme", default)]
    pub ieeg_placement_scheme: Option<String>,
    #[serde(rename = "iEEGElectrodeGroups", default)]
    pub ieeg_electrode_groups: Option<serde_json::Value>,
}

impl IeegMetadata {
    pub fn from_metadata(md: &BidsMetadata) -> Option<Self> {
        md.deserialize_as()
    }

    pub fn total_channel_count(&self) -> u32 {
        self.ecog_channel_count.unwrap_or(0)
            + self.seeg_channel_count.unwrap_or(0)
            + self.eeg_channel_count.unwrap_or(0)
            + self.eog_channel_count.unwrap_or(0)
            + self.ecg_channel_count.unwrap_or(0)
            + self.emg_channel_count.unwrap_or(0)
            + self.misc_channel_count.unwrap_or(0)
            + self.trigger_channel_count.unwrap_or(0)
    }
}
