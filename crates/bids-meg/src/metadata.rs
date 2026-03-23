//! Typed MEG metadata from JSON sidecars.

use bids_core::metadata::BidsMetadata;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct MegMetadata {
    pub sampling_frequency: f64,
    #[serde(rename = "MEGChannelCount", default)]
    pub meg_channel_count: Option<u32>,
    #[serde(rename = "MEGREFChannelCount", default)]
    pub megref_channel_count: Option<u32>,
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
    pub power_line_frequency: Option<f64>,
    #[serde(default)]
    pub recording_duration: Option<f64>,
    #[serde(default)]
    pub recording_type: Option<String>,
    #[serde(default)]
    pub task_name: Option<String>,
    #[serde(default)]
    pub manufacturer: Option<String>,
    #[serde(default)]
    pub dewar_position: Option<String>,
    #[serde(default)]
    pub software_filters: Option<serde_json::Value>,
    #[serde(default)]
    pub digitized_head_points: Option<bool>,
    #[serde(default)]
    pub digitized_landmark: Option<bool>,
}

impl MegMetadata {
    pub fn from_metadata(md: &BidsMetadata) -> Option<Self> {
        md.deserialize_as()
    }
}
