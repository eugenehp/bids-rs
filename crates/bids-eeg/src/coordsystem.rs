//! Coordinate system information from `_coordsystem.json`.
//!
//! Describes the coordinate system used for electrode positions, including
//! the system name, units, and description.

use serde::{Deserialize, Serialize};

/// Coordinate system information from _coordsystem.json.
///
/// See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct CoordinateSystem {
    /// Coordinate system name (e.g., "CapTrak", "EEGLAB", "Other").
    #[serde(rename = "EEGCoordinateSystem")]
    pub eeg_coordinate_system: String,

    /// Units of the electrode coordinates (e.g., "mm", "cm", "m").
    #[serde(rename = "EEGCoordinateUnits")]
    pub eeg_coordinate_units: String,

    /// Description of the coordinate system if "Other" is used.
    #[serde(rename = "EEGCoordinateSystemDescription", default)]
    pub eeg_coordinate_system_description: Option<String>,

    /// Method used to determine electrode positions.
    #[serde(rename = "EEGCoordinateProcessingDescription", default)]
    pub eeg_coordinate_processing_description: Option<String>,

    /// Reference for the iEEG coordinate system.
    #[serde(rename = "iEEGCoordinateSystem", default)]
    pub ieeg_coordinate_system: Option<String>,

    /// Units for iEEG coordinates.
    #[serde(rename = "iEEGCoordinateUnits", default)]
    pub ieeg_coordinate_units: Option<String>,

    /// Description for iEEG coordinate system.
    #[serde(rename = "iEEGCoordinateSystemDescription", default)]
    pub ieeg_coordinate_system_description: Option<String>,

    /// Processing description for iEEG coordinates.
    #[serde(rename = "iEEGCoordinateProcessingDescription", default)]
    pub ieeg_coordinate_processing_description: Option<String>,

    /// Fiducials in the coordinate system.
    #[serde(default)]
    pub fiducials: Option<serde_json::Value>,

    /// Anatomical landmarks.
    #[serde(rename = "AnatomicalLandmarkCoordinates", default)]
    pub anatomical_landmark_coordinates: Option<serde_json::Value>,

    /// Coordinate system for anatomical landmarks.
    #[serde(rename = "AnatomicalLandmarkCoordinateSystem", default)]
    pub anatomical_landmark_coordinate_system: Option<String>,

    /// Units for anatomical landmark coordinates.
    #[serde(rename = "AnatomicalLandmarkCoordinateUnits", default)]
    pub anatomical_landmark_coordinate_units: Option<String>,

    /// Photo of the head with landmark markers.
    #[serde(rename = "IntendedFor", default)]
    pub intended_for: Option<serde_json::Value>,
}

impl CoordinateSystem {
    /// Load from a JSON file.
    pub fn from_file(path: &std::path::Path) -> bids_core::error::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let cs: Self = serde_json::from_str(&contents)?;
        Ok(cs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordsystem_parse() {
        let json = r#"{
            "EEGCoordinateSystem": "CapTrak",
            "EEGCoordinateUnits": "mm",
            "EEGCoordinateSystemDescription": "RAS orientation"
        }"#;

        let cs: CoordinateSystem = serde_json::from_str(json).unwrap();
        assert_eq!(cs.eeg_coordinate_system, "CapTrak");
        assert_eq!(cs.eeg_coordinate_units, "mm");
    }
}
