//! Typed PET metadata from JSON sidecars.

use bids_core::metadata::BidsMetadata;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PetMetadata {
    #[serde(default)]
    pub manufacturer: Option<String>,
    #[serde(default)]
    pub manufacturers_model_name: Option<String>,
    #[serde(default)]
    pub units: Option<String>,
    #[serde(default)]
    pub tracer_name: Option<String>,
    #[serde(default)]
    pub tracer_radionuclide: Option<String>,
    #[serde(default)]
    pub tracer_molecular_weight: Option<f64>,
    #[serde(default)]
    pub injected_radioactivity: Option<f64>,
    #[serde(default)]
    pub injected_radioactivity_units: Option<String>,
    #[serde(default)]
    pub injected_mass: Option<f64>,
    #[serde(default)]
    pub injected_mass_units: Option<String>,
    #[serde(default)]
    pub specific_radioactivity: Option<f64>,
    #[serde(default)]
    pub mode_of_administration: Option<String>,
    #[serde(default)]
    pub time_zero: Option<String>,
    #[serde(default)]
    pub scan_start: Option<f64>,
    #[serde(default)]
    pub injection_start: Option<f64>,
    #[serde(default)]
    pub frame_times_start: Option<Vec<f64>>,
    #[serde(default)]
    pub frame_duration: Option<Vec<f64>>,
    #[serde(default)]
    pub body_part: Option<String>,
}

impl PetMetadata {
    pub fn from_metadata(md: &BidsMetadata) -> Option<Self> {
        md.deserialize_as()
    }
    pub fn total_duration(&self) -> Option<f64> {
        self.frame_duration.as_ref().map(|d| d.iter().sum())
    }
}
