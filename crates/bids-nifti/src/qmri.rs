//! Quantitative MRI (qMRI) metadata for parametric maps and acquisition files.
//!
//! BIDS supports a wide range of quantitative MRI methods: relaxometry
//! (T1map, T2map, T2starmap), magnetization transfer (MTsat, MTRmap),
//! B1 field mapping (TB1DAM, TB1EPI, etc.), and multi-parameter mapping
//! (MPM, VFA, MP2RAGE, etc.).
//!
//! This module provides typed metadata for qMRI JSON sidecars.

use serde::{Deserialize, Serialize};

/// Metadata for quantitative MRI acquisitions and derived maps.
///
/// Covers fields from the BIDS specification for qMRI-specific JSON sidecars,
/// including relaxometry, magnetization transfer, and B1 field mapping.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct QmriMetadata {
    // ── Common acquisition parameters ───────────────────────────────────
    /// Repetition time in seconds.
    #[serde(default)]
    pub repetition_time: Option<f64>,
    /// Echo time(s) in seconds.
    #[serde(default)]
    pub echo_time: Option<serde_json::Value>,
    /// Flip angle(s) in degrees.
    #[serde(default)]
    pub flip_angle: Option<serde_json::Value>,
    /// Inversion time in seconds.
    #[serde(default)]
    pub inversion_time: Option<f64>,

    // ── Magnetization transfer ──────────────────────────────────────────
    /// Whether MT pulse is on or off (`"on"` or `"off"`).
    #[serde(rename = "MTState", default)]
    pub mt_state: Option<String>,
    /// MT pulse shape.
    #[serde(rename = "MTPulseBandwidth", default)]
    pub mt_pulse_bandwidth: Option<f64>,
    /// MT pulse duration in seconds.
    #[serde(rename = "MTPulseDuration", default)]
    pub mt_pulse_duration: Option<f64>,
    /// Number of MT pulses per excitation.
    #[serde(rename = "MTPulseShape", default)]
    pub mt_pulse_shape: Option<String>,

    // ── Multi-echo / multi-flip ─────────────────────────────────────────
    /// Number of echoes.
    #[serde(default)]
    pub number_of_echoes: Option<u32>,
    /// Number of flip angles.
    #[serde(default)]
    pub number_of_flip_angles: Option<u32>,

    // ── MP2RAGE-specific ────────────────────────────────────────────────
    /// First inversion time for MP2RAGE.
    #[serde(rename = "InversionTime1", default)]
    pub inversion_time_1: Option<f64>,
    /// Second inversion time for MP2RAGE.
    #[serde(rename = "InversionTime2", default)]
    pub inversion_time_2: Option<f64>,
    /// Number of excitations between inversions for MP2RAGE.
    #[serde(rename = "NumberShots", default)]
    pub number_shots: Option<u32>,

    // ── B1 mapping ──────────────────────────────────────────────────────
    /// Method used for B1+ mapping.
    #[serde(rename = "B1MappingMethod", default)]
    pub b1_mapping_method: Option<String>,

    // ── Derived map metadata ────────────────────────────────────────────
    /// Units of the parametric map values.
    #[serde(default)]
    pub units: Option<String>,
    /// Fitting method used to derive the map.
    #[serde(default)]
    pub fitting_method: Option<String>,
    /// Whether the map was estimated with B1+ correction.
    #[serde(rename = "B1Corrected", default)]
    pub b1_corrected: Option<bool>,

    // ── General ─────────────────────────────────────────────────────────
    /// Manufacturer of the scanner.
    #[serde(default)]
    pub manufacturer: Option<String>,
    /// Magnetic field strength in Tesla.
    #[serde(default)]
    pub magnetic_field_strength: Option<f64>,
    /// Pulse sequence type.
    #[serde(default)]
    pub pulse_sequence_type: Option<String>,
}

impl QmriMetadata {
    /// Parse from a `BidsMetadata` dictionary.
    pub fn from_metadata(md: &bids_core::metadata::BidsMetadata) -> Option<Self> {
        md.deserialize_as()
    }

    /// Get echo time(s) as a Vec of f64.
    #[must_use]
    pub fn echo_times(&self) -> Vec<f64> {
        match &self.echo_time {
            Some(serde_json::Value::Number(n)) => {
                n.as_f64().map_or_else(Vec::new, |v| vec![v])
            }
            Some(serde_json::Value::Array(arr)) => {
                arr.iter().filter_map(|v| v.as_f64()).collect()
            }
            _ => Vec::new(),
        }
    }

    /// Get flip angle(s) as a Vec of f64.
    #[must_use]
    pub fn flip_angles(&self) -> Vec<f64> {
        match &self.flip_angle {
            Some(serde_json::Value::Number(n)) => {
                n.as_f64().map_or_else(Vec::new, |v| vec![v])
            }
            Some(serde_json::Value::Array(arr)) => {
                arr.iter().filter_map(|v| v.as_f64()).collect()
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qmri_metadata_parse() {
        let json = r#"{
            "RepetitionTime": 0.025,
            "EchoTime": [0.002, 0.006, 0.010],
            "FlipAngle": 6,
            "MTState": "on",
            "MagneticFieldStrength": 3.0
        }"#;
        let md: QmriMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.repetition_time, Some(0.025));
        assert_eq!(md.echo_times(), vec![0.002, 0.006, 0.010]);
        assert_eq!(md.flip_angles(), vec![6.0]);
        assert_eq!(md.mt_state.as_deref(), Some("on"));
        assert_eq!(md.magnetic_field_strength, Some(3.0));
    }

    #[test]
    fn test_qmri_mp2rage() {
        let json = r#"{
            "InversionTime1": 0.7,
            "InversionTime2": 2.5,
            "FlipAngle": [4, 5],
            "NumberShots": 176
        }"#;
        let md: QmriMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.inversion_time_1, Some(0.7));
        assert_eq!(md.inversion_time_2, Some(2.5));
        assert_eq!(md.flip_angles(), vec![4.0, 5.0]);
        assert_eq!(md.number_shots, Some(176));
    }
}
