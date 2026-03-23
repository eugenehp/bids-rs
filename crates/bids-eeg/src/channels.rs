//! EEG channel definitions and `_channels.tsv` parsing.
//!
//! Provides [`Channel`] (name, type, units, sampling frequency, status) and
//! [`ChannelType`] covering all BIDS-defined EEG/MEG/iEEG channel types.

use bids_core::error::{BidsError, Result};
use bids_io::tsv::read_tsv;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Type of EEG channel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelType {
    EEG,
    EOG,
    ECG,
    EMG,
    MISC,
    TRIG,
    REF,
    MEGMAG,
    MEGGRAD,
    MEGREF,
    ECOG,
    SEEG,
    DBS,
    VEOG,
    HEOG,
    Audio,
    PD,
    EYEGAZE,
    PUPIL,
    SysClock,
    ADC,
    DAC,
    HLU,
    FITERR,
    Other(String),
}

impl ChannelType {
    pub fn parse(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "EEG" => Self::EEG,
            "EOG" => Self::EOG,
            "ECG" | "EKG" => Self::ECG,
            "EMG" => Self::EMG,
            "MISC" => Self::MISC,
            "TRIG" | "TRIGGER" => Self::TRIG,
            "REF" => Self::REF,
            "MEGMAG" => Self::MEGMAG,
            "MEGGRAD" => Self::MEGGRAD,
            "MEGREF" | "MEGREFMAG" | "MEGREFGRAD" => Self::MEGREF,
            "ECOG" => Self::ECOG,
            "SEEG" => Self::SEEG,
            "DBS" => Self::DBS,
            "VEOG" => Self::VEOG,
            "HEOG" => Self::HEOG,
            "AUDIO" => Self::Audio,
            "PD" | "PHOTODIODE" => Self::PD,
            "EYEGAZE" => Self::EYEGAZE,
            "PUPIL" => Self::PUPIL,
            "SYSCLOCK" => Self::SysClock,
            "ADC" => Self::ADC,
            "DAC" => Self::DAC,
            "HLU" => Self::HLU,
            "FITERR" => Self::FITERR,
            other => Self::Other(other.to_string()),
        }
    }
}

impl std::fmt::Display for ChannelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Other(s) => write!(f, "{s}"),
            _ => write!(f, "{self:?}"),
        }
    }
}

/// A single EEG channel definition from _channels.tsv.
///
/// See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    /// Channel name (required).
    pub name: String,
    /// Channel type (required): EEG, EOG, ECG, EMG, MISC, TRIG, etc.
    pub channel_type: ChannelType,
    /// Units of the channel data (required): e.g., "µV", "microV", "mV".
    pub units: String,
    /// Description of the channel.
    pub description: Option<String>,
    /// Sampling frequency in Hz (if different from the main recording).
    pub sampling_frequency: Option<f64>,
    /// Low cutoff frequency of the hardware filter in Hz.
    pub low_cutoff: Option<f64>,
    /// High cutoff frequency of the hardware filter in Hz.
    pub high_cutoff: Option<f64>,
    /// Notch filter frequency in Hz.
    pub notch: Option<f64>,
    /// Reference for this channel.
    pub reference: Option<String>,
    /// Status: "good" or "bad".
    pub status: Option<String>,
    /// Description of why the channel is bad.
    pub status_description: Option<String>,
}

impl Channel {
    /// Whether this channel is marked as "bad".
    pub fn is_bad(&self) -> bool {
        self.status.as_deref() == Some("bad")
    }

    /// Whether this is an EEG-type channel.
    pub fn is_eeg(&self) -> bool {
        self.channel_type == ChannelType::EEG
    }
}

/// Read channels from a BIDS _channels.tsv file.
pub fn read_channels_tsv(path: &Path) -> Result<Vec<Channel>> {
    let rows = read_tsv(path)?;
    let mut channels = Vec::with_capacity(rows.len());

    for row in &rows {
        let name = row
            .get("name")
            .ok_or_else(|| BidsError::Csv("Missing 'name' column in channels.tsv".into()))?
            .trim()
            .to_string();

        let channel_type = row
            .get("type")
            .map(|s| ChannelType::parse(s.trim()))
            .unwrap_or(ChannelType::MISC);

        let units = row
            .get("units")
            .cloned()
            .unwrap_or_else(|| "n/a".to_string())
            .trim()
            .to_string();

        let channel = Channel {
            name,
            channel_type,
            units,
            description: non_empty(row.get("description")),
            sampling_frequency: parse_f64(row.get("sampling_frequency")),
            low_cutoff: parse_f64(row.get("low_cutoff")),
            high_cutoff: parse_f64(row.get("high_cutoff")),
            notch: parse_f64(row.get("notch")),
            reference: non_empty(row.get("reference")),
            status: non_empty(row.get("status")),
            status_description: non_empty(row.get("status_description")),
        };
        channels.push(channel);
    }

    Ok(channels)
}

fn non_empty(val: Option<&String>) -> Option<String> {
    val.filter(|s| !s.is_empty() && *s != "n/a")
        .map(|s| s.trim().to_string())
}

fn parse_f64(val: Option<&String>) -> Option<f64> {
    val.and_then(|s| s.trim().parse().ok())
}

/// Count channels by type.
pub fn count_by_type(channels: &[Channel]) -> std::collections::HashMap<ChannelType, usize> {
    let mut counts = std::collections::HashMap::new();
    for ch in channels {
        *counts.entry(ch.channel_type.clone()).or_insert(0) += 1;
    }
    counts
}

/// Get only the EEG channels.
pub fn eeg_channels(channels: &[Channel]) -> Vec<&Channel> {
    channels.iter().filter(|c| c.is_eeg()).collect()
}

/// Get only the bad channels.
pub fn bad_channels(channels: &[Channel]) -> Vec<&Channel> {
    channels.iter().filter(|c| c.is_bad()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_channels_tsv() {
        let dir = std::env::temp_dir().join("bids_eeg_channels_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("channels.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(
            f,
            "name\ttype\tunits\tsampling_frequency\tlow_cutoff\thigh_cutoff\tnotch"
        )
        .unwrap();
        writeln!(f, "Fp1\tEEG\tmicroV\t256\t0.5\t100\t50").unwrap();
        writeln!(f, "Fp2\tEEG\tmicroV\t256\t0.5\t100\t50").unwrap();
        writeln!(f, "EOG1\tEOG\tmicroV\t256\tn/a\tn/a\tn/a").unwrap();

        let channels = read_channels_tsv(&path).unwrap();
        assert_eq!(channels.len(), 3);
        assert_eq!(channels[0].name, "Fp1");
        assert_eq!(channels[0].channel_type, ChannelType::EEG);
        assert_eq!(channels[0].sampling_frequency, Some(256.0));
        assert_eq!(channels[2].channel_type, ChannelType::EOG);
        assert_eq!(channels[2].notch, None);

        assert_eq!(eeg_channels(&channels).len(), 2);
        assert_eq!(bad_channels(&channels).len(), 0);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
