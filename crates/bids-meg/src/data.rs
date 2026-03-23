//! MEG data reading via the FIFF format.
//!
//! Requires the `fiff` feature flag: `bids-meg = { features = ["fiff"] }`
//!
//! FIFF (Functional Imaging File Format) is the native format for
//! Elekta/Neuromag MEG systems. This module provides high-level access
//! to raw MEG signal data from `.fif` files.

use bids_core::error::{BidsError, Result};
use std::path::Path;

/// Raw MEG signal data read from a FIFF file.
#[derive(Debug, Clone)]
pub struct MegData {
    /// Channel names in order.
    pub channel_names: Vec<String>,
    /// Signal data: one `Vec<f64>` per channel, calibrated to physical units.
    pub data: Vec<Vec<f64>>,
    /// Sampling frequency in Hz.
    pub sfreq: f64,
    /// Total number of samples per channel.
    pub n_samples: usize,
    /// Indices of bad channels (from the FIFF file's bad channel list).
    pub bad_channels: Vec<String>,
}

impl MegData {
    pub fn n_channels(&self) -> usize { self.data.len() }
}

impl bids_core::timeseries::TimeSeries for MegData {
    fn n_channels(&self) -> usize { self.data.len() }
    fn n_samples(&self) -> usize { self.n_samples }
    fn channel_names(&self) -> &[String] { &self.channel_names }
    fn sampling_rate(&self) -> f64 { self.sfreq }
    fn channel_data(&self, index: usize) -> Option<&[f64]> { self.data.get(index).map(|v| v.as_slice()) }
    fn duration(&self) -> f64 { if self.sfreq > 0.0 { self.n_samples as f64 / self.sfreq } else { 0.0 } }
}

/// Read raw MEG data from a FIFF (.fif) file.
///
/// Uses the `fiff` crate to parse the FIFF tree, extract measurement info
/// (channels, sampling rate, bad channels), and read raw data buffers.
pub fn read_fiff(path: &Path) -> Result<MegData> {
    use fiff::{open_fiff, MeasInfo};
    use fiff::tree::dir_tree_find;
    use fiff::tag::Tag;
    use fiff::constants::*;

    let (mut reader, tree) = open_fiff(path)
        .map_err(|e| BidsError::Io(std::io::Error::new(
            std::io::ErrorKind::Other, format!("FIFF open error: {}", e)
        )))?;

    // Read measurement info
    let meas_info = MeasInfo::read(&mut reader, &tree)
        .map_err(|e| BidsError::Csv(format!("Cannot read FIFF measurement info: {}", e)))?;

    let sfreq = meas_info.sfreq;
    let n_channels = meas_info.nchan as usize;
    let channel_names: Vec<String> = meas_info.channels.iter()
        .map(|ch| ch.ch_name.clone())
        .collect();
    let bad_channels = meas_info.bads.clone();

    // Calibration factors per channel
    let cals: Vec<f64> = meas_info.channels.iter()
        .map(|ch| ch.calibration())
        .collect();

    // Find raw data blocks in the FIFF tree
    let raw_blocks = dir_tree_find(&tree, FIFFB_RAW_DATA);
    let data_blocks = if raw_blocks.is_empty() {
        // Try continuous data block as fallback
        dir_tree_find(&tree, FIFFB_CONTINUOUS_DATA)
    } else {
        raw_blocks
    };

    // Read all raw data buffers
    let mut all_data: Vec<Vec<f64>> = vec![Vec::new(); n_channels];

    for block in &data_blocks {
        for entry in &block.directory {
            if entry.kind == FIFF_DATA_BUFFER {
                let tag = Tag::read_at(&mut reader, entry.pos as u64)
                    .map_err(|e| BidsError::Csv(format!("Cannot read FIFF data buffer: {}", e)))?;

                if let Ok(samples) = tag.as_samples(n_channels) {
                    // samples is Vec<Vec<f64>> — channels × samples_in_buffer
                    for (ch, ch_samples) in samples.iter().enumerate() {
                        if ch < n_channels {
                            let cal = cals.get(ch).copied().unwrap_or(1.0);
                            for &s in ch_samples {
                                all_data[ch].push(s * cal);
                            }
                        }
                    }
                }
            }
        }
    }

    let n_samples = all_data.first().map(|v| v.len()).unwrap_or(0);

    Ok(MegData {
        channel_names,
        data: all_data,
        sfreq,
        n_samples,
        bad_channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meg_data_struct() {
        let data = MegData {
            channel_names: vec!["MEG0111".into(), "MEG0112".into()],
            data: vec![vec![1e-13, 2e-13], vec![3e-13, 4e-13]],
            sfreq: 1000.0,
            n_samples: 2,
            bad_channels: vec![],
        };
        assert_eq!(data.n_channels(), 2);
        assert_eq!(data.n_samples, 2);
    }
}
