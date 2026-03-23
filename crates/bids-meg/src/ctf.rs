//! CTF MEG `.ds` directory reading.
//!
//! CTF MEG systems store data in `.ds` directories containing multiple files:
//! `.meg4` (raw data), `.res4` (header/resources), and others.
//!
//! This module provides header-level parsing to extract channel count,
//! sampling rate, and trial structure without reading the full dataset.

use bids_core::error::{BidsError, Result};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// CTF MEG dataset header information parsed from the `.res4` file.
#[derive(Debug, Clone)]
pub struct CtfHeader {
    /// Number of channels.
    pub n_channels: usize,
    /// Number of time samples per trial.
    pub n_samples: usize,
    /// Number of trials (epochs).
    pub n_trials: usize,
    /// Sampling rate in Hz.
    pub sample_rate: f64,
    /// Channel names.
    pub channel_names: Vec<String>,
}

/// Read the header from a CTF `.ds` directory.
///
/// Parses the `.res4` resource file inside the directory to extract
/// channel count, sampling rate, and trial structure.
///
/// # Errors
///
/// Returns an error if the directory doesn't contain a valid `.res4` file
/// or the file can't be parsed.
pub fn read_ctf_header(ds_path: &Path) -> Result<CtfHeader> {
    // Find the .res4 file
    let res4_path = find_res4(ds_path)?;

    let file = std::fs::File::open(&res4_path)?;
    let mut reader = BufReader::new(file);

    // CTF res4 header layout (big-endian):
    // offset 0: 8 bytes — run ID (string)
    // offset 776: 2 bytes — number of channels (i16 big-endian)
    // offset 778: 2 bytes — number of samples per trial (i16)
    // offset 1844: 8 bytes — sample rate (f64 big-endian)
    // Actually, the exact layout varies by CTF version.
    // We use the standard layout documented in FieldTrip/MNE sources.

    // Read run identification
    let mut header_buf = [0u8; 1848];
    reader.read_exact(&mut header_buf).map_err(|_| {
        BidsError::DataFormat(format!("CTF .res4 file too short: {}", res4_path.display()))
    })?;

    let n_channels = i16::from_be_bytes([header_buf[776], header_buf[777]]) as usize;
    let n_samples = i16::from_be_bytes([header_buf[778], header_buf[779]]) as usize;
    let n_trials = i16::from_be_bytes([header_buf[780], header_buf[781]]) as usize;

    let sample_rate = f64::from_be_bytes([
        header_buf[1840],
        header_buf[1841],
        header_buf[1842],
        header_buf[1843],
        header_buf[1844],
        header_buf[1845],
        header_buf[1846],
        header_buf[1847],
    ]);

    // Read channel names: each channel record starts at offset 1848 + ch * 1352
    // Channel name is the first 32 bytes of each record
    let mut channel_names = Vec::with_capacity(n_channels);
    reader.seek(SeekFrom::Start(1848))?;
    for _ in 0..n_channels {
        let mut name_buf = [0u8; 32];
        reader.read_exact(&mut name_buf)?;
        let name = String::from_utf8_lossy(&name_buf)
            .trim_end_matches('\0')
            .trim()
            .to_string();
        channel_names.push(name);
        // Skip rest of channel record (1352 - 32 = 1320 bytes)
        reader.seek(SeekFrom::Current(1320))?;
    }

    if n_channels == 0 || sample_rate <= 0.0 {
        return Err(BidsError::DataFormat(format!(
            "Invalid CTF header in {}: n_channels={}, sample_rate={}",
            ds_path.display(),
            n_channels,
            sample_rate
        )));
    }

    Ok(CtfHeader {
        n_channels,
        n_samples,
        n_trials,
        sample_rate,
        channel_names,
    })
}

fn find_res4(ds_path: &Path) -> Result<std::path::PathBuf> {
    if ds_path.is_dir() {
        for entry in std::fs::read_dir(ds_path)?.flatten() {
            if entry.path().extension().is_some_and(|e| e == "res4") {
                return Ok(entry.path());
            }
        }
    }
    Err(BidsError::DataFormat(format!(
        "No .res4 file found in CTF directory: {}",
        ds_path.display()
    )))
}

impl std::fmt::Display for CtfHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CTF: {} channels × {} samples × {} trials @ {:.1} Hz",
            self.n_channels, self.n_samples, self.n_trials, self.sample_rate
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_res4() {
        let dir = std::env::temp_dir().join("bids_ctf_empty");
        std::fs::create_dir_all(&dir).unwrap();
        let result = read_ctf_header(&dir);
        assert!(result.is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
