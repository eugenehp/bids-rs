//! NIRS data reading via the SNIRF (HDF5) format.
//!
//! Requires the `snirf` feature flag: `bids-nirs = { features = ["snirf"] }`
//!
//! SNIRF (Shared Near Infrared Spectroscopy Format) stores fNIRS data in HDF5
//! files. This module reads the `/nirs/data1/dataTimeSeries` dataset and
//! associated metadata.
//!
//! See: <https://github.com/fNIRS/snirf/blob/master/snirf_specification.md>

use bids_core::error::{BidsError, Result};
use std::path::Path;

/// NIRS signal data read from a SNIRF file.
#[derive(Debug, Clone)]
pub struct NirsData {
    /// Time series data: channels × samples, in physical units.
    pub data: Vec<Vec<f64>>,
    /// Time vector in seconds (one value per sample).
    pub time: Vec<f64>,
    /// Channel names.
    pub channel_names: Vec<String>,
    /// Number of channels.
    pub n_channels: usize,
    /// Number of time points.
    pub n_samples: usize,
    /// Source-detector-wavelength measurement list.
    /// Each entry is (source_index, detector_index, wavelength_nm).
    pub measurement_list: Vec<(u32, u32, f64)>,
}

impl NirsData {
    /// Sampling frequency derived from time vector (1 / median dt).
    pub fn sfreq(&self) -> f64 {
        if self.time.len() < 2 { return 0.0; }
        let dt = self.time[1] - self.time[0];
        if dt > 0.0 { 1.0 / dt } else { 0.0 }
    }

    /// Channel names (generated as "ch1", "ch2", ... if not available).
    pub fn channel_labels(&self) -> Vec<String> {
        (1..=self.n_channels).map(|i| format!("ch{}", i)).collect()
    }
}

impl bids_core::timeseries::TimeSeries for NirsData {
    fn n_channels(&self) -> usize { self.n_channels }
    fn n_samples(&self) -> usize { self.n_samples }
    fn channel_names(&self) -> &[String] { &self.channel_names }
    fn sampling_rate(&self) -> f64 { self.sfreq() }
    fn channel_data(&self, index: usize) -> Option<&[f64]> { self.data.get(index).map(|v| v.as_slice()) }
    fn duration(&self) -> f64 { self.time.last().copied().unwrap_or(0.0) - self.time.first().copied().unwrap_or(0.0) }
}

/// Read NIRS data from a SNIRF (.snirf) file.
///
/// Reads the first nirs group (`/nirs/data1`). The data is returned as a
/// channels × samples matrix (transposed from SNIRF's samples × channels layout).
pub fn read_snirf(path: &Path) -> Result<NirsData> {
    let file = hdf5::File::open(path)
        .map_err(|e| BidsError::Io(std::io::Error::new(
            std::io::ErrorKind::Other, format!("HDF5 error: {}", e)
        )))?;

    // Navigate to /nirs or /nirs1
    let nirs_group = file.group("nirs")
        .or_else(|_| file.group("nirs1"))
        .map_err(|e| BidsError::Csv(format!("Cannot find /nirs group: {}", e)))?;

    // Read data from data1
    let data_group = nirs_group.group("data1")
        .map_err(|e| BidsError::Csv(format!("Cannot find /nirs/data1: {}", e)))?;

    // dataTimeSeries: samples × channels (f64 or f32)
    let ts_dataset = data_group.dataset("dataTimeSeries")
        .map_err(|e| BidsError::Csv(format!("Cannot find dataTimeSeries: {}", e)))?;

    let shape = ts_dataset.shape();
    if shape.len() != 2 {
        return Err(BidsError::Csv(format!("Expected 2D dataTimeSeries, got {}D", shape.len())));
    }
    let n_samples = shape[0];
    let n_channels = shape[1];

    // Read as flat f64 array, then reshape
    let flat: Vec<f64> = ts_dataset.read_raw()
        .map_err(|e| BidsError::Csv(format!("Cannot read dataTimeSeries: {}", e)))?;

    // Transpose: SNIRF is samples×channels (row-major), we want channels×samples
    let mut data = vec![Vec::with_capacity(n_samples); n_channels];
    for s in 0..n_samples {
        for ch in 0..n_channels {
            data[ch].push(flat[s * n_channels + ch]);
        }
    }

    // Read time vector
    let time_dataset = data_group.dataset("time")
        .map_err(|e| BidsError::Csv(format!("Cannot find time dataset: {}", e)))?;
    let time: Vec<f64> = time_dataset.read_raw()
        .map_err(|e| BidsError::Csv(format!("Cannot read time: {}", e)))?;

    // Read measurement list (best effort)
    let mut measurement_list = Vec::new();
    for i in 1..=n_channels {
        let ml_name = format!("measurementList{}", i);
        if let Ok(ml_i) = data_group.group(&ml_name) {
            let src = ml_i.dataset("sourceIndex")
                .and_then(|d| d.read_scalar::<i32>()).unwrap_or(0) as u32;
            let det = ml_i.dataset("detectorIndex")
                .and_then(|d| d.read_scalar::<i32>()).unwrap_or(0) as u32;
            let wl = ml_i.dataset("wavelengthIndex")
                .and_then(|d| d.read_scalar::<i32>()).unwrap_or(0) as f64;
            measurement_list.push((src, det, wl));
        } else {
            measurement_list.push((0, 0, 0.0));
        }
    }

    let channel_names: Vec<String> = (1..=n_channels).map(|i| format!("ch{}", i)).collect();

    Ok(NirsData {
        data,
        time,
        channel_names,
        n_channels,
        n_samples,
        measurement_list,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nirs_data_struct() {
        let data = NirsData {
            data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            time: vec![0.0, 0.1, 0.2],
            channel_names: vec!["ch1".into(), "ch2".into()],
            n_channels: 2,
            n_samples: 3,
            measurement_list: vec![(1, 1, 760.0), (1, 1, 850.0)],
        };
        assert_eq!(data.n_channels, 2);
        assert!((data.sfreq() - 10.0).abs() < 0.01);
    }
}
