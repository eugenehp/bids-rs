//! High-level EEG interface on top of [`BidsLayout`].
//!
//! [`EegLayout`] wraps a `BidsLayout` to provide EEG-specific queries:
//! listing EEG files, reading channels/electrodes/events/metadata, loading
//! signal data, and generating dataset summaries.

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_io::tsv::read_tsv_gz;
use bids_layout::BidsLayout;

use crate::channels::{Channel, read_channels_tsv};
use crate::coordsystem::CoordinateSystem;
use crate::data::{Annotation, EegData, ReadOptions, read_brainvision_markers, read_eeg_data};
use crate::electrodes::{Electrode, read_electrodes_tsv};
use crate::events::{EegEvent, read_events_tsv};
use crate::metadata::EegMetadata;

/// High-level EEG-specific interface built on top of BidsLayout.
pub struct EegLayout<'a> {
    layout: &'a BidsLayout,
}

impl<'a> EegLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self {
        Self { layout }
    }

    pub fn get_eeg_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("eeg").collect()
    }
    pub fn get_eeg_files_for_subject(&self, subject: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("eeg").subject(subject).collect()
    }
    pub fn get_eeg_files_for_task(&self, task: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("eeg").task(task).collect()
    }

    pub fn get_channels(&self, f: &BidsFile) -> Result<Option<Vec<Channel>>> {
        bids_core::try_read_companion(&f.companion("channels", "tsv"), read_channels_tsv)
    }
    pub fn get_electrodes(&self, f: &BidsFile) -> Result<Option<Vec<Electrode>>> {
        bids_core::try_read_companion(&f.companion("electrodes", "tsv"), read_electrodes_tsv)
    }
    pub fn get_events(&self, f: &BidsFile) -> Result<Option<Vec<EegEvent>>> {
        bids_core::try_read_companion(&f.companion("events", "tsv"), read_events_tsv)
    }
    pub fn get_eeg_metadata(&self, f: &BidsFile) -> Result<Option<EegMetadata>> {
        let md = self.layout.get_metadata(&f.path)?;
        Ok(EegMetadata::from_metadata(&md))
    }
    pub fn get_coordsystem(&self, f: &BidsFile) -> Result<Option<CoordinateSystem>> {
        let p = f.companion("coordsystem", "json");
        if p.exists() {
            return Ok(Some(CoordinateSystem::from_file(&p)?));
        }
        // Relaxed: try without task entity
        let stem = f.filename.split('.').next().unwrap_or("");
        let base_parts: Vec<&str> = stem
            .split('_')
            .filter(|p| p.starts_with("sub-") || p.starts_with("ses-") || p.starts_with("acq-"))
            .collect();
        if !base_parts.is_empty() {
            let relaxed = f
                .dirname
                .join(format!("{}_coordsystem.json", base_parts.join("_")));
            if relaxed.exists() {
                return Ok(Some(CoordinateSystem::from_file(&relaxed)?));
            }
        }
        Ok(None)
    }

    pub fn get_physio(&self, f: &BidsFile) -> Result<Option<PhysioData>> {
        let p = f.companion("physio", "tsv.gz");
        if !p.exists() {
            return Ok(None);
        }
        let rows = read_tsv_gz(&p)?;
        let json_path = p.with_extension("").with_extension("json");
        let md = if json_path.exists() {
            Some(bids_io::json::read_json_sidecar(&json_path)?)
        } else {
            None
        };
        let sampling_freq = md
            .as_ref()
            .and_then(|m| m.get_f64("SamplingFrequency"))
            .unwrap_or(1.0);
        let start_time = md
            .as_ref()
            .and_then(|m| m.get_f64("StartTime"))
            .unwrap_or(0.0);
        let columns: Vec<String> = md
            .as_ref()
            .and_then(|m| m.get_array("Columns"))
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        Ok(Some(PhysioData {
            rows,
            sampling_frequency: sampling_freq,
            start_time,
            columns,
        }))
    }

    pub fn read_edf_header(&self, f: &BidsFile) -> Result<Option<EdfHeader>> {
        if !f.filename.ends_with(".edf") && !f.filename.ends_with(".bdf") {
            return Ok(None);
        }
        EdfHeader::from_file(&f.path).map(Some)
    }

    /// Read the actual signal data from an EEG data file.
    ///
    /// Automatically detects the format from the file extension (.edf, .bdf, .vhdr)
    /// and returns the multichannel time-series data in physical units.
    ///
    /// For BrainVision files, the companion binary file (.eeg/.dat) is resolved
    /// relative to the .vhdr header file.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use bids_layout::BidsLayout;
    /// use bids_eeg::{EegLayout, ReadOptions};
    ///
    /// # let layout = BidsLayout::new("/path").unwrap();
    /// let eeg = EegLayout::new(&layout);
    /// let files = eeg.get_eeg_files().unwrap();
    /// if let Some(f) = files.first() {
    ///     let data = eeg.read_data(f, &ReadOptions::default()).unwrap();
    ///     println!("{} channels, {} samples", data.n_channels(), data.n_samples(0));
    /// }
    /// ```
    pub fn read_data(&self, f: &BidsFile, opts: &ReadOptions) -> Result<EegData> {
        read_eeg_data(&f.path, opts)
    }

    /// Read the signal data from an EEG file, selecting specific channels.
    ///
    /// Convenience wrapper around [`read_data`](Self::read_data) that pre-fills
    /// the channel filter in [`ReadOptions`].
    pub fn read_data_channels(&self, f: &BidsFile, channels: &[&str]) -> Result<EegData> {
        let opts = ReadOptions::new().with_channels(
            channels
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
        );
        self.read_data(f, &opts)
    }

    /// Read a time window from an EEG data file.
    ///
    /// Convenience wrapper around [`read_data`](Self::read_data) that pre-fills
    /// the time range in [`ReadOptions`].
    pub fn read_data_time_range(&self, f: &BidsFile, start: f64, end: f64) -> Result<EegData> {
        let opts = ReadOptions::new().with_time_range(start, end);
        self.read_data(f, &opts)
    }

    /// Read annotations/markers from a BrainVision .vmrk file associated with
    /// the given EEG file.
    pub fn get_brainvision_markers(
        &self,
        f: &BidsFile,
        sampling_rate: f64,
    ) -> Result<Vec<Annotation>> {
        let vmrk_path = f.path.with_extension("vmrk");
        if vmrk_path.exists() {
            read_brainvision_markers(&vmrk_path, sampling_rate)
        } else {
            Ok(Vec::new())
        }
    }

    pub fn get_all_channels_files(&self) -> Result<Vec<BidsFile>> {
        self.layout
            .get()
            .suffix("channels")
            .datatype("eeg")
            .extension("tsv")
            .collect()
    }
    pub fn get_all_electrodes_files(&self) -> Result<Vec<BidsFile>> {
        self.layout
            .get()
            .suffix("electrodes")
            .datatype("eeg")
            .extension("tsv")
            .collect()
    }
    pub fn get_all_events_files(&self) -> Result<Vec<BidsFile>> {
        self.layout
            .get()
            .suffix("events")
            .datatype("eeg")
            .extension("tsv")
            .collect()
    }
    pub fn get_eeg_subjects(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("eeg").return_unique("subject")
    }
    pub fn get_eeg_tasks(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("eeg").return_unique("task")
    }

    pub fn summary(&self) -> Result<EegDatasetSummary> {
        let eeg_files = self.get_eeg_files()?;
        let subjects = self.get_eeg_subjects()?;
        let tasks = self.get_eeg_tasks()?;
        let sampling_frequency = eeg_files
            .first()
            .and_then(|f| self.get_eeg_metadata(f).ok().flatten())
            .map(|m| m.sampling_frequency);
        let channel_count = eeg_files
            .first()
            .and_then(|f| self.get_channels(f).ok().flatten())
            .map(|c| c.len());
        Ok(EegDatasetSummary {
            n_subjects: subjects.len(),
            n_recordings: eeg_files.len(),
            subjects,
            tasks,
            sampling_frequency,
            channel_count,
        })
    }
}

#[derive(Debug)]
pub struct EegDatasetSummary {
    pub n_subjects: usize,
    pub n_recordings: usize,
    pub subjects: Vec<String>,
    pub tasks: Vec<String>,
    pub sampling_frequency: Option<f64>,
    pub channel_count: Option<usize>,
}

impl std::fmt::Display for EegDatasetSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "EEG Dataset Summary:")?;
        writeln!(f, "  Subjects: {}", self.n_subjects)?;
        writeln!(f, "  Recordings: {}", self.n_recordings)?;
        writeln!(f, "  Tasks: {:?}", self.tasks)?;
        if let Some(sf) = self.sampling_frequency {
            writeln!(f, "  Sampling Frequency: {sf} Hz")?;
        }
        if let Some(cc) = self.channel_count {
            writeln!(f, "  Channels: {cc}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct PhysioData {
    pub rows: Vec<std::collections::HashMap<String, String>>,
    pub sampling_frequency: f64,
    pub start_time: f64,
    pub columns: Vec<String>,
}

#[derive(Debug)]
pub struct EdfHeader {
    pub version: String,
    pub patient_id: String,
    pub recording_id: String,
    pub n_channels: usize,
    pub n_records: i64,
    pub record_duration: f64,
    pub sampling_rates: Vec<f64>,
    pub channel_labels: Vec<String>,
}

impl EdfHeader {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut header = vec![0u8; 256];
        file.read_exact(&mut header)?;
        let version = String::from_utf8_lossy(&header[0..8]).trim().to_string();
        let patient_id = String::from_utf8_lossy(&header[8..88]).trim().to_string();
        let recording_id = String::from_utf8_lossy(&header[88..168]).trim().to_string();
        let n_channels: usize = String::from_utf8_lossy(&header[252..256])
            .trim()
            .parse()
            .unwrap_or(0);
        let n_records: i64 = String::from_utf8_lossy(&header[236..244])
            .trim()
            .parse()
            .unwrap_or(-1);
        let record_duration: f64 = String::from_utf8_lossy(&header[244..252])
            .trim()
            .parse()
            .unwrap_or(1.0);
        let ext_size = n_channels * 256;
        let mut ext_header = vec![0u8; ext_size];
        file.read_exact(&mut ext_header)?;
        let mut channel_labels = Vec::new();
        for i in 0..n_channels {
            let offset = i * 16;
            if offset + 16 <= ext_header.len() {
                channel_labels.push(
                    String::from_utf8_lossy(&ext_header[offset..offset + 16])
                        .trim()
                        .to_string(),
                );
            }
        }
        let mut sampling_rates = Vec::new();
        let samples_offset = n_channels * 216;
        for i in 0..n_channels {
            let offset = samples_offset + i * 8;
            if offset + 8 <= ext_header.len() {
                let samples: f64 = String::from_utf8_lossy(&ext_header[offset..offset + 8])
                    .trim()
                    .parse()
                    .unwrap_or(0.0);
                if record_duration > 0.0 {
                    sampling_rates.push(samples / record_duration);
                }
            }
        }
        Ok(Self {
            version,
            patient_id,
            recording_id,
            n_channels,
            n_records,
            record_duration,
            sampling_rates,
            channel_labels,
        })
    }
    pub fn duration(&self) -> f64 {
        if self.n_records > 0 {
            self.n_records as f64 * self.record_duration
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_companion() {
        let bf = BidsFile::new("/data/sub-01/eeg/sub-01_task-rest_eeg.edf");
        assert_eq!(
            bf.companion("channels", "tsv").to_string_lossy(),
            "/data/sub-01/eeg/sub-01_task-rest_channels.tsv"
        );
        assert_eq!(
            bf.companion("events", "tsv").to_string_lossy(),
            "/data/sub-01/eeg/sub-01_task-rest_events.tsv"
        );
    }
}
