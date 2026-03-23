//! EEG signal data reading: EDF, BDF, and BrainVision formats.
//!
//! Reads raw signal data from all BIDS-EEG formats into [`EegData`], a
//! channel × samples matrix with physical-unit values, annotations, and
//! stimulus channel detection. Implements the [`TimeSeries`](bids_core::timeseries::TimeSeries)
//! trait for modality-agnostic processing.

use bids_core::error::{BidsError, Result};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// ─── Annotation ────────────────────────────────────────────────────────────────

/// A time-stamped annotation from EDF+, BDF+, or BrainVision marker files.
///
/// Corresponds to MNE's `raw.annotations`. EDF+ TAL (Time-stamped Annotation
/// Lists) entries and BrainVision `.vmrk` markers are both parsed into this type.
#[derive(Debug, Clone, PartialEq)]
pub struct Annotation {
    /// Onset time in seconds from the start of the recording.
    pub onset: f64,
    /// Duration in seconds (0.0 if instantaneous).
    pub duration: f64,
    /// Description / label of the annotation.
    pub description: String,
}

// ─── EegData ───────────────────────────────────────────────────────────────────

/// Raw EEG signal data read from a data file.
///
/// Stores multichannel time-series data as a channel × samples matrix,
/// where each inner `Vec<f64>` represents one channel's signal in physical units.
/// Also carries annotations parsed from the file (EDF+ TAL, BDF+ status, or
/// BrainVision markers).
#[derive(Clone)]
pub struct EegData {
    /// Channel labels in order.
    pub channel_labels: Vec<String>,
    /// Signal data: one `Vec<f64>` per channel, all in physical units (e.g., µV).
    pub data: Vec<Vec<f64>>,
    /// Sampling rate per channel in Hz.
    pub sampling_rates: Vec<f64>,
    /// Total duration in seconds.
    pub duration: f64,
    /// Annotations parsed from the data file (EDF+ TAL, BDF+ status, .vmrk markers).
    pub annotations: Vec<Annotation>,
    /// Indices of channels detected as stimulus/trigger channels.
    pub stim_channel_indices: Vec<usize>,
    /// Whether this recording is from an EDF+D/BDF+D discontinuous file.
    /// If true, there may be gaps in the time axis — use `record_onsets`
    /// to reconstruct the true timeline.
    pub is_discontinuous: bool,
    /// Actual onset time of each data record in seconds (from EDF+ TAL).
    /// For continuous recordings this is empty or `[0, dur, 2*dur, ...]`.
    /// For discontinuous (EDF+D), these give the real timestamps of each record,
    /// which may have gaps.
    pub record_onsets: Vec<f64>,
}

impl std::fmt::Debug for EegData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EegData")
            .field("n_channels", &self.data.len())
            .field("n_samples", &self.data.first().map_or(0, std::vec::Vec::len))
            .field("channel_labels", &self.channel_labels)
            .field("sampling_rates", &self.sampling_rates)
            .field("duration", &self.duration)
            .field("annotations", &self.annotations.len())
            .field("is_discontinuous", &self.is_discontinuous)
            .finish()
    }
}

impl std::fmt::Display for EegData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sr = self.sampling_rates.first().copied().unwrap_or(0.0);
        write!(f, "EegData({} ch × {} samples @ {:.0} Hz, {:.1}s)",
            self.data.len(),
            self.data.first().map_or(0, std::vec::Vec::len),
            sr,
            self.duration)
    }
}

impl EegData {
    /// Number of channels.
    #[inline]
    pub fn n_channels(&self) -> usize { self.data.len() }

    /// Number of samples for the given channel index.
    #[inline]
    pub fn n_samples(&self, channel: usize) -> usize {
        self.data.get(channel).map_or(0, std::vec::Vec::len)
    }

    /// Get a single channel's data by index.
    #[inline]
    pub fn channel(&self, index: usize) -> Option<&[f64]> {
        self.data.get(index).map(std::vec::Vec::as_slice)
    }

    /// Get a single channel's data by label.
    pub fn channel_by_name(&self, name: &str) -> Option<&[f64]> {
        let idx = self.channel_labels.iter().position(|l| l == name)?;
        self.channel(idx)
    }

    /// Generate a times array in seconds for the given channel (like MNE's `raw.times`).
    ///
    /// Returns one f64 per sample: `[0.0, 1/sr, 2/sr, ...]`.
    pub fn times(&self, channel: usize) -> Option<Vec<f64>> {
        let n = self.n_samples(channel);
        if n == 0 { return None; }
        let sr = self.sampling_rates.get(channel).copied().unwrap_or(1.0);
        Some((0..n).map(|i| i as f64 / sr).collect())
    }

    /// Get the data together with the times array (like MNE's `get_data(return_times=True)`).
    pub fn get_data_with_times(&self) -> (Vec<Vec<f64>>, Vec<f64>) {
        let times = self.times(0).unwrap_or_default();
        (self.data.clone(), times)
    }

    /// Select a subset of channels by name (include), returning a new `EegData`.
    #[must_use]
    pub fn select_channels(&self, names: &[&str]) -> EegData {
        let mut labels = Vec::with_capacity(names.len());
        let mut data = Vec::with_capacity(names.len());
        let mut rates = Vec::with_capacity(names.len());
        let mut stim = Vec::new();
        for name in names {
            if let Some(idx) = self.channel_labels.iter().position(|l| l == *name) {
                if self.stim_channel_indices.contains(&idx) {
                    stim.push(labels.len());
                }
                labels.push(self.channel_labels[idx].clone());
                data.push(self.data[idx].clone());
                rates.push(self.sampling_rates[idx]);
            }
        }
        EegData {
            channel_labels: labels, data, sampling_rates: rates,
            duration: self.duration, annotations: self.annotations.clone(),
            stim_channel_indices: stim, is_discontinuous: false, record_onsets: Vec::new(),
        }
    }

    /// Exclude channels by name, returning a new `EegData` with all other channels.
    #[must_use]
    pub fn exclude_channels(&self, names: &[&str]) -> EegData {
        let keep: Vec<&str> = self.channel_labels.iter()
            .filter(|l| !names.contains(&l.as_str()))
            .map(std::string::String::as_str)
            .collect();
        self.select_channels(&keep)
    }

    /// Extract a time window (in seconds) from all channels.
    #[must_use]
    pub fn time_slice(&self, start_sec: f64, end_sec: f64) -> EegData {
        let mut data = Vec::with_capacity(self.data.len());
        let mut rates = Vec::with_capacity(self.data.len());
        for (i, ch_data) in self.data.iter().enumerate() {
            let sr = self.sampling_rates[i];
            let start_sample = ((start_sec * sr).round() as usize).min(ch_data.len());
            let end_sample = ((end_sec * sr).round() as usize).min(ch_data.len());
            data.push(ch_data[start_sample..end_sample].to_vec());
            rates.push(sr);
        }
        // Filter annotations to the time window
        let anns = self.annotations.iter()
            .filter(|a| a.onset + a.duration >= start_sec && a.onset < end_sec)
            .map(|a| Annotation {
                onset: (a.onset - start_sec).max(0.0),
                duration: a.duration,
                description: a.description.clone(),
            })
            .collect();
        EegData {
            channel_labels: self.channel_labels.clone(),
            data, sampling_rates: rates,
            duration: end_sec - start_sec,
            annotations: anns,
            stim_channel_indices: self.stim_channel_indices.clone(), is_discontinuous: self.is_discontinuous, record_onsets: self.record_onsets.clone(),
        }
    }

    /// Convert channel data to different units by applying a scale factor.
    ///
    /// `unit_map` maps channel name → scale factor. For example, to convert
    /// from µV to V: `{"EEG1": 1e-6}`.
    pub fn convert_units(&mut self, unit_map: &std::collections::HashMap<String, f64>) {
        for (i, label) in self.channel_labels.iter().enumerate() {
            if let Some(&scale) = unit_map.get(label) {
                for v in &mut self.data[i] {
                    *v *= scale;
                }
            }
        }
    }

    /// Select channels by type (like MNE's `pick_types`).
    ///
    /// `types` should be a list of `ChannelType` variants to keep. Requires
    /// that `channel_types` is available (from channels.tsv).
    #[must_use]
    pub fn pick_types(&self, types: &[crate::ChannelType], channel_types: &[crate::ChannelType]) -> EegData {
        let names: Vec<&str> = self.channel_labels.iter().enumerate()
            .filter(|(i, _)| channel_types.get(*i).is_some_and(|ct| types.contains(ct)))
            .map(|(_, name)| name.as_str())
            .collect();
        self.select_channels(&names)
    }

    /// Concatenate another `EegData` in time (appending samples).
    ///
    /// Both must have the same channels in the same order. Annotations from
    /// `other` are time-shifted by `self.duration`.
    pub fn concatenate(&mut self, other: &EegData) -> std::result::Result<(), String> {
        if self.channel_labels != other.channel_labels {
            return Err("Channel labels must match for concatenation".into());
        }
        if self.data.len() != other.data.len() {
            return Err("Channel count must match for concatenation".into());
        }
        let time_offset = self.duration;
        for (i, ch) in self.data.iter_mut().enumerate() {
            ch.extend_from_slice(&other.data[i]);
        }
        for ann in &other.annotations {
            self.annotations.push(Annotation {
                onset: ann.onset + time_offset,
                duration: ann.duration,
                description: ann.description.clone(),
            });
        }
        self.duration += other.duration;
        Ok(())
    }

    /// Remove (zero-out) data segments that overlap with annotations matching
    /// a description pattern (like MNE's `reject_by_annotation`).
    ///
    /// Returns a new `EegData` where samples overlapping "BAD" annotations
    /// (or annotations matching `pattern`) are replaced with `f64::NAN`.
    #[must_use]
    pub fn reject_by_annotation(&self, pattern: &str) -> EegData {
        let mut new_data = self.data.clone();
        let pattern_upper = pattern.to_uppercase();

        for ann in &self.annotations {
            if !ann.description.to_uppercase().contains(&pattern_upper) {
                continue;
            }
            for (ch, ch_data) in new_data.iter_mut().enumerate() {
                let sr = self.sampling_rates[ch];
                let start = (ann.onset * sr).round() as usize;
                let end = ((ann.onset + ann.duration) * sr).round() as usize;
                let start = start.min(ch_data.len());
                let end = end.min(ch_data.len());
                for v in &mut ch_data[start..end] {
                    *v = f64::NAN;
                }
            }
        }

        EegData {
            channel_labels: self.channel_labels.clone(),
            data: new_data,
            sampling_rates: self.sampling_rates.clone(),
            duration: self.duration,
            annotations: self.annotations.clone(),
            stim_channel_indices: self.stim_channel_indices.clone(), is_discontinuous: self.is_discontinuous, record_onsets: self.record_onsets.clone(),
        }
    }
}

// ─── MNE-inspired signal processing methods ────────────────────────────────────

impl EegData {
    /// Apply a bandpass filter to all channels (like MNE's `raw.filter(l_freq, h_freq)`).
    ///
    /// Uses a Butterworth IIR filter with zero-phase `filtfilt` application.
    /// `l_freq` and `h_freq` are in Hz. If `l_freq` is `None`, applies lowpass only.
    /// If `h_freq` is `None`, applies highpass only.
    #[must_use]
    pub fn filter(&self, l_freq: Option<f64>, h_freq: Option<f64>, order: usize) -> EegData {
        let sr = self.sampling_rates.first().copied().unwrap_or(1.0);
        let nyquist = sr / 2.0;
        let mut new_data = self.data.clone();

        for ch_data in &mut new_data {
            let filtered = match (l_freq, h_freq) {
                (Some(lo), Some(hi)) => {
                    let (b, a) = bids_filter::butter_bandpass(
                        order,
                        (lo / nyquist).clamp(0.001, 0.999),
                        (hi / nyquist).clamp(0.001, 0.999),
                    );
                    bids_filter::filtfilt(&b, &a, ch_data)
                }
                (Some(lo), None) => {
                    let (b, a) = bids_filter::butter_highpass(order, (lo / nyquist).clamp(0.001, 0.999));
                    bids_filter::filtfilt(&b, &a, ch_data)
                }
                (None, Some(hi)) => {
                    let (b, a) = bids_filter::butter_lowpass(order, (hi / nyquist).clamp(0.001, 0.999));
                    bids_filter::filtfilt(&b, &a, ch_data)
                }
                (None, None) => continue,
            };
            *ch_data = filtered;
        }

        EegData {
            data: new_data,
            ..self.clone()
        }
    }

    /// Remove power line noise at `freq` Hz and its harmonics (like MNE's `raw.notch_filter()`).
    #[must_use]
    pub fn notch_filter(&self, freq: f64, quality: f64) -> EegData {
        let sr = self.sampling_rates.first().copied().unwrap_or(1.0);
        let mut new_data = self.data.clone();
        let nyquist = sr / 2.0;

        // Apply notch at fundamental and harmonics up to Nyquist
        let mut f = freq;
        while f < nyquist {
            for ch in &mut new_data {
                *ch = bids_filter::notch_filter(ch, f, sr, quality);
            }
            f += freq;
        }

        EegData {
            data: new_data,
            ..self.clone()
        }
    }

    /// Resample all channels to a new sampling rate (like MNE's `raw.resample()`).
    ///
    /// Applies an anti-aliasing lowpass filter before downsampling.
    #[must_use]
    pub fn resample(&self, new_sr: f64) -> EegData {
        let old_sr = self.sampling_rates.first().copied().unwrap_or(1.0);
        let new_data: Vec<Vec<f64>> = self.data.iter()
            .map(|ch| bids_filter::resample(ch, old_sr, new_sr))
            .collect();
        let new_duration = new_data.first().map_or(0.0, |ch| ch.len() as f64 / new_sr);

        EegData {
            data: new_data,
            sampling_rates: vec![new_sr; self.channel_labels.len()],
            duration: new_duration,
            channel_labels: self.channel_labels.clone(),
            annotations: self.annotations.clone(),
            stim_channel_indices: self.stim_channel_indices.clone(),
            is_discontinuous: self.is_discontinuous,
            record_onsets: Vec::new(),
        }
    }

    /// Re-reference to the average of all channels (like MNE's `raw.set_eeg_reference('average')`).
    ///
    /// Subtracts the mean across channels at each time point.
    #[must_use]
    pub fn set_average_reference(&self) -> EegData {
        let n_ch = self.data.len();
        let n_s = self.data.first().map_or(0, std::vec::Vec::len);
        if n_ch == 0 || n_s == 0 { return self.clone(); }

        let mut new_data = self.data.clone();

        for s in 0..n_s {
            let mean: f64 = self.data.iter().map(|ch| ch[s]).sum::<f64>() / n_ch as f64;
            for ch in &mut new_data {
                ch[s] -= mean;
            }
        }

        EegData { data: new_data, ..self.clone() }
    }

    /// Re-reference to a specific channel (like MNE's `raw.set_eeg_reference([ch_name])`).
    #[must_use]
    pub fn set_reference(&self, ref_channel: &str) -> EegData {
        let ref_idx = match self.channel_labels.iter().position(|l| l == ref_channel) {
            Some(i) => i,
            None => return self.clone(),
        };
        let ref_data = self.data[ref_idx].clone();
        let mut new_data = self.data.clone();

        for (i, ch) in new_data.iter_mut().enumerate() {
            if i != ref_idx {
                for (s, v) in ch.iter_mut().enumerate() {
                    *v -= ref_data[s];
                }
            }
        }

        EegData { data: new_data, ..self.clone() }
    }

    /// Extract epochs around events (like MNE's `mne.Epochs(raw, events, tmin, tmax)`).
    ///
    /// Returns a Vec of `EegData`, one per event matching `event_desc`.
    /// `tmin` and `tmax` are relative to event onset in seconds.
    /// If `event_desc` is `None`, epochs around all annotations.
    pub fn epoch(
        &self, tmin: f64, tmax: f64, event_desc: Option<&str>,
    ) -> Vec<EegData> {
        let sr = self.sampling_rates.first().copied().unwrap_or(1.0);
        let events: Vec<&Annotation> = self.annotations.iter()
            .filter(|a| event_desc.is_none_or(|d| a.description == d))
            .collect();

        let n_before = ((-tmin) * sr).round() as usize;
        let n_after = (tmax * sr).round() as usize;
        let epoch_len = n_before + n_after;

        let mut epochs = Vec::with_capacity(events.len());
        for event in &events {
            let center = (event.onset * sr).round() as isize;
            let start = center - n_before as isize;
            let end = center + n_after as isize;

            // Skip if epoch would go out of bounds
            let n_samples = self.data.first().map_or(0, std::vec::Vec::len) as isize;
            if start < 0 || end > n_samples { continue; }

            let start = start as usize;
            let data: Vec<Vec<f64>> = self.data.iter()
                .map(|ch| ch[start..start + epoch_len].to_vec())
                .collect();

            epochs.push(EegData {
                channel_labels: self.channel_labels.clone(),
                data,
                sampling_rates: self.sampling_rates.clone(),
                duration: epoch_len as f64 / sr,
                annotations: vec![Annotation {
                    onset: -tmin,
                    duration: event.duration,
                    description: event.description.clone(),
                }],
                stim_channel_indices: self.stim_channel_indices.clone(),
                is_discontinuous: false,
                record_onsets: Vec::new(),
            });
        }
        epochs
    }

    /// Average a list of epochs to compute an ERP (Event-Related Potential).
    ///
    /// All epochs must have the same shape. Like MNE's `epochs.average()`.
    pub fn average_epochs(epochs: &[EegData]) -> Option<EegData> {
        if epochs.is_empty() { return None; }
        let n_ch = epochs[0].data.len();
        let n_s = epochs[0].data.first().map_or(0, std::vec::Vec::len);
        let n_epochs = epochs.len() as f64;

        let mut avg_data = vec![vec![0.0; n_s]; n_ch];
        for epoch in epochs {
            for (ch, ch_data) in epoch.data.iter().enumerate() {
                for (s, &v) in ch_data.iter().enumerate() {
                    if ch < n_ch && s < n_s {
                        avg_data[ch][s] += v;
                    }
                }
            }
        }
        for ch in &mut avg_data {
            for v in ch.iter_mut() {
                *v /= n_epochs;
            }
        }

        Some(EegData {
            channel_labels: epochs[0].channel_labels.clone(),
            data: avg_data,
            sampling_rates: epochs[0].sampling_rates.clone(),
            duration: epochs[0].duration,
            annotations: Vec::new(),
            stim_channel_indices: epochs[0].stim_channel_indices.clone(),
            is_discontinuous: false,
            record_onsets: Vec::new(),
        })
    }

    /// Compute power spectral density using Welch's method (like MNE's `raw.compute_psd()`).
    ///
    /// Returns `(frequencies, psd)` where `psd[ch]` is the power spectrum for each channel.
    /// `n_fft` is the FFT window size (default: sampling rate = 1 Hz resolution).
    pub fn compute_psd(&self, n_fft: Option<usize>) -> (Vec<f64>, Vec<Vec<f64>>) {
        let sr = self.sampling_rates.first().copied().unwrap_or(1.0);
        let n_fft = n_fft.unwrap_or(sr as usize);
        let n_freqs = n_fft / 2 + 1;

        let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * sr / n_fft as f64).collect();

        let psd: Vec<Vec<f64>> = self.data.iter()
            .map(|ch| welch_psd(ch, n_fft, sr))
            .collect();

        (freqs, psd)
    }
}

/// Simple Welch PSD estimate using the periodogram method.
///
/// Splits the signal into overlapping segments, computes the squared magnitude
/// of the DFT for each, and averages. Uses a Hann window.
fn welch_psd(x: &[f64], n_fft: usize, _fs: f64) -> Vec<f64> {
    let n_freqs = n_fft / 2 + 1;
    if x.len() < n_fft {
        return vec![0.0; n_freqs];
    }

    let hop = n_fft / 2; // 50% overlap
    let n_segments = (x.len() - n_fft) / hop + 1;
    let mut psd = vec![0.0; n_freqs];

    // Hann window
    let window: Vec<f64> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n_fft - 1) as f64).cos()))
        .collect();
    let window_power: f64 = window.iter().map(|w| w * w).sum::<f64>();

    for seg in 0..n_segments {
        let start = seg * hop;
        // Apply window and compute DFT magnitudes via Goertzel / direct DFT
        for (k, psd_bin) in psd.iter_mut().enumerate() {
            let freq = 2.0 * std::f64::consts::PI * k as f64 / n_fft as f64;
            let mut re = 0.0;
            let mut im = 0.0;
            for n in 0..n_fft {
                let windowed = x[start + n] * window[n];
                re += windowed * (freq * n as f64).cos();
                im -= windowed * (freq * n as f64).sin();
            }
            *psd_bin += (re * re + im * im) / window_power;
        }
    }

    // Average over segments and scale
    let scale = 1.0 / n_segments as f64;
    for v in &mut psd { *v *= scale; }

    psd
}

#[cfg(feature = "safetensors")]
impl EegData {
    /// Save signal data as a safetensors file.
    ///
    /// The file contains one tensor "data" with shape [n_channels, n_samples]
    /// and dtype f64, plus metadata with channel names and sampling rate.
    pub fn save_safetensors(&self, path: &std::path::Path) -> std::io::Result<()> {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let n_ch = self.n_channels();
        let n_s = self.n_samples(0);

        // Flatten channels × samples into a contiguous buffer
        let mut flat = Vec::with_capacity(n_ch * n_s);
        for ch in &self.data {
            flat.extend_from_slice(ch);
        }

        let flat_bytes: &[u8] = bytemuck::cast_slice(&flat);

        let tensor = TensorView::new(
            Dtype::F64,
            vec![n_ch, n_s],
            flat_bytes,
        ).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let mut tensors = HashMap::new();
        tensors.insert("data".to_string(), tensor);

        // Store metadata as JSON in the safetensors header
        let mut metadata = HashMap::new();
        metadata.insert("channel_names".to_string(), serde_json::to_string(&self.channel_labels).unwrap_or_default());
        metadata.insert("sampling_rate".to_string(), self.sampling_rates.first().map(|r| r.to_string()).unwrap_or_default());
        metadata.insert("duration".to_string(), self.duration.to_string());
        metadata.insert("n_channels".to_string(), n_ch.to_string());
        metadata.insert("n_samples".to_string(), n_s.to_string());

        let bytes = safetensors::tensor::serialize(&tensors, &Some(metadata))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        std::fs::write(path, bytes)
    }
}

#[cfg(feature = "ndarray")]
impl EegData {
    /// Convert to an ndarray Array2<f64> with shape (n_channels, n_samples).
    pub fn to_ndarray(&self) -> ndarray::Array2<f64> {
        let n_ch = self.n_channels();
        let n_s = self.n_samples(0);
        let mut arr = ndarray::Array2::zeros((n_ch, n_s));
        for (i, ch) in self.data.iter().enumerate() {
            for (j, &v) in ch.iter().enumerate() {
                arr[(i, j)] = v;
            }
        }
        arr
    }

    /// Create from an ndarray Array2<f64> (n_channels × n_samples).
    pub fn from_ndarray(
        arr: &ndarray::Array2<f64>, channel_labels: Vec<String>, sampling_rate: f64,
    ) -> Self {
        let n_ch = arr.nrows();
        let n_s = arr.ncols();
        let data: Vec<Vec<f64>> = (0..n_ch)
            .map(|i| arr.row(i).to_vec())
            .collect();
        Self {
            channel_labels,
            data,
            sampling_rates: vec![sampling_rate; n_ch],
            duration: n_s as f64 / sampling_rate,
            annotations: Vec::new(),
            stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        }
    }
}

impl bids_core::timeseries::TimeSeries for EegData {
    fn n_channels(&self) -> usize { self.data.len() }
    fn n_samples(&self) -> usize { self.data.first().map_or(0, std::vec::Vec::len) }
    fn channel_names(&self) -> &[String] { &self.channel_labels }
    fn sampling_rate(&self) -> f64 { self.sampling_rates.first().copied().unwrap_or(1.0) }
    fn channel_data(&self, index: usize) -> Option<&[f64]> { self.data.get(index).map(std::vec::Vec::as_slice) }
    fn duration(&self) -> f64 { self.duration }
}

// ─── ReadOptions ───────────────────────────────────────────────────────────────

/// Options for reading EEG data files.
#[derive(Debug, Clone, Default)]
pub struct ReadOptions {
    /// If set, only read these channels (by name). Corresponds to MNE's `include`.
    pub channels: Option<Vec<String>>,
    /// If set, exclude these channels by name. Applied after `channels`.
    /// Corresponds to MNE's `exclude` parameter.
    pub exclude: Option<Vec<String>>,
    /// If set, only read starting from this time in seconds.
    pub start_time: Option<f64>,
    /// If set, only read up to this time in seconds.
    pub end_time: Option<f64>,
    /// Override stim channel detection. If `Some`, these channel names are
    /// treated as stimulus channels. If `None`, auto-detection is used
    /// (channels named "Status", "Trigger", "STI", case-insensitive).
    pub stim_channel: Option<Vec<String>>,
}

impl ReadOptions {
    pub fn new() -> Self { Self::default() }

    /// Include only these channels (like MNE's `include` / `picks`).
    pub fn with_channels(mut self, channels: Vec<String>) -> Self {
        self.channels = Some(channels); self
    }

    /// Exclude these channels (like MNE's `exclude`).
    pub fn with_exclude(mut self, exclude: Vec<String>) -> Self {
        self.exclude = Some(exclude); self
    }

    pub fn with_time_range(mut self, start: f64, end: f64) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// Override stim channel names. Pass empty vec to disable auto-detection.
    pub fn with_stim_channel(mut self, names: Vec<String>) -> Self {
        self.stim_channel = Some(names); self
    }
}

// ─── EDF / BDF Reader ─────────────────────────────────────────────────────────

/// Read EEG data from an EDF (European Data Format) or BDF (BioSemi) file.
///
/// Parses the header to determine channel layout, then reads the raw data records
/// and converts digital values to physical units using the calibration parameters.
///
/// Handles EDF files where `n_records` is `-1` (unknown) by computing the
/// record count from the file size.
///
/// Performance notes:
/// - Uses buffered I/O and bulk reads (single `read_exact` for all data)
/// - Pre-computes a channel→output lookup table (no hash or linear scan in hot loop)
/// - Channel-major decode loop for sequential output writes (cache-friendly)
/// - Branchless format dispatch: EDF (i16) and BDF (i24) use separate decode paths
pub fn read_edf(path: &Path, opts: &ReadOptions) -> Result<EegData> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::with_capacity(256 * 1024, file);

    let mut hdr_buf = [0u8; 256];
    reader.read_exact(&mut hdr_buf)?;

    let is_bdf = hdr_buf[0] == 0xFF;
    let bytes_per_sample: usize = if is_bdf { 3 } else { 2 };

    let n_channels: usize = parse_header_int(&hdr_buf[252..256])?;
    let n_records: i64 = parse_header_int(&hdr_buf[236..244])?;
    let record_duration: f64 = parse_header_f64(&hdr_buf[244..252])?;

    if n_channels == 0 {
        return Err(BidsError::DataFormat("EDF/BDF file has 0 channels".into()));
    }
    if n_channels > 10_000 {
        return Err(BidsError::DataFormat(format!(
            "EDF/BDF file claims {n_channels} channels — likely corrupt header")));
    }

    // Detect EDF+ / BDF+ from the reserved field
    let reserved = String::from_utf8_lossy(&hdr_buf[192..236]).trim().to_string();
    let is_edf_plus = reserved.starts_with("EDF+");
    let is_edf_plus_discontinuous = reserved.contains("EDF+D") || reserved.contains("BDF+D");

    // Read extended header
    let ext_size = n_channels * 256;
    let mut ext = vec![0u8; ext_size];
    reader.read_exact(&mut ext)?;

    // Parse per-channel fields
    let mut labels = Vec::with_capacity(n_channels);
    let mut phys_min = Vec::with_capacity(n_channels);
    let mut phys_max = Vec::with_capacity(n_channels);
    let mut dig_min = Vec::with_capacity(n_channels);
    let mut dig_max = Vec::with_capacity(n_channels);
    let mut samples_per_record = Vec::with_capacity(n_channels);

    for i in 0..n_channels {
        labels.push(read_field(&ext, i, 16, 0));
        phys_min.push(read_field_f64(&ext, i, 8, n_channels * 104)?);
        phys_max.push(read_field_f64(&ext, i, 8, n_channels * 112)?);
        dig_min.push(read_field_f64(&ext, i, 8, n_channels * 120)?);
        dig_max.push(read_field_f64(&ext, i, 8, n_channels * 128)?);
        samples_per_record.push(read_field_int(&ext, i, 8, n_channels * 216)?);
    }

    let sampling_rates: Vec<f64> = samples_per_record.iter()
        .map(|&s| if record_duration > 0.0 { s as f64 / record_duration } else { s as f64 })
        .collect();

    // Identify EDF+ annotation channels and BDF status channels
    let annotation_channel_indices: Vec<usize> = labels.iter().enumerate()
        .filter(|(_, l)| l.as_str() == "EDF Annotations" || l.as_str() == "BDF Status")
        .map(|(i, _)| i)
        .collect();

    // Detect stim channels: user-specified, or auto-detect by name
    let stim_names: Vec<String> = if let Some(ref names) = opts.stim_channel {
        names.clone()
    } else {
        // Auto-detect: channels named "status", "trigger", "sti *" (case insensitive)
        labels.iter()
            .filter(|l| {
                let lower = l.to_lowercase();
                lower == "status" || lower == "trigger" || lower.starts_with("sti ")
                    || lower.starts_with("sti\t")
            })
            .cloned()
            .collect()
    };

    // Determine which channels to read (include then exclude)
    let mut channel_indices: Vec<usize> = if let Some(ref wanted) = opts.channels {
        wanted.iter().filter_map(|name| labels.iter().position(|l| l == name)).collect()
    } else {
        // By default, exclude annotation channels from signal data
        (0..n_channels)
            .filter(|i| !annotation_channel_indices.contains(i))
            .collect()
    };

    // Apply exclude
    if let Some(ref excl) = opts.exclude {
        let excl_indices: Vec<usize> = excl.iter()
            .filter_map(|name| labels.iter().position(|l| l == name))
            .collect();
        channel_indices.retain(|i| !excl_indices.contains(i));
    }

    // Map stim channel names to output indices
    let stim_channel_indices: Vec<usize> = channel_indices.iter().enumerate()
        .filter(|(_, ch)| stim_names.iter().any(|n| n == &labels[**ch]))
        .map(|(out_idx, _)| out_idx)
        .collect();

    // Pre-compute channel→output-index lookup table (O(1) per channel in hot loop)
    let mut ch_to_out: Vec<usize> = vec![usize::MAX; n_channels];
    for (out_idx, &ch) in channel_indices.iter().enumerate() {
        ch_to_out[ch] = out_idx;
    }

    // Also need to read annotation channels even if excluded from output
    let need_annotation_read = is_edf_plus || is_bdf
        || annotation_channel_indices.iter().any(|i| labels[*i] == "BDF Status");
    let mut ann_ch_to_out: Vec<usize> = vec![usize::MAX; n_channels];
    if need_annotation_read {
        for (ann_idx, &ch) in annotation_channel_indices.iter().enumerate() {
            ann_ch_to_out[ch] = ann_idx;
        }
    }

    // EDF spec: n_records == -1 means "unknown" — compute from file size.
    let n_records = if n_records < 0 {
        let record_bytes: usize = samples_per_record.iter()
            .map(|&s| s * bytes_per_sample)
            .sum();
        let file_len = reader.seek(SeekFrom::End(0))? as usize;
        let header_size = 256 + ext_size;
        if record_bytes > 0 && file_len > header_size {
            ((file_len - header_size) / record_bytes) as i64
        } else {
            0
        }
    } else {
        n_records
    };
    let total_duration = n_records as f64 * record_duration;

    let start_record = opts.start_time
        .map(|t| ((t / record_duration).floor() as i64).clamp(0, n_records) as usize)
        .unwrap_or(0);
    let end_record = opts.end_time
        .map(|t| ((t / record_duration).ceil() as i64).clamp(0, n_records) as usize)
        .unwrap_or(n_records as usize);

    if start_record >= end_record {
        return Ok(EegData {
            channel_labels: channel_indices.iter().map(|&i| labels[i].clone()).collect(),
            data: vec![Vec::new(); channel_indices.len()],
            sampling_rates: channel_indices.iter().map(|&i| sampling_rates[i]).collect(),
            duration: 0.0,
            annotations: Vec::new(),
            stim_channel_indices: stim_channel_indices.clone(), is_discontinuous: false, record_onsets: Vec::new(),
        });
    }

    let records_to_read = end_record - start_record;

    // Calculate record size in bytes and per-channel byte offsets within a record
    let mut ch_byte_offsets = Vec::with_capacity(n_channels);
    let mut offset = 0usize;
    for (ch, &spr) in samples_per_record.iter().enumerate() {
        ch_byte_offsets.push(offset);
        let ch_bytes = spr.checked_mul(bytes_per_sample)
            .ok_or_else(|| BidsError::DataFormat(format!(
                "Channel {ch} samples_per_record ({spr}) overflows byte calculation")))?;
        offset = offset.checked_add(ch_bytes)
            .ok_or_else(|| BidsError::DataFormat("Record size overflow".into()))?;
    }
    let record_byte_size = offset;

    // Pre-allocate output with exact sizes
    let mut out_data: Vec<Vec<f64>> = channel_indices.iter()
        .map(|&i| vec![0.0f64; samples_per_record[i] as usize * records_to_read])
        .collect();

    // Pre-compute gain and offset for each channel: physical = digital * gain + cal_offset
    // where gain = (phys_max - phys_min) / (dig_max - dig_min)
    //       cal_offset = phys_min - dig_min * gain
    let mut gains = vec![0.0f64; n_channels];
    let mut cal_offsets = vec![0.0f64; n_channels];
    for i in 0..n_channels {
        let dd = dig_max[i] - dig_min[i];
        let pd = phys_max[i] - phys_min[i];
        let g = if dd.abs() > f64::EPSILON { pd / dd } else { 1.0 };
        gains[i] = g;
        cal_offsets[i] = phys_min[i] - dig_min[i] * g;
    }

    // Seek to start record — bulk read all needed data at once
    let data_start = 256 + ext_size + start_record * record_byte_size;
    reader.seek(SeekFrom::Start(data_start as u64))?;

    let total_data_bytes = records_to_read * record_byte_size;
    let mut all_data = vec![0u8; total_data_bytes];
    reader.read_exact(&mut all_data)?;

    // Decode — split by format to avoid branch in inner loop
    let params = DecodeParams {
        all_data: &all_data, records_to_read, n_channels,
        samples_per_record: &samples_per_record, ch_byte_offsets: &ch_byte_offsets,
        ch_to_out: &ch_to_out, gains: &gains, cal_offsets: &cal_offsets,
        record_byte_size,
    };
    if is_bdf {
        decode_records_bdf(&params, &mut out_data);
    } else {
        decode_records_edf(&params, &mut out_data);
    }

    // Extract annotation channel raw bytes (if EDF+/BDF+)
    let mut annotations = Vec::new();
    let mut record_onsets: Vec<f64> = Vec::new(); // actual onset time of each record (for EDF+D)
    if need_annotation_read && !annotation_channel_indices.is_empty() {
        for &ann_ch in &annotation_channel_indices {
            let label = &labels[ann_ch];
            if label == "EDF Annotations" {
                // Parse TAL (Time-stamped Annotation Lists) from each record
                let spr = samples_per_record[ann_ch];
                let ch_bytes = spr * bytes_per_sample;
                for rec in 0..records_to_read {
                    let rec_base = rec * record_byte_size;
                    let start = rec_base + ch_byte_offsets[ann_ch];
                    let end = start + ch_bytes;
                    if end <= all_data.len() {
                        let tal_bytes = &all_data[start..end];
                        if let Some(onset) = parse_edf_tal(tal_bytes, &mut annotations) {
                            record_onsets.push(onset);
                        }
                    }
                }
            } else if label == "BDF Status" {
                // BDF status channel: extract trigger events from the 24-bit status word.
                // prev_val must persist across record boundaries to avoid duplicate
                // events when a trigger is held high across the boundary.
                let spr = samples_per_record[ann_ch];
                let mut prev_val: i32 = 0;
                for rec in 0..records_to_read {
                    let rec_base = rec * record_byte_size;
                    let src = &all_data[rec_base + ch_byte_offsets[ann_ch]..];
                    let t_base = (start_record + rec) as f64 * record_duration;
                    for s in 0..spr {
                        let off = s * 3;
                        let b0 = src[off] as u32;
                        let b1 = src[off + 1] as u32;
                        let b2 = src[off + 2] as u32;
                        let raw_val = (b0 | (b1 << 8) | (b2 << 16)) as i32;
                        // Lower 16 bits are the trigger value in BDF
                        let trigger = raw_val & 0xFFFF;
                        if trigger != 0 && trigger != prev_val {
                            let onset = t_base + s as f64 / (spr as f64 / record_duration);
                            annotations.push(Annotation {
                                onset,
                                duration: 0.0,
                                description: format!("{trigger}"),
                            });
                        }
                        prev_val = trigger;
                    }
                }
            }
        }
    }

    // Sub-record precision trimming
    trim_time_range(&mut out_data, &channel_indices, &sampling_rates,
                    record_duration, start_record, opts);

    let actual_duration = (opts.end_time.unwrap_or(total_duration)
        - opts.start_time.unwrap_or(0.0)).min(total_duration);

    Ok(EegData {
        channel_labels: channel_indices.iter().map(|&i| labels[i].clone()).collect(),
        data: out_data,
        sampling_rates: channel_indices.iter().map(|&i| sampling_rates[i]).collect(),
        duration: actual_duration,
        annotations,
        stim_channel_indices, is_discontinuous: is_edf_plus_discontinuous, record_onsets,
    })
}

/// Bulk decode EDF (16-bit) records.
/// Shared parameters for EDF/BDF record decoding.
struct DecodeParams<'a> {
    all_data: &'a [u8],
    records_to_read: usize,
    n_channels: usize,
    samples_per_record: &'a [usize],
    ch_byte_offsets: &'a [usize],
    ch_to_out: &'a [usize],
    gains: &'a [f64],
    cal_offsets: &'a [f64],
    record_byte_size: usize,
}

/// Bulk decode EDF (16-bit) records.
///
/// Iterates channels in the outer loop so that writes to each output
/// channel's `Vec<f64>` are sequential, giving much better cache locality
/// on the write side. The input reads stride by `record_byte_size` which
/// is typically in L2 cache for common channel counts.
#[inline(never)]
fn decode_records_edf(p: &DecodeParams, out_data: &mut [Vec<f64>]) {
    for ch in 0..p.n_channels {
        let out_idx = p.ch_to_out[ch];
        if out_idx == usize::MAX { continue; }

        let n_samples = p.samples_per_record[ch];
        let gain = p.gains[ch];
        let cal_offset = p.cal_offsets[ch];
        let ch_off = p.ch_byte_offsets[ch];
        let dst = &mut out_data[out_idx];

        for rec in 0..p.records_to_read {
            let src = &p.all_data[rec * p.record_byte_size + ch_off..];
            let dst_start = rec * n_samples;

            for s in 0..n_samples {
                let off = s * 2;
                let digital = i16::from_le_bytes([src[off], src[off + 1]]) as f64;
                dst[dst_start + s] = digital * gain + cal_offset;
            }
        }
    }
}

/// Bulk decode BDF (24-bit) records.
///
/// Channel-major iteration for sequential output writes.
#[inline(never)]
fn decode_records_bdf(p: &DecodeParams, out_data: &mut [Vec<f64>]) {
    for ch in 0..p.n_channels {
        let out_idx = p.ch_to_out[ch];
        if out_idx == usize::MAX { continue; }

        let n_samples = p.samples_per_record[ch];
        let gain = p.gains[ch];
        let cal_offset = p.cal_offsets[ch];
        let ch_off = p.ch_byte_offsets[ch];
        let dst = &mut out_data[out_idx];

        for rec in 0..p.records_to_read {
            let src = &p.all_data[rec * p.record_byte_size + ch_off..];
            let dst_start = rec * n_samples;

            for s in 0..n_samples {
                let off = s * 3;
                let b0 = src[off] as u32;
                let b1 = src[off + 1] as u32;
                let b2 = src[off + 2] as u32;
                let val = b0 | (b1 << 8) | (b2 << 16);
                // Sign extend from 24-bit
                let digital = if val & 0x800000 != 0 {
                    (val | 0xFF000000) as i32
                } else {
                    val as i32
                } as f64;
                dst[dst_start + s] = digital * gain + cal_offset;
            }
        }
    }
}

/// Trim samples at sub-record precision for time range requests.
fn trim_time_range(
    out_data: &mut [Vec<f64>],
    channel_indices: &[usize],
    sampling_rates: &[f64],
    record_duration: f64,
    start_record: usize,
    opts: &ReadOptions,
) {
    // Trim from the start
    if let Some(start_t) = opts.start_time {
        let record_start_t = start_record as f64 * record_duration;
        if start_t > record_start_t {
            for (out_idx, &ch) in channel_indices.iter().enumerate() {
                let skip = ((start_t - record_start_t) * sampling_rates[ch]).round() as usize;
                if skip > 0 && skip < out_data[out_idx].len() {
                    // In-place shift via drain — avoids reallocation
                    out_data[out_idx].drain(..skip);
                }
            }
        }
    }
    // Trim from the end
    if let Some(end_t) = opts.end_time {
        let actual_start = opts.start_time.unwrap_or(0.0);
        let desired_dur = end_t - actual_start;
        for (out_idx, &ch) in channel_indices.iter().enumerate() {
            let max_samples = (desired_dur * sampling_rates[ch]).round() as usize;
            out_data[out_idx].truncate(max_samples);
        }
    }
}

// ─── EDF+ TAL Parser ───────────────────────────────────────────────────────────

/// Parse EDF+ Time-stamped Annotation Lists (TAL) from raw annotation channel bytes.
///
/// TAL format per record:
/// `+T\x14\x14\x00` — time-keeping annotation (onset only, gives record start time)
/// `+T\x15D\x14description\x14\x00` — annotation with onset T, duration D, description
///
/// Returns the record onset time (from the first TAL entry without a description).
/// This is critical for EDF+D (discontinuous) files where records may not be contiguous.
fn parse_edf_tal(data: &[u8], annotations: &mut Vec<Annotation>) -> Option<f64> {
    let mut record_onset = None;
    // Split on \x00 to get individual TAL entries
    for entry in data.split(|&b| b == 0) {
        if entry.is_empty() { continue; }

        let s = String::from_utf8_lossy(entry);
        let s = s.trim_matches(|c: char| c == '\x14' || c == '\x00' || c == '\x15');
        if s.is_empty() { continue; }

        // Split on \x14 (annotation separator)
        let parts: Vec<&str> = s.split('\x14').collect();
        if parts.is_empty() { continue; }

        // First part: onset and optional duration
        let onset_dur = parts[0];
        let (onset, duration) = if let Some(dur_sep) = onset_dur.find('\x15') {
            let onset_str = &onset_dur[..dur_sep];
            let dur_str = &onset_dur[dur_sep + 1..];
            (
                onset_str.trim_start_matches('+').parse::<f64>().unwrap_or(0.0),
                dur_str.parse::<f64>().unwrap_or(0.0),
            )
        } else {
            (onset_dur.trim_start_matches('+').parse::<f64>().unwrap_or(0.0), 0.0)
        };

        // Remaining parts are descriptions
        let has_description = parts[1..].iter().any(|d| !d.trim().is_empty());
        if !has_description && record_onset.is_none() {
            // First TAL entry without description = record onset time
            record_onset = Some(onset);
        }
        for desc in &parts[1..] {
            let desc = desc.trim();
            if desc.is_empty() { continue; }
            annotations.push(Annotation {
                onset,
                duration,
                description: desc.to_string(),
            });
        }
    }
    record_onset
}

// ─── BrainVision Reader ────────────────────────────────────────────────────────

/// Read EEG data from BrainVision format (.vhdr + .eeg/.dat).
///
/// Parses the `.vhdr` header to determine data layout, then reads the binary
/// data file and applies channel-specific resolution scaling.
///
/// Performance notes:
/// - Single bulk read of entire binary file
/// - For INT_16/INT_32: batch decode with pre-computed resolution
/// - For IEEE_FLOAT_32: safe reinterpret via `from_le_bytes` batched over slices
/// - Vectorized layout gets direct contiguous slice access per channel
pub fn read_brainvision(vhdr_path: &Path, opts: &ReadOptions) -> Result<EegData> {
    let header_text = std::fs::read_to_string(vhdr_path)?;
    let bv = parse_vhdr(&header_text)?;

    let parent = vhdr_path.parent().unwrap_or(Path::new("."));
    let data_path = parent.join(&bv.data_file);
    if !data_path.exists() {
        return Err(BidsError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("BrainVision data file not found: {}", data_path.display()),
        )));
    }

    let n_channels = bv.channels.len();
    let bps = bv.bytes_per_sample();
    let raw_data = std::fs::read(&data_path)?;
    let total_samples = raw_data.len() / bps / n_channels;
    let sampling_rate = bv.sampling_interval_us.map(|us| 1_000_000.0 / us).unwrap_or(1.0);

    // Channel selection: include then exclude
    let mut channel_indices: Vec<usize> = if let Some(ref wanted) = opts.channels {
        wanted.iter().filter_map(|name| bv.channels.iter().position(|c| c.name == *name)).collect()
    } else {
        (0..n_channels).collect()
    };

    if let Some(ref excl) = opts.exclude {
        let excl_indices: Vec<usize> = excl.iter()
            .filter_map(|name| bv.channels.iter().position(|c| c.name == *name))
            .collect();
        channel_indices.retain(|i| !excl_indices.contains(i));
    }

    // Stim channel detection
    let stim_names: Vec<String> = if let Some(ref names) = opts.stim_channel {
        names.clone()
    } else {
        bv.channels.iter()
            .filter(|c| {
                let lower = c.name.to_lowercase();
                lower == "status" || lower == "trigger" || lower.starts_with("sti ")
            })
            .map(|c| c.name.clone())
            .collect()
    };
    let stim_channel_indices: Vec<usize> = channel_indices.iter().enumerate()
        .filter(|(_, ch)| stim_names.iter().any(|n| n == &bv.channels[**ch].name))
        .map(|(out_idx, _)| out_idx)
        .collect();

    let start_sample = opts.start_time
        .map(|t| (t * sampling_rate).round() as usize)
        .unwrap_or(0)
        .min(total_samples);
    let end_sample = opts.end_time
        .map(|t| (t * sampling_rate).round() as usize)
        .unwrap_or(total_samples)
        .min(total_samples);

    let n_out = end_sample.saturating_sub(start_sample);

    let out_data = if bv.data_orientation == BvOrientation::Multiplexed {
        decode_bv_multiplexed(&raw_data, &bv, &channel_indices, start_sample, n_out, n_channels, bps)
    } else {
        decode_bv_vectorized(&raw_data, &bv, &channel_indices, start_sample, n_out, total_samples, n_channels, bps)
    };

    // Read .vmrk marker file if it exists
    let annotations = if let Some(ref marker_file) = bv.marker_file {
        let vmrk_path = parent.join(marker_file);
        if vmrk_path.exists() {
            let vmrk_text = std::fs::read_to_string(&vmrk_path)?;
            parse_vmrk(&vmrk_text, sampling_rate)
        } else {
            Vec::new()
        }
    } else {
        // Try conventional name: same stem as .vhdr but .vmrk
        let vmrk_path = vhdr_path.with_extension("vmrk");
        if vmrk_path.exists() {
            let vmrk_text = std::fs::read_to_string(&vmrk_path)?;
            parse_vmrk(&vmrk_text, sampling_rate)
        } else {
            Vec::new()
        }
    };

    Ok(EegData {
        channel_labels: channel_indices.iter().map(|&i| bv.channels[i].name.clone()).collect(),
        data: out_data,
        sampling_rates: vec![sampling_rate; channel_indices.len()],
        duration: n_out as f64 / sampling_rate,
        annotations,
        stim_channel_indices, is_discontinuous: false, record_onsets: Vec::new(),
    })
}

/// Decode multiplexed BrainVision data using cache-friendly tiled decoding.
/// Layout: [ch0_s0, ch1_s0, ..., chN_s0, ch0_s1, ch1_s1, ...]
///
/// For multiplexed data, per-channel iteration re-reads the entire file from
/// memory for each channel (N×file_size memory traffic), while per-sample
/// iteration scatters writes across N output buffers.
///
/// Tiled approach: process in blocks of TILE samples. Within each tile,
/// both the input block and output tile fit in L2 cache, giving good locality
/// for both reads and writes.
#[inline(never)]
fn decode_bv_multiplexed(
    raw: &[u8], bv: &BvHeader, indices: &[usize],
    start: usize, count: usize, n_ch: usize, bps: usize,
) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = indices.iter()
        .map(|_| vec![0.0f64; count])
        .collect();

    // Build ch→(out_idx, resolution) lookup
    let mut ch_map: Vec<(usize, f64)> = vec![(usize::MAX, 0.0); n_ch];
    for (out_idx, &ch) in indices.iter().enumerate() {
        ch_map[ch] = (out_idx, bv.channels[ch].resolution);
    }

    let frame_bytes = n_ch * bps;

    // Tile size: chosen so tile_input + tile_output fits in L2 cache (~256KB).
    // tile_input = TILE * n_ch * bps, tile_output = TILE * n_out * 8
    // For 64ch × 2B: input = TILE*128, output = TILE*64*8 = TILE*512
    // Total = TILE * 640 → TILE = 256K/640 ≈ 400. Use 512 for power-of-2.
    const TILE: usize = 512;
    let n_out = indices.len();

    match bv.data_format {
        BvDataFormat::Int16 => {
            let mut s = 0;
            while s < count {
                let tile_end = (s + TILE).min(count);
                for oi in 0..n_out {
                    let ch = indices[oi];
                    let res = ch_map[ch].1;
                    let dst = &mut out[oi][s..tile_end];
                    let mut base = (start + s) * frame_bytes + ch * 2;
                    for d in dst.iter_mut() {
                        *d = i16::from_le_bytes([raw[base], raw[base + 1]]) as f64 * res;
                        base += frame_bytes;
                    }
                }
                s = tile_end;
            }
        }
        BvDataFormat::Float32 => {
            let mut s = 0;
            while s < count {
                let tile_end = (s + TILE).min(count);
                for oi in 0..n_out {
                    let ch = indices[oi];
                    let res = ch_map[ch].1;
                    let dst = &mut out[oi][s..tile_end];
                    let mut base = (start + s) * frame_bytes + ch * 4;
                    for d in dst.iter_mut() {
                        *d = f32::from_le_bytes([raw[base], raw[base+1], raw[base+2], raw[base+3]]) as f64 * res;
                        base += frame_bytes;
                    }
                }
                s = tile_end;
            }
        }
        BvDataFormat::Int32 => {
            let mut s = 0;
            while s < count {
                let tile_end = (s + TILE).min(count);
                for oi in 0..n_out {
                    let ch = indices[oi];
                    let res = ch_map[ch].1;
                    let dst = &mut out[oi][s..tile_end];
                    let mut base = (start + s) * frame_bytes + ch * 4;
                    for d in dst.iter_mut() {
                        *d = i32::from_le_bytes([raw[base], raw[base+1], raw[base+2], raw[base+3]]) as f64 * res;
                        base += frame_bytes;
                    }
                }
                s = tile_end;
            }
        }
    }
    out
}

/// Decode vectorized BrainVision data.
/// Layout: [ch0_s0, ch0_s1, ..., ch0_sN, ch1_s0, ch1_s1, ...]
#[inline(never)]
fn decode_bv_vectorized(
    raw: &[u8], bv: &BvHeader, indices: &[usize],
    start: usize, count: usize, total: usize, _n_ch: usize, bps: usize,
) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = indices.iter()
        .map(|_| vec![0.0f64; count])
        .collect();

    let ch_stride = total * bps; // bytes per channel's contiguous block

    match bv.data_format {
        BvDataFormat::Int16 => {
            for (out_idx, &ch) in indices.iter().enumerate() {
                let res = bv.channels[ch].resolution;
                let ch_base = ch * ch_stride + start * 2;
                let src = &raw[ch_base..];
                for (s, d) in out[out_idx].iter_mut().enumerate() {
                    let off = s * 2;
                    *d = i16::from_le_bytes([src[off], src[off + 1]]) as f64 * res;
                }
            }
        }
        BvDataFormat::Float32 => {
            for (out_idx, &ch) in indices.iter().enumerate() {
                let res = bv.channels[ch].resolution;
                let ch_base = ch * ch_stride + start * 4;
                let src = &raw[ch_base..];
                for (s, d) in out[out_idx].iter_mut().enumerate() {
                    let off = s * 4;
                    *d = f32::from_le_bytes([src[off], src[off+1], src[off+2], src[off+3]]) as f64 * res;
                }
            }
        }
        BvDataFormat::Int32 => {
            for (out_idx, &ch) in indices.iter().enumerate() {
                let res = bv.channels[ch].resolution;
                let ch_base = ch * ch_stride + start * 4;
                let src = &raw[ch_base..];
                for (s, d) in out[out_idx].iter_mut().enumerate() {
                    let off = s * 4;
                    *d = i32::from_le_bytes([src[off], src[off+1], src[off+2], src[off+3]]) as f64 * res;
                }
            }
        }
    }
    out
}

// ─── Unified reader ────────────────────────────────────────────────────────────

/// Detect format from file extension and read EEG data.
///
/// Supported formats:
/// - `.edf` — European Data Format
/// - `.bdf` — BioSemi Data Format
/// - `.vhdr` — BrainVision (reads companion `.eeg`/`.dat` file)
pub fn read_eeg_data(path: &Path, opts: &ReadOptions) -> Result<EegData> {
    let ext = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "edf" | "bdf" => read_edf(path, opts),
        "vhdr" => read_brainvision(path, opts),
        "set" => read_eeglab_set(path, opts),
        _ => Err(BidsError::FileType(format!(
            "Unsupported EEG data format: .{ext}. Supported: .edf, .bdf, .vhdr, .set"
        ))),
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────────

fn parse_header_int<T: std::str::FromStr>(bytes: &[u8]) -> Result<T> {
    String::from_utf8_lossy(bytes)
        .trim()
        .parse::<T>()
        .map_err(|_| BidsError::Csv(format!(
            "Failed to parse header field: '{}'",
            String::from_utf8_lossy(bytes).trim()
        )))
}

fn parse_header_f64(bytes: &[u8]) -> Result<f64> { parse_header_int(bytes) }

fn read_field(ext: &[u8], ch: usize, width: usize, base_offset: usize) -> String {
    let offset = base_offset + ch * width;
    if offset + width <= ext.len() {
        String::from_utf8_lossy(&ext[offset..offset + width]).trim().to_string()
    } else {
        String::new()
    }
}

fn read_field_f64(ext: &[u8], ch: usize, width: usize, base_offset: usize) -> Result<f64> {
    let s = read_field(ext, ch, width, base_offset);
    s.parse::<f64>().map_err(|_| BidsError::Csv(format!(
        "Failed to parse channel {ch} field at offset {base_offset}: '{s}'"
    )))
}

fn read_field_int(ext: &[u8], ch: usize, width: usize, base_offset: usize) -> Result<usize> {
    let s = read_field(ext, ch, width, base_offset);
    s.parse::<usize>().map_err(|_| BidsError::Csv(format!(
        "Failed to parse channel {ch} field at offset {base_offset}: '{s}'"
    )))
}

// ─── BrainVision header parsing ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum BvOrientation { Multiplexed, Vectorized }

#[derive(Debug, Clone, PartialEq)]
enum BvDataFormat { Int16, Float32, Int32 }

#[derive(Debug, Clone)]
struct BvChannel { name: String, resolution: f64 }

#[derive(Debug, Clone)]
struct BvHeader {
    data_file: String,
    marker_file: Option<String>,
    data_format: BvDataFormat,
    data_orientation: BvOrientation,
    channels: Vec<BvChannel>,
    sampling_interval_us: Option<f64>,
}

impl BvHeader {
    fn bytes_per_sample(&self) -> usize {
        match self.data_format {
            BvDataFormat::Int16 => 2,
            BvDataFormat::Float32 | BvDataFormat::Int32 => 4,
        }
    }
}

fn parse_vhdr(text: &str) -> Result<BvHeader> {
    let mut data_file = String::new();
    let mut marker_file: Option<String> = None;
    let mut data_format = BvDataFormat::Int16;
    let mut orientation = BvOrientation::Multiplexed;
    let mut sampling_interval: Option<f64> = None;
    let mut channels = Vec::new();
    let mut section = String::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') { continue; }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].to_lowercase();
            continue;
        }

        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim();
            let value = value.trim();
            match section.as_str() {
                "common infos" => match key {
                    "DataFile" => data_file = value.to_string(),
                    "MarkerFile" => marker_file = Some(value.to_string()),
                    "DataOrientation" => {
                        orientation = if value.to_uppercase().contains("VECTORIZED") {
                            BvOrientation::Vectorized
                        } else { BvOrientation::Multiplexed };
                    }
                    "SamplingInterval" => { sampling_interval = value.parse().ok(); }
                    _ => {}
                },
                "binary infos" => if key == "BinaryFormat" {
                    data_format = match value.to_uppercase().as_str() {
                        "IEEE_FLOAT_32" => BvDataFormat::Float32,
                        "INT_32" => BvDataFormat::Int32,
                        _ => BvDataFormat::Int16,
                    };
                },
                "channel infos" => if key.starts_with("Ch") || key.starts_with("ch") {
                    let parts: Vec<&str> = value.splitn(4, ',').collect();
                    let name = parts.first().map(|s| s.trim().to_string()).unwrap_or_default();
                    let resolution = parts.get(2).and_then(|s| s.trim().parse::<f64>().ok()).unwrap_or(1.0);
                    channels.push(BvChannel { name, resolution });
                },
                _ => {}
            }
        }
    }

    if data_file.is_empty() {
        return Err(BidsError::Csv("BrainVision header missing DataFile".into()));
    }
    if channels.is_empty() {
        return Err(BidsError::Csv("BrainVision header has no channels".into()));
    }

    Ok(BvHeader { data_file, marker_file, data_format, data_orientation: orientation, channels, sampling_interval_us: sampling_interval })
}

// ─── BrainVision .vmrk marker parser ───────────────────────────────────────────

/// Parse BrainVision marker file (.vmrk) into annotations.
///
/// Marker format: `Mk<n>=<type>,<description>,<position>,<size>,<channel>,<date>`
/// where position is 1-indexed sample number.
fn parse_vmrk(text: &str, sampling_rate: f64) -> Vec<Annotation> {
    let mut annotations = Vec::new();
    let mut section = String::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') { continue; }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].to_lowercase();
            continue;
        }

        if section == "marker infos" {
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                if key.starts_with("Mk") || key.starts_with("mk") {
                    // Mk1=Stimulus,S  1,26214,1,0
                    // type, description, position, size, channel[, date]
                    let parts: Vec<&str> = value.splitn(6, ',').collect();
                    if parts.len() >= 3 {
                        let marker_type = parts[0].trim();
                        let description = parts[1].trim();
                        let position: usize = parts[2].trim().parse().unwrap_or(1);
                        let size: usize = parts.get(3)
                            .and_then(|s| s.trim().parse().ok())
                            .unwrap_or(1);

                        // Position is 1-indexed sample number
                        let onset = (position.saturating_sub(1)) as f64 / sampling_rate;
                        let duration = if size > 1 {
                            size as f64 / sampling_rate
                        } else {
                            0.0
                        };

                        // Build description like MNE: "type/description" or just description
                        let desc = if marker_type.is_empty()
                            || marker_type == "Stimulus"
                            || marker_type == "Response"
                            || marker_type == "Comment"
                        {
                            description.to_string()
                        } else {
                            format!("{marker_type}/{description}")
                        };

                        if !desc.is_empty() {
                            annotations.push(Annotation { onset, duration, description: desc });
                        }
                    }
                }
            }
        }
    }

    annotations
}

/// Read BrainVision markers from a .vmrk file directly.
///
/// This is a standalone function for reading markers without reading the
/// signal data. Useful for event-only analysis.
pub fn read_brainvision_markers(vmrk_path: &Path, sampling_rate: f64) -> Result<Vec<Annotation>> {
    let text = std::fs::read_to_string(vmrk_path)?;
    Ok(parse_vmrk(&text, sampling_rate))
}

// ─── EEGLAB .set/.fdt reader ───────────────────────────────────────────────────

/// Read EEG data from an EEGLAB `.set` file and its companion `.fdt` data file.
///
/// The `.set` file is a MATLAB MAT v5 file containing metadata (channel names,
/// sampling rate, etc.). The actual signal data is stored in a companion `.fdt`
/// file as a flat binary array of `f32` values in channels × samples order.
///
/// # Limitations
///
/// This reader handles the common case where `.set` metadata is paired with
/// a binary `.fdt` file. Complex `.set` files that embed data directly in
/// the MAT structure (MATLAB v7.3 / HDF5) are not supported — convert to
/// EDF or BrainVision first.
///
/// # Errors
///
/// Returns an error if the `.fdt` companion doesn't exist, or the `.set` file
/// can't be parsed for the required metadata fields.
pub fn read_eeglab_set(path: &Path, opts: &ReadOptions) -> Result<EegData> {
    // Read the MAT v5 file to extract basic metadata
    let set_bytes = std::fs::read(path)?;

    // Parse minimal metadata from MAT v5 header + data elements
    let (n_channels, n_samples, srate, channel_labels) = parse_set_metadata(&set_bytes, path)?;

    // Find companion .fdt file
    let fdt_path = path.with_extension("fdt");
    if !fdt_path.exists() {
        return Err(BidsError::DataFormat(format!(
            "Companion .fdt file not found for {}. \
             If the data is embedded in the .set file (MATLAB v7.3/HDF5), \
             convert to EDF or BrainVision format first.",
            path.display()
        )));
    }

    // Read binary .fdt: float32, channels × samples, little-endian
    let fdt_bytes = std::fs::read(&fdt_path)?;
    let expected_size = n_channels * n_samples * 4;
    if fdt_bytes.len() < expected_size {
        return Err(BidsError::DataFormat(format!(
            ".fdt file too small: expected {} bytes ({} ch × {} samp × 4), got {}",
            expected_size, n_channels, n_samples, fdt_bytes.len()
        )));
    }

    let mut data = vec![Vec::with_capacity(n_samples); n_channels];
    #[allow(clippy::needless_range_loop)]
    for s in 0..n_samples {
        for ch in 0..n_channels {
            let offset = (s * n_channels + ch) * 4;
            let val = f32::from_le_bytes([
                fdt_bytes[offset],
                fdt_bytes[offset + 1],
                fdt_bytes[offset + 2],
                fdt_bytes[offset + 3],
            ]);
            data[ch].push(val as f64);
        }
    }

    // Apply channel selection from opts
    let (data, channel_labels) = if let Some(ref include) = opts.channels {
        let mut new_data = Vec::new();
        let mut new_labels = Vec::new();
        for label in include {
            if let Some(idx) = channel_labels.iter().position(|l| l == label) {
                new_data.push(data[idx].clone());
                new_labels.push(label.clone());
            }
        }
        (new_data, new_labels)
    } else if let Some(ref exclude) = opts.exclude {
        let mut new_data = Vec::new();
        let mut new_labels = Vec::new();
        for (idx, label) in channel_labels.iter().enumerate() {
            if !exclude.contains(label) {
                new_data.push(data[idx].clone());
                new_labels.push(label.clone());
            }
        }
        (new_data, new_labels)
    } else {
        (data, channel_labels)
    };

    let n_ch = data.len();
    let duration = if srate > 0.0 { n_samples as f64 / srate } else { 0.0 };

    Ok(EegData {
        channel_labels,
        data,
        sampling_rates: vec![srate; n_ch],
        duration,
        annotations: Vec::new(),
        stim_channel_indices: Vec::new(),
        is_discontinuous: false,
        record_onsets: Vec::new(),
    })
}

/// Parse minimal metadata from a MAT v5 `.set` file.
///
/// Extracts: nbchan (number of channels), pnts (number of samples),
/// srate (sampling rate), and channel labels.
///
/// This is a minimal parser for the MATLAB Level 5 MAT-file format,
/// reading just enough to get the EEG struct's scalar fields and chanlocs.
fn parse_set_metadata(bytes: &[u8], path: &Path) -> Result<(usize, usize, f64, Vec<String>)> {
    // MAT v5 files start with a 128-byte header: 116 bytes text + 8 reserved + 4 version + 2 endian
    if bytes.len() < 128 {
        return Err(BidsError::DataFormat(format!(
            "{}: File too small to be a valid MAT v5 file", path.display()
        )));
    }

    let header_text = String::from_utf8_lossy(&bytes[..116]);
    if !header_text.contains("MATLAB") {
        return Err(BidsError::DataFormat(format!(
            "{}: Not a MATLAB MAT v5 file (header doesn't contain 'MATLAB'). \
             If this is a MATLAB v7.3 (HDF5) file, convert with: \
             pop_saveset(EEG, 'filename', 'output.set', 'savemode', 'onefile', 'version', '7')",
            path.display()
        )));
    }

    // Scan the binary for known field name patterns
    // Look for common EEGLAB field values as ASCII strings
    let text = String::from_utf8_lossy(bytes);

    // Try to find nbchan, pnts, srate by scanning for field names
    // In MAT v5, struct field names are stored as arrays of fixed-width strings
    let mut n_channels = 0usize;
    let mut n_samples = 0usize;
    let mut srate = 0.0f64;
    let mut channel_labels = Vec::new();

    // Heuristic: scan for ASCII patterns that encode the metadata.
    // This is simplified — a full MAT parser would decode the tag/data structure.
    // We look for the pattern: field_name followed by a numeric value.
    for window in bytes.windows(6) {
        if window == b"nbchan" {
            // Look for a double value in the next ~50 bytes
            if let Some(v) = find_next_double(bytes, bytes.len().min(offset_of(bytes, window) + 100), offset_of(bytes, window)) {
                n_channels = v as usize;
            }
        }
        if window[..4] == *b"pnts" {
            if let Some(v) = find_next_double(bytes, bytes.len().min(offset_of(bytes, window) + 100), offset_of(bytes, window)) {
                n_samples = v as usize;
            }
        }
        if window[..5] == *b"srate" {
            if let Some(v) = find_next_double(bytes, bytes.len().min(offset_of(bytes, window) + 100), offset_of(bytes, window)) {
                srate = v;
            }
        }
    }

    // Generate default channel labels if we couldn't parse them from chanlocs
    if channel_labels.is_empty() && n_channels > 0 {
        channel_labels = (0..n_channels).map(|i| format!("EEG{:03}", i + 1)).collect();
    }

    // Try to extract channel labels from chanlocs.labels
    // Look for sequences of short ASCII strings after "labels"
    if let Some(pos) = text.find("labels") {
        let search_region = &bytes[pos..bytes.len().min(pos + n_channels * 20 + 200)];
        let mut labels = Vec::new();
        let mut i = 0;
        while i < search_region.len() && labels.len() < n_channels {
            // Look for runs of printable ASCII that could be channel names
            if search_region[i].is_ascii_alphanumeric() {
                let start = i;
                while i < search_region.len() && search_region[i].is_ascii_graphic() && search_region[i] != 0 {
                    i += 1;
                }
                let candidate = String::from_utf8_lossy(&search_region[start..i]).to_string();
                if candidate.len() >= 2 && candidate.len() <= 10 && candidate != "labels" {
                    labels.push(candidate);
                }
            } else {
                i += 1;
            }
        }
        if labels.len() == n_channels {
            channel_labels = labels;
        }
    }

    if n_channels == 0 || n_samples == 0 || srate == 0.0 {
        return Err(BidsError::DataFormat(format!(
            "{}: Could not extract EEG metadata from .set file \
             (nbchan={}, pnts={}, srate={}). The file may use an unsupported \
             MAT format. Convert with EEGLAB: pop_saveset(EEG, 'savemode', 'onefile')",
            path.display(), n_channels, n_samples, srate
        )));
    }

    Ok((n_channels, n_samples, srate, channel_labels))
}

fn offset_of(haystack: &[u8], needle: &[u8]) -> usize {
    needle.as_ptr() as usize - haystack.as_ptr() as usize
}

fn find_next_double(bytes: &[u8], end: usize, start: usize) -> Option<f64> {
    // MAT v5 stores doubles as 8-byte little-endian IEEE 754
    // Look for a double that makes sense as a positive integer or frequency
    let search = &bytes[start..end.min(bytes.len())];
    for offset in (0..search.len().saturating_sub(7)).step_by(8) {
        let val = f64::from_le_bytes([
            search[offset], search[offset + 1], search[offset + 2], search[offset + 3],
            search[offset + 4], search[offset + 5], search[offset + 6], search[offset + 7],
        ]);
        if val.is_finite() && val > 0.0 && val < 1e9 {
            return Some(val);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_test_edf(path: &Path, n_channels: usize, n_records: usize, samples_per_record: usize) {
        let mut file = std::fs::File::create(path).unwrap();
        let mut hdr = [b' '; 256];
        hdr[0..1].copy_from_slice(b"0");
        hdr[168..176].copy_from_slice(b"01.01.01");
        hdr[176..184].copy_from_slice(b"00.00.00");
        let hs = format!("{:<8}", 256 + n_channels * 256);
        hdr[184..192].copy_from_slice(hs.as_bytes());
        let nr = format!("{:<8}", n_records);
        hdr[236..244].copy_from_slice(nr.as_bytes());
        hdr[244..252].copy_from_slice(b"1       ");
        let nc = format!("{:<4}", n_channels);
        hdr[252..256].copy_from_slice(nc.as_bytes());
        file.write_all(&hdr).unwrap();

        let mut ext = vec![b' '; n_channels * 256];
        for i in 0..n_channels {
            let label = format!("{:<16}", format!("EEG{}", i + 1));
            ext[i * 16..i * 16 + 16].copy_from_slice(label.as_bytes());
            let o = n_channels * 96 + i * 8;
            ext[o..o + 2].copy_from_slice(b"uV");
            let s = format!("{:<8}", "-3200");
            ext[n_channels * 104 + i * 8..n_channels * 104 + i * 8 + 8].copy_from_slice(s.as_bytes());
            let s = format!("{:<8}", "3200");
            ext[n_channels * 112 + i * 8..n_channels * 112 + i * 8 + 8].copy_from_slice(s.as_bytes());
            let s = format!("{:<8}", "-32768");
            ext[n_channels * 120 + i * 8..n_channels * 120 + i * 8 + 8].copy_from_slice(s.as_bytes());
            let s = format!("{:<8}", "32767");
            ext[n_channels * 128 + i * 8..n_channels * 128 + i * 8 + 8].copy_from_slice(s.as_bytes());
            let s = format!("{:<8}", samples_per_record);
            ext[n_channels * 216 + i * 8..n_channels * 216 + i * 8 + 8].copy_from_slice(s.as_bytes());
        }
        file.write_all(&ext).unwrap();

        // Write data records in bulk (one buffer per record)
        let rec_bytes = n_channels * samples_per_record * 2;
        let mut buf = vec![0u8; rec_bytes];
        for rec in 0..n_records {
            for ch in 0..n_channels {
                for s in 0..samples_per_record {
                    let t = rec as f64 + s as f64 / samples_per_record as f64;
                    let value = (1000.0 * (2.0 * std::f64::consts::PI * (ch as f64 + 1.0) * t).sin()) as i16;
                    let off = (ch * samples_per_record + s) * 2;
                    buf[off..off + 2].copy_from_slice(&value.to_le_bytes());
                }
            }
            file.write_all(&buf).unwrap();
        }
    }

    #[test]
    fn test_read_edf_basic() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_edf");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.edf");
        create_test_edf(&path, 3, 2, 256);

        let data = read_edf(&path, &ReadOptions::default()).unwrap();
        assert_eq!(data.n_channels(), 3);
        assert_eq!(data.n_samples(0), 512);
        assert_eq!(data.channel_labels, vec!["EEG1", "EEG2", "EEG3"]);
        assert!((data.sampling_rates[0] - 256.0).abs() < 0.01);
        assert!((data.duration - 2.0).abs() < 0.01);
        for ch_data in &data.data {
            for &v in ch_data {
                assert!(v >= -3200.1 && v <= 3200.1, "Value {} out of range", v);
            }
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_edf_unknown_n_records() {
        // EDF spec allows n_records == -1 meaning "unknown".
        // The reader should compute the count from the file size.
        let dir = std::env::temp_dir().join("bids_eeg_data_test_nrec");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.edf");

        // Create a normal EDF with 2 channels, 3 records, 128 spr
        create_test_edf(&path, 2, 3, 128);

        // Patch the header to set n_records = -1
        let mut bytes = std::fs::read(&path).unwrap();
        let neg1 = format!("{:<8}", "-1");
        bytes[236..244].copy_from_slice(neg1.as_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let data = read_edf(&path, &ReadOptions::default()).unwrap();
        assert_eq!(data.n_channels(), 2);
        assert_eq!(data.n_samples(0), 3 * 128); // should infer 3 records from file size
        assert!((data.duration - 3.0).abs() < 0.01);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_edf_channel_select() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_chsel");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.edf");
        create_test_edf(&path, 4, 1, 128);

        let opts = ReadOptions::new().with_channels(vec!["EEG1".into(), "EEG3".into()]);
        let data = read_edf(&path, &opts).unwrap();
        assert_eq!(data.n_channels(), 2);
        assert_eq!(data.channel_labels, vec!["EEG1", "EEG3"]);
        assert_eq!(data.n_samples(0), 128);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_edf_time_range() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_time");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.edf");
        create_test_edf(&path, 2, 4, 256);

        let opts = ReadOptions::new().with_time_range(1.0, 3.0);
        let data = read_edf(&path, &opts).unwrap();
        assert_eq!(data.n_channels(), 2);
        assert_eq!(data.n_samples(0), 512);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_eeg_data_select_channels() {
        let data = EegData {
            channel_labels: vec!["Fp1".into(), "Fp2".into(), "Cz".into()],
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
            sampling_rates: vec![256.0; 3],
            duration: 0.0078125,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let subset = data.select_channels(&["Fp1", "Cz"]);
        assert_eq!(subset.n_channels(), 2);
        assert_eq!(subset.channel_labels, vec!["Fp1", "Cz"]);
        assert_eq!(subset.channel(0), Some(&[1.0, 2.0][..]));
        assert_eq!(subset.channel(1), Some(&[5.0, 6.0][..]));
    }

    #[test]
    fn test_eeg_data_time_slice() {
        let data = EegData {
            channel_labels: vec!["Fp1".into()],
            data: vec![(0..256).map(|i| i as f64).collect()],
            sampling_rates: vec![256.0],
            duration: 1.0,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let slice = data.time_slice(0.0, 0.5);
        assert_eq!(slice.n_samples(0), 128);
        assert_eq!(slice.channel(0).unwrap()[0], 0.0);
        assert_eq!(slice.channel(0).unwrap()[127], 127.0);
    }

    #[test]
    fn test_read_eeg_data_dispatch() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_dispatch");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.edf");
        create_test_edf(&path, 2, 1, 128);
        let data = read_eeg_data(&path, &ReadOptions::default()).unwrap();
        assert_eq!(data.n_channels(), 2);
        let bad = dir.join("test.xyz");
        std::fs::write(&bad, b"").unwrap();
        assert!(read_eeg_data(&bad, &ReadOptions::default()).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_brainvision_header_parse() {
        let vhdr = r#"
Brain Vision Data Exchange Header File Version 1.0
; comment line

[Common Infos]
DataFile=test.eeg
DataOrientation=MULTIPLEXED
SamplingInterval=3906.25

[Binary Infos]
BinaryFormat=INT_16

[Channel Infos]
Ch1=Fp1,,0.1
Ch2=Fp2,,0.1
Ch3=Cz,,0.1
"#;
        let hdr = parse_vhdr(vhdr).unwrap();
        assert_eq!(hdr.data_file, "test.eeg");
        assert_eq!(hdr.data_format, BvDataFormat::Int16);
        assert_eq!(hdr.data_orientation, BvOrientation::Multiplexed);
        assert_eq!(hdr.channels.len(), 3);
        assert_eq!(hdr.channels[0].name, "Fp1");
        assert!((hdr.channels[0].resolution - 0.1).abs() < 0.001);
        let sr = 1_000_000.0 / hdr.sampling_interval_us.unwrap();
        assert!((sr - 256.0).abs() < 0.01);
    }

    #[test]
    fn test_read_brainvision() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_bv");
        std::fs::create_dir_all(&dir).unwrap();

        let vhdr_path = dir.join("test.vhdr");
        std::fs::write(&vhdr_path, r#"Brain Vision Data Exchange Header File Version 1.0

[Common Infos]
DataFile=test.eeg
DataOrientation=MULTIPLEXED
SamplingInterval=3906.25

[Binary Infos]
BinaryFormat=INT_16

[Channel Infos]
Ch1=Fp1,,0.1
Ch2=Fp2,,0.1
"#).unwrap();

        let eeg_path = dir.join("test.eeg");
        let mut eeg_data = Vec::with_capacity(256 * 4);
        for s in 0..256 {
            let v1 = (1000.0 * (2.0 * std::f64::consts::PI * s as f64 / 256.0).sin()) as i16;
            let v2 = (500.0 * (2.0 * std::f64::consts::PI * 2.0 * s as f64 / 256.0).sin()) as i16;
            eeg_data.extend_from_slice(&v1.to_le_bytes());
            eeg_data.extend_from_slice(&v2.to_le_bytes());
        }
        std::fs::write(&eeg_path, &eeg_data).unwrap();

        let data = read_brainvision(&vhdr_path, &ReadOptions::default()).unwrap();
        assert_eq!(data.n_channels(), 2);
        assert_eq!(data.n_samples(0), 256);
        assert_eq!(data.channel_labels, vec!["Fp1", "Fp2"]);
        assert!((data.sampling_rates[0] - 256.0).abs() < 0.01);
        assert!(data.data[0].iter().all(|v| v.abs() <= 3276.8));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_channel_by_name() {
        let data = EegData {
            channel_labels: vec!["Fp1".into(), "Fp2".into()],
            data: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            sampling_rates: vec![256.0; 2],
            duration: 0.01171875,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        assert_eq!(data.channel_by_name("Fp1"), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(data.channel_by_name("Fp2"), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(data.channel_by_name("Cz"), None);
    }

    /// Benchmark-style test: 64 channels, 600 seconds @ 2048 Hz (typical clinical EEG).
    /// ~150 MB of data. Ensures we can handle realistic sizes.
    #[test]
    fn test_read_edf_large() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_large");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("large.edf");

        let n_ch = 64;
        let n_rec = 60; // 60 seconds (keep test fast)
        let spr = 2048;
        create_test_edf(&path, n_ch, n_rec, spr);

        let start = std::time::Instant::now();
        let data = read_edf(&path, &ReadOptions::default()).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(data.n_channels(), n_ch);
        assert_eq!(data.n_samples(0), n_rec * spr);

        // Should be well under 1 second for 60s × 64ch × 2048Hz (~15MB)
        assert!(elapsed.as_millis() < 1000,
            "Reading took {}ms, expected < 1000ms", elapsed.as_millis());

        // Channel selection should be faster
        let start = std::time::Instant::now();
        let _data = read_edf(&path, &ReadOptions::new()
            .with_channels(vec!["EEG1".into(), "EEG32".into()])).unwrap();
        let elapsed2 = start.elapsed();
        assert!(elapsed2 <= elapsed || elapsed2.as_millis() < 500,
            "Channel-select took {}ms", elapsed2.as_millis());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_times() {
        let data = EegData {
            channel_labels: vec!["Fp1".into()],
            data: vec![vec![0.0; 512]],
            sampling_rates: vec![256.0],
            duration: 2.0,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let times = data.times(0).unwrap();
        assert_eq!(times.len(), 512);
        assert!((times[0] - 0.0).abs() < 1e-10);
        assert!((times[1] - 1.0 / 256.0).abs() < 1e-10);
        assert!((times[511] - 511.0 / 256.0).abs() < 1e-10);
    }

    #[test]
    fn test_exclude_channels() {
        let data = EegData {
            channel_labels: vec!["Fp1".into(), "Fp2".into(), "Cz".into()],
            data: vec![vec![1.0], vec![2.0], vec![3.0]],
            sampling_rates: vec![256.0; 3],
            duration: 0.00390625,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let excl = data.exclude_channels(&["Fp2"]);
        assert_eq!(excl.n_channels(), 2);
        assert_eq!(excl.channel_labels, vec!["Fp1", "Cz"]);
    }

    #[test]
    fn test_convert_units() {
        let mut data = EegData {
            channel_labels: vec!["Fp1".into()],
            data: vec![vec![100.0, 200.0]],
            sampling_rates: vec![256.0],
            duration: 0.0078125,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let mut map = std::collections::HashMap::new();
        map.insert("Fp1".into(), 1e-6);
        data.convert_units(&map);
        assert!((data.data[0][0] - 100e-6).abs() < 1e-15);
        assert!((data.data[0][1] - 200e-6).abs() < 1e-15);
    }

    #[test]
    fn test_reject_by_annotation() {
        let data = EegData {
            channel_labels: vec!["Fp1".into()],
            data: vec![(0..256).map(|i| i as f64).collect()],
            sampling_rates: vec![256.0],
            duration: 1.0,
            annotations: vec![
                Annotation { onset: 0.25, duration: 0.25, description: "BAD_segment".into() },
            ],
            stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let rejected = data.reject_by_annotation("BAD");
        // Samples from 0.25s to 0.5s (64..128) should be NAN
        assert!(!rejected.data[0][63].is_nan());
        assert!(rejected.data[0][64].is_nan());
        assert!(rejected.data[0][127].is_nan());
        assert!(!rejected.data[0][128].is_nan());
    }

    #[test]
    fn test_edf_tal_parse() {
        let mut annotations = Vec::new();
        // TAL format: +onset\x14\x14\x00 (record onset) +onset\x14description\x14\x00
        let tal = b"+0.0\x14\x14\x00+0.5\x14stimulus\x14\x00+1.5\x150.5\x14response\x14\x00";
        let record_onset = parse_edf_tal(tal, &mut annotations);
        assert_eq!(record_onset, Some(0.0)); // first entry without description = record onset
        assert_eq!(annotations.len(), 2);
        assert!((annotations[0].onset - 0.5).abs() < 1e-10);
        assert_eq!(annotations[0].description, "stimulus");
        assert!((annotations[0].duration - 0.0).abs() < 1e-10);
        assert!((annotations[1].onset - 1.5).abs() < 1e-10);
        assert_eq!(annotations[1].description, "response");
        assert!((annotations[1].duration - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vmrk_parse() {
        let vmrk = r#"Brain Vision Data Exchange Marker File Version 1.0

[Common Infos]
Codepage=UTF-8
DataFile=test.eeg

[Marker Infos]
Mk1=Stimulus,S  1,512,1,0
Mk2=Stimulus,S  2,1024,1,0
Mk3=Response,R  1,2048,1,0
Mk4=Comment,hello world,3072,1,0
"#;
        let anns = parse_vmrk(vmrk, 256.0);
        assert_eq!(anns.len(), 4);
        // Mk1: position 512 → onset (512-1)/256 = 1.99609375
        assert!((anns[0].onset - 511.0 / 256.0).abs() < 1e-10);
        assert_eq!(anns[0].description, "S  1");
        assert!((anns[1].onset - 1023.0 / 256.0).abs() < 1e-10);
        assert_eq!(anns[1].description, "S  2");
        assert_eq!(anns[2].description, "R  1");
        assert_eq!(anns[3].description, "hello world");
    }

    #[test]
    fn test_brainvision_with_markers() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_bv_vmrk");
        std::fs::create_dir_all(&dir).unwrap();

        std::fs::write(dir.join("test.vhdr"), r#"Brain Vision Data Exchange Header File Version 1.0

[Common Infos]
DataFile=test.eeg
MarkerFile=test.vmrk
DataOrientation=MULTIPLEXED
SamplingInterval=3906.25

[Binary Infos]
BinaryFormat=INT_16

[Channel Infos]
Ch1=Fp1,,0.1
Ch2=Fp2,,0.1
"#).unwrap();

        std::fs::write(dir.join("test.vmrk"), r#"Brain Vision Data Exchange Marker File Version 1.0

[Marker Infos]
Mk1=Stimulus,S1,50,1,0
Mk2=Stimulus,S2,150,1,0
"#).unwrap();

        // Create binary data (2 ch × 256 samples × INT_16)
        let mut buf = Vec::with_capacity(256 * 2 * 2);
        for s in 0..256 {
            let v = (100.0 * (s as f64)).round() as i16;
            buf.extend_from_slice(&v.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("test.eeg"), &buf).unwrap();

        let data = read_brainvision(&dir.join("test.vhdr"), &ReadOptions::default()).unwrap();
        assert_eq!(data.annotations.len(), 2);
        assert_eq!(data.annotations[0].description, "S1");
        assert_eq!(data.annotations[1].description, "S2");
        // onset = (position - 1) / sampling_rate
        assert!((data.annotations[0].onset - 49.0 / 256.0).abs() < 1e-6);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_edf_with_exclude() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_excl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.edf");
        create_test_edf(&path, 4, 1, 128);

        let opts = ReadOptions::new().with_exclude(vec!["EEG2".into(), "EEG4".into()]);
        let data = read_edf(&path, &opts).unwrap();
        assert_eq!(data.n_channels(), 2);
        assert_eq!(data.channel_labels, vec!["EEG1", "EEG3"]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_stim_channel_detection() {
        let dir = std::env::temp_dir().join("bids_eeg_data_test_stim");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stim.edf");

        // Create EDF with a "Status" channel
        let n_ch = 3;
        let spr = 128;
        let mut file = std::fs::File::create(&path).unwrap();
        let mut hdr = [b' '; 256];
        hdr[0..1].copy_from_slice(b"0");
        hdr[168..176].copy_from_slice(b"01.01.01");
        hdr[176..184].copy_from_slice(b"00.00.00");
        let hs = format!("{:<8}", 256 + n_ch * 256);
        hdr[184..192].copy_from_slice(hs.as_bytes());
        hdr[236..244].copy_from_slice(b"1       ");
        hdr[244..252].copy_from_slice(b"1       ");
        let nc = format!("{:<4}", n_ch);
        hdr[252..256].copy_from_slice(nc.as_bytes());
        file.write_all(&hdr).unwrap();

        let mut ext = vec![b' '; n_ch * 256];
        let ch_labels = ["EEG1", "EEG2", "Status"];
        for i in 0..n_ch {
            let label = format!("{:<16}", ch_labels[i]);
            ext[i*16..i*16+16].copy_from_slice(label.as_bytes());
            ext[n_ch*96+i*8..n_ch*96+i*8+2].copy_from_slice(b"uV");
            let s = format!("{:<8}", "-3200"); ext[n_ch*104+i*8..n_ch*104+i*8+8].copy_from_slice(s.as_bytes());
            let s = format!("{:<8}", "3200"); ext[n_ch*112+i*8..n_ch*112+i*8+8].copy_from_slice(s.as_bytes());
            let s = format!("{:<8}", "-32768"); ext[n_ch*120+i*8..n_ch*120+i*8+8].copy_from_slice(s.as_bytes());
            let s = format!("{:<8}", "32767"); ext[n_ch*128+i*8..n_ch*128+i*8+8].copy_from_slice(s.as_bytes());
            let s = format!("{:<8}", spr); ext[n_ch*216+i*8..n_ch*216+i*8+8].copy_from_slice(s.as_bytes());
        }
        file.write_all(&ext).unwrap();

        let rec_bytes = n_ch * spr * 2;
        let buf = vec![0u8; rec_bytes];
        file.write_all(&buf).unwrap();
        drop(file);

        let data = read_edf(&path, &ReadOptions::default()).unwrap();
        assert_eq!(data.n_channels(), 3);
        // "Status" channel should be detected as stim
        assert_eq!(data.stim_channel_indices, vec![2]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn make_sine_data(freq: f64, sr: f64, duration: f64, n_ch: usize) -> EegData {
        let n = (duration * sr) as usize;
        let data: Vec<Vec<f64>> = (0..n_ch)
            .map(|_| (0..n).map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * freq * t).sin()
            }).collect())
            .collect();
        EegData {
            channel_labels: (0..n_ch).map(|i| format!("Ch{}", i + 1)).collect(),
            data,
            sampling_rates: vec![sr; n_ch],
            duration,
            annotations: Vec::new(),
            stim_channel_indices: Vec::new(),
            is_discontinuous: false,
            record_onsets: Vec::new(),
        }
    }

    #[test]
    fn test_filter_lowpass() {
        // 10 Hz signal + 100 Hz noise at 500 Hz sampling
        let sr = 500.0;
        let n = 1000;
        let data = EegData {
            channel_labels: vec!["Ch1".into()],
            data: vec![(0..n).map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * 10.0 * t).sin()
                    + (2.0 * std::f64::consts::PI * 100.0 * t).sin()
            }).collect()],
            sampling_rates: vec![sr],
            duration: n as f64 / sr,
            annotations: Vec::new(), stim_channel_indices: Vec::new(),
            is_discontinuous: false, record_onsets: Vec::new(),
        };

        let filtered = data.filter(None, Some(30.0), 5);
        assert_eq!(filtered.data[0].len(), n);
        // High-freq noise should be greatly reduced
        let orig_energy: f64 = data.data[0][n/2..].iter().map(|v| v*v).sum::<f64>();
        let filt_energy: f64 = filtered.data[0][n/2..].iter().map(|v| v*v).sum::<f64>();
        assert!(filt_energy < orig_energy * 0.7);
    }

    #[test]
    fn test_notch_filter() {
        let sr = 500.0;
        let n = 2000;
        let data = EegData {
            channel_labels: vec!["Ch1".into()],
            data: vec![(0..n).map(|i| {
                let t = i as f64 / sr;
                (2.0 * std::f64::consts::PI * 10.0 * t).sin()
                    + 0.5 * (2.0 * std::f64::consts::PI * 50.0 * t).sin()
            }).collect()],
            sampling_rates: vec![sr],
            duration: n as f64 / sr,
            annotations: Vec::new(), stim_channel_indices: Vec::new(),
            is_discontinuous: false, record_onsets: Vec::new(),
        };

        let filtered = data.notch_filter(50.0, 30.0);
        assert_eq!(filtered.data[0].len(), n);
    }

    #[test]
    fn test_resample() {
        let data = make_sine_data(5.0, 1000.0, 1.0, 2);
        assert_eq!(data.data[0].len(), 1000);

        let resampled = data.resample(250.0);
        assert_eq!(resampled.data[0].len(), 250);
        assert_eq!(resampled.data.len(), 2);
        assert!((resampled.sampling_rates[0] - 250.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_average_reference() {
        let data = EegData {
            channel_labels: vec!["Ch1".into(), "Ch2".into(), "Ch3".into()],
            data: vec![vec![3.0, 6.0], vec![1.0, 2.0], vec![2.0, 4.0]],
            sampling_rates: vec![256.0; 3],
            duration: 2.0 / 256.0,
            annotations: Vec::new(), stim_channel_indices: Vec::new(),
            is_discontinuous: false, record_onsets: Vec::new(),
        };
        let reref = data.set_average_reference();
        // Mean at t=0: (3+1+2)/3 = 2.0
        assert!((reref.data[0][0] - 1.0).abs() < 1e-10); // 3 - 2
        assert!((reref.data[1][0] - (-1.0)).abs() < 1e-10); // 1 - 2
        assert!((reref.data[2][0] - 0.0).abs() < 1e-10); // 2 - 2
    }

    #[test]
    fn test_set_reference() {
        let data = EegData {
            channel_labels: vec!["Fp1".into(), "Cz".into(), "Pz".into()],
            data: vec![vec![10.0, 20.0], vec![5.0, 10.0], vec![8.0, 16.0]],
            sampling_rates: vec![256.0; 3],
            duration: 2.0 / 256.0,
            annotations: Vec::new(), stim_channel_indices: Vec::new(),
            is_discontinuous: false, record_onsets: Vec::new(),
        };
        let reref = data.set_reference("Cz");
        assert!((reref.data[0][0] - 5.0).abs() < 1e-10); // 10 - 5
        assert!((reref.data[1][0] - 5.0).abs() < 1e-10); // Cz unchanged
        assert!((reref.data[2][0] - 3.0).abs() < 1e-10); // 8 - 5
    }

    #[test]
    fn test_epoch_and_average() {
        let sr = 100.0;
        let n = 500;
        let data = EegData {
            channel_labels: vec!["Ch1".into()],
            data: vec![(0..n).map(|i| (i as f64 / sr * 2.0 * std::f64::consts::PI).sin()).collect()],
            sampling_rates: vec![sr],
            duration: n as f64 / sr,
            annotations: vec![
                Annotation { onset: 1.0, duration: 0.0, description: "stim".into() },
                Annotation { onset: 2.0, duration: 0.0, description: "stim".into() },
                Annotation { onset: 3.0, duration: 0.0, description: "stim".into() },
            ],
            stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };

        let epochs = data.epoch(-0.2, 0.5, Some("stim"));
        assert_eq!(epochs.len(), 3);
        assert_eq!(epochs[0].data[0].len(), 70); // 0.2 + 0.5 = 0.7s * 100 Hz = 70 samples

        // Average
        let avg = EegData::average_epochs(&epochs).unwrap();
        assert_eq!(avg.data[0].len(), 70);
        assert_eq!(avg.data.len(), 1);
    }

    #[test]
    fn test_compute_psd() {
        let data = make_sine_data(10.0, 256.0, 2.0, 1);
        let (freqs, psd) = data.compute_psd(Some(256));
        assert_eq!(freqs.len(), 129); // 256/2 + 1
        assert_eq!(psd.len(), 1);
        assert_eq!(psd[0].len(), 129);
        // Peak should be at ~10 Hz
        let peak_idx = psd[0].iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
        let peak_freq = freqs[peak_idx];
        assert!((peak_freq - 10.0).abs() < 2.0,
            "PSD peak at {:.1} Hz, expected ~10 Hz", peak_freq);
    }

    #[test]
    fn test_display_debug() {
        let data = make_sine_data(10.0, 256.0, 1.0, 4);
        let display = format!("{}", data);
        assert!(display.contains("4 ch"), "Display: {}", display);
        assert!(display.contains("256"), "Display: {}", display);
        let debug = format!("{:?}", data);
        assert!(debug.contains("n_channels: 4"), "Debug: {}", debug);
        // Should NOT contain raw sample data
        assert!(!debug.contains("0."), "Debug should not dump samples: {}", &debug[..100.min(debug.len())]);
    }

    #[test]
    fn test_get_data_with_times() {
        let data = EegData {
            channel_labels: vec!["Fp1".into()],
            data: vec![vec![1.0, 2.0, 3.0, 4.0]],
            sampling_rates: vec![256.0],
            duration: 4.0 / 256.0,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let (d, t) = data.get_data_with_times();
        assert_eq!(d.len(), 1);
        assert_eq!(t.len(), 4);
        assert!((t[0] - 0.0).abs() < 1e-10);
        assert!((t[3] - 3.0 / 256.0).abs() < 1e-10);
    }

    #[test]
    fn test_pick_types() {
        use crate::ChannelType;
        let data = EegData {
            channel_labels: vec!["Fp1".into(), "EOG1".into(), "Cz".into(), "ECG1".into()],
            data: vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
            sampling_rates: vec![256.0; 4],
            duration: 1.0 / 256.0,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let types = vec![ChannelType::EEG, ChannelType::EEG, ChannelType::EEG, ChannelType::ECG];
        let picked = data.pick_types(&[ChannelType::EEG], &types);
        assert_eq!(picked.n_channels(), 3);
        assert_eq!(picked.channel_labels, vec!["Fp1", "EOG1", "Cz"]);
    }

    #[test]
    fn test_concatenate() {
        let mut data1 = EegData {
            channel_labels: vec!["Fp1".into(), "Fp2".into()],
            data: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            sampling_rates: vec![256.0; 2],
            duration: 2.0 / 256.0,
            annotations: vec![Annotation { onset: 0.0, duration: 0.0, description: "A".into() }],
            stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        let data2 = EegData {
            channel_labels: vec!["Fp1".into(), "Fp2".into()],
            data: vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            sampling_rates: vec![256.0; 2],
            duration: 2.0 / 256.0,
            annotations: vec![Annotation { onset: 0.0, duration: 0.0, description: "B".into() }],
            stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        data1.concatenate(&data2).unwrap();
        assert_eq!(data1.n_samples(0), 4);
        assert_eq!(data1.data[0], vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(data1.data[1], vec![3.0, 4.0, 7.0, 8.0]);
        assert_eq!(data1.annotations.len(), 2);
        assert_eq!(data1.annotations[1].description, "B");
        assert!((data1.annotations[1].onset - 2.0 / 256.0).abs() < 1e-10);

        // Mismatched channels should fail
        let data3 = EegData {
            channel_labels: vec!["Cz".into()],
            data: vec![vec![9.0]],
            sampling_rates: vec![256.0],
            duration: 1.0 / 256.0,
            annotations: Vec::new(), stim_channel_indices: Vec::new(), is_discontinuous: false, record_onsets: Vec::new(),
        };
        assert!(data1.concatenate(&data3).is_err());
    }
}
