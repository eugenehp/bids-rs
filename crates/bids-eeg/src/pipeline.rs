//! Composable preprocessing pipeline for EEG data.
//!
//! Chains filter → epoch → resample → normalize into a single declarative
//! pipeline that transforms raw [`EegData`] + [`EegEvent`]s into ML-ready
//! `(X, y)` arrays.
//!
//! Inspired by [MOABB](https://github.com/NeuroTechX/moabb)'s paradigm system.
//!
//! # Example
//!
//! ```no_run
//! use bids_eeg::pipeline::Pipeline;
//!
//! let pipeline = Pipeline::new()
//!     .select_channels(&["Fz", "Cz", "Pz", "Oz"])
//!     .bandpass(8.0, 30.0, 5)
//!     .notch(50.0)
//!     .epoch("left_hand", -0.5, 3.5)
//!     .epoch("right_hand", -0.5, 3.5)
//!     .baseline(-0.5, 0.0)
//!     .resample(128.0)
//!     .z_score();
//!
//! // let result = pipeline.transform(&eeg_data, &events);
//! ```

use crate::data::EegData;
use crate::events::EegEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single preprocessing step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Step {
    SelectChannels(Vec<String>),
    ExcludeChannels(Vec<String>),
    AverageReference,
    ChannelReference(String),
    Bandpass(f64, f64, usize),
    Highpass(f64, usize),
    Lowpass(f64, usize),
    Notch(f64),
    Resample(f64),
    /// Epoch around events: (trial_type, tmin, tmax)
    Epoch(String, f64, f64),
    /// Baseline correction: subtract mean of (bmin, bmax) from each epoch
    Baseline(f64, f64),
    ZScore,
    MinMaxNormalize,
}

/// Result of a pipeline transformation.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Epoch data: `[n_epochs][n_channels][n_samples]`.
    pub x: Vec<Vec<Vec<f64>>>,
    /// Labels: one per epoch.
    pub y: Vec<String>,
    /// Metadata per epoch: maps of (key → value).
    pub metadata: Vec<HashMap<String, String>>,
    /// Channel names after pipeline processing.
    pub channel_names: Vec<String>,
    /// Sampling rate after pipeline processing.
    pub sampling_rate: f64,
}

impl PipelineResult {
    /// Number of epochs.
    #[must_use]
    pub fn n_epochs(&self) -> usize { self.x.len() }

    /// Number of channels per epoch.
    #[must_use]
    pub fn n_channels(&self) -> usize {
        self.x.first().map_or(0, |e| e.len())
    }

    /// Number of samples per epoch.
    #[must_use]
    pub fn n_samples(&self) -> usize {
        self.x.first().and_then(|e| e.first()).map_or(0, |c| c.len())
    }

    /// Shape as (n_epochs, n_channels, n_samples).
    #[must_use]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.n_epochs(), self.n_channels(), self.n_samples())
    }

    /// Flatten all epochs into a single contiguous Vec for model input.
    ///
    /// Layout: `[epoch0_ch0_s0, epoch0_ch0_s1, ..., epoch0_chN_sM, epoch1_ch0_s0, ...]`
    /// Shape: `(n_epochs, n_channels * n_samples)`.
    #[must_use]
    pub fn to_flat_features(&self) -> Vec<Vec<f64>> {
        self.x.iter().map(|epoch| {
            epoch.iter().flat_map(|ch| ch.iter().copied()).collect()
        }).collect()
    }

    /// Convert to row-major contiguous array: `[n_epochs * n_channels * n_samples]`.
    #[must_use]
    pub fn to_contiguous(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_epochs() * self.n_channels() * self.n_samples());
        for epoch in &self.x {
            for ch in epoch {
                out.extend_from_slice(ch);
            }
        }
        out
    }

    /// Unique labels sorted.
    #[must_use]
    pub fn classes(&self) -> Vec<String> {
        let mut c: Vec<String> = self.y.iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter().cloned().collect();
        c.sort();
        c
    }

    /// Integer-encoded labels (sorted class order).
    #[must_use]
    pub fn y_encoded(&self) -> Vec<usize> {
        let classes = self.classes();
        self.y.iter().map(|label| {
            classes.iter().position(|c| c == label).unwrap_or(0)
        }).collect()
    }
}

/// A composable EEG preprocessing pipeline.
///
/// Steps are applied in order. Epoch steps collect events and cut windows;
/// non-epoch steps are applied either to the raw data (before epoching) or
/// to each epoch (after epoching).
///
/// Pipelines can be serialized to JSON/YAML for reproducibility and sharing:
///
/// ```no_run
/// # use bids_eeg::pipeline::Pipeline;
/// let pipeline = Pipeline::new()
///     .bandpass(8.0, 30.0, 5)
///     .epoch("left_hand", 0.0, 3.0)
///     .z_score();
///
/// // Save to JSON
/// pipeline.save_json("my_pipeline.json").unwrap();
///
/// // Load back
/// let loaded = Pipeline::load_json("my_pipeline.json").unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    /// Optional human-readable name.
    #[serde(default)]
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: String,
    steps: Vec<Step>,
}

impl Default for Pipeline {
    fn default() -> Self { Self::new() }
}

impl Pipeline {
    #[must_use]
    pub fn new() -> Self {
        Self { name: String::new(), description: String::new(), steps: Vec::new() }
    }

    /// Set a name for this pipeline (for display / serialization).
    #[must_use]
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.into();
        self
    }

    /// Set a description for this pipeline.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.into();
        self
    }

    // ── Persistence ────────────────────────────────────────────────────

    /// Save the pipeline configuration to a JSON file.
    pub fn save_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    /// Load a pipeline configuration from a JSON file.
    pub fn load_json(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Serialize to a JSON string.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get the steps (for inspection).
    #[must_use]
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    // ── Channel selection ──────────────────────────────────────────────

    #[must_use]
    pub fn select_channels(mut self, names: &[&str]) -> Self {
        self.steps.push(Step::SelectChannels(names.iter().map(|s| (*s).to_string()).collect()));
        self
    }

    #[must_use]
    pub fn exclude_channels(mut self, names: &[&str]) -> Self {
        self.steps.push(Step::ExcludeChannels(names.iter().map(|s| (*s).to_string()).collect()));
        self
    }

    // ── Re-referencing ─────────────────────────────────────────────────

    #[must_use]
    pub fn average_reference(mut self) -> Self {
        self.steps.push(Step::AverageReference);
        self
    }

    #[must_use]
    pub fn reference_channel(mut self, name: &str) -> Self {
        self.steps.push(Step::ChannelReference(name.into()));
        self
    }

    // ── Filtering ──────────────────────────────────────────────────────

    #[must_use]
    pub fn bandpass(mut self, l_freq: f64, h_freq: f64, order: usize) -> Self {
        self.steps.push(Step::Bandpass(l_freq, h_freq, order));
        self
    }

    #[must_use]
    pub fn highpass(mut self, freq: f64, order: usize) -> Self {
        self.steps.push(Step::Highpass(freq, order));
        self
    }

    #[must_use]
    pub fn lowpass(mut self, freq: f64, order: usize) -> Self {
        self.steps.push(Step::Lowpass(freq, order));
        self
    }

    #[must_use]
    pub fn notch(mut self, freq: f64) -> Self {
        self.steps.push(Step::Notch(freq));
        self
    }

    // ── Resampling ─────────────────────────────────────────────────────

    #[must_use]
    pub fn resample(mut self, target_hz: f64) -> Self {
        self.steps.push(Step::Resample(target_hz));
        self
    }

    // ── Epoching ───────────────────────────────────────────────────────

    /// Add an epoch extraction step for a specific trial type.
    /// Multiple calls accumulate — all specified trial types produce epochs.
    #[must_use]
    pub fn epoch(mut self, trial_type: &str, tmin: f64, tmax: f64) -> Self {
        self.steps.push(Step::Epoch(trial_type.into(), tmin, tmax));
        self
    }

    // ── Normalization ──────────────────────────────────────────────────

    /// Baseline correction: subtract mean of `[bmin, bmax]` from each epoch.
    #[must_use]
    pub fn baseline(mut self, bmin: f64, bmax: f64) -> Self {
        self.steps.push(Step::Baseline(bmin, bmax));
        self
    }

    #[must_use]
    pub fn z_score(mut self) -> Self {
        self.steps.push(Step::ZScore);
        self
    }

    #[must_use]
    pub fn min_max_normalize(mut self) -> Self {
        self.steps.push(Step::MinMaxNormalize);
        self
    }

    // ── Execution ──────────────────────────────────────────────────────

    /// Transform raw EEG data + events into ML-ready `(X, y)`.
    ///
    /// Steps are split into three phases:
    /// 1. **Pre-epoch**: channel selection, referencing, filtering, resampling
    /// 2. **Epoching**: cut windows around events
    /// 3. **Post-epoch**: baseline correction, normalization
    pub fn transform(&self, data: &EegData, events: &[EegEvent]) -> PipelineResult {
        // Partition steps into pre-epoch, epoch specs, post-epoch
        let mut pre_steps = Vec::new();
        let mut epoch_specs: Vec<(String, f64, f64)> = Vec::new();
        let mut post_steps = Vec::new();
        let mut past_epoch = false;

        for step in &self.steps {
            match step {
                Step::Epoch(tt, tmin, tmax) => {
                    epoch_specs.push((tt.clone(), *tmin, *tmax));
                    past_epoch = true;
                }
                other => {
                    if past_epoch {
                        post_steps.push(other.clone());
                    } else {
                        pre_steps.push(other.clone());
                    }
                }
            }
        }

        // Phase 1: apply pre-epoch steps to the continuous data
        let mut processed = data.clone();
        for step in &pre_steps {
            processed = apply_step_to_data(processed, step);
        }

        let sr = processed.sampling_rates.first().copied().unwrap_or(1.0);
        let channel_names = processed.channel_labels.clone();

        // Phase 2: epoch extraction
        let (mut epochs, labels, metas) = if epoch_specs.is_empty() {
            // No epoching — treat the entire recording as a single epoch
            (vec![processed.data.clone()], vec!["_whole_".into()], vec![HashMap::new()])
        } else {
            extract_epochs(&processed, events, &epoch_specs)
        };

        // Phase 3: apply post-epoch steps to each epoch
        let mut current_sr = sr;
        for step in &post_steps {
            for epoch in &mut epochs {
                apply_step_to_epoch(epoch, step, current_sr);
            }
            if let Step::Resample(target) = step {
                current_sr = *target;
            }
        }

        let final_sr = current_sr;

        PipelineResult {
            x: epochs,
            y: labels,
            metadata: metas,
            channel_names,
            sampling_rate: final_sr,
        }
    }
}

// ─── Internal helpers ──────────────────────────────────────────────────────────

fn apply_step_to_data(data: EegData, step: &Step) -> EegData {
    match step {
        Step::SelectChannels(names) => {
            let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            data.select_channels(&refs)
        }
        Step::ExcludeChannels(names) => {
            let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            data.exclude_channels(&refs)
        }
        Step::AverageReference => data.set_average_reference(),
        Step::ChannelReference(ch) => data.set_reference(ch),
        Step::Bandpass(lo, hi, order) => data.filter(Some(*lo), Some(*hi), *order),
        Step::Highpass(freq, order) => data.filter(Some(*freq), None, *order),
        Step::Lowpass(freq, order) => data.filter(None, Some(*freq), *order),
        Step::Notch(freq) => data.notch_filter(*freq, 30.0),
        Step::Resample(hz) => data.resample(*hz),
        Step::ZScore | Step::MinMaxNormalize | Step::Baseline(..) | Step::Epoch(..) => data,
    }
}

fn apply_step_to_epoch(epoch: &mut [Vec<f64>], step: &Step, sr: f64) {
    match step {
        Step::Baseline(bmin, bmax) => {
            for ch in epoch.iter_mut() {
                let start = (bmin.max(0.0) * sr).round() as usize;
                let end = (bmax.max(0.0) * sr).round() as usize;
                let end = end.min(ch.len());
                let start = start.min(end);
                if start < end {
                    let mean: f64 = ch[start..end].iter().sum::<f64>() / (end - start) as f64;
                    for v in ch.iter_mut() {
                        *v -= mean;
                    }
                }
            }
        }
        Step::ZScore => {
            for ch in epoch.iter_mut() {
                let n = ch.len() as f64;
                if n < 2.0 { continue; }
                let mean = ch.iter().sum::<f64>() / n;
                let std = (ch.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
                let std = if std > f64::EPSILON { std } else { 1.0 };
                for v in ch.iter_mut() {
                    *v = (*v - mean) / std;
                }
            }
        }
        Step::MinMaxNormalize => {
            for ch in epoch.iter_mut() {
                let min = ch.iter().copied().fold(f64::INFINITY, f64::min);
                let max = ch.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;
                let range = if range > f64::EPSILON { range } else { 1.0 };
                for v in ch.iter_mut() {
                    *v = (*v - min) / range;
                }
            }
        }
        Step::Resample(target_hz) => {
            for ch in epoch.iter_mut() {
                *ch = bids_filter::resample(ch, sr, *target_hz);
            }
        }
        _ => {} // Other steps handled at data level
    }
}

/// Epochs, labels, and per-epoch metadata extracted from continuous data.
type ExtractedEpochs = (Vec<Vec<Vec<f64>>>, Vec<String>, Vec<HashMap<String, String>>);

/// Extract epochs from continuous data around events.
fn extract_epochs(
    data: &EegData,
    events: &[EegEvent],
    epoch_specs: &[(String, f64, f64)],
) -> ExtractedEpochs {
    let sr = data.sampling_rates.first().copied().unwrap_or(1.0);
    let n_total = data.data.first().map_or(0, |ch| ch.len()) as isize;

    let mut epochs = Vec::new();
    let mut labels = Vec::new();
    let mut metas = Vec::new();

    for (trial_type, tmin, tmax) in epoch_specs {
        let n_before = ((-tmin) * sr).round() as usize;
        let n_after = (tmax * sr).round() as usize;
        let epoch_len = n_before + n_after;

        for event in events {
            let tt = event.trial_type.as_deref().unwrap_or("");
            if tt != trial_type { continue; }

            let center = (event.onset * sr).round() as isize;
            let start = center - n_before as isize;

            if start < 0 || start + epoch_len as isize > n_total { continue; }
            let start = start as usize;

            let epoch: Vec<Vec<f64>> = data.data.iter()
                .map(|ch| ch[start..start + epoch_len].to_vec())
                .collect();

            let mut meta = HashMap::new();
            meta.insert("trial_type".into(), trial_type.clone());
            meta.insert("onset".into(), event.onset.to_string());
            if let Some(ref v) = event.value { meta.insert("value".into(), v.clone()); }
            for (k, v) in &event.extra { meta.insert(k.clone(), v.clone()); }

            epochs.push(epoch);
            labels.push(trial_type.clone());
            metas.push(meta);
        }
    }

    (epochs, labels, metas)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::EegData;
    use crate::events::EegEvent;

    fn make_test_data() -> EegData {
        let sr = 256.0;
        let dur = 10.0;
        let n = (sr * dur) as usize;
        EegData {
            channel_labels: vec!["Fz".into(), "Cz".into(), "Pz".into()],
            data: vec![
                (0..n).map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / sr).sin()).collect(),
                (0..n).map(|i| (2.0 * std::f64::consts::PI * 20.0 * i as f64 / sr).sin()).collect(),
                (0..n).map(|i| (2.0 * std::f64::consts::PI * 5.0 * i as f64 / sr).sin()).collect(),
            ],
            sampling_rates: vec![sr; 3],
            duration: dur,
            annotations: Vec::new(),
            stim_channel_indices: Vec::new(),
            is_discontinuous: false,
            record_onsets: Vec::new(),
        }
    }

    fn make_test_events() -> Vec<EegEvent> {
        vec![
            EegEvent { onset: 1.0, duration: 0.0, trial_type: Some("left_hand".into()), value: None, sample: None, response_time: None, extra: HashMap::new() },
            EegEvent { onset: 3.0, duration: 0.0, trial_type: Some("right_hand".into()), value: None, sample: None, response_time: None, extra: HashMap::new() },
            EegEvent { onset: 5.0, duration: 0.0, trial_type: Some("left_hand".into()), value: None, sample: None, response_time: None, extra: HashMap::new() },
            EegEvent { onset: 7.0, duration: 0.0, trial_type: Some("right_hand".into()), value: None, sample: None, response_time: None, extra: HashMap::new() },
        ]
    }

    #[test]
    fn test_basic_pipeline() {
        let data = make_test_data();
        let events = make_test_events();

        let pipeline = Pipeline::new()
            .select_channels(&["Fz", "Cz"])
            .bandpass(1.0, 40.0, 4)
            .epoch("left_hand", 0.0, 2.0)
            .epoch("right_hand", 0.0, 2.0)
            .baseline(0.0, 0.5)
            .z_score();

        let result = pipeline.transform(&data, &events);
        assert_eq!(result.n_epochs(), 4);
        assert_eq!(result.n_channels(), 2);
        assert_eq!(result.y.iter().filter(|l| *l == "left_hand").count(), 2);
        assert_eq!(result.y.iter().filter(|l| *l == "right_hand").count(), 2);
    }

    #[test]
    fn test_pipeline_with_resample() {
        let data = make_test_data();
        let events = make_test_events();

        let pipeline = Pipeline::new()
            .epoch("left_hand", 0.0, 2.0)
            .resample(128.0);

        let result = pipeline.transform(&data, &events);
        assert_eq!(result.n_epochs(), 2);
        assert_eq!(result.sampling_rate, 128.0);
        // 2 seconds at 128 Hz = 256 samples
        assert!((result.n_samples() as i32 - 256).abs() <= 2);
    }

    #[test]
    fn test_pipeline_result_helpers() {
        let data = make_test_data();
        let events = make_test_events();

        let pipeline = Pipeline::new()
            .epoch("left_hand", 0.0, 1.0)
            .epoch("right_hand", 0.0, 1.0);

        let result = pipeline.transform(&data, &events);
        let (ne, nc, ns) = result.shape();
        assert_eq!(ne, 4);
        assert_eq!(nc, 3);

        let flat = result.to_flat_features();
        assert_eq!(flat.len(), 4);
        assert_eq!(flat[0].len(), nc * ns);

        let classes = result.classes();
        assert_eq!(classes, vec!["left_hand", "right_hand"]);

        let encoded = result.y_encoded();
        assert_eq!(encoded.len(), 4);
    }

    #[test]
    fn test_no_epoch_pipeline() {
        let data = make_test_data();
        let pipeline = Pipeline::new()
            .select_channels(&["Fz"])
            .z_score();

        let result = pipeline.transform(&data, &[]);
        assert_eq!(result.n_epochs(), 1);
        assert_eq!(result.n_channels(), 1);
    }

    #[test]
    fn test_json_roundtrip() {
        let pipeline = Pipeline::new()
            .with_name("motor_imagery_baseline")
            .with_description("Standard MI preprocessing")
            .select_channels(&["Fz", "Cz", "Pz", "C3", "C4"])
            .average_reference()
            .bandpass(8.0, 30.0, 5)
            .notch(50.0)
            .epoch("left_hand", -0.5, 3.5)
            .epoch("right_hand", -0.5, 3.5)
            .baseline(-0.5, 0.0)
            .resample(128.0)
            .z_score();

        let json = pipeline.to_json();
        assert!(json.contains("motor_imagery_baseline"));
        assert!(json.contains("Bandpass"));
        assert!(json.contains("Notch"));
        assert!(json.contains("left_hand"));

        let loaded = Pipeline::from_json(&json).unwrap();
        assert_eq!(loaded.name, "motor_imagery_baseline");
        assert_eq!(loaded.steps().len(), pipeline.steps().len());

        // Loaded pipeline should produce the same results
        let data = make_test_data();
        let events = make_test_events();
        let r1 = pipeline.transform(&data, &events);
        let r2 = loaded.transform(&data, &events);
        assert_eq!(r1.shape(), r2.shape());
        assert_eq!(r1.y, r2.y);
    }

    #[test]
    fn test_save_load_json_file() {
        let dir = std::env::temp_dir().join("bids_pipeline_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pipeline.json");

        let pipeline = Pipeline::new()
            .with_name("test_save")
            .bandpass(1.0, 40.0, 4)
            .epoch("stimulus", 0.0, 1.0)
            .z_score();

        pipeline.save_json(path.to_str().unwrap()).unwrap();
        assert!(path.exists());

        let loaded = Pipeline::load_json(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.name, "test_save");
        assert_eq!(loaded.steps().len(), 3);

        // Verify the JSON is human-readable
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"name\""));
        assert!(content.contains("test_save"));
        assert!(content.contains("Bandpass"));

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_step_visibility() {
        let pipeline = Pipeline::new()
            .highpass(1.0, 4)
            .lowpass(40.0, 4)
            .resample(256.0);

        let steps = pipeline.steps();
        assert_eq!(steps.len(), 3);
        assert!(matches!(steps[0], Step::Highpass(..)));
        assert!(matches!(steps[1], Step::Lowpass(..)));
        assert!(matches!(steps[2], Step::Resample(..)));
    }
}
