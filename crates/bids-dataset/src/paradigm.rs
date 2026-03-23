//! BCI paradigm definitions for ML pipelines.
//!
//! Inspired by [MOABB](https://github.com/NeuroTechX/moabb)'s paradigm system,
//! this module defines how raw BIDS data should be preprocessed and epoched
//! for specific experimental paradigms (motor imagery, P300, SSVEP, resting state).
//!
//! A paradigm specifies:
//! - Which events / event codes to use for labeling
//! - Bandpass filter range
//! - Epoch time window (tmin/tmax relative to event onset)
//! - Target sampling rate
//! - Channel selection
//! - Which datasets are compatible
//!
//! This replaces MOABB's `BaseParadigm` / `MotorImagery` / `P300` / `SSVEP` classes.

use std::collections::HashMap;

/// A bandpass filter specification (low cutoff Hz, high cutoff Hz).
pub type BandpassFilter = (f64, f64);

/// Which events from the dataset to use, and what label to assign.
pub type EventMap = HashMap<String, i64>;

/// A paradigm defines how to extract labeled epochs from continuous data.
///
/// This is the Rust equivalent of MOABB's `BaseParadigm`. It tells the
/// data loading pipeline:
/// 1. Which events to epoch around (and what labels to assign)
/// 2. What time window to cut (tmin, tmax relative to event onset)
/// 3. What bandpass filter(s) to apply
/// 4. What sampling rate to resample to
/// 5. Which channels to keep
///
/// # Example
///
/// ```
/// use bids_dataset::paradigm::Paradigm;
///
/// let mi = Paradigm::motor_imagery()
///     .events(&["left_hand", "right_hand"])
///     .bandpass(8.0, 30.0)
///     .interval(0.5, 3.5)
///     .resample(128.0);
/// ```
#[derive(Debug, Clone)]
pub struct Paradigm {
    /// Human-readable name.
    pub name: String,
    /// The paradigm type (for dataset compatibility checks).
    pub paradigm_type: ParadigmType,
    /// Bandpass filter banks. Each entry is (fmin, fmax).
    /// Multiple entries = filter bank approach.
    pub filters: Vec<BandpassFilter>,
    /// Event names to epoch around. If empty, use all events.
    pub events: Vec<String>,
    /// Start of epoch window in seconds, relative to event onset.
    pub tmin: f64,
    /// End of epoch window in seconds, relative to event onset.
    /// `None` = use dataset default.
    pub tmax: Option<f64>,
    /// Baseline correction interval. `None` = no baseline correction.
    pub baseline: Option<(f64, f64)>,
    /// Channels to keep. `None` = all EEG channels.
    pub channels: Option<Vec<String>>,
    /// Target sampling rate in Hz. `None` = keep original.
    pub resample_hz: Option<f64>,
}

/// Paradigm type for dataset compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParadigmType {
    MotorImagery,
    P300,
    SSVEP,
    RestingState,
    FixedIntervalWindows,
    Custom,
}

impl Paradigm {
    /// Create a motor imagery paradigm with sensible defaults.
    ///
    /// Default: 8–30 Hz bandpass, 0.5–3.5s window.
    #[must_use]
    pub fn motor_imagery() -> Self {
        Self {
            name: "MotorImagery".into(),
            paradigm_type: ParadigmType::MotorImagery,
            filters: vec![(8.0, 30.0)],
            events: vec![],
            tmin: 0.0,
            tmax: None,
            baseline: None,
            channels: None,
            resample_hz: None,
        }
    }

    /// Create a P300 paradigm with sensible defaults.
    ///
    /// Default: 1–24 Hz bandpass, 0.0–0.8s window.
    #[must_use]
    pub fn p300() -> Self {
        Self {
            name: "P300".into(),
            paradigm_type: ParadigmType::P300,
            filters: vec![(1.0, 24.0)],
            events: vec!["Target".into(), "NonTarget".into()],
            tmin: 0.0,
            tmax: Some(0.8),
            baseline: None,
            channels: None,
            resample_hz: None,
        }
    }

    /// Create an SSVEP paradigm with sensible defaults.
    ///
    /// Default: 7–45 Hz bandpass.
    #[must_use]
    pub fn ssvep() -> Self {
        Self {
            name: "SSVEP".into(),
            paradigm_type: ParadigmType::SSVEP,
            filters: vec![(7.0, 45.0)],
            events: vec![],
            tmin: 0.0,
            tmax: None,
            baseline: None,
            channels: None,
            resample_hz: None,
        }
    }

    /// Create a resting-state paradigm with sensible defaults.
    ///
    /// Default: 1–35 Hz bandpass, 10–50s window, resampled to 128 Hz.
    #[must_use]
    pub fn resting_state() -> Self {
        Self {
            name: "RestingState".into(),
            paradigm_type: ParadigmType::RestingState,
            filters: vec![(1.0, 35.0)],
            events: vec![],
            tmin: 10.0,
            tmax: Some(50.0),
            baseline: None,
            channels: None,
            resample_hz: Some(128.0),
        }
    }

    /// Create a fixed-interval windowing paradigm (no events needed).
    ///
    /// Useful for resting-state or continuous data where you want to
    /// extract windows at regular intervals.
    #[must_use]
    pub fn fixed_interval(window_sec: f64, _stride_sec: f64) -> Self {
        Self {
            name: "FixedIntervalWindows".into(),
            paradigm_type: ParadigmType::FixedIntervalWindows,
            filters: vec![(1.0, 45.0)],
            events: vec![],
            tmin: 0.0,
            tmax: Some(window_sec),
            baseline: None,
            channels: None,
            resample_hz: None,
        }
    }

    /// Create a custom paradigm.
    #[must_use]
    pub fn custom(name: &str) -> Self {
        Self {
            name: name.into(),
            paradigm_type: ParadigmType::Custom,
            filters: vec![],
            events: vec![],
            tmin: 0.0,
            tmax: None,
            baseline: None,
            channels: None,
            resample_hz: None,
        }
    }

    // ── Builder methods ────────────────────────────────────────────────

    /// Set the event names to epoch around.
    #[must_use]
    pub fn events(mut self, events: &[&str]) -> Self {
        self.events = events.iter().map(|s| (*s).to_string()).collect();
        self
    }

    /// Set a single bandpass filter.
    #[must_use]
    pub fn bandpass(mut self, fmin: f64, fmax: f64) -> Self {
        self.filters = vec![(fmin, fmax)];
        self
    }

    /// Set a filter bank (multiple bandpass filters).
    #[must_use]
    pub fn filter_bank(mut self, filters: &[(f64, f64)]) -> Self {
        self.filters = filters.to_vec();
        self
    }

    /// Set the epoch time interval relative to event onset.
    #[must_use]
    pub fn interval(mut self, tmin: f64, tmax: f64) -> Self {
        self.tmin = tmin;
        self.tmax = Some(tmax);
        self
    }

    /// Set baseline correction interval.
    #[must_use]
    pub fn baseline(mut self, bmin: f64, bmax: f64) -> Self {
        self.baseline = Some((bmin, bmax));
        self
    }

    /// Set target sampling rate.
    #[must_use]
    pub fn resample(mut self, hz: f64) -> Self {
        self.resample_hz = Some(hz);
        self
    }

    /// Set channel selection.
    #[must_use]
    pub fn channels(mut self, channels: &[&str]) -> Self {
        self.channels = Some(channels.iter().map(|s| (*s).to_string()).collect());
        self
    }

    // ── Derived properties ─────────────────────────────────────────────

    /// Epoch duration in seconds.
    #[must_use]
    pub fn epoch_duration(&self) -> Option<f64> {
        self.tmax.map(|tmax| tmax - self.tmin)
    }

    /// Number of samples per epoch at the target sampling rate.
    #[must_use]
    pub fn epoch_samples(&self) -> Option<usize> {
        let sr = self.resample_hz?;
        let dur = self.epoch_duration()?;
        Some((dur * sr).round() as usize)
    }

    /// Number of filter banks.
    #[must_use]
    pub fn n_filters(&self) -> usize {
        self.filters.len()
    }

    /// Check if a dataset's paradigm type is compatible.
    #[must_use]
    pub fn is_compatible(&self, dataset_paradigm: &str) -> bool {
        match self.paradigm_type {
            ParadigmType::MotorImagery => dataset_paradigm == "imagery",
            ParadigmType::P300 => dataset_paradigm == "p300",
            ParadigmType::SSVEP => dataset_paradigm == "ssvep",
            ParadigmType::RestingState => dataset_paradigm == "rstate",
            ParadigmType::FixedIntervalWindows | ParadigmType::Custom => true,
        }
    }
}

impl std::fmt::Display for Paradigm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let filters: Vec<String> = self.filters.iter()
            .map(|(lo, hi)| format!("{lo}–{hi} Hz"))
            .collect();
        write!(f, "{} ({})", self.name, filters.join(", "))?;
        if let Some(dur) = self.epoch_duration() {
            write!(f, " [{:.1}–{:.1}s = {:.1}s]", self.tmin, self.tmax.unwrap(), dur)?;
        }
        if let Some(sr) = self.resample_hz {
            write!(f, " @{sr:.0}Hz")?;
        }
        Ok(())
    }
}

// ─── Cross-session / Cross-subject Splitters ───────────────────────────────────

/// Evaluation strategy mirroring MOABB's evaluation types.
///
/// These determine how train/test splits are organized relative to
/// the BIDS hierarchy (sessions, subjects, datasets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalStrategy {
    /// K-fold CV within each session (MOABB's `WithinSessionEvaluation`).
    WithinSession,
    /// Leave-one-session-out per subject (MOABB's `CrossSessionEvaluation`).
    /// Requires ≥ 2 sessions.
    CrossSession,
    /// Leave-one-subject-out (MOABB's `CrossSubjectEvaluation`).
    /// Requires ≥ 2 subjects.
    CrossSubject,
}

impl std::fmt::Display for EvalStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WithinSession => write!(f, "WithinSession"),
            Self::CrossSession => write!(f, "CrossSession"),
            Self::CrossSubject => write!(f, "CrossSubject"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motor_imagery_defaults() {
        let mi = Paradigm::motor_imagery()
            .events(&["left_hand", "right_hand"])
            .interval(0.5, 3.5)
            .resample(128.0);

        assert_eq!(mi.paradigm_type, ParadigmType::MotorImagery);
        assert_eq!(mi.filters, vec![(8.0, 30.0)]);
        assert_eq!(mi.events, vec!["left_hand", "right_hand"]);
        assert_eq!(mi.epoch_duration(), Some(3.0));
        assert_eq!(mi.epoch_samples(), Some(384)); // 3.0 * 128
        assert!(mi.is_compatible("imagery"));
        assert!(!mi.is_compatible("p300"));
    }

    #[test]
    fn test_p300_defaults() {
        let p = Paradigm::p300();
        assert_eq!(p.paradigm_type, ParadigmType::P300);
        assert_eq!(p.events, vec!["Target", "NonTarget"]);
        assert!(p.is_compatible("p300"));
    }

    #[test]
    fn test_filter_bank() {
        let fb = Paradigm::motor_imagery()
            .filter_bank(&[(8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0)]);
        assert_eq!(fb.n_filters(), 4);
    }

    #[test]
    fn test_display() {
        let mi = Paradigm::motor_imagery().interval(0.5, 3.5).resample(128.0);
        let s = format!("{mi}");
        assert!(s.contains("MotorImagery"));
        assert!(s.contains("8–30 Hz"));
        assert!(s.contains("@128Hz"));
    }

    #[test]
    fn test_resting_state() {
        let rs = Paradigm::resting_state();
        assert_eq!(rs.resample_hz, Some(128.0));
        assert_eq!(rs.epoch_duration(), Some(40.0));
        assert!(rs.is_compatible("rstate"));
    }

    #[test]
    fn test_fixed_interval() {
        let fi = Paradigm::fixed_interval(2.0, 1.0);
        assert_eq!(fi.epoch_duration(), Some(2.0));
        assert!(fi.is_compatible("anything"));
    }
}
