//! Channel harmonization across datasets.
//!
//! When combining data from multiple datasets for cross-dataset ML, channels
//! and sampling rates must be aligned. This module provides functions to
//! find common channels, harmonize sampling rates, and standardize channel
//! ordering.
//!
//! Inspired by MOABB's `BaseProcessing.match_all()`.

use crate::data::EegData;
use std::collections::{BTreeSet, HashSet};

/// Strategy for combining channel sets across datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelStrategy {
    /// Keep only channels present in ALL datasets (safe, no missing data).
    Intersect,
    /// Keep all channels from any dataset (requires interpolation for missing).
    Union,
}

/// Result of channel harmonization analysis.
#[derive(Debug, Clone)]
pub struct HarmonizationPlan {
    /// Final channel list to use, in sorted order.
    pub channels: Vec<String>,
    /// Minimum sampling rate across all inputs.
    pub target_sr: f64,
    /// Per-input: which channels are present (true) or missing (false).
    pub channel_availability: Vec<Vec<bool>>,
}

impl std::fmt::Display for HarmonizationPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HarmonizationPlan({} channels @ {:.1} Hz",
            self.channels.len(),
            self.target_sr
        )?;
        let n_inputs = self.channel_availability.len();
        if n_inputs > 0 {
            let all_present = self
                .channel_availability
                .iter()
                .all(|avail| avail.iter().all(|&a| a));
            if all_present {
                write!(f, ", all channels present in all inputs")?;
            } else {
                let missing: usize = self
                    .channel_availability
                    .iter()
                    .map(|avail| avail.iter().filter(|&&a| !a).count())
                    .sum();
                write!(f, ", {missing} missing channel-input pairs")?;
            }
        }
        write!(f, ")")
    }
}

/// Analyze multiple EegData recordings and produce a harmonization plan.
///
/// Examines all inputs to find the common channel set (or union) and
/// the minimum sampling rate.
///
/// # Arguments
/// - `inputs`: slice of `EegData` references to harmonize
/// - `strategy`: how to combine channel sets
/// - `ignore_channels`: channel names to exclude (e.g., `["STIM", "STATUS"]`)
/// - `sr_shift`: small negative shift to avoid off-by-one sample counts
///   (MOABB uses `-0.5` for this; pass `0.0` if not needed)
#[must_use]
pub fn plan_harmonization(
    inputs: &[&EegData],
    strategy: ChannelStrategy,
    ignore_channels: &[&str],
    sr_shift: f64,
) -> HarmonizationPlan {
    if inputs.is_empty() {
        return HarmonizationPlan {
            channels: Vec::new(),
            target_sr: 0.0,
            channel_availability: Vec::new(),
        };
    }

    let ignore: HashSet<&str> = ignore_channels.iter().copied().collect();

    // Collect channel sets
    let channel_sets: Vec<BTreeSet<String>> = inputs
        .iter()
        .map(|d| {
            d.channel_labels
                .iter()
                .filter(|ch| !ignore.contains(ch.as_str()))
                .cloned()
                .collect()
        })
        .collect();

    let channels: BTreeSet<String> = match strategy {
        ChannelStrategy::Intersect => {
            let mut result = channel_sets[0].clone();
            for set in &channel_sets[1..] {
                result = result.intersection(set).cloned().collect();
            }
            result
        }
        ChannelStrategy::Union => {
            let mut result = BTreeSet::new();
            for set in &channel_sets {
                result = result.union(set).cloned().collect();
            }
            result
        }
    };

    let channels: Vec<String> = channels.into_iter().collect();

    // Find minimum sampling rate
    let target_sr = inputs
        .iter()
        .filter_map(|d| d.sampling_rates.first().copied())
        .fold(f64::INFINITY, f64::min)
        + sr_shift;

    // Build availability map
    let channel_availability = inputs
        .iter()
        .map(|d| {
            let input_channels: HashSet<&str> =
                d.channel_labels.iter().map(|s| s.as_str()).collect();
            channels
                .iter()
                .map(|ch| input_channels.contains(ch.as_str()))
                .collect()
        })
        .collect();

    HarmonizationPlan {
        channels,
        target_sr,
        channel_availability,
    }
}

/// Apply a harmonization plan to a single `EegData`, producing a new one
/// with the specified channels (in order) and sampling rate.
///
/// Missing channels are filled with zeros.
#[must_use]
pub fn apply_harmonization(data: &EegData, plan: &HarmonizationPlan) -> EegData {
    let n_samples = data.data.first().map_or(0, |ch| ch.len());
    let channel_map: std::collections::HashMap<&str, usize> = data
        .channel_labels
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    let mut new_data = Vec::with_capacity(plan.channels.len());
    let mut new_rates = Vec::with_capacity(plan.channels.len());
    let sr = data
        .sampling_rates
        .first()
        .copied()
        .unwrap_or(plan.target_sr);

    for ch_name in &plan.channels {
        if let Some(&idx) = channel_map.get(ch_name.as_str()) {
            new_data.push(data.data[idx].clone());
            new_rates.push(data.sampling_rates.get(idx).copied().unwrap_or(sr));
        } else {
            // Missing channel — fill with zeros
            new_data.push(vec![0.0; n_samples]);
            new_rates.push(sr);
        }
    }

    let mut result = EegData {
        channel_labels: plan.channels.clone(),
        data: new_data,
        sampling_rates: new_rates,
        duration: data.duration,
        annotations: data.annotations.clone(),
        stim_channel_indices: Vec::new(), // stim channels may not be in the plan
        is_discontinuous: data.is_discontinuous,
        record_onsets: data.record_onsets.clone(),
    };

    // Resample if needed
    if (sr - plan.target_sr).abs() > 0.01 {
        result = result.resample(plan.target_sr);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::EegData;

    fn make_data(channels: &[&str], sr: f64, n_samples: usize) -> EegData {
        EegData {
            channel_labels: channels.iter().map(|s| (*s).to_string()).collect(),
            data: channels
                .iter()
                .enumerate()
                .map(|(i, _)| (0..n_samples).map(|s| (s + i) as f64).collect())
                .collect(),
            sampling_rates: vec![sr; channels.len()],
            duration: n_samples as f64 / sr,
            annotations: Vec::new(),
            stim_channel_indices: Vec::new(),
            is_discontinuous: false,
            record_onsets: Vec::new(),
        }
    }

    #[test]
    fn test_intersect_strategy() {
        let d1 = make_data(&["Fz", "Cz", "Pz", "STIM"], 256.0, 256);
        let d2 = make_data(&["Fz", "Cz", "Oz", "STIM"], 512.0, 512);

        let plan = plan_harmonization(&[&d1, &d2], ChannelStrategy::Intersect, &["STIM"], 0.0);
        assert_eq!(plan.channels, vec!["Cz", "Fz"]); // sorted, intersected, STIM excluded
        assert!((plan.target_sr - 256.0).abs() < 1e-10); // min SR
    }

    #[test]
    fn test_union_strategy() {
        let d1 = make_data(&["Fz", "Cz"], 256.0, 256);
        let d2 = make_data(&["Cz", "Pz"], 256.0, 256);

        let plan = plan_harmonization(&[&d1, &d2], ChannelStrategy::Union, &[], 0.0);
        assert_eq!(plan.channels, vec!["Cz", "Fz", "Pz"]);
        // d1 has Fz, Cz but not Pz
        assert_eq!(plan.channel_availability[0], vec![true, true, false]);
        // d2 has Cz, Pz but not Fz
        assert_eq!(plan.channel_availability[1], vec![true, false, true]);
    }

    #[test]
    fn test_apply_harmonization() {
        let d1 = make_data(&["Fz", "Cz", "Pz"], 256.0, 256);
        let plan = HarmonizationPlan {
            channels: vec!["Cz".into(), "Fz".into(), "Oz".into()],
            target_sr: 256.0,
            channel_availability: vec![vec![true, true, false]],
        };

        let result = apply_harmonization(&d1, &plan);
        assert_eq!(result.channel_labels, vec!["Cz", "Fz", "Oz"]);
        assert_eq!(result.data.len(), 3);
        // Oz should be zeros (missing)
        assert!(result.data[2].iter().all(|&v| v == 0.0));
        // Cz and Fz should have data
        assert!(result.data[0].iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_sr_shift() {
        let d1 = make_data(&["Fz"], 128.0, 128);
        let plan = plan_harmonization(&[&d1], ChannelStrategy::Intersect, &[], -0.5);
        assert!((plan.target_sr - 127.5).abs() < 1e-10);
    }
}
