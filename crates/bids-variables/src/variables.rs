//! BIDS variable types: simple, sparse-run, and dense-run.
//!
//! Implements the three variable types used in BIDS statistical modeling:
//! [`SimpleVariable`] (participant/session-level), [`SparseRunVariable`]
//! (event-level with onset/duration), and [`DenseRunVariable`] (continuous
//! regressors sampled at TR).

use bids_core::entities::StringEntities;

/// A simple variable with no timing information.
///
/// Represents a column from a BIDS tabular file such as `participants.tsv`,
/// `sessions.tsv`, or `scans.tsv`. Each variable has a name, source file
/// identifier, string values, and an entity index that maps each row to
/// its BIDS entities (e.g., which subject each value belongs to).
///
/// Values are stored as both strings and parsed floats (NaN for non-numeric).
/// The `is_numeric` flag indicates whether all values are parseable as f64.
///
/// Corresponds to PyBIDS' `SimpleVariable` class.
///
/// # Example
///
/// ```
/// use bids_variables::SimpleVariable;
/// use std::collections::HashMap;
///
/// let index = vec![
///     HashMap::from([("subject".into(), "01".into())]),
///     HashMap::from([("subject".into(), "02".into())]),
/// ];
/// let var = SimpleVariable::new("age", "participants",
///     vec!["25".into(), "30".into()], index);
///
/// assert_eq!(var.len(), 2);
/// assert!(var.is_numeric);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleVariable {
    pub name: String,
    pub source: String,
    pub values: Vec<f64>,
    pub str_values: Vec<String>,
    pub index: Vec<StringEntities>,
    pub entities: StringEntities,
    pub is_numeric: bool,
}

impl SimpleVariable {
    pub fn new(
        name: &str,
        source: &str,
        values: Vec<String>,
        index: Vec<StringEntities>,
    ) -> Self {
        let numeric_values: Vec<f64> = values.iter()
            .map(|v| v.parse().unwrap_or(f64::NAN))
            .collect();
        let is_numeric = values.iter()
            .all(|v| v.parse::<f64>().is_ok() || v.is_empty());
        let entities = extract_common_entities(&index);

        Self {
            name: name.to_string(),
            source: source.to_string(),
            values: numeric_values,
            str_values: values,
            index,
            entities,
            is_numeric,
        }
    }

    pub fn len(&self) -> usize { self.str_values.len() }
    pub fn is_empty(&self) -> bool { self.str_values.is_empty() }

    /// Clone with optional data/name replacement.
    pub fn clone_with(&self, data: Option<Vec<String>>, name: Option<&str>) -> Self {
        let mut cloned = self.clone();
        if let Some(d) = data {
            cloned.values = d.iter().map(|v| v.parse().unwrap_or(f64::NAN)).collect();
            cloned.str_values = d;
        }
        if let Some(n) = name {
            cloned.name = n.to_string();
        }
        cloned
    }

    /// Filter rows matching given entity criteria.
    pub fn filter(&self, filters: &StringEntities) -> Self {
        let mut values = Vec::new();
        let mut index = Vec::new();

        for (i, row_ents) in self.index.iter().enumerate() {
            if filters.iter().all(|(k, v)| row_ents.get(k).is_none_or(|rv| rv == v)) {
                values.push(self.str_values[i].clone());
                index.push(row_ents.clone());
            }
        }

        Self::new(&self.name, &self.source, values, index)
    }

    /// Convert to tabular rows.
    pub fn to_rows(&self) -> Vec<StringEntities> {
        self.str_values.iter().enumerate().map(|(i, val)| {
            let mut row = self.index.get(i).cloned().unwrap_or_default();
            row.insert("amplitude".into(), val.clone());
            row.insert("condition".into(), self.name.clone());
            row
        }).collect()
    }
}

/// A sparse run variable representing events with onset, duration, and amplitude.
///
/// Loaded from `_events.tsv` files, each event has a time point (onset),
/// a duration, and an amplitude value. Multiple runs can be represented in
/// a single variable via the `run_info` vector.
///
/// Can be converted to a [`DenseRunVariable`] via [`to_dense()`](Self::to_dense)
/// for convolution with hemodynamic response functions or other time-domain
/// operations.
///
/// Corresponds to PyBIDS' `SparseRunVariable` class.
#[derive(Debug, Clone)]
pub struct SparseRunVariable {
    pub name: String,
    pub source: String,
    pub onset: Vec<f64>,
    pub duration: Vec<f64>,
    pub amplitude: Vec<f64>,
    pub str_amplitude: Vec<String>,
    pub index: Vec<StringEntities>,
    pub entities: StringEntities,
    pub run_info: Vec<super::node::RunInfo>,
}

impl SparseRunVariable {
    pub fn new(
        name: &str,
        source: &str,
        onset: Vec<f64>,
        duration: Vec<f64>,
        amplitude: Vec<String>,
        index: Vec<StringEntities>,
        run_info: Vec<super::node::RunInfo>,
    ) -> Self {
        let numeric_amp: Vec<f64> = amplitude.iter()
            .map(|v| v.parse().unwrap_or(f64::NAN))
            .collect();
        let mut entities = extract_common_entities(&index);
        // Also include common entities from run_info
        if let Some(first_run) = run_info.first() {
            for (k, v) in &first_run.entities {
                if run_info.iter().all(|r| r.entities.get(k) == Some(v)) {
                    entities.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }

        Self {
            name: name.to_string(),
            source: source.to_string(),
            onset,
            duration,
            amplitude: numeric_amp,
            str_amplitude: amplitude,
            index,
            entities,
            run_info,
        }
    }

    pub fn len(&self) -> usize { self.onset.len() }
    pub fn is_empty(&self) -> bool { self.onset.is_empty() }

    /// Total duration of all runs.
    pub fn get_duration(&self) -> f64 {
        self.run_info.iter().map(|r| r.duration).sum()
    }

    /// Convert sparse to dense representation using GCD-based bin size.
    pub fn to_dense(&self, sampling_rate: Option<f64>) -> DenseRunVariable {
        let onsets_ms: Vec<i64> = self.onset.iter()
            .map(|o| (o * 1000.0).round() as i64).collect();
        let durations_ms: Vec<i64> = self.duration.iter()
            .map(|d| (d * 1000.0).round() as i64).collect();

        let all_vals: Vec<i64> = onsets_ms.iter().chain(durations_ms.iter())
            .copied().filter(|&v| v > 0).collect();
        let gcd_val = all_vals.iter().copied()
            .reduce(gcd_pair).unwrap_or(1).max(1);

        let bin_sr = 1000.0 / gcd_val as f64;
        let sr = sampling_rate.map_or(bin_sr, |s| s.max(bin_sr));
        let total_duration = self.get_duration();
        let n_samples = (total_duration * sr).ceil() as usize;
        let mut ts = vec![0.0f64; n_samples];

        let mut run_offset = 0.0;
        let mut last_onset = -1.0f64;
        let mut run_i = 0;

        for i in 0..self.onset.len() {
            if self.onset[i] < last_onset && run_i + 1 < self.run_info.len() {
                run_offset += self.run_info[run_i].duration;
                run_i += 1;
            }
            let onset_sample = ((run_offset + self.onset[i]) * sr).round() as usize;
            let dur_samples = (self.duration[i] * sr).round() as usize;
            let offset_sample = (onset_sample + dur_samples).min(n_samples);
            for ts_val in ts.iter_mut().take(offset_sample).skip(onset_sample) {
                *ts_val = self.amplitude[i];
            }
            last_onset = self.onset[i];
        }

        let final_sr = sampling_rate.unwrap_or(sr);
        if (final_sr - sr).abs() > 0.001 {
            let new_n = (total_duration * final_sr).ceil() as usize;
            ts = linear_resample(&ts, new_n);
        }

        DenseRunVariable::new(&self.name, &self.source, ts, final_sr, self.run_info.clone())
    }

    /// Filter events by entity criteria.
    pub fn filter(&self, filters: &StringEntities) -> Self {
        let mut onset = Vec::new();
        let mut duration = Vec::new();
        let mut amplitude = Vec::new();
        let mut index = Vec::new();

        for (i, row_ents) in self.index.iter().enumerate() {
            if filters.iter().all(|(k, v)| row_ents.get(k).is_none_or(|rv| rv == v)) {
                onset.push(self.onset[i]);
                duration.push(self.duration[i]);
                amplitude.push(self.str_amplitude[i].clone());
                index.push(row_ents.clone());
            }
        }

        Self::new(&self.name, &self.source, onset, duration, amplitude, index, self.run_info.clone())
    }

    /// Convert to tabular rows.
    pub fn to_rows(&self) -> Vec<StringEntities> {
        (0..self.onset.len()).map(|i| {
            let mut row = self.index.get(i).cloned().unwrap_or_default();
            row.insert("onset".into(), self.onset[i].to_string());
            row.insert("duration".into(), self.duration[i].to_string());
            row.insert("amplitude".into(), self.str_amplitude[i].clone());
            row.insert("condition".into(), self.name.clone());
            row
        }).collect()
    }
}

/// A dense run variable with uniformly-sampled time series data.
///
/// Represents continuous signals such as physiological recordings (`_physio.tsv.gz`),
/// stimulus waveforms (`_stim.tsv.gz`), or confound regressors (`_regressors.tsv`).
/// Data is stored as a vector of f64 values at a fixed sampling rate.
///
/// Supports resampling to different rates via [`resample()`](Self::resample) and
/// TR-based downsampling via [`resample_to_tr()`](Self::resample_to_tr).
///
/// Corresponds to PyBIDS' `DenseRunVariable` class.
#[derive(Debug, Clone)]
pub struct DenseRunVariable {
    pub name: String,
    pub source: String,
    pub values: Vec<f64>,
    pub sampling_rate: f64,
    pub run_info: Vec<super::node::RunInfo>,
    pub entities: StringEntities,
}

impl DenseRunVariable {
    pub fn new(
        name: &str,
        source: &str,
        values: Vec<f64>,
        sampling_rate: f64,
        run_info: Vec<super::node::RunInfo>,
    ) -> Self {
        let mut entities = StringEntities::new();
        for ri in &run_info {
            for (k, v) in &ri.entities {
                entities.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        Self {
            name: name.into(), source: source.into(),
            values, sampling_rate, run_info, entities,
        }
    }

    pub fn len(&self) -> usize { self.values.len() }
    pub fn is_empty(&self) -> bool { self.values.is_empty() }

    /// Resample to a different sampling rate.
    pub fn resample(&self, new_sr: f64) -> Self {
        if (new_sr - self.sampling_rate).abs() < 0.001 {
            return self.clone();
        }
        let new_n = ((self.values.len() as f64) * new_sr / self.sampling_rate).ceil() as usize;
        Self {
            name: self.name.clone(), source: self.source.clone(),
            values: linear_resample(&self.values, new_n),
            sampling_rate: new_sr, run_info: self.run_info.clone(),
            entities: self.entities.clone(),
        }
    }

    /// Resample to TR-based sampling rate.
    pub fn resample_to_tr(&self) -> Self {
        self.run_info.first()
            .filter(|ri| ri.tr > 0.0)
            .map(|ri| self.resample(1.0 / ri.tr))
            .unwrap_or_else(|| self.clone())
    }

    /// Convert to tabular rows.
    pub fn to_rows(&self) -> Vec<StringEntities> {
        let interval = 1.0 / self.sampling_rate;
        self.values.iter().enumerate().map(|(i, val)| {
            let mut row = self.entities.clone();
            row.insert("onset".into(), (i as f64 * interval).to_string());
            row.insert("duration".into(), interval.to_string());
            row.insert("amplitude".into(), val.to_string());
            row.insert("condition".into(), self.name.clone());
            row
        }).collect()
    }
}

impl SparseRunVariable {
    /// Select specific row indices.
    pub fn select_rows(&self, indices: &[usize]) -> Self {
        Self::new(
            &self.name, &self.source,
            indices.iter().filter_map(|&i| self.onset.get(i).copied()).collect(),
            indices.iter().filter_map(|&i| self.duration.get(i).copied()).collect(),
            indices.iter().filter_map(|&i| self.str_amplitude.get(i).cloned()).collect(),
            indices.iter().filter_map(|&i| self.index.get(i).cloned()).collect(),
            self.run_info.clone(),
        )
    }

    /// Split into multiple variables based on a grouper.
    pub fn split(&self, group_col: &str) -> Vec<Self> {
        let mut groups: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (i, row) in self.index.iter().enumerate() {
            let key = row.get(group_col).cloned().unwrap_or_default();
            groups.entry(key).or_default().push(i);
        }
        groups.into_iter().map(|(key, indices)| {
            let mut var = self.select_rows(&indices);
            var.name = format!("{}.{}", self.name, key);
            var
        }).collect()
    }
}

impl DenseRunVariable {
    /// Build entity index with timestamps for each sample.
    pub fn build_entity_index(&self) -> Vec<(f64, StringEntities)> {
        let interval = 1.0 / self.sampling_rate;
        let mut result = Vec::with_capacity(self.values.len());
        let mut offset = 0.0;
        let mut run_i = 0;
        for (i, _) in self.values.iter().enumerate() {
            let t = i as f64 * interval;
            // Advance run if we've passed the current run's duration
            while run_i + 1 < self.run_info.len() && t >= offset + self.run_info[run_i].duration {
                offset += self.run_info[run_i].duration;
                run_i += 1;
            }
            let ents = self.run_info.get(run_i)
                .map(|ri| ri.entities.clone())
                .unwrap_or_default();
            result.push((t, ents));
        }
        result
    }
}

/// Get a grouper key for groupby operations.
pub fn get_grouper(index: &[StringEntities], group_by: &[&str]) -> Vec<String> {
    index.iter().map(|row| {
        group_by.iter()
            .map(|k| row.get(*k).cloned().unwrap_or_default())
            .collect::<Vec<_>>()
            .join("@@@")
    }).collect()
}

/// Apply a function to groups defined by a grouper.
pub fn apply_grouped<F>(values: &[f64], grouper: &[String], func: F) -> Vec<f64>
where F: Fn(&[f64]) -> Vec<f64>
{
    let mut groups: std::collections::HashMap<&str, Vec<(usize, f64)>> = std::collections::HashMap::new();
    for (i, (val, key)) in values.iter().zip(grouper).enumerate() {
        groups.entry(key.as_str()).or_default().push((i, *val));
    }
    let mut result = vec![0.0; values.len()];
    for group in groups.values() {
        let group_vals: Vec<f64> = group.iter().map(|(_, v)| *v).collect();
        let transformed = func(&group_vals);
        for ((idx, _), new_val) in group.iter().zip(transformed) {
            result[*idx] = new_val;
        }
    }
    result
}

// ──────────────────────── Merge functions ────────────────────────

/// Merge a list of simple variables with the same name.
pub fn merge_simple(variables: &[&SimpleVariable]) -> Option<SimpleVariable> {
    let first = variables.first()?;
    let mut all_values = Vec::new();
    let mut all_index = Vec::new();
    for v in variables {
        all_values.extend(v.str_values.iter().cloned());
        all_index.extend(v.index.iter().cloned());
    }
    Some(SimpleVariable::new(&first.name, &first.source, all_values, all_index))
}

/// Merge sparse run variables.
pub fn merge_sparse(variables: &[&SparseRunVariable]) -> Option<SparseRunVariable> {
    let first = variables.first()?;
    let mut onset = Vec::new();
    let mut duration = Vec::new();
    let mut amplitude = Vec::new();
    let mut index = Vec::new();
    let mut run_info = Vec::new();
    for v in variables {
        onset.extend(&v.onset);
        duration.extend(&v.duration);
        amplitude.extend(v.str_amplitude.iter().cloned());
        index.extend(v.index.iter().cloned());
        run_info.extend(v.run_info.iter().cloned());
    }
    Some(SparseRunVariable::new(&first.name, &first.source, onset, duration, amplitude, index, run_info))
}

// ──────────────────────── Helpers ────────────────────────

fn extract_common_entities(index: &[StringEntities]) -> StringEntities {
    let mut common = StringEntities::new();
    if let Some(first) = index.first() {
        for (k, v) in first {
            if index.iter().all(|row| row.get(k) == Some(v)) {
                common.insert(k.clone(), v.clone());
            }
        }
    }
    common
}

fn gcd_pair(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 { let t = b; b = a % b; a = t; }
    a
}

fn linear_resample(values: &[f64], new_n: usize) -> Vec<f64> {
    if new_n == 0 || values.is_empty() { return vec![]; }
    if new_n == values.len() { return values.to_vec(); }
    let old_n = values.len();
    (0..new_n).map(|i| {
        let t = if new_n > 1 {
            (i as f64) * (old_n as f64 - 1.0) / (new_n as f64 - 1.0)
        } else { 0.0 };
        let lo = t.floor() as usize;
        let hi = (lo + 1).min(old_n - 1);
        let frac = t - lo as f64;
        values[lo] * (1.0 - frac) + values[hi] * frac
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::RunInfo;
    use std::collections::HashMap;

    #[test]
    fn test_sparse_to_dense() {
        let ri = RunInfo {
            entities: StringEntities::new(), duration: 10.0, tr: 2.0,
            image: None, n_vols: 5,
        };
        let sparse = SparseRunVariable::new(
            "trial_type", "events",
            vec![1.0, 3.0], vec![1.0, 2.0],
            vec!["1".into(), "1".into()],
            vec![StringEntities::new(), StringEntities::new()],
            vec![ri],
        );
        let dense = sparse.to_dense(Some(10.0));
        assert_eq!(dense.sampling_rate, 10.0);
        assert_eq!(dense.values.len(), 100);
        assert_eq!(dense.values[10], 1.0);
        assert_eq!(dense.values[0], 0.0);
    }

    #[test]
    fn test_simple_filter() {
        let idx = vec![
            HashMap::from([("subject".into(), "01".into())]),
            HashMap::from([("subject".into(), "02".into())]),
        ];
        let var = SimpleVariable::new("age", "participants",
            vec!["25".into(), "30".into()], idx);
        let filtered = var.filter(&HashMap::from([("subject".into(), "01".into())]));
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.str_values[0], "25");
    }

    #[test]
    fn test_merge_simple() {
        let v1 = SimpleVariable::new("age", "participants",
            vec!["25".into()],
            vec![HashMap::from([("subject".into(), "01".into())])]);
        let v2 = SimpleVariable::new("age", "participants",
            vec!["30".into()],
            vec![HashMap::from([("subject".into(), "02".into())])]);
        let merged = merge_simple(&[&v1, &v2]).unwrap();
        assert_eq!(merged.len(), 2);
    }
}
