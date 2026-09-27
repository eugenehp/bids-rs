//! Variable collections for grouping and merging BIDS variables.
//!
//! Collections group multiple variables at the same level of the BIDS hierarchy,
//! providing operations like variable lookup, wildcard matching, filtering,
//! and merging across runs or subjects.

use crate::node::RunInfo;
use crate::variables::{DenseRunVariable, SimpleVariable, SparseRunVariable, merge_simple};
use bids_core::entities::StringEntities;
use std::collections::{HashMap, HashSet};

fn source_to_level(source: &str) -> &str {
    match source {
        "events" | "physio" | "stim" | "regressors" => "run",
        "scans" => "session",
        "sessions" => "subject",
        "participants" => "dataset",
        other => other,
    }
}

/// A collection of simple BIDS variables at a single level of the hierarchy.
///
/// Groups [`SimpleVariable`]s by name, automatically merging variables with
/// the same name and inferring the hierarchy level from the variable source
/// (events→run, participants→dataset, etc.).
///
/// Supports variable lookup by name, wildcard pattern matching, entity-based
/// filtering, and subset selection.
///
/// Corresponds to PyBIDS' `BIDSVariableCollection` class.
///
/// # Run-level timing
///
/// When a collection is built from a [`RunVariableCollection`] via
/// [`from_run_collection()`](Self::from_run_collection), each variable keeps
/// the timing information it was loaded with. Variables that came from
/// `_events.tsv` retain their onsets and durations in [`timing`](Self::timing);
/// variables that are uniformly sampled are listed in [`dense`](Self::dense).
/// This is what allows the `ToDense`, `Resample`, `Assign`, and `Convolve`
/// transformations to operate — without it a collection is a flat table with
/// no notion of time.
#[derive(Debug, Clone)]
pub struct VariableCollection {
    pub variables: HashMap<String, SimpleVariable>,
    pub level: String,
    pub entities: StringEntities,
    /// Onset/duration per variable, for variables in sparse (event) form.
    pub timing: HashMap<String, VariableTiming>,
    /// Names of variables held in dense (uniformly sampled) form.
    pub dense: HashSet<String>,
    /// Sampling rate in Hz used for dense variables.
    pub sampling_rate: f64,
    /// Run metadata (duration, TR) when this collection is run-level.
    pub run_info: Vec<RunInfo>,
    /// Named variable groups created by the `Group` transformation.
    pub groups: HashMap<String, Vec<String>>,
}

/// Onset and duration vectors for a variable in sparse (event) form.
///
/// Parallel to the variable's own value vector: entry `i` of `onset` and
/// `duration` describes event `i`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VariableTiming {
    pub onset: Vec<f64>,
    pub duration: Vec<f64>,
}

/// Default sampling rate (Hz) for dense variables, matching PyBIDS.
pub const DEFAULT_SAMPLING_RATE: f64 = 10.0;

impl Default for VariableCollection {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            level: String::new(),
            entities: StringEntities::new(),
            timing: HashMap::new(),
            dense: HashSet::new(),
            sampling_rate: DEFAULT_SAMPLING_RATE,
            run_info: Vec::new(),
            groups: HashMap::new(),
        }
    }
}

impl VariableCollection {
    pub fn new(vars: Vec<SimpleVariable>) -> Self {
        let level = vars
            .first()
            .map(|v| source_to_level(&v.source).to_string())
            .unwrap_or_default();

        // Merge variables with same name
        let mut by_name: HashMap<String, Vec<&SimpleVariable>> = HashMap::new();
        for v in &vars {
            by_name.entry(v.name.clone()).or_default().push(v);
        }
        let mut variables = HashMap::new();
        for (name, var_list) in &by_name {
            if let Some(m) = merge_simple(var_list) {
                variables.insert(name.clone(), m);
            }
        }

        let entities = index_common_entities(&variables);
        Self {
            variables,
            level,
            entities,
            ..Default::default()
        }
    }

    pub fn get(&self, name: &str) -> Option<&SimpleVariable> {
        self.variables.get(name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.variables
            .keys()
            .map(std::string::String::as_str)
            .collect()
    }

    /// Match variable names against a pattern (regex or glob).
    pub fn match_variables(&self, pattern: &str, use_regex: bool) -> Vec<&str> {
        if use_regex {
            let re =
                regex::Regex::new(pattern).unwrap_or_else(|_| regex::Regex::new("$^").unwrap());
            self.variables
                .keys()
                .filter(|k| re.is_match(k))
                .map(std::string::String::as_str)
                .collect()
        } else {
            self.variables
                .keys()
                .filter(|k| glob_match(pattern, k))
                .map(std::string::String::as_str)
                .collect()
        }
    }

    /// Convert all variables to tabular rows (long format).
    pub fn to_rows(&self) -> Vec<StringEntities> {
        self.variables
            .values()
            .flat_map(super::variables::SimpleVariable::to_rows)
            .collect()
    }

    /// Convert to wide format: each variable becomes a column.
    /// Returns `(column_names, rows)` where each row is a `Vec<String>`.
    pub fn to_wide(&self) -> (Vec<String>, Vec<Vec<String>>) {
        // Collect all entity columns
        let mut entity_cols: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for var in self.variables.values() {
            for row in &var.index {
                entity_cols.extend(row.keys().cloned());
            }
        }
        let var_names: Vec<String> = self.variables.keys().cloned().collect();
        let mut col_names: Vec<String> = entity_cols.into_iter().collect();
        col_names.extend(var_names.iter().cloned());

        // Build rows grouped by entity combination
        let n_rows = self
            .variables
            .values()
            .next()
            .map_or(0, super::variables::SimpleVariable::len);
        let mut rows = Vec::with_capacity(n_rows);
        for i in 0..n_rows {
            let mut row = Vec::with_capacity(col_names.len());
            // Entity columns
            let first_var = self.variables.values().next();
            let ent_row = first_var.and_then(|v| v.index.get(i));
            for col in &col_names {
                if self.variables.contains_key(col) {
                    // Variable column
                    let val = self
                        .variables
                        .get(col)
                        .and_then(|v| v.str_values.get(i))
                        .cloned()
                        .unwrap_or_default();
                    row.push(val);
                } else {
                    // Entity column
                    let val = ent_row
                        .and_then(|r| r.get(col))
                        .cloned()
                        .unwrap_or_default();
                    row.push(val);
                }
            }
            rows.push(row);
        }
        (col_names, rows)
    }

    /// Create collection from tabular rows.
    pub fn from_rows(rows: &[StringEntities], source: &str) -> Self {
        let mut by_name: HashMap<String, (Vec<String>, Vec<StringEntities>)> = HashMap::new();
        for row in rows {
            let name = row
                .get("condition")
                .cloned()
                .unwrap_or_else(|| "unknown".into());
            let amp = row.get("amplitude").cloned().unwrap_or_default();
            let ents: StringEntities = row
                .iter()
                .filter(|(k, _)| k.as_str() != "condition" && k.as_str() != "amplitude")
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let entry = by_name.entry(name).or_default();
            entry.0.push(amp);
            entry.1.push(ents);
        }
        let vars: Vec<SimpleVariable> = by_name
            .into_iter()
            .map(|(name, (values, index))| SimpleVariable::new(&name, source, values, index))
            .collect();
        Self::new(vars)
    }

    /// Build a collection from a run-level collection, preserving timing.
    ///
    /// Sparse variables keep their onsets and durations in [`timing`](Self::timing);
    /// dense variables are recorded in [`dense`](Self::dense) and their sample
    /// values become the variable's values. This is the constructor that makes a
    /// collection usable by the time-aware transformations (`ToDense`,
    /// `Resample`, `Assign`, `Convolve`).
    pub fn from_run_collection(rc: &RunVariableCollection) -> Self {
        let mut variables = HashMap::new();
        let mut timing = HashMap::new();
        let mut dense = HashSet::new();
        let mut run_info: Vec<RunInfo> = Vec::new();

        for v in &rc.sparse {
            let var =
                SimpleVariable::new(&v.name, &v.source, v.str_amplitude.clone(), v.index.clone());
            timing.insert(
                v.name.clone(),
                VariableTiming {
                    onset: v.onset.clone(),
                    duration: v.duration.clone(),
                },
            );
            variables.insert(v.name.clone(), var);
            if run_info.is_empty() {
                run_info = v.run_info.clone();
            }
        }

        for v in &rc.dense {
            let str_values: Vec<String> = v
                .values
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            let index = dense_index(v);
            let var = SimpleVariable::new(&v.name, &v.source, str_values, index);
            dense.insert(v.name.clone());
            variables.insert(v.name.clone(), var);
            if run_info.is_empty() {
                run_info = v.run_info.clone();
            }
        }

        let entities = index_common_entities(&variables);
        let level = variables
            .values()
            .next()
            .map(|v| source_to_level(&v.source).to_string())
            .unwrap_or_else(|| "run".into());

        Self {
            variables,
            level,
            entities,
            timing,
            dense,
            sampling_rate: rc.sampling_rate,
            run_info,
            groups: HashMap::new(),
        }
    }

    /// Total duration in seconds across all runs in this collection.
    #[must_use]
    pub fn run_duration(&self) -> f64 {
        self.run_info.iter().map(|r| r.duration).sum()
    }

    /// Whether `name` is held in dense (uniformly sampled) form.
    #[must_use]
    pub fn is_dense(&self, name: &str) -> bool {
        self.dense.contains(name)
    }

    /// Whether `name` is held in sparse (event) form, i.e. it has timing.
    #[must_use]
    pub fn is_sparse(&self, name: &str) -> bool {
        self.timing.contains_key(name)
    }

    /// Insert a variable, replacing any timing/dense state under its name.
    pub fn insert_var(&mut self, var: SimpleVariable) {
        self.timing.remove(&var.name);
        self.dense.remove(&var.name);
        self.variables.insert(var.name.clone(), var);
    }

    /// Insert a sparse variable together with its event timing.
    pub fn insert_sparse(&mut self, var: SimpleVariable, timing: VariableTiming) {
        self.dense.remove(&var.name);
        self.timing.insert(var.name.clone(), timing);
        self.variables.insert(var.name.clone(), var);
    }

    /// Insert a dense variable sampled at [`sampling_rate`](Self::sampling_rate).
    pub fn insert_dense(&mut self, var: SimpleVariable) {
        self.timing.remove(&var.name);
        self.dense.insert(var.name.clone());
        self.variables.insert(var.name.clone(), var);
    }

    /// Remove a variable and all of its associated state.
    pub fn remove_var(&mut self, name: &str) -> Option<SimpleVariable> {
        self.timing.remove(name);
        self.dense.remove(name);
        for members in self.groups.values_mut() {
            members.retain(|m| m != name);
        }
        self.variables.remove(name)
    }

    /// Rename a variable, carrying its timing/dense state to the new name.
    pub fn rename_var(&mut self, from: &str, to: &str) {
        if let Some(mut var) = self.variables.remove(from) {
            var.name = to.to_string();
            self.variables.insert(to.to_string(), var);
        } else {
            return;
        }
        if let Some(t) = self.timing.remove(from) {
            self.timing.insert(to.to_string(), t);
        }
        if self.dense.remove(from) {
            self.dense.insert(to.to_string());
        }
        for members in self.groups.values_mut() {
            for m in members.iter_mut() {
                if m == from {
                    *m = to.to_string();
                }
            }
        }
    }

    /// Copy a variable under a new name, carrying timing/dense state along.
    pub fn copy_var(&mut self, from: &str, to: &str) {
        let Some(src) = self.variables.get(from).cloned() else {
            return;
        };
        let mut copy = src;
        copy.name = to.to_string();
        self.variables.insert(to.to_string(), copy);
        if let Some(t) = self.timing.get(from).cloned() {
            self.timing.insert(to.to_string(), t);
        }
        if self.dense.contains(from) {
            self.dense.insert(to.to_string());
        }
    }

    /// Reconstruct a [`SparseRunVariable`] for `name`, if it has timing.
    #[must_use]
    pub fn as_sparse(&self, name: &str) -> Option<SparseRunVariable> {
        let var = self.variables.get(name)?;
        let t = self.timing.get(name)?;
        Some(SparseRunVariable::new(
            &var.name,
            &var.source,
            t.onset.clone(),
            t.duration.clone(),
            var.str_values.clone(),
            var.index.clone(),
            self.run_info.clone(),
        ))
    }

    /// Reconstruct a [`DenseRunVariable`] for `name`, if it is dense.
    #[must_use]
    pub fn as_dense(&self, name: &str) -> Option<DenseRunVariable> {
        if !self.dense.contains(name) {
            return None;
        }
        let var = self.variables.get(name)?;
        Some(DenseRunVariable::new(
            &var.name,
            &var.source,
            var.values.clone(),
            self.sampling_rate,
            self.run_info.clone(),
        ))
    }

    /// Store a [`DenseRunVariable`] back into the collection.
    ///
    /// Sets the collection's sampling rate to the variable's when the
    /// collection has no other dense variables, so that a lone `Resample`
    /// keeps the collection self-consistent.
    pub fn put_dense(&mut self, dv: &DenseRunVariable) {
        let only =
            self.dense.is_empty() || (self.dense.len() == 1 && self.dense.contains(&dv.name));
        if only {
            self.sampling_rate = dv.sampling_rate;
        }
        let str_values: Vec<String> = dv
            .values
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        let index = dense_index(dv);
        let var = SimpleVariable::new(&dv.name, &dv.source, str_values, index);
        self.insert_dense(var);
    }

    /// Convert a sparse variable to dense form in place.
    ///
    /// No-op if the variable is missing or already dense. Mirrors PyBIDS'
    /// `ToDense` transformation.
    pub fn to_dense_var(&mut self, name: &str, sampling_rate: f64) {
        if self.dense.contains(name) {
            return;
        }
        let Some(sparse) = self.as_sparse(name) else {
            return;
        };
        let dv = sparse.to_dense(Some(sampling_rate));
        self.timing.remove(name);
        self.put_dense(&dv);
    }

    /// Overwrite a variable's values in place, keeping timing/dense state.
    ///
    /// Returns `false` if the variable is not present.
    pub fn set_values(&mut self, name: &str, values: Vec<f64>) -> bool {
        let Some(var) = self.variables.get_mut(name) else {
            return false;
        };
        var.str_values = values
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        var.values = values;
        var.is_numeric = true;
        true
    }

    /// Create or replace `to`, reusing `from`'s index and source.
    ///
    /// When `to` and `from` have the same row count, `from`'s timing/dense
    /// state carries over, so a derived variable stays convolvable. Returns
    /// `false` if `from` is not present.
    pub fn derive_var(&mut self, from: &str, to: &str, values: Vec<f64>) -> bool {
        let Some(src) = self.variables.get(from) else {
            return false;
        };
        let same_len = src.len() == values.len();
        let str_values: Vec<String> = values
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        let index = src.index.clone();
        let source = src.source.clone();
        let mut var = SimpleVariable::new(to, &source, str_values, index);
        var.values = values;
        var.is_numeric = true;

        let timing = if same_len {
            self.timing.get(from).cloned()
        } else {
            None
        };
        let was_dense = same_len && self.dense.contains(from);
        match timing {
            Some(t) => self.insert_sparse(var, t),
            None if was_dense => self.insert_dense(var),
            None => self.insert_var(var),
        }
        true
    }

    /// Keep only the rows at `keep` for `name`, trimming timing in step.
    ///
    /// Dense variables have no per-row timing, so their samples are simply
    /// subset. Returns `false` if the variable is not present.
    pub fn select_rows(&mut self, name: &str, keep: &[usize]) -> bool {
        let Some(var) = self.variables.get(name) else {
            return false;
        };
        let str_values: Vec<String> = keep
            .iter()
            .filter_map(|&i| var.str_values.get(i).cloned())
            .collect();
        let index: Vec<StringEntities> = keep
            .iter()
            .filter_map(|&i| var.index.get(i).cloned())
            .collect();
        let source = var.source.clone();
        let new_var = SimpleVariable::new(name, &source, str_values, index);

        if let Some(t) = self.timing.get(name) {
            let trimmed = VariableTiming {
                onset: keep
                    .iter()
                    .filter_map(|&i| t.onset.get(i).copied())
                    .collect(),
                duration: keep
                    .iter()
                    .filter_map(|&i| t.duration.get(i).copied())
                    .collect(),
            };
            self.insert_sparse(new_var, trimmed);
        } else if self.dense.contains(name) {
            self.insert_dense(new_var);
        } else {
            self.insert_var(new_var);
        }
        true
    }

    /// Number of rows in `name`, or 0 if absent.
    #[must_use]
    pub fn n_rows(&self, name: &str) -> usize {
        self.variables.get(name).map_or(0, SimpleVariable::len)
    }

    /// Resample a dense variable in place. No-op for sparse variables.
    pub fn resample_var(&mut self, name: &str, sampling_rate: f64) {
        let Some(dv) = self.as_dense(name) else {
            return;
        };
        let resampled = dv.resample(sampling_rate);
        self.put_dense(&resampled);
    }
}

/// Build a per-sample entity index for a dense variable.
fn dense_index(v: &DenseRunVariable) -> Vec<StringEntities> {
    v.build_entity_index()
        .into_iter()
        .map(|(t, mut ents)| {
            ents.insert("onset".into(), t.to_string());
            ents
        })
        .collect()
}

/// A collection of run-level BIDS variables.
#[derive(Debug, Default, Clone)]
pub struct RunVariableCollection {
    pub sparse: Vec<SparseRunVariable>,
    pub dense: Vec<DenseRunVariable>,
    pub sampling_rate: f64,
}

impl RunVariableCollection {
    pub fn new(
        sparse: Vec<SparseRunVariable>,
        dense: Vec<DenseRunVariable>,
        sampling_rate: Option<f64>,
    ) -> Self {
        Self {
            sparse,
            dense,
            sampling_rate: sampling_rate.unwrap_or(10.0),
        }
    }

    pub fn get_sparse(&self, name: &str) -> Option<&SparseRunVariable> {
        self.sparse.iter().find(|v| v.name == name)
    }
    pub fn get_dense(&self, name: &str) -> Option<&DenseRunVariable> {
        self.dense.iter().find(|v| v.name == name)
    }

    pub fn sparse_names(&self) -> Vec<&str> {
        self.sparse.iter().map(|v| v.name.as_str()).collect()
    }
    pub fn dense_names(&self) -> Vec<&str> {
        self.dense.iter().map(|v| v.name.as_str()).collect()
    }
    pub fn all_sparse(&self) -> bool {
        self.dense.is_empty()
    }
    pub fn all_dense(&self) -> bool {
        self.sparse.is_empty()
    }

    pub fn names(&self) -> Vec<&str> {
        let mut n: Vec<&str> = self
            .sparse_names()
            .into_iter()
            .chain(self.dense_names())
            .collect();
        n.sort();
        n.dedup();
        n
    }

    /// Convert all sparse variables to dense.
    pub fn to_dense(&mut self, sampling_rate: Option<f64>) {
        let sr = sampling_rate.unwrap_or(self.sampling_rate);
        let sparse = std::mem::take(&mut self.sparse);
        for var in sparse {
            if var.amplitude.iter().all(|v| v.is_finite()) {
                self.dense.push(var.to_dense(Some(sr)));
            }
        }
        self.sampling_rate = sr;
    }

    /// Resample all dense variables.
    pub fn resample(&mut self, sampling_rate: f64) {
        self.dense = self
            .dense
            .iter()
            .map(|v| v.resample(sampling_rate))
            .collect();
        self.sampling_rate = sampling_rate;
    }

    /// Combined densify + resample pipeline.
    pub fn densify_and_resample(
        &mut self,
        sampling_rate: Option<f64>,
        force_dense: bool,
        resample_dense: bool,
    ) {
        let sr = sampling_rate.unwrap_or(self.sampling_rate);
        if force_dense {
            self.to_dense(Some(sr));
        }
        if resample_dense {
            self.resample(sr);
        }
        self.sampling_rate = sr;
    }

    /// Convert to rows with sampling rate option.
    pub fn to_rows_with_options(
        &self,
        include_sparse: bool,
        include_dense: bool,
        sampling_rate: Option<f64>,
    ) -> Vec<StringEntities> {
        let mut all_rows = Vec::new();
        if include_sparse {
            for var in &self.sparse {
                all_rows.extend(var.to_rows());
            }
        }
        if include_dense {
            let dense_vars: Vec<_> = if let Some(sr) = sampling_rate {
                self.dense.iter().map(|v| v.resample(sr)).collect()
            } else {
                self.dense.clone()
            };
            for var in &dense_vars {
                all_rows.extend(var.to_rows());
            }
        }
        all_rows
    }

    /// Resolve sampling rate from string ('TR', 'highest', or numeric).
    pub fn resolve_sampling_rate(&self, requested: Option<&str>) -> f64 {
        match requested {
            Some("TR") => {
                let trs: std::collections::HashSet<i64> = self
                    .dense
                    .iter()
                    .flat_map(|v| v.run_info.iter())
                    .map(|r| (r.tr * 1_000_000.0).round() as i64)
                    .collect();
                if trs.len() == 1 {
                    1.0 / (trs.into_iter().next().unwrap() as f64 / 1_000_000.0)
                } else {
                    self.sampling_rate
                }
            }
            Some("highest") => self
                .dense
                .iter()
                .map(|v| v.sampling_rate)
                .fold(self.sampling_rate, f64::max),
            Some(s) => s.parse().unwrap_or(self.sampling_rate),
            None => self.sampling_rate,
        }
    }
}

/// Merge multiple variable collections.
pub fn merge_collections(collections: &[VariableCollection]) -> Option<VariableCollection> {
    if collections.is_empty() {
        return None;
    }
    let all_vars: Vec<SimpleVariable> = collections
        .iter()
        .flat_map(|c| c.variables.values().cloned())
        .collect();
    Some(VariableCollection::new(all_vars))
}

// ─────────── Helpers ───────────

fn index_common_entities(variables: &HashMap<String, SimpleVariable>) -> StringEntities {
    let all_ents: Vec<&StringEntities> = variables.values().map(|v| &v.entities).collect();
    let mut common = StringEntities::new();
    if let Some(first) = all_ents.first() {
        for (k, v) in *first {
            if all_ents.iter().all(|e| e.get(k) == Some(v)) {
                common.insert(k.clone(), v.clone());
            }
        }
    }
    common
}

fn glob_match(pattern: &str, text: &str) -> bool {
    let re_str = format!(
        "^{}$",
        pattern
            .replace('.', r"\.")
            .replace('*', ".*")
            .replace('?', ".")
    );
    regex::Regex::new(&re_str).is_ok_and(|re| re.is_match(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_variable_collection() {
        let idx = vec![
            HashMap::from([("subject".into(), "01".into())]),
            HashMap::from([("subject".into(), "02".into())]),
        ];
        let v = SimpleVariable::new("age", "participants", vec!["25".into(), "30".into()], idx);
        let col = VariableCollection::new(vec![v]);
        assert_eq!(col.level, "dataset");
        assert!(col.get("age").is_some());
    }

    #[test]
    fn test_match_variables() {
        let v1 = SimpleVariable::new("age", "participants", vec![], vec![]);
        let v2 = SimpleVariable::new("sex", "participants", vec![], vec![]);
        let v3 = SimpleVariable::new("age_group", "participants", vec![], vec![]);
        let col = VariableCollection::new(vec![v1, v2, v3]);
        let matches = col.match_variables("age*", false);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_merge_collections() {
        let v1 = SimpleVariable::new(
            "age",
            "participants",
            vec!["25".into()],
            vec![HashMap::from([("subject".into(), "01".into())])],
        );
        let v2 = SimpleVariable::new(
            "age",
            "participants",
            vec!["30".into()],
            vec![HashMap::from([("subject".into(), "02".into())])],
        );
        let c1 = VariableCollection::new(vec![v1]);
        let c2 = VariableCollection::new(vec![v2]);
        let merged = merge_collections(&[c1, c2]).unwrap();
        assert_eq!(merged.get("age").unwrap().len(), 2);
    }
}
