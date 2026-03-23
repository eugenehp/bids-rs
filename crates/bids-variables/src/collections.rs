//! Variable collections for grouping and merging BIDS variables.
//!
//! Collections group multiple variables at the same level of the BIDS hierarchy,
//! providing operations like variable lookup, wildcard matching, filtering,
//! and merging across runs or subjects.

use crate::variables::{DenseRunVariable, SimpleVariable, SparseRunVariable, merge_simple};
use bids_core::entities::StringEntities;
use std::collections::HashMap;

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
#[derive(Debug, Default, Clone)]
pub struct VariableCollection {
    pub variables: HashMap<String, SimpleVariable>,
    pub level: String,
    pub entities: StringEntities,
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
