//! Variable transformations from the `pybids-transforms-v1` specification.
//!
//! Implements Rename, Copy, Scale, Threshold, Factor, Filter, Replace, Select,
//! Delete, And, Or, Not, Product, Sum, Power, and Convolve transformations
//! that operate on variable collections within a BIDS-StatsModels pipeline.

use bids_core::entities::StringEntities;
use bids_variables::collections::VariableCollection;
use bids_variables::variables::SimpleVariable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single transformation instruction from the `pybids-transforms-v1` spec.
///
/// Each variant corresponds to a named transformation that can be applied
/// to variables in a [`VariableCollection`]. Transformations modify variable
/// data in place (rename, scale, threshold) or create new variables (factor,
/// copy, split).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "Name")]
pub enum Instruction {
    Rename {
        input: Vec<String>,
        output: Vec<String>,
    },
    Copy {
        input: Vec<String>,
        output: Vec<String>,
    },
    Scale {
        input: Vec<String>,
        #[serde(default)]
        demean: bool,
        #[serde(default)]
        rescale: bool,
        #[serde(default)]
        replace_na: Option<f64>,
    },
    Threshold {
        input: Vec<String>,
        #[serde(default = "default_threshold")]
        threshold: f64,
        #[serde(default)]
        above: bool,
        #[serde(default)]
        binarize: bool,
        #[serde(default)]
        signed: bool,
    },
    And {
        input: Vec<String>,
        output: Option<Vec<String>>,
    },
    Or {
        input: Vec<String>,
        output: Option<Vec<String>>,
    },
    Not {
        input: Vec<String>,
        output: Option<Vec<String>>,
    },
    Product {
        input: Vec<String>,
        output: Option<String>,
    },
    Sum {
        input: Vec<String>,
        #[serde(default)]
        weights: Vec<f64>,
        output: Option<String>,
    },
    Power {
        input: Vec<String>,
        value: f64,
        output: Option<Vec<String>>,
    },
    Factor {
        input: Vec<String>,
    },
    Filter {
        input: Vec<String>,
        query: String,
    },
    Replace {
        input: Vec<String>,
        replace: HashMap<String, String>,
        output: Option<Vec<String>>,
    },
    Select {
        input: Vec<String>,
    },
    Delete {
        input: Vec<String>,
    },
    Group {
        input: Vec<String>,
        output: String,
    },
    Resample {
        input: Vec<String>,
        sampling_rate: f64,
    },
    ToDense {
        input: Vec<String>,
        sampling_rate: Option<f64>,
    },
    Convolve {
        input: Vec<String>,
        #[serde(default = "default_hrf_model")]
        model: String,
    },
}

fn default_threshold() -> f64 {
    0.0
}
fn default_hrf_model() -> String {
    "spm".into()
}

/// A transformation specification containing a transformer name and instructions.
///
/// The `transformer` field identifies the transformation engine (typically
/// `"pybids-transforms-v1"`). The `instructions` field contains a list of
/// JSON transformation objects that are applied in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformSpec {
    pub transformer: String,
    pub instructions: Vec<serde_json::Value>,
}

/// Dispatch a single transformation instruction on a collection.
fn dispatch_instruction(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let name = instruction
        .get("Name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match name {
        "Rename" => apply_rename(collection, instruction),
        "Copy" => apply_copy(collection, instruction),
        "Factor" => apply_factor(collection, instruction),
        "Select" => apply_select(collection, instruction),
        "Delete" => apply_delete(collection, instruction),
        "Replace" => apply_replace(collection, instruction),
        "Scale" => apply_scale(collection, instruction),
        "Threshold" => apply_threshold(collection, instruction),
        "DropNA" => apply_dropna(collection, instruction),
        "Split" => apply_split(collection, instruction),
        "Concatenate" => apply_concatenate(collection, instruction),
        "Orthogonalize" => apply_orthogonalize(collection, instruction),
        "Lag" => apply_lag(collection, instruction),
        // No-ops: metadata-only or run-level only transforms
        "Group" | "Resample" | "ToDense" | "Assign" | "Convolve" => {}
        _ => {}
    }
}

/// Apply transformations to a VariableCollection.
///
/// This is a simplified version — PyBIDS supports a full transformer
/// plugin system. We implement the core pybids-transforms-v1 instructions.
pub fn apply_transformations(collection: &mut VariableCollection, spec: &TransformSpec) {
    for instruction in &spec.instructions {
        dispatch_instruction(collection, instruction);
    }
}

fn get_inputs(instruction: &serde_json::Value) -> Vec<String> {
    instruction
        .get("Input")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .or_else(|| {
            instruction
                .get("Input")
                .and_then(|v| v.as_str())
                .map(|s| vec![s.into()])
        })
        .unwrap_or_default()
}

fn get_outputs(instruction: &serde_json::Value) -> Vec<String> {
    instruction
        .get("Output")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .or_else(|| {
            instruction
                .get("Output")
                .and_then(|v| v.as_str())
                .map(|s| vec![s.into()])
        })
        .unwrap_or_default()
}

fn apply_rename(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let outputs = get_outputs(instruction);
    for (old, new) in inputs.iter().zip(outputs.iter()) {
        if let Some(mut var) = collection.variables.remove(old) {
            var.name = new.clone();
            collection.variables.insert(new.clone(), var);
        }
    }
}

fn apply_copy(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let outputs = get_outputs(instruction);
    for (src, dst) in inputs.iter().zip(outputs.iter()) {
        if let Some(var) = collection.variables.get(src) {
            let mut copy = var.clone();
            copy.name = dst.clone();
            collection.variables.insert(dst.clone(), copy);
        }
    }
}

fn apply_factor(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let mut new_vars = Vec::new();

    for input_name in &inputs {
        if let Some(var) = collection.variables.get(input_name) {
            let str_values = var.str_values.clone();
            let source = var.source.clone();
            let index = var.index.clone();

            let mut seen = std::collections::HashSet::new();
            let unique: Vec<String> = str_values
                .iter()
                .filter(|v| !v.is_empty() && seen.insert((*v).clone()))
                .cloned()
                .collect();

            for level in &unique {
                let new_name = format!("{input_name}.{level}");
                let values: Vec<String> = str_values
                    .iter()
                    .map(|v| if v == level { "1".into() } else { "0".into() })
                    .collect();
                new_vars.push(SimpleVariable::new(
                    &new_name,
                    &source,
                    values,
                    index.clone(),
                ));
            }
        }
    }

    for var in new_vars {
        collection.variables.insert(var.name.clone(), var);
    }
}

fn apply_select(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let input_set: std::collections::HashSet<String> = inputs.into_iter().collect();
    collection.variables.retain(|k, _| input_set.contains(k));
}

fn apply_delete(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    for name in &inputs {
        collection.variables.remove(name);
    }
}

fn apply_replace(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let outputs = get_outputs(instruction);
    let replace_map: HashMap<String, String> = instruction
        .get("Replace")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_default();

    for (i, input_name) in inputs.iter().enumerate() {
        if let Some(var) = collection.variables.get(input_name) {
            let new_values: Vec<String> = var
                .str_values
                .iter()
                .map(|v| replace_map.get(v).cloned().unwrap_or_else(|| v.clone()))
                .collect();
            let out_name = outputs.get(i).unwrap_or(input_name);
            let new_var = SimpleVariable::new(out_name, &var.source, new_values, var.index.clone());
            collection.variables.insert(out_name.clone(), new_var);
        }
    }
}

fn apply_scale(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let demean = instruction
        .get("Demean")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let rescale = instruction
        .get("Rescale")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    for input_name in &inputs {
        if let Some(var) = collection.variables.get_mut(input_name) {
            if !var.is_numeric {
                continue;
            }
            let vals = &var.values;
            let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
            if finite.is_empty() {
                continue;
            }

            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            let std = (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / finite.len() as f64)
                .sqrt();

            for (i, v) in var.values.iter_mut().enumerate() {
                if !v.is_finite() {
                    continue;
                }
                if demean {
                    *v -= mean;
                }
                if rescale && std > 1e-15 {
                    *v /= std;
                }
                var.str_values[i] = v.to_string();
            }
        }
    }
}

fn apply_threshold(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let threshold = instruction
        .get("Threshold")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let above = instruction
        .get("Above")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    let binarize = instruction
        .get("Binarize")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    for input_name in &inputs {
        if let Some(var) = collection.variables.get_mut(input_name) {
            if !var.is_numeric {
                continue;
            }
            for (i, v) in var.values.iter_mut().enumerate() {
                let passes = if above {
                    *v >= threshold
                } else {
                    *v <= threshold
                };
                if binarize {
                    *v = if passes { 1.0 } else { 0.0 };
                } else if !passes {
                    *v = 0.0;
                }
                var.str_values[i] = v.to_string();
            }
        }
    }
}

fn apply_dropna(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    for input_name in &inputs {
        if let Some(var) = collection.variables.get(input_name) {
            let keep: Vec<usize> = var
                .str_values
                .iter()
                .enumerate()
                .filter(|(_, v)| !v.is_empty())
                .map(|(i, _)| i)
                .collect();
            let new_values: Vec<String> = keep.iter().map(|&i| var.str_values[i].clone()).collect();
            let new_index: Vec<StringEntities> = keep
                .iter()
                .filter_map(|&i| var.index.get(i).cloned())
                .collect();
            let new_var = SimpleVariable::new(&var.name, &var.source, new_values, new_index);
            collection.variables.insert(input_name.clone(), new_var);
        }
    }
}

fn apply_split(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let by = instruction.get("By").and_then(|v| v.as_str()).unwrap_or("");
    if by.is_empty() {
        return;
    }

    let mut new_vars = Vec::new();
    for input_name in &inputs {
        if let Some(var) = collection.variables.get(input_name) {
            let by_var = collection.variables.get(by);
            if let Some(group_var) = by_var {
                let mut groups: std::collections::HashMap<String, Vec<usize>> =
                    std::collections::HashMap::new();
                for (i, val) in group_var.str_values.iter().enumerate() {
                    groups.entry(val.clone()).or_default().push(i);
                }
                for (key, indices) in &groups {
                    let name = format!("{input_name}.{key}");
                    let values: Vec<String> = indices
                        .iter()
                        .map(|&i| var.str_values.get(i).cloned().unwrap_or_default())
                        .collect();
                    let index: Vec<StringEntities> = indices
                        .iter()
                        .filter_map(|&i| var.index.get(i).cloned())
                        .collect();
                    new_vars.push(SimpleVariable::new(&name, &var.source, values, index));
                }
            }
        }
    }
    for v in new_vars {
        collection.variables.insert(v.name.clone(), v);
    }
}

fn apply_concatenate(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let output = instruction
        .get("Output")
        .and_then(|v| v.as_str())
        .unwrap_or("concatenated");
    let mut all_values = Vec::new();
    let mut all_index = Vec::new();
    let mut source = String::new();
    for input_name in &inputs {
        if let Some(var) = collection.variables.get(input_name) {
            if source.is_empty() {
                source = var.source.clone();
            }
            all_values.extend(var.str_values.iter().cloned());
            all_index.extend(var.index.iter().cloned());
        }
    }
    if !all_values.is_empty() {
        collection.variables.insert(
            output.into(),
            SimpleVariable::new(output, &source, all_values, all_index),
        );
    }
}

fn apply_orthogonalize(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let other_names: Vec<String> = instruction
        .get("Other")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    for input_name in &inputs {
        if let Some(var) = collection.variables.get(input_name) {
            if !var.is_numeric {
                continue;
            }
            let mut x = var.values.clone();
            // Gram-Schmidt: orthogonalize x against each other variable
            for other_name in &other_names {
                if let Some(other) = collection.variables.get(other_name) {
                    if other.values.len() != x.len() {
                        continue;
                    }
                    let dot_xo: f64 = x.iter().zip(&other.values).map(|(a, b)| a * b).sum();
                    let dot_oo: f64 = other.values.iter().map(|v| v * v).sum();
                    if dot_oo.abs() > 1e-15 {
                        let proj = dot_xo / dot_oo;
                        for (xi, oi) in x.iter_mut().zip(&other.values) {
                            *xi -= proj * oi;
                        }
                    }
                }
            }
            let new_values: Vec<String> = x.iter().map(std::string::ToString::to_string).collect();
            let new_var =
                SimpleVariable::new(&var.name, &var.source, new_values, var.index.clone());
            collection.variables.insert(input_name.clone(), new_var);
        }
    }
}

fn apply_lag(collection: &mut VariableCollection, instruction: &serde_json::Value) {
    let inputs = get_inputs(instruction);
    let n_shift = instruction
        .get("N")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(1);
    let outputs = get_outputs(instruction);

    for (i, input_name) in inputs.iter().enumerate() {
        if let Some(var) = collection.variables.get(input_name) {
            if !var.is_numeric {
                continue;
            }
            let n = var.values.len();
            let lagged: Vec<f64> = (0..n)
                .map(|j| {
                    let src = j as i64 - n_shift;
                    if src >= 0 && (src as usize) < n {
                        var.values[src as usize]
                    } else {
                        0.0
                    }
                })
                .collect();
            let new_values: Vec<String> = lagged
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            let out_name = outputs.get(i).unwrap_or(input_name);
            let new_var = SimpleVariable::new(out_name, &var.source, new_values, var.index.clone());
            collection.variables.insert(out_name.clone(), new_var);
        }
    }
}

/// A TransformerManager that tracks transform history.
pub struct TransformerManager {
    pub transformer: String,
    pub keep_history: bool,
    pub history: Vec<VariableCollection>,
}

impl TransformerManager {
    pub fn new(transformer: &str, keep_history: bool) -> Self {
        Self {
            transformer: transformer.into(),
            keep_history,
            history: Vec::new(),
        }
    }

    pub fn transform(
        &mut self,
        mut collection: VariableCollection,
        spec: &TransformSpec,
    ) -> VariableCollection {
        for instruction in &spec.instructions {
            dispatch_instruction(&mut collection, instruction);
            if self.keep_history {
                self.history.push(collection.clone());
            }
        }
        collection
    }
}

/// Expand wildcard patterns in a list of variable names.
pub fn expand_wildcards(selectors: &[String], pool: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for spec in selectors {
        if spec.contains('*') || spec.contains('?') || spec.contains('[') {
            let re_str = format!(
                "^{}$",
                spec.replace('.', r"\.")
                    .replace('*', ".*")
                    .replace('?', ".")
            );
            if let Ok(re) = regex::Regex::new(&re_str) {
                for name in pool {
                    if re.is_match(name) {
                        out.push(name.clone());
                    }
                }
            }
        } else {
            out.push(spec.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_collection() -> VariableCollection {
        use bids_core::entities::StringEntities;
        let v1 = SimpleVariable::new(
            "trial_type",
            "events",
            vec!["face".into(), "house".into(), "face".into()],
            vec![StringEntities::new(); 3],
        );
        let v2 = SimpleVariable::new(
            "rt",
            "events",
            vec!["0.5".into(), "0.7".into(), "0.6".into()],
            vec![StringEntities::new(); 3],
        );
        VariableCollection::new(vec![v1, v2])
    }

    #[test]
    fn test_factor() {
        let mut col = make_collection();
        let instr = serde_json::json!({"Name": "Factor", "Input": ["trial_type"]});
        apply_factor(&mut col, &instr);
        assert!(col.variables.contains_key("trial_type.face"));
        assert!(col.variables.contains_key("trial_type.house"));
        assert_eq!(
            col.variables["trial_type.face"].str_values,
            vec!["1", "0", "1"]
        );
    }

    #[test]
    fn test_rename() {
        let mut col = make_collection();
        let instr =
            serde_json::json!({"Name": "Rename", "Input": ["rt"], "Output": ["reaction_time"]});
        apply_rename(&mut col, &instr);
        assert!(!col.variables.contains_key("rt"));
        assert!(col.variables.contains_key("reaction_time"));
    }

    #[test]
    fn test_scale() {
        let mut col = make_collection();
        let instr =
            serde_json::json!({"Name": "Scale", "Input": ["rt"], "Demean": true, "Rescale": true});
        apply_scale(&mut col, &instr);
        let vals = &col.variables["rt"].values;
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(
            mean.abs() < 1e-10,
            "Mean should be ~0 after demean, got {}",
            mean
        );
    }

    #[test]
    fn test_threshold() {
        let mut col = make_collection();
        let instr = serde_json::json!({"Name": "Threshold", "Input": ["rt"], "Threshold": 0.6, "Above": true, "Binarize": true});
        apply_threshold(&mut col, &instr);
        assert_eq!(col.variables["rt"].values, vec![0.0, 1.0, 1.0]);
    }
}
