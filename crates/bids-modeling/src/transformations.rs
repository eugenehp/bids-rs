//! Variable transformations from the `pybids-transforms-v1` specification.
//!
//! Implements the full set of transformations defined by PyBIDS'
//! `bids.modeling.transformations` module, operating on a
//! [`VariableCollection`] within a BIDS-StatsModels pipeline.
//!
//! # Coverage
//!
//! | Group | Transformations |
//! |-------|-----------------|
//! | Naming / structure | `Rename`, `Copy`, `Delete`, `Select`, `Assign`, `Group` |
//! | Munging | `Factor`, `Filter`, `Replace`, `Split`, `DropNA`, `ToDense`, `Resample` |
//! | Computation | `Convolve`, `Demean`, `Orthogonalize`, `Product`, `Scale`, `Sum`, `Threshold`, `And`, `Or`, `Not`, `Lag` |
//! | Extensions | `Concatenate`, `Power` |
//!
//! # Output naming
//!
//! As in PyBIDS, every transformation accepts the generic `Output`,
//! `OutputPrefix`, and `OutputSuffix` keys. With none of them set, a
//! transformation overwrites its input in place. `Copy`, `Rename`, `Product`,
//! `Sum`, `And`, and `Or` require an output name and report an error without
//! one.
//!
//! # Errors
//!
//! [`apply_transformations()`] returns a [`TransformError`] for every
//! instruction it could not carry out — an unknown transformation name, a
//! missing input, a malformed argument. Nothing is skipped silently: a model
//! that asks for `Convolve` and does not get it produces an error rather than
//! an unconvolved design matrix.
//!
//! # Known deviations from PyBIDS
//!
//! - `And`/`Or`/`Not` treat NaN as false. Pandas treats NaN as truthy, which
//!   silently flips the result of a conjunction over missing data.
//! - `Lag` applies an exact shift for integer shifts and linear interpolation
//!   for fractional ones; PyBIDS defaults to cubic spline interpolation
//!   (`scipy.ndimage.shift(order=3)`).
//! - `Convolve` names its derivative/dispersion outputs `<name>_derivative`
//!   and `<name>_dispersion`. PyBIDS derives the suffix by splitting on the
//!   last `_`, which mangles variable names that already contain one.

use crate::hrf::{HrfModel, compute_regressor, regressor_names};
use crate::spec::ols;
use bids_core::entities::StringEntities;
use bids_variables::collections::{VariableCollection, VariableTiming};
use bids_variables::variables::SimpleVariable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// A JSON value that may be either a single string or a list of strings.
///
/// BIDS-StatsModels accepts both forms for `Input` and `Output`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    #[must_use]
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            Self::One(s) => vec![s.clone()],
            Self::Many(v) => v.clone(),
        }
    }
}

/// A single transformation instruction from the `pybids-transforms-v1` spec.
///
/// This is the typed view of an instruction, used by
/// [`TransformSpec::parse_instructions()`] and [`TransformSpec::validate()`].
/// Execution itself is driven from the raw JSON so that the generic
/// `Output`/`OutputPrefix`/`OutputSuffix` keys apply uniformly to every
/// transformation, exactly as in PyBIDS.
///
/// Field names deserialize from the PascalCase keys used in BIDS-StatsModels
/// JSON (`Input`, `Output`, `Demean`, …).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "Name")]
pub enum Instruction {
    #[serde(rename_all = "PascalCase")]
    Rename {
        input: StringOrVec,
        output: StringOrVec,
    },
    #[serde(rename_all = "PascalCase")]
    Copy {
        input: StringOrVec,
        output: StringOrVec,
    },
    #[serde(rename_all = "PascalCase")]
    Delete { input: StringOrVec },
    #[serde(rename_all = "PascalCase")]
    Select { input: StringOrVec },
    #[serde(rename_all = "PascalCase")]
    Assign {
        input: StringOrVec,
        target: String,
        #[serde(default)]
        input_attr: Option<String>,
        #[serde(default)]
        target_attr: Option<String>,
    },
    #[serde(rename_all = "PascalCase")]
    Group { input: StringOrVec, output: String },
    #[serde(rename_all = "PascalCase")]
    Factor {
        input: StringOrVec,
        #[serde(default)]
        constraint: Option<String>,
        #[serde(default)]
        ref_level: Option<String>,
        #[serde(default)]
        sep: Option<String>,
    },
    #[serde(rename_all = "PascalCase")]
    Filter {
        input: StringOrVec,
        query: String,
        #[serde(default)]
        by: Option<StringOrVec>,
    },
    #[serde(rename_all = "PascalCase")]
    Replace {
        input: StringOrVec,
        replace: HashMap<String, serde_json::Value>,
        #[serde(default)]
        attribute: Option<String>,
    },
    #[serde(rename_all = "PascalCase")]
    Split { input: StringOrVec, by: StringOrVec },
    #[serde(rename_all = "PascalCase")]
    DropNA { input: StringOrVec },
    #[serde(rename_all = "PascalCase")]
    ToDense {
        input: StringOrVec,
        #[serde(default)]
        sampling_rate: Option<f64>,
    },
    #[serde(rename_all = "PascalCase")]
    Resample {
        input: StringOrVec,
        #[serde(default)]
        sampling_rate: Option<f64>,
    },
    #[serde(rename_all = "PascalCase")]
    Convolve {
        input: StringOrVec,
        #[serde(default = "default_hrf_model")]
        model: String,
        #[serde(default)]
        derivative: bool,
        #[serde(default)]
        dispersion: bool,
        #[serde(default)]
        fir_delays: Option<Vec<f64>>,
    },
    #[serde(rename_all = "PascalCase")]
    Demean { input: StringOrVec },
    #[serde(rename_all = "PascalCase")]
    Orthogonalize {
        input: StringOrVec,
        other: StringOrVec,
    },
    #[serde(rename_all = "PascalCase")]
    Product { input: StringOrVec, output: String },
    #[serde(rename_all = "PascalCase")]
    Scale {
        input: StringOrVec,
        #[serde(default = "default_true")]
        demean: bool,
        #[serde(default = "default_true")]
        rescale: bool,
        #[serde(default)]
        replace_na: Option<String>,
    },
    #[serde(rename_all = "PascalCase")]
    Sum {
        input: StringOrVec,
        output: String,
        #[serde(default)]
        weights: Option<Vec<f64>>,
    },
    #[serde(rename_all = "PascalCase")]
    Threshold {
        input: StringOrVec,
        #[serde(default)]
        threshold: f64,
        #[serde(default = "default_true")]
        above: bool,
        #[serde(default)]
        binarize: bool,
        #[serde(default = "default_true")]
        signed: bool,
    },
    #[serde(rename_all = "PascalCase")]
    And { input: StringOrVec, output: String },
    #[serde(rename_all = "PascalCase")]
    Or { input: StringOrVec, output: String },
    #[serde(rename_all = "PascalCase")]
    Not { input: StringOrVec },
    #[serde(rename_all = "PascalCase")]
    Lag {
        input: StringOrVec,
        #[serde(default = "default_shift")]
        shift: f64,
        #[serde(default)]
        difference: bool,
        #[serde(default)]
        mode: Option<String>,
        #[serde(default)]
        constant: Option<f64>,
    },
    /// Non-PyBIDS extension: concatenate several variables end to end.
    #[serde(rename_all = "PascalCase")]
    Concatenate { input: StringOrVec, output: String },
    /// Non-PyBIDS extension: raise a variable to a power.
    #[serde(rename_all = "PascalCase")]
    Power { input: StringOrVec, value: f64 },
}

impl Instruction {
    /// The transformation name as it appears in the `Name` key.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rename { .. } => "Rename",
            Self::Copy { .. } => "Copy",
            Self::Delete { .. } => "Delete",
            Self::Select { .. } => "Select",
            Self::Assign { .. } => "Assign",
            Self::Group { .. } => "Group",
            Self::Factor { .. } => "Factor",
            Self::Filter { .. } => "Filter",
            Self::Replace { .. } => "Replace",
            Self::Split { .. } => "Split",
            Self::DropNA { .. } => "DropNA",
            Self::ToDense { .. } => "ToDense",
            Self::Resample { .. } => "Resample",
            Self::Convolve { .. } => "Convolve",
            Self::Demean { .. } => "Demean",
            Self::Orthogonalize { .. } => "Orthogonalize",
            Self::Product { .. } => "Product",
            Self::Scale { .. } => "Scale",
            Self::Sum { .. } => "Sum",
            Self::Threshold { .. } => "Threshold",
            Self::And { .. } => "And",
            Self::Or { .. } => "Or",
            Self::Not { .. } => "Not",
            Self::Lag { .. } => "Lag",
            Self::Concatenate { .. } => "Concatenate",
            Self::Power { .. } => "Power",
        }
    }
}

/// Every transformation name this module can execute.
pub const SUPPORTED_TRANSFORMATIONS: &[&str] = &[
    "Rename",
    "Copy",
    "Delete",
    "Select",
    "Assign",
    "Group",
    "Factor",
    "Filter",
    "Replace",
    "Split",
    "DropNA",
    "ToDense",
    "Resample",
    "Convolve",
    "Demean",
    "Orthogonalize",
    "Product",
    "Scale",
    "Sum",
    "Threshold",
    "And",
    "Or",
    "Not",
    "Lag",
    "Concatenate",
    "Power",
];

fn default_hrf_model() -> String {
    "spm".into()
}
fn default_true() -> bool {
    true
}
fn default_shift() -> f64 {
    1.0
}

/// A transformation specification containing a transformer name and instructions.
///
/// The `transformer` field identifies the transformation engine (typically
/// `"pybids-transforms-v1"`). The `instructions` field contains the list of
/// JSON transformation objects that are applied in order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformSpec {
    pub transformer: String,
    pub instructions: Vec<serde_json::Value>,
}

impl TransformSpec {
    /// Parse each instruction into its typed [`Instruction`] form.
    ///
    /// Returns `Err` on the first instruction that does not parse, naming the
    /// transformation and the serde error.
    pub fn parse_instructions(&self) -> Result<Vec<Instruction>, TransformError> {
        self.instructions
            .iter()
            .map(|raw| {
                let name = instruction_name(raw).to_string();
                serde_json::from_value::<Instruction>(raw.clone()).map_err(|e| TransformError {
                    instruction: name,
                    message: e.to_string(),
                })
            })
            .collect()
    }

    /// Check that every instruction names a supported transformation and
    /// carries well-formed arguments.
    pub fn validate(&self) -> Vec<TransformError> {
        let mut errors = Vec::new();
        for raw in &self.instructions {
            let name = instruction_name(raw);
            if name.is_empty() {
                errors.push(TransformError::new("", "instruction has no 'Name' key"));
                continue;
            }
            if !SUPPORTED_TRANSFORMATIONS.contains(&name) {
                errors.push(TransformError::new(
                    name,
                    format!("unknown transformation '{name}'"),
                ));
                continue;
            }
            if let Err(e) = serde_json::from_value::<Instruction>(raw.clone()) {
                errors.push(TransformError::new(name, e.to_string()));
            }
        }
        errors
    }
}

/// A transformation that could not be applied.
///
/// Produced instead of skipping an instruction, so that a model asking for a
/// transformation the pipeline cannot perform fails loudly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformError {
    /// The transformation's `Name`, or empty when the name was missing.
    pub instruction: String,
    pub message: String,
}

impl TransformError {
    fn new(instruction: &str, message: impl Into<String>) -> Self {
        Self {
            instruction: instruction.to_string(),
            message: message.into(),
        }
    }
}

impl fmt::Display for TransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.instruction.is_empty() {
            write!(f, "{}", self.message)
        } else {
            write!(f, "{}: {}", self.instruction, self.message)
        }
    }
}

impl std::error::Error for TransformError {}

fn instruction_name(instruction: &serde_json::Value) -> &str {
    instruction
        .get("Name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
}

/// Apply transformations to a [`VariableCollection`] in order.
///
/// Returns one [`TransformError`] per instruction that could not be applied;
/// an empty vector means every instruction succeeded. Instructions after a
/// failing one still run, matching the "apply what we can, report the rest"
/// behaviour that makes a partially-specified model debuggable.
pub fn apply_transformations(
    collection: &mut VariableCollection,
    spec: &TransformSpec,
) -> Vec<TransformError> {
    let mut errors = Vec::new();
    for instruction in &spec.instructions {
        let name = instruction_name(instruction).to_string();
        if let Err(msg) = dispatch_instruction(collection, instruction) {
            errors.push(TransformError::new(&name, msg));
        }
    }
    errors
}

/// Dispatch a single transformation instruction on a collection.
fn dispatch_instruction(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let name = instruction_name(instruction);
    match name {
        "" => Err("instruction has no 'Name' key".into()),
        "Rename" => apply_rename(collection, instruction),
        "Copy" => apply_copy(collection, instruction),
        "Delete" => apply_delete(collection, instruction),
        "Select" => apply_select(collection, instruction),
        "Assign" => apply_assign(collection, instruction),
        "Group" => apply_group(collection, instruction),
        "Factor" => apply_factor(collection, instruction),
        "Filter" => apply_filter(collection, instruction),
        "Replace" => apply_replace(collection, instruction),
        "Split" => apply_split(collection, instruction),
        "DropNA" => apply_dropna(collection, instruction),
        "ToDense" => apply_to_dense(collection, instruction),
        "Resample" => apply_resample(collection, instruction),
        "Convolve" => apply_convolve(collection, instruction),
        "Demean" => apply_demean(collection, instruction),
        "Orthogonalize" => apply_orthogonalize(collection, instruction),
        "Product" => apply_product(collection, instruction),
        "Scale" => apply_scale(collection, instruction),
        "Sum" => apply_sum(collection, instruction),
        "Threshold" => apply_threshold(collection, instruction),
        "And" => apply_logical(collection, instruction, LogicalOp::And),
        "Or" => apply_logical(collection, instruction, LogicalOp::Or),
        "Not" => apply_not(collection, instruction),
        "Lag" => apply_lag(collection, instruction),
        "Concatenate" => apply_concatenate(collection, instruction),
        "Power" => apply_power(collection, instruction),
        other => Err(format!("unknown transformation '{other}'")),
    }
}

// ───────────────────────── Argument helpers ─────────────────────────

fn str_or_vec(instruction: &serde_json::Value, key: &str) -> Vec<String> {
    match instruction.get(key) {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

fn get_inputs(instruction: &serde_json::Value) -> Vec<String> {
    str_or_vec(instruction, "Input")
}

fn get_outputs(instruction: &serde_json::Value) -> Vec<String> {
    str_or_vec(instruction, "Output")
}

fn get_bool(instruction: &serde_json::Value, key: &str, default: bool) -> bool {
    instruction
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(default)
}

fn get_f64(instruction: &serde_json::Value, key: &str, default: f64) -> f64 {
    instruction
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default)
}

fn get_str<'a>(instruction: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    instruction.get(key).and_then(|v| v.as_str())
}

/// Resolve the destination names for a transformation.
///
/// Mirrors PyBIDS: `Output` supplies explicit names (one per input, or a single
/// name when the transformation collapses inputs), and `OutputPrefix` /
/// `OutputSuffix` decorate either those names or the input names. With none of
/// the three set, each input is overwritten in place.
fn resolve_outputs(instruction: &serde_json::Value, inputs: &[String]) -> Vec<String> {
    let explicit = get_outputs(instruction);
    let prefix = get_str(instruction, "OutputPrefix").unwrap_or("");
    let suffix = get_str(instruction, "OutputSuffix").unwrap_or("");

    let base: Vec<String> = if explicit.is_empty() {
        inputs.to_vec()
    } else if explicit.len() == inputs.len() {
        explicit
    } else if explicit.len() == 1 {
        // A single name for many inputs: PyBIDS prefixes each input's name.
        inputs
            .iter()
            .map(|i| format!("{}_{}", explicit[0], i))
            .collect()
    } else {
        explicit
    };

    base.into_iter()
        .map(|n| format!("{prefix}{n}{suffix}"))
        .collect()
}

/// A single destination name for transformations that collapse many inputs.
fn resolve_single_output(instruction: &serde_json::Value) -> Option<String> {
    let explicit = get_outputs(instruction);
    let prefix = get_str(instruction, "OutputPrefix").unwrap_or("");
    let suffix = get_str(instruction, "OutputSuffix").unwrap_or("");
    let base = explicit.first()?;
    Some(format!("{prefix}{base}{suffix}"))
}

fn require_var<'a>(
    collection: &'a VariableCollection,
    name: &str,
) -> Result<&'a SimpleVariable, String> {
    collection
        .variables
        .get(name)
        .ok_or_else(|| format!("no such variable '{name}'"))
}

fn require_numeric<'a>(
    collection: &'a VariableCollection,
    name: &str,
) -> Result<&'a SimpleVariable, String> {
    let var = require_var(collection, name)?;
    if var.is_numeric {
        Ok(var)
    } else {
        Err(format!("variable '{name}' is not numeric"))
    }
}

/// Values of several variables, checked for equal length.
fn aligned_values(
    collection: &VariableCollection,
    names: &[String],
) -> Result<Vec<Vec<f64>>, String> {
    let mut out = Vec::with_capacity(names.len());
    let mut len: Option<usize> = None;
    for name in names {
        let var = require_numeric(collection, name)?;
        match len {
            Some(n) if n != var.values.len() => {
                return Err(format!(
                    "variable '{}' has {} rows, expected {n}",
                    name,
                    var.values.len()
                ));
            }
            _ => len = Some(var.values.len()),
        }
        out.push(var.values.clone());
    }
    Ok(out)
}

// ───────────────────────── Numeric helpers ─────────────────────────

/// Mean over finite values, matching pandas' `skipna=True`.
fn nan_mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

/// Sample standard deviation (ddof=1) over finite values, matching
/// `pandas.Series.std()`. PyBIDS' `Scale` divides by this, so using the
/// population standard deviation here would put every scaled regressor off by
/// a factor of sqrt(n/(n-1)).
fn nan_std(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 2 {
        return f64::NAN;
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let ss: f64 = finite.iter().map(|v| (v - mean).powi(2)).sum();
    (ss / (finite.len() - 1) as f64).sqrt()
}

fn gcd_u64(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd_u64(b, a % b) }
}

/// Greatest common divisor of a set of values at resolution `res`.
///
/// Mirrors PyBIDS' `_fractional_gcd`: values are rounded to multiples of `res`
/// and reduced with an integer GCD. Zeros are ignored, so zero onsets and
/// impulse (zero-duration) events do not collapse the result.
fn fractional_gcd(values: &[f64], res: f64) -> f64 {
    let g = values
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v / res).round().abs() as u64)
        .fold(0u64, gcd_u64);
    g as f64 * res
}

// ───────────────────── Naming and structure ─────────────────────

fn apply_rename(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    if get_outputs(instruction).is_empty()
        && get_str(instruction, "OutputPrefix").is_none()
        && get_str(instruction, "OutputSuffix").is_none()
    {
        return Err("Rename requires 'Output', 'OutputPrefix', or 'OutputSuffix'".into());
    }
    let outputs = resolve_outputs(instruction, &inputs);
    for (old, new) in inputs.iter().zip(outputs.iter()) {
        require_var(collection, old)?;
        collection.rename_var(old, new);
    }
    Ok(())
}

fn apply_copy(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    if get_outputs(instruction).is_empty()
        && get_str(instruction, "OutputPrefix").is_none()
        && get_str(instruction, "OutputSuffix").is_none()
    {
        return Err("Copy requires 'Output', 'OutputPrefix', or 'OutputSuffix'".into());
    }
    let outputs = resolve_outputs(instruction, &inputs);
    for (src, dst) in inputs.iter().zip(outputs.iter()) {
        require_var(collection, src)?;
        collection.copy_var(src, dst);
    }
    Ok(())
}

fn apply_delete(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    for name in &get_inputs(instruction) {
        collection.remove_var(name);
    }
    Ok(())
}

fn apply_select(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let keep: std::collections::HashSet<String> = get_inputs(instruction).into_iter().collect();
    let drop: Vec<String> = collection
        .variables
        .keys()
        .filter(|k| !keep.contains(*k))
        .cloned()
        .collect();
    for name in &drop {
        collection.remove_var(name);
    }
    Ok(())
}

/// Assign one variable's amplitude, duration, or onset to another.
///
/// Both variables must be sparse (dense variables have no onsets or
/// durations) and must hold the same number of events. As in PyBIDS the
/// result is a modified copy of the *target*, written back under the target's
/// name unless an explicit output is given.
fn apply_assign(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    const VALID: [&str; 3] = ["amplitude", "duration", "onset"];

    let inputs = get_inputs(instruction);
    let input = inputs
        .first()
        .ok_or_else(|| "Assign requires an 'Input'".to_string())?;
    let target =
        get_str(instruction, "Target").ok_or_else(|| "Assign requires a 'Target'".to_string())?;
    let input_attr = get_str(instruction, "InputAttr").unwrap_or("amplitude");
    let target_attr = get_str(instruction, "TargetAttr").unwrap_or("amplitude");

    if !VALID.contains(&input_attr) {
        return Err(format!(
            "invalid 'InputAttr' '{input_attr}', expected one of {VALID:?}"
        ));
    }
    if !VALID.contains(&target_attr) {
        return Err(format!(
            "invalid 'TargetAttr' '{target_attr}', expected one of {VALID:?}"
        ));
    }

    let in_timing = collection
        .timing
        .get(input)
        .ok_or_else(|| {
            format!("Assign only applies to sparse variables; input '{input}' is dense or absent")
        })?
        .clone();
    let tgt_timing = collection
        .timing
        .get(target)
        .ok_or_else(|| {
            format!("Assign only applies to sparse variables; target '{target}' is dense or absent")
        })?
        .clone();

    let in_var = require_var(collection, input)?;
    let l_s = in_var.len();
    let l_t = collection.n_rows(target);
    if l_s != l_t {
        return Err(format!(
            "input and target do not contain the same number of events ({l_s} vs. {l_t})"
        ));
    }

    let vals: Vec<f64> = match input_attr {
        "amplitude" => in_var.values.clone(),
        "duration" => in_timing.duration.clone(),
        _ => in_timing.onset.clone(),
    };

    let tgt_var = require_var(collection, target)?;
    let mut new_var = tgt_var.clone();
    let mut new_timing = tgt_timing;
    match target_attr {
        "amplitude" => {
            new_var.str_values = vals.iter().map(std::string::ToString::to_string).collect();
            new_var.values = vals;
            new_var.is_numeric = true;
        }
        "duration" => new_timing.duration = vals,
        _ => new_timing.onset = vals,
    }

    let out = resolve_single_output(instruction).unwrap_or_else(|| target.to_string());
    new_var.name = out;
    collection.insert_sparse(new_var, new_timing);
    Ok(())
}

/// Record a named group of variables on the collection.
///
/// PyBIDS' `Group` takes its group name in a `name` argument, which collides
/// with the `Name` key that selects the transformation itself; this
/// implementation reads the group name from `Output` instead.
fn apply_group(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let name = get_str(instruction, "Output")
        .or_else(|| get_str(instruction, "GroupName"))
        .ok_or_else(|| "Group requires 'Output' naming the group".to_string())?;
    if collection.variables.contains_key(name) {
        return Err(format!(
            "group name '{name}' conflicts with an existing variable name"
        ));
    }
    for input in &inputs {
        require_var(collection, input)?;
    }
    collection.groups.insert(name.to_string(), inputs);
    Ok(())
}

// ───────────────────────────── Munging ─────────────────────────────

/// Dummy-code a categorical variable, one new variable per level.
///
/// Levels are sorted lexicographically, as in PyBIDS. `Constraint` may be
/// `"none"` (default), `"drop_one"`, or `"mean_zero"`; the reference level
/// defaults to the first sorted level. The original variable is removed.
fn apply_factor(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let constraint = get_str(instruction, "Constraint").unwrap_or("none");
    if !["none", "drop_one", "mean_zero"].contains(&constraint) {
        return Err(format!(
            "invalid 'Constraint' '{constraint}', expected 'none', 'drop_one', or 'mean_zero'"
        ));
    }
    let sep = get_str(instruction, "Sep").unwrap_or(".").to_string();
    let explicit_ref = get_str(instruction, "RefLevel").map(String::from);

    for input_name in &inputs {
        let var = require_var(collection, input_name)?.clone();
        let mut levels: Vec<String> = var
            .str_values
            .iter()
            .filter(|v| !v.is_empty())
            .cloned()
            .collect();
        levels.sort();
        levels.dedup();
        if levels.is_empty() {
            return Err(format!("variable '{input_name}' has no levels to factor"));
        }

        let drop = levels.len() > 1 && matches!(constraint, "drop_one" | "mean_zero");
        let ref_level = if drop {
            Some(explicit_ref.clone().unwrap_or_else(|| levels[0].clone()))
        } else {
            explicit_ref.clone().filter(|_| false)
        };
        let mean_zero_fill = -1.0 / (levels.len() - 1).max(1) as f64;

        let timing = collection.timing.get(input_name).cloned();
        let was_dense = collection.dense.contains(input_name);

        for level in &levels {
            if ref_level.as_ref() == Some(level) {
                continue;
            }
            let new_name = format!("{input_name}{sep}{level}");
            let values: Vec<f64> = var
                .str_values
                .iter()
                .map(|v| {
                    if v == level {
                        1.0
                    } else if constraint == "mean_zero" && Some(v) == ref_level.as_ref() {
                        mean_zero_fill
                    } else {
                        0.0
                    }
                })
                .collect();
            let str_values: Vec<String> = values
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            let mut new_var =
                SimpleVariable::new(&new_name, &var.source, str_values, var.index.clone());
            new_var.values = values;
            new_var.is_numeric = true;
            match (&timing, was_dense) {
                (Some(t), _) => collection.insert_sparse(new_var, t.clone()),
                (None, true) => collection.insert_dense(new_var),
                (None, false) => collection.insert_var(new_var),
            }
        }
        collection.remove_var(input_name);
    }
    Ok(())
}

/// Keep only the rows of a variable that satisfy a query.
///
/// The query is a conjunction/disjunction of comparisons — for example
/// `"trial_type == 'face'"` or `"RT > 0.5 and accuracy == 1"`. Operands may
/// name other variables in the collection, entities in the variable's index,
/// numbers, or quoted strings. Parentheses are not supported.
fn apply_filter(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let query =
        get_str(instruction, "Query").ok_or_else(|| "Filter requires a 'Query'".to_string())?;
    let expr = parse_query(query)?;

    for input_name in &inputs {
        let n = collection.n_rows(input_name);
        if n == 0 {
            return Err(format!("no such variable '{input_name}'"));
        }
        let mut keep = Vec::new();
        for i in 0..n {
            if eval_query(&expr, collection, input_name, i)? {
                keep.push(i);
            }
        }
        collection.select_rows(input_name, &keep);
        if let Some(out) = resolve_single_output(instruction) {
            if &out != input_name {
                collection.rename_var(input_name, &out);
            }
        }
    }
    Ok(())
}

fn apply_replace(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let outputs = resolve_outputs(instruction, &inputs);
    let replace_map: HashMap<String, serde_json::Value> = instruction
        .get("Replace")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .ok_or_else(|| "Replace requires a 'Replace' mapping".to_string())?;

    let as_text = |v: &serde_json::Value| match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    for (i, input_name) in inputs.iter().enumerate() {
        let var = require_var(collection, input_name)?.clone();
        let new_values: Vec<String> = var
            .str_values
            .iter()
            .map(|v| replace_map.get(v).map_or_else(|| v.clone(), &as_text))
            .collect();
        let out_name = outputs.get(i).unwrap_or(input_name).clone();
        let new_var = SimpleVariable::new(&out_name, &var.source, new_values, var.index.clone());
        match (
            collection.timing.get(input_name).cloned(),
            collection.dense.contains(input_name),
        ) {
            (Some(t), _) => collection.insert_sparse(new_var, t),
            (None, true) => collection.insert_dense(new_var),
            (None, false) => collection.insert_var(new_var),
        }
    }
    Ok(())
}

/// Split a variable into one variable per level of the `By` variables.
fn apply_split(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let by = {
        let mut b = str_or_vec(instruction, "By");
        if b.is_empty() {
            if let Some(s) = get_str(instruction, "By") {
                b = vec![s.to_string()];
            }
        }
        b
    };
    if by.is_empty() {
        return Err("Split requires a 'By'".into());
    }

    for input_name in &inputs {
        let var = require_var(collection, input_name)?.clone();
        let timing = collection.timing.get(input_name).cloned();
        let was_dense = collection.dense.contains(input_name);

        // Group rows by the joined level of every `By` key. A key may name a
        // variable in the collection or an entity in the variable's index.
        let mut groups: std::collections::BTreeMap<String, Vec<usize>> =
            std::collections::BTreeMap::new();
        for i in 0..var.len() {
            let mut parts = Vec::with_capacity(by.len());
            for key in &by {
                let val = if let Some(other) = collection.variables.get(key) {
                    other.str_values.get(i).cloned().unwrap_or_default()
                } else {
                    var.index
                        .get(i)
                        .and_then(|e| e.get(key.as_str()))
                        .cloned()
                        .unwrap_or_default()
                };
                parts.push(val);
            }
            groups.entry(parts.join(".")).or_default().push(i);
        }

        for (key, indices) in &groups {
            let name = format!("{input_name}.{key}");
            let str_values: Vec<String> = indices
                .iter()
                .filter_map(|&i| var.str_values.get(i).cloned())
                .collect();
            let index: Vec<StringEntities> = indices
                .iter()
                .filter_map(|&i| var.index.get(i).cloned())
                .collect();
            let new_var = SimpleVariable::new(&name, &var.source, str_values, index);
            match (&timing, was_dense) {
                (Some(t), _) => {
                    let trimmed = VariableTiming {
                        onset: indices
                            .iter()
                            .filter_map(|&i| t.onset.get(i).copied())
                            .collect(),
                        duration: indices
                            .iter()
                            .filter_map(|&i| t.duration.get(i).copied())
                            .collect(),
                    };
                    collection.insert_sparse(new_var, trimmed);
                }
                (None, true) => collection.insert_dense(new_var),
                (None, false) => collection.insert_var(new_var),
            }
        }
    }
    Ok(())
}

fn apply_dropna(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    for input_name in &get_inputs(instruction) {
        let var = require_var(collection, input_name)?;
        let keep: Vec<usize> = var
            .str_values
            .iter()
            .enumerate()
            .filter(|(i, s)| !s.is_empty() && var.values.get(*i).is_none_or(|v| !v.is_nan()))
            .map(|(i, _)| i)
            .collect();
        collection.select_rows(input_name, &keep);
    }
    Ok(())
}

fn apply_to_dense(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let sr = get_f64(instruction, "SamplingRate", 10.0);
    if sr <= 0.0 {
        return Err(format!("'SamplingRate' must be positive, got {sr}"));
    }
    for input_name in &get_inputs(instruction) {
        require_var(collection, input_name)?;
        if collection.is_dense(input_name) {
            continue;
        }
        if !collection.is_sparse(input_name) {
            return Err(format!(
                "variable '{input_name}' has no event timing, so it cannot be made dense; \
                 load the collection with VariableCollection::from_run_collection"
            ));
        }
        collection.to_dense_var(input_name, sr);
    }
    Ok(())
}

fn apply_resample(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let sr = get_f64(instruction, "SamplingRate", 10.0);
    if sr <= 0.0 {
        return Err(format!("'SamplingRate' must be positive, got {sr}"));
    }
    for input_name in &get_inputs(instruction) {
        require_var(collection, input_name)?;
        if !collection.is_dense(input_name) {
            return Err(format!(
                "Resample applies to dense variables; '{input_name}' is sparse (apply ToDense first)"
            ));
        }
        collection.resample_var(input_name, sr);
    }
    Ok(())
}

// ─────────────────────────── Computation ───────────────────────────

/// Convolve one or more variables with a haemodynamic response function.
///
/// Sparse (event) variables are sampled onto a regular grid spanning the run
/// before convolution; dense variables are convolved on their existing grid.
/// The oversampling factor is derived from the greatest common divisor of the
/// event onsets and durations, bounded to 1–100 Hz, matching PyBIDS.
///
/// Requesting a derivative or dispersion basis produces additional variables
/// named `<name>_derivative` and `<name>_dispersion`.
fn apply_convolve(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    const MAX_FREQ: f64 = 100.0;
    const MIN_FREQ: f64 = 1.0;
    const MIN_INTERVAL: f64 = 0.01;

    let inputs = get_inputs(instruction);
    if inputs.is_empty() {
        return Err("Convolve requires an 'Input'".into());
    }
    let mut model = get_str(instruction, "Model")
        .unwrap_or("spm")
        .to_lowercase();
    let derivative = get_bool(instruction, "Derivative", false);
    let dispersion = get_bool(instruction, "Dispersion", false);
    let fir_delays: Option<Vec<f64>> = instruction
        .get("FirDelays")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(serde_json::Value::as_f64).collect());

    if model == "spm" || model == "glover" {
        if derivative {
            model.push_str(" + derivative");
        }
        if dispersion {
            model.push_str(" + dispersion");
        }
    } else if model != "fir" {
        return Err(format!(
            "'Model' must be one of 'spm', 'glover', or 'fir', got '{model}'"
        ));
    }
    let hrf_model = HrfModel::parse(&model);
    if matches!(hrf_model, HrfModel::None) {
        return Err(format!("unsupported HRF model '{model}'"));
    }

    let outputs = resolve_outputs(instruction, &inputs);

    for (i, input_name) in inputs.iter().enumerate() {
        let var = require_numeric(collection, input_name)?.clone();
        let out_base = outputs.get(i).unwrap_or(input_name).clone();

        let (onsets, durations, amplitudes, frame_times, sampling_rate, safety) =
            if let Some(t) = collection.timing.get(input_name).cloned() {
                let sr = collection.sampling_rate;
                let dur = collection.run_duration();
                if dur <= 0.0 {
                    return Err(format!(
                        "cannot convolve '{input_name}': the collection has no run duration"
                    ));
                }
                let n = (dur * sr).ceil() as usize;
                if n < 2 {
                    return Err(format!(
                        "cannot convolve '{input_name}': run is too short at {sr} Hz"
                    ));
                }
                // linspace(0, dur, n, endpoint=False)
                let step = dur / n as f64;
                let frames: Vec<f64> = (0..n).map(|k| k as f64 * step).collect();
                (t.onset, t.duration, var.values.clone(), frames, sr, 2.0)
            } else if collection.is_dense(input_name) {
                let sr = collection.sampling_rate;
                let n = var.values.len();
                if n < 2 {
                    return Err(format!(
                        "cannot convolve '{input_name}': fewer than 2 samples"
                    ));
                }
                let frames: Vec<f64> = (0..n).map(|k| k as f64 / sr).collect();
                let step = 1.0 / sr;
                (
                    frames.clone(),
                    vec![step; n],
                    var.values.clone(),
                    frames,
                    sr,
                    1.0,
                )
            } else {
                return Err(format!(
                    "cannot convolve '{input_name}': it has neither event timing nor a sampling \
                     rate; load the collection with VariableCollection::from_run_collection"
                ));
            };

        let mut grid: Vec<f64> = onsets.clone();
        grid.extend_from_slice(&durations);
        let resolution = fractional_gcd(&grid, MIN_INTERVAL);
        let resolution = if resolution > 0.0 {
            resolution
        } else {
            MIN_INTERVAL
        };
        let effective_sr = (safety / resolution).clamp(MIN_FREQ, MAX_FREQ);
        let oversampling = (effective_sr / sampling_rate).ceil().max(1.0) as usize;

        let regressors = compute_regressor(
            &onsets,
            &durations,
            &amplitudes,
            &hrf_model,
            &frame_times,
            oversampling,
            fir_delays.as_deref(),
        );
        let names = regressor_names(&out_base, &hrf_model, fir_delays.as_deref());

        collection.remove_var(input_name);
        for (values, name) in regressors.into_iter().zip(names) {
            let str_values: Vec<String> = values
                .iter()
                .map(std::string::ToString::to_string)
                .collect();
            let index: Vec<StringEntities> = frame_times
                .iter()
                .map(|t| {
                    let mut e = collection.entities.clone();
                    e.insert("onset".into(), t.to_string());
                    e
                })
                .collect();
            let mut new_var = SimpleVariable::new(&name, &var.source, str_values, index);
            new_var.values = values;
            new_var.is_numeric = true;
            collection.insert_dense(new_var);
        }
        collection.sampling_rate = sampling_rate;
    }
    Ok(())
}

fn apply_demean(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let outputs = resolve_outputs(instruction, &inputs);
    for (i, input_name) in inputs.iter().enumerate() {
        let var = require_numeric(collection, input_name)?;
        let mean = nan_mean(&var.values);
        let values: Vec<f64> = var.values.iter().map(|v| v - mean).collect();
        let out = outputs.get(i).unwrap_or(input_name).clone();
        collection.derive_var(input_name, &out, values);
    }
    Ok(())
}

/// Residualize each input against one or more other variables.
///
/// Matches PyBIDS: the input is regressed on an intercept plus the `Other`
/// variables, and the fitted contribution of the `Other` variables alone is
/// subtracted. Omitting the intercept — as a plain Gram-Schmidt projection
/// does — leaves a bias whenever the regressors are not mean-centred.
fn apply_orthogonalize(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let others = str_or_vec(instruction, "Other");
    if others.is_empty() {
        return Err("Orthogonalize requires 'Other'".into());
    }
    let outputs = resolve_outputs(instruction, &inputs);

    for (i, input_name) in inputs.iter().enumerate() {
        let y = require_numeric(collection, input_name)?.values.clone();
        let x = aligned_values(collection, &others)?;
        if x.iter().any(|col| col.len() != y.len()) {
            return Err(format!(
                "'Other' variables do not align with '{input_name}' ({} rows)",
                y.len()
            ));
        }
        let mut design = vec![vec![1.0; y.len()]];
        design.extend(x.iter().cloned());
        let coefs = ols(&design, &y)
            .ok_or_else(|| format!("orthogonalization of '{input_name}' is rank-deficient"))?;

        let values: Vec<f64> = (0..y.len())
            .map(|r| {
                let fitted: f64 = x
                    .iter()
                    .zip(&coefs[1..])
                    .map(|(col, beta)| col[r] * beta)
                    .sum();
                y[r] - fitted
            })
            .collect();
        let out = outputs.get(i).unwrap_or(input_name).clone();
        collection.derive_var(input_name, &out, values);
    }
    Ok(())
}

fn apply_product(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let out = resolve_single_output(instruction)
        .ok_or_else(|| "Product requires an 'Output'".to_string())?;
    let cols = aligned_values(collection, &inputs)?;
    let n = cols.first().map_or(0, Vec::len);
    let values: Vec<f64> = (0..n)
        .map(|r| {
            cols.iter()
                .map(|c| c[r])
                .filter(|v| v.is_finite())
                .product::<f64>()
        })
        .collect();
    let first = inputs
        .first()
        .ok_or_else(|| "Product requires an 'Input'".to_string())?;
    collection.derive_var(first, &out, values);
    Ok(())
}

/// Centre and/or standardise variables.
///
/// Defaults to `Demean: true, Rescale: true` — i.e. a z-score — as in PyBIDS.
/// The standard deviation is the sample (ddof=1) standard deviation that
/// `pandas.Series.std()` computes.
fn apply_scale(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let demean = get_bool(instruction, "Demean", true);
    let rescale = get_bool(instruction, "Rescale", true);
    let replace_na = get_str(instruction, "ReplaceNA").map(str::to_lowercase);
    if let Some(mode) = replace_na.as_deref() {
        if mode != "before" && mode != "after" {
            return Err(format!(
                "invalid 'ReplaceNA' '{mode}', expected 'before' or 'after'"
            ));
        }
    }
    let outputs = resolve_outputs(instruction, &inputs);

    for (i, input_name) in inputs.iter().enumerate() {
        let var = require_numeric(collection, input_name)?;
        let mut values = var.values.clone();

        let distinct: std::collections::HashSet<u64> = values
            .iter()
            .filter(|v| v.is_finite())
            .map(|v| v.to_bits())
            .collect();
        if distinct.len() == 1 && replace_na.as_deref() != Some("after") {
            let val = values
                .iter()
                .find(|v| v.is_finite())
                .copied()
                .unwrap_or(0.0);
            return Err(format!(
                "cannot scale '{input_name}': it has the constant value {val}. To get a column \
                 of zeros back, set 'ReplaceNA' to 'after'."
            ));
        }

        if replace_na.as_deref() == Some("before") {
            for v in &mut values {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
        }
        if demean {
            let mean = nan_mean(&values);
            for v in &mut values {
                *v -= mean;
            }
        }
        if rescale {
            let std = nan_std(&values);
            for v in &mut values {
                *v /= std;
            }
        }
        if replace_na.as_deref() == Some("after") {
            for v in &mut values {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
        }
        let out = outputs.get(i).unwrap_or(input_name).clone();
        collection.derive_var(input_name, &out, values);
    }
    Ok(())
}

fn apply_sum(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let out =
        resolve_single_output(instruction).ok_or_else(|| "Sum requires an 'Output'".to_string())?;
    let cols = aligned_values(collection, &inputs)?;
    let weights: Vec<f64> = match instruction.get("Weights").and_then(|v| v.as_array()) {
        Some(arr) => {
            let w: Vec<f64> = arr.iter().filter_map(serde_json::Value::as_f64).collect();
            if w.len() != cols.len() {
                return Err(format!(
                    "'Weights' has {} entries but {} variables are being summed",
                    w.len(),
                    cols.len()
                ));
            }
            w
        }
        None => vec![1.0; cols.len()],
    };
    let n = cols.first().map_or(0, Vec::len);
    let values: Vec<f64> = (0..n)
        .map(|r| {
            cols.iter()
                .zip(&weights)
                .map(|(c, w)| c[r] * w)
                .filter(|v| v.is_finite())
                .sum()
        })
        .collect();
    let first = inputs
        .first()
        .ok_or_else(|| "Sum requires an 'Input'".to_string())?;
    collection.derive_var(first, &out, values);
    Ok(())
}

fn apply_threshold(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let mut threshold = get_f64(instruction, "Threshold", 0.0);
    let above = get_bool(instruction, "Above", true);
    let binarize = get_bool(instruction, "Binarize", false);
    let signed = get_bool(instruction, "Signed", true);
    let outputs = resolve_outputs(instruction, &inputs);

    if !signed {
        threshold = threshold.abs();
    }

    for (i, input_name) in inputs.iter().enumerate() {
        let var = require_numeric(collection, input_name)?;
        // Unsigned thresholding compares — and returns — absolute values.
        let mut values: Vec<f64> = if signed {
            var.values.clone()
        } else {
            var.values.iter().map(|v| v.abs()).collect()
        };
        for v in &mut values {
            let keep = if above {
                *v >= threshold
            } else {
                *v <= threshold
            };
            if !keep {
                *v = 0.0;
            } else if binarize {
                *v = 1.0;
            }
        }
        let out = outputs.get(i).unwrap_or(input_name).clone();
        collection.derive_var(input_name, &out, values);
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum LogicalOp {
    And,
    Or,
}

/// Elementwise conjunction or disjunction across inputs.
///
/// NaN counts as false; see the module-level note on deviations.
fn apply_logical(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
    op: LogicalOp,
) -> Result<(), String> {
    let label = match op {
        LogicalOp::And => "And",
        LogicalOp::Or => "Or",
    };
    let inputs = get_inputs(instruction);
    let out = resolve_single_output(instruction)
        .ok_or_else(|| format!("{label} requires an 'Output'"))?;
    let cols = aligned_values(collection, &inputs)?;
    let n = cols.first().map_or(0, Vec::len);
    let truthy = |v: f64| v.is_finite() && v != 0.0;
    let values: Vec<f64> = (0..n)
        .map(|r| {
            let hit = match op {
                LogicalOp::And => cols.iter().all(|c| truthy(c[r])),
                LogicalOp::Or => cols.iter().any(|c| truthy(c[r])),
            };
            f64::from(u8::from(hit))
        })
        .collect();
    let first = inputs
        .first()
        .ok_or_else(|| format!("{label} requires an 'Input'"))?;
    collection.derive_var(first, &out, values);
    Ok(())
}

fn apply_not(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let outputs = resolve_outputs(instruction, &inputs);
    for (i, input_name) in inputs.iter().enumerate() {
        let var = require_numeric(collection, input_name)?;
        let values: Vec<f64> = var
            .values
            .iter()
            .map(|v| f64::from(u8::from(!(v.is_finite() && *v != 0.0))))
            .collect();
        let out = outputs.get(i).unwrap_or(input_name).clone();
        collection.derive_var(input_name, &out, values);
    }
    Ok(())
}

/// Shift a variable in time, optionally returning the difference.
///
/// `Shift` is the number of positions to move values towards the end of the
/// series (negative moves them towards the beginning). Out-of-range positions
/// are filled according to `Mode`: `"nearest"` (default), `"reflect"`, or
/// `"constant"` with `Constant`.
fn apply_lag(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let shift = instruction
        .get("Shift")
        .or_else(|| instruction.get("N"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1.0);
    let difference = get_bool(instruction, "Difference", false);
    let mode = get_str(instruction, "Mode")
        .unwrap_or("nearest")
        .to_lowercase();
    let constant = get_f64(instruction, "Constant", 0.0);
    if !["nearest", "reflect", "constant"].contains(&mode.as_str()) {
        return Err(format!(
            "invalid 'Mode' '{mode}', expected 'nearest', 'reflect', or 'constant'"
        ));
    }
    let outputs = resolve_outputs(instruction, &inputs);

    for (i, input_name) in inputs.iter().enumerate() {
        let var = require_numeric(collection, input_name)?;
        let src = &var.values;
        let n = src.len();
        let sample = |pos: f64| -> f64 {
            if pos.fract() == 0.0 {
                boundary_value(src, pos as i64, &mode, constant)
            } else {
                let lo = pos.floor();
                let hi = pos.ceil();
                let w = pos - lo;
                let a = boundary_value(src, lo as i64, &mode, constant);
                let b = boundary_value(src, hi as i64, &mode, constant);
                a * (1.0 - w) + b * w
            }
        };
        let shifted: Vec<f64> = (0..n).map(|j| sample(j as f64 - shift)).collect();
        let values: Vec<f64> = if difference {
            if shift >= 0.0 {
                src.iter().zip(&shifted).map(|(a, b)| a - b).collect()
            } else {
                shifted.iter().zip(src).map(|(a, b)| a - b).collect()
            }
        } else {
            shifted
        };
        let out = outputs.get(i).unwrap_or(input_name).clone();
        collection.derive_var(input_name, &out, values);
    }
    Ok(())
}

/// Read `values[idx]`, extending the series past its edges per `mode`.
fn boundary_value(values: &[f64], idx: i64, mode: &str, constant: f64) -> f64 {
    let n = values.len() as i64;
    if n == 0 {
        return constant;
    }
    if idx >= 0 && idx < n {
        return values[idx as usize];
    }
    match mode {
        "constant" => constant,
        "reflect" => {
            // (d c b a | a b c d | d c b a): reflect about the outer edge,
            // repeating for indices further out than one period.
            let period = 2 * n;
            let mut m = idx.rem_euclid(period);
            if m >= n {
                m = period - m - 1;
            }
            values[m as usize]
        }
        // "nearest": replicate the boundary value.
        _ => values[idx.clamp(0, n - 1) as usize],
    }
}

fn apply_power(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let exponent = instruction
        .get("Value")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| "Power requires a numeric 'Value'".to_string())?;
    let outputs = resolve_outputs(instruction, &inputs);
    for (i, input_name) in inputs.iter().enumerate() {
        let var = require_numeric(collection, input_name)?;
        let values: Vec<f64> = var.values.iter().map(|v| v.powf(exponent)).collect();
        let out = outputs.get(i).unwrap_or(input_name).clone();
        collection.derive_var(input_name, &out, values);
    }
    Ok(())
}

fn apply_concatenate(
    collection: &mut VariableCollection,
    instruction: &serde_json::Value,
) -> Result<(), String> {
    let inputs = get_inputs(instruction);
    let output = resolve_single_output(instruction).unwrap_or_else(|| "concatenated".into());
    let mut all_values = Vec::new();
    let mut all_index = Vec::new();
    let mut source = String::new();
    for input_name in &inputs {
        let var = require_var(collection, input_name)?;
        if source.is_empty() {
            source = var.source.clone();
        }
        all_values.extend(var.str_values.iter().cloned());
        all_index.extend(var.index.iter().cloned());
    }
    if all_values.is_empty() {
        return Err("Concatenate produced no values".into());
    }
    collection.insert_var(SimpleVariable::new(&output, &source, all_values, all_index));
    Ok(())
}

// ─────────────────────────── Query parsing ───────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cmp {
    Eq,
    Ne,
    Ge,
    Le,
    Gt,
    Lt,
}

#[derive(Debug, Clone)]
enum Operand {
    Number(f64),
    Text(String),
    /// A variable in the collection, or an entity in the target's index.
    Name(String),
}

#[derive(Debug, Clone)]
struct Predicate {
    lhs: Operand,
    op: Cmp,
    rhs: Operand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Join {
    And,
    Or,
}

/// A flat conjunction/disjunction of comparisons. Parentheses are not
/// supported; joins are evaluated left to right.
#[derive(Debug, Clone)]
struct Query {
    first: Predicate,
    rest: Vec<(Join, Predicate)>,
}

fn parse_query(query: &str) -> Result<Query, String> {
    let (clauses, joins) = split_joins(query);
    let mut parsed = clauses
        .iter()
        .map(|c| parse_predicate(c))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter();
    let first = parsed
        .next()
        .ok_or_else(|| format!("empty query '{query}'"))?;
    Ok(Query {
        first,
        rest: joins.into_iter().zip(parsed).collect(),
    })
}

/// Split a query on top-level `and`/`or` (also `&&`/`||`, `&`/`|`),
/// leaving quoted strings intact.
fn split_joins(query: &str) -> (Vec<String>, Vec<Join>) {
    let bytes: Vec<char> = query.chars().collect();
    let mut clauses = Vec::new();
    let mut joins = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = quote {
            current.push(c);
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        if c == '\'' || c == '"' {
            quote = Some(c);
            current.push(c);
            i += 1;
            continue;
        }

        let rest: String = bytes[i..].iter().collect();
        let lower = rest.to_lowercase();
        let boundary = current.chars().last().is_none_or(char::is_whitespace);
        let matched = if boundary && lower.starts_with("and ") {
            Some((Join::And, 4))
        } else if boundary && lower.starts_with("or ") {
            Some((Join::Or, 3))
        } else if rest.starts_with("&&") {
            Some((Join::And, 2))
        } else if rest.starts_with("||") {
            Some((Join::Or, 2))
        } else if c == '&' {
            Some((Join::And, 1))
        } else if c == '|' {
            Some((Join::Or, 1))
        } else {
            None
        };

        if let Some((join, width)) = matched {
            clauses.push(current.trim().to_string());
            joins.push(join);
            current = String::new();
            i += width;
            continue;
        }
        current.push(c);
        i += 1;
    }
    clauses.push(current.trim().to_string());
    (clauses, joins)
}

fn parse_predicate(clause: &str) -> Result<Predicate, String> {
    const OPS: [(&str, Cmp); 6] = [
        ("==", Cmp::Eq),
        ("!=", Cmp::Ne),
        (">=", Cmp::Ge),
        ("<=", Cmp::Le),
        (">", Cmp::Gt),
        ("<", Cmp::Lt),
    ];
    for (token, op) in OPS {
        if let Some(pos) = clause.find(token) {
            let lhs = clause[..pos].trim();
            let rhs = clause[pos + token.len()..].trim();
            if lhs.is_empty() || rhs.is_empty() {
                return Err(format!("malformed comparison '{clause}'"));
            }
            return Ok(Predicate {
                lhs: parse_operand(lhs),
                op,
                rhs: parse_operand(rhs),
            });
        }
    }
    Err(format!(
        "no comparison operator in '{clause}' (expected one of ==, !=, >=, <=, >, <)"
    ))
}

fn parse_operand(token: &str) -> Operand {
    let t = token.trim();
    if (t.starts_with('\'') && t.ends_with('\'') && t.len() >= 2)
        || (t.starts_with('"') && t.ends_with('"') && t.len() >= 2)
    {
        return Operand::Text(t[1..t.len() - 1].to_string());
    }
    t.parse::<f64>()
        .map_or_else(|_| Operand::Name(t.to_string()), Operand::Number)
}

/// Resolve an operand for row `row` of `target`.
fn operand_value(
    operand: &Operand,
    collection: &VariableCollection,
    target: &str,
    row: usize,
) -> Result<(Option<f64>, String), String> {
    match operand {
        Operand::Number(n) => Ok((Some(*n), n.to_string())),
        Operand::Text(s) => Ok((s.parse::<f64>().ok(), s.clone())),
        Operand::Name(name) => {
            if let Some(var) = collection.variables.get(name) {
                let text = var.str_values.get(row).cloned().unwrap_or_default();
                let num = var.values.get(row).copied().filter(|v| v.is_finite());
                return Ok((num, text));
            }
            let text = collection
                .variables
                .get(target)
                .and_then(|v| v.index.get(row))
                .and_then(|e| e.get(name.as_str()))
                .cloned()
                .ok_or_else(|| format!("query references unknown variable or entity '{name}'"))?;
            let num = text.parse::<f64>().ok();
            Ok((num, text))
        }
    }
}

fn eval_predicate(
    pred: &Predicate,
    collection: &VariableCollection,
    target: &str,
    row: usize,
) -> Result<bool, String> {
    let (lnum, ltext) = operand_value(&pred.lhs, collection, target, row)?;
    let (rnum, rtext) = operand_value(&pred.rhs, collection, target, row)?;

    if let (Some(a), Some(b)) = (lnum, rnum) {
        return Ok(match pred.op {
            Cmp::Eq => (a - b).abs() < f64::EPSILON,
            Cmp::Ne => (a - b).abs() >= f64::EPSILON,
            Cmp::Ge => a >= b,
            Cmp::Le => a <= b,
            Cmp::Gt => a > b,
            Cmp::Lt => a < b,
        });
    }
    match pred.op {
        Cmp::Eq => Ok(ltext == rtext),
        Cmp::Ne => Ok(ltext != rtext),
        Cmp::Ge => Ok(ltext >= rtext),
        Cmp::Le => Ok(ltext <= rtext),
        Cmp::Gt => Ok(ltext > rtext),
        Cmp::Lt => Ok(ltext < rtext),
    }
}

fn eval_query(
    query: &Query,
    collection: &VariableCollection,
    target: &str,
    row: usize,
) -> Result<bool, String> {
    let mut acc = eval_predicate(&query.first, collection, target, row)?;
    for (join, pred) in &query.rest {
        let rhs = eval_predicate(pred, collection, target, row)?;
        acc = match join {
            Join::And => acc && rhs,
            Join::Or => acc || rhs,
        };
    }
    Ok(acc)
}

// ───────────────────────── Transformer manager ─────────────────────────

/// A user-supplied transformation.
///
/// Receives the collection to modify and the raw instruction JSON, and returns
/// `Err` with a message the manager turns into a [`TransformError`].
pub type CustomTransformation =
    Box<dyn Fn(&mut VariableCollection, &serde_json::Value) -> Result<(), String>>;

/// Applies transformations while optionally recording intermediate states.
///
/// Corresponds to PyBIDS' `TransformerManager`.
///
/// # Custom transformations
///
/// [`register()`](Self::register) adds a handler under a name. Registered
/// handlers are consulted *before* the built-in table, so registering under a
/// built-in name overrides it — the same precedence PyBIDS gives its own
/// registry.
///
/// ```
/// use bids_modeling::transformations::{TransformSpec, TransformerManager};
/// use bids_variables::collections::VariableCollection;
/// use bids_variables::variables::SimpleVariable;
///
/// let mut manager = TransformerManager::new("pybids-transforms-v1", false);
/// manager.register("Double", Box::new(|coll: &mut VariableCollection, instr| {
///     let name = instr["Input"][0].as_str().ok_or("Double needs an Input")?;
///     let doubled: Vec<f64> = coll
///         .variables
///         .get(name)
///         .ok_or_else(|| format!("no such variable '{name}'"))?
///         .values
///         .iter()
///         .map(|v| v * 2.0)
///         .collect();
///     coll.set_values(name, doubled);
///     Ok(())
/// }));
///
/// let var = SimpleVariable::new("x", "events", vec!["1".into(), "2".into()], vec![]);
/// let coll = VariableCollection::new(vec![var]);
/// let spec = TransformSpec {
///     transformer: "pybids-transforms-v1".into(),
///     instructions: vec![serde_json::json!({"Name": "Double", "Input": ["x"]})],
/// };
///
/// let out = manager.transform(coll, &spec);
/// assert!(manager.errors.is_empty());
/// assert_eq!(out.variables["x"].values, vec![2.0, 4.0]);
/// ```
#[derive(Default)]
pub struct TransformerManager {
    pub transformer: String,
    pub keep_history: bool,
    pub history: Vec<VariableCollection>,
    /// Errors from the most recent [`transform()`](Self::transform) call.
    pub errors: Vec<TransformError>,
    custom: HashMap<String, CustomTransformation>,
}

impl fmt::Debug for TransformerManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names: Vec<&str> = self.custom.keys().map(String::as_str).collect();
        names.sort_unstable();
        f.debug_struct("TransformerManager")
            .field("transformer", &self.transformer)
            .field("keep_history", &self.keep_history)
            .field("history", &self.history.len())
            .field("errors", &self.errors)
            .field("registered", &names)
            .finish()
    }
}

impl TransformerManager {
    #[must_use]
    pub fn new(transformer: &str, keep_history: bool) -> Self {
        Self {
            transformer: transformer.into(),
            keep_history,
            history: Vec::new(),
            errors: Vec::new(),
            custom: HashMap::new(),
        }
    }

    /// Register a handler for `name`, replacing any previous registration.
    ///
    /// Corresponds to PyBIDS' `TransformerManager.register`. As there, `And`
    /// and `Or` are accepted with or without a trailing underscore.
    pub fn register(&mut self, name: &str, handler: CustomTransformation) {
        self.custom.insert(sanitize_name(name), handler);
    }

    /// Whether a handler is registered under `name`.
    #[must_use]
    pub fn is_registered(&self, name: &str) -> bool {
        self.custom.contains_key(&sanitize_name(name))
    }

    /// Names of the registered handlers, sorted.
    #[must_use]
    pub fn registered(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.custom.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Apply `spec` to `collection`, recording any errors in
    /// [`errors`](Self::errors).
    pub fn transform(
        &mut self,
        mut collection: VariableCollection,
        spec: &TransformSpec,
    ) -> VariableCollection {
        self.errors.clear();
        for instruction in &spec.instructions {
            let name = instruction_name(instruction).to_string();
            let result = match self.custom.get(&sanitize_name(&name)) {
                Some(handler) => handler(&mut collection, instruction),
                None => dispatch_instruction(&mut collection, instruction),
            };
            if let Err(msg) = result {
                self.errors.push(TransformError::new(&name, msg));
            }
            if self.keep_history {
                self.history.push(collection.clone());
            }
        }
        collection
    }
}

/// Normalise a transformation name.
///
/// PyBIDS stores the Python-keyword-avoiding `And_`/`Or_` under those names;
/// the BIDS-StatsModels JSON uses `And`/`Or`. Accept both.
fn sanitize_name(name: &str) -> String {
    match name {
        "And_" => "And".into(),
        "Or_" => "Or".into(),
        other => other.to_string(),
    }
}

/// Expand wildcard patterns in a list of variable names.
#[must_use]
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
    use bids_variables::RunInfo;

    fn spec(instructions: Vec<serde_json::Value>) -> TransformSpec {
        TransformSpec {
            transformer: "pybids-transforms-v1".into(),
            instructions,
        }
    }

    fn simple(name: &str, values: &[&str]) -> SimpleVariable {
        let index: Vec<StringEntities> = (0..values.len())
            .map(|i| {
                let mut e = StringEntities::new();
                e.insert("run".into(), format!("{}", i % 2 + 1));
                e
            })
            .collect();
        SimpleVariable::new(
            name,
            "events",
            values.iter().map(|s| (*s).to_string()).collect(),
            index,
        )
    }

    fn collection(vars: Vec<SimpleVariable>) -> VariableCollection {
        VariableCollection::new(vars)
    }

    /// A run-level collection: one 3-event variable over a 20 s run.
    fn run_collection() -> VariableCollection {
        let mut c = collection(vec![simple("face", &["1", "1", "1"])]);
        c.run_info = vec![RunInfo {
            entities: StringEntities::new(),
            duration: 20.0,
            tr: 2.0,
            image: None,
            n_vols: 10,
        }];
        c.sampling_rate = 10.0;
        c.timing.insert(
            "face".into(),
            VariableTiming {
                onset: vec![0.0, 5.0, 10.0],
                duration: vec![1.0, 1.0, 1.0],
            },
        );
        c
    }

    fn vals(c: &VariableCollection, name: &str) -> Vec<f64> {
        c.variables.get(name).unwrap().values.clone()
    }

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    // ── Coverage ──

    #[test]
    fn every_pybids_transformation_is_dispatched() {
        // The 24 transformations defined by bids.modeling.transformations.
        let pybids = [
            "Assign",
            "Copy",
            "Delete",
            "DropNA",
            "Factor",
            "Filter",
            "Group",
            "Rename",
            "Replace",
            "Select",
            "Split",
            "ToDense",
            "Resample",
            "Convolve",
            "Demean",
            "Orthogonalize",
            "Product",
            "Scale",
            "Sum",
            "Threshold",
            "And",
            "Or",
            "Not",
            "Lag",
        ];
        for name in pybids {
            assert!(
                SUPPORTED_TRANSFORMATIONS.contains(&name),
                "'{name}' is not dispatched"
            );
        }
    }

    #[test]
    fn unknown_transformation_reports_an_error() {
        let mut c = collection(vec![simple("a", &["1"])]);
        let errors =
            apply_transformations(&mut c, &spec(vec![serde_json::json!({"Name": "Nope"})]));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unknown transformation"));
    }

    #[test]
    fn missing_input_reports_an_error_rather_than_passing_silently() {
        let mut c = collection(vec![simple("a", &["1"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Demean", "Input": ["nope"]}),
            ]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("no such variable"));
    }

    // ── Computation ──

    #[test]
    fn scale_defaults_to_z_score_with_sample_sd() {
        let mut c = collection(vec![simple("x", &["1", "2", "3", "4"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({"Name": "Scale", "Input": ["x"]})]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        // mean 2.5, sample sd (ddof=1) = sqrt(5/3) ≈ 1.2909944487358056
        let sd = (5.0f64 / 3.0).sqrt();
        let got = vals(&c, "x");
        approx(got[0], -1.5 / sd);
        approx(got[3], 1.5 / sd);
    }

    #[test]
    fn scale_rejects_a_constant_column() {
        let mut c = collection(vec![simple("x", &["2", "2", "2"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({"Name": "Scale", "Input": ["x"]})]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("constant value"));
    }

    #[test]
    fn demean_centres_on_zero() {
        let mut c = collection(vec![simple("x", &["1", "2", "3"])]);
        apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({"Name": "Demean", "Input": ["x"]})]),
        );
        assert_eq!(vals(&c, "x"), vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn sum_applies_weights() {
        let mut c = collection(vec![simple("a", &["1", "2"]), simple("b", &["10", "20"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Sum", "Input": ["a", "b"], "Output": "s", "Weights": [1, -1]
            })]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(vals(&c, "s"), vec![-9.0, -18.0]);
    }

    #[test]
    fn sum_rejects_a_weight_length_mismatch() {
        let mut c = collection(vec![simple("a", &["1"]), simple("b", &["2"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Sum", "Input": ["a", "b"], "Output": "s", "Weights": [1]
            })]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("Weights"));
    }

    #[test]
    fn product_multiplies_elementwise() {
        let mut c = collection(vec![simple("a", &["2", "3"]), simple("b", &["4", "5"])]);
        apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Product", "Input": ["a", "b"], "Output": "p"}),
            ]),
        );
        assert_eq!(vals(&c, "p"), vec![8.0, 15.0]);
    }

    #[test]
    fn product_and_sum_require_an_output() {
        let mut c = collection(vec![simple("a", &["1"]), simple("b", &["2"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Product", "Input": ["a", "b"]}),
                serde_json::json!({"Name": "Sum", "Input": ["a", "b"]}),
            ]),
        );
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn logical_transformations() {
        let mut c = collection(vec![
            simple("a", &["1", "1", "0"]),
            simple("b", &["1", "0", "0"]),
        ]);
        apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "And", "Input": ["a", "b"], "Output": "and"}),
                serde_json::json!({"Name": "Or", "Input": ["a", "b"], "Output": "or"}),
                serde_json::json!({"Name": "Not", "Input": ["a"], "Output": "not_a"}),
            ]),
        );
        assert_eq!(vals(&c, "and"), vec![1.0, 0.0, 0.0]);
        assert_eq!(vals(&c, "or"), vec![1.0, 1.0, 0.0]);
        assert_eq!(vals(&c, "not_a"), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn threshold_signed_and_unsigned() {
        let mut c = collection(vec![
            simple("s", &["-4", "1", "4"]),
            simple("u", &["-4", "1", "4"]),
        ]);
        apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Threshold", "Input": ["s"], "Threshold": 3}),
                serde_json::json!({
                    "Name": "Threshold", "Input": ["u"], "Threshold": 3, "Signed": false
                }),
            ]),
        );
        assert_eq!(vals(&c, "s"), vec![0.0, 0.0, 4.0]);
        // Unsigned compares — and returns — absolute values.
        assert_eq!(vals(&c, "u"), vec![4.0, 0.0, 4.0]);
    }

    #[test]
    fn threshold_binarize() {
        let mut c = collection(vec![simple("x", &["0", "5"])]);
        apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Threshold", "Input": ["x"], "Threshold": 1, "Binarize": true
            })]),
        );
        assert_eq!(vals(&c, "x"), vec![0.0, 1.0]);
    }

    #[test]
    fn orthogonalize_accounts_for_the_intercept() {
        // y = 2x + 3 with an offset: a plain projection would leave a residual
        // slope because neither column is mean-centred.
        let mut c = collection(vec![
            simple("y", &["5", "7", "9", "11"]),
            simple("x", &["1", "2", "3", "4"]),
        ]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Orthogonalize", "Input": ["y"], "Other": ["x"]
            })]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        // y - 2x is the constant 3 everywhere.
        for v in vals(&c, "y") {
            approx(v, 3.0);
        }
    }

    #[test]
    fn lag_shifts_and_differences() {
        let mut c = collection(vec![
            simple("a", &["1", "2", "3"]),
            simple("b", &["1", "2", "3"]),
            simple("d", &["1", "2", "4"]),
        ]);
        apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Lag", "Input": ["a"], "Shift": 1}),
                serde_json::json!({
                    "Name": "Lag", "Input": ["b"], "Shift": 1, "Mode": "constant", "Constant": 0
                }),
                serde_json::json!({"Name": "Lag", "Input": ["d"], "Difference": true}),
            ]),
        );
        // "nearest" replicates the boundary value.
        assert_eq!(vals(&c, "a"), vec![1.0, 1.0, 2.0]);
        assert_eq!(vals(&c, "b"), vec![0.0, 1.0, 2.0]);
        // Backward difference against the nearest-extended series.
        assert_eq!(vals(&c, "d"), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn lag_reflect_mode() {
        let mut c = collection(vec![simple("a", &["1", "2", "3"])]);
        apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Lag", "Input": ["a"], "Shift": 1, "Mode": "reflect"
            })]),
        );
        assert_eq!(vals(&c, "a"), vec![1.0, 1.0, 2.0]);
    }

    // ── Munging ──

    #[test]
    fn factor_dummy_codes_and_drops_the_original() {
        let mut c = collection(vec![simple("tt", &["face", "house", "face"])]);
        apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({"Name": "Factor", "Input": ["tt"]})]),
        );
        assert!(
            !c.variables.contains_key("tt"),
            "original should be removed"
        );
        assert_eq!(vals(&c, "tt.face"), vec![1.0, 0.0, 1.0]);
        assert_eq!(vals(&c, "tt.house"), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn factor_drop_one_omits_the_reference_level() {
        let mut c = collection(vec![simple("tt", &["face", "house", "scene"])]);
        apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Factor", "Input": ["tt"], "Constraint": "drop_one"
            })]),
        );
        // Levels sort to [face, house, scene]; "face" is the reference.
        assert!(!c.variables.contains_key("tt.face"));
        assert!(c.variables.contains_key("tt.house"));
        assert!(c.variables.contains_key("tt.scene"));
    }

    #[test]
    fn factor_mean_zero_codes_the_reference_level_negatively() {
        let mut c = collection(vec![simple("tt", &["a", "b", "c"])]);
        apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Factor", "Input": ["tt"], "Constraint": "mean_zero"
            })]),
        );
        approx(vals(&c, "tt.b")[0], -0.5);
        approx(vals(&c, "tt.c")[0], -0.5);
    }

    #[test]
    fn filter_keeps_matching_rows() {
        let mut c = collection(vec![
            simple("rt", &["100", "200", "300"]),
            simple("acc", &["1", "0", "1"]),
        ]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Filter", "Input": ["rt"], "Query": "acc == 1 and rt > 150"
            })]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(vals(&c, "rt"), vec![300.0]);
    }

    #[test]
    fn filter_matches_strings_and_entities() {
        let mut c = collection(vec![simple("tt", &["face", "house", "face"])]);
        apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Filter", "Input": ["tt"], "Query": "tt == 'face'"
            })]),
        );
        assert_eq!(c.n_rows("tt"), 2);

        let mut c2 = collection(vec![simple("tt", &["a", "b", "c", "d"])]);
        apply_transformations(
            &mut c2,
            &spec(vec![serde_json::json!({
                "Name": "Filter", "Input": ["tt"], "Query": "run == 1"
            })]),
        );
        assert_eq!(c2.n_rows("tt"), 2);
    }

    #[test]
    fn filter_rejects_a_malformed_query() {
        let mut c = collection(vec![simple("x", &["1"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Filter", "Input": ["x"], "Query": "x"}),
            ]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("comparison operator"));
    }

    #[test]
    fn split_creates_one_variable_per_level() {
        let mut c = collection(vec![
            simple("rt", &["1", "2", "3", "4"]),
            simple("cond", &["a", "b", "a", "b"]),
        ]);
        apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Split", "Input": ["rt"], "By": ["cond"]}),
            ]),
        );
        assert_eq!(vals(&c, "rt.a"), vec![1.0, 3.0]);
        assert_eq!(vals(&c, "rt.b"), vec![2.0, 4.0]);
    }

    #[test]
    fn rename_and_copy_require_an_output() {
        let mut c = collection(vec![simple("a", &["1"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Rename", "Input": ["a"]}),
                serde_json::json!({"Name": "Copy", "Input": ["a"]}),
            ]),
        );
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn output_prefix_and_suffix_apply_generically() {
        let mut c = collection(vec![simple("x", &["1", "2", "3"])]);
        apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Demean", "Input": ["x"], "OutputSuffix": "_c"
            })]),
        );
        assert!(c.variables.contains_key("x_c"));
        assert!(c.variables.contains_key("x"), "input is left in place");
    }

    #[test]
    fn group_records_membership() {
        let mut c = collection(vec![simple("a", &["1"]), simple("b", &["2"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Group", "Input": ["a", "b"], "Output": "both"
            })]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            c.groups.get("both").unwrap(),
            &["a".to_string(), "b".into()]
        );
    }

    #[test]
    fn group_rejects_a_name_clash() {
        let mut c = collection(vec![simple("a", &["1"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Group", "Input": ["a"], "Output": "a"}),
            ]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("conflicts"));
    }

    #[test]
    fn assign_copies_an_attribute_onto_the_target() {
        let mut c = run_collection();
        c.insert_sparse(
            simple("target", &["0", "0", "0"]),
            VariableTiming {
                onset: vec![0.0, 5.0, 10.0],
                duration: vec![0.0, 0.0, 0.0],
            },
        );
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Assign", "Input": ["face"], "Target": "target",
                "InputAttr": "duration", "TargetAttr": "duration"
            })]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            c.timing.get("target").unwrap().duration,
            vec![1.0, 1.0, 1.0]
        );
    }

    #[test]
    fn assign_rejects_a_length_mismatch() {
        let mut c = run_collection();
        c.insert_sparse(
            simple("target", &["0"]),
            VariableTiming {
                onset: vec![0.0],
                duration: vec![0.0],
            },
        );
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Assign", "Input": ["face"], "Target": "target"
            })]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("same number of events"));
    }

    // ── Run-level: densify, resample, convolve ──

    #[test]
    fn to_dense_then_resample() {
        let mut c = run_collection();
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "ToDense", "Input": ["face"], "SamplingRate": 10}),
            ]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert!(c.is_dense("face"));
        assert!(!c.is_sparse("face"));
        let n = c.n_rows("face");
        assert_eq!(n, 200, "20 s at 10 Hz");
        // Three 1 s boxcars at amplitude 1 → 30 non-zero samples.
        let nonzero = vals(&c, "face").iter().filter(|v| **v > 0.5).count();
        assert!(
            (25..=35).contains(&nonzero),
            "got {nonzero} non-zero samples"
        );

        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Resample", "Input": ["face"], "SamplingRate": 5}),
            ]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(c.n_rows("face"), 100);
        approx(c.sampling_rate, 5.0);
    }

    #[test]
    fn resample_on_a_sparse_variable_is_an_error_not_a_no_op() {
        let mut c = run_collection();
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Resample", "Input": ["face"]}),
            ]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("ToDense"));
    }

    #[test]
    fn convolve_produces_a_dense_hrf_regressor() {
        let mut c = run_collection();
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Convolve", "Input": ["face"], "Model": "spm"}),
            ]),
        );
        assert!(errors.is_empty(), "{errors:?}");

        let v = vals(&c, "face");
        assert!(c.is_dense("face"));
        assert_eq!(v.len(), 200, "20 s at 10 Hz");
        assert!(
            v.iter().all(|x| x.is_finite()),
            "regressor has non-finite values"
        );

        // The HRF is causal and peaks several seconds after the first onset.
        approx(v[0], 0.0);
        let peak = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(*peak.1 > 0.0, "convolved regressor is flat");
        assert!(peak.0 > 10, "peak at sample {} is too early", peak.0);
    }

    #[test]
    fn convolve_emits_derivative_and_dispersion_bases() {
        let mut c = run_collection();
        let errors = apply_transformations(
            &mut c,
            &spec(vec![serde_json::json!({
                "Name": "Convolve", "Input": ["face"], "Model": "spm",
                "Derivative": true, "Dispersion": true
            })]),
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert!(c.variables.contains_key("face"));
        assert!(c.variables.contains_key("face_derivative"));
        assert!(c.variables.contains_key("face_dispersion"));
    }

    #[test]
    fn convolve_rejects_an_unknown_model() {
        let mut c = run_collection();
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Convolve", "Input": ["face"], "Model": "wat"}),
            ]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("'spm', 'glover', or 'fir'"));
    }

    #[test]
    fn convolve_without_timing_reports_why() {
        // A collection built without run timing cannot be convolved, and must
        // say so instead of silently leaving the regressor unconvolved.
        let mut c = collection(vec![simple("face", &["1", "1"])]);
        let errors = apply_transformations(
            &mut c,
            &spec(vec![
                serde_json::json!({"Name": "Convolve", "Input": ["face"]}),
            ]),
        );
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("from_run_collection"));
    }

    // ── Spec parsing ──

    #[test]
    fn instructions_parse_from_pascal_case_json() {
        let s = spec(vec![
            serde_json::json!({"Name": "Scale", "Input": ["x"], "Demean": true}),
            serde_json::json!({"Name": "Convolve", "Input": ["y"], "Model": "glover"}),
        ]);
        let parsed = s.parse_instructions().unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name(), "Scale");
        assert_eq!(parsed[1].name(), "Convolve");
        assert!(s.validate().is_empty());
    }

    #[test]
    fn validate_flags_unknown_names() {
        let s = spec(vec![serde_json::json!({"Name": "Bogus", "Input": ["x"]})]);
        let errors = s.validate();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].instruction, "Bogus");
    }

    #[test]
    fn expand_wildcards_matches_patterns() {
        let pool = vec!["tt.face".to_string(), "tt.house".into(), "rt".into()];
        let got = expand_wildcards(&["tt.*".to_string()], &pool);
        assert_eq!(got.len(), 2);
    }
}
