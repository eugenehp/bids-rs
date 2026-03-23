//! BIDS entity definitions, parsing, and canonical ordering.
//!
//! Entities are the key-value pairs encoded in BIDS filenames (e.g.,
//! `sub-01`, `task-rest`, `run-02`). This module provides:
//!
//! - [`Entity`] — A named entity definition with a regex pattern for extraction
//! - [`EntityValue`] — A typed value (string, padded int, float, bool, JSON)
//! - [`parse_file_entities()`] — Extract all entities from a file path
//! - [`ENTITY_ORDER`] — Canonical BIDS entity ordering
//!
//! All other crates use these types to represent and match BIDS entities.

use indexmap::IndexMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::padded_int::PaddedInt;

/// Represents a single entity defined in configuration.
///
/// Corresponds to PyBIDS `Entity` — a named key (e.g., "subject", "task")
/// with a regex pattern to extract values from file paths.
///
/// # Example
///
/// ```
/// use bids_core::Entity;
///
/// let ent = Entity::new("subject", r"[/\\]+sub-([a-zA-Z0-9]+)");
/// let val = ent.match_path("/sub-01/anat/sub-01_T1w.nii.gz");
/// assert_eq!(val.unwrap().as_str_lossy(), "01");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub pattern: String,
    #[serde(default)]
    pub mandatory: bool,
    #[serde(default)]
    pub directory: Option<String>,
    #[serde(default = "default_dtype")]
    pub dtype: String,

    /// Lazily compiled regex — uses `OnceLock` so matching only needs `&self`.
    #[serde(skip)]
    compiled_regex: OnceLock<Option<Regex>>,
}

fn default_dtype() -> String {
    "str".to_string()
}

impl std::fmt::Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Entity('{}', dtype={})", self.name, self.dtype)?;
        if self.mandatory {
            write!(f, " [mandatory]")?;
        }
        Ok(())
    }
}

impl Entity {
    /// Create a new entity.
    ///
    /// Eagerly compiles the regex. If the pattern is invalid, the entity will
    /// never match any path. Prefer [`Entity::try_new()`] when working with
    /// user-supplied patterns so you can surface the error.
    pub fn new(name: &str, pattern: &str) -> Self {
        let lock = OnceLock::new();
        let compiled = Regex::new(pattern).ok();
        #[cfg(debug_assertions)]
        if compiled.is_none() {
            log::warn!("invalid regex pattern for entity '{name}': {pattern}");
        }
        let _ = lock.set(compiled);
        Self {
            name: name.to_string(),
            pattern: pattern.to_string(),
            mandatory: false,
            directory: None,
            dtype: "str".to_string(),
            compiled_regex: lock,
        }
    }

    /// Create a new entity, returning an error if the regex pattern is invalid.
    ///
    /// # Errors
    ///
    /// Returns `regex::Error` if `pattern` is not a valid regular expression.
    pub fn try_new(name: &str, pattern: &str) -> Result<Self, regex::Error> {
        let compiled = Regex::new(pattern)?;
        let lock = OnceLock::new();
        let _ = lock.set(Some(compiled));
        Ok(Self {
            name: name.to_string(),
            pattern: pattern.to_string(),
            mandatory: false,
            directory: None,
            dtype: "str".to_string(),
            compiled_regex: lock,
        })
    }

    /// Set the data type (`"str"`, `"int"`, `"float"`, `"bool"`).
    #[must_use]
    pub fn with_dtype(mut self, dtype: &str) -> Self {
        self.dtype = dtype.to_string();
        self
    }

    /// Set the directory pattern for this entity.
    #[must_use]
    pub fn with_directory(mut self, directory: &str) -> Self {
        self.directory = Some(directory.to_string());
        self
    }

    /// Mark this entity as mandatory.
    #[must_use]
    pub fn with_mandatory(mut self, mandatory: bool) -> Self {
        self.mandatory = mandatory;
        self
    }

    /// Return the compiled regex, lazily compiling on first access.
    ///
    /// Returns `None` if the pattern is invalid. Only requires `&self`.
    pub fn regex(&self) -> Option<&Regex> {
        self.compiled_regex
            .get_or_init(|| Regex::new(&self.pattern).ok())
            .as_ref()
    }

    /// Match the entity pattern against a file path.
    /// Returns the captured value if found.
    ///
    /// Only requires `&self` (no mutable borrow needed).
    pub fn match_path(&self, path: &str) -> Option<EntityValue> {
        let regex = self.regex()?;
        let caps = regex.captures(path)?;
        let val_str = caps.get(1)?.as_str();
        Some(self.coerce_value(val_str))
    }

    /// Coerce a string value to the appropriate type.
    pub fn coerce_value(&self, val: &str) -> EntityValue {
        match self.dtype.as_str() {
            "int" => EntityValue::Int(PaddedInt::new(val)),
            "float" => EntityValue::Float(val.parse().unwrap_or(0.0)),
            "bool" => EntityValue::Bool(val.parse().unwrap_or(false)),
            _ => EntityValue::Str(val.to_string()),
        }
    }
}

/// A typed entity value, preserving the original representation where needed.
///
/// Most entity values are strings (e.g., `sub-01` → `Str("01")`), but some
/// are typed as integers (preserving zero-padding via [`PaddedInt`]),
/// floats, or booleans. The [`Json`](EntityValue::Json) variant is used for
/// metadata values merged from JSON sidecars.
///
/// # Conversions
///
/// - `From<&str>` and `From<String>` → `EntityValue::Str`
/// - `From<i32>` and `From<i64>` → `EntityValue::Int`
/// - `From<f64>` → `EntityValue::Float`
/// - `From<bool>` → `EntityValue::Bool`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EntityValue {
    Str(String),
    Int(PaddedInt),
    Float(f64),
    Bool(bool),
    Json(serde_json::Value),
}

impl From<&str> for EntityValue {
    fn from(s: &str) -> Self {
        EntityValue::Str(s.to_string())
    }
}

impl From<String> for EntityValue {
    fn from(s: String) -> Self {
        EntityValue::Str(s)
    }
}

impl From<i32> for EntityValue {
    fn from(v: i32) -> Self {
        EntityValue::Int(PaddedInt::from(v))
    }
}

impl From<i64> for EntityValue {
    fn from(v: i64) -> Self {
        EntityValue::Int(PaddedInt::from(v))
    }
}

impl From<f64> for EntityValue {
    fn from(v: f64) -> Self {
        EntityValue::Float(v)
    }
}

impl From<bool> for EntityValue {
    fn from(v: bool) -> Self {
        EntityValue::Bool(v)
    }
}

impl EntityValue {
    /// Get the value as a string representation.
    ///
    /// Returns a `Cow::Borrowed` for `Str` variants (zero-copy) and
    /// `Cow::Owned` for all others.
    #[must_use]
    pub fn as_str_lossy(&self) -> std::borrow::Cow<'_, str> {
        match self {
            EntityValue::Str(s) => std::borrow::Cow::Borrowed(s),
            EntityValue::Int(i) => std::borrow::Cow::Owned(i.to_string()),
            EntityValue::Float(f) => std::borrow::Cow::Owned(f.to_string()),
            EntityValue::Bool(b) => std::borrow::Cow::Owned(b.to_string()),
            EntityValue::Json(v) => std::borrow::Cow::Owned(v.to_string()),
        }
    }

    /// Try to extract the value as an `i64`.
    ///
    /// Returns `Some` for `Int`, `Float` (truncated), and `Str` (parsed).
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            EntityValue::Int(p) => Some(p.value()),
            EntityValue::Float(f) => Some(*f as i64),
            EntityValue::Str(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Try to extract the value as an `f64`.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            EntityValue::Float(f) => Some(*f),
            EntityValue::Int(p) => Some(p.value() as f64),
            EntityValue::Str(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Try to extract the value as a `bool`.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            EntityValue::Bool(b) => Some(*b),
            EntityValue::Str(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// Returns `true` if this is a `Str` variant.
    #[must_use]
    pub fn is_str(&self) -> bool {
        matches!(self, EntityValue::Str(_))
    }

    /// Returns `true` if this is an `Int` variant.
    #[must_use]
    pub fn is_int(&self) -> bool {
        matches!(self, EntityValue::Int(_))
    }
}

impl std::fmt::Display for EntityValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str_lossy())
    }
}

impl PartialEq for EntityValue {
    fn eq(&self, other: &Self) -> bool {
        // Use canonical string form for all comparisons so that Eq and Hash
        // are consistent — two values are equal iff they produce the same
        // canonical string.  This avoids the previous bug where float epsilon
        // comparison could disagree with the string-based Hash.
        *self.as_str_lossy() == *other.as_str_lossy()
    }
}

impl Eq for EntityValue {}

impl std::hash::Hash for EntityValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str_lossy().hash(state);
    }
}

/// A map from entity names to their typed values for a single file.
///
/// Uses `IndexMap` to preserve insertion order, which gives deterministic
/// iteration in canonical BIDS entity order when entities are inserted
/// following [`ENTITY_ORDER`].
pub type Entities = IndexMap<String, EntityValue>;

/// A map from entity names to string values (used in variables/collections).
pub type StringEntities = HashMap<String, String>;

/// Standard BIDS entities in their canonical order.
pub const ENTITY_ORDER: &[&str] = &[
    "subject",
    "session",
    "sample",
    "task",
    "tracksys",
    "acquisition",
    "ceagent",
    "staining",
    "tracer",
    "reconstruction",
    "direction",
    "run",
    "modality",
    "echo",
    "flip",
    "inversion",
    "mtransfer",
    "part",
    "processing",
    "hemisphere",
    "space",
    "split",
    "recording",
    "chunk",
    "atlas",
    "resolution",
    "density",
    "label",
    "description",
    "suffix",
    "extension",
    "datatype",
];

/// Parse entities from a filename using the provided entity definitions.
///
/// Only requires `&[Entity]` — no mutable borrow needed thanks to lazy regex
/// compilation via `OnceLock`.
#[must_use]
pub fn parse_file_entities(path: &str, entities: &[Entity]) -> Entities {
    let mut result = Entities::new();
    for entity in entities.iter() {
        if let Some(val) = entity.match_path(path) {
            result.insert(entity.name.clone(), val);
        }
    }
    result
}

/// Sort entity keys according to the canonical BIDS ordering.
#[must_use]
pub fn sort_entities(entities: &Entities) -> Vec<(String, EntityValue)> {
    let mut pairs: Vec<_> = entities
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    pairs.sort_by_key(|(k, _)| {
        ENTITY_ORDER
            .iter()
            .position(|&e| e == k.as_str())
            .unwrap_or(ENTITY_ORDER.len())
    });

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_matching() {
        let ent = Entity::new("subject", r"[/\\]+sub-([a-zA-Z0-9]+)");
        let val = ent.match_path("/sub-01/anat/sub-01_T1w.nii.gz");
        assert!(val.is_some());
        assert_eq!(val.unwrap().as_str_lossy(), "01");
    }

    #[test]
    fn test_int_entity() {
        let ent = Entity::new("run", r"[_/\\]+run-(\d+)").with_dtype("int");
        let val = ent.match_path("sub-01_task-rest_run-02_bold.nii.gz");
        assert!(val.is_some());
        match val.unwrap() {
            EntityValue::Int(p) => {
                assert_eq!(p.value(), 2);
                assert_eq!(p.to_string(), "02");
            }
            _ => panic!("Expected Int"),
        }
    }

    #[test]
    fn test_parse_file_entities() {
        let entities = vec![
            Entity::new("subject", r"[/\\]+sub-([a-zA-Z0-9]+)"),
            Entity::new("session", r"[_/\\]+ses-([a-zA-Z0-9]+)"),
            Entity::new("task", r"[_/\\]+task-([a-zA-Z0-9]+)"),
            Entity::new("suffix", r"[_/\\]([a-zA-Z0-9]+)\.[^/\\]+$"),
        ];
        let result = parse_file_entities(
            "/sub-01/ses-02/eeg/sub-01_ses-02_task-rest_eeg.edf",
            &entities,
        );
        assert_eq!(result.get("subject").unwrap().as_str_lossy(), "01");
        assert_eq!(result.get("session").unwrap().as_str_lossy(), "02");
        assert_eq!(result.get("task").unwrap().as_str_lossy(), "rest");
    }
}
