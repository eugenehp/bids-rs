//! Metadata dictionary for BIDS files.
//!
//! Provides [`BidsMetadata`], an ordered key-value store for JSON sidecar
//! metadata that supports typed accessors, iteration, merging, and
//! deserialization into arbitrary structs.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Ordered metadata dictionary for a BIDS file.
///
/// Wraps an `IndexMap<String, serde_json::Value>` preserving insertion order.
/// Provides typed accessors for common value types and can be deserialized
/// into arbitrary structs via [`deserialize_as()`](Self::deserialize_as).
///
/// Metadata is populated from JSON sidecar files following the BIDS
/// inheritance principle, where more-specific sidecars (closer to the data
/// file) override less-specific ones (closer to the dataset root).
///
/// Corresponds to PyBIDS' `BIDSMetadata` class.
///
/// # Example
///
/// ```
/// use bids_core::metadata::BidsMetadata;
/// use serde_json::json;
///
/// let mut md = BidsMetadata::new();
/// md.insert("SamplingFrequency".into(), json!(256.0));
/// md.insert("EEGReference".into(), json!("Cz"));
///
/// assert_eq!(md.get_f64("SamplingFrequency"), Some(256.0));
/// assert_eq!(md.get_str("EEGReference"), Some("Cz"));
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BidsMetadata {
    inner: IndexMap<String, Value>,
    /// The source file this metadata is associated with.
    #[serde(skip)]
    pub source_file: Option<String>,
}

impl BidsMetadata {
    /// Create an empty metadata dictionary.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty metadata dictionary tagged with a source file path.
    pub fn with_source(source_file: &str) -> Self {
        Self {
            inner: IndexMap::new(),
            source_file: Some(source_file.to_string()),
        }
    }

    /// Get a raw JSON value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.inner.get(key)
    }

    /// Get a string value by key (returns `None` if missing or not a string).
    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.inner.get(key).and_then(|v| v.as_str())
    }

    /// Get a float value by key (returns `None` if missing or not numeric).
    #[must_use]
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.inner.get(key).and_then(serde_json::Value::as_f64)
    }

    /// Get an integer value by key (returns `None` if missing or not numeric).
    #[must_use]
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.inner.get(key).and_then(serde_json::Value::as_i64)
    }

    /// Get a boolean value by key.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.inner.get(key).and_then(serde_json::Value::as_bool)
    }

    /// Get a JSON array value by key.
    #[must_use]
    pub fn get_array(&self, key: &str) -> Option<&Vec<Value>> {
        self.inner.get(key).and_then(|v| v.as_array())
    }

    /// Insert a key-value pair, replacing any existing value.
    pub fn insert(&mut self, key: String, value: Value) {
        self.inner.insert(key, value);
    }

    /// Merge all entries from `other` into this metadata (overwrites on conflict).
    pub fn extend(&mut self, other: BidsMetadata) {
        self.inner.extend(other.inner);
    }

    /// Merge entries from an `IndexMap` into this metadata.
    pub fn update_from_map(&mut self, map: IndexMap<String, Value>) {
        self.inner.extend(map);
    }

    /// Check if a key exists.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(key)
    }

    /// Iterate over keys.
    #[must_use]
    pub fn keys(&self) -> indexmap::map::Keys<'_, String, Value> {
        self.inner.keys()
    }

    /// Iterate over key-value pairs.
    #[must_use]
    pub fn iter(&self) -> indexmap::map::Iter<'_, String, Value> {
        self.inner.iter()
    }

    /// Number of metadata entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if the metadata is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl FromIterator<(String, Value)> for BidsMetadata {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(iter: I) -> Self {
        Self {
            inner: IndexMap::from_iter(iter),
            source_file: None,
        }
    }
}

impl BidsMetadata {
    /// Try to deserialize this metadata into a typed struct.
    ///
    /// Works via JSON roundtrip: metadata map → `serde_json::Value` → `T`.
    pub fn deserialize_as<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        let json = serde_json::to_value(&self.inner).ok()?;
        serde_json::from_value(json).ok()
    }
}

impl IntoIterator for BidsMetadata {
    type Item = (String, Value);
    type IntoIter = indexmap::map::IntoIter<String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl std::ops::Index<&str> for BidsMetadata {
    type Output = Value;

    /// Index into metadata by key.
    ///
    /// # Panics
    ///
    /// Panics if the key is not present. Use [`get()`](Self::get) for a
    /// non-panicking alternative.
    fn index(&self, key: &str) -> &Value {
        &self.inner[key]
    }
}

impl From<IndexMap<String, Value>> for BidsMetadata {
    fn from(map: IndexMap<String, Value>) -> Self {
        Self {
            inner: map,
            source_file: None,
        }
    }
}

impl From<serde_json::Map<String, Value>> for BidsMetadata {
    fn from(map: serde_json::Map<String, Value>) -> Self {
        Self {
            inner: map.into_iter().collect(),
            source_file: None,
        }
    }
}

impl std::fmt::Display for BidsMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BidsMetadata({} keys", self.inner.len())?;
        if let Some(src) = &self.source_file {
            write!(f, " from {src}")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_metadata_typed_accessors() {
        let mut md = BidsMetadata::new();
        md.insert("SamplingFrequency".into(), json!(256.0));
        md.insert("EEGReference".into(), json!("Cz"));
        md.insert("RecordingDuration".into(), json!(600));
        md.insert("EEGGround".into(), json!(true));
        md.insert("TaskName".into(), json!(null));

        assert_eq!(md.get_f64("SamplingFrequency"), Some(256.0));
        assert_eq!(md.get_str("EEGReference"), Some("Cz"));
        assert_eq!(md.get_i64("RecordingDuration"), Some(600));
        assert_eq!(md.get_bool("EEGGround"), Some(true));
        assert!(md.get_str("TaskName").is_none());
        assert!(md.get_f64("Missing").is_none());
    }

    #[test]
    fn test_metadata_extend_overrides() {
        let mut base = BidsMetadata::new();
        base.insert("A".into(), json!(1));
        base.insert("B".into(), json!(2));

        let mut child = BidsMetadata::new();
        child.insert("B".into(), json!(99));
        child.insert("C".into(), json!(3));

        base.extend(child);
        assert_eq!(base.get_i64("A"), Some(1));
        assert_eq!(base.get_i64("B"), Some(99)); // overridden
        assert_eq!(base.get_i64("C"), Some(3));
        assert_eq!(base.len(), 3);
    }

    #[test]
    fn test_metadata_deserialize_as() {
        #[derive(serde::Deserialize)]
        struct EegMeta {
            #[serde(rename = "SamplingFrequency")]
            sampling_frequency: f64,
            #[serde(rename = "EEGReference")]
            eeg_reference: String,
        }

        let mut md = BidsMetadata::new();
        md.insert("SamplingFrequency".into(), json!(256.0));
        md.insert("EEGReference".into(), json!("Cz"));

        let typed: EegMeta = md.deserialize_as().unwrap();
        assert_eq!(typed.sampling_frequency, 256.0);
        assert_eq!(typed.eeg_reference, "Cz");
    }

    #[test]
    fn test_metadata_from_iterator() {
        let md: BidsMetadata = vec![
            ("A".to_string(), json!(1)),
            ("B".to_string(), json!("hello")),
        ]
        .into_iter()
        .collect();
        assert_eq!(md.len(), 2);
        assert_eq!(md.get_i64("A"), Some(1));
    }

    #[test]
    fn test_metadata_source_file() {
        let md = BidsMetadata::with_source("/data/sub-01_eeg.json");
        assert_eq!(md.source_file.as_deref(), Some("/data/sub-01_eeg.json"));
        assert!(md.is_empty());
    }
}
