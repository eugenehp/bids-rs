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

/// BIDS' sentinel for a value that exists in the schema but is unavailable here.
const NA: &str = "n/a";

/// Recursively replace BIDS' `"n/a"` sentinel with JSON `null`.
///
/// BIDS permits `"n/a"` wherever a value is unknown, including for fields the
/// specification types as numeric — `"PowerLineFrequency": "n/a"` is valid and
/// appears in real datasets. Deserializing that into an `Option<f64>` fails with
/// `invalid type: string "n/a", expected f64`, which is not what the dataset
/// means: `"n/a"` means absent, and absent is `None`.
///
/// Mapping it to `null` lets every `Option<T>` field take it, for every modality
/// struct at once, without annotating each field. It matches what this codebase
/// already does elsewhere — the TSV readers turn `n/a` cells into empty strings,
/// and the electrode/channel parsers filter `"n/a"` out by hand.
///
/// Applied only on the typed path. The untyped accessors ([`BidsMetadata::get`],
/// [`BidsMetadata::get_str`], ...) still return the sidecar verbatim, so
/// anything comparing raw metadata against PyBIDS keeps seeing `"n/a"`.
fn na_to_null(value: Value) -> Value {
    match value {
        Value::String(ref s) if s == NA => Value::Null,
        Value::Array(items) => Value::Array(items.into_iter().map(na_to_null).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(k, v)| (k, na_to_null(v)))
                .collect(),
        ),
        other => other,
    }
}

impl BidsMetadata {
    /// Try to deserialize this metadata into a typed struct.
    ///
    /// Works via JSON roundtrip: metadata map → `serde_json::Value` → `T`.
    /// BIDS' `"n/a"` sentinel is treated as absent — see
    /// [`try_deserialize_as`](Self::try_deserialize_as) for the details and for
    /// the error when this returns `None`.
    pub fn deserialize_as<T: serde::de::DeserializeOwned>(&self) -> Option<T> {
        self.try_deserialize_as().ok()
    }

    /// Deserialize this metadata into a typed struct, reporting why on failure.
    ///
    /// Same conversion as [`deserialize_as`](Self::deserialize_as), which
    /// discards the error and yields `None`. That is a poor way to debug a
    /// sidecar: one unparseable field drops the whole struct, and the caller
    /// cannot see which key was at fault. Prefer this when you need to know.
    ///
    /// Any `"n/a"` in the metadata — at any depth — is converted to `null`
    /// first, so fields the BIDS schema types as numeric still deserialize when
    /// a dataset marks them unavailable. `Option<T>` fields take it as `None`;
    /// a *required* field set to `"n/a"` is still an error, which is correct —
    /// it genuinely has no value.
    ///
    /// # Errors
    ///
    /// Returns the `serde_json` error if the metadata cannot be represented as
    /// JSON, or does not fit `T`.
    ///
    /// The error describes the mismatch (`invalid type: null, expected f64`)
    /// but does not name the key — `serde_json::from_value` carries no field
    /// path. Note that a `"n/a"` reaching a required field surfaces as
    /// `null`, since that is what it was rewritten to. Wiring up
    /// `serde_path_to_error` here would recover the key name, at the cost of a
    /// new dependency in every crate that builds on this one.
    ///
    /// # Example
    ///
    /// ```
    /// use bids_core::metadata::BidsMetadata;
    /// use serde::Deserialize;
    /// use serde_json::json;
    ///
    /// #[derive(Deserialize)]
    /// #[serde(rename_all = "PascalCase")]
    /// struct Eeg {
    ///     sampling_frequency: f64,
    ///     #[serde(default)]
    ///     power_line_frequency: Option<f64>,
    /// }
    ///
    /// let mut md = BidsMetadata::new();
    /// md.insert("SamplingFrequency".into(), json!(500.0));
    /// md.insert("PowerLineFrequency".into(), json!("n/a"));
    ///
    /// let eeg: Eeg = md.try_deserialize_as().unwrap();
    /// assert_eq!(eeg.sampling_frequency, 500.0);
    /// assert_eq!(eeg.power_line_frequency, None);
    /// ```
    pub fn try_deserialize_as<T: serde::de::DeserializeOwned>(
        &self,
    ) -> std::result::Result<T, serde_json::Error> {
        let json = serde_json::to_value(&self.inner)?;
        serde_json::from_value(na_to_null(json))
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

    // ── BIDS' "n/a" sentinel ──────────────────────────────────────────────

    #[derive(Debug, serde::Deserialize)]
    #[serde(rename_all = "PascalCase")]
    struct NaProbe {
        sampling_frequency: f64,
        #[serde(default)]
        power_line_frequency: Option<f64>,
        #[serde(rename = "EEGChannelCount", default)]
        eeg_channel_count: Option<u32>,
        #[serde(default)]
        manufacturer: Option<String>,
    }

    /// Regression: `"PowerLineFrequency": "n/a"` is valid BIDS and appears in
    /// real datasets (ds005420). It used to fail the whole struct with
    /// `invalid type: string "n/a", expected f64`.
    #[test]
    fn na_in_a_numeric_field_deserializes_as_none() {
        let mut md = BidsMetadata::new();
        md.insert("SamplingFrequency".into(), json!(500.0));
        md.insert("PowerLineFrequency".into(), json!("n/a"));
        md.insert("EEGChannelCount".into(), json!(20));
        md.insert("Manufacturer".into(), json!("n/a"));

        let typed: NaProbe = md.deserialize_as().expect("n/a must not sink the struct");
        assert_eq!(typed.sampling_frequency, 500.0);
        assert_eq!(typed.power_line_frequency, None);
        assert_eq!(typed.eeg_channel_count, Some(20));
        // "n/a" means absent for strings too, matching what the TSV readers and
        // the electrode/channel parsers already do.
        assert_eq!(typed.manufacturer, None);
    }

    #[test]
    fn na_is_rewritten_at_any_depth() {
        let mut md = BidsMetadata::new();
        md.insert(
            "Nested".into(),
            json!({ "inner": { "value": "n/a" }, "list": ["n/a", 1] }),
        );
        let out: serde_json::Value = md.try_deserialize_as().unwrap();
        assert_eq!(out["Nested"]["inner"]["value"], Value::Null);
        assert_eq!(out["Nested"]["list"][0], Value::Null);
        assert_eq!(out["Nested"]["list"][1], json!(1));
    }

    /// `"n/a"` is only rewritten on the typed path — raw accessors still hand
    /// back what the sidecar said, which is what PyBIDS-parity tests compare.
    #[test]
    fn na_survives_on_the_untyped_path() {
        let mut md = BidsMetadata::new();
        md.insert("SoftwareFilters".into(), json!("n/a"));
        assert_eq!(md.get_str("SoftwareFilters"), Some("n/a"));
        assert_eq!(md.get("SoftwareFilters"), Some(&json!("n/a")));
    }

    /// A required field set to `"n/a"` genuinely has no value, so it must still
    /// fail — and `try_deserialize_as` has to say why.
    #[test]
    fn na_in_a_required_field_still_errors_with_a_reason() {
        let mut md = BidsMetadata::new();
        md.insert("SamplingFrequency".into(), json!("n/a"));

        assert!(md.deserialize_as::<NaProbe>().is_none());

        // The error describes the mismatch but not which key caused it —
        // `serde_json::from_value` does not carry a field path. Still strictly
        // more than the bare `None` that `deserialize_as` gives back. Note the
        // `null`: that is the rewritten `"n/a"`.
        let err = md
            .try_deserialize_as::<NaProbe>()
            .expect_err("required field must not be silently defaulted");
        assert!(
            err.to_string().contains("expected f64"),
            "error should describe the mismatch, got: {err}"
        );
    }
}
