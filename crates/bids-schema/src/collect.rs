//! Loading a published BIDS schema document.
//!
//! [`BidsSchema::built_in()`](crate::BidsSchema::built_in) is the schema this
//! crate ships and validates against. [`collect_schema()`] is for the other
//! case: reading the official schema JSON for some *other* version, to diff
//! against or to drive tooling.
//!
//! Corresponds to PyBIDS' `bids.utils.collect_schema`.
//!
//! Fetching over the network requires the `http` feature; reading from a local
//! path always works.

use bids_core::error::{BidsError, Result};
use std::path::Path;

/// Where to read a schema document from.
#[derive(Debug, Clone, Copy)]
pub enum SchemaSource<'a> {
    /// A local `schema.json`.
    Path(&'a Path),
    /// An explicit URL. Requires the `http` feature.
    Url(&'a str),
    /// A BIDS version such as `"1.9.0"`, or `"latest"` / `"stable"`.
    /// Requires the `http` feature.
    Version(&'a str),
}

/// Base URL of the published schema documents.
const SCHEMA_BASE: &str = "https://bids-specification.readthedocs.io/en";

/// The published schema URL for a version string.
///
/// Accepts `"1.9.0"`, `"v1.9.0"`, `"latest"` and `"stable"`; a bare version
/// number is normalised to a `v`-prefixed tag, matching the paths the BIDS
/// specification site publishes.
///
/// # Errors
///
/// Returns an error if no version can be read out of `version`.
///
/// # Example
///
/// ```
/// use bids_schema::collect::schema_url;
///
/// assert_eq!(
///     schema_url("1.9.0").unwrap(),
///     "https://bids-specification.readthedocs.io/en/v1.9.0/schema.json"
/// );
/// assert_eq!(
///     schema_url("latest").unwrap(),
///     "https://bids-specification.readthedocs.io/en/latest/schema.json"
/// );
/// ```
pub fn schema_url(version: &str) -> Result<String> {
    let lower = version.to_lowercase();
    let tag = if lower.contains("latest") {
        "latest".to_string()
    } else if lower.contains("stable") {
        "stable".to_string()
    } else {
        let re =
            regex::Regex::new(r"\d+(?:\.\d+){1,}").map_err(|e| BidsError::Config(e.to_string()))?;
        let found = re.find(version).ok_or_else(|| {
            BidsError::Config(format!("unable to determine a version from '{version}'"))
        })?;
        format!("v{}", found.as_str())
    };
    Ok(format!("{SCHEMA_BASE}/{tag}/schema.json"))
}

/// Read a BIDS schema document.
///
/// # Errors
///
/// Returns an error if the source cannot be read, or if the document is not
/// valid JSON. Network sources return an error when the `http` feature is off.
///
/// # Example
///
/// ```no_run
/// use bids_schema::collect::{SchemaSource, collect_schema};
/// use std::path::Path;
///
/// let schema = collect_schema(SchemaSource::Path(Path::new("schema.json"))).unwrap();
/// println!("{}", schema["bids_version"]);
/// ```
pub fn collect_schema(source: SchemaSource<'_>) -> Result<serde_json::Value> {
    let text = match source {
        SchemaSource::Path(p) => std::fs::read_to_string(p)?,
        SchemaSource::Url(url) => fetch(url)?,
        SchemaSource::Version(v) => fetch(&schema_url(v)?)?,
    };
    Ok(serde_json::from_str(&text)?)
}

#[cfg(feature = "http")]
fn fetch(url: &str) -> Result<String> {
    ureq::get(url)
        .call()
        .map_err(|e| BidsError::Config(format!("schema endpoint '{url}' is not reachable: {e}")))?
        .body_mut()
        .read_to_string()
        .map_err(|e| BidsError::Config(format!("could not read schema from '{url}': {e}")))
}

#[cfg(not(feature = "http"))]
fn fetch(url: &str) -> Result<String> {
    Err(BidsError::Config(format!(
        "cannot fetch '{url}': bids-schema was built without the 'http' feature. \
         Enable it, or pass SchemaSource::Path to read a local schema.json."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_strings_map_to_published_urls() {
        assert_eq!(
            schema_url("1.9.0").unwrap(),
            "https://bids-specification.readthedocs.io/en/v1.9.0/schema.json"
        );
        // A `v` prefix is accepted and not doubled.
        assert_eq!(
            schema_url("v1.10.0").unwrap(),
            schema_url("1.10.0").unwrap()
        );
        assert!(
            schema_url("latest")
                .unwrap()
                .ends_with("/latest/schema.json")
        );
        assert!(
            schema_url("stable")
                .unwrap()
                .ends_with("/stable/schema.json")
        );
    }

    #[test]
    fn an_unparseable_version_is_an_error() {
        assert!(schema_url("not-a-version").is_err());
    }

    #[test]
    fn a_local_schema_round_trips() {
        let dir = std::env::temp_dir().join("bids_rs_collect_schema");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("schema.json");
        std::fs::write(&path, r#"{"bids_version": "1.9.0"}"#).unwrap();

        let schema = collect_schema(SchemaSource::Path(&path)).unwrap();
        assert_eq!(schema["bids_version"], "1.9.0");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(not(feature = "http"))]
    #[test]
    fn network_sources_explain_the_missing_feature() {
        let err = collect_schema(SchemaSource::Version("1.9.0")).unwrap_err();
        assert!(err.to_string().contains("'http' feature"));
    }
}
