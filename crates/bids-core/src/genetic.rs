//! Genetic descriptor files (`genetic_info.json`, `genetic_database.json`).
//!
//! BIDS datasets may include genetic descriptor files at the root level
//! describing the genetic data associated with the dataset. See the
//! [BIDS specification appendix](https://bids-specification.readthedocs.io/en/stable/appendices/genetic-database.html).
//!
//! # Example
//!
//! ```
//! use bids_core::genetic::GeneticInfo;
//!
//! let json = r#"{"Dataset": "OpenNeuro", "Genetics": {"Database": "https://example.com", "Descriptors": "APOE"}}"#;
//! let info: GeneticInfo = serde_json::from_str(json).unwrap();
//! assert_eq!(info.genetics.as_ref().unwrap().database.as_deref(), Some("https://example.com"));
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::Result;

/// Contents of `genetic_info.json`.
///
/// Describes the genetic information associated with participants in the
/// dataset. This is a root-level file that links to external genetic databases.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct GeneticInfo {
    /// Name of the dataset.
    #[serde(default)]
    pub dataset: Option<String>,
    /// Genetic database reference information.
    #[serde(default)]
    pub genetics: Option<GeneticsField>,
}

/// Genetic database reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct GeneticsField {
    /// URI of the genetic database.
    #[serde(default)]
    pub database: Option<String>,
    /// Descriptors of the genetic data (e.g., genotype, SNP, allele).
    #[serde(default)]
    pub descriptors: Option<serde_json::Value>,
}

/// Contents of `genetic_database.json`.
///
/// Maps participant IDs to genetic database identifiers, enabling linkage
/// between BIDS participant data and external genetic resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticDatabase {
    /// Mapping of `participant_id` → database record ID.
    #[serde(flatten)]
    pub entries: std::collections::HashMap<String, serde_json::Value>,
}

impl GeneticInfo {
    /// Load `genetic_info.json` from a dataset root directory.
    ///
    /// Returns `Ok(None)` if the file doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but can't be read or parsed.
    pub fn from_dir(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("genetic_info.json");
        if !path.exists() {
            return Ok(None);
        }
        let contents = std::fs::read_to_string(&path)?;
        let info: Self = serde_json::from_str(&contents)?;
        Ok(Some(info))
    }
}

impl GeneticDatabase {
    /// Load `genetic_database.json` from a dataset root directory.
    ///
    /// Returns `Ok(None)` if the file doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but can't be read or parsed.
    pub fn from_dir(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("genetic_database.json");
        if !path.exists() {
            return Ok(None);
        }
        let contents = std::fs::read_to_string(&path)?;
        let db: Self = serde_json::from_str(&contents)?;
        Ok(Some(db))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genetic_info_parse() {
        let json = r#"{
            "Dataset": "Example",
            "Genetics": {
                "Database": "https://www.ncbi.nlm.nih.gov/gap/",
                "Descriptors": ["APOE", "BDNF"]
            }
        }"#;
        let info: GeneticInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.dataset.as_deref(), Some("Example"));
        let genetics = info.genetics.unwrap();
        assert!(genetics.database.as_ref().unwrap().contains("ncbi"));
    }

    #[test]
    fn test_genetic_database_parse() {
        let json = r#"{
            "sub-01": {"sample_id": "GENO_001"},
            "sub-02": {"sample_id": "GENO_002"}
        }"#;
        let db: GeneticDatabase = serde_json::from_str(json).unwrap();
        assert_eq!(db.entries.len(), 2);
        assert!(db.entries.contains_key("sub-01"));
    }

    #[test]
    fn test_genetic_info_missing_file() {
        let dir = std::env::temp_dir().join("bids_genetic_test_missing");
        std::fs::create_dir_all(&dir).unwrap();
        let result = GeneticInfo::from_dir(&dir).unwrap();
        assert!(result.is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
