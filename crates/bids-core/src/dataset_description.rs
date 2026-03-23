//! Typed representation of `dataset_description.json`.
//!
//! Every BIDS dataset root contains a `dataset_description.json` file with
//! mandatory fields (`Name`, `BIDSVersion`) and optional fields (authors,
//! license, funding, etc.). Derivative datasets additionally declare
//! `GeneratedBy` (or the deprecated `PipelineDescription`).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

use crate::error::{BidsError, Result};

/// Contents of a BIDS `dataset_description.json` file.
///
/// See: <https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files.html>
///
/// # Example
///
/// ```
/// use bids_core::DatasetDescription;
///
/// let json = r#"{"Name": "My Dataset", "BIDSVersion": "1.9.0"}"#;
/// let desc: DatasetDescription = serde_json::from_str(json).unwrap();
/// assert_eq!(desc.name, "My Dataset");
/// assert!(!desc.is_derivative());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct DatasetDescription {
    /// Name of the dataset.
    pub name: String,
    /// The version of the BIDS standard used.
    #[serde(rename = "BIDSVersion")]
    pub bids_version: String,
    /// What license the dataset is distributed under.
    #[serde(default)]
    pub license: Option<String>,
    /// List of individuals who contributed to the creation/curation of the dataset.
    #[serde(default)]
    pub authors: Option<Vec<String>>,
    /// Text acknowledging contributions of individuals or institutions.
    #[serde(default)]
    pub acknowledgements: Option<String>,
    /// How to acknowledge this dataset when used in publications.
    #[serde(default)]
    pub how_to_acknowledge: Option<String>,
    /// List of sources of funding.
    #[serde(default)]
    pub funding: Option<Vec<String>>,
    /// List of ethics committee approvals.
    #[serde(default)]
    pub ethics_approvals: Option<Vec<String>>,
    /// List of references to publications about the dataset.
    #[serde(default)]
    pub references_and_links: Option<Vec<String>>,
    /// The DOI of the dataset.
    #[serde(rename = "DatasetDOI", default)]
    pub dataset_doi: Option<String>,
    /// Type of dataset: "raw" or "derivative".
    #[serde(default)]
    pub dataset_type: Option<String>,
    /// Information about the pipeline that generated a derivative dataset.
    #[serde(default)]
    pub generated_by: Option<Vec<GeneratedBy>>,
    /// Datasets that were used to generate this derivative dataset.
    #[serde(default)]
    pub source_datasets: Option<Vec<Value>>,
    /// Legacy field (deprecated in BIDS 1.4.0).
    #[serde(default)]
    pub pipeline_description: Option<PipelineDescription>,
}

/// Information about a pipeline that generated a derivative dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct GeneratedBy {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub code_url: Option<String>,
    #[serde(default)]
    pub container: Option<Value>,
}

/// Legacy pipeline description (deprecated).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PipelineDescription {
    pub name: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

impl DatasetDescription {
    /// Load `dataset_description.json` from the given directory.
    ///
    /// # Errors
    ///
    /// Returns [`BidsError::MissingDatasetDescription`] if the file doesn't exist,
    /// or an I/O or JSON error if the file can't be read or parsed.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("dataset_description.json");
        if !path.exists() {
            return Err(BidsError::MissingDatasetDescription);
        }
        let contents = std::fs::read_to_string(&path)?;
        let desc: Self = serde_json::from_str(&contents)?;
        Ok(desc)
    }

    /// Validate that mandatory fields are present.
    ///
    /// # Errors
    ///
    /// Returns [`BidsError::MissingMandatoryField`] if `Name` or `BIDSVersion`
    /// is empty.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(BidsError::MissingMandatoryField { field: "Name".into() });
        }
        if self.bids_version.is_empty() {
            return Err(BidsError::MissingMandatoryField { field: "BIDSVersion".into() });
        }
        Ok(())
    }

    /// Whether this is a derivative dataset.
    #[must_use]
    pub fn is_derivative(&self) -> bool {
        self.dataset_type.as_deref() == Some("derivative")
    }

    /// Get the pipeline name for derivative datasets.
    #[must_use]
    pub fn pipeline_name(&self) -> Option<&str> {
        // Try GeneratedBy first (BIDS >= 1.4.0)
        if let Some(generated_by) = &self.generated_by
            && let Some(first) = generated_by.first() {
                return Some(&first.name);
            }
        // Fall back to PipelineDescription (deprecated)
        if let Some(pd) = &self.pipeline_description {
            return Some(&pd.name);
        }
        None
    }

    /// Save this description to a `dataset_description.json` file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the directory doesn't exist or the file can't
    /// be written, or a JSON error if serialization fails.
    pub fn save_to(&self, dir: &Path) -> Result<()> {
        let path = dir.join("dataset_description.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

impl std::fmt::Display for DatasetDescription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (BIDS {})", self.name, self.bids_version)?;
        if self.is_derivative() {
            if let Some(pipeline) = self.pipeline_name() {
                write!(f, " [derivative: {pipeline}]")?;
            } else {
                write!(f, " [derivative]")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_description() {
        let json = r#"{
            "Name": "Test Dataset",
            "BIDSVersion": "1.6.0",
            "License": "CC0",
            "Authors": ["Test Author"]
        }"#;
        let desc: DatasetDescription = serde_json::from_str(json).unwrap();
        assert_eq!(desc.name, "Test Dataset");
        assert_eq!(desc.bids_version, "1.6.0");
        assert!(!desc.is_derivative());
    }

    #[test]
    fn test_derivative_description() {
        let json = r#"{
            "Name": "fmriprep",
            "BIDSVersion": "1.6.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "fmriprep", "Version": "20.2.0"}]
        }"#;
        let desc: DatasetDescription = serde_json::from_str(json).unwrap();
        assert!(desc.is_derivative());
        assert_eq!(desc.pipeline_name(), Some("fmriprep"));
    }
}
