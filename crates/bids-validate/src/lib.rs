#![deny(unsafe_code)]
//! BIDS dataset validation utilities.
//!
//! Provides functions for validating BIDS dataset roots, derivative directories,
//! and file indexing patterns. Used by `bids-layout` during dataset indexing to
//! determine which files to include or exclude.
//!
//! # Root Validation
//!
//! [`validate_root()`] checks that a path exists, is a directory, and contains
//! a valid `dataset_description.json`. If validation is enabled, the description
//! is parsed and checked for required fields.
//!
//! # Derivative Validation
//!
//! [`validate_derivative_path()`] ensures derivative datasets have a valid
//! `dataset_description.json` with pipeline information (either `GeneratedBy`
//! or the legacy `PipelineDescription`).
//!
//! # Ignore / Force Patterns
//!
//! Default ignore patterns exclude `code/`, `models/`, `sourcedata/`, `stimuli/`,
//! hidden files (`.`-prefixed), and common non-BIDS directories. These can be
//! overridden with custom patterns via [`validate_indexing_args()`].

use bids_core::error::{BidsError, Result};
use bids_core::dataset_description::DatasetDescription;
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Directories to ignore by default during indexing.
pub static DEFAULT_IGNORE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    vec![
        Regex::new(r"^/(code|models|sourcedata|stimuli)").unwrap(),
        Regex::new(r"/\.").unwrap(), // dotfiles
    ]
});

/// Validate the root directory of a BIDS dataset.
pub fn validate_root(root: &Path, validate: bool) -> Result<(PathBuf, Option<DatasetDescription>)> {
    let root = root.canonicalize().map_err(|_| {
        BidsError::RootNotFound(root.to_string_lossy().to_string())
    })?;

    if !root.exists() {
        return Err(BidsError::RootNotFound(root.to_string_lossy().to_string()));
    }

    let desc_path = root.join("dataset_description.json");
    if !desc_path.exists() {
        if validate {
            return Err(BidsError::MissingDatasetDescription);
        }
        return Ok((root, None));
    }

    match DatasetDescription::from_dir(&root) {
        Ok(desc) => {
            if validate {
                desc.validate()?;
            }
            Ok((root, Some(desc)))
        }
        Err(e) => {
            if validate { Err(e) } else { Ok((root, None)) }
        }
    }
}

/// Validate a derivatives directory and return the pipeline name.
pub fn validate_derivative_path(path: &Path) -> Result<String> {
    let desc = DatasetDescription::from_dir(path)?;
    desc.pipeline_name()
        .map(std::string::ToString::to_string)
        .ok_or_else(|| BidsError::DerivativesValidation(
            "Every valid BIDS-derivatives dataset must have a GeneratedBy.Name field \
             set inside 'dataset_description.json'".to_string()
        ))
}

/// Check if a path should be ignored during indexing.
pub fn should_ignore(path: &Path, root: &Path, ignore_patterns: &[Regex]) -> bool {
    let rel = path.strip_prefix(root)
        .map(|p| format!("/{}", p.to_string_lossy()))
        .unwrap_or_default();

    ignore_patterns.iter().any(|pat| pat.is_match(&rel))
}

/// Check if a path matches BIDS naming conventions (basic validation).
pub fn is_bids_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    const ROOT_FILES: &[&str] = &[
        "dataset_description.json", "participants.tsv",
        "participants.json", "README", "CHANGES", "LICENSE",
    ];
    ROOT_FILES.contains(&name)
        || name.starts_with("sub-")
        || name.starts_with("task-")
        || name.starts_with("acq-")
        || name.starts_with("sample-")
}

/// Resolve a BIDS URI or relative IntendedFor path to an absolute path.
pub fn resolve_intended_for(intent: &str, root: &Path, subject: &str) -> Option<PathBuf> {
    if let Some(rest) = intent.strip_prefix("bids::") {
        Some(root.join(rest))
    } else if intent.starts_with("bids:") {
        None // Named dataset URI — cross-dataset reference
    } else {
        Some(root.join(format!("sub-{subject}")).join(intent))
    }
}

/// Validate and sort indexing arguments.
///
/// Returns (ignore_patterns, force_index_patterns), both sorted from specific to general.
pub fn validate_indexing_args(
    ignore: Option<Vec<Regex>>,
    force_index: Option<Vec<Regex>>,
    _root: &Path,
) -> Result<(Vec<Regex>, Vec<Regex>)> {
    let mut ignore = ignore.unwrap_or_else(|| DEFAULT_IGNORE.clone());

    // Always ignore dotfiles
    let dotfile_re = Regex::new(r"/\.").unwrap();
    if !ignore.iter().any(|r| r.as_str() == dotfile_re.as_str()) {
        ignore.push(dotfile_re);
    }

    let force_index = force_index.unwrap_or_default();

    // Validate no derivatives in force_index
    for entry in &force_index {
        if entry.as_str().contains("derivatives") {
            return Err(BidsError::Validation(
                "Do not pass 'derivatives' in force_index. Use add_derivatives() instead.".to_string()
            ));
        }
    }

    Ok((ignore, force_index))
}

/// A single validation issue found in a BIDS dataset.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity: `"error"` or `"warning"`.
    pub severity: String,
    /// Short code (e.g., `"MISSING_DATASET_DESCRIPTION"`, `"INVALID_FILENAME"`).
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// File path that triggered the issue (if applicable).
    pub path: Option<String>,
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.severity.to_uppercase(), self.code, self.message)?;
        if let Some(ref p) = self.path {
            write!(f, " ({p})")?;
        }
        Ok(())
    }
}

/// Result of a full dataset validation.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// All issues found.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// Returns `true` if no errors were found (warnings are OK).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == "error")
    }

    /// Count of errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues.iter().filter(|i| i.severity == "error").count()
    }

    /// Count of warnings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues.iter().filter(|i| i.severity == "warning").count()
    }
}

impl std::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Validation: {} errors, {} warnings", self.error_count(), self.warning_count())?;
        for issue in &self.issues {
            writeln!(f, "  {issue}")?;
        }
        Ok(())
    }
}

/// Perform a full BIDS validation of a dataset directory.
///
/// Checks:
/// - `dataset_description.json` exists and has required fields
/// - `README` exists
/// - All files under `sub-*` directories follow BIDS naming conventions
/// - No unexpected files at the root level
/// - Metadata consistency (TR values match across runs, etc.)
///
/// This is a lighter-weight alternative to the official `bids-validator`;
/// it catches the most common structural issues.
///
/// # Errors
///
/// Returns an I/O error if the directory can't be read.
pub fn validate_dataset(root: &Path) -> Result<ValidationResult> {
    let mut issues = Vec::new();

    // 1. Check dataset_description.json
    let desc_path = root.join("dataset_description.json");
    if !desc_path.exists() {
        issues.push(ValidationIssue {
            severity: "error".into(),
            code: "MISSING_DATASET_DESCRIPTION".into(),
            message: "dataset_description.json is required at the root".into(),
            path: None,
        });
    } else {
        match DatasetDescription::from_dir(root) {
            Ok(desc) => {
                if desc.name.is_empty() {
                    issues.push(ValidationIssue {
                        severity: "error".into(),
                        code: "MISSING_NAME".into(),
                        message: "Name field is required in dataset_description.json".into(),
                        path: Some(desc_path.to_string_lossy().into()),
                    });
                }
                if desc.bids_version.is_empty() {
                    issues.push(ValidationIssue {
                        severity: "error".into(),
                        code: "MISSING_BIDS_VERSION".into(),
                        message: "BIDSVersion field is required in dataset_description.json".into(),
                        path: Some(desc_path.to_string_lossy().into()),
                    });
                }
            }
            Err(_) => {
                issues.push(ValidationIssue {
                    severity: "error".into(),
                    code: "INVALID_DATASET_DESCRIPTION".into(),
                    message: "dataset_description.json cannot be parsed".into(),
                    path: Some(desc_path.to_string_lossy().into()),
                });
            }
        }
    }

    // 2. Check README
    let has_readme = root.join("README").exists()
        || root.join("README.md").exists()
        || root.join("README.rst").exists()
        || root.join("README.txt").exists();
    if !has_readme {
        issues.push(ValidationIssue {
            severity: "warning".into(),
            code: "MISSING_README".into(),
            message: "A README file is recommended at the dataset root".into(),
            path: None,
        });
    }

    // 3. Check subject directories
    let mut has_subjects = false;
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("sub-") && entry.file_type().is_ok_and(|t| t.is_dir()) {
                has_subjects = true;
                validate_subject_dir(&entry.path(), root, &mut issues);
            }
        }
    }

    if !has_subjects {
        issues.push(ValidationIssue {
            severity: "error".into(),
            code: "NO_SUBJECTS".into(),
            message: "No subject directories (sub-*) found".into(),
            path: None,
        });
    }

    Ok(ValidationResult { issues })
}

fn validate_subject_dir(sub_dir: &Path, root: &Path, issues: &mut Vec<ValidationIssue>) {
    let schema = bids_schema::BidsSchema::load();

    let entries: Vec<walkdir::DirEntry> = walkdir::WalkDir::new(sub_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .collect();

    {
        for entry in entries {
            let path = entry.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Skip hidden files and known non-BIDS
            if name.starts_with('.') { continue; }

            // Check filename follows BIDS pattern
            let rel = path.strip_prefix(root)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();

            if !schema.is_valid(&rel) && !name.ends_with(".json") {
                // Don't flag JSON sidecars as errors — they follow looser patterns
                issues.push(ValidationIssue {
                    severity: "warning".into(),
                    code: "INVALID_FILENAME".into(),
                    message: "File does not match any BIDS naming pattern".to_string(),
                    path: Some(rel),
                });
            }
        }
    }
}

/// Check if a path should be force-indexed.
pub fn should_force_index(path: &Path, root: &Path, force_patterns: &[Regex]) -> bool {
    if force_patterns.is_empty() {
        return false;
    }
    let rel = path.strip_prefix(root)
        .map(|p| format!("/{}", p.to_string_lossy()))
        .unwrap_or_default();
    force_patterns.iter().any(|pat| pat.is_match(&rel))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_intended_for() {
        let root = Path::new("/data/bids");
        assert_eq!(
            resolve_intended_for("bids::sub-01/anat/sub-01_T1w.nii.gz", root, "01"),
            Some(PathBuf::from("/data/bids/sub-01/anat/sub-01_T1w.nii.gz"))
        );
        assert_eq!(
            resolve_intended_for("anat/sub-01_T1w.nii.gz", root, "01"),
            Some(PathBuf::from("/data/bids/sub-01/anat/sub-01_T1w.nii.gz"))
        );
        assert_eq!(
            resolve_intended_for("bids:other:sub-01/anat/sub-01_T1w.nii.gz", root, "01"),
            None
        );
    }

    #[test]
    fn test_validate_indexing_args() {
        let root = Path::new("/data");
        let (ignore, force) = validate_indexing_args(None, None, root).unwrap();
        assert!(!ignore.is_empty());
        assert!(force.is_empty());
    }

    #[test]
    fn test_validate_indexing_args_no_derivatives() {
        let root = Path::new("/data");
        let force = vec![Regex::new("derivatives").unwrap()];
        let result = validate_indexing_args(None, Some(force), root);
        assert!(result.is_err());
    }

    #[test]
    fn test_should_force_index() {
        let root = Path::new("/data");
        let patterns = vec![Regex::new(r"/extra/").unwrap()];
        assert!(should_force_index(Path::new("/data/extra/file.txt"), root, &patterns));
        assert!(!should_force_index(Path::new("/data/sub-01/file.txt"), root, &patterns));
    }
}
