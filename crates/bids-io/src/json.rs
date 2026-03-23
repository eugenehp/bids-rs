//! JSON sidecar file handling for BIDS metadata.
//!
//! BIDS uses JSON sidecar files to provide metadata for data files. These
//! sidecars follow the BIDS inheritance principle: a sidecar at a higher
//! directory level applies to all matching data files below it, and can
//! be overridden by more-specific sidecars closer to the data file.
//!
//! This module provides functions for reading, merging, and discovering
//! JSON sidecars.

use bids_core::error::Result;
use bids_core::metadata::BidsMetadata;
use indexmap::IndexMap;
use serde_json::Value;
use std::path::Path;

/// Read a JSON sidecar file and return it as `BidsMetadata`.
///
/// # Errors
///
/// Returns an error if the file can't be read or contains invalid JSON.
pub fn read_json_sidecar(path: &Path) -> Result<BidsMetadata> {
    let contents = std::fs::read_to_string(path)?;
    let map: IndexMap<String, Value> = serde_json::from_str(&contents)?;
    let mut md = BidsMetadata::with_source(&path.to_string_lossy());
    md.update_from_map(map);
    Ok(md)
}

/// Read a JSON file and return it as a generic `serde_json::Value`.
///
/// # Errors
///
/// Returns an error if the file can't be read or contains invalid JSON.
pub fn read_json(path: &Path) -> Result<Value> {
    let contents = std::fs::read_to_string(path)?;
    let val: Value = serde_json::from_str(&contents)?;
    Ok(val)
}

/// Merge JSON sidecars following BIDS inheritance: more-specific files
/// override less-specific ones.
///
/// `sidecars` should be ordered from most specific (closest to data file)
/// to least specific (dataset root).
///
/// # Errors
///
/// Returns an error if any sidecar file can't be read or parsed.
pub fn merge_json_sidecars(sidecars: &[&Path]) -> Result<BidsMetadata> {
    let mut merged = BidsMetadata::new();
    // Process from least specific to most specific, so specific values win
    for path in sidecars.iter().rev() {
        let md = read_json_sidecar(path)?;
        merged.extend(md);
    }
    Ok(merged)
}

/// Find all applicable JSON sidecar files for a given data file,
/// walking up the directory tree (BIDS inheritance principle).
///
/// Returns paths ordered from most specific to least specific.
pub fn find_sidecars(data_file: &Path, root: &Path) -> Vec<std::path::PathBuf> {
    let mut sidecars = Vec::new();

    // Get the suffix from the data file
    let stem = data_file.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    // Handle .tsv.gz double extension
    let stem = stem.strip_suffix(".tsv").unwrap_or(stem);

    let suffix = stem.rsplit('_').next().unwrap_or("");
    if suffix.is_empty() {
        return sidecars;
    }

    // Walk from the data file's directory up to root
    let mut dir = data_file.parent();
    while let Some(current_dir) = dir {
        // Look for JSON files in this directory that might match
        if let Ok(entries) = std::fs::read_dir(current_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "json") {
                    let json_stem = path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    let json_suffix = json_stem.rsplit('_').next().unwrap_or("");

                    if json_suffix == suffix && is_sidecar_for(&path, data_file) {
                        sidecars.push(path);
                    }
                }
            }
        }

        if current_dir == root {
            break;
        }
        dir = current_dir.parent();
    }

    sidecars
}

/// Check whether a JSON sidecar applies to a data file.
///
/// A sidecar applies if all key-value entity pairs in its filename (parts
/// containing `-`) also appear in the data file's filename. Suffix parts
/// (without `-`) must match when both are present.
fn is_sidecar_for(sidecar: &Path, data_file: &Path) -> bool {
    let sc_stem = sidecar.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let df_stem = data_file.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    // Handle .tsv.gz double extension
    let df_stem = df_stem.strip_suffix(".tsv").unwrap_or(df_stem);

    let sc_parts: Vec<&str> = sc_stem.split('_').collect();
    let df_parts: Vec<&str> = df_stem.split('_').collect();

    for part in &sc_parts {
        if part.contains('-') {
            // Entity key-value pair must appear in the data file
            if !df_parts.contains(part) {
                return false;
            }
        }
        // Suffix parts (no '-') are implicitly matched by find_sidecars()
        // which already filters by suffix, so no extra check needed here.
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_json_sidecar() {
        let dir = std::env::temp_dir().join("bids_io_json_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sub-01_task-rest_eeg.json");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(f, r#"{{"SamplingFrequency": 256, "EEGReference": "Cz"}}"#).unwrap();

        let md = read_json_sidecar(&path).unwrap();
        assert_eq!(md.get_f64("SamplingFrequency"), Some(256.0));
        assert_eq!(md.get_str("EEGReference"), Some("Cz"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
