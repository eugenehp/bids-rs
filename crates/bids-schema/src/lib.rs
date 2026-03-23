#![deny(unsafe_code)]
//! BIDS specification schema loader and version tracking.
//!
//! Provides entity definitions, valid datatypes/suffixes/extensions,
//! BIDS filename validation, and specification version management —
//! replacing bidsschematools + bids-validator.
//!
//! # Spec Version Tracking
//!
//! All spec-derived knowledge is concentrated in this crate and in
//! `bids-core/src/configs/`. When the BIDS spec releases a new version:
//!
//! 1. Add a [`SpecChange`](version::SpecChange) entry to
//!    [`version::CHANGELOG`].
//! 2. Update [`version::SUPPORTED_BIDS_VERSION`].
//! 3. Update `BidsSchema::built_in()` with new entities/datatypes/suffixes.
//! 4. Update `bids-core/src/configs/bids.json` with new patterns.
//! 5. Run `cargo test --workspace` to catch regressions.
//!
//! See the [`version`] module for full documentation on the migration process.

pub mod version;

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub use version::{BidsVersion, Compatibility, SUPPORTED_BIDS_VERSION, MIN_COMPATIBLE_VERSION};

/// A BIDS entity definition from the schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityDef {
    pub name: String,
    /// The BIDS key prefix (e.g., "sub" for subject, "ses" for session).
    pub key: String,
    /// Format: "label", "index", etc.
    pub format: String,
    /// Whether this entity creates a directory level.
    pub is_directory: bool,
}

/// Full BIDS schema.
#[derive(Debug, Clone)]
pub struct BidsSchema {
    pub version: String,
    pub entities: Vec<EntityDef>,
    pub datatypes: HashSet<String>,
    pub suffixes: HashSet<String>,
    pub extensions: HashSet<String>,
    /// Filename validation patterns: datatype → vec of regex.
    pub file_patterns: HashMap<String, Vec<Regex>>,
}

impl BidsSchema {
    /// Load the bundled default schema.
    pub fn load() -> Self {
        Self::built_in()
    }

    /// Check compatibility between this schema and a dataset's declared BIDS version.
    ///
    /// # Example
    ///
    /// ```
    /// use bids_schema::{BidsSchema, BidsVersion};
    ///
    /// let schema = BidsSchema::load();
    /// let compat = schema.check_dataset_version("1.8.0");
    /// assert!(compat.is_ok());
    /// ```
    #[must_use]
    pub fn check_dataset_version(&self, dataset_version_str: &str) -> Compatibility {
        match BidsVersion::parse(dataset_version_str) {
            Some(dv) => {
                let lib_ver = BidsVersion::parse(&self.version)
                    .unwrap_or(SUPPORTED_BIDS_VERSION);
                lib_ver.check_compatibility(&dv)
            }
            None => Compatibility::Incompatible {
                reason: format!("Cannot parse BIDS version: '{dataset_version_str}'"),
            },
        }
    }

    /// Built-in schema derived from BIDS 1.9.0 specification.
    ///
    /// **Maintainer note:** When updating to a new BIDS spec version, update:
    /// - The entity, datatype, suffix, and extension lists below
    /// - The `version` field
    /// - [`version::SUPPORTED_BIDS_VERSION`]
    /// - [`version::CHANGELOG`]
    /// - `bids-core/src/configs/bids.json`
    fn built_in() -> Self {
        let entities = vec![
            ent("subject", "sub", "label", true),
            ent("session", "ses", "label", true),
            ent("sample", "sample", "label", false),
            ent("task", "task", "label", false),
            ent("tracksys", "tracksys", "label", false),
            ent("acquisition", "acq", "label", false),
            ent("ceagent", "ce", "label", false),
            ent("staining", "stain", "label", false),
            ent("tracer", "trc", "label", false),
            ent("reconstruction", "rec", "label", false),
            ent("direction", "dir", "label", false),
            ent("run", "run", "index", false),
            ent("modality", "mod", "label", false),
            ent("echo", "echo", "index", false),
            ent("flip", "flip", "index", false),
            ent("inversion", "inv", "index", false),
            ent("mtransfer", "mt", "label", false),
            ent("part", "part", "label", false),
            ent("processing", "proc", "label", false),
            ent("hemisphere", "hemi", "label", false),
            ent("space", "space", "label", false),
            ent("split", "split", "index", false),
            ent("recording", "recording", "label", false),
            ent("chunk", "chunk", "index", false),
            ent("atlas", "atlas", "label", false),
            ent("resolution", "res", "label", false),
            ent("density", "den", "label", false),
            ent("label", "label", "label", false),
            ent("description", "desc", "label", false),
        ];

        let datatypes: HashSet<String> = [
            "anat", "beh", "dwi", "eeg", "fmap", "func", "ieeg", "meg",
            "micr", "motion", "mrs", "nirs", "perf", "pet",
        ].iter().map(std::string::ToString::to_string).collect();

        let suffixes: HashSet<String> = [
            "T1w", "T2w", "T2star", "FLAIR", "PD", "PDT2", "inplaneT1", "inplaneT2",
            "angio", "defacemask", "bold", "cbv", "sbref", "phase", "dwi",
            "phasediff", "magnitude1", "magnitude2", "phase1", "phase2", "fieldmap", "epi",
            "events", "physio", "stim", "channels", "electrodes", "coordsystem",
            "eeg", "ieeg", "meg", "headshape", "photo",
            "pet", "blood", "asl", "m0scan", "aslcontext", "asllabeling",
            "motion", "nirs", "optodes",
            "svs", "mrsi", "unloc", "mrsref",
            "TEM", "SEM", "uCT", "BF", "DF", "PC", "DIC", "FLUO", "CONF",
            "participants", "scans", "sessions", "regressors", "timeseries",
        ].iter().map(std::string::ToString::to_string).collect();

        let extensions: HashSet<String> = [
            ".nii", ".nii.gz", ".json", ".tsv", ".tsv.gz",
            ".bval", ".bvec",
            ".edf", ".bdf", ".set", ".fdt", ".vhdr", ".vmrk", ".eeg",
            ".fif", ".dat", ".pos", ".sqd", ".con", ".ds",
            ".snirf",
            ".mefd", ".nwb",
            ".png", ".tif", ".ome.tif", ".ome.btf", ".jpg",
        ].iter().map(std::string::ToString::to_string).collect();

        // Core filename validation patterns
        let mut file_patterns: HashMap<String, Vec<Regex>> = HashMap::new();
        let sub = r"sub-[a-zA-Z0-9]+";
        let ses = r"(?:_ses-[a-zA-Z0-9]+)?";
        let entities_pat = r"(?:_[a-z]+-[a-zA-Z0-9]+)*";

        // Pattern with optional ses- directory AND ses- entity in filename
        let ses_dir = r"(?:/ses-[a-zA-Z0-9]+)?";
        for dt in &datatypes {
            let pat = format!(
                r"^{sub}{ses_dir}/{dt}/{sub}{ses}{entities_pat}_[a-zA-Z0-9]+\.[a-zA-Z0-9.]+$");
            if let Ok(re) = Regex::new(&pat) {
                file_patterns.entry(dt.clone()).or_default().push(re);
            }
        }

        // Root-level files
        let root_patterns = vec![
            Regex::new(r"^participants\.tsv$").unwrap(),
            Regex::new(r"^participants\.json$").unwrap(),
            Regex::new(r"^dataset_description\.json$").unwrap(),
            Regex::new(r"^README.*$").unwrap(),
            Regex::new(r"^CHANGES$").unwrap(),
            Regex::new(r"^LICENSE$").unwrap(),
            // Task/acq-level sidecars
            Regex::new(r"^(?:task-[a-zA-Z0-9]+_)?(?:acq-[a-zA-Z0-9]+_)?[a-zA-Z0-9]+\.json$").unwrap(),
        ];
        file_patterns.insert("root".into(), root_patterns);

        // Scans files
        let scans_pat = Regex::new(&format!(r"^{sub}(?:/ses-[a-zA-Z0-9]+)?/{sub}(?:_ses-[a-zA-Z0-9]+)?_scans\.tsv$")).unwrap();
        file_patterns.entry("scans".into()).or_default().push(scans_pat);

        // Session files
        let ses_pat = Regex::new(&format!(r"^{sub}/{sub}_sessions\.tsv$")).unwrap();
        file_patterns.entry("sessions".into()).or_default().push(ses_pat);

        Self { version: SUPPORTED_BIDS_VERSION.to_string(), entities, datatypes, suffixes, extensions, file_patterns }
    }

    /// Validate a relative file path against BIDS naming rules.
    pub fn is_valid(&self, relative_path: &str) -> bool {
        let path = relative_path.trim_start_matches('/');

        // Check root-level files
        if !path.contains('/') {
            return self.file_patterns.get("root")
                .is_some_and(|pats| pats.iter().any(|p| p.is_match(path)));
        }

        // Check all datatype patterns
        for patterns in self.file_patterns.values() {
            if patterns.iter().any(|p| p.is_match(path)) {
                return true;
            }
        }
        false
    }

    /// Get entity definition by name.
    pub fn get_entity(&self, name: &str) -> Option<&EntityDef> {
        self.entities.iter().find(|e| e.name == name)
    }

    /// Get entity definition by BIDS key (e.g., "sub", "ses").
    pub fn get_entity_by_key(&self, key: &str) -> Option<&EntityDef> {
        self.entities.iter().find(|e| e.key == key)
    }

    /// Check if a datatype is valid.
    pub fn is_valid_datatype(&self, dt: &str) -> bool { self.datatypes.contains(dt) }

    /// Check if a suffix is valid.
    pub fn is_valid_suffix(&self, s: &str) -> bool { self.suffixes.contains(s) }

    /// Check if an extension is valid.
    pub fn is_valid_extension(&self, e: &str) -> bool { self.extensions.contains(e) }

    /// Generate a regex pattern string for an entity.
    pub fn entity_pattern(&self, name: &str) -> Option<String> {
        let ent = self.get_entity(name)?;
        let value_pattern = match ent.format.as_str() {
            "index" => r"\d+",
            _ => r"[a-zA-Z0-9]+",
        };
        if ent.is_directory {
            Some(format!(r"[/\\]+{}-({value})", ent.key, value = value_pattern))
        } else {
            Some(format!(r"[_/\\]+{}-({value})", ent.key, value = value_pattern))
        }
    }
}

fn ent(name: &str, key: &str, format: &str, is_dir: bool) -> EntityDef {
    EntityDef { name: name.into(), key: key.into(), format: format.into(), is_directory: is_dir }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_schema() {
        let schema = BidsSchema::load();
        assert_eq!(schema.version, "1.9.0");
        assert!(schema.entities.len() >= 25);
        assert!(schema.datatypes.contains("eeg"));
        assert!(schema.datatypes.contains("func"));
        assert!(schema.suffixes.contains("bold"));
        assert!(schema.extensions.contains(".nii.gz"));
    }

    #[test]
    fn test_entity_pattern() {
        let schema = BidsSchema::load();
        let pat = schema.entity_pattern("subject").unwrap();
        assert!(pat.contains("sub-"));
        let pat = schema.entity_pattern("run").unwrap();
        assert!(pat.contains(r"\d+"));
    }

    #[test]
    fn test_is_valid() {
        let schema = BidsSchema::load();
        assert!(schema.is_valid("participants.tsv"));
        assert!(schema.is_valid("dataset_description.json"));
        assert!(schema.is_valid("sub-01/eeg/sub-01_task-rest_eeg.edf"));
        assert!(schema.is_valid("sub-01/func/sub-01_task-rest_bold.nii.gz"));
    }

    #[test]
    fn test_valid_types() {
        let schema = BidsSchema::load();
        assert!(schema.is_valid_datatype("eeg"));
        assert!(!schema.is_valid_datatype("xyz"));
        assert!(schema.is_valid_suffix("bold"));
        assert!(schema.is_valid_extension(".nii.gz"));
    }
}
