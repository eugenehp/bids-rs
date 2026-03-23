//! BIDS specification version tracking and compatibility checking.
//!
//! This module provides the infrastructure for pinning `bids-rs` to a specific
//! BIDS specification version, detecting version mismatches, and planning
//! migration paths when the spec evolves.
//!
//! # Design
//!
//! The BIDS specification is versioned using [SemVer](https://semver.org/).
//! Each release of `bids-rs` targets a specific spec version, declared in
//! [`SUPPORTED_BIDS_VERSION`]. When loading a dataset whose
//! `dataset_description.json` declares a different `BIDSVersion`, the library
//! can warn or error depending on the compatibility policy.
//!
//! ## How spec updates flow through the codebase
//!
//! All spec-derived knowledge is concentrated in two places:
//!
//! 1. **`bids-schema`** — Entity definitions, valid datatypes/suffixes/extensions,
//!    and filename validation patterns.  When the BIDS spec adds a new entity
//!    or datatype, only this crate needs updating.
//!
//! 2. **`bids-core/src/configs/`** — `bids.json` and `derivatives.json` config
//!    files containing entity regex patterns and path-building templates.
//!    These are `include_str!`'d at compile time.
//!
//! All other crates pull their spec knowledge from these two sources rather
//! than hardcoding it.

use std::fmt;

/// The BIDS specification version that this release of `bids-rs` targets.
///
/// Update this constant (and the schema/config data) when adopting a new
/// spec version.  CI should verify that this matches the embedded schema.
pub const SUPPORTED_BIDS_VERSION: BidsVersion = BidsVersion::new(1, 9, 0);

/// The minimum BIDS version that this release can read without data loss.
///
/// Datasets older than this may be missing required fields or use deprecated
/// conventions that the library cannot handle correctly.
pub const MIN_COMPATIBLE_VERSION: BidsVersion = BidsVersion::new(1, 4, 0);

/// A parsed BIDS specification version (SemVer).
///
/// # Examples
///
/// ```
/// use bids_schema::version::BidsVersion;
///
/// let v = BidsVersion::parse("1.9.0").unwrap();
/// assert_eq!(v.major, 1);
/// assert_eq!(v.minor, 9);
/// assert_eq!(v.patch, 0);
///
/// let older = BidsVersion::parse("1.6.0").unwrap();
/// assert!(v > older);
/// assert!(v.is_compatible_with(&older));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BidsVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl BidsVersion {
    /// Create a new version.
    #[must_use]
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version string like `"1.9.0"`.
    ///
    /// Accepts 2-part (`"1.9"`) and 3-part (`"1.9.0"`) versions.
    /// Returns `None` if parsing fails.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.trim().split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return None;
        }
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0);
        Some(Self {
            major,
            minor,
            patch,
        })
    }

    /// Check whether this version is compatible with `other`.
    ///
    /// Two versions are compatible if they share the same major version and
    /// `other` is at least [`MIN_COMPATIBLE_VERSION`].
    #[must_use]
    pub fn is_compatible_with(&self, other: &BidsVersion) -> bool {
        self.major == other.major && *other >= MIN_COMPATIBLE_VERSION
    }

    /// Check whether `other` is from a newer spec than this version.
    #[must_use]
    pub fn is_older_than(&self, other: &BidsVersion) -> bool {
        self < other
    }

    /// Return the compatibility status between this library version and
    /// a dataset's declared BIDS version.
    #[must_use]
    pub fn check_compatibility(&self, dataset_version: &BidsVersion) -> Compatibility {
        if *dataset_version == *self {
            Compatibility::Exact
        } else if dataset_version.major != self.major {
            Compatibility::Incompatible {
                reason: format!(
                    "Major version mismatch: dataset is BIDS {dataset_version}, \
                     library targets BIDS {self}"
                ),
            }
        } else if *dataset_version < MIN_COMPATIBLE_VERSION {
            Compatibility::Incompatible {
                reason: format!(
                    "Dataset BIDS version {dataset_version} is below minimum \
                     compatible version {MIN_COMPATIBLE_VERSION}"
                ),
            }
        } else if dataset_version > self {
            Compatibility::Newer {
                dataset: *dataset_version,
                library: *self,
            }
        } else {
            Compatibility::Compatible {
                dataset: *dataset_version,
                library: *self,
            }
        }
    }
}

impl fmt::Display for BidsVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl std::str::FromStr for BidsVersion {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| format!("Invalid BIDS version: '{s}'"))
    }
}

/// Result of checking a dataset's BIDS version against the library's supported version.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Compatibility {
    /// Dataset version exactly matches the library's target version.
    Exact,
    /// Dataset is from an older (but supported) spec version.
    /// Some newer features may not be present in the dataset.
    Compatible {
        dataset: BidsVersion,
        library: BidsVersion,
    },
    /// Dataset declares a *newer* spec version than the library supports.
    /// The library may not understand new entities, datatypes, or conventions.
    Newer {
        dataset: BidsVersion,
        library: BidsVersion,
    },
    /// Dataset is fundamentally incompatible (different major version or
    /// below minimum supported version).
    Incompatible { reason: String },
}

impl Compatibility {
    /// Returns `true` if the dataset can be safely processed.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Exact | Self::Compatible { .. })
    }

    /// Returns `true` if processing may lose information or produce warnings.
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        matches!(self, Self::Newer { .. })
    }

    /// Returns `true` if the dataset should not be processed.
    #[must_use]
    pub fn is_incompatible(&self) -> bool {
        matches!(self, Self::Incompatible { .. })
    }
}

impl fmt::Display for Compatibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => write!(f, "exact match"),
            Self::Compatible { dataset, library } => {
                write!(f, "compatible (dataset {dataset}, library {library})")
            }
            Self::Newer { dataset, library } => {
                write!(
                    f,
                    "WARNING: dataset uses BIDS {dataset}, but library only \
                     supports up to {library}. Some features may not be recognized."
                )
            }
            Self::Incompatible { reason } => write!(f, "INCOMPATIBLE: {reason}"),
        }
    }
}

/// Structured record of what changed between two BIDS spec versions.
///
/// Used by [`CHANGELOG`] to drive automated migration and validation.
/// When bumping the supported BIDS version, add a new entry describing
/// what changed so the library can:
/// - Warn about deprecated features
/// - Recognize new entities/datatypes
/// - Validate against the correct rules
#[derive(Debug, Clone)]
pub struct SpecChange {
    /// The version that introduced this change.
    pub version: BidsVersion,
    /// Human-readable summary.
    pub summary: &'static str,
    /// New entities added in this version (if any).
    pub new_entities: &'static [&'static str],
    /// New datatypes added (if any).
    pub new_datatypes: &'static [&'static str],
    /// New suffixes added (if any).
    pub new_suffixes: &'static [&'static str],
    /// Entities deprecated or removed (if any).
    pub deprecated_entities: &'static [&'static str],
    /// Breaking changes that require code updates.
    pub breaking: bool,
}

/// Registry of known spec changes.  
///
/// When updating `SUPPORTED_BIDS_VERSION`, add an entry here documenting
/// what the new version adds/changes.  This serves as both documentation
/// and a machine-readable migration guide.
///
/// # Adding a new BIDS version
///
/// 1. Add a `SpecChange` entry to [`CHANGELOG`] below.
/// 2. Update [`SUPPORTED_BIDS_VERSION`] at the top of this file.
/// 3. Update `bids-schema/src/lib.rs` — add new entities/datatypes/suffixes/extensions.
/// 4. Update `bids-core/src/configs/bids.json` — add regex patterns for new entities
///    and path patterns for new file types.
/// 5. Update `bids-core/src/configs/derivatives.json` if derivative patterns changed.
/// 6. Update `bids-core/src/entities.rs` `ENTITY_ORDER` if new entities were added.
/// 7. Run `cargo test --workspace` and fix any failing validations.
/// 8. Update the `bids-cli upgrade` command if migration logic is needed.
pub const CHANGELOG: &[SpecChange] = &[
    SpecChange {
        version: BidsVersion::new(1, 7, 0),
        summary: "Added microscopy (micr) datatype, near-infrared spectroscopy (NIRS), \
                  and genetic descriptor files",
        new_entities: &["sample", "staining", "chunk"],
        new_datatypes: &["micr"],
        new_suffixes: &[
            "TEM", "SEM", "uCT", "BF", "DF", "PC", "DIC", "FLUO", "CONF", "PLI", "CARS", "2PE",
            "MPE", "SR", "NLO", "OCT", "SPIM",
        ],
        deprecated_entities: &[],
        breaking: false,
    },
    SpecChange {
        version: BidsVersion::new(1, 8, 0),
        summary: "Added motion capture (motion), MR spectroscopy (mrs), PET, perfusion (perf), \
                  and NIRS datatypes. Added quantitative MRI entities and suffixes.",
        new_entities: &["tracksys", "nucleus", "volume"],
        new_datatypes: &["motion", "mrs", "nirs", "perf"],
        new_suffixes: &[
            "motion",
            "nirs",
            "optodes",
            "svs",
            "mrsi",
            "unloc",
            "mrsref",
            "asl",
            "m0scan",
            "aslcontext",
            "asllabeling",
        ],
        deprecated_entities: &[],
        breaking: false,
    },
    SpecChange {
        version: BidsVersion::new(1, 9, 0),
        summary: "Stabilized positional encoding entities, added atlas entity, \
                  refined qMRI suffixes and file patterns",
        new_entities: &["atlas"],
        new_datatypes: &[],
        new_suffixes: &[],
        deprecated_entities: &[],
        breaking: false,
    },
];

/// Get all spec changes between two versions (exclusive of `from`, inclusive of `to`).
#[must_use]
pub fn changes_between(from: &BidsVersion, to: &BidsVersion) -> Vec<&'static SpecChange> {
    CHANGELOG
        .iter()
        .filter(|c| c.version > *from && c.version <= *to)
        .collect()
}

/// Get all entities that were added after a given version.
#[must_use]
pub fn entities_added_since(version: &BidsVersion) -> Vec<&'static str> {
    CHANGELOG
        .iter()
        .filter(|c| c.version > *version)
        .flat_map(|c| c.new_entities.iter().copied())
        .collect()
}

/// Get all datatypes that were added after a given version.
#[must_use]
pub fn datatypes_added_since(version: &BidsVersion) -> Vec<&'static str> {
    CHANGELOG
        .iter()
        .filter(|c| c.version > *version)
        .flat_map(|c| c.new_datatypes.iter().copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_version() {
        let v = BidsVersion::parse("1.9.0").unwrap();
        assert_eq!(v, BidsVersion::new(1, 9, 0));

        let v2 = BidsVersion::parse("1.8").unwrap();
        assert_eq!(v2, BidsVersion::new(1, 8, 0));

        assert!(BidsVersion::parse("").is_none());
        assert!(BidsVersion::parse("abc").is_none());
        assert!(BidsVersion::parse("1.2.3.4").is_none());
    }

    #[test]
    fn test_version_ordering() {
        let v190 = BidsVersion::new(1, 9, 0);
        let v180 = BidsVersion::new(1, 8, 0);
        let v160 = BidsVersion::new(1, 6, 0);
        assert!(v190 > v180);
        assert!(v180 > v160);
    }

    #[test]
    fn test_compatibility_exact() {
        let lib = SUPPORTED_BIDS_VERSION;
        let compat = lib.check_compatibility(&lib);
        assert_eq!(compat, Compatibility::Exact);
        assert!(compat.is_ok());
    }

    #[test]
    fn test_compatibility_older_dataset() {
        let lib = SUPPORTED_BIDS_VERSION;
        let dataset = BidsVersion::new(1, 6, 0);
        let compat = lib.check_compatibility(&dataset);
        assert!(compat.is_ok());
        assert!(!compat.has_warnings());
    }

    #[test]
    fn test_compatibility_newer_dataset() {
        let lib = BidsVersion::new(1, 9, 0);
        let dataset = BidsVersion::new(1, 11, 0);
        let compat = lib.check_compatibility(&dataset);
        assert!(compat.has_warnings());
        assert!(!compat.is_incompatible());
    }

    #[test]
    fn test_compatibility_too_old() {
        let lib = SUPPORTED_BIDS_VERSION;
        let dataset = BidsVersion::new(1, 2, 0);
        let compat = lib.check_compatibility(&dataset);
        assert!(compat.is_incompatible());
    }

    #[test]
    fn test_compatibility_major_mismatch() {
        let lib = SUPPORTED_BIDS_VERSION;
        let dataset = BidsVersion::new(2, 0, 0);
        let compat = lib.check_compatibility(&dataset);
        assert!(compat.is_incompatible());
    }

    #[test]
    fn test_changes_between() {
        let from = BidsVersion::new(1, 6, 0);
        let to = BidsVersion::new(1, 9, 0);
        let changes = changes_between(&from, &to);
        assert!(!changes.is_empty());
        // Should include 1.7, 1.8, 1.9 but not 1.6
        assert!(changes.iter().all(|c| c.version > from));
        assert!(changes.iter().all(|c| c.version <= to));
    }

    #[test]
    fn test_entities_added_since() {
        let v160 = BidsVersion::new(1, 6, 0);
        let added = entities_added_since(&v160);
        assert!(added.contains(&"sample"));
        assert!(added.contains(&"tracksys"));
        assert!(added.contains(&"atlas"));
    }

    #[test]
    fn test_datatypes_added_since() {
        let v160 = BidsVersion::new(1, 6, 0);
        let added = datatypes_added_since(&v160);
        assert!(added.contains(&"micr"));
        assert!(added.contains(&"motion"));
        assert!(added.contains(&"nirs"));
    }

    #[test]
    fn test_display() {
        assert_eq!(SUPPORTED_BIDS_VERSION.to_string(), "1.9.0");
    }

    #[test]
    fn test_from_str() {
        let v: BidsVersion = "1.9.0".parse().unwrap();
        assert_eq!(v, BidsVersion::new(1, 9, 0));
    }

    #[test]
    fn test_changelog_is_sorted() {
        for window in CHANGELOG.windows(2) {
            assert!(
                window[0].version < window[1].version,
                "CHANGELOG must be sorted by version: {} >= {}",
                window[0].version,
                window[1].version,
            );
        }
    }

    #[test]
    fn test_supported_version_matches_last_changelog() {
        let last = CHANGELOG.last().expect("CHANGELOG should not be empty");
        assert_eq!(
            last.version, SUPPORTED_BIDS_VERSION,
            "SUPPORTED_BIDS_VERSION ({SUPPORTED_BIDS_VERSION}) must match the \
             last CHANGELOG entry ({}). Did you forget to update one?",
            last.version,
        );
    }
}
