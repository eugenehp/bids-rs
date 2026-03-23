//! BIDS file representation with type detection, entity access, and companion lookup.
//!
//! [`BidsFile`] is the central type representing a single file in a BIDS dataset.
//! It carries the file's path, extracted entities, metadata from JSON sidecars,
//! and provides methods for reading JSON/TSV content, finding companion files,
//! and copying/symlinking.

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

use crate::entities::{Entities, EntityValue};
use crate::metadata::BidsMetadata;

/// How to copy a file to a new location.
///
/// # Example
///
/// ```
/// use bids_core::file::CopyMode;
///
/// let mode = CopyMode::Symlink;
/// let mode_from_bool: CopyMode = true.into(); // Symlink
/// assert_eq!(mode, mode_from_bool);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CopyMode {
    /// Create a regular file copy (default).
    #[default]
    Copy,
    /// Create a symbolic link (falls back to copy on non-Unix platforms).
    Symlink,
}

impl From<bool> for CopyMode {
    /// `true` → `Symlink`, `false` → `Copy` (backwards compatibility).
    fn from(symbolic: bool) -> Self {
        if symbolic { CopyMode::Symlink } else { CopyMode::Copy }
    }
}

impl std::fmt::Display for CopyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Copy => write!(f, "copy"),
            Self::Symlink => write!(f, "symlink"),
        }
    }
}

/// The type of a BIDS file, determining what extra operations are available.
///
/// Inferred automatically from file extensions by [`FileType::from_path()`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FileType {
    /// Generic file (no special handling)
    Generic,
    /// Tabular data file (.tsv, .tsv.gz)
    Data,
    /// Neuroimaging file (.nii, .nii.gz, .gii, .dtseries.nii, .func.gii)
    Image,
    /// JSON sidecar
    Json,
    /// EEG data file (.edf, .bdf, .set, .vhdr, .eeg, .fdt)
    Eeg,
    /// MEG data file (.fif, .ds, .sqd, .con, .raw, .pdf)
    Meg,
    /// PET image (.nii, .nii.gz — identified by suffix, but this catches
    /// the common `.blood.tsv` companion pattern)
    Pet,
    /// Microscopy image (.tif, .tiff, .ome.tif, .ome.tiff, .png, .svs)
    Microscopy,
    /// NIRS data file (.snirf)
    Nirs,
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Generic => write!(f, "generic"),
            Self::Data => write!(f, "data"),
            Self::Image => write!(f, "image"),
            Self::Json => write!(f, "json"),
            Self::Eeg => write!(f, "eeg"),
            Self::Meg => write!(f, "meg"),
            Self::Pet => write!(f, "pet"),
            Self::Microscopy => write!(f, "microscopy"),
            Self::Nirs => write!(f, "nirs"),
        }
    }
}

/// Extension-to-FileType mapping table.
///
/// Compound extensions (`.nii.gz`) must come before simple ones (`.nii`)
/// so that `ends_with` matching works correctly.
const EXTENSION_MAP: &[(&[&str], FileType)] = &[
    (&[".dtseries.nii", ".func.gii", ".nii.gz", ".nii", ".gii"], FileType::Image),
    (&[".tsv.gz", ".tsv"], FileType::Data),
    (&[".json"], FileType::Json),
    (&[".edf", ".bdf", ".set", ".vhdr", ".eeg", ".fdt"], FileType::Eeg),
    (&[".fif", ".ds", ".sqd", ".con", ".raw", ".pdf"], FileType::Meg),
    (&[".snirf"], FileType::Nirs),
    (&[".ome.tif", ".ome.tiff", ".tif", ".tiff", ".svs"], FileType::Microscopy),
];

impl FileType {
    /// Infer file type from the file extension(s).
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        for &(extensions, file_type) in EXTENSION_MAP {
            if extensions.iter().any(|ext| name.ends_with(ext)) {
                return file_type;
            }
        }
        FileType::Generic
    }
}

/// Represents a single file in a BIDS dataset.
///
/// This is the Rust equivalent of PyBIDS' `BIDSFile` hierarchy
/// (BIDSFile, BIDSDataFile, BIDSImageFile, BIDSJSONFile).
///
/// # Example
///
/// ```
/// use bids_core::file::BidsFile;
///
/// let f = BidsFile::new("/data/sub-01/eeg/sub-01_task-rest_eeg.edf");
/// assert_eq!(f.suffix(), Some("eeg"));
/// assert_eq!(f.extension(), ".edf");
/// assert_eq!(f.file_type, bids_core::FileType::Eeg);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidsFile {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// Just the filename component.
    pub filename: String,
    /// The parent directory.
    pub dirname: PathBuf,
    /// Whether this entry represents a directory.
    pub is_dir: bool,
    /// File type determined from extension.
    pub file_type: FileType,
    /// Entities extracted from the filename.
    pub entities: Entities,
    /// Metadata loaded from JSON sidecars (populated during metadata indexing).
    ///
    /// **Note:** This field is skipped during serialization/deserialization.
    /// If you serialize a `BidsFile` and deserialize it back, metadata will be
    /// empty. Use `BidsLayout::get_metadata()` to re-populate after
    /// deserialization, or serialize metadata separately.
    #[serde(skip)]
    pub metadata: BidsMetadata,
}

impl BidsFile {
    /// Create a new BidsFile from a path.
    pub fn new(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let filename = path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let dirname = path.parent()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_default();
        let is_dir = filename.is_empty();
        let file_type = FileType::from_path(&path);

        Self {
            path,
            filename,
            dirname,
            is_dir,
            file_type,
            entities: Entities::new(),
            metadata: BidsMetadata::new(),
        }
    }

    /// Set entities on this file, returning `self` for chaining.
    #[must_use]
    pub fn with_entities(mut self, entities: Entities) -> Self {
        self.entities = entities;
        self
    }

    /// Set metadata on this file, returning `self` for chaining.
    #[must_use]
    pub fn with_metadata(mut self, metadata: BidsMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Get the path relative to a root directory.
    #[must_use]
    pub fn relpath(&self, root: &Path) -> Option<PathBuf> {
        self.path.strip_prefix(root).ok().map(std::path::Path::to_path_buf)
    }

    /// Get a combined view of filename entities and metadata.
    #[must_use]
    pub fn get_entities(&self, metadata: Option<bool>) -> Entities {
        match metadata {
            Some(true) => {
                // Only metadata entities
                self.metadata.iter()
                    .map(|(k, v)| (k.clone(), EntityValue::Json(v.clone())))
                    .collect()
            }
            Some(false) => self.entities.clone(),
            None => {
                let mut merged = self.entities.clone();
                for (k, v) in self.metadata.iter() {
                    if !merged.contains_key(k) {
                        merged.insert(k.clone(), EntityValue::Json(v.clone()));
                    }
                }
                merged
            }
        }
    }

    /// Get metadata for this file (from JSON sidecars).
    #[must_use]
    pub fn get_metadata(&self) -> &BidsMetadata {
        &self.metadata
    }

    /// Get the full extension, including compound extensions like `.tsv.gz`.
    ///
    /// Compound extensions are checked first (longest match wins).
    #[must_use]
    pub fn extension(&self) -> &str {
        const COMPOUND_EXTENSIONS: &[&str] = &[
            ".dtseries.nii", ".func.gii", ".ome.tif", ".ome.tiff",
            ".nii.gz", ".tsv.gz",
        ];
        let name = &self.filename;
        for ext in COMPOUND_EXTENSIONS {
            if name.ends_with(ext) {
                return ext;
            }
        }
        name.rfind('.')
            .map(|start| &name[start..])
            .unwrap_or("")
    }

    /// Get the suffix (the part before the extension, after the last underscore).
    #[must_use]
    pub fn suffix(&self) -> Option<&str> {
        let stem = self.filename.split('.').next()?;
        stem.rsplit('_').next()
    }

    /// Find a companion file by replacing the suffix and extension.
    ///
    /// E.g., for `sub-01_task-rest_eeg.edf`, `companion("channels", "tsv")`
    /// returns `sub-01_task-rest_channels.tsv` in the same directory.
    #[must_use]
    pub fn companion(&self, suffix: &str, ext: &str) -> PathBuf {
        let stem = self.filename.split('.').next().unwrap_or("");
        let base = stem.rsplit_once('_').map(|(b, _)| b).unwrap_or(stem);
        self.dirname.join(format!("{base}_{suffix}.{ext}"))
    }

    /// Read a JSON file and return as `serde_json::Value`.
    ///
    /// # Errors
    ///
    /// Returns a `BidsError::FileType` if this file is not a JSON file, or
    /// an I/O / JSON parse error.
    pub fn get_json(&self) -> Result<serde_json::Value, crate::error::BidsError> {
        if self.file_type != FileType::Json {
            return Err(crate::error::BidsError::FileType(
                format!("{} is not a JSON file", self.path.display())
            ));
        }
        let contents = std::fs::read_to_string(&self.path)?;
        let val: serde_json::Value = serde_json::from_str(&contents)?;
        Ok(val)
    }

    /// Read a JSON file and return as a `HashMap`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not JSON, can't be read, or the
    /// top-level JSON value is not an object.
    pub fn get_dict(&self) -> Result<std::collections::HashMap<String, serde_json::Value>, crate::error::BidsError> {
        let val = self.get_json()?;
        match val {
            serde_json::Value::Object(map) => {
                Ok(map.into_iter().collect())
            }
            _ => Err(crate::error::BidsError::FileType(
                format!("{} is a JSON containing {}, not an object", self.path.display(), val)
            )),
        }
    }

    /// Read a TSV/TSV.GZ file and return rows as `Vec<HashMap<String, String>>`.
    ///
    /// Handles both plain `.tsv` and gzip-compressed `.tsv.gz` files. The
    /// BIDS sentinel value `n/a` is automatically converted to empty strings.
    ///
    /// For bulk TSV processing, prefer [`bids_io::read_tsv`] / [`bids_io::read_tsv_gz`]
    /// which share the same parsing logic but don't require a `BidsFile`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read, is empty, or is not a
    /// tabular data file.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use bids_core::file::BidsFile;
    /// let f = BidsFile::new("/data/sub-01/eeg/sub-01_events.tsv");
    /// let rows = f.get_df().unwrap();
    /// for row in &rows {
    ///     println!("onset={}, type={}", row["onset"], row["trial_type"]);
    /// }
    /// ```
    pub fn get_df(&self) -> Result<Vec<std::collections::HashMap<String, String>>, crate::error::BidsError> {
        let file = std::fs::File::open(&self.path)?;
        let reader: Box<dyn std::io::Read> = if self.filename.ends_with(".tsv.gz") {
            Box::new(flate2::read::GzDecoder::new(file))
        } else {
            Box::new(file)
        };
        parse_tsv_reader(reader)
    }

    /// Copy this file to a new location.
    ///
    /// Accepts a [`CopyMode`] to control whether the file is copied or
    /// symlinked. Prefer the enum over a bare boolean for clarity.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the source can't be read or the destination
    /// can't be created.
    pub fn copy_to(
        &self,
        new_path: &std::path::Path,
        mode: CopyMode,
    ) -> Result<(), crate::error::BidsError> {
        if let Some(parent) = new_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match mode {
            CopyMode::Symlink => {
                #[cfg(unix)]
                std::os::unix::fs::symlink(&self.path, new_path)?;
                #[cfg(not(unix))]
                std::fs::copy(&self.path, new_path)?;
            }
            CopyMode::Copy => {
                std::fs::copy(&self.path, new_path)?;
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for BidsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<BidsFile '{}'>", self.path.display())
    }
}

impl PartialEq for BidsFile {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl Eq for BidsFile {}

impl PartialOrd for BidsFile {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BidsFile {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        natural_cmp(&self.path.to_string_lossy(), &other.path.to_string_lossy())
    }
}

/// Natural sort comparison: numeric substrings are compared as numbers.
fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();

    loop {
        match (ai.peek(), bi.peek()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(&ac), Some(&bc)) => {
                if ac.is_ascii_digit() && bc.is_ascii_digit() {
                    let mut an = String::new();
                    while let Some(&c) = ai.peek() {
                        if c.is_ascii_digit() { an.push(c); ai.next(); } else { break; }
                    }
                    let mut bn = String::new();
                    while let Some(&c) = bi.peek() {
                        if c.is_ascii_digit() { bn.push(c); bi.next(); } else { break; }
                    }
                    let av: u64 = an.parse().unwrap_or(0);
                    let bv: u64 = bn.parse().unwrap_or(0);
                    match av.cmp(&bv) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                } else {
                    let al = ac.to_lowercase().next().unwrap_or(ac);
                    let bl = bc.to_lowercase().next().unwrap_or(bc);
                    match al.cmp(&bl) {
                        Ordering::Equal => { ai.next(); bi.next(); }
                        ord => return ord,
                    }
                }
            }
        }
    }
}

/// Parse TSV rows from any reader. Shared logic for `BidsFile::get_df()`.
///
/// The `n/a` sentinel is converted to empty strings per BIDS convention.
pub(crate) fn parse_tsv_reader(reader: impl std::io::Read) -> Result<Vec<std::collections::HashMap<String, String>>, crate::error::BidsError> {
    use std::io::{BufRead, BufReader};

    let mut lines = BufReader::new(reader).lines();
    let header_line = lines.next()
        .ok_or_else(|| crate::error::BidsError::Csv("Empty TSV file".into()))??;
    let headers: Vec<String> = header_line.split('\t').map(|s| s.trim().to_string()).collect();

    let mut rows = Vec::new();
    for line_result in lines {
        let line = line_result?;
        if line.trim().is_empty() { continue; }
        let values: Vec<&str> = line.split('\t').collect();
        let mut row = std::collections::HashMap::new();
        for (i, header) in headers.iter().enumerate() {
            let val = values.get(i).copied().unwrap_or("").trim();
            row.insert(header.clone(), if val == "n/a" { String::new() } else { val.to_string() });
        }
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_type_detection() {
        assert_eq!(FileType::from_path(Path::new("sub-01_T1w.nii.gz")), FileType::Image);
        assert_eq!(FileType::from_path(Path::new("sub-01_events.tsv")), FileType::Data);
        assert_eq!(FileType::from_path(Path::new("sub-01_eeg.json")), FileType::Json);
        assert_eq!(FileType::from_path(Path::new("sub-01_eeg.edf")), FileType::Eeg);
        assert_eq!(FileType::from_path(Path::new("sub-01_eeg.bdf")), FileType::Eeg);
        assert_eq!(FileType::from_path(Path::new("sub-01_meg.fif")), FileType::Meg);
        assert_eq!(FileType::from_path(Path::new("sub-01_nirs.snirf")), FileType::Nirs);
        assert_eq!(FileType::from_path(Path::new("sub-01_sample-A_FLUO.tif")), FileType::Microscopy);
        assert_eq!(FileType::from_path(Path::new("README")), FileType::Generic);
    }

    #[test]
    fn test_bids_file() {
        let f = BidsFile::new("/data/sub-01/eeg/sub-01_task-rest_eeg.edf");
        assert_eq!(f.filename, "sub-01_task-rest_eeg.edf");
        assert_eq!(f.file_type, FileType::Eeg);
        assert_eq!(f.suffix(), Some("eeg"));
        assert_eq!(f.extension(), ".edf");
    }

    #[test]
    fn test_natural_sort() {
        let mut files: Vec<String> = vec![
            "sub-10".into(), "sub-2".into(), "sub-1".into(), "sub-20".into(),
        ];
        files.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(files, vec!["sub-1", "sub-2", "sub-10", "sub-20"]);
    }
}
