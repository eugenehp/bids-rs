//! Filter BIDS files by modality, subject, task, extension, and custom predicates.

use bids_core::file::BidsFile;
use std::path::Path;

/// A composable filter for selecting BIDS files from a dataset.
///
/// Filters can be chained and are applied conjunctively (AND).
#[derive(Debug, Clone, Default)]
pub struct DatasetFilter {
    /// Only include files from this datatype/modality (e.g., "eeg", "anat", "func").
    pub modality: Option<String>,
    /// Only include files matching this suffix (e.g., "bold", "eeg", "T1w").
    pub suffix: Option<String>,
    /// Only include these subjects (e.g., ["01", "02"]).
    pub subjects: Option<Vec<String>>,
    /// Exclude these subjects.
    pub exclude_subjects: Option<Vec<String>>,
    /// Only include these tasks.
    pub tasks: Option<Vec<String>>,
    /// Only include these sessions.
    pub sessions: Option<Vec<String>>,
    /// Only include these runs.
    pub runs: Option<Vec<String>>,
    /// Only include files with this extension (e.g., ".edf", ".nii.gz").
    pub extensions: Option<Vec<String>>,
    /// Minimum file size in bytes.
    pub min_size: Option<u64>,
}

impl DatasetFilter {
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn modality(mut self, m: &str) -> Self {
        self.modality = Some(m.into());
        self
    }
    #[must_use]
    pub fn suffix(mut self, s: &str) -> Self {
        self.suffix = Some(s.into());
        self
    }
    #[must_use]
    pub fn subjects(mut self, s: Vec<String>) -> Self {
        self.subjects = Some(s);
        self
    }
    #[must_use]
    pub fn exclude_subjects(mut self, s: Vec<String>) -> Self {
        self.exclude_subjects = Some(s);
        self
    }
    #[must_use]
    pub fn tasks(mut self, t: Vec<String>) -> Self {
        self.tasks = Some(t);
        self
    }
    #[must_use]
    pub fn sessions(mut self, s: Vec<String>) -> Self {
        self.sessions = Some(s);
        self
    }
    #[must_use]
    pub fn runs(mut self, r: Vec<String>) -> Self {
        self.runs = Some(r);
        self
    }
    #[must_use]
    pub fn extension(mut self, e: &str) -> Self {
        self.extensions.get_or_insert_with(Vec::new).push(e.into());
        self
    }
    #[must_use]
    pub fn min_size(mut self, bytes: u64) -> Self {
        self.min_size = Some(bytes);
        self
    }

    /// Test whether a BidsFile matches this filter.
    pub fn matches(&self, f: &BidsFile) -> bool {
        let ent_str = |key: &str| -> Option<String> {
            f.entities.get(key).map(std::string::ToString::to_string)
        };

        if let Some(ref m) = self.modality {
            if !ent_str("datatype").is_some_and(|d| d.eq_ignore_ascii_case(m)) {
                return false;
            }
        }
        if let Some(ref s) = self.suffix {
            if ent_str("suffix").is_none_or(|d| d != *s) {
                return false;
            }
        }
        if let Some(ref subs) = self.subjects {
            if !ent_str("subject")
                .as_ref()
                .is_some_and(|d| subs.iter().any(|s| s == d))
            {
                return false;
            }
        }
        if let Some(ref excl) = self.exclude_subjects {
            if ent_str("subject")
                .as_ref()
                .is_some_and(|d| excl.iter().any(|s| s == d))
            {
                return false;
            }
        }
        if let Some(ref tasks) = self.tasks {
            if !ent_str("task")
                .as_ref()
                .is_some_and(|d| tasks.iter().any(|t| t == d))
            {
                return false;
            }
        }
        if let Some(ref sess) = self.sessions {
            if !ent_str("session")
                .as_ref()
                .is_some_and(|d| sess.iter().any(|s| s == d))
            {
                return false;
            }
        }
        if let Some(ref runs) = self.runs {
            if !ent_str("run")
                .as_ref()
                .is_some_and(|d| runs.iter().any(|r| r == d))
            {
                return false;
            }
        }
        if let Some(ref exts) = self.extensions {
            let fname = &f.filename;
            if !exts.iter().any(|e| fname.ends_with(e.as_str())) {
                return false;
            }
        }
        true
    }

    /// Test whether a remote file path matches this filter.
    /// This is a lightweight check based on the path string, used for
    /// filtering remote file listings before download.
    pub fn matches_path(&self, path: &str) -> bool {
        if let Some(ref m) = self.modality {
            // Check if path contains the datatype directory
            if !path.contains(&format!("/{m}/")) && !path.starts_with(&format!("{m}/")) {
                return false;
            }
        }
        if let Some(ref exts) = self.extensions {
            if !exts.iter().any(|e| path.ends_with(e.as_str())) {
                return false;
            }
        }
        if let Some(ref subs) = self.subjects {
            // Check sub-XX in path
            if !subs
                .iter()
                .any(|s| path.contains(&format!("sub-{s}/")) || path.contains(&format!("sub-{s}_")))
            {
                return false;
            }
        }
        if let Some(ref tasks) = self.tasks {
            if !tasks.iter().any(|t| {
                path.contains(&format!("task-{t}_")) || path.contains(&format!("task-{t}."))
            }) {
                return false;
            }
        }
        true
    }

    /// Filter a remote file list for selective download.
    pub fn filter_remote<'a>(&self, files: &'a [crate::RemoteFile]) -> Vec<&'a crate::RemoteFile> {
        files
            .iter()
            .filter(|f| {
                if let Some(min) = self.min_size {
                    if f.size < min {
                        return false;
                    }
                }
                self.matches_path(&f.path)
            })
            .collect()
    }
}

/// Apply a `DatasetFilter` to a local BIDS dataset directory.
///
/// Walks the directory and returns matching `BidsFile` entries.
pub fn filter_local(root: &Path, filter: &DatasetFilter) -> crate::Result<Vec<BidsFile>> {
    let mut results = Vec::new();
    walk_dir(root, root, filter, &mut results)?;
    Ok(results)
}

fn walk_dir(
    base: &Path,
    dir: &Path,
    filter: &DatasetFilter,
    out: &mut Vec<BidsFile>,
) -> crate::Result<()> {
    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            walk_dir(base, &path, filter, out)?;
        } else {
            let path_str = path.to_str().unwrap_or("");
            let mut bf = BidsFile::new(path_str);
            // Parse entities from the path since BidsFile::new doesn't do it
            populate_entities_from_path(&mut bf, base);
            if filter.matches(&bf) {
                if let Some(min) = filter.min_size {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        if meta.len() < min {
                            continue;
                        }
                    }
                }
                out.push(bf);
            }
        }
    }
    Ok(())
}

/// Parse BIDS entities from file path when no layout is available.
fn populate_entities_from_path(bf: &mut BidsFile, root: &Path) {
    use bids_core::entities::EntityValue;

    let rel = bf.path.strip_prefix(root).unwrap_or(&bf.path);
    let components: Vec<&str> = rel.iter().filter_map(|c| c.to_str()).collect();

    // Datatype = directory containing the file if it's a known BIDS datatype.
    // Uses the schema's datatype list so it stays in sync with spec updates.
    let schema = bids_schema::BidsSchema::load();
    if components.len() >= 2 {
        let parent_dir = components[components.len() - 2];
        if schema.is_valid_datatype(parent_dir) {
            bf.entities
                .insert("datatype".into(), EntityValue::Str(parent_dir.into()));
        }
    }

    // Parse entities from filename: sub-XX_ses-YY_task-ZZ_run-NN_suffix.ext
    let fname = &bf.filename;
    let stem = fname.split('.').next().unwrap_or(fname);
    let parts: Vec<&str> = stem.split('_').collect();
    for part in &parts {
        if let Some(val) = part.strip_prefix("sub-") {
            bf.entities
                .insert("subject".into(), EntityValue::Str(val.into()));
        } else if let Some(val) = part.strip_prefix("ses-") {
            bf.entities
                .insert("session".into(), EntityValue::Str(val.into()));
        } else if let Some(val) = part.strip_prefix("task-") {
            bf.entities
                .insert("task".into(), EntityValue::Str(val.into()));
        } else if let Some(val) = part.strip_prefix("run-") {
            bf.entities
                .insert("run".into(), EntityValue::Str(val.into()));
        } else if let Some(val) = part.strip_prefix("acq-") {
            bf.entities
                .insert("acquisition".into(), EntityValue::Str(val.into()));
        }
    }
    // Last part before extension is the suffix
    if let Some(last) = parts.last() {
        if !last.contains('-') {
            bf.entities
                .insert("suffix".into(), EntityValue::Str((*last).into()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_matches_path() {
        let f = DatasetFilter::new().modality("eeg").extension(".edf");
        assert!(f.matches_path("sub-01/eeg/sub-01_task-rest_eeg.edf"));
        assert!(!f.matches_path("sub-01/anat/sub-01_T1w.nii.gz"));
        assert!(!f.matches_path("sub-01/eeg/sub-01_task-rest_eeg.json"));
    }

    #[test]
    fn test_filter_subjects() {
        let f = DatasetFilter::new().subjects(vec!["01".into(), "03".into()]);
        assert!(f.matches_path("sub-01/eeg/sub-01_eeg.edf"));
        assert!(f.matches_path("sub-03/eeg/sub-03_eeg.edf"));
        assert!(!f.matches_path("sub-02/eeg/sub-02_eeg.edf"));
    }

    #[test]
    fn test_filter_tasks() {
        let f = DatasetFilter::new().tasks(vec!["rest".into()]);
        assert!(f.matches_path("sub-01/eeg/sub-01_task-rest_eeg.edf"));
        assert!(!f.matches_path("sub-01/eeg/sub-01_task-motor_eeg.edf"));
    }

    #[test]
    fn test_filter_remote() {
        let files = vec![
            crate::RemoteFile {
                path: "sub-01/eeg/sub-01_eeg.edf".into(),
                size: 1000,
            },
            crate::RemoteFile {
                path: "sub-01/eeg/sub-01_eeg.json".into(),
                size: 500,
            },
            crate::RemoteFile {
                path: "sub-01/anat/sub-01_T1w.nii.gz".into(),
                size: 5000,
            },
        ];
        let f = DatasetFilter::new().modality("eeg").extension(".edf");
        let matched = f.filter_remote(&files);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].path, "sub-01/eeg/sub-01_eeg.edf");
    }
}
