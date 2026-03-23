#![deny(unsafe_code)]
//! Behavioral data support for BIDS datasets.
//!
//! Provides access to behavioral task data files (`_beh.tsv`), associated
//! events files, and dataset summaries for the `beh` datatype.
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/behavioral-experiments.html>

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_io::tsv::read_tsv;
use bids_layout::BidsLayout;
use std::collections::HashMap;

pub struct BehLayout<'a> { layout: &'a BidsLayout }
impl<'a> BehLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self { Self { layout } }
    pub fn get_beh_files(&self) -> Result<Vec<BidsFile>> { self.layout.get().datatype("beh").collect() }
    pub fn get_beh_files_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> { self.layout.get().datatype("beh").subject(s).collect() }
    pub fn get_beh_files_for_task(&self, t: &str) -> Result<Vec<BidsFile>> { self.layout.get().datatype("beh").task(t).collect() }
    pub fn get_beh_subjects(&self) -> Result<Vec<String>> { self.layout.get().datatype("beh").return_unique("subject") }
    pub fn get_beh_tasks(&self) -> Result<Vec<String>> { self.layout.get().datatype("beh").return_unique("task") }

    pub fn get_events(&self, f: &BidsFile) -> Result<Option<Vec<HashMap<String, String>>>> {
        let p = f.companion("events", "tsv");
        if p.exists() { Ok(Some(read_tsv(&p)?)) } else { Ok(None) }
    }

    pub fn get_beh_data(&self, f: &BidsFile) -> Result<Option<Vec<HashMap<String, String>>>> {
        if f.filename.ends_with("_beh.tsv") && f.path.exists() {
            Ok(Some(read_tsv(&f.path)?))
        } else { Ok(None) }
    }

    pub fn summary(&self) -> Result<BehSummary> {
        let files = self.get_beh_files()?;
        let subjects = self.get_beh_subjects()?;
        let tasks = self.get_beh_tasks()?;
        Ok(BehSummary { n_subjects: subjects.len(), n_files: files.len(), subjects, tasks })
    }
}

#[derive(Debug)]
pub struct BehSummary { pub n_subjects: usize, pub n_files: usize, pub subjects: Vec<String>, pub tasks: Vec<String> }
impl std::fmt::Display for BehSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Behavioral Summary: {} subjects, {} files, tasks: {:?}", self.n_subjects, self.n_files, self.tasks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_beh_events() {
        let dir = std::env::temp_dir().join("bids_beh_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sub-01_task-test_events.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "onset\tduration\ttrial_type").unwrap();
        writeln!(f, "1.0\t0.5\tgo").unwrap();
        writeln!(f, "2.0\t0.5\tstop").unwrap();

        let rows = read_tsv(&path).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["trial_type"], "go");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
