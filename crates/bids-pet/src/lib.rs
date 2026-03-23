#![deny(unsafe_code)]
//! Positron Emission Tomography (PET) support for BIDS datasets.
//!
//! Provides typed access to PET-specific files, blood samples, tracer metadata
//! (name, radionuclide, injected activity/mass), frame timing, and PET events.
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/positron-emission-tomography.html>

pub mod metadata;
use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_io::tsv::read_tsv;
use bids_layout::BidsLayout;
use serde::{Deserialize, Serialize};
pub use metadata::PetMetadata;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloodSample {
    pub time: f64,
    pub whole_blood_radioactivity: Option<f64>,
    pub plasma_radioactivity: Option<f64>,
    pub metabolite_parent_fraction: Option<f64>,
}

pub fn read_blood_tsv(path: &std::path::Path) -> Result<Vec<BloodSample>> {
    let rows = read_tsv(path)?;
    Ok(rows.iter().map(|r| BloodSample {
        time: r.get("time").and_then(|v| v.parse().ok()).unwrap_or(0.0),
        whole_blood_radioactivity: r.get("whole_blood_radioactivity").and_then(|v| v.parse().ok()),
        plasma_radioactivity: r.get("plasma_radioactivity").and_then(|v| v.parse().ok()),
        metabolite_parent_fraction: r.get("metabolite_parent_fraction").and_then(|v| v.parse().ok()),
    }).collect())
}

pub struct PetLayout<'a> { layout: &'a BidsLayout }
impl<'a> PetLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self { Self { layout } }
    pub fn get_pet_files(&self) -> Result<Vec<BidsFile>> { self.layout.get().suffix("pet").collect() }
    pub fn get_pet_files_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> { self.layout.get().suffix("pet").subject(s).collect() }
    pub fn get_pet_files_for_task(&self, t: &str) -> Result<Vec<BidsFile>> { self.layout.get().suffix("pet").task(t).collect() }
    pub fn get_blood(&self, f: &BidsFile) -> Result<Option<Vec<BloodSample>>> {
        // blood files have recording- entity, try to find any matching blood file
        let p = f.companion("blood", "tsv");
        if p.exists() { return Ok(Some(read_blood_tsv(&p)?)); }
        // Try with recording entity variations
        let stem = f.filename.split('.').next().unwrap_or("");
        let base = stem.rsplit_once('_').map(|(b, _)| b).unwrap_or(stem);
        for entry in std::fs::read_dir(&f.dirname).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.contains("blood") && name.ends_with(".tsv") && name.starts_with(&base[..base.len().min(10)]) {
                return Ok(Some(read_blood_tsv(&entry.path())?));
            }
        }
        Ok(None)
    }
    pub fn get_events(&self, f: &BidsFile) -> Result<Option<Vec<bids_eeg::EegEvent>>> {
        bids_core::try_read_companion(&f.companion("events", "tsv"), bids_eeg::read_events_tsv)
    }
    pub fn get_metadata(&self, f: &BidsFile) -> Result<Option<PetMetadata>> { Ok(self.layout.get_metadata(&f.path)?.deserialize_as()) }

    /// Load the PET NIfTI image data.
    pub fn read_image(&self, f: &BidsFile) -> std::result::Result<bids_nifti::NiftiImage, bids_nifti::NiftiError> {
        bids_nifti::NiftiImage::from_file(&f.path)
    }
    /// Read only the NIfTI header (fast, no data loading).
    pub fn read_header(&self, f: &BidsFile) -> std::result::Result<bids_nifti::NiftiHeader, bids_nifti::NiftiError> {
        bids_nifti::NiftiHeader::from_file(&f.path)
    }
    pub fn get_pet_subjects(&self) -> Result<Vec<String>> { self.layout.get().suffix("pet").return_unique("subject") }
    pub fn get_pet_tasks(&self) -> Result<Vec<String>> { self.layout.get().suffix("pet").return_unique("task") }
    pub fn summary(&self) -> Result<PetSummary> {
        let files = self.get_pet_files()?;
        let subjects = self.get_pet_subjects()?;
        let md = files.first().and_then(|f| self.get_metadata(f).ok().flatten());
        let tracer = md.as_ref().and_then(|m| m.tracer_name.clone());
        let dur = md.as_ref().and_then(metadata::PetMetadata::total_duration);
        Ok(PetSummary { n_subjects: subjects.len(), n_scans: files.len(), subjects, tracer, total_duration: dur })
    }
}
#[derive(Debug)]
pub struct PetSummary { pub n_subjects: usize, pub n_scans: usize, pub subjects: Vec<String>, pub tracer: Option<String>, pub total_duration: Option<f64> }
impl std::fmt::Display for PetSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PET Summary: {} subjects, {} scans", self.n_subjects, self.n_scans)?;
        if let Some(ref t) = self.tracer { writeln!(f, "  Tracer: {t}")?; }
        if let Some(d) = self.total_duration { writeln!(f, "  Duration: {d:.0}s")?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pet_metadata() {
        let json = r#"{"TracerName":"CIMBI-36","TracerRadionuclide":"C11","InjectedRadioactivity":573,"FrameDuration":[10,20,30]}"#;
        let md: PetMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.tracer_name.as_deref(), Some("CIMBI-36"));
        assert_eq!(md.total_duration(), Some(60.0));
    }
}
