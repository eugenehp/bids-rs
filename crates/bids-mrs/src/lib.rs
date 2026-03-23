#![deny(unsafe_code)]
//! Magnetic Resonance Spectroscopy (MRS) support for BIDS datasets.
//!
//! Provides access to single-voxel spectroscopy (SVS) and MR spectroscopic
//! imaging (MRSI) data files and metadata including resonant nucleus, spectral
//! width, echo time, and repetition time.
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-spectroscopy.html>

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_core::metadata::BidsMetadata;
use bids_layout::BidsLayout;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct MrsMetadata {
    #[serde(default)] pub resonant_nucleus: Option<String>,
    #[serde(default)] pub spectral_width: Option<f64>,
    #[serde(default)] pub echo_time: Option<f64>,
    #[serde(default)] pub repetition_time: Option<f64>,
    #[serde(default)] pub number_of_spectral_points: Option<u32>,
    #[serde(default)] pub mixing_time: Option<f64>,
    #[serde(default)] pub flip_angle: Option<f64>,
    #[serde(default)] pub manufacturer: Option<String>,
    #[serde(default)] pub task_name: Option<String>,
}
impl MrsMetadata { pub fn from_metadata(md: &BidsMetadata) -> Option<Self> { md.deserialize_as() } }

pub struct MrsLayout<'a> { layout: &'a BidsLayout }
impl<'a> MrsLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self { Self { layout } }
    pub fn get_svs_files(&self) -> Result<Vec<BidsFile>> { self.layout.get().suffix("svs").collect() }
    pub fn get_mrsi_files(&self) -> Result<Vec<BidsFile>> { self.layout.get().suffix("mrsi").collect() }
    pub fn get_svs_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> { self.layout.get().suffix("svs").subject(s).collect() }
    pub fn get_mrsi_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> { self.layout.get().suffix("mrsi").subject(s).collect() }
    pub fn get_metadata(&self, f: &BidsFile) -> Result<Option<MrsMetadata>> { Ok(self.layout.get_metadata(&f.path)?.deserialize_as()) }

    /// Load the MRS NIfTI image data (NIfTI-MRS format).
    pub fn read_image(&self, f: &BidsFile) -> std::result::Result<bids_nifti::NiftiImage, bids_nifti::NiftiError> {
        bids_nifti::NiftiImage::from_file(&f.path)
    }
    /// Read only the NIfTI header (fast, no data loading).
    pub fn read_header(&self, f: &BidsFile) -> std::result::Result<bids_nifti::NiftiHeader, bids_nifti::NiftiError> {
        bids_nifti::NiftiHeader::from_file(&f.path)
    }

    pub fn get_mrs_subjects(&self) -> Result<Vec<String>> {
        let mut s = self.layout.get().suffix("svs").return_unique("subject")?;
        s.extend(self.layout.get().suffix("mrsi").return_unique("subject")?);
        s.sort(); s.dedup(); Ok(s)
    }
    pub fn get_mrs_tasks(&self) -> Result<Vec<String>> {
        let mut t = self.layout.get().suffix("svs").return_unique("task")?;
        t.extend(self.layout.get().suffix("mrsi").return_unique("task")?);
        t.sort(); t.dedup(); Ok(t)
    }
    pub fn summary(&self) -> Result<MrsSummary> {
        let svs = self.get_svs_files()?;
        let mrsi = self.get_mrsi_files()?;
        let subjects = self.get_mrs_subjects()?;
        let md = svs.first().or(mrsi.first()).and_then(|f| self.get_metadata(f).ok().flatten());
        let nucleus = md.and_then(|m| m.resonant_nucleus);
        Ok(MrsSummary { n_subjects: subjects.len(), n_svs: svs.len(), n_mrsi: mrsi.len(), subjects, nucleus })
    }
}

#[derive(Debug)]
pub struct MrsSummary { pub n_subjects: usize, pub n_svs: usize, pub n_mrsi: usize, pub subjects: Vec<String>, pub nucleus: Option<String> }
impl std::fmt::Display for MrsSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MRS Summary: {} subjects, {} SVS, {} MRSI", self.n_subjects, self.n_svs, self.n_mrsi)?;
        if let Some(ref n) = self.nucleus { writeln!(f, "  Nucleus: {n}")?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mrs_metadata() {
        let json = r#"{"ResonantNucleus":"1H","SpectralWidth":2000,"EchoTime":0.03,"RepetitionTime":2.0}"#;
        let md: MrsMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.resonant_nucleus.as_deref(), Some("1H"));
        assert_eq!(md.spectral_width, Some(2000.0));
    }
}
