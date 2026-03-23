#![deny(unsafe_code)]
//! Perfusion imaging (ASL) support for BIDS datasets.
//!
//! Provides access to Arterial Spin Labeling data, M0 scans, ASL context
//! files, and perfusion-specific metadata (labeling type, post-labeling delay,
//! background suppression, etc.).
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html#arterial-spin-labeling-perfusion-data>

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_core::metadata::BidsMetadata;
use bids_layout::BidsLayout;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PerfMetadata {
    #[serde(default)]
    pub arterial_spin_labeling_type: Option<String>,
    #[serde(default)]
    pub post_labeling_delay: Option<serde_json::Value>,
    #[serde(default)]
    pub background_suppression: Option<bool>,
    #[serde(default)]
    pub m0_type: Option<String>,
    #[serde(default)]
    pub repetition_time_preparation: Option<f64>,
    #[serde(default)]
    pub echo_time: Option<f64>,
    #[serde(default)]
    pub flip_angle: Option<f64>,
    #[serde(default)]
    pub labeling_duration: Option<f64>,
    #[serde(default)]
    pub total_acquired_pairs: Option<u32>,
    #[serde(default)]
    pub vascular_crushing: Option<bool>,
    #[serde(default)]
    pub manufacturer: Option<String>,
}
impl PerfMetadata {
    pub fn from_metadata(md: &BidsMetadata) -> Option<Self> {
        md.deserialize_as()
    }
}

pub struct PerfLayout<'a> {
    layout: &'a BidsLayout,
}
impl<'a> PerfLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self {
        Self { layout }
    }
    pub fn get_asl_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("asl").collect()
    }
    pub fn get_asl_files_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("asl").subject(s).collect()
    }
    pub fn get_m0scan_files(&self) -> Result<Vec<BidsFile>> {
        self.layout.get().suffix("m0scan").collect()
    }
    pub fn get_aslcontext(&self, f: &BidsFile) -> Result<Option<Vec<String>>> {
        let p = f.companion("aslcontext", "tsv");
        if !p.exists() {
            return Ok(None);
        }
        let rows = bids_io::tsv::read_tsv(&p)?;
        Ok(Some(
            rows.iter()
                .filter_map(|r| r.get("volume_type").cloned())
                .collect(),
        ))
    }
    pub fn get_metadata(&self, f: &BidsFile) -> Result<Option<PerfMetadata>> {
        Ok(self.layout.get_metadata(&f.path)?.deserialize_as())
    }

    /// Load the ASL/M0 NIfTI image data.
    pub fn read_image(
        &self,
        f: &BidsFile,
    ) -> std::result::Result<bids_nifti::NiftiImage, bids_nifti::NiftiError> {
        bids_nifti::NiftiImage::from_file(&f.path)
    }
    /// Read only the NIfTI header (fast, no data loading).
    pub fn read_header(
        &self,
        f: &BidsFile,
    ) -> std::result::Result<bids_nifti::NiftiHeader, bids_nifti::NiftiError> {
        bids_nifti::NiftiHeader::from_file(&f.path)
    }

    pub fn get_perf_subjects(&self) -> Result<Vec<String>> {
        self.layout.get().suffix("asl").return_unique("subject")
    }
    pub fn summary(&self) -> Result<PerfSummary> {
        let files = self.get_asl_files()?;
        let subjects = self.get_perf_subjects()?;
        let md = files
            .first()
            .and_then(|f| self.get_metadata(f).ok().flatten());
        let asl_type = md.and_then(|m| m.arterial_spin_labeling_type);
        Ok(PerfSummary {
            n_subjects: subjects.len(),
            n_scans: files.len(),
            subjects,
            asl_type,
        })
    }
}

#[derive(Debug)]
pub struct PerfSummary {
    pub n_subjects: usize,
    pub n_scans: usize,
    pub subjects: Vec<String>,
    pub asl_type: Option<String>,
}
impl std::fmt::Display for PerfSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Perfusion Summary: {} subjects, {} scans",
            self.n_subjects, self.n_scans
        )?;
        if let Some(ref t) = self.asl_type {
            writeln!(f, "  ASL Type: {t}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_perf_metadata() {
        let json = r#"{"ArterialSpinLabelingType":"PCASL","PostLabelingDelay":1.8,"BackgroundSuppression":true}"#;
        let md: PerfMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.arterial_spin_labeling_type.as_deref(), Some("PCASL"));
        assert_eq!(md.background_suppression, Some(true));
    }
}
