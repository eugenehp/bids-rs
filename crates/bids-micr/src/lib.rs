#![deny(unsafe_code)]
//! Microscopy support for BIDS datasets.
//!
//! Provides access to microscopy image files and metadata including pixel size,
//! magnification, sample staining/fixation/embedding, and slice thickness.
//!
//! ## Feature flags
//!
//! - **`tiff`** — Enable TIFF/OME-TIFF image reading. Pure Rust, no system dependencies.
//!   Adds the `data` module with `MicrImage`, `read_tiff`, and `read_tiff_stack`.
//!
//! See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/microscopy.html>

pub mod ome;
#[cfg(feature = "tiff")]
pub mod data;

pub use ome::{OmeMetadata, read_ome_metadata};

use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_core::metadata::BidsMetadata;
use bids_layout::BidsLayout;
use serde::{Deserialize, Serialize};

#[cfg(feature = "tiff")]
pub use data::{MicrImage, read_tiff, read_tiff_stack};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct MicrMetadata {
    #[serde(default)] pub pixel_size: Option<Vec<f64>>,
    #[serde(default)] pub pixel_size_units: Option<String>,
    #[serde(default)] pub magnification: Option<f64>,
    #[serde(default)] pub sample_environment: Option<String>,
    #[serde(default)] pub sample_staining: Option<String>,
    #[serde(default)] pub sample_primary_antibody: Option<String>,
    #[serde(default)] pub sample_embedding: Option<String>,
    #[serde(default)] pub sample_fixation: Option<String>,
    #[serde(default)] pub slice_thickness: Option<f64>,
    #[serde(default)] pub manufacturer: Option<String>,
}
impl MicrMetadata { pub fn from_metadata(md: &BidsMetadata) -> Option<Self> { md.deserialize_as() } }

pub struct MicrLayout<'a> { layout: &'a BidsLayout }
impl<'a> MicrLayout<'a> {
    pub fn new(layout: &'a BidsLayout) -> Self { Self { layout } }
    pub fn get_micr_files(&self) -> Result<Vec<BidsFile>> { self.layout.get().datatype("micr").collect() }
    pub fn get_micr_files_for_subject(&self, s: &str) -> Result<Vec<BidsFile>> { self.layout.get().datatype("micr").subject(s).collect() }
    pub fn get_metadata(&self, f: &BidsFile) -> Result<Option<MicrMetadata>> { Ok(self.layout.get_metadata(&f.path)?.deserialize_as()) }

    /// Read a microscopy image from a TIFF/OME-TIFF file.
    ///
    /// Requires the `tiff` feature flag.
    #[cfg(feature = "tiff")]
    pub fn read_image(&self, f: &BidsFile) -> Result<MicrImage> {
        read_tiff(&f.path)
    }

    /// Read all pages from a multi-page TIFF stack.
    ///
    /// Requires the `tiff` feature flag.
    #[cfg(feature = "tiff")]
    pub fn read_image_stack(&self, f: &BidsFile) -> Result<Vec<MicrImage>> {
        read_tiff_stack(&f.path)
    }
    pub fn get_micr_subjects(&self) -> Result<Vec<String>> { self.layout.get().datatype("micr").return_unique("subject") }
    pub fn get_samples(&self) -> Result<Vec<String>> { self.layout.get().datatype("micr").return_unique("sample") }
    pub fn get_stainings(&self) -> Result<Vec<String>> { self.layout.get().datatype("micr").return_unique("staining") }
    pub fn summary(&self) -> Result<MicrSummary> {
        let files = self.get_micr_files()?;
        let subjects = self.get_micr_subjects()?;
        let samples = self.get_samples()?;
        let stainings = self.get_stainings()?;
        let md = files.first().and_then(|f| self.get_metadata(f).ok().flatten());
        let mag = md.and_then(|m| m.magnification);
        Ok(MicrSummary { n_subjects: subjects.len(), n_images: files.len(), subjects, samples, stainings, magnification: mag })
    }
}

#[derive(Debug)]
pub struct MicrSummary { pub n_subjects: usize, pub n_images: usize, pub subjects: Vec<String>, pub samples: Vec<String>, pub stainings: Vec<String>, pub magnification: Option<f64> }
impl std::fmt::Display for MicrSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Microscopy Summary: {} subjects, {} images, {} samples", self.n_subjects, self.n_images, self.samples.len())?;
        if !self.stainings.is_empty() { writeln!(f, "  Stainings: {:?}", self.stainings)?; }
        if let Some(m) = self.magnification { writeln!(f, "  Magnification: {m}x")?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_micr_metadata() {
        let json = r#"{"PixelSize":[0.001,0.001],"PixelSizeUnits":"mm","Magnification":40,"SampleStaining":"LFB"}"#;
        let md: MicrMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(md.magnification, Some(40.0));
        assert_eq!(md.sample_staining.as_deref(), Some("LFB"));
    }
}
