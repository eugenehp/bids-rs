//! GIFTI surface file reading (`.gii`, `.surf.gii`, `.func.gii`).
//!
//! GIFTI (Geometry Format under the Informatics Technology Infrastructure)
//! is an XML-based format for cortical surface meshes and surface-mapped data.
//!
//! This module provides a minimal reader that extracts the metadata and
//! base64-encoded data arrays from GIFTI files without a full XML parser —
//! just regex-based extraction of the key elements.

use bids_core::error::{BidsError, Result};
use std::path::Path;

/// A GIFTI file's parsed contents.
#[derive(Debug, Clone)]
pub struct GiftiImage {
    /// Number of data arrays in the file.
    pub n_arrays: usize,
    /// Metadata key-value pairs from the top-level `<MetaData>` element.
    pub metadata: Vec<(String, String)>,
    /// Intent codes for each data array (e.g., 1008 = POINTSET, 1009 = TRIANGLE).
    pub intents: Vec<i32>,
    /// Data type codes for each array (e.g., 16 = FLOAT32).
    pub data_types: Vec<i32>,
    /// Dimensions for each data array: `[n_rows, n_cols]`.
    pub dimensions: Vec<Vec<usize>>,
    /// Total number of vertices (from the POINTSET array, if present).
    pub n_vertices: Option<usize>,
    /// Total number of faces/triangles (from the TRIANGLE array, if present).
    pub n_faces: Option<usize>,
}

/// Read metadata and structure from a GIFTI XML file.
///
/// Parses the XML text to extract data array attributes (intent, datatype,
/// dimensions) and top-level metadata. Does not decode the base64 data
/// arrays themselves — use this for inspection and validation.
///
/// # Errors
///
/// Returns an error if the file can't be read or doesn't look like GIFTI XML.
pub fn read_gifti_header(path: &Path) -> Result<GiftiImage> {
    let text = std::fs::read_to_string(path)?;

    if !text.contains("<GIFTI") {
        return Err(BidsError::DataFormat(format!(
            "{}: Not a GIFTI file (no <GIFTI> root element)",
            path.display()
        )));
    }

    let mut intents = Vec::new();
    let mut data_types = Vec::new();
    let mut dimensions = Vec::new();
    let mut metadata = Vec::new();
    let mut n_vertices = None;
    let mut n_faces = None;

    // Parse DataArray elements
    for da_match in text.match_indices("<DataArray") {
        let start = da_match.0;
        let end = text[start..].find('>').map(|e| start + e).unwrap_or(text.len());
        let attrs = &text[start..end];

        let intent = extract_int_attr(attrs, "Intent").unwrap_or(0);
        let dtype = extract_int_attr(attrs, "DataType").unwrap_or(0);

        let dim0 = extract_int_attr(attrs, "Dim0").unwrap_or(0) as usize;
        let dim1 = extract_int_attr(attrs, "Dim1");

        let mut dims = vec![dim0];
        if let Some(d1) = dim1 {
            dims.push(d1 as usize);
        }

        // NIFTI_INTENT_POINTSET = 1008, NIFTI_INTENT_TRIANGLE = 1009
        match intent {
            1008 => n_vertices = Some(dim0),
            1009 => n_faces = Some(dim0),
            _ => {}
        }

        intents.push(intent);
        data_types.push(dtype);
        dimensions.push(dims);
    }

    // Parse top-level metadata
    if let Some(md_start) = text.find("<MetaData>") {
        let md_end = text[md_start..].find("</MetaData>").unwrap_or(0) + md_start;
        let md_section = &text[md_start..md_end];

        let mut pos = 0;
        while let Some(name_start) = md_section[pos..].find("<Name>") {
            let name_s = pos + name_start + 6;
            let name_e = md_section[name_s..].find("</Name>").map(|e| name_s + e).unwrap_or(name_s);
            let name = md_section[name_s..name_e].trim().to_string();

            if let Some(val_start) = md_section[name_e..].find("<Value>") {
                let val_s = name_e + val_start + 7;
                let val_e = md_section[val_s..].find("</Value>").map(|e| val_s + e).unwrap_or(val_s);
                let value = md_section[val_s..val_e].trim().to_string();
                metadata.push((name, value));
                pos = val_e;
            } else {
                pos = name_e;
            }
        }
    }

    Ok(GiftiImage {
        n_arrays: intents.len(),
        metadata,
        intents,
        data_types,
        dimensions,
        n_vertices,
        n_faces,
    })
}

fn extract_int_attr(attrs: &str, name: &str) -> Option<i32> {
    let pattern = format!("{}=\"", name);
    let start = attrs.find(&pattern)? + pattern.len();
    let end = attrs[start..].find('"')? + start;
    attrs[start..end].parse().ok()
}

impl std::fmt::Display for GiftiImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GIFTI({} arrays", self.n_arrays)?;
        if let Some(v) = self.n_vertices {
            write!(f, ", {v} vertices")?;
        }
        if let Some(t) = self.n_faces {
            write!(f, ", {t} faces")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_gifti_header() {
        let dir = std::env::temp_dir().join("bids_gifti_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.surf.gii");

        std::fs::write(&path, r#"<?xml version="1.0" encoding="UTF-8"?>
<GIFTI Version="1.0">
  <MetaData>
    <MD><Name>AnatomicalStructurePrimary</Name><Value>CortexLeft</Value></MD>
  </MetaData>
  <DataArray Intent="1008" DataType="16" Dim0="32492" Dim1="3" Encoding="Base64Binary" Endian="LittleEndian">
    <Data>AAAA</Data>
  </DataArray>
  <DataArray Intent="1009" DataType="8" Dim0="64980" Dim1="3" Encoding="Base64Binary" Endian="LittleEndian">
    <Data>AAAA</Data>
  </DataArray>
</GIFTI>
"#).unwrap();

        let img = read_gifti_header(&path).unwrap();
        assert_eq!(img.n_arrays, 2);
        assert_eq!(img.n_vertices, Some(32492));
        assert_eq!(img.n_faces, Some(64980));
        assert_eq!(img.intents, vec![1008, 1009]);
        assert!(!img.metadata.is_empty());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_not_gifti() {
        let dir = std::env::temp_dir().join("bids_gifti_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.gii");
        std::fs::write(&path, "not a gifti file").unwrap();

        assert!(read_gifti_header(&path).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
