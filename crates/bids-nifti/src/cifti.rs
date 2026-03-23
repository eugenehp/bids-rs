//! CIFTI-2 header parsing (`.dtseries.nii`, `.dscalar.nii`, `.dlabel.nii`).
//!
//! CIFTI-2 files are NIfTI-2 files with an XML extension that describes
//! brain models (surface vertices + volume voxels) mapped to matrix dimensions.
//!
//! This module reads the NIfTI-2 header and extracts the CIFTI XML metadata
//! to determine the matrix structure (number of time points, brain models, etc.).

use bids_core::error::{BidsError, Result};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Parsed CIFTI-2 header information.
#[derive(Debug, Clone)]
pub struct CiftiHeader {
    /// Matrix dimensions from the NIfTI header.
    pub dimensions: Vec<usize>,
    /// The CIFTI intent code (e.g., 3002 = ConnDenseSeries, 3006 = ConnDenseScalar).
    pub intent_code: i16,
    /// Human-readable intent name.
    pub intent_name: String,
    /// Number of time points (for dtseries) or scalars (for dscalar).
    pub n_maps: usize,
    /// Total number of brain ordinates (vertices + voxels).
    pub n_brainordinates: usize,
    /// The raw CIFTI XML extension string (for advanced parsing).
    pub cifti_xml: String,
    /// Brain model info extracted from the XML.
    pub brain_models: Vec<BrainModel>,
}

/// A brain model mapping surface vertices or volume voxels to matrix indices.
#[derive(Debug, Clone)]
pub struct BrainModel {
    /// Type: `"CIFTI_MODEL_TYPE_SURFACE"` or `"CIFTI_MODEL_TYPE_VOXELS"`.
    pub model_type: String,
    /// Brain structure name (e.g., `"CIFTI_STRUCTURE_CORTEX_LEFT"`).
    pub brain_structure: String,
    /// Number of vertices or voxels in this model.
    pub count: usize,
    /// Surface number of vertices (for surface models).
    pub surface_number_of_vertices: Option<usize>,
}

/// Read CIFTI-2 header and XML metadata from a `.dtseries.nii` or similar file.
///
/// # Errors
///
/// Returns an error if the file is not a valid NIfTI-2 file or doesn't
/// contain a CIFTI extension.
pub fn read_cifti_header(path: &Path) -> Result<CiftiHeader> {
    let nifti_header = crate::NiftiHeader::from_file(path)
        .map_err(|e| BidsError::DataFormat(format!("Cannot read NIfTI header: {e}")))?;

    let intent_code = nifti_header.sform_code;
    let n_maps = nifti_header.dim[5].max(1) as usize;
    let n_brainordinates = nifti_header.dim[6].max(1) as usize;

    // Read NIfTI extensions — look for CIFTI XML (ecode = 32)
    let vox_offset = nifti_header.vox_offset;
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    reader.seek(SeekFrom::Start(if nifti_header.version == 2 { 544 } else { 352 }))?;

    let mut cifti_xml = String::new();

    // Extensions start after header, before vox_offset
    let mut pos = if nifti_header.version == 2 { 544u64 } else { 352u64 };
    while pos + 8 < vox_offset {
        let mut ext_header = [0u8; 8];
        if reader.read_exact(&mut ext_header).is_err() {
            break;
        }
        let esize = i32::from_le_bytes([ext_header[0], ext_header[1], ext_header[2], ext_header[3]]) as usize;
        let ecode = i32::from_le_bytes([ext_header[4], ext_header[5], ext_header[6], ext_header[7]]);

        if esize < 8 {
            break;
        }

        if ecode == 32 {
            // CIFTI extension
            let data_len = esize - 8;
            let mut xml_bytes = vec![0u8; data_len];
            reader.read_exact(&mut xml_bytes)?;
            cifti_xml = String::from_utf8_lossy(&xml_bytes)
                .trim_end_matches('\0')
                .to_string();
            break;
        } else {
            // Skip this extension
            reader.seek(SeekFrom::Current((esize - 8) as i64))?;
        }
        pos += esize as u64;
    }

    if cifti_xml.is_empty() {
        return Err(BidsError::DataFormat(format!(
            "{}: No CIFTI XML extension found (ecode=32). This may not be a CIFTI file.",
            path.display()
        )));
    }

    // Parse brain models from the XML
    let brain_models = parse_brain_models(&cifti_xml);

    // Determine actual intent from XML
    let intent_name = if cifti_xml.contains("CIFTI_INDEX_TYPE_SERIES") {
        "ConnDenseSeries"
    } else if cifti_xml.contains("CIFTI_INDEX_TYPE_SCALARS") {
        "ConnDenseScalar"
    } else if cifti_xml.contains("CIFTI_INDEX_TYPE_LABELS") {
        "ConnDenseLabel"
    } else if cifti_xml.contains("CIFTI_INDEX_TYPE_PARCELS") {
        "ConnParcels"
    } else {
        "Unknown"
    };

    let dimensions: Vec<usize> = nifti_header.dim[1..=7]
        .iter()
        .take_while(|&&d| d > 0)
        .map(|&d| d as usize)
        .collect();

    Ok(CiftiHeader {
        dimensions,
        intent_code,
        intent_name: intent_name.to_string(),
        n_maps,
        n_brainordinates,
        cifti_xml,
        brain_models,
    })
}

fn parse_brain_models(xml: &str) -> Vec<BrainModel> {
    let mut models = Vec::new();

    for bm_start in xml.match_indices("<BrainModel") {
        let start = bm_start.0;
        let end = xml[start..].find("/>")
            .or_else(|| xml[start..].find('>'))
            .map(|e| start + e)
            .unwrap_or(xml.len());
        let attrs = &xml[start..end];

        let model_type = extract_str_attr(attrs, "ModelType").unwrap_or_default();
        let brain_structure = extract_str_attr(attrs, "BrainStructure").unwrap_or_default();
        let count = extract_usize_attr(attrs, "IndexCount").unwrap_or(0);
        let surface_verts = extract_usize_attr(attrs, "SurfaceNumberOfVertices");

        models.push(BrainModel {
            model_type,
            brain_structure,
            count,
            surface_number_of_vertices: surface_verts,
        });
    }

    models
}

fn extract_str_attr(attrs: &str, name: &str) -> Option<String> {
    let pattern = format!("{}=\"", name);
    let start = attrs.find(&pattern)? + pattern.len();
    let end = attrs[start..].find('"')? + start;
    Some(attrs[start..end].to_string())
}

fn extract_usize_attr(attrs: &str, name: &str) -> Option<usize> {
    extract_str_attr(attrs, name)?.parse().ok()
}

impl std::fmt::Display for CiftiHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CIFTI-2 {}: {} maps × {} brainordinates, {} brain models",
            self.intent_name, self.n_maps, self.n_brainordinates, self.brain_models.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_brain_models() {
        let xml = r#"
            <BrainModel ModelType="CIFTI_MODEL_TYPE_SURFACE"
                        BrainStructure="CIFTI_STRUCTURE_CORTEX_LEFT"
                        IndexCount="29696"
                        SurfaceNumberOfVertices="32492"/>
            <BrainModel ModelType="CIFTI_MODEL_TYPE_SURFACE"
                        BrainStructure="CIFTI_STRUCTURE_CORTEX_RIGHT"
                        IndexCount="29716"
                        SurfaceNumberOfVertices="32492"/>
        "#;
        let models = parse_brain_models(xml);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].count, 29696);
        assert_eq!(models[0].surface_number_of_vertices, Some(32492));
        assert!(models[0].brain_structure.contains("LEFT"));
    }
}
