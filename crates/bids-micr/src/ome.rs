//! OME-XML metadata parsing from OME-TIFF files.
//!
//! OME-TIFF files embed an XML string in the TIFF ImageDescription tag
//! that describes the image dimensions, pixel sizes, channel information,
//! and acquisition metadata following the [OME data model](https://www.openmicroscopy.org/ome-model/).
//!
//! This module extracts and parses that XML to provide typed access to
//! the most common metadata fields.

use bids_core::error::{BidsError, Result};
use std::path::Path;

/// Parsed OME-XML metadata from an OME-TIFF file.
#[derive(Debug, Clone)]
pub struct OmeMetadata {
    /// Image name.
    pub name: Option<String>,
    /// Number of pixels in X.
    pub size_x: Option<usize>,
    /// Number of pixels in Y.
    pub size_y: Option<usize>,
    /// Number of Z slices.
    pub size_z: Option<usize>,
    /// Number of channels.
    pub size_c: Option<usize>,
    /// Number of time points.
    pub size_t: Option<usize>,
    /// Pixel size in X (micrometers).
    pub physical_size_x: Option<f64>,
    /// Pixel size in Y (micrometers).
    pub physical_size_y: Option<f64>,
    /// Pixel size in Z (micrometers).
    pub physical_size_z: Option<f64>,
    /// Pixel type (e.g., "uint8", "uint16", "float").
    pub pixel_type: Option<String>,
    /// Channel names.
    pub channel_names: Vec<String>,
    /// The raw OME-XML string.
    pub raw_xml: String,
}

/// Extract OME-XML metadata from an OME-TIFF file.
///
/// Reads the TIFF ImageDescription tag and parses the OME-XML within it.
///
/// # Errors
///
/// Returns an error if the file can't be read, isn't a TIFF, or doesn't
/// contain OME-XML metadata.
pub fn read_ome_metadata(path: &Path) -> Result<OmeMetadata> {
    let bytes = std::fs::read(path)?;

    // Check TIFF magic
    if bytes.len() < 8 {
        return Err(BidsError::DataFormat("File too small to be TIFF".into()));
    }

    let little_endian = match &bytes[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return Err(BidsError::DataFormat("Not a TIFF file".into())),
    };

    // Read first IFD offset
    let ifd_offset = read_u32(&bytes, 4, little_endian) as usize;
    if ifd_offset >= bytes.len() {
        return Err(BidsError::DataFormat("Invalid IFD offset".into()));
    }

    // Read IFD entries to find ImageDescription (tag 270)
    let n_entries = read_u16(&bytes, ifd_offset, little_endian) as usize;
    let mut xml_str = String::new();

    for i in 0..n_entries {
        let entry_offset = ifd_offset + 2 + i * 12;
        if entry_offset + 12 > bytes.len() {
            break;
        }
        let tag = read_u16(&bytes, entry_offset, little_endian);
        if tag == 270 {
            // ImageDescription
            let count = read_u32(&bytes, entry_offset + 4, little_endian) as usize;
            let value_offset = if count <= 4 {
                entry_offset + 8
            } else {
                read_u32(&bytes, entry_offset + 8, little_endian) as usize
            };
            if value_offset + count <= bytes.len() {
                xml_str = String::from_utf8_lossy(&bytes[value_offset..value_offset + count])
                    .trim_end_matches('\0')
                    .to_string();
            }
            break;
        }
    }

    if !xml_str.contains("<OME") && !xml_str.contains("<ome:") {
        return Err(BidsError::DataFormat(format!(
            "{}: No OME-XML found in TIFF ImageDescription tag",
            path.display()
        )));
    }

    parse_ome_xml(&xml_str)
}

fn parse_ome_xml(xml: &str) -> Result<OmeMetadata> {
    let name = extract_xml_attr(xml, "Image", "Name");
    let size_x = extract_xml_attr(xml, "Pixels", "SizeX").and_then(|s| s.parse().ok());
    let size_y = extract_xml_attr(xml, "Pixels", "SizeY").and_then(|s| s.parse().ok());
    let size_z = extract_xml_attr(xml, "Pixels", "SizeZ").and_then(|s| s.parse().ok());
    let size_c = extract_xml_attr(xml, "Pixels", "SizeC").and_then(|s| s.parse().ok());
    let size_t = extract_xml_attr(xml, "Pixels", "SizeT").and_then(|s| s.parse().ok());
    let physical_size_x = extract_xml_attr(xml, "Pixels", "PhysicalSizeX").and_then(|s| s.parse().ok());
    let physical_size_y = extract_xml_attr(xml, "Pixels", "PhysicalSizeY").and_then(|s| s.parse().ok());
    let physical_size_z = extract_xml_attr(xml, "Pixels", "PhysicalSizeZ").and_then(|s| s.parse().ok());
    let pixel_type = extract_xml_attr(xml, "Pixels", "Type");

    // Extract channel names
    let mut channel_names = Vec::new();
    for ch_start in xml.match_indices("<Channel") {
        let start = ch_start.0;
        let end = xml[start..].find("/>")
            .or_else(|| xml[start..].find('>'))
            .map(|e| start + e)
            .unwrap_or(xml.len());
        let attrs = &xml[start..end];
        if let Some(name) = extract_attr(attrs, "Name") {
            channel_names.push(name);
        }
    }

    Ok(OmeMetadata {
        name,
        size_x,
        size_y,
        size_z,
        size_c,
        size_t,
        physical_size_x,
        physical_size_y,
        physical_size_z,
        pixel_type,
        channel_names,
        raw_xml: xml.to_string(),
    })
}

fn extract_xml_attr(xml: &str, element: &str, attr: &str) -> Option<String> {
    let elem_start = xml.find(&format!("<{element}"))?;
    let elem_end = xml[elem_start..].find('>')? + elem_start;
    extract_attr(&xml[elem_start..elem_end], attr)
}

fn extract_attr(tag: &str, name: &str) -> Option<String> {
    let pattern = format!("{name}=\"");
    let start = tag.find(&pattern)? + pattern.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_string())
}

fn read_u16(bytes: &[u8], offset: usize, little_endian: bool) -> u16 {
    if little_endian {
        u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
    } else {
        u16::from_be_bytes([bytes[offset], bytes[offset + 1]])
    }
}

fn read_u32(bytes: &[u8], offset: usize, little_endian: bool) -> u32 {
    if little_endian {
        u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
    } else {
        u32::from_be_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
    }
}

impl std::fmt::Display for OmeMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OME")?;
        if let (Some(x), Some(y)) = (self.size_x, self.size_y) {
            write!(f, " {x}×{y}")?;
            if let Some(z) = self.size_z.filter(|&z| z > 1) {
                write!(f, "×{z}")?;
            }
        }
        if let Some(ref px) = self.physical_size_x {
            write!(f, " ({px:.3} µm/px)")?;
        }
        if !self.channel_names.is_empty() {
            write!(f, " ch: {:?}", self.channel_names)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ome_xml() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image Name="sample-A_FLUO">
    <Pixels SizeX="1024" SizeY="1024" SizeZ="20" SizeC="3" SizeT="1"
            PhysicalSizeX="0.325" PhysicalSizeY="0.325" PhysicalSizeZ="1.0"
            Type="uint16">
      <Channel Name="DAPI"/>
      <Channel Name="GFP"/>
      <Channel Name="mCherry"/>
    </Pixels>
  </Image>
</OME>"#;

        let md = parse_ome_xml(xml).unwrap();
        assert_eq!(md.size_x, Some(1024));
        assert_eq!(md.size_y, Some(1024));
        assert_eq!(md.size_z, Some(20));
        assert_eq!(md.size_c, Some(3));
        assert_eq!(md.physical_size_x, Some(0.325));
        assert_eq!(md.pixel_type.as_deref(), Some("uint16"));
        assert_eq!(md.channel_names, vec!["DAPI", "GFP", "mCherry"]);
        assert_eq!(md.name.as_deref(), Some("sample-A_FLUO"));
    }
}
