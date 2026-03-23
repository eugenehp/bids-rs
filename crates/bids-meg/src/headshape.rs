//! MEG headshape digitization point parsing (`.pos` files).
//!
//! The `_headshape.pos` file contains 3D coordinates of digitized head points
//! used for MEG-MRI coregistration. Each line has a label and x/y/z coordinates.

use bids_core::error::{BidsError, Result};
use std::path::Path;

/// A single digitization point from a headshape file.
#[derive(Debug, Clone, PartialEq)]
pub struct DigPoint {
    /// Point label (e.g., "nasion", "lpa", "rpa", or numeric index).
    pub label: String,
    /// X coordinate in mm.
    pub x: f64,
    /// Y coordinate in mm.
    pub y: f64,
    /// Z coordinate in mm.
    pub z: f64,
    /// Point category if detectable (fiducial, extra, EEG, HPI).
    pub kind: PointKind,
}

/// Category of a digitization point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointKind {
    /// Anatomical fiducial (nasion, LPA, RPA).
    Fiducial,
    /// Head Position Indicator coil.
    Hpi,
    /// EEG electrode position.
    Eeg,
    /// Extra head surface point (for coregistration).
    Extra,
}

/// Read a headshape `.pos` file.
///
/// The format is whitespace-delimited with columns: label x y z
/// (some files omit labels, in which case indices are used).
///
/// # Errors
///
/// Returns an error if the file can't be read or contains malformed lines.
pub fn read_headshape_pos(path: &Path) -> Result<Vec<DigPoint>> {
    let contents = std::fs::read_to_string(path)?;
    let mut points = Vec::new();

    for (line_no, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('%') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        let (label, x, y, z) = if parts.len() >= 4 {
            // label x y z
            let x = parts[1].parse::<f64>().map_err(|_| parse_err(path, line_no))?;
            let y = parts[2].parse::<f64>().map_err(|_| parse_err(path, line_no))?;
            let z = parts[3].parse::<f64>().map_err(|_| parse_err(path, line_no))?;
            (parts[0].to_string(), x, y, z)
        } else if parts.len() == 3 {
            // x y z (no label)
            let x = parts[0].parse::<f64>().map_err(|_| parse_err(path, line_no))?;
            let y = parts[1].parse::<f64>().map_err(|_| parse_err(path, line_no))?;
            let z = parts[2].parse::<f64>().map_err(|_| parse_err(path, line_no))?;
            (format!("{}", points.len() + 1), x, y, z)
        } else {
            continue; // skip malformed lines
        };

        let kind = classify_point(&label);
        points.push(DigPoint { label, x, y, z, kind });
    }

    Ok(points)
}

fn classify_point(label: &str) -> PointKind {
    let lower = label.to_lowercase();
    if lower == "nasion" || lower == "nas" || lower == "lpa" || lower == "rpa"
        || lower == "nz" || lower == "left" || lower == "right"
    {
        PointKind::Fiducial
    } else if lower.starts_with("hpi") || lower.starts_with("coil") {
        PointKind::Hpi
    } else if lower.starts_with("eeg") || lower.starts_with("e") && lower.len() <= 4 {
        PointKind::Eeg
    } else {
        PointKind::Extra
    }
}

fn parse_err(path: &Path, line: usize) -> BidsError {
    BidsError::DataFormat(format!(
        "Cannot parse headshape file {} at line {}",
        path.display(),
        line + 1,
    ))
}

/// Count of points by category.
#[must_use]
pub fn count_by_kind(points: &[DigPoint]) -> (usize, usize, usize, usize) {
    let mut fid = 0;
    let mut hpi = 0;
    let mut eeg = 0;
    let mut extra = 0;
    for p in points {
        match p.kind {
            PointKind::Fiducial => fid += 1,
            PointKind::Hpi => hpi += 1,
            PointKind::Eeg => eeg += 1,
            PointKind::Extra => extra += 1,
        }
    }
    (fid, hpi, eeg, extra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_headshape_pos() {
        let dir = std::env::temp_dir().join("bids_headshape_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.pos");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "nasion 0.0 85.0 -40.0").unwrap();
        writeln!(f, "lpa -80.0 0.0 -40.0").unwrap();
        writeln!(f, "rpa 80.0 0.0 -40.0").unwrap();
        writeln!(f, "1 10.0 20.0 30.0").unwrap();
        writeln!(f, "2 11.0 21.0 31.0").unwrap();

        let points = read_headshape_pos(&path).unwrap();
        assert_eq!(points.len(), 5);
        assert_eq!(points[0].kind, PointKind::Fiducial);
        assert_eq!(points[0].label, "nasion");
        assert!((points[0].y - 85.0).abs() < 1e-10);
        assert_eq!(points[3].kind, PointKind::Extra);

        let (fid, _, _, extra) = count_by_kind(&points);
        assert_eq!(fid, 3);
        assert_eq!(extra, 2);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_no_labels() {
        let dir = std::env::temp_dir().join("bids_headshape_nolabel");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.pos");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "10.0 20.0 30.0").unwrap();
        writeln!(f, "11.0 21.0 31.0").unwrap();

        let points = read_headshape_pos(&path).unwrap();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].label, "1");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
