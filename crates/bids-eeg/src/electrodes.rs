//! Electrode positions from `_electrodes.tsv`.
//!
//! Provides [`Electrode`] with name, 3D coordinates (x/y/z), material,
//! and impedance fields as defined by the BIDS-EEG specification.

use bids_core::error::{BidsError, Result};
use bids_io::tsv::read_tsv;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// An electrode position from _electrodes.tsv.
///
/// See: <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html>
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Electrode {
    /// Electrode name (must match channel name).
    pub name: String,
    /// X coordinate.
    pub x: Option<f64>,
    /// Y coordinate.
    pub y: Option<f64>,
    /// Z coordinate.
    pub z: Option<f64>,
    /// Type of electrode material.
    pub material: Option<String>,
    /// Impedance of the electrode in kOhm.
    pub impedance: Option<f64>,
}

impl Electrode {
    /// Whether this electrode has valid 3D coordinates.
    pub fn has_position(&self) -> bool {
        self.x.is_some() && self.y.is_some() && self.z.is_some()
    }

    /// Get position as (x, y, z) tuple.
    pub fn position(&self) -> Option<(f64, f64, f64)> {
        match (self.x, self.y, self.z) {
            (Some(x), Some(y), Some(z)) => Some((x, y, z)),
            _ => None,
        }
    }
}

/// Read electrodes from a BIDS _electrodes.tsv file.
pub fn read_electrodes_tsv(path: &Path) -> Result<Vec<Electrode>> {
    let rows = read_tsv(path)?;
    let mut electrodes = Vec::with_capacity(rows.len());

    for row in &rows {
        let name = row
            .get("name")
            .ok_or_else(|| BidsError::Csv("Missing 'name' column in electrodes.tsv".into()))?
            .trim()
            .to_string();

        let electrode = Electrode {
            name,
            x: parse_f64(row.get("x")),
            y: parse_f64(row.get("y")),
            z: parse_f64(row.get("z")),
            material: non_empty(row.get("material")),
            impedance: parse_f64(row.get("impedance")),
        };
        electrodes.push(electrode);
    }

    Ok(electrodes)
}

fn non_empty(val: Option<&String>) -> Option<String> {
    val.filter(|s| !s.is_empty() && *s != "n/a")
        .map(|s| s.trim().to_string())
}

fn parse_f64(val: Option<&String>) -> Option<f64> {
    val.and_then(|s| {
        let s = s.trim();
        if s.is_empty() || s == "n/a" {
            None
        } else {
            s.parse().ok()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_electrodes_tsv() {
        let dir = std::env::temp_dir().join("bids_eeg_electrodes_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("electrodes.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "name\tx\ty\tz").unwrap();
        writeln!(f, "Fp1\t-0.03\t0.08\t0.01").unwrap();
        writeln!(f, "Fp2\t0.03\t0.08\t0.01").unwrap();

        let electrodes = read_electrodes_tsv(&path).unwrap();
        assert_eq!(electrodes.len(), 2);
        assert_eq!(electrodes[0].name, "Fp1");
        assert!(electrodes[0].has_position());
        assert_eq!(electrodes[0].position(), Some((-0.03, 0.08, 0.01)));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
