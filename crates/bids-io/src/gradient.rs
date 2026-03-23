//! DWI gradient table parsing (`.bval` and `.bvec` files).
//!
//! BIDS diffusion-weighted imaging stores gradient information in two
//! companion files alongside the NIfTI image:
//!
//! - `_dwi.bval` — b-values, one per volume, whitespace-separated
//! - `_dwi.bvec` — gradient directions, 3 rows × N columns
//!
//! This module parses both files into typed structures.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//! use bids_io::gradient::{read_bvals, read_bvecs, GradientTable};
//!
//! let bvals = read_bvals(Path::new("sub-01_dwi.bval")).unwrap();
//! let bvecs = read_bvecs(Path::new("sub-01_dwi.bvec")).unwrap();
//! let table = GradientTable::new(bvals, bvecs).unwrap();
//!
//! println!("{} volumes, {} non-zero", table.n_volumes(), table.n_diffusion_volumes());
//! for (i, (bval, dir)) in table.iter().enumerate() {
//!     println!("vol {i}: b={bval:.0}, dir=[{:.3}, {:.3}, {:.3}]", dir[0], dir[1], dir[2]);
//! }
//! ```

use bids_core::error::{BidsError, Result};
use std::path::Path;

/// A parsed b-value file: one f64 per volume.
pub type Bvals = Vec<f64>;

/// A parsed b-vector file: one [x, y, z] direction per volume.
pub type Bvecs = Vec<[f64; 3]>;

/// Read a `.bval` file — whitespace-separated b-values on one or more lines.
///
/// # Errors
///
/// Returns an error if the file can't be read or contains non-numeric values.
pub fn read_bvals(path: &Path) -> Result<Bvals> {
    let contents = std::fs::read_to_string(path)?;
    let values: std::result::Result<Vec<f64>, _> = contents
        .split_whitespace()
        .map(|s| s.parse::<f64>())
        .collect();
    values.map_err(|e| BidsError::DataFormat(format!("Invalid bval file {}: {e}", path.display())))
}

/// Read a `.bvec` file — 3 rows of whitespace-separated values (x, y, z directions).
///
/// # Errors
///
/// Returns an error if the file can't be read, doesn't have exactly 3 rows,
/// or contains non-numeric values.
pub fn read_bvecs(path: &Path) -> Result<Bvecs> {
    let contents = std::fs::read_to_string(path)?;
    let rows: Vec<Vec<f64>> = contents
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .map(|s| s.parse::<f64>())
                .collect::<std::result::Result<Vec<f64>, _>>()
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| BidsError::DataFormat(format!("Invalid bvec file {}: {e}", path.display())))?;

    if rows.len() != 3 {
        return Err(BidsError::DataFormat(format!(
            "bvec file {} has {} rows, expected 3",
            path.display(),
            rows.len()
        )));
    }

    let n = rows[0].len();
    if rows[1].len() != n || rows[2].len() != n {
        return Err(BidsError::DataFormat(format!(
            "bvec file {} has inconsistent row lengths ({}, {}, {})",
            path.display(),
            rows[0].len(),
            rows[1].len(),
            rows[2].len()
        )));
    }

    let bvecs: Vec<[f64; 3]> = (0..n)
        .map(|i| [rows[0][i], rows[1][i], rows[2][i]])
        .collect();
    Ok(bvecs)
}

/// Combined b-value + b-vector gradient table for a DWI acquisition.
///
/// Provides convenience methods for counting volumes, identifying b=0 images,
/// and iterating over gradient directions.
#[derive(Debug, Clone)]
pub struct GradientTable {
    /// B-values for each volume.
    pub bvals: Bvals,
    /// Gradient directions for each volume (unit vectors for diffusion, zero for b=0).
    pub bvecs: Bvecs,
}

impl GradientTable {
    /// Create a gradient table from b-values and b-vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if the lengths don't match.
    pub fn new(bvals: Bvals, bvecs: Bvecs) -> Result<Self> {
        if bvals.len() != bvecs.len() {
            return Err(BidsError::DataFormat(format!(
                "bval count ({}) != bvec count ({})",
                bvals.len(),
                bvecs.len()
            )));
        }
        Ok(Self { bvals, bvecs })
    }

    /// Load from companion `.bval` and `.bvec` files.
    ///
    /// # Errors
    ///
    /// Returns an error if either file can't be read/parsed or they have
    /// different volume counts.
    pub fn from_files(bval_path: &Path, bvec_path: &Path) -> Result<Self> {
        let bvals = read_bvals(bval_path)?;
        let bvecs = read_bvecs(bvec_path)?;
        Self::new(bvals, bvecs)
    }

    /// Number of volumes.
    #[must_use]
    pub fn n_volumes(&self) -> usize {
        self.bvals.len()
    }

    /// Number of diffusion-weighted volumes (b > 0).
    #[must_use]
    pub fn n_diffusion_volumes(&self) -> usize {
        self.bvals.iter().filter(|&&b| b > 0.0).count()
    }

    /// Number of b=0 volumes.
    #[must_use]
    pub fn n_b0_volumes(&self) -> usize {
        self.bvals.iter().filter(|&&b| b == 0.0).count()
    }

    /// Unique b-values (sorted).
    #[must_use]
    pub fn unique_bvals(&self) -> Vec<f64> {
        let mut unique: Vec<f64> = Vec::new();
        for &b in &self.bvals {
            let rounded = (b * 10.0).round() / 10.0;
            if !unique.iter().any(|&u| (u - rounded).abs() < 1.0) {
                unique.push(rounded);
            }
        }
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique
    }

    /// Iterate over (b-value, direction) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (f64, [f64; 3])> + '_ {
        self.bvals.iter().copied().zip(self.bvecs.iter().copied())
    }

    /// Get indices of b=0 volumes.
    #[must_use]
    pub fn b0_indices(&self) -> Vec<usize> {
        self.bvals
            .iter()
            .enumerate()
            .filter(|(_, b)| **b == 0.0)
            .map(|(i, _)| i)
            .collect()
    }

    /// Check if the gradient directions are approximately unit-normalized
    /// for non-zero b-values.
    #[must_use]
    pub fn is_normalized(&self) -> bool {
        self.iter().all(|(b, [x, y, z])| {
            if b == 0.0 {
                true
            } else {
                let norm = (x * x + y * y + z * z).sqrt();
                (norm - 1.0).abs() < 0.1
            }
        })
    }
}

impl std::fmt::Display for GradientTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let unique = self.unique_bvals();
        write!(
            f,
            "GradientTable({} volumes: {} b=0, {} diffusion, b-values: {:?})",
            self.n_volumes(),
            self.n_b0_volumes(),
            self.n_diffusion_volumes(),
            unique
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_bvals() {
        let dir = std::env::temp_dir().join("bids_gradient_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.bval");
        std::fs::write(&path, "0 1000 1000 2000 0\n").unwrap();

        let bvals = read_bvals(&path).unwrap();
        assert_eq!(bvals, vec![0.0, 1000.0, 1000.0, 2000.0, 0.0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_read_bvecs() {
        let dir = std::env::temp_dir().join("bids_bvec_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.bvec");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "1.0 0.0 0.5").unwrap();
        writeln!(f, "0.0 1.0 0.5").unwrap();
        writeln!(f, "0.0 0.0 0.707").unwrap();

        let bvecs = read_bvecs(&path).unwrap();
        assert_eq!(bvecs.len(), 3);
        assert_eq!(bvecs[0], [1.0, 0.0, 0.0]);
        assert_eq!(bvecs[1], [0.0, 1.0, 0.0]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_gradient_table() {
        let bvals = vec![0.0, 1000.0, 1000.0, 2000.0, 0.0];
        let bvecs = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ];
        let table = GradientTable::new(bvals, bvecs).unwrap();

        assert_eq!(table.n_volumes(), 5);
        assert_eq!(table.n_b0_volumes(), 2);
        assert_eq!(table.n_diffusion_volumes(), 3);
        assert_eq!(table.b0_indices(), vec![0, 4]);

        let unique = table.unique_bvals();
        assert_eq!(unique.len(), 3); // 0, 1000, 2000
    }

    #[test]
    fn test_gradient_table_mismatch() {
        let bvals = vec![0.0, 1000.0];
        let bvecs = vec![[0.0, 0.0, 0.0]]; // length mismatch
        assert!(GradientTable::new(bvals, bvecs).is_err());
    }

    #[test]
    fn test_display() {
        let bvals = vec![0.0, 1000.0, 1000.0];
        let bvecs = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let table = GradientTable::new(bvals, bvecs).unwrap();
        let s = table.to_string();
        assert!(s.contains("3 volumes"));
        assert!(s.contains("1 b=0"));
    }
}
