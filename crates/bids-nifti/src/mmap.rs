//! Memory-mapped NIfTI reader for large files.
//!
//! Instead of reading the entire file into a `Vec<f64>`, this opens the file
//! via `mmap` and decodes voxels on-demand. For a 4 GB fMRI file, the RSS
//! stays at ~zero until you actually access specific volumes or voxels.
//!
//! # Example
//!
//! ```no_run
//! use bids_nifti::mmap::MmapNifti;
//!
//! let nii = MmapNifti::open("big_bold.nii").unwrap();
//! println!("Shape: {:?}, {} volumes", nii.header.shape(), nii.header.n_vols());
//!
//! // Only decodes volume 50 — no 4 GB malloc
//! let vol = nii.read_volume(50).unwrap();
//! let ts = nii.read_timeseries(32, 32, 16).unwrap();
//! ```

use crate::{NiftiHeader, NiftiError, DataType};
use std::path::Path;
use std::fs::File;
use std::io::Read;

/// A memory-mapped NIfTI file that decodes voxels on demand.
pub struct MmapNifti {
    /// Parsed header.
    pub header: NiftiHeader,
    /// Raw bytes of the file (either true mmap or read into Vec).
    #[cfg(feature = "mmap")]
    mmap: memmap2::Mmap,
    #[cfg(not(feature = "mmap"))]
    mmap: Vec<u8>,
    /// Byte offset where voxel data starts within the mmap.
    data_offset: usize,
}

impl MmapNifti {
    /// Open a NIfTI file for lazy access.
    ///
    /// For `.nii` files, reads the header then memory-maps the data region.
    /// For `.nii.gz` files, decompresses into an in-memory buffer (gzip
    /// doesn't support random access, so this is the best we can do).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NiftiError> {
        let path = path.as_ref();
        let name = path.to_string_lossy();

        if name.ends_with(".gz") {
            return Self::open_gz(path);
        }

        // Read header
        let mut file = File::open(path)?;
        let header = NiftiHeader::parse_reader(&mut file)?;
        let hdr_size = if header.version == 1 { 348usize } else { 540 };
        let data_offset = if header.vox_offset as usize > hdr_size {
            header.vox_offset as usize
        } else {
            hdr_size
        };

        // Memory-map the file (zero-copy if feature enabled)
        #[cfg(feature = "mmap")]
        let mmap = {
            let file = File::open(path)?;
            // SAFETY: we treat the mmap as read-only and the file is not modified.
            unsafe { memmap2::Mmap::map(&file)? }
        };
        #[cfg(not(feature = "mmap"))]
        let mmap = std::fs::read(path)?;

        Ok(Self { header, mmap, data_offset })
    }

    fn open_gz(path: &Path) -> Result<Self, NiftiError> {
        let file = File::open(path)?;
        let mut gz = flate2::read::GzDecoder::new(std::io::BufReader::new(file));
        let mut buf = Vec::new();
        gz.read_to_end(&mut buf)?;

        let mut cursor = std::io::Cursor::new(&buf);
        let header = NiftiHeader::parse_reader(&mut cursor)?;
        let hdr_size = if header.version == 1 { 348usize } else { 540 };
        let data_offset = if header.vox_offset as usize > hdr_size {
            header.vox_offset as usize
        } else {
            hdr_size
        };

        // For .nii.gz we must decompress to memory (can't mmap gzip)
        #[cfg(feature = "mmap")]
        {
            // Write decompressed data to a temp file, then mmap it
            let tmp = std::env::temp_dir().join(format!("bids_nifti_{:x}.tmp",
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos()));
            std::fs::write(&tmp, &buf)?;
            let file = File::open(&tmp)?;
            // SAFETY: read-only mmap, file not modified
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let _ = std::fs::remove_file(&tmp); // cleanup (mmap keeps handle)
            return Ok(Self { header, mmap, data_offset });
        }
        #[cfg(not(feature = "mmap"))]
        Ok(Self { header, mmap: buf, data_offset })
    }

    /// Read a single 3D volume from a 4D image.
    pub fn read_volume(&self, t: usize) -> Result<Vec<f64>, NiftiError> {
        let n_vols = self.header.n_vols();
        if t >= n_vols {
            return Err(NiftiError::VolumeOutOfRange { requested: t, available: n_vols });
        }
        let dt = self.header.data_type();
        let bpv = dt.bytes_per_voxel();
        let vol_size = self.header.n_voxels() / n_vols;
        let start = self.data_offset + t * vol_size * bpv;
        let end = start + vol_size * bpv;

        if end > self.mmap.len() {
            return Err(NiftiError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof, "Volume extends past end of file")));
        }

        crate::decode_raw_to_f64(&self.mmap[start..end], dt,
            self.header.scl_slope, self.header.scl_inter)
    }

    /// Read a voxel's time series across all volumes.
    pub fn read_timeseries(&self, x: usize, y: usize, z: usize) -> Result<Vec<f64>, NiftiError> {
        let (nx, ny, nz) = self.header.matrix_size();
        if x >= nx || y >= ny || z >= nz {
            return Err(NiftiError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput, "Voxel coordinates out of bounds")));
        }

        let dt = self.header.data_type();
        let bpv = dt.bytes_per_voxel();
        let vol_size = nx * ny * nz;
        let n_vols = self.header.n_vols();
        let voxel_offset = x + nx * (y + ny * z);

        let slope = if self.header.scl_slope == 0.0 { 1.0 } else { self.header.scl_slope };
        let inter = self.header.scl_inter;
        let apply_scaling = self.header.has_scaling();

        let mut ts = Vec::with_capacity(n_vols);
        for t in 0..n_vols {
            let byte_offset = self.data_offset + (t * vol_size + voxel_offset) * bpv;
            let byte_end = byte_offset + bpv;
            if byte_end > self.mmap.len() {
                return Err(NiftiError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof, "Voxel data extends past end of file")));
            }
            let raw = &self.mmap[byte_offset..byte_end];
            let mut val = decode_single_voxel(raw, dt);
            if apply_scaling { val = val * slope + inter; }
            ts.push(val);
        }
        Ok(ts)
    }

    /// Read a single voxel value.
    pub fn read_voxel(&self, idx: &[usize]) -> Result<f64, NiftiError> {
        let dt = self.header.data_type();
        let bpv = dt.bytes_per_voxel();
        let mut linear = 0usize;
        let mut stride = 1usize;
        for (i, &ix) in idx.iter().enumerate() {
            let dim = self.header.dim[i + 1] as usize;
            if ix >= dim {
                return Err(NiftiError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput, "Index out of bounds")));
            }
            linear += ix * stride;
            stride *= dim;
        }
        let byte_offset = self.data_offset + linear * bpv;
        let byte_end = byte_offset + bpv;
        if byte_end > self.mmap.len() {
            return Err(NiftiError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof, "Voxel data extends past end of file")));
        }
        let raw = &self.mmap[byte_offset..byte_end];
        let mut val = decode_single_voxel(raw, dt);
        if self.header.has_scaling() {
            let slope = if self.header.scl_slope == 0.0 { 1.0 } else { self.header.scl_slope };
            val = val * slope + self.header.scl_inter;
        }
        Ok(val)
    }
}

/// Decode a single voxel from raw bytes.
fn decode_single_voxel(raw: &[u8], dt: DataType) -> f64 {
    match dt {
        DataType::UInt8 => raw[0] as f64,
        DataType::Int8 => raw[0] as i8 as f64,
        DataType::Int16 => i16::from_le_bytes([raw[0], raw[1]]) as f64,
        DataType::UInt16 => u16::from_le_bytes([raw[0], raw[1]]) as f64,
        DataType::Int32 => i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f64,
        DataType::UInt32 => u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f64,
        DataType::Float32 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f64,
        DataType::Float64 => f64::from_le_bytes([raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]]),
        DataType::Int64 => i64::from_le_bytes([raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]]) as f64,
        DataType::UInt64 => u64::from_le_bytes([raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7]]) as f64,
        DataType::Unknown => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_lazy_volume() {
        // Create a test NIfTI file
        let dir = std::env::temp_dir().join("bids_nifti_mmap_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bold.nii");

        let nx = 8; let ny = 8; let nz = 4; let nt = 20;
        let vol_size = nx * ny * nz;

        // Build NIfTI-1 file
        let mut bytes = vec![0u8; 352]; // 348 header + 4 extension
        bytes[0..4].copy_from_slice(&348i32.to_le_bytes());
        let dims = [4i16, nx as i16, ny as i16, nz as i16, nt as i16, 1, 1, 1];
        for (i, &d) in dims.iter().enumerate() {
            let off = 40 + i * 2;
            bytes[off..off + 2].copy_from_slice(&d.to_le_bytes());
        }
        bytes[70..72].copy_from_slice(&16i16.to_le_bytes()); // FLOAT32
        bytes[72..74].copy_from_slice(&32i16.to_le_bytes());
        let pixdims = [1.0f32; 8];
        for (i, &p) in pixdims.iter().enumerate() {
            let off = 76 + i * 4;
            bytes[off..off + 4].copy_from_slice(&p.to_le_bytes());
        }
        bytes[108..112].copy_from_slice(&352.0f32.to_le_bytes()); // vox_offset
        bytes[112..116].copy_from_slice(&1.0f32.to_le_bytes()); // scl_slope
        bytes[344..348].copy_from_slice(b"n+1\0");

        // Data: each volume filled with its index
        for t in 0..nt {
            for _ in 0..vol_size {
                bytes.extend_from_slice(&(t as f32).to_le_bytes());
            }
        }
        std::fs::write(&path, &bytes).unwrap();

        // Test lazy access
        let nii = MmapNifti::open(&path).unwrap();
        assert_eq!(nii.header.n_vols(), nt);

        // Read volume 5
        let vol = nii.read_volume(5).unwrap();
        assert_eq!(vol.len(), vol_size);
        assert!((vol[0] - 5.0).abs() < 0.01);

        // Read timeseries for voxel [0,0,0]
        let ts = nii.read_timeseries(0, 0, 0).unwrap();
        assert_eq!(ts.len(), nt);
        for (t, &v) in ts.iter().enumerate() {
            assert!((v - t as f64).abs() < 0.01, "ts[{}] = {} expected {}", t, v, t);
        }

        // Out of range
        assert!(nii.read_volume(nt).is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
