#![deny(unsafe_code)]
//! NIfTI-1 / NIfTI-2 reader for BIDS datasets.
//!
//! Reads headers to extract dimensions, voxel sizes, TR, and volume count.
//! Also loads voxel data as `f64` arrays with automatic type conversion and
//! `scl_slope`/`scl_inter` scaling.
//!
//! Handles `.nii`, `.nii.gz`, `.dtseries.nii`, and legacy `.hdr`/`.img` pairs.
//!
//! # Performance
//!
//! - Bulk reads entire data region in one I/O call
//! - Type-specific decode paths (no branch in inner loop)
//! - Pre-computed scaling: `value = raw * slope + inter`
//! - For `.nii.gz`: streams through gzip decompressor
//!
//! # Example
//!
//! ```no_run
//! use bids_nifti::{NiftiHeader, NiftiImage};
//!
//! // Header only (fast)
//! let hdr = NiftiHeader::from_file("sub-01_bold.nii.gz".as_ref()).unwrap();
//! println!("{}D, {} vols, TR={:?}s", hdr.ndim, hdr.n_vols(), hdr.tr());
//!
//! // Full image load
//! let img = NiftiImage::from_file("sub-01_bold.nii.gz".as_ref()).unwrap();
//! println!("shape: {:?}, {} voxels", img.shape(), img.data.len());
//! let val = img.get_voxel(&[32, 32, 16, 0]);
//! ```

#[allow(unsafe_code)]
pub mod mmap;
pub mod qmri;
pub mod gifti;
pub mod cifti;

pub use qmri::QmriMetadata;
pub use gifti::{GiftiImage, read_gifti_header};
pub use cifti::{CiftiHeader, BrainModel, read_cifti_header};

use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// ─── NIfTI datatype codes ──────────────────────────────────────────────────────

/// NIfTI datatype codes (from nifti1.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i16)]
pub enum DataType {
    UInt8 = 2,
    Int16 = 4,
    Int32 = 8,
    Float32 = 16,
    Float64 = 64,
    Int8 = 256,
    UInt16 = 512,
    UInt32 = 768,
    Int64 = 1024,
    UInt64 = 1280,
    Unknown = -1,
}

impl DataType {
    fn from_code(code: i16) -> Self {
        match code {
            2 => Self::UInt8,
            4 => Self::Int16,
            8 => Self::Int32,
            16 => Self::Float32,
            64 => Self::Float64,
            256 => Self::Int8,
            512 => Self::UInt16,
            768 => Self::UInt32,
            1024 => Self::Int64,
            1280 => Self::UInt64,
            _ => Self::Unknown,
        }
    }

    fn bytes_per_voxel(self) -> usize {
        match self {
            Self::UInt8 | Self::Int8 => 1,
            Self::Int16 | Self::UInt16 => 2,
            Self::Int32 | Self::UInt32 | Self::Float32 => 4,
            Self::Float64 | Self::Int64 | Self::UInt64 => 8,
            Self::Unknown => 0,
        }
    }
}

// ─── NiftiHeader ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NiftiHeader {
    /// Number of dimensions (1-7).
    pub ndim: usize,
    /// Size of each dimension: `dim[0]`=ndim, `dim[1..8]`.
    pub dim: [i64; 8],
    /// Voxel sizes / TR: `pixdim[1..3]` = voxel mm, `pixdim[4]` = TR for 4D.
    pub pixdim: [f64; 8],
    /// Data type code.
    pub datatype: i16,
    /// Bits per voxel.
    pub bitpix: i16,
    /// NIfTI version (1 or 2).
    pub version: u8,
    /// Byte offset to the start of voxel data.
    pub vox_offset: u64,
    /// Slope for intensity scaling: scaled = raw * scl_slope + scl_inter.
    /// If 0.0, no scaling is applied.
    pub scl_slope: f64,
    /// Intercept for intensity scaling.
    pub scl_inter: f64,
    /// sform affine matrix (4×4, row-major). Maps voxel (i,j,k) → (x,y,z) in mm.
    /// Set from sform if sform_code > 0, else from qform.
    pub affine: [[f64; 4]; 4],
    /// sform code (0=unknown, 1=scanner, 2=aligned, 3=Talairach, 4=MNI).
    pub sform_code: i16,
    /// qform code.
    pub qform_code: i16,
}

impl NiftiHeader {
    /// Read header from a NIfTI file (.nii, .nii.gz, .hdr).
    pub fn from_file(path: &Path) -> Result<Self, NiftiError> {
        let name = path.to_string_lossy();
        if name.ends_with(".gz") {
            let file = std::fs::File::open(path)?;
            let mut gz = flate2::read::GzDecoder::new(file);
            Self::parse_reader(&mut gz)
        } else {
            let mut file = std::fs::File::open(path)?;
            Self::parse_reader(&mut file)
        }
    }

    fn parse_reader<R: Read>(reader: &mut R) -> Result<Self, NiftiError> {
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let sizeof_hdr = i32::from_le_bytes(buf4);

        if sizeof_hdr == 348 {
            Self::parse_nifti1(reader)
        } else if sizeof_hdr == 540 {
            Self::parse_nifti2(reader)
        } else {
            let sizeof_be = i32::from_be_bytes(buf4);
            if sizeof_be == 348 || sizeof_be == 540 {
                Err(NiftiError::BigEndian)
            } else {
                Err(NiftiError::InvalidHeader(sizeof_hdr))
            }
        }
    }

    fn parse_nifti1<R: Read>(reader: &mut R) -> Result<Self, NiftiError> {
        let mut buf = [0u8; 344];
        reader.read_exact(&mut buf)?;

        let f32_at = |off: usize| -> f64 {
            f32::from_le_bytes([buf[off], buf[off+1], buf[off+2], buf[off+3]]) as f64
        };
        let i16_at = |off: usize| -> i16 {
            i16::from_le_bytes([buf[off], buf[off+1]])
        };

        let mut dim = [0i64; 8];
        for (i, d) in dim.iter_mut().enumerate() {
            *d = i16_at(36 + i * 2) as i64;
        }

        let datatype = i16_at(66);
        let bitpix = i16_at(68);

        let mut pixdim = [0.0f64; 8];
        for (i, p) in pixdim.iter_mut().enumerate() {
            *p = f32_at(72 + i * 4);
        }

        let vox_offset = f32_at(104) as u64;
        let scl_slope = f32_at(108);
        let scl_inter = f32_at(112);

        // qform_code: byte 252 → offset 248; sform_code: byte 254 → offset 250
        let qform_code = i16_at(248);
        let sform_code = i16_at(250);

        // sform affine: bytes 280-327 → offset 276-323 (3 rows × 4 cols of f32)
        let mut affine = [[0.0f64; 4]; 4];
        if sform_code > 0 {
            for (row, arow) in affine[..3].iter_mut().enumerate() {
                for (col, acol) in arow.iter_mut().enumerate() {
                    *acol = f32_at(276 + (row * 4 + col) * 4);
                }
            }
            affine[3][3] = 1.0;
        } else if qform_code > 0 {
            // Build affine from quaternion (qoffset + quatern params)
            let qb = f32_at(256);
            let qc = f32_at(260);
            let qd = f32_at(264);
            let qx = f32_at(268);
            let qy = f32_at(272);
            let qz = f32_at(276); // Note: overlaps with sform — only used when sform_code==0
            // Actually qoffset_x/y/z are at different offsets for NIfTI-1:
            // qoffset_x=268, qoffset_y=272, qoffset_z=276
            let qa = (1.0 - qb*qb - qc*qc - qd*qd).max(0.0).sqrt();
            let (i, j, k) = (pixdim[1], pixdim[2], pixdim[3]);
            let qfac = if pixdim[0] < 0.0 { -1.0 } else { 1.0 };

            affine[0][0] = (qa*qa + qb*qb - qc*qc - qd*qd) * i;
            affine[0][1] = 2.0*(qb*qc - qa*qd) * j;
            affine[0][2] = 2.0*(qb*qd + qa*qc) * k * qfac;
            affine[0][3] = qx;
            affine[1][0] = 2.0*(qb*qc + qa*qd) * i;
            affine[1][1] = (qa*qa + qc*qc - qb*qb - qd*qd) * j;
            affine[1][2] = 2.0*(qc*qd - qa*qb) * k * qfac;
            affine[1][3] = qy;
            affine[2][0] = 2.0*(qb*qd - qa*qc) * i;
            affine[2][1] = 2.0*(qc*qd + qa*qb) * j;
            affine[2][2] = (qa*qa + qd*qd - qb*qb - qc*qc) * k * qfac;
            affine[2][3] = qz;
            affine[3][3] = 1.0;
        } else {
            // Method 1: use pixdim as diagonal
            affine[0][0] = pixdim[1];
            affine[1][1] = pixdim[2];
            affine[2][2] = pixdim[3];
            affine[3][3] = 1.0;
        }

        let ndim = dim[0].clamp(0, 7) as usize;

        Ok(Self {
            ndim, dim, pixdim, datatype, bitpix, version: 1,
            vox_offset, scl_slope, scl_inter, affine, sform_code, qform_code,
        })
    }

    fn parse_nifti2<R: Read>(reader: &mut R) -> Result<Self, NiftiError> {
        let mut buf = [0u8; 536];
        reader.read_exact(&mut buf)?;

        let f64_at = |off: usize| -> f64 {
            f64::from_le_bytes(buf[off..off+8].try_into().unwrap())
        };
        let i16_at = |off: usize| -> i16 {
            i16::from_le_bytes([buf[off], buf[off+1]])
        };
        let i64_at = |off: usize| -> i64 {
            i64::from_le_bytes(buf[off..off+8].try_into().unwrap())
        };

        let datatype = i16_at(8);
        let bitpix = i16_at(10);

        let mut dim = [0i64; 8];
        for (i, d) in dim.iter_mut().enumerate() { *d = i64_at(12 + i * 8); }

        let mut pixdim = [0.0f64; 8];
        for (i, p) in pixdim.iter_mut().enumerate() { *p = f64_at(100 + i * 8); }

        let vox_offset = i64_at(164) as u64;
        let scl_slope = f64_at(172);
        let scl_inter = f64_at(180);

        // NIfTI-2: qform_code at 344, sform_code at 346
        let qform_code = i16_at(340);
        let sform_code = i16_at(342);

        // sform: 3×4 f64 at bytes 400-495 → offset 396-491
        let mut affine = [[0.0f64; 4]; 4];
        if sform_code > 0 {
            for (row, arow) in affine[..3].iter_mut().enumerate() {
                for (col, acol) in arow.iter_mut().enumerate() {
                    *acol = f64_at(396 + (row * 4 + col) * 8);
                }
            }
            affine[3][3] = 1.0;
        } else {
            affine[0][0] = pixdim[1];
            affine[1][1] = pixdim[2];
            affine[2][2] = pixdim[3];
            affine[3][3] = 1.0;
        }

        let ndim = dim[0].clamp(0, 7) as usize;

        Ok(Self {
            ndim, dim, pixdim, datatype, bitpix, version: 2,
            vox_offset, scl_slope, scl_inter, affine, sform_code, qform_code,
        })
    }

    /// Number of volumes (4th dimension). Returns 1 for 3D images.
    pub fn n_vols(&self) -> usize {
        if self.ndim >= 4 { self.dim[4].max(1) as usize } else { 1 }
    }

    /// Voxel dimensions in mm: (x, y, z).
    pub fn voxel_size(&self) -> (f64, f64, f64) {
        (self.pixdim[1], self.pixdim[2], self.pixdim[3])
    }

    /// Matrix size: (nx, ny, nz).
    pub fn matrix_size(&self) -> (usize, usize, usize) {
        (self.dim[1].max(0) as usize, self.dim[2].max(0) as usize, self.dim[3].max(0) as usize)
    }

    /// Repetition time in seconds (`pixdim[4]` for 4D time series).
    pub fn tr(&self) -> Option<f64> {
        if self.ndim >= 4 && self.pixdim[4] > 0.0 {
            Some(self.pixdim[4])
        } else {
            None
        }
    }

    /// Total number of voxels across all dimensions.
    pub fn n_voxels(&self) -> usize {
        let mut n = 1usize;
        for i in 1..=self.ndim {
            n = n.saturating_mul(self.dim[i].max(0) as usize);
        }
        n
    }

    /// Shape as a Vec (dims 1..ndim).
    pub fn shape(&self) -> Vec<usize> {
        (1..=self.ndim).map(|i| self.dim[i].max(0) as usize).collect()
    }

    /// Whether slope/intercept scaling should be applied.
    pub fn has_scaling(&self) -> bool {
        self.scl_slope != 0.0 && (self.scl_slope != 1.0 || self.scl_inter != 0.0)
    }

    /// Parsed datatype enum.
    pub fn data_type(&self) -> DataType {
        DataType::from_code(self.datatype)
    }

    /// Get the 4×4 affine matrix mapping voxel indices to mm coordinates.
    /// Equivalent to nibabel's `img.affine`.
    pub fn affine(&self) -> &[[f64; 4]; 4] {
        &self.affine
    }

    /// Transform voxel indices [i, j, k] to world coordinates [x, y, z] in mm.
    pub fn voxel_to_world(&self, ijk: [f64; 3]) -> [f64; 3] {
        let a = &self.affine;
        [
            a[0][0] * ijk[0] + a[0][1] * ijk[1] + a[0][2] * ijk[2] + a[0][3],
            a[1][0] * ijk[0] + a[1][1] * ijk[1] + a[1][2] * ijk[2] + a[1][3],
            a[2][0] * ijk[0] + a[2][1] * ijk[1] + a[2][2] * ijk[2] + a[2][3],
        ]
    }
}

// ─── NiftiImage ────────────────────────────────────────────────────────────────

/// A loaded NIfTI image with header and voxel data.
///
/// All voxel data is stored as `f64` regardless of the on-disk type. If the
/// header has non-trivial `scl_slope`/`scl_inter`, the scaling is applied
/// automatically: `value = raw * slope + inter`.
#[derive(Clone)]
pub struct NiftiImage {
    /// Parsed header.
    pub header: NiftiHeader,
    /// Voxel data as a flat f64 array in row-major order.
    /// For a 4D image (x, y, z, t), the index is: x + nx*(y + ny*(z + nz*t)).
    pub data: Vec<f64>,
}

impl std::fmt::Debug for NiftiImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NiftiImage")
            .field("shape", &self.header.shape())
            .field("datatype", &self.header.data_type())
            .field("n_voxels", &self.data.len())
            .field("tr", &self.header.tr())
            .finish()
    }
}

impl std::fmt::Display for NiftiImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.header.shape();
        let shape_str: Vec<String> = shape.iter().map(std::string::ToString::to_string).collect();
        write!(f, "NiftiImage({})", shape_str.join("×"))
    }
}

impl NiftiImage {
    /// Load a complete NIfTI image (header + data) from a file.
    ///
    /// Supports `.nii`, `.nii.gz`, and `.hdr`/`.img` pairs.
    pub fn from_file(path: &Path) -> Result<Self, NiftiError> {
        let name = path.to_string_lossy();
        if name.ends_with(".gz") {
            Self::load_gz(path)
        } else if name.ends_with(".hdr") {
            // .hdr/.img pair
            let img_path = path.with_extension("img");
            Self::load_hdr_img(path, &img_path)
        } else {
            Self::load_nii(path)
        }
    }

    /// Load a single volume from a 4D image (0-indexed).
    pub fn from_file_volume(path: &Path, volume: usize) -> Result<Self, NiftiError> {
        let name = path.to_string_lossy();
        if name.ends_with(".gz") {
            // For gzip, we have to decompress sequentially, so load all then slice
            let mut img = Self::load_gz(path)?;
            img.extract_volume_in_place(volume)?;
            Ok(img)
        } else {
            Self::load_nii_volume(path, volume)
        }
    }

    fn load_nii(path: &Path) -> Result<Self, NiftiError> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        let header = NiftiHeader::parse_reader(&mut reader)?;
        let data = read_voxel_data_seekable(&mut reader, &header)?;
        Ok(Self { header, data })
    }

    fn load_nii_volume(path: &Path, volume: usize) -> Result<Self, NiftiError> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        let header = NiftiHeader::parse_reader(&mut reader)?;
        let data = read_single_volume_seekable(&mut reader, &header, volume)?;
        let mut vol_header = header.clone();
        // Adjust header to 3D
        if vol_header.ndim >= 4 {
            vol_header.dim[4] = 1;
        }
        Ok(Self { header: vol_header, data })
    }

    fn load_gz(path: &Path) -> Result<Self, NiftiError> {
        let file = std::fs::File::open(path)?;
        let mut gz = flate2::read::GzDecoder::new(BufReader::with_capacity(256 * 1024, file));
        let header = NiftiHeader::parse_reader(&mut gz)?;
        let data = read_voxel_data_stream(&mut gz, &header)?;
        Ok(Self { header, data })
    }

    fn load_hdr_img(hdr_path: &Path, img_path: &Path) -> Result<Self, NiftiError> {
        // Parse header from .hdr
        let mut hdr_file = std::fs::File::open(hdr_path)?;
        let header = NiftiHeader::parse_reader(&mut hdr_file)?;
        // Read data from .img
        let file = std::fs::File::open(img_path)?;
        let mut reader = BufReader::with_capacity(256 * 1024, file);
        let data = read_voxel_data_stream(&mut reader, &header)?;
        Ok(Self { header, data })
    }

    fn extract_volume_in_place(&mut self, volume: usize) -> Result<(), NiftiError> {
        if self.header.ndim < 4 {
            return if volume == 0 { Ok(()) } else {
                Err(NiftiError::VolumeOutOfRange { requested: volume, available: 1 })
            };
        }
        let n_vols = self.header.n_vols();
        if volume >= n_vols {
            return Err(NiftiError::VolumeOutOfRange { requested: volume, available: n_vols });
        }
        let vol_size = self.header.n_voxels() / n_vols;
        let start = volume * vol_size;
        let end = start + vol_size;
        self.data = self.data[start..end].to_vec();
        self.header.dim[4] = 1;
        Ok(())
    }

    /// Save image data as a safetensors file.
    ///
    /// Tensor name "data" with shape matching the NIfTI dimensions.
    /// Metadata includes affine, voxel size, and TR.
    #[cfg(feature = "safetensors")]
    pub fn save_safetensors(&self, path: &std::path::Path) -> std::io::Result<()> {
        use safetensors::tensor::{Dtype, TensorView};
        use std::collections::HashMap;

        let shape = self.header.shape();
        let flat_bytes: &[u8] = bytemuck::cast_slice(&self.data);

        let tensor = TensorView::new(Dtype::F64, shape.clone(), flat_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        let mut tensors = HashMap::new();
        tensors.insert("data".to_string(), tensor);

        let mut metadata = HashMap::new();
        metadata.insert("shape".to_string(), format!("{:?}", shape));
        metadata.insert("voxel_size".to_string(), format!("{:?}", self.header.voxel_size()));
        if let Some(tr) = self.header.tr() {
            metadata.insert("tr".to_string(), tr.to_string());
        }
        // Flatten affine to string
        let aff: Vec<f64> = self.header.affine.iter().flat_map(|r| r.iter().copied()).collect();
        metadata.insert("affine".to_string(), format!("{:?}", aff));

        let bytes = safetensors::tensor::serialize(&tensors, &Some(metadata))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        std::fs::write(path, bytes)
    }

    /// Shape of the image as a Vec.
    pub fn shape(&self) -> Vec<usize> { self.header.shape() }

    /// Convert to an ndarray ArrayD<f64> with the image's native shape.
    #[cfg(feature = "ndarray")]
    pub fn to_ndarray(&self) -> ndarray::ArrayD<f64> {
        let shape: Vec<usize> = self.header.shape();
        ndarray::ArrayD::from_shape_vec(
            ndarray::IxDyn(&shape), self.data.clone()
        ).unwrap_or_else(|_| ndarray::ArrayD::zeros(ndarray::IxDyn(&shape)))
    }

    /// Get a voxel value by multi-dimensional index [x, y, z] or [x, y, z, t].
    pub fn get_voxel(&self, idx: &[usize]) -> Option<f64> {
        let linear = self.linear_index(idx)?;
        self.data.get(linear).copied()
    }

    /// Get a 3D slice at time point t from a 4D image. Returns None for 3D images
    /// if t > 0.
    pub fn get_volume(&self, t: usize) -> Option<Vec<f64>> {
        let (nx, ny, nz) = self.header.matrix_size();
        let vol_size = nx * ny * nz;
        let n_vols = self.header.n_vols();
        if t >= n_vols { return None; }
        let start = t * vol_size;
        let end = start + vol_size;
        if end <= self.data.len() {
            Some(self.data[start..end].to_vec())
        } else {
            None
        }
    }

    /// Extract a time series for a single voxel [x, y, z] across all volumes.
    pub fn get_timeseries(&self, x: usize, y: usize, z: usize) -> Option<Vec<f64>> {
        let (nx, ny, nz) = self.header.matrix_size();
        if x >= nx || y >= ny || z >= nz { return None; }
        let vol_size = nx * ny * nz;
        let n_vols = self.header.n_vols();
        let voxel_offset = x + nx * (y + ny * z);
        let mut ts = Vec::with_capacity(n_vols);
        for t in 0..n_vols {
            ts.push(self.data[t * vol_size + voxel_offset]);
        }
        Some(ts)
    }

    /// Compute the mean across all voxels (useful for QC).
    pub fn mean(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// Compute the mean image across time (for 4D data, returns 3D mean).
    pub fn temporal_mean(&self) -> Vec<f64> {
        let (nx, ny, nz) = self.header.matrix_size();
        let vol_size = nx * ny * nz;
        let n_vols = self.header.n_vols();
        if n_vols <= 1 { return self.data.clone(); }
        let mut mean = vec![0.0f64; vol_size];
        for t in 0..n_vols {
            let start = t * vol_size;
            for (m, &d) in mean.iter_mut().zip(self.data[start..start + vol_size].iter()) {
                *m += d;
            }
        }
        let scale = 1.0 / n_vols as f64;
        for v in &mut mean { *v *= scale; }
        mean
    }

    fn linear_index(&self, idx: &[usize]) -> Option<usize> {
        if idx.len() > self.header.ndim { return None; }
        let mut linear = 0;
        let mut stride = 1;
        for (i, &ix) in idx.iter().enumerate() {
            let dim_size = self.header.dim[i + 1] as usize;
            if ix >= dim_size { return None; }
            linear += ix * stride;
            stride *= dim_size;
        }
        if linear < self.data.len() { Some(linear) } else { None }
    }
}

// ─── Data reading ──────────────────────────────────────────────────────────────

/// Read voxel data from a seekable reader (uncompressed .nii).
fn read_voxel_data_seekable<R: Read + Seek>(
    reader: &mut R, header: &NiftiHeader,
) -> Result<Vec<f64>, NiftiError> {
    let hdr_size = if header.version == 1 { 348u64 } else { 540 };
    let offset = if header.vox_offset > hdr_size { header.vox_offset } else { hdr_size };
    reader.seek(SeekFrom::Start(offset))?;
    decode_voxels(reader, header)
}

/// Read a single volume from a seekable reader.
fn read_single_volume_seekable<R: Read + Seek>(
    reader: &mut R, header: &NiftiHeader, volume: usize,
) -> Result<Vec<f64>, NiftiError> {
    let n_vols = header.n_vols();
    if volume >= n_vols {
        return Err(NiftiError::VolumeOutOfRange { requested: volume, available: n_vols });
    }
    let dt = DataType::from_code(header.datatype);
    let bpv = dt.bytes_per_voxel();
    if bpv == 0 { return Err(NiftiError::UnsupportedDataType(header.datatype)); }

    let vol_voxels = header.n_voxels() / n_vols;
    let vol_bytes = vol_voxels * bpv;

    let hdr_size = if header.version == 1 { 348u64 } else { 540 };
    let offset = if header.vox_offset > hdr_size { header.vox_offset } else { hdr_size };
    reader.seek(SeekFrom::Start(offset + volume as u64 * vol_bytes as u64))?;

    let mut raw = vec![0u8; vol_bytes];
    reader.read_exact(&mut raw)?;
    decode_raw_to_f64(&raw, dt, header.scl_slope, header.scl_inter)
}

/// Read voxel data from a streaming reader (gzip, .img pair).
fn read_voxel_data_stream<R: Read>(
    reader: &mut R, header: &NiftiHeader,
) -> Result<Vec<f64>, NiftiError> {
    // Skip to vox_offset (for .nii.gz the header was already consumed by parse_reader,
    // so we need to skip: vox_offset - header_size bytes)
    let hdr_size = if header.version == 1 { 348u64 } else { 540 };
    let offset = if header.vox_offset > hdr_size { header.vox_offset } else { hdr_size };
    let skip = offset - hdr_size;
    if skip > 0 {
        let mut discard = vec![0u8; skip as usize];
        reader.read_exact(&mut discard)?;
    }
    decode_voxels(reader, header)
}

/// Decode all voxels from current reader position.
fn decode_voxels<R: Read>(reader: &mut R, header: &NiftiHeader) -> Result<Vec<f64>, NiftiError> {
    let dt = DataType::from_code(header.datatype);
    let bpv = dt.bytes_per_voxel();
    if bpv == 0 { return Err(NiftiError::UnsupportedDataType(header.datatype)); }

    let n_voxels = header.n_voxels();
    let total_bytes = n_voxels * bpv;
    let mut raw = vec![0u8; total_bytes];
    reader.read_exact(&mut raw)?;

    decode_raw_to_f64(&raw, dt, header.scl_slope, header.scl_inter)
}

/// Convert raw bytes to f64 with optional scaling. Type-specific decode paths
/// with no branch in the inner loop.
#[inline(never)]
pub(crate) fn decode_raw_to_f64(
    raw: &[u8], dt: DataType, scl_slope: f64, scl_inter: f64,
) -> Result<Vec<f64>, NiftiError> {
    let bpv = dt.bytes_per_voxel();
    let n = raw.len() / bpv;

    let mut data = vec![0.0f64; n];

    let apply_scaling = scl_slope != 0.0 && (scl_slope != 1.0 || scl_inter != 0.0);
    let slope = if scl_slope == 0.0 { 1.0 } else { scl_slope };
    let inter = scl_inter;

    // Helper: read a LE value from a byte slice at a known-good offset.
    // Using explicit byte indexing avoids try_into().unwrap().
    macro_rules! le_read {
        (i16, $src:expr, $off:expr) => {
            i16::from_le_bytes([$src[$off], $src[$off + 1]])
        };
        (u16, $src:expr, $off:expr) => {
            u16::from_le_bytes([$src[$off], $src[$off + 1]])
        };
        (i32, $src:expr, $off:expr) => {
            i32::from_le_bytes([$src[$off], $src[$off+1], $src[$off+2], $src[$off+3]])
        };
        (u32, $src:expr, $off:expr) => {
            u32::from_le_bytes([$src[$off], $src[$off+1], $src[$off+2], $src[$off+3]])
        };
        (f32, $src:expr, $off:expr) => {
            f32::from_le_bytes([$src[$off], $src[$off+1], $src[$off+2], $src[$off+3]])
        };
        (f64, $src:expr, $off:expr) => {
            f64::from_le_bytes([$src[$off], $src[$off+1], $src[$off+2], $src[$off+3],
                                $src[$off+4], $src[$off+5], $src[$off+6], $src[$off+7]])
        };
        (i64, $src:expr, $off:expr) => {
            i64::from_le_bytes([$src[$off], $src[$off+1], $src[$off+2], $src[$off+3],
                                $src[$off+4], $src[$off+5], $src[$off+6], $src[$off+7]])
        };
        (u64, $src:expr, $off:expr) => {
            u64::from_le_bytes([$src[$off], $src[$off+1], $src[$off+2], $src[$off+3],
                                $src[$off+4], $src[$off+5], $src[$off+6], $src[$off+7]])
        };
    }

    match dt {
        DataType::UInt8 => {
            for (dst, &src) in data.iter_mut().zip(raw.iter()) { *dst = src as f64; }
        }
        DataType::Int8 => {
            for (dst, &src) in data.iter_mut().zip(raw.iter()) { *dst = (src as i8) as f64; }
        }
        DataType::Int16 => {
            for (i, dst) in data.iter_mut().enumerate() { *dst = le_read!(i16, raw, i * 2) as f64; }
        }
        DataType::UInt16 => {
            for (i, dst) in data.iter_mut().enumerate() { *dst = le_read!(u16, raw, i * 2) as f64; }
        }
        DataType::Int32 => {
            for (i, dst) in data.iter_mut().enumerate() { *dst = le_read!(i32, raw, i * 4) as f64; }
        }
        DataType::UInt32 => {
            for (i, dst) in data.iter_mut().enumerate() { *dst = le_read!(u32, raw, i * 4) as f64; }
        }
        DataType::Float32 => {
            for (i, dst) in data.iter_mut().enumerate() { *dst = le_read!(f32, raw, i * 4) as f64; }
        }
        DataType::Float64 => {
            for (i, dst) in data.iter_mut().enumerate() { *dst = le_read!(f64, raw, i * 8); }
        }
        DataType::Int64 => {
            for (i, dst) in data.iter_mut().enumerate() { *dst = le_read!(i64, raw, i * 8) as f64; }
        }
        DataType::UInt64 => {
            for (i, dst) in data.iter_mut().enumerate() { *dst = le_read!(u64, raw, i * 8) as f64; }
        }
        DataType::Unknown => {
            return Err(NiftiError::UnsupportedDataType(dt as i16));
        }
    }

    if apply_scaling {
        for v in &mut data { *v = *v * slope + inter; }
    }

    Ok(data)
}

// ─── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum NiftiError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid NIfTI header size: {0}")]
    InvalidHeader(i32),
    #[error("Big-endian NIfTI not supported (swap bytes externally)")]
    BigEndian,
    #[error("Unsupported NIfTI datatype code: {0}")]
    UnsupportedDataType(i16),
    #[error("Volume {requested} out of range (available: {available})")]
    VolumeOutOfRange { requested: usize, available: usize },
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a NIfTI-1 .nii file in memory with given dimensions and float32 data.
    fn build_nifti1(dims: &[i16; 8], pixdims: &[f32; 8], data: &[f32]) -> Vec<u8> {
        let mut header = vec![0u8; 348];
        header[0..4].copy_from_slice(&348i32.to_le_bytes());
        for (i, &d) in dims.iter().enumerate() {
            let off = 40 + i * 2;
            header[off..off + 2].copy_from_slice(&d.to_le_bytes());
        }
        // datatype = FLOAT32 (16)
        header[70..72].copy_from_slice(&16i16.to_le_bytes());
        // bitpix = 32
        header[72..74].copy_from_slice(&32i16.to_le_bytes());
        for (i, &p) in pixdims.iter().enumerate() {
            let off = 76 + i * 4;
            header[off..off + 4].copy_from_slice(&p.to_le_bytes());
        }
        // vox_offset = 352 (348 header + 4 extension bytes)
        header[108..112].copy_from_slice(&352.0f32.to_le_bytes());
        // scl_slope = 1.0
        header[112..116].copy_from_slice(&1.0f32.to_le_bytes());
        // scl_inter = 0.0
        header[116..120].copy_from_slice(&0.0f32.to_le_bytes());
        // magic
        header[344..348].copy_from_slice(b"n+1\0");
        // 4 extension bytes (required by NIfTI-1 .nii)
        header.extend_from_slice(&[0u8; 4]);
        // data
        for &v in data {
            header.extend_from_slice(&v.to_le_bytes());
        }
        header
    }

    fn write_nifti1_file(path: &Path, dims: &[i16; 8], pixdims: &[f32; 8], data: &[f32]) {
        let bytes = build_nifti1(dims, pixdims, data);
        std::fs::write(path, &bytes).unwrap();
    }

    fn write_nifti1_gz(path: &Path, dims: &[i16; 8], pixdims: &[f32; 8], data: &[f32]) {
        let bytes = build_nifti1(dims, pixdims, data);
        let file = std::fs::File::create(path).unwrap();
        let mut gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        gz.write_all(&bytes).unwrap();
        gz.finish().unwrap();
    }

    #[test]
    fn test_header_nifti1() {
        let dir = std::env::temp_dir().join("bids_nifti_hdr1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.nii");
        let dims: [i16; 8] = [4, 64, 64, 32, 100, 1, 1, 1];
        let pixdims: [f32; 8] = [1.0, 3.0, 3.0, 3.5, 2.0, 0.0, 0.0, 0.0];
        let data: Vec<f32> = vec![0.0; 64 * 64 * 32 * 100];
        write_nifti1_file(&path, &dims, &pixdims, &data);

        let hdr = NiftiHeader::from_file(&path).unwrap();
        assert_eq!(hdr.version, 1);
        assert_eq!(hdr.ndim, 4);
        assert_eq!(hdr.n_vols(), 100);
        assert_eq!(hdr.matrix_size(), (64, 64, 32));
        assert!((hdr.tr().unwrap() - 2.0).abs() < 0.01);
        assert_eq!(hdr.n_voxels(), 64 * 64 * 32 * 100);
        assert_eq!(hdr.shape(), vec![64, 64, 32, 100]);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_3d_image() {
        let dir = std::env::temp_dir().join("bids_nifti_3d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("anat.nii");
        let dims: [i16; 8] = [3, 4, 4, 3, 1, 1, 1, 1];
        let pixdims: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        // 4×4×3 = 48 voxels, fill with index values
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        write_nifti1_file(&path, &dims, &pixdims, &data);

        let img = NiftiImage::from_file(&path).unwrap();
        assert_eq!(img.shape(), vec![4, 4, 3]);
        assert_eq!(img.data.len(), 48);
        // Voxel at [0,0,0] = 0.0
        assert_eq!(img.get_voxel(&[0, 0, 0]), Some(0.0));
        // Voxel at [1,0,0] = 1.0 (x varies fastest)
        assert_eq!(img.get_voxel(&[1, 0, 0]), Some(1.0));
        // Voxel at [0,1,0] = 4.0 (stride = nx = 4)
        assert_eq!(img.get_voxel(&[0, 1, 0]), Some(4.0));
        // Voxel at [0,0,1] = 16.0 (stride = nx*ny = 16)
        assert_eq!(img.get_voxel(&[0, 0, 1]), Some(16.0));
        // Out of bounds
        assert_eq!(img.get_voxel(&[4, 0, 0]), None);

        assert!((img.mean() - 23.5).abs() < 0.01);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_4d_image() {
        let dir = std::env::temp_dir().join("bids_nifti_4d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bold.nii");
        let nx = 4; let ny = 4; let nz = 3; let nt = 10;
        let dims: [i16; 8] = [4, nx as i16, ny as i16, nz as i16, nt as i16, 1, 1, 1];
        let pixdims: [f32; 8] = [1.0, 2.0, 2.0, 2.0, 1.5, 0.0, 0.0, 0.0];
        let vol_size = nx * ny * nz;
        let data: Vec<f32> = (0..(vol_size * nt)).map(|i| i as f32).collect();
        write_nifti1_file(&path, &dims, &pixdims, &data);

        let img = NiftiImage::from_file(&path).unwrap();
        assert_eq!(img.shape(), vec![4, 4, 3, 10]);
        assert_eq!(img.data.len(), vol_size * nt);
        assert_eq!(img.header.n_vols(), 10);

        // Get volume 0
        let vol0 = img.get_volume(0).unwrap();
        assert_eq!(vol0.len(), vol_size);
        assert_eq!(vol0[0], 0.0);
        assert_eq!(vol0[vol_size - 1], (vol_size - 1) as f64);

        // Get volume 1
        let vol1 = img.get_volume(1).unwrap();
        assert_eq!(vol1[0], vol_size as f64);

        // Timeseries for voxel [0,0,0]
        let ts = img.get_timeseries(0, 0, 0).unwrap();
        assert_eq!(ts.len(), nt);
        for t in 0..nt {
            assert_eq!(ts[t], (t * vol_size) as f64);
        }

        // Temporal mean
        let tmean = img.temporal_mean();
        assert_eq!(tmean.len(), vol_size);
        // Mean of voxel [0,0,0] across time: mean of 0, 48, 96, ..., 432
        let expected_mean = (0..nt).map(|t| (t * vol_size) as f64).sum::<f64>() / nt as f64;
        assert!((tmean[0] - expected_mean).abs() < 0.01);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_single_volume() {
        let dir = std::env::temp_dir().join("bids_nifti_vol");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bold.nii");
        let nx = 4; let ny = 4; let nz = 3; let nt = 5;
        let dims: [i16; 8] = [4, nx as i16, ny as i16, nz as i16, nt as i16, 1, 1, 1];
        let pixdims: [f32; 8] = [1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0];
        let vol_size = nx * ny * nz;
        let data: Vec<f32> = (0..(vol_size * nt)).map(|i| i as f32).collect();
        write_nifti1_file(&path, &dims, &pixdims, &data);

        let img = NiftiImage::from_file_volume(&path, 2).unwrap();
        assert_eq!(img.data.len(), vol_size);
        assert_eq!(img.data[0], (2 * vol_size) as f64);

        // Out of range
        assert!(NiftiImage::from_file_volume(&path, 5).is_err());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_gzip() {
        let dir = std::env::temp_dir().join("bids_nifti_gz");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.nii.gz");
        let dims: [i16; 8] = [3, 8, 8, 4, 1, 1, 1, 1];
        let pixdims: [f32; 8] = [1.0, 1.5, 1.5, 1.5, 0.0, 0.0, 0.0, 0.0];
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
        write_nifti1_gz(&path, &dims, &pixdims, &data);

        let img = NiftiImage::from_file(&path).unwrap();
        assert_eq!(img.shape(), vec![8, 8, 4]);
        assert_eq!(img.data.len(), 256);
        assert!((img.data[0] - 0.0).abs() < 0.01);
        assert!((img.data[1] - 0.5).abs() < 0.01);
        assert!((img.data[255] - 127.5).abs() < 0.01);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_scaling() {
        let dir = std::env::temp_dir().join("bids_nifti_scl");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scaled.nii");
        let dims: [i16; 8] = [3, 2, 2, 2, 1, 1, 1, 1];
        let pixdims: [f32; 8] = [1.0; 8];
        let raw_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Build with custom slope/inter
        let mut bytes = build_nifti1(&dims, &pixdims, &raw_data);
        // scl_slope at byte 112, scl_inter at 116
        bytes[112..116].copy_from_slice(&2.0f32.to_le_bytes()); // slope = 2
        bytes[116..120].copy_from_slice(&10.0f32.to_le_bytes()); // inter = 10
        std::fs::write(&path, &bytes).unwrap();

        let img = NiftiImage::from_file(&path).unwrap();
        assert!(img.header.has_scaling());
        // value = raw * 2 + 10
        assert!((img.data[0] - 12.0).abs() < 0.01); // 1*2+10
        assert!((img.data[7] - 26.0).abs() < 0.01); // 8*2+10

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_int16_datatype() {
        let dir = std::env::temp_dir().join("bids_nifti_i16");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("int16.nii");

        let dims: [i16; 8] = [3, 4, 4, 2, 1, 1, 1, 1];
        let n_voxels = 32usize;

        let mut header = vec![0u8; 348];
        header[0..4].copy_from_slice(&348i32.to_le_bytes());
        for (i, &d) in dims.iter().enumerate() {
            let off = 40 + i * 2;
            header[off..off + 2].copy_from_slice(&d.to_le_bytes());
        }
        // datatype = INT16 (4)
        header[70..72].copy_from_slice(&4i16.to_le_bytes());
        header[72..74].copy_from_slice(&16i16.to_le_bytes());
        let pixdims = [1.0f32; 8];
        for (i, &p) in pixdims.iter().enumerate() {
            let off = 76 + i * 4;
            header[off..off + 4].copy_from_slice(&p.to_le_bytes());
        }
        header[108..112].copy_from_slice(&352.0f32.to_le_bytes());
        header[112..116].copy_from_slice(&0.0f32.to_le_bytes()); // slope=0 → no scaling
        header[344..348].copy_from_slice(b"n+1\0");
        header.extend_from_slice(&[0u8; 4]);

        // Write int16 data
        for i in 0..n_voxels as i16 {
            header.extend_from_slice(&(i * 100).to_le_bytes());
        }
        std::fs::write(&path, &header).unwrap();

        let img = NiftiImage::from_file(&path).unwrap();
        assert_eq!(img.data.len(), n_voxels);
        assert_eq!(img.data[0], 0.0);
        assert_eq!(img.data[1], 100.0);
        assert_eq!(img.data[31], 3100.0);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_large_4d_performance() {
        let dir = std::env::temp_dir().join("bids_nifti_perf");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bold.nii");

        // 64×64×32×100 float32 = ~50 MB
        let nx = 64; let ny = 64; let nz = 32; let nt = 100;
        let dims: [i16; 8] = [4, nx, ny, nz, nt, 1, 1, 1];
        let pixdims: [f32; 8] = [1.0, 3.0, 3.0, 3.5, 2.0, 0.0, 0.0, 0.0];
        let n = (nx as usize) * (ny as usize) * (nz as usize) * (nt as usize);
        // Generate data in bulk
        let data: Vec<f32> = (0..n).map(|i| (i % 1000) as f32 * 0.1).collect();
        write_nifti1_file(&path, &dims, &pixdims, &data);

        let file_mb = std::fs::metadata(&path).unwrap().len() as f64 / 1e6;

        // Warm up
        let _ = NiftiImage::from_file(&path).unwrap();

        let t = std::time::Instant::now();
        let img = NiftiImage::from_file(&path).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;

        assert_eq!(img.data.len(), n);
        assert_eq!(img.header.n_vols(), nt as usize);

        eprintln!(
            "  NIfTI 64×64×32×100 float32 ({:.0}MB): {:.1}ms ({:.0} MB/s)",
            file_mb, ms, file_mb / (ms / 1000.0)
        );

        // Should be well under 500ms for 50MB uncompressed
        assert!(ms < 500.0, "NIfTI read took {:.0}ms, expected <500ms", ms);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_header_nifti2_synthetic() {
        let mut header = vec![0u8; 540];
        header[0..4].copy_from_slice(&540i32.to_le_bytes());
        let dims: [i64; 8] = [4, 96, 96, 48, 200, 1, 1, 1];
        for (i, &d) in dims.iter().enumerate() {
            let off = 16 + i * 8;
            header[off..off + 8].copy_from_slice(&d.to_le_bytes());
        }
        let pixdims: [f64; 8] = [1.0, 2.0, 2.0, 2.0, 1.5, 0.0, 0.0, 0.0];
        for (i, &p) in pixdims.iter().enumerate() {
            let off = 104 + i * 8;
            header[off..off + 8].copy_from_slice(&p.to_le_bytes());
        }

        let mut cursor = std::io::Cursor::new(header);
        let hdr = NiftiHeader::parse_reader(&mut cursor).unwrap();

        assert_eq!(hdr.version, 2);
        assert_eq!(hdr.n_vols(), 200);
        assert_eq!(hdr.matrix_size(), (96, 96, 48));
        assert!((hdr.tr().unwrap() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_sform_affine() {
        let dir = std::env::temp_dir().join("bids_nifti_affine");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("affine.nii");

        let dims: [i16; 8] = [3, 4, 4, 3, 1, 1, 1, 1];
        let pixdims: [f32; 8] = [1.0, 2.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        let data: Vec<f32> = vec![0.0; 48];
        let mut bytes = build_nifti1(&dims, &pixdims, &data);

        // Set sform_code = 1 (scanner) at byte 254 (offset 254 in full header)
        bytes[254..256].copy_from_slice(&1i16.to_le_bytes());
        // sform affine at bytes 280-327: set diagonal = [2, 2, 3] + offset = [10, 20, 30]
        // Row 0: [2, 0, 0, 10]
        let sform_vals: [f32; 12] = [2.0, 0.0, 0.0, 10.0,
                                      0.0, 2.0, 0.0, 20.0,
                                      0.0, 0.0, 3.0, 30.0];
        for (i, &v) in sform_vals.iter().enumerate() {
            let off = 280 + i * 4;
            bytes[off..off+4].copy_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&path, &bytes).unwrap();

        let hdr = NiftiHeader::from_file(&path).unwrap();
        assert_eq!(hdr.sform_code, 1);
        assert!((hdr.affine[0][0] - 2.0).abs() < 1e-6);
        assert!((hdr.affine[1][1] - 2.0).abs() < 1e-6);
        assert!((hdr.affine[2][2] - 3.0).abs() < 1e-6);
        assert!((hdr.affine[0][3] - 10.0).abs() < 1e-6);
        assert!((hdr.affine[1][3] - 20.0).abs() < 1e-6);
        assert!((hdr.affine[2][3] - 30.0).abs() < 1e-6);
        assert!((hdr.affine[3][3] - 1.0).abs() < 1e-6);

        // voxel_to_world: voxel [1, 2, 3] → [2*1+10, 2*2+20, 3*3+30] = [12, 24, 39]
        let world = hdr.voxel_to_world([1.0, 2.0, 3.0]);
        assert!((world[0] - 12.0).abs() < 1e-6);
        assert!((world[1] - 24.0).abs() < 1e-6);
        assert!((world[2] - 39.0).abs() < 1e-6);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
