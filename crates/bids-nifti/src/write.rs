//! NIfTI-1 writing, and reading from bytes already in memory (a browser has
//! no paths).
//!
//! The writer is the parser's mirror: every offset below is the one in
//! `nifti1.h`, for the same reason [`NiftiHeader::parse_nifti1`] keeps its
//! buffer aligned to the file. Geometry goes in both records: the `sform` is
//! the affine as given, the `qform` is its nearest rotation plus voxel sizes
//! (`nifti_mat44_to_quatern` from `nifti1_io.c`, polar decomposition
//! included), so readers that trust only one record still place the voxels.
//!
//! [`NiftiHeader::parse_nifti1`]: crate::NiftiHeader

use std::borrow::Cow;
use std::io::{Cursor, Read, Write};
use std::path::Path;

use crate::{DataType, GeometrySource, NiftiError, NiftiHeader, NiftiImage};

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
/// Header plus the four-byte extension flag: where the voxels start.
const VOX_OFFSET: usize = 352;

impl NiftiHeader {
    /// Parse the header of a `.nii` or `.nii.gz` file held in memory
    /// (gzip is recognised by its magic bytes, not a name).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NiftiError> {
        if bytes.starts_with(&GZIP_MAGIC) {
            Self::parse_reader(&mut flate2::read::GzDecoder::new(bytes))
        } else {
            Self::parse_reader(&mut Cursor::new(bytes))
        }
    }

    /// A NIfTI-1 header for a volume of `shape` voxels (x fastest, up to
    /// seven axes) stored as `datatype`, placed in the world by `affine`.
    ///
    /// Both geometry records carry the affine, as nibabel writes them for a
    /// new image: `sform_code` 2 (aligned) and `qform_code` 0 (unknown, but
    /// filled in), and no units (`xyzt_units` 0). Use [`Self::set_affine`] to
    /// choose the codes.
    #[must_use]
    pub fn for_volume(shape: &[usize], datatype: DataType, affine: [[f64; 4]; 4]) -> Self {
        assert!(
            (1..=7).contains(&shape.len()),
            "NIfTI holds 1 to 7 axes, not {}",
            shape.len()
        );
        assert!(
            datatype != DataType::Unknown,
            "cannot write DataType::Unknown"
        );
        let mut dim = [1i64; 8];
        dim[0] = shape.len() as i64;
        for (d, &n) in dim[1..].iter_mut().zip(shape) {
            *d = n as i64;
        }
        let mut h = NiftiHeader {
            ndim: shape.len(),
            dim,
            pixdim: [1.0; 8],
            datatype: datatype as i16,
            bitpix: (datatype.bytes_per_voxel() * 8) as i16,
            version: 1,
            vox_offset: VOX_OFFSET as u64,
            // nibabel writes (1, 0) for unscaled data.
            scl_slope: 1.0,
            scl_inter: 0.0,
            ..NiftiHeader::default()
        };
        h.set_affine(affine, 2, 0);
        h
    }

    /// What nibabel does when an image is built from this header and
    /// `affine`: an affine not `allclose` (rtol 1e-5, atol 1e-8) to the
    /// header's own goes into the sform with code 2 (aligned), the qform code
    /// becomes 0 (unknown) and the voxel sizes follow the affine; a matching
    /// affine leaves the header as it is. Returns whether the header changed.
    pub fn harmonize_affine(&mut self, affine: [[f64; 4]; 4]) -> bool {
        let current = *self.affine();
        let close = (0..4).all(|r| {
            (0..4)
                .all(|c| (affine[r][c] - current[r][c]).abs() <= 1e-8 + 1e-5 * current[r][c].abs())
        });
        if close {
            return false;
        }
        self.set_affine(affine, 2, 0);
        true
    }

    /// Store `affine` in both geometry records: the `sform` exactly, the
    /// `qform` as its nearest rotation, voxel sizes (`pixdim[1..=3]`) and
    /// `qfac` (`pixdim[0]`). [`Self::affine`] becomes `affine`, taken from
    /// the record NIfTI's precedence picks for these codes.
    pub fn set_affine(&mut self, affine: [[f64; 4]; 4], sform_code: i16, qform_code: i16) {
        let (quatern, qoffset, voxel, qfac) = mat44_to_quatern(&affine);
        self.quatern = quatern;
        self.qoffset = qoffset;
        self.pixdim[0] = qfac;
        self.pixdim[1..4].copy_from_slice(&voxel);
        self.srow = [affine[0], affine[1], affine[2]];
        self.sform_code = sform_code;
        self.qform_code = qform_code;
        if sform_code > 0 {
            self.affine = affine;
            self.geometry_source = GeometrySource::Sform;
        } else {
            self.affine = crate::affine_from_qform(quatern, qoffset, &self.pixdim);
            self.geometry_source = if qform_code > 0 {
                GeometrySource::Qform
            } else {
                GeometrySource::Pixdim
            };
        }
    }

    /// The 348-byte NIfTI-1 header, little-endian, followed by the empty
    /// extension flag: everything before the voxels.
    #[must_use]
    pub fn to_nifti1_bytes(&self) -> Vec<u8> {
        let mut b = vec![0u8; VOX_OFFSET];
        let put_i16 =
            |b: &mut [u8], off: usize, v: i16| b[off..off + 2].copy_from_slice(&v.to_le_bytes());
        let put_f32 = |b: &mut [u8], off: usize, v: f64| {
            b[off..off + 4].copy_from_slice(&(v as f32).to_le_bytes())
        };
        b[0..4].copy_from_slice(&348i32.to_le_bytes());
        // `regular` stays empty, as nibabel writes a new header.
        for (i, &d) in self.dim.iter().enumerate() {
            put_i16(
                &mut b,
                40 + 2 * i,
                d.clamp(i16::MIN as i64, i16::MAX as i64) as i16,
            );
        }
        put_i16(&mut b, 70, self.datatype);
        put_i16(&mut b, 72, self.bitpix);
        for (i, &p) in self.pixdim.iter().enumerate() {
            put_f32(&mut b, 76 + 4 * i, p);
        }
        put_f32(&mut b, 108, VOX_OFFSET as f64);
        put_f32(&mut b, 112, self.scl_slope);
        put_f32(&mut b, 116, self.scl_inter);
        b[123] = self.xyzt_units;
        put_i16(&mut b, 252, self.qform_code);
        put_i16(&mut b, 254, self.sform_code);
        for i in 0..3 {
            put_f32(&mut b, 256 + 4 * i, self.quatern[i]);
            put_f32(&mut b, 268 + 4 * i, self.qoffset[i]);
        }
        for (row, srow) in self.srow.iter().enumerate() {
            for (col, &v) in srow.iter().enumerate() {
                put_f32(&mut b, 280 + (row * 4 + col) * 4, v);
            }
        }
        b[344..348].copy_from_slice(b"n+1\0");
        b
    }
}

/// The header of a `.nii` or `.nii.gz` held in memory and its voxels' bytes
/// (`n_voxels * bytes_per_voxel`, little-endian, x fastest): borrowed from
/// an uncompressed file, decompressed into one buffer from a gzipped one.
/// [`crate::decode_voxels_as`] turns them into values.
pub fn voxel_bytes(bytes: &[u8]) -> Result<(NiftiHeader, Cow<'_, [u8]>), NiftiError> {
    let gzip = bytes.starts_with(&GZIP_MAGIC);
    let mut gz = flate2::read::GzDecoder::new(bytes);
    let mut cursor = Cursor::new(bytes);
    let mut reader: &mut dyn Read = if gzip { &mut gz } else { &mut cursor };
    let header = NiftiHeader::parse_reader(&mut reader)?;
    let dt = DataType::from_code(header.datatype);
    if dt.bytes_per_voxel() == 0 {
        return Err(NiftiError::UnsupportedDataType(header.datatype));
    }
    let hdr_size = if header.version == 1 { 348u64 } else { 540 };
    let offset = header.vox_offset.max(hdr_size);
    let total = header.n_voxels() * dt.bytes_per_voxel();
    let short = || NiftiError::Io(std::io::ErrorKind::UnexpectedEof.into());
    if !gzip {
        let start = usize::try_from(offset).map_err(|_| short())?;
        let end = start.checked_add(total).ok_or_else(short)?;
        return Ok((
            header,
            Cow::Borrowed(bytes.get(start..end).ok_or_else(short)?),
        ));
    }
    std::io::copy(&mut (&mut gz).take(offset - hdr_size), &mut std::io::sink())?;
    let mut raw = vec![0u8; total];
    gz.read_exact(&mut raw)?;
    Ok((header, Cow::Owned(raw)))
}

/// A single-file NIfTI-1 (`.nii`, gzip-compressed when `gzip`) of `header`
/// and `values` (x fastest), encoded as the header's datatype: when
/// `scl_slope` is non-zero they are stored as `(value - inter) / slope`,
/// and integer types round to nearest and saturate.
pub fn nifti1_bytes(
    header: &NiftiHeader,
    values: impl ExactSizeIterator<Item = f64>,
    gzip: bool,
) -> Result<Vec<u8>, NiftiError> {
    let mut out = header.to_nifti1_bytes();
    let n = header.n_voxels();
    if values.len() != n {
        return Err(NiftiError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} voxels for a header of {n}", values.len()),
        )));
    }
    encode_voxels(&mut out, values, header)?;
    if !gzip {
        return Ok(out);
    }
    // level 1, nibabel's default for `.gz` (`Opener.default_compresslevel`)
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    gz.write_all(&out)?;
    Ok(gz.finish()?)
}

impl NiftiImage {
    /// Load a `.nii` or `.nii.gz` file held in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, NiftiError> {
        let (header, raw) = voxel_bytes(bytes)?;
        let data = crate::decode_raw_to_f64(
            &raw,
            DataType::from_code(header.datatype),
            header.scl_slope,
            header.scl_inter,
        )?;
        Ok(Self { header, data })
    }

    /// The image as a single-file NIfTI-1 (`.nii`), gzip-compressed when
    /// `gzip` (`.nii.gz`); see [`nifti1_bytes`].
    pub fn to_bytes(&self, gzip: bool) -> Result<Vec<u8>, NiftiError> {
        nifti1_bytes(&self.header, self.data.iter().copied(), gzip)
    }

    /// Write to `path`, gzip-compressed when the name ends in `.gz`.
    pub fn write(&self, path: &Path) -> Result<(), NiftiError> {
        let gzip = path.to_string_lossy().ends_with(".gz");
        std::fs::write(path, self.to_bytes(gzip)?)?;
        Ok(())
    }
}

/// Append `data` encoded as `header`'s datatype (inverse of the reader's
/// scaling).
fn encode_voxels(
    out: &mut Vec<u8>,
    data: impl ExactSizeIterator<Item = f64>,
    header: &NiftiHeader,
) -> Result<(), NiftiError> {
    let dt = DataType::from_code(header.datatype);
    let (slope, inter) = if header.scl_slope != 0.0 && header.scl_slope.is_finite() {
        (header.scl_slope, header.scl_inter)
    } else {
        (1.0, 0.0)
    };
    let raw = |v: f64| (v - inter) / slope;
    // sized once and filled in place
    let start = out.len();
    out.resize(start + data.len() * dt.bytes_per_voxel(), 0);
    let dst = &mut out[start..];
    macro_rules! put {
        ($t:ty, $f:expr) => {
            for (d, v) in dst.chunks_exact_mut(std::mem::size_of::<$t>()).zip(data) {
                let x: $t = $f(raw(v));
                d.copy_from_slice(&x.to_le_bytes());
            }
        };
    }
    match dt {
        DataType::UInt8 => put!(u8, |x: f64| round(x) as u8),
        DataType::Int8 => put!(i8, |x: f64| round(x) as i8),
        DataType::Int16 => put!(i16, |x: f64| round(x) as i16),
        DataType::UInt16 => put!(u16, |x: f64| round(x) as u16),
        DataType::Int32 => put!(i32, |x: f64| round(x) as i32),
        DataType::UInt32 => put!(u32, |x: f64| round(x) as u32),
        DataType::Int64 => put!(i64, |x: f64| round(x) as i64),
        DataType::UInt64 => put!(u64, |x: f64| round(x) as u64),
        DataType::Float32 => put!(f32, |x: f64| x as f32),
        DataType::Float64 => put!(f64, |x: f64| x),
        DataType::Unknown => return Err(NiftiError::UnsupportedDataType(header.datatype)),
    }
    Ok(())
}

/// `x.round()` (half away from zero) from `trunc`, which wasm has as an
/// instruction where `round` is a library call: the fraction `x - trunc(x)`
/// is exact, and NaN and infinities pass through as `round` passes them.
#[inline(always)]
fn round(x: f64) -> f64 {
    let t = x.trunc();
    if (x - t).abs() >= 0.5 {
        t + 1f64.copysign(x)
    } else {
        t
    }
}

/// `nifti_mat44_to_quatern`: the qform (quaternion b, c, d; offset; voxel
/// sizes; qfac) nearest to `affine`. Columns are normalised, the rotation is
/// orthogonalised by polar decomposition, and a reflection becomes
/// `qfac = -1` with the third column negated.
fn mat44_to_quatern(m: &[[f64; 4]; 4]) -> ([f64; 3], [f64; 3], [f64; 3], f64) {
    let qoffset = [m[0][3], m[1][3], m[2][3]];
    let mut r = [[0.0f64; 3]; 3];
    let mut voxel = [0.0f64; 3];
    for c in 0..3 {
        let norm = (0..3).map(|i| m[i][c] * m[i][c]).sum::<f64>().sqrt();
        for i in 0..3 {
            r[i][c] = if norm == 0.0 {
                if i == c { 1.0 } else { 0.0 }
            } else {
                m[i][c] / norm
            };
        }
        voxel[c] = if norm == 0.0 { 1.0 } else { norm };
    }
    let mut r = polar(r);
    let qfac = if det3(&r) > 0.0 { 1.0 } else { -1.0 };
    if qfac < 0.0 {
        for row in &mut r {
            row[2] = -row[2];
        }
    }
    // (q, b, c, d): q the scalar part, kept non-negative by negating the rest.
    let trace = r[0][0] + r[1][1] + r[2][2] + 1.0;
    let (q, mut b, mut c, mut d);
    if trace > 0.5 {
        q = 0.5 * trace.sqrt();
        b = 0.25 * (r[2][1] - r[1][2]) / q;
        c = 0.25 * (r[0][2] - r[2][0]) / q;
        d = 0.25 * (r[1][0] - r[0][1]) / q;
    } else {
        let xd = 1.0 + r[0][0] - (r[1][1] + r[2][2]);
        let yd = 1.0 + r[1][1] - (r[0][0] + r[2][2]);
        let zd = 1.0 + r[2][2] - (r[0][0] + r[1][1]);
        if xd > 1.0 {
            b = 0.5 * xd.sqrt();
            c = 0.25 * (r[0][1] + r[1][0]) / b;
            d = 0.25 * (r[0][2] + r[2][0]) / b;
            q = 0.25 * (r[2][1] - r[1][2]) / b;
        } else if yd > 1.0 {
            c = 0.5 * yd.sqrt();
            b = 0.25 * (r[0][1] + r[1][0]) / c;
            d = 0.25 * (r[1][2] + r[2][1]) / c;
            q = 0.25 * (r[0][2] - r[2][0]) / c;
        } else {
            d = 0.5 * zd.sqrt();
            b = 0.25 * (r[0][2] + r[2][0]) / d;
            c = 0.25 * (r[1][2] + r[2][1]) / d;
            q = 0.25 * (r[1][0] - r[0][1]) / d;
        }
    }
    if q < 0.0 {
        (b, c, d) = (-b, -c, -d);
    }
    ([b, c, d], qoffset, voxel, qfac)
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inverse3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = det3(m);
    // Adjugate over the determinant: out[i][j] is the cofactor of m[j][i].
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (a, b) = ((j + 1) % 3, (j + 2) % 3);
            let (c, d) = ((i + 1) % 3, (i + 2) % 3);
            (m[a][c] * m[b][d] - m[a][d] * m[b][c]) / det
        })
    })
}

fn row_norm(m: &[[f64; 3]; 3]) -> f64 {
    m.iter()
        .map(|r| r.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn col_norm(m: &[[f64; 3]; 3]) -> f64 {
    (0..3)
        .map(|c| (0..3).map(|r| m[r][c].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// `nifti_mat33_polar`: the orthogonal factor of `a` (Higham's scaled
/// Newton iteration).
fn polar(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut x = a;
    let mut gam = det3(&x);
    while gam == 0.0 {
        // Perturb a singular matrix so the iteration can start.
        let g = 0.00001 * (0.001 + row_norm(&x));
        for (i, row) in x.iter_mut().enumerate() {
            row[i] += g;
        }
        gam = det3(&x);
    }
    let mut dif = 1.0;
    let mut k = 0;
    loop {
        let y = inverse3(&x);
        let (gam_scale, gmi) = if dif > 0.3 {
            let alp = (row_norm(&x) * col_norm(&x)).sqrt();
            let bet = (row_norm(&y) * col_norm(&y)).sqrt();
            let g = (bet / alp).sqrt();
            (g, 1.0 / g)
        } else {
            (1.0, 1.0)
        };
        let mut z = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                z[i][j] = 0.5 * (gam_scale * x[i][j] + gmi * y[j][i]);
            }
        }
        dif = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| (z[i][j] - x[i][j]).abs())
            .sum();
        k += 1;
        if k > 100 || dif < 3.0e-6 {
            return z;
        }
        x = z;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn round_is_f64_round() {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let check = |x: f64| assert_eq!(super::round(x).to_bits(), x.round().to_bits(), "{x:e}");
        for x in [
            0.0,
            -0.0,
            0.5,
            -0.5,
            1.5,
            2.5,
            -2.5,
            0.49999999999999994,
            4503599627370495.5,
            4503599627370496.0,
            1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            5e-324,
        ] {
            check(x);
            check(-x);
        }
        for _ in 0..1_000_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let x = f64::from_bits(s);
            if !x.is_nan() {
                check(x);
            }
            check((s % 100_000) as f64 / 64.0 - 700.0);
        }
        assert!(super::round(f64::NAN).is_nan());
    }

    use super::*;

    fn close(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4], tol: f64) -> bool {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn harmonize_affine_follows_nibabel() {
        let a = [
            [2.0, 0.0, 0.0, -10.0],
            [0.0, 2.0, 0.0, 5.0],
            [0.0, 0.0, 3.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut h = NiftiHeader::for_volume(&[4, 4, 4], DataType::Float32, a);
        h.set_affine(a, 1, 1);
        // An affine within allclose leaves the codes alone.
        assert!(!h.harmonize_affine(a));
        assert_eq!((h.sform_code, h.qform_code), (1, 1));
        // Another goes to sform (aligned), qform unknown, zooms from it.
        let b = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert!(h.harmonize_affine(b));
        assert_eq!((h.sform_code, h.qform_code), (2, 0));
        assert_eq!(h.pixdim[1..4], [1.0, 1.0, 1.5]);
        assert!(close(h.affine(), &b, 1e-12));
    }

    /// A rotated, anisotropic, left-handed affine survives the round trip in
    /// both records, and every datatype round-trips its values.
    #[test]
    fn write_then_read() {
        let (s, c) = (0.3f64.sin(), 0.3f64.cos());
        let affine = [
            [-0.8 * c, 0.0, 2.5 * s, 10.0],
            [0.8 * s, 0.0, 2.5 * c, -20.5],
            [0.0, 1.2, 0.0, 30.25],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let values: Vec<f64> = (0..24).map(|v| v as f64 - 7.0).collect();
        for dt in [
            DataType::Int16,
            DataType::Int32,
            DataType::Float32,
            DataType::Float64,
            DataType::Int8,
            DataType::Int64,
        ] {
            let img = NiftiImage {
                header: NiftiHeader::for_volume(&[2, 3, 4], dt, affine),
                data: values.clone(),
            };
            for gzip in [false, true] {
                let back = NiftiImage::from_bytes(&img.to_bytes(gzip).unwrap()).unwrap();
                assert_eq!(back.header.shape(), vec![2, 3, 4]);
                assert_eq!(back.data, values, "{dt:?} gzip={gzip}");
                assert!(
                    close(back.header.affine(), &affine, 1e-5),
                    "{dt:?}: sform {:?}",
                    back.header.affine()
                );
                assert!(
                    close(&back.header.qform_affine(), &affine, 1e-5),
                    "{dt:?}: qform {:?}",
                    back.header.qform_affine()
                );
                assert_eq!(back.header.sform_code, 2);
            }
        }
        // Unsigned types hold the non-negative part.
        let pos: Vec<f64> = (0..24).map(f64::from).collect();
        for dt in [
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
        ] {
            let img = NiftiImage {
                header: NiftiHeader::for_volume(&[24], dt, affine),
                data: pos.clone(),
            };
            assert_eq!(
                NiftiImage::from_bytes(&img.to_bytes(true).unwrap())
                    .unwrap()
                    .data,
                pos,
                "{dt:?}"
            );
        }
    }

    /// Slope and intercept: stored values are the inverse map, read back as given.
    #[test]
    fn units_round_trip() {
        let mut h = NiftiHeader::for_volume(
            &[2, 2, 2],
            DataType::UInt8,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        );
        assert_eq!(h.xyzt_units, 0);
        h.xyzt_units = 3 | 16; // micron, msec
        let img = NiftiImage {
            header: h,
            data: vec![0.0; 8],
        };
        assert_eq!(
            NiftiImage::from_bytes(&img.to_bytes(true).unwrap())
                .unwrap()
                .header
                .xyzt_units,
            3 | 16
        );
    }

    #[test]
    fn scaled_ints() {
        let id = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut h = NiftiHeader::for_volume(&[3], DataType::Int16, id);
        h.scl_slope = 0.5;
        h.scl_inter = -100.0;
        let img = NiftiImage {
            header: h,
            data: vec![-100.0, -99.5, 50.0],
        };
        let bytes = img.to_bytes(false).unwrap();
        let raw: Vec<i16> = bytes[VOX_OFFSET..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(raw, vec![0, 1, 300]);
        assert_eq!(NiftiImage::from_bytes(&bytes).unwrap().data, img.data);
    }
}
