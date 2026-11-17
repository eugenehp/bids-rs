//! nibabel's orientation algebra (`nibabel.orientations`): which world axis
//! and direction each voxel axis runs along (`io_orientation`), axis codes
//! such as `"RAS"` (`axcodes2ornt`, `aff2axcodes`), the flips and permutation
//! between two orientations (`ornt_transform`), the voxel affine that undoes
//! them (`inv_ornt_aff`) and the array reordering itself
//! (`apply_orientation`), with nibabel's labels `(("L", "R"), ("P", "A"),
//! ("I", "S"))`.

use crate::NiftiError;

/// A 4 x 4 voxel-to-world affine.
pub type Affine = [[f64; 4]; 4];

/// Per input axis: the world axis it runs along and its direction (+1 or -1).
pub type Ornt = [(usize, f64); 3];

const LABELS: [(char, char); 3] = [('L', 'R'), ('P', 'A'), ('I', 'S')];

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Orthogonal polar factor (scaled Newton iteration); nibabel takes it from
/// an SVD, which gives the same matrix for a full-rank input.
fn polar(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut x = a;
    for _ in 0..100 {
        let det = det3(&x);
        let inv_t: [[f64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
                let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
                (x[r0][c0] * x[r1][c1] - x[r0][c1] * x[r1][c0]) / det
            })
        });
        let z: [[f64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (x[i][j] + inv_t[i][j])));
        let dif: f64 = (0..9)
            .map(|k| (z[k / 3][k % 3] - x[k / 3][k % 3]).abs())
            .sum();
        x = z;
        if dif < 1e-15 {
            break;
        }
    }
    x
}

/// `io_orientation(affine)`.
#[must_use]
pub fn io_orientation(affine: &Affine) -> Ornt {
    let mut rs = [[0.0f64; 3]; 3];
    for c in 0..3 {
        let z = (0..3)
            .map(|r| affine[r][c] * affine[r][c])
            .sum::<f64>()
            .sqrt();
        let z = if z == 0.0 { 1.0 } else { z };
        for r in 0..3 {
            rs[r][c] = affine[r][c] / z;
        }
    }
    let mut r = polar(rs);
    let mut out = [(0usize, 1.0f64); 3];
    for (in_ax, o) in out.iter_mut().enumerate() {
        let col: [f64; 3] = std::array::from_fn(|i| r[i][in_ax]);
        let out_ax = (0..3)
            .max_by(|&a, &b| col[a].abs().total_cmp(&col[b].abs()).then(b.cmp(&a)))
            .unwrap();
        *o = (out_ax, if col[out_ax] < 0.0 { -1.0 } else { 1.0 });
        r[out_ax] = [0.0; 3];
    }
    out
}

/// `axcodes2ornt` for three codes such as `"RAS"` or `"LPS"`.
pub fn axcodes2ornt(codes: &str) -> Result<Ornt, NiftiError> {
    let chars: Vec<char> = codes.chars().collect();
    if chars.len() != 3 {
        return Err(NiftiError::InvalidOrientation(format!(
            "axis codes {codes:?}: expected three"
        )));
    }
    let mut out = [(0usize, 1.0f64); 3];
    let mut seen = [false; 3];
    for (i, ch) in chars.iter().enumerate() {
        let Some((axis, (lo, _))) = LABELS
            .iter()
            .enumerate()
            .find(|(_, (lo, hi))| lo == ch || hi == ch)
        else {
            return Err(NiftiError::InvalidOrientation(format!(
                "unknown axis code {ch:?}"
            )));
        };
        if seen[axis] {
            return Err(NiftiError::InvalidOrientation(format!(
                "axis codes {codes:?} repeat an axis"
            )));
        }
        seen[axis] = true;
        out[i] = (axis, if ch == lo { -1.0 } else { 1.0 });
    }
    Ok(out)
}

/// `aff2axcodes`: the codes an affine's axes point to.
#[must_use]
pub fn aff2axcodes(affine: &Affine) -> String {
    io_orientation(affine)
        .iter()
        .map(|&(axis, dir)| {
            if dir < 0.0 {
                LABELS[axis].0
            } else {
                LABELS[axis].1
            }
        })
        .collect()
}

/// `ornt_transform(start, end)`: the flips and permutation taking `start`
/// to `end`.
pub fn ornt_transform(start: &Ornt, end: &Ornt) -> Result<Ornt, NiftiError> {
    let mut out = [(0usize, 1.0f64); 3];
    for (end_in, &(end_out, end_flip)) in end.iter().enumerate() {
        let Some(start_in) = start
            .iter()
            .position(|&(start_out, _)| start_out == end_out)
        else {
            return Err(NiftiError::InvalidOrientation(format!(
                "unknown axis {end_out} in the target orientation"
            )));
        };
        out[start_in] = (
            end_in,
            if start[start_in].1 == end_flip {
                1.0
            } else {
                -1.0
            },
        );
    }
    Ok(out)
}

/// `inv_ornt_aff(ornt, shape)`: the voxel affine that undoes `ornt`'s flips
/// and transpose for an array of `shape` (multiply the source affine by it).
#[must_use]
pub fn inv_ornt_aff(ornt: &Ornt, shape: [usize; 3]) -> Affine {
    const EYE: Affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut undo_reorder = [[0.0f64; 4]; 4];
    for (row, &(axis, _)) in ornt.iter().enumerate() {
        undo_reorder[row] = EYE[axis];
    }
    undo_reorder[3] = EYE[3];
    let mut undo_flip = EYE;
    for (i, &(_, flip)) in ornt.iter().enumerate() {
        undo_flip[i][i] = flip;
        let center = -(shape[i] as f64 - 1.0) / 2.0;
        undo_flip[i][3] = flip * center - center;
    }
    std::array::from_fn(|r| {
        std::array::from_fn(|c| (0..4).map(|k| undo_flip[r][k] * undo_reorder[k][c]).sum())
    })
}

/// `ornt` as `(flip, axes)`: input axes to reverse, and the input axis behind
/// each output axis (`np.argsort(ornt[:, 0])`).
#[must_use]
pub fn flips_and_axes(ornt: &Ornt) -> ([bool; 3], [usize; 3]) {
    let flip = [ornt[0].1 < 0.0, ornt[1].1 < 0.0, ornt[2].1 < 0.0];
    let mut axes = [0usize; 3];
    for (in_ax, &(out_ax, _)) in ornt.iter().enumerate() {
        axes[out_ax] = in_ax;
    }
    (flip, axes)
}

/// `apply_orientation(arr, ornt)` for a C-order volume of `dims`: returns the
/// reordered values and their shape.
pub fn apply_orientation<T: Copy>(
    data: &[T],
    dims: [usize; 3],
    ornt: &Ornt,
) -> (Vec<T>, [usize; 3]) {
    let (flip, axes) = flips_and_axes(ornt);
    flip_permute(data, dims, flip, axes)
}

/// Reverse the `flip` axes, then take input axis `axes[i]` as output axis `i`.
pub fn flip_permute<T: Copy>(
    data: &[T],
    dims: [usize; 3],
    flip: [bool; 3],
    axes: [usize; 3],
) -> (Vec<T>, [usize; 3]) {
    let nd = [dims[axes[0]], dims[axes[1]], dims[axes[2]]];
    let strides = [dims[1] * dims[2], dims[2], 1];
    let mut out = Vec::with_capacity(data.len());
    for i in 0..nd[0] {
        for j in 0..nd[1] {
            for k in 0..nd[2] {
                let pos = [i, j, k];
                let mut at = 0;
                for (a, &axis) in axes.iter().enumerate() {
                    at += strides[axis]
                        * if flip[axis] {
                            dims[axis] - 1 - pos[a]
                        } else {
                            pos[a]
                        };
                }
                out.push(data[at]);
            }
        }
    }
    (out, nd)
}

/// `affine @ b` as NumPy computes a 4 x 4 product (sequential sums).
#[must_use]
pub fn matmul(a: &Affine, b: &Affine) -> Affine {
    std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let mut t = 0.0;
            for k in 0..4 {
                t += a[r][k] * b[k][c];
            }
            t
        })
    })
}

/// Reorient a C-order volume to `codes` (nibabel's `as_reoriented` with
/// `ornt_transform(io_orientation(affine), axcodes2ornt(codes))`): the
/// values, their shape and the new affine.
pub fn reorient<T: Copy>(
    data: &[T],
    dims: [usize; 3],
    affine: &Affine,
    codes: &str,
) -> Result<(Vec<T>, [usize; 3], Affine), NiftiError> {
    let t = ornt_transform(&io_orientation(affine), &axcodes2ornt(codes)?)?;
    let (values, nd) = apply_orientation(data, dims, &t);
    Ok((values, nd, matmul(affine, &inv_ornt_aff(&t, dims))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_and_round_trip() {
        let lps = [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(aff2axcodes(&lps), "LPS");
        let swapped = [
            [0.0, 0.0, 2.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_eq!(aff2axcodes(&swapped), "PSR");
        let data: Vec<u16> = (0..24).collect();
        let (ras, dims, a) = reorient(&data, [2, 3, 4], &swapped, "RAS").unwrap();
        assert_eq!((aff2axcodes(&a), dims), ("RAS".to_string(), [4, 2, 3]));
        let (back, bdims, b) = reorient(&ras, dims, &a, "PSR").unwrap();
        assert_eq!((back, bdims), (data, [2, 3, 4]));
        for r in 0..4 {
            for c in 0..4 {
                assert!((b[r][c] - swapped[r][c]).abs() < 1e-12);
            }
        }
        assert!(axcodes2ornt("RAR").is_err());
    }
}
