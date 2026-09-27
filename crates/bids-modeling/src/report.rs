//! Model reports: contrast matrices, VIF diagnostics, and HTML output.
//!
//! Corresponds to PyBIDS' `bids.modeling.report` package. Where PyBIDS renders
//! plots with Altair, this module emits self-contained inline SVG, so a report
//! is a single HTML file with no runtime dependencies.
//!
//! # Entry points
//!
//! - [`node_report()`] — diagnostics for one node output.
//! - [`generate_report()`] — a full HTML document for a set of node outputs.
//! - [`est_vif()`], [`get_eff_reg_vif()`], [`get_all_contrast_vif()`] —
//!   variance inflation factors for regressors and for contrasts.
//! - [`generate_contrast_matrix()`] — contrasts as a weight matrix over the
//!   design matrix columns.
//! - [`plot_design_matrix()`], [`plot_corr_matrix()`], [`plot_vif()`] — inline
//!   SVG figures, usable on their own.

use crate::node::{ContrastInfo, StatsModelsNodeOutput};
use crate::spec::{compute_vif, pearson_r};
use std::collections::HashMap;

// ───────────────────────── Small dense linear algebra ─────────────────────────

/// Eigendecomposition of a real symmetric matrix by the cyclic Jacobi method.
///
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors held column-major:
/// `vectors[j][i]` is row `i` of eigenvector `j`.
// The rotation loops read and write two columns (or rows) at the same index,
// so an iterator over one of them cannot express the update.
#[allow(clippy::needless_range_loop)]
fn jacobi_eigen(input: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = input.len();
    let mut a = input.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..n).map(|i| f64::from(u8::from(i == j))).collect())
        .collect();

    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-18 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[p][k];
                    let vkq = v[q][k];
                    v[p][k] = c * vkp - s * vkq;
                    v[q][k] = s * vkp + c * vkq;
                }
            }
        }
    }
    let values: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    (values, v)
}

/// Moore-Penrose pseudo-inverse of a real symmetric matrix.
fn pinv_sym(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    if n == 0 {
        return Vec::new();
    }
    let (vals, vecs) = jacobi_eigen(m);
    let max = vals.iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let tol = max * n as f64 * f64::EPSILON;

    let mut out = vec![vec![0.0; n]; n];
    for (k, &lambda) in vals.iter().enumerate() {
        if lambda.abs() <= tol {
            continue;
        }
        let inv = 1.0 / lambda;
        for i in 0..n {
            for j in 0..n {
                out[i][j] += inv * vecs[k][i] * vecs[k][j];
            }
        }
    }
    out
}

/// Orthonormal basis for the null space of a single row vector.
///
/// Returns the basis column-major; the result has `len(row) - 1` columns when
/// `row` is non-zero. This is the `scipy.linalg.null_space` call PyBIDS makes
/// on a contrast vector, specialised to the 1 x m case it always uses.
fn null_space_of_row(row: &[f64]) -> Vec<Vec<f64>> {
    let m = row.len();
    let norm_sq: f64 = row.iter().map(|v| v * v).sum();
    if norm_sq < 1e-24 {
        // Every direction is in the null space.
        return (0..m)
            .map(|j| (0..m).map(|i| f64::from(u8::from(i == j))).collect())
            .collect();
    }

    // Project each standard basis vector off `row`, then Gram-Schmidt.
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for j in 0..m {
        let mut candidate: Vec<f64> = (0..m)
            .map(|i| f64::from(u8::from(i == j)) - row[j] * row[i] / norm_sq)
            .collect();
        for existing in &basis {
            let dot: f64 = candidate.iter().zip(existing).map(|(a, b)| a * b).sum();
            for (c, e) in candidate.iter_mut().zip(existing) {
                *c -= dot * e;
            }
        }
        let norm: f64 = candidate.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-9 {
            for c in &mut candidate {
                *c /= norm;
            }
            basis.push(candidate);
        }
        if basis.len() == m - 1 {
            break;
        }
    }
    basis
}

/// Multiply column-major `a` (k x n) by column-major `b` (p x k), giving p x n.
fn matmul_cols(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.first().map_or(0, Vec::len);
    b.iter()
        .map(|bcol| {
            (0..n)
                .map(|r| a.iter().zip(bcol).map(|(acol, bv)| acol[r] * bv).sum())
                .collect()
        })
        .collect()
}

/// Gram matrix `Xᵀ X` for column-major `x`.
fn gram(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|ci| {
            x.iter()
                .map(|cj| ci.iter().zip(cj).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect()
}

// ───────────────────────────── VIF diagnostics ─────────────────────────────

/// Variance inflation factor for every regressor in a design matrix.
///
/// An intercept is added internally, as in PyBIDS' `est_vif`, and is not
/// itself reported. `columns` is column-major. VIF above ~5 conventionally
/// indicates problematic collinearity.
#[must_use]
pub fn est_vif(names: &[String], columns: &[Vec<f64>]) -> Vec<(String, f64)> {
    let keep: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.as_str() != "intercept")
        .map(|(i, _)| i)
        .collect();
    let sub: Vec<Vec<f64>> = keep.iter().map(|&i| columns[i].clone()).collect();
    let vifs = compute_vif(&sub);
    keep.iter()
        .zip(vifs)
        .map(|(&i, v)| (names[i].clone(), v))
        .collect()
}

/// Contrast weights laid out over the design matrix columns.
#[derive(Debug, Clone, Default)]
pub struct ContrastMatrix {
    /// One entry per contrast, in order.
    pub contrasts: Vec<String>,
    /// Design matrix column names.
    pub columns: Vec<String>,
    /// `weights[contrast][column]`, zero where a contrast does not load.
    pub weights: Vec<Vec<f64>>,
}

impl ContrastMatrix {
    /// The weight row for a named contrast.
    #[must_use]
    pub fn row(&self, contrast: &str) -> Option<&[f64]> {
        let i = self.contrasts.iter().position(|c| c == contrast)?;
        Some(&self.weights[i])
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.contrasts.is_empty()
    }
}

/// Build a contrast matrix from contrast definitions and design matrix columns.
///
/// Each contrast's `conditions`/`weights` pairs are scattered onto the design
/// matrix columns; columns a contrast does not mention get zero.
#[must_use]
pub fn generate_contrast_matrix(contrasts: &[ContrastInfo], cols: &[String]) -> ContrastMatrix {
    let weights = contrasts
        .iter()
        .map(|con| {
            let lookup: HashMap<&str, f64> = con
                .conditions
                .iter()
                .map(String::as_str)
                .zip(con.weights.iter().copied())
                .collect();
            cols.iter()
                .map(|c| lookup.get(c.as_str()).copied().unwrap_or(0.0))
                .collect()
        })
        .collect();
    ContrastMatrix {
        contrasts: contrasts.iter().map(|c| c.name.clone()).collect(),
        columns: cols.to_vec(),
        weights,
    }
}

/// Variance inflation factor for a single contrast.
///
/// Extends the effective-regressor construction of Smith et al. (2007),
/// *Meaningful design and contrast estimability* (NeuroImage): the regressors
/// the contrast loads on are rotated so that one axis is the effective
/// regressor and the rest span an orthogonal basis; regressors the contrast
/// does not touch are left alone. The VIF of the effective regressor in that
/// rebuilt design is the contrast's VIF.
///
/// `columns` is column-major; `contrast` has one weight per column. Returns
/// `None` when the contrast is all zeros or the rotation is degenerate.
#[must_use]
pub fn get_eff_reg_vif(columns: &[Vec<f64>], contrast: &[f64]) -> Option<f64> {
    if columns.len() != contrast.len() {
        return None;
    }
    let active: Vec<usize> = (0..contrast.len())
        .filter(|&i| contrast[i] != 0.0)
        .collect();
    if active.is_empty() {
        return None;
    }
    let nuisance: Vec<usize> = (0..contrast.len())
        .filter(|&i| contrast[i] == 0.0)
        .collect();

    // x: the contrast-involved regressors; con: their weights.
    let x: Vec<Vec<f64>> = active.iter().map(|&i| columns[i].clone()).collect();
    let con: Vec<f64> = active.iter().map(|&i| contrast[i]).collect();
    let n_obs = x.first().map_or(0, Vec::len);
    if n_obs == 0 {
        return None;
    }

    let q = pinv_sym(&gram(&x));

    // f1 = pinv(con q conᵀ) — con is a single row, so this is a scalar.
    let q_con: Vec<f64> = q
        .iter()
        .map(|row| row.iter().zip(&con).map(|(a, b)| a * b).sum())
        .collect();
    let con_q_con: f64 = con.iter().zip(&q_con).map(|(a, b)| a * b).sum();
    if con_q_con.abs() < 1e-18 {
        return None;
    }
    let f1 = 1.0 / con_q_con;

    // eff_reg = x q conᵀ f1
    let eff_reg: Vec<f64> = (0..n_obs)
        .map(|r| x.iter().zip(&q_con).map(|(col, w)| col[r] * w).sum::<f64>() * f1)
        .collect();

    // con2_t: an orthonormal basis for the null space of con.
    let con2_t = null_space_of_row(&con);

    // pc = conᵀ f1 con q ; con3_t = con2_t - pc con2_t
    let con3_t: Vec<Vec<f64>> = con2_t
        .iter()
        .map(|basis_col| {
            let con_q_basis: f64 = q_con.iter().zip(basis_col).map(|(a, b)| a * b).sum();
            basis_col
                .iter()
                .zip(&con)
                .map(|(b, c)| b - c * f1 * con_q_basis)
                .collect()
        })
        .collect();

    let mut other_reg: Vec<Vec<f64>> = Vec::new();
    if !con3_t.is_empty() {
        // f3 = pinv(con3_tᵀ q con3_t)
        let q_con3 = matmul_cols(&q, &con3_t);
        let inner: Vec<Vec<f64>> = con3_t
            .iter()
            .map(|ci| {
                q_con3
                    .iter()
                    .map(|cj| ci.iter().zip(cj).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect();
        let f3 = pinv_sym(&inner);
        // other_reg = x q con3_t f3
        let x_q_con3 = matmul_cols(&x, &q_con3);
        other_reg = matmul_cols(&x_q_con3, &f3);
    }

    // The rebuilt design: effective regressor first, then the orthogonal
    // basis, then the untouched nuisance regressors.
    let mut design = vec![eff_reg];
    design.extend(other_reg);
    design.extend(nuisance.iter().map(|&i| columns[i].clone()));

    compute_vif(&design).first().copied()
}

/// Variance inflation factors for every contrast on a node output.
///
/// Returns `(contrast_name, vif)` pairs. Contrasts whose VIF cannot be
/// computed (all-zero weights, or a degenerate rotation) are omitted.
#[must_use]
pub fn get_all_contrast_vif(output: &StatsModelsNodeOutput) -> Vec<(String, f64)> {
    let Some((names, columns)) = output.design_matrix.as_ref() else {
        return Vec::new();
    };
    let matrix = generate_contrast_matrix(&output.contrasts, names);
    matrix
        .contrasts
        .iter()
        .zip(&matrix.weights)
        .filter_map(|(name, weights)| get_eff_reg_vif(columns, weights).map(|v| (name.clone(), v)))
        .collect()
}

/// Pairwise Pearson correlations between design matrix columns.
///
/// Returned row-major: `corr[i][j]`.
#[must_use]
pub fn correlation_matrix(columns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    columns
        .iter()
        .map(|ci| columns.iter().map(|cj| pearson_r(ci, cj)).collect())
        .collect()
}

/// Diagnostics for a single node output.
#[derive(Debug, Clone, Default)]
pub struct NodeReport {
    pub node_name: String,
    pub entities: bids_core::entities::StringEntities,
    pub level: String,
    pub n_obs: usize,
    pub n_predictors: usize,
    /// VIF per regressor.
    pub vif: Vec<(String, f64)>,
    /// VIF per contrast.
    pub contrast_vif: Vec<(String, f64)>,
    pub contrast_matrix: ContrastMatrix,
    /// Column names and column-major data of the design matrix.
    pub design_matrix: Option<(Vec<String>, Vec<Vec<f64>>)>,
    /// Transformations that failed for this node.
    pub transform_errors: Vec<String>,
}

impl NodeReport {
    /// The largest regressor VIF, if any regressors were assessed.
    #[must_use]
    pub fn max_vif(&self) -> Option<f64> {
        self.vif
            .iter()
            .map(|(_, v)| *v)
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.max(v)))
            })
    }
}

/// Build diagnostics for one node output.
///
/// Unlike PyBIDS, which returns an empty report for anything other than a run
/// level node, this reports at every level — a collinear group-level design is
/// just as worth catching.
#[must_use]
pub fn node_report(output: &StatsModelsNodeOutput) -> NodeReport {
    let (names, columns) = output
        .design_matrix
        .clone()
        .unwrap_or_else(|| (Vec::new(), Vec::new()));
    NodeReport {
        node_name: output.node_name.clone(),
        entities: output.entities.clone(),
        level: String::new(),
        n_obs: columns.first().map_or(0, Vec::len),
        n_predictors: names.len(),
        vif: est_vif(&names, &columns),
        contrast_vif: get_all_contrast_vif(output),
        contrast_matrix: generate_contrast_matrix(&output.contrasts, &names),
        design_matrix: output.design_matrix.clone(),
        transform_errors: output
            .transform_errors
            .iter()
            .map(std::string::ToString::to_string)
            .collect(),
    }
}

// ───────────────────────── Naming helpers ─────────────────────────

/// Strip a dataset root prefix from a path.
///
/// Reports are more readable with dataset-relative paths, and they stay
/// reproducible across machines. Corresponds to PyBIDS' `deroot`.
///
/// # Example
///
/// ```
/// use bids_modeling::report::deroot;
///
/// assert_eq!(deroot("/data/ds001/sub-01/func/x.nii", "/data/ds001"), "sub-01/func/x.nii");
/// assert_eq!(deroot("/elsewhere/x.nii", "/data/ds001"), "/elsewhere/x.nii");
/// ```
#[must_use]
pub fn deroot(value: &str, root: &str) -> String {
    let Some(rest) = value.strip_prefix(root) else {
        return value.to_string();
    };
    rest.strip_prefix('/').unwrap_or(rest).to_string()
}

/// Turn a contrast name into something readable.
///
/// `_gt_`, `_lt_` and `_vs_` become `>`, `<` and `vs.`. Corresponds to
/// PyBIDS' `displayify`, except that PyBIDS emits the HTML entities `&gt;`
/// and `&lt;`; this returns the plain characters, because everything placed
/// into a report here is escaped on the way out.
///
/// # Example
///
/// ```
/// use bids_modeling::report::displayify;
///
/// assert_eq!(displayify("face_gt_house"), "face > house");
/// assert_eq!(displayify("a_vs_b"), "a vs. b");
/// ```
#[must_use]
pub fn displayify(contrast_name: &str) -> String {
    contrast_name
        .replace("_gt_", " > ")
        .replace("_lt_", " < ")
        .replace("_vs_", " vs. ")
}

/// Convert a string to an alphanumeric camelCase identifier.
///
/// Every non-alphanumeric character becomes a word break, then the words are
/// joined camelCase. Useful for turning variable names into HTML ids and
/// filenames. Corresponds to PyBIDS' `to_alphanum`.
///
/// # Example
///
/// ```
/// use bids_modeling::report::to_alphanum;
///
/// assert_eq!(to_alphanum("abc123"), "abc123");
/// assert_eq!(to_alphanum("a four word phrase"), "aFourWordPhrase");
/// assert_eq!(to_alphanum("trial_type.face"), "trialTypeFace");
/// assert_eq!(to_alphanum("array[0]"), "array0");
/// ```
#[must_use]
pub fn to_alphanum(string: &str) -> String {
    let mut out = String::with_capacity(string.len());
    let mut capitalize = false;
    let mut started = false;
    for c in string.chars() {
        if c.is_ascii_alphanumeric() {
            if capitalize && started {
                out.extend(c.to_uppercase());
            } else {
                out.push(c);
            }
            capitalize = false;
            started = true;
        } else {
            // A run of separators is one word break.
            capitalize = true;
        }
    }
    out
}

// ─────────────────────────────── Figures ───────────────────────────────
//
// Colour roles are emitted as CSS custom properties and referenced from the
// SVG, so light and dark are two selected sets of steps rather than an
// automatic inversion.
//
// Sequential (non-negative magnitude) uses the documented blue ramp. Diverging
// (signed values, correlations) pairs that blue against red across a neutral
// gray midpoint; the red arm is interpolated from the midpoint to the palette's
// red pole, since only the blue ramp is enumerated. Each arm validates as a
// one-hue ordinal ramp in both modes; the near-surface end is allowed to
// recede because these are continuous encodings, not discrete ordinal marks.

/// Number of steps per sequential ramp.
const SEQ_STEPS: usize = 7;
/// Number of steps per diverging arm, excluding the midpoint.
const DIV_STEPS: usize = 4;

const STYLE: &str = r#"
.bidsrs-report {
  color-scheme: light;
  --surface-1: #fcfcfb;
  --page: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --gridline: #e1e0d9;
  --baseline: #c3c2b7;
  --border: rgba(11,11,11,0.10);
  --critical: #d03b3b;
  --seq-1: #cde2fb; --seq-2: #9ec5f4; --seq-3: #6da7ec; --seq-4: #3987e5;
  --seq-5: #256abf; --seq-6: #184f95; --seq-7: #0d366b;
  --div-mid: #f0efec;
  --div-pos-1: #9ec5f4; --div-pos-2: #5598e7; --div-pos-3: #2a78d6; --div-pos-4: #0d366b;
  --div-neg-1: #edc6c3; --div-neg-2: #e99492; --div-neg-3: #e34948; --div-neg-4: #8d2d2d;
  background: var(--page);
  color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px;
  line-height: 1.5;
  margin: 0;
  padding: 32px;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .bidsrs-report { color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
    --text-secondary: #c3c2b7; --muted: #898781; --gridline: #2c2c2a;
    --baseline: #383835; --border: rgba(255,255,255,0.10); --critical: #d03b3b;
    --seq-1: #0d366b; --seq-2: #184f95; --seq-3: #256abf; --seq-4: #3987e5;
    --seq-5: #6da7ec; --seq-6: #9ec5f4; --seq-7: #cde2fb;
    --div-mid: #383835;
    --div-pos-1: #184f95; --div-pos-2: #256abf; --div-pos-3: #3987e5; --div-pos-4: #86b6ef;
    --div-neg-1: #644442; --div-neg-2: #985251; --div-neg-3: #e66767; --div-neg-4: #ef9c9c;
  }
}
:root[data-theme="dark"] .bidsrs-report { color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d; --text-primary: #ffffff;
  --text-secondary: #c3c2b7; --muted: #898781; --gridline: #2c2c2a;
  --baseline: #383835; --border: rgba(255,255,255,0.10); --critical: #d03b3b;
  --seq-1: #0d366b; --seq-2: #184f95; --seq-3: #256abf; --seq-4: #3987e5;
  --seq-5: #6da7ec; --seq-6: #9ec5f4; --seq-7: #cde2fb;
  --div-mid: #383835;
  --div-pos-1: #184f95; --div-pos-2: #256abf; --div-pos-3: #3987e5; --div-pos-4: #86b6ef;
  --div-neg-1: #644442; --div-neg-2: #985251; --div-neg-3: #e66767; --div-neg-4: #ef9c9c;
}
.bidsrs-report h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
.bidsrs-report h2 { font-size: 16px; font-weight: 600; margin: 32px 0 12px; }
.bidsrs-report h3 { font-size: 14px; font-weight: 600; margin: 24px 0 8px; }
.bidsrs-report .sub { color: var(--text-secondary); margin: 0 0 24px; }
.bidsrs-report .node { background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; }
.bidsrs-report .tiles { display: flex; flex-wrap: wrap; gap: 24px; margin: 0 0 8px; }
.bidsrs-report .tile .v { font-size: 26px; font-weight: 600; }
.bidsrs-report .tile .k { color: var(--text-secondary); font-size: 12px;
  text-transform: uppercase; letter-spacing: 0.04em; }
.bidsrs-report .tile .v.alarm { color: var(--critical); }
.bidsrs-report figure { margin: 0 0 8px; }
.bidsrs-report figcaption { color: var(--text-secondary); font-size: 12px; margin-top: 4px; }
.bidsrs-report svg text { font-family: inherit; }
.bidsrs-report svg .cell { stroke: var(--surface-1); stroke-width: 0; }
.bidsrs-report svg .cell:hover { stroke: var(--text-primary); stroke-width: 1.5; }
.bidsrs-report svg .bar:hover { opacity: 0.82; }
.bidsrs-report details { margin: 8px 0 0; }
.bidsrs-report summary { cursor: pointer; color: var(--text-secondary); font-size: 13px; }
.bidsrs-report table { border-collapse: collapse; margin-top: 8px; font-size: 12px;
  font-variant-numeric: tabular-nums; }
.bidsrs-report th, .bidsrs-report td { border: 1px solid var(--gridline);
  padding: 3px 8px; text-align: right; }
.bidsrs-report th:first-child, .bidsrs-report td:first-child { text-align: left; }
.bidsrs-report th { color: var(--text-secondary); font-weight: 600; }
.bidsrs-report .warn { color: var(--critical); }
.bidsrs-report .empty { color: var(--text-secondary); font-style: italic; }
@media (forced-colors: active) {
  .bidsrs-report svg .cell { forced-color-adjust: none; stroke: CanvasText; stroke-width: 0.5; }
  .bidsrs-report details { display: block; }
  .bidsrs-report details > summary::after { content: " (figures above rely on colour; use this table)"; }
}
"#;

/// Escape text for inclusion in XML/HTML content or an attribute value.
fn esc(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '&' => "&amp;".into(),
            '<' => "&lt;".into(),
            '>' => "&gt;".into(),
            '"' => "&quot;".into(),
            '\'' => "&#39;".into(),
            other => other.to_string(),
        })
        .collect()
}

/// Format a number compactly: integers bare, everything else to 3 decimals.
fn fmt(v: f64) -> String {
    if !v.is_finite() {
        return if v.is_nan() {
            "n/a".into()
        } else if v > 0.0 {
            "∞".into()
        } else {
            "-∞".into()
        };
    }
    if (v - v.round()).abs() < 1e-9 && v.abs() < 1e6 {
        format!("{v:.0}")
    } else {
        format!("{v:.3}")
    }
}

/// Map a value in `[0, 1]` onto a sequential ramp variable.
fn seq_var(t: f64) -> String {
    let idx = ((t.clamp(0.0, 1.0) * (SEQ_STEPS - 1) as f64).round() as usize) + 1;
    format!("var(--seq-{idx})")
}

/// Map a value in `[-1, 1]` onto the diverging ramp.
fn div_var(t: f64) -> String {
    let t = t.clamp(-1.0, 1.0);
    let mag = t.abs();
    // The innermost band around zero is the neutral midpoint.
    let step = (mag * DIV_STEPS as f64).ceil() as usize;
    if step == 0 {
        return "var(--div-mid)".into();
    }
    if t > 0.0 {
        format!("var(--div-pos-{step})")
    } else {
        format!("var(--div-neg-{step})")
    }
}

/// Layout knobs for the design matrix figure.
#[derive(Debug, Clone)]
pub struct FigureOptions {
    /// Maximum number of observations drawn before the rows are subsampled.
    pub max_rows: usize,
    /// Plot height in pixels.
    pub height: f64,
    /// Width of one design matrix column in pixels.
    pub column_width: f64,
}

impl Default for FigureOptions {
    fn default() -> Self {
        Self {
            max_rows: 200,
            height: 340.0,
            column_width: 34.0,
        }
    }
}

/// Render a design matrix as an inline SVG heatmap.
///
/// Rows are observations (time, at the run level) and columns are regressors.
/// **Each regressor is scaled to its own range**, because design matrix columns
/// routinely differ by orders of magnitude — an intercept of 1.0 beside an HRF
/// regressor of ±0.1 on one shared scale renders the interesting columns as
/// noise. The colour therefore encodes each regressor's relative magnitude, not
/// an absolute value; read absolute values from the table view or the
/// timecourse.
///
/// Rows beyond [`FigureOptions::max_rows`] are subsampled evenly. `columns` is
/// column-major.
#[must_use]
pub fn plot_design_matrix(names: &[String], columns: &[Vec<f64>], opts: &FigureOptions) -> String {
    if names.is_empty() || columns.is_empty() {
        return String::new();
    }
    let n_obs = columns.first().map_or(0, Vec::len);
    if n_obs == 0 {
        return String::new();
    }

    let shown: Vec<usize> = if n_obs > opts.max_rows {
        (0..opts.max_rows)
            .map(|i| i * (n_obs - 1) / (opts.max_rows - 1).max(1))
            .collect()
    } else {
        (0..n_obs).collect()
    };

    let ranges: Vec<(f64, f64)> = columns.iter().map(|c| finite_range(c)).collect();

    let gap = 2.0;
    let pad_left = 52.0;
    let pad_top = 76.0;
    let col_w = opts.column_width;
    let plot_w = names.len() as f64 * col_w;
    let row_h = (opts.height / shown.len() as f64).max(1.0);
    let plot_h = row_h * shown.len() as f64;
    let width = pad_left + plot_w + 24.0;
    let height = pad_top + plot_h + 56.0;

    let mut svg = format!(
        "<svg viewBox=\"0 0 {width:.0} {height:.0}\" width=\"{width:.0}\" height=\"{height:.0}\" \
         role=\"img\" aria-label=\"Design matrix heatmap\" \
         xmlns=\"http://www.w3.org/2000/svg\">"
    );

    // Column labels, angled so long regressor names do not collide.
    for (j, name) in names.iter().enumerate() {
        let x = pad_left + j as f64 * col_w + col_w / 2.0;
        svg.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{y:.1}\" transform=\"rotate(-38 {x:.1} {y:.1})\" \
             text-anchor=\"start\" font-size=\"11\" fill=\"var(--text-secondary)\">{}</text>",
            esc(name),
            y = pad_top - 8.0
        ));
    }

    // Cells. A 2px gap separates regressors; rows stay contiguous so a run's
    // time course reads as continuous.
    for (j, col) in columns.iter().enumerate() {
        let x = pad_left + j as f64 * col_w;
        let (lo, hi) = ranges[j];
        let span = (hi - lo).max(1e-12);
        for (r, &obs) in shown.iter().enumerate() {
            let v = col.get(obs).copied().unwrap_or(f64::NAN);
            let fill = if v.is_finite() {
                seq_var((v - lo) / span)
            } else {
                "var(--gridline)".to_string()
            };
            let y = pad_top + r as f64 * row_h;
            svg.push_str(&format!(
                "<rect class=\"cell\" x=\"{x:.1}\" y=\"{y:.2}\" width=\"{w:.1}\" \
                 height=\"{row_h:.2}\" fill=\"{fill}\"><title>{} · row {obs} · {}</title></rect>",
                esc(&names[j]),
                fmt(v),
                w = col_w - gap
            ));
        }
    }

    // Row axis: first, middle and last observation index.
    for r in [0usize, shown.len() / 2, shown.len() - 1] {
        let obs = shown[r];
        let y = pad_top + r as f64 * row_h + row_h / 2.0;
        svg.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"end\" dominant-baseline=\"middle\" \
             font-size=\"10\" fill=\"var(--muted)\">{obs}</text>",
            x = pad_left - 8.0
        ));
    }
    svg.push_str(&format!(
        "<text x=\"14\" y=\"{y:.1}\" text-anchor=\"middle\" font-size=\"10\" \
         fill=\"var(--muted)\" transform=\"rotate(-90 14 {y:.1})\">observation</text>",
        y = pad_top + plot_h / 2.0
    ));

    // Legend: the scale is relative, so label the ends rather than numbers.
    svg.push_str(&relative_legend(pad_left, pad_top + plot_h + 28.0));
    svg.push_str("</svg>");
    svg
}

/// Min and max over the finite entries of a column.
fn finite_range(col: &[f64]) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in col.iter().filter(|v| v.is_finite()) {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    if lo.is_finite() { (lo, hi) } else { (0.0, 1.0) }
}

/// A sequential legend labelled by relative position rather than by value.
fn relative_legend(x: f64, y: f64) -> String {
    let sw = 18.0;
    let mut out = String::new();
    for i in 0..SEQ_STEPS {
        out.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"10\" fill=\"{}\"/>",
            x + i as f64 * sw,
            sw - 2.0,
            seq_var(i as f64 / (SEQ_STEPS - 1) as f64)
        ));
    }
    let end = x + SEQ_STEPS as f64 * sw - 2.0;
    out.push_str(&format!(
        "<text x=\"{x:.1}\" y=\"{ty:.1}\" font-size=\"10\" \
         fill=\"var(--muted)\">regressor min</text>\
         <text x=\"{end:.1}\" y=\"{ty:.1}\" text-anchor=\"end\" font-size=\"10\" \
         fill=\"var(--muted)\">max</text>",
        ty = y + 22.0
    ));
    out
}

/// Render each regressor's time course as a small multiple.
///
/// One panel per regressor, sharing an x axis and each scaled to its own
/// range. This is the form that actually answers "what does this regressor do
/// over the run" — the heatmap above is the compact overview.
///
/// `columns` is column-major.
#[must_use]
pub fn plot_design_timecourse(names: &[String], columns: &[Vec<f64>]) -> String {
    if names.is_empty() || columns.is_empty() {
        return String::new();
    }
    let n_obs = columns.first().map_or(0, Vec::len);
    if n_obs < 2 {
        return String::new();
    }

    let pad_left = 160.0;
    let panel_h = 44.0;
    let panel_gap = 8.0;
    let plot_w = 620.0;
    let pad_top = 12.0;
    let n = names.len();
    let width = pad_left + plot_w + 24.0;
    let height = pad_top + n as f64 * (panel_h + panel_gap) + 28.0;

    let mut svg = format!(
        "<svg viewBox=\"0 0 {width:.0} {height:.0}\" width=\"{width:.0}\" height=\"{height:.0}\" \
         role=\"img\" aria-label=\"Regressor time courses\" \
         xmlns=\"http://www.w3.org/2000/svg\">"
    );

    for (j, (name, col)) in names.iter().zip(columns).enumerate() {
        let top = pad_top + j as f64 * (panel_h + panel_gap);
        let (lo, hi) = finite_range(col);
        let span = (hi - lo).max(1e-12);

        // Panel label and range, in text ink — the line carries no identity
        // colour because each panel holds exactly one series.
        svg.push_str(&format!(
            "<text x=\"{lx:.1}\" y=\"{ly:.1}\" text-anchor=\"end\" font-size=\"11\" \
             fill=\"var(--text-secondary)\">{}</text>\
             <text x=\"{lx:.1}\" y=\"{ly2:.1}\" text-anchor=\"end\" font-size=\"10\" \
             fill=\"var(--muted)\">{} … {}</text>",
            esc(name),
            fmt(lo),
            fmt(hi),
            lx = pad_left - 12.0,
            ly = top + panel_h / 2.0 - 2.0,
            ly2 = top + panel_h / 2.0 + 12.0
        ));

        // Baseline at the panel's zero, when zero is inside the range.
        if lo <= 0.0 && hi >= 0.0 {
            let zy = top + panel_h - (0.0 - lo) / span * panel_h;
            svg.push_str(&format!(
                "<line x1=\"{pad_left:.1}\" y1=\"{zy:.1}\" x2=\"{x2:.1}\" y2=\"{zy:.1}\" \
                 stroke=\"var(--gridline)\" stroke-width=\"1\"/>",
                x2 = pad_left + plot_w
            ));
        }

        let mut d = String::new();
        for (i, v) in col.iter().enumerate() {
            let x = pad_left + i as f64 / (n_obs - 1) as f64 * plot_w;
            let y = top + panel_h - (v.clamp(lo, hi) - lo) / span * panel_h;
            d.push_str(&format!("{}{x:.1} {y:.2}", if i == 0 { "M" } else { "L" }));
            d.push(' ');
        }
        svg.push_str(&format!(
            "<path d=\"{}\" fill=\"none\" stroke=\"var(--seq-4)\" stroke-width=\"2\" \
             stroke-linejoin=\"round\" stroke-linecap=\"round\"><title>{}</title></path>",
            d.trim_end(),
            esc(name)
        ));
    }

    let axis_y = pad_top + n as f64 * (panel_h + panel_gap) - panel_gap + 4.0;
    svg.push_str(&format!(
        "<text x=\"{pad_left:.1}\" y=\"{ty:.1}\" font-size=\"10\" \
         fill=\"var(--muted)\">0</text>\
         <text x=\"{x2:.1}\" y=\"{ty:.1}\" text-anchor=\"end\" font-size=\"10\" \
         fill=\"var(--muted)\">{} observations</text>",
        n_obs,
        ty = axis_y + 14.0,
        x2 = pad_left + plot_w
    ));
    svg.push_str("</svg>");
    svg
}

/// A horizontal colour-scale strip with end labels.
fn scale_legend(x: f64, y: f64, lo: f64, hi: f64, signed: bool) -> String {
    let steps: Vec<String> = if signed {
        (0..=(DIV_STEPS * 2))
            .map(|i| div_var(i as f64 / DIV_STEPS as f64 - 1.0))
            .collect()
    } else {
        (0..SEQ_STEPS)
            .map(|i| seq_var(i as f64 / (SEQ_STEPS - 1) as f64))
            .collect()
    };
    let sw = 18.0;
    let mut out = String::new();
    for (i, fill) in steps.iter().enumerate() {
        out.push_str(&format!(
            "<rect x=\"{:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"10\" fill=\"{fill}\"/>",
            x + i as f64 * sw,
            sw - 2.0
        ));
    }
    let end = x + steps.len() as f64 * sw - 2.0;
    let (left_label, right_label) = if signed {
        (fmt(-lo.abs().max(hi.abs())), fmt(lo.abs().max(hi.abs())))
    } else {
        (fmt(lo), fmt(hi))
    };
    out.push_str(&format!(
        "<text x=\"{x:.1}\" y=\"{ty:.1}\" font-size=\"10\" fill=\"var(--muted)\">{left_label}</text>\
         <text x=\"{end:.1}\" y=\"{ty:.1}\" text-anchor=\"end\" font-size=\"10\" \
         fill=\"var(--muted)\">{right_label}</text>",
        ty = y + 22.0
    ));
    out
}

/// Render the correlation matrix of a design matrix as an inline SVG heatmap.
///
/// Uses a diverging ramp: red for negative correlation, a neutral midpoint at
/// zero, blue for positive. Cells are labelled with the coefficient when the
/// matrix is small enough for the text to fit.
#[must_use]
pub fn plot_corr_matrix(names: &[String], columns: &[Vec<f64>]) -> String {
    let n = names.len().min(columns.len());
    if n == 0 {
        return String::new();
    }
    let corr = correlation_matrix(&columns[..n]);

    let cell = if n <= 8 { 46.0 } else { 26.0 };
    let gap = 2.0;
    let pad_left = 96.0;
    let pad_top = 96.0;
    let plot = n as f64 * cell;
    let width = pad_left + plot + 24.0;
    let height = pad_top + plot + 56.0;

    let mut svg = format!(
        "<svg viewBox=\"0 0 {width:.0} {height:.0}\" width=\"{width:.0}\" height=\"{height:.0}\" \
         role=\"img\" aria-label=\"Regressor correlation matrix\" \
         xmlns=\"http://www.w3.org/2000/svg\">"
    );

    for (j, name) in names.iter().take(n).enumerate() {
        let x = pad_left + j as f64 * cell + cell / 2.0;
        svg.push_str(&format!(
            "<text x=\"{x:.1}\" y=\"{y:.1}\" transform=\"rotate(-38 {x:.1} {y:.1})\" \
             text-anchor=\"start\" font-size=\"11\" fill=\"var(--text-secondary)\">{}</text>",
            esc(name),
            y = pad_top - 8.0
        ));
        let y = pad_top + j as f64 * cell + cell / 2.0;
        svg.push_str(&format!(
            "<text x=\"{x2:.1}\" y=\"{y:.1}\" text-anchor=\"end\" dominant-baseline=\"middle\" \
             font-size=\"11\" fill=\"var(--text-secondary)\">{}</text>",
            esc(name),
            x2 = pad_left - 8.0
        ));
    }

    for (i, row) in corr.iter().enumerate() {
        for (j, &r) in row.iter().enumerate() {
            let x = pad_left + j as f64 * cell;
            let y = pad_top + i as f64 * cell;
            svg.push_str(&format!(
                "<rect class=\"cell\" x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{w:.1}\" \
                 rx=\"2\" fill=\"{}\"><title>{} × {}: r = {}</title></rect>",
                div_var(r),
                esc(&names[i]),
                esc(&names[j]),
                fmt(r),
                w = cell - gap
            ));
            if n <= 8 {
                // Ink stays in text tokens; the cell colour carries magnitude.
                svg.push_str(&format!(
                    "<text x=\"{tx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" \
                     dominant-baseline=\"middle\" font-size=\"10\" pointer-events=\"none\" \
                     fill=\"var(--text-primary)\">{:.2}</text>",
                    r,
                    tx = x + (cell - gap) / 2.0,
                    ty = y + (cell - gap) / 2.0
                ));
            }
        }
    }

    svg.push_str(&scale_legend(
        pad_left,
        pad_top + plot + 28.0,
        -1.0,
        1.0,
        true,
    ));
    svg.push_str("</svg>");
    svg
}

/// Render variance inflation factors as a horizontal bar chart.
///
/// A reference line marks VIF = 5, the conventional threshold above which
/// collinearity is treated as a problem.
#[must_use]
pub fn plot_vif(vifs: &[(String, f64)]) -> String {
    if vifs.is_empty() {
        return String::new();
    }
    const THRESHOLD: f64 = 5.0;
    let finite_max = vifs
        .iter()
        .map(|(_, v)| *v)
        .filter(|v| v.is_finite())
        .fold(0.0f64, f64::max);
    let axis_max = finite_max.max(THRESHOLD * 1.2).max(1.0);

    let bar_h = 18.0;
    let step = 26.0;
    let pad_left = 160.0;
    let pad_top = 16.0;
    let plot_w = 320.0;
    let width = pad_left + plot_w + 72.0;
    let height = pad_top + vifs.len() as f64 * step + 28.0;

    let mut svg = format!(
        "<svg viewBox=\"0 0 {width:.0} {height:.0}\" width=\"{width:.0}\" height=\"{height:.0}\" \
         role=\"img\" aria-label=\"Variance inflation factor per regressor\" \
         xmlns=\"http://www.w3.org/2000/svg\">"
    );

    // Threshold reference line, behind the bars.
    let tx = pad_left + THRESHOLD / axis_max * plot_w;
    svg.push_str(&format!(
        "<line x1=\"{tx:.1}\" y1=\"{pad_top:.1}\" x2=\"{tx:.1}\" y2=\"{y2:.1}\" \
         stroke=\"var(--baseline)\" stroke-width=\"1\" stroke-dasharray=\"3 3\"/>\
         <text x=\"{tx:.1}\" y=\"{ty:.1}\" text-anchor=\"middle\" font-size=\"10\" \
         fill=\"var(--muted)\">VIF 5</text>",
        y2 = pad_top + vifs.len() as f64 * step,
        ty = pad_top + vifs.len() as f64 * step + 14.0
    ));

    for (i, (name, v)) in vifs.iter().enumerate() {
        let y = pad_top + i as f64 * step;
        let drawn = if v.is_finite() { *v } else { axis_max };
        let w = (drawn / axis_max * plot_w).max(2.0);
        let over = !v.is_finite() || *v > THRESHOLD;
        let fill = if over {
            "var(--div-neg-3)"
        } else {
            "var(--seq-4)"
        };
        svg.push_str(&format!(
            "<text x=\"{lx:.1}\" y=\"{cy:.1}\" text-anchor=\"end\" dominant-baseline=\"middle\" \
             font-size=\"11\" fill=\"var(--text-secondary)\">{}</text>\
             <rect class=\"bar\" x=\"{pad_left:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" \
             height=\"{bar_h}\" rx=\"4\" fill=\"{fill}\"><title>{}: VIF {}</title></rect>\
             <text x=\"{vx:.1}\" y=\"{cy:.1}\" dominant-baseline=\"middle\" font-size=\"11\" \
             fill=\"var(--text-primary)\">{}</text>",
            esc(name),
            esc(name),
            fmt(*v),
            fmt(*v),
            lx = pad_left - 10.0,
            cy = y + bar_h / 2.0,
            vx = pad_left + w + 8.0
        ));
    }

    // Baseline at zero.
    svg.push_str(&format!(
        "<line x1=\"{pad_left:.1}\" y1=\"{pad_top:.1}\" x2=\"{pad_left:.1}\" y2=\"{y2:.1}\" \
         stroke=\"var(--baseline)\" stroke-width=\"1\"/>",
        y2 = pad_top + vifs.len() as f64 * step
    ));
    svg.push_str("</svg>");
    svg
}

// ──────────────────────────── HTML document ────────────────────────────

/// Render a table of `(label, value)` rows.
fn value_table(header: (&str, &str), rows: &[(String, f64)], warn_above: Option<f64>) -> String {
    let mut out = format!(
        "<table><thead><tr><th>{}</th><th>{}</th></tr></thead><tbody>",
        esc(header.0),
        esc(header.1)
    );
    for (name, v) in rows {
        let class = match warn_above {
            Some(t) if !v.is_finite() || *v > t => " class=\"warn\"",
            _ => "",
        };
        out.push_str(&format!(
            "<tr><td>{}</td><td{class}>{}</td></tr>",
            esc(name),
            fmt(*v)
        ));
    }
    out.push_str("</tbody></table>");
    out
}

/// Render a matrix as an HTML table.
fn matrix_table(row_labels: &[String], col_labels: &[String], rows: &[Vec<f64>]) -> String {
    let mut out = String::from("<table><thead><tr><th></th>");
    for c in col_labels {
        out.push_str(&format!("<th>{}</th>", esc(c)));
    }
    out.push_str("</tr></thead><tbody>");
    for (label, row) in row_labels.iter().zip(rows) {
        out.push_str(&format!("<tr><td>{}</td>", esc(label)));
        for v in row {
            out.push_str(&format!("<td>{}</td>", fmt(*v)));
        }
        out.push_str("</tr>");
    }
    out.push_str("</tbody></table>");
    out
}

/// A figure with its accompanying table view.
fn figure(svg: &str, caption: &str, table_label: &str, table: &str) -> String {
    if svg.is_empty() {
        return String::new();
    }
    format!(
        "<figure>{svg}<figcaption>{}</figcaption></figure>\
         <details><summary>{}</summary>{table}</details>",
        esc(caption),
        esc(table_label)
    )
}

/// A design matrix rendered row-major for the table view, capped in length.
fn design_rows(columns: &[Vec<f64>], limit: usize) -> (Vec<String>, Vec<Vec<f64>>) {
    let n_obs = columns.first().map_or(0, Vec::len);
    let shown = n_obs.min(limit);
    let labels = (0..shown).map(|i| i.to_string()).collect();
    let rows = (0..shown)
        .map(|r| columns.iter().map(|c| c[r]).collect())
        .collect();
    (labels, rows)
}

/// Render the diagnostics for one node as an HTML fragment.
#[must_use]
pub fn node_report_html(report: &NodeReport) -> String {
    let mut out = String::from("<section class=\"node\">");
    let ents: Vec<String> = {
        let mut e: Vec<String> = report
            .entities
            .iter()
            .map(|(k, v)| format!("{k}-{v}"))
            .collect();
        e.sort();
        e
    };
    out.push_str(&format!(
        "<h2>{}</h2><p class=\"sub\">{}</p>",
        esc(&report.node_name),
        if ents.is_empty() {
            "all data".to_string()
        } else {
            esc(&ents.join(" · "))
        }
    ));

    // Stat tiles: the headline numbers, no chart needed.
    let max_vif = report.max_vif();
    let alarm = max_vif.is_some_and(|v| !v.is_finite() || v > 5.0);
    out.push_str(&format!(
        "<div class=\"tiles\">\
         <div class=\"tile\"><div class=\"v\">{}</div><div class=\"k\">observations</div></div>\
         <div class=\"tile\"><div class=\"v\">{}</div><div class=\"k\">predictors</div></div>\
         <div class=\"tile\"><div class=\"v\">{}</div><div class=\"k\">contrasts</div></div>\
         <div class=\"tile\"><div class=\"v{}\">{}</div><div class=\"k\">max VIF</div></div>\
         </div>",
        report.n_obs,
        report.n_predictors,
        report.contrast_matrix.contrasts.len(),
        if alarm { " alarm" } else { "" },
        max_vif.map_or_else(|| "–".into(), fmt),
    ));

    if !report.transform_errors.is_empty() {
        out.push_str("<h3 class=\"warn\">Transformation errors</h3><ul>");
        for e in &report.transform_errors {
            out.push_str(&format!("<li class=\"warn\">{}</li>", esc(e)));
        }
        out.push_str("</ul>");
    }

    if let Some((names, columns)) = &report.design_matrix {
        out.push_str("<h3>Design matrix</h3>");
        let (row_labels, rows) = design_rows(columns, 200);
        out.push_str(&figure(
            &plot_design_matrix(names, columns, &FigureOptions::default()),
            "Regressors across observations; each regressor is scaled to its own range.",
            "Table view",
            &matrix_table(&row_labels, names, &rows),
        ));
        let timecourse = plot_design_timecourse(names, columns);
        if !timecourse.is_empty() {
            out.push_str(&format!(
                "<figure>{timecourse}<figcaption>{}</figcaption></figure>",
                esc("Time course per regressor, each on its own scale.")
            ));
        }

        out.push_str("<h3>Regressor correlations</h3>");
        let corr = correlation_matrix(columns);
        out.push_str(&figure(
            &plot_corr_matrix(names, columns),
            "Pearson correlation between regressors.",
            "Table view",
            &matrix_table(names, names, &corr),
        ));
    }

    if !report.vif.is_empty() {
        out.push_str("<h3>Variance inflation</h3>");
        out.push_str(&figure(
            &plot_vif(&report.vif),
            "VIF per regressor; above 5 indicates problematic collinearity.",
            "Table view",
            &value_table(("Regressor", "VIF"), &report.vif, Some(5.0)),
        ));
    }

    if !report.contrast_vif.is_empty() {
        out.push_str("<h3>Contrast variance inflation</h3>");
        out.push_str(&figure(
            &plot_vif(&report.contrast_vif),
            "VIF per contrast, via the effective-regressor construction.",
            "Table view",
            &value_table(("Contrast", "VIF"), &report.contrast_vif, Some(5.0)),
        ));
    }

    if !report.contrast_matrix.is_empty() {
        out.push_str("<h3>Contrast matrix</h3>");
        out.push_str(&matrix_table(
            &report.contrast_matrix.contrasts,
            &report.contrast_matrix.columns,
            &report.contrast_matrix.weights,
        ));
    }

    out.push_str("</section>");
    out
}

/// Render a complete, self-contained HTML report for a set of node outputs.
///
/// The document has no external dependencies: styles are inline, figures are
/// inline SVG, and every figure is paired with a table view so the data is
/// reachable without colour.
#[must_use]
pub fn generate_report(outputs: &[StatsModelsNodeOutput], title: &str) -> String {
    let mut body = String::new();
    if outputs.is_empty() {
        body.push_str("<p class=\"empty\">No node outputs to report.</p>");
    }
    for output in outputs {
        body.push_str(&node_report_html(&node_report(output)));
    }

    format!(
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <title>{t}</title><style>{STYLE}</style></head>\
         <body class=\"bidsrs-report\"><h1>{t}</h1>\
         <p class=\"sub\">BIDS Stats Model report · {n} node output(s)</p>{body}</body></html>\n",
        t = esc(title),
        n = outputs.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bids_core::entities::StringEntities;

    fn contrast(name: &str, conditions: &[&str], weights: &[f64]) -> ContrastInfo {
        ContrastInfo {
            name: name.into(),
            conditions: conditions.iter().map(|s| (*s).to_string()).collect(),
            weights: weights.to_vec(),
            test: Some("t".into()),
            entities: StringEntities::new(),
        }
    }

    fn output() -> StatsModelsNodeOutput {
        let names = vec!["intercept".to_string(), "a".into(), "b".into()];
        let n = 24;
        let a: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let b: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 3 == 0))).collect();
        let columns = vec![vec![1.0; n as usize], a, b];
        StatsModelsNodeOutput {
            node_name: "run".into(),
            entities: StringEntities::from([("subject".to_string(), "01".to_string())]),
            x_variables: names.clone(),
            contrasts: vec![
                contrast("a_gt_b", &["a", "b"], &[1.0, -1.0]),
                contrast("a", &["a"], &[1.0]),
            ],
            design_matrix: Some((names, columns)),
            transform_errors: Vec::new(),
        }
    }

    #[test]
    fn contrast_matrix_scatters_weights_onto_columns() {
        let cols = vec!["intercept".to_string(), "a".into(), "b".into()];
        let m = generate_contrast_matrix(&[contrast("a_gt_b", &["a", "b"], &[1.0, -1.0])], &cols);
        assert_eq!(m.row("a_gt_b").unwrap(), &[0.0, 1.0, -1.0]);
    }

    #[test]
    fn est_vif_skips_the_intercept() {
        let o = output();
        let (names, columns) = o.design_matrix.as_ref().unwrap();
        let vifs = est_vif(names, columns);
        assert_eq!(vifs.len(), 2);
        assert!(vifs.iter().all(|(n, _)| n != "intercept"));
        assert!(vifs.iter().all(|(_, v)| *v >= 1.0 - 1e-9));
    }

    #[test]
    fn eff_reg_vif_is_computed_for_a_contrast() {
        let o = output();
        let (names, columns) = o.design_matrix.as_ref().unwrap();
        let m = generate_contrast_matrix(&o.contrasts, names);
        let vif = get_eff_reg_vif(columns, m.row("a_gt_b").unwrap()).unwrap();
        assert!(vif.is_finite() && vif >= 1.0 - 1e-6, "got {vif}");
    }

    #[test]
    fn eff_reg_vif_rejects_an_all_zero_contrast() {
        let o = output();
        let (_, columns) = o.design_matrix.as_ref().unwrap();
        assert!(get_eff_reg_vif(columns, &[0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn contrast_vif_matches_regressor_vif_for_a_single_regressor_contrast() {
        // A contrast that loads on exactly one regressor reduces to that
        // regressor's own VIF.
        let o = output();
        let (names, columns) = o.design_matrix.as_ref().unwrap();
        let m = generate_contrast_matrix(&o.contrasts, names);
        let con_vif = get_eff_reg_vif(columns, m.row("a").unwrap()).unwrap();
        let reg_vif = est_vif(names, columns)
            .into_iter()
            .find(|(n, _)| n == "a")
            .unwrap()
            .1;
        assert!(
            (con_vif - reg_vif).abs() < 1e-6,
            "contrast VIF {con_vif} != regressor VIF {reg_vif}"
        );
    }

    #[test]
    fn all_contrast_vif_covers_every_contrast() {
        let vifs = get_all_contrast_vif(&output());
        assert_eq!(vifs.len(), 2);
    }

    #[test]
    fn pinv_of_a_symmetric_matrix_round_trips() {
        let m = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let inv = pinv_sym(&m);
        // m * inv should be the identity.
        for (i, row) in m.iter().enumerate() {
            for j in 0..row.len() {
                let v: f64 = row.iter().zip(&inv).map(|(mv, icol)| mv * icol[j]).sum();
                let want = f64::from(u8::from(i == j));
                assert!((v - want).abs() < 1e-9, "({i},{j}) = {v}");
            }
        }
    }

    #[test]
    fn null_space_of_a_row_is_orthogonal_and_complete() {
        let row = [1.0, -1.0, 2.0];
        let basis = null_space_of_row(&row);
        assert_eq!(basis.len(), 2);
        for col in &basis {
            let dot: f64 = col.iter().zip(&row).map(|(a, b)| a * b).sum();
            assert!(dot.abs() < 1e-9, "basis vector not orthogonal to the row");
        }
        let cross: f64 = basis[0].iter().zip(&basis[1]).map(|(a, b)| a * b).sum();
        assert!(cross.abs() < 1e-9, "basis vectors not mutually orthogonal");
    }

    #[test]
    fn node_report_collects_diagnostics() {
        let r = node_report(&output());
        assert_eq!(r.n_obs, 24);
        assert_eq!(r.n_predictors, 3);
        assert_eq!(r.vif.len(), 2);
        assert_eq!(r.contrast_vif.len(), 2);
        assert!(r.max_vif().is_some());
    }

    #[test]
    fn html_report_is_self_contained_and_escaped() {
        let mut o = output();
        o.node_name = "run <script>".into();
        let html = generate_report(&[o], "Model & report");
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("<svg"));
        assert!(html.contains("Model &amp; report"));
        assert!(!html.contains("<script>"), "node name was not escaped");
        // No external resources: the only URL is the SVG namespace.
        assert!(!html.contains("src="), "report loads an external resource");
        assert!(!html.contains("href="), "report links out");
        assert!(!html.contains("@import"), "stylesheet imports");
        assert!(!html.contains("<img"), "report embeds an external image");
        // Every figure section is reachable as a table too.
        assert!(html.matches("<details>").count() >= 3);
        assert!(html.contains("<table>"));
    }

    #[test]
    fn html_report_handles_no_outputs() {
        let html = generate_report(&[], "Empty");
        assert!(html.contains("No node outputs"));
    }

    #[test]
    fn figures_render_for_signed_and_unsigned_matrices() {
        let names = vec!["a".to_string(), "b".into()];
        let signed = vec![vec![-1.0, 0.0, 1.0], vec![1.0, 0.0, -1.0]];
        let unsigned = vec![vec![0.0, 1.0, 2.0], vec![2.0, 1.0, 0.0]];
        let opts = FigureOptions::default();
        // The design matrix is per-regressor normalised, so it is always
        // sequential; the correlation matrix is always diverging.
        assert!(plot_design_matrix(&names, &signed, &opts).contains("--seq-"));
        assert!(plot_design_matrix(&names, &unsigned, &opts).contains("--seq-"));
        assert!(plot_corr_matrix(&names, &signed).contains("--div-"));
        assert!(plot_vif(&[("a".into(), 1.2), ("b".into(), 9.0)]).contains("<svg"));
        assert!(plot_design_timecourse(&names, &signed).contains("<path"));
    }

    #[test]
    fn design_matrix_scales_each_regressor_independently() {
        // A constant intercept beside a small-amplitude regressor must not
        // flatten the second one to a single colour.
        let names = vec!["intercept".to_string(), "hrf".into()];
        let hrf: Vec<f64> = (0..20).map(|i| f64::from(i % 5) * 0.01).collect();
        let columns = vec![vec![1.0; 20], hrf];
        let svg = plot_design_matrix(&names, &columns, &FigureOptions::default());
        // The hrf column must use more than one ramp step.
        let steps: std::collections::HashSet<&str> = (1..=SEQ_STEPS)
            .map(|i| match i {
                1 => "--seq-1",
                2 => "--seq-2",
                3 => "--seq-3",
                4 => "--seq-4",
                5 => "--seq-5",
                6 => "--seq-6",
                _ => "--seq-7",
            })
            .filter(|v| svg.contains(v))
            .collect();
        assert!(
            steps.len() > 2,
            "expected a spread of ramp steps, got {steps:?}"
        );
    }

    #[test]
    fn figures_are_empty_for_empty_input() {
        let opts = FigureOptions::default();
        assert!(plot_design_matrix(&[], &[], &opts).is_empty());
        assert!(plot_corr_matrix(&[], &[]).is_empty());
        assert!(plot_vif(&[]).is_empty());
        assert!(plot_design_timecourse(&[], &[]).is_empty());
    }
}
