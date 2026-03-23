//! Common Spatial Patterns (CSP) for motor imagery BCI.
//!
//! CSP finds spatial filters that maximize the variance ratio between two
//! classes. It's the standard baseline feature extraction for motor imagery
//! BCI systems, typically paired with LDA or SVM classification.
//!
//! This is a pure-Rust implementation — no BLAS/LAPACK dependency.
//!
//! # Algorithm
//!
//! 1. Compute per-class average covariance matrices C₁, C₂
//! 2. Solve the generalized eigenvalue problem: C₁ w = λ (C₁ + C₂) w
//! 3. Select the m eigenvectors with largest/smallest eigenvalues as spatial filters
//! 4. Project data: Z = W^T X, features = log(var(Z))
//!
//! # Example
//!
//! ```no_run
//! use bids_eeg::csp::CSP;
//!
//! let mut csp = CSP::new(3); // 3 pairs = 6 components
//! // csp.fit(&epochs_class1, &epochs_class2);
//! // let features = csp.transform(&test_epoch);
//! ```

/// Common Spatial Patterns filter.
#[derive(Debug, Clone)]
pub struct CSP {
    /// Number of spatial filter pairs (2*n_components total filters).
    n_components: usize,
    /// Spatial filter matrix W: each row is a spatial filter.
    /// Shape: `[2*n_components][n_channels]`.
    filters: Option<Vec<Vec<f64>>>,
    /// Eigenvalues corresponding to the filters.
    eigenvalues: Vec<f64>,
}

impl CSP {
    /// Create a new CSP with `n_components` pairs of spatial filters.
    ///
    /// The total number of output features will be `2 * n_components`.
    #[must_use]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            filters: None,
            eigenvalues: Vec::new(),
        }
    }

    /// Fit CSP filters from two classes of epoch data.
    ///
    /// Each epoch is `[n_channels][n_samples]`. All epochs must have the
    /// same number of channels.
    pub fn fit(&mut self, class1: &[Vec<Vec<f64>>], class2: &[Vec<Vec<f64>>]) {
        if class1.is_empty() || class2.is_empty() {
            return;
        }

        let n_ch = class1[0].len();

        // Compute mean covariance for each class
        let cov1 = mean_covariance(class1, n_ch);
        let cov2 = mean_covariance(class2, n_ch);

        // Composite covariance
        let mut cov_sum = vec![0.0; n_ch * n_ch];
        for i in 0..cov_sum.len() {
            cov_sum[i] = cov1[i] + cov2[i];
        }

        // Solve generalized eigenvalue problem via whitening:
        // 1. Eigendecompose cov_sum = U D U^T
        // 2. Whitening: P = D^(-1/2) U^T
        // 3. S = P C₁ P^T
        // 4. Eigendecompose S to get the CSP filters
        let (eig_vals_sum, eig_vecs_sum) = symmetric_eigen(n_ch, &cov_sum);

        // Whitening matrix P: D^(-1/2) * U^T
        let mut p = vec![0.0; n_ch * n_ch];
        for i in 0..n_ch {
            let d = eig_vals_sum[i];
            let scale = if d > 1e-12 { 1.0 / d.sqrt() } else { 0.0 };
            for j in 0..n_ch {
                p[i * n_ch + j] = eig_vecs_sum[j * n_ch + i] * scale; // U^T scaled
            }
        }

        // S = P * C₁ * P^T
        let pc1 = mat_mul(n_ch, &p, &cov1);
        let p_t = transpose(n_ch, &p);
        let s = mat_mul(n_ch, &pc1, &p_t);

        // Eigendecompose S
        let (eig_vals_s, eig_vecs_s) = symmetric_eigen(n_ch, &s);

        // Sort eigenvalues (descending)
        let mut indices: Vec<usize> = (0..n_ch).collect();
        indices.sort_by(|&a, &b| {
            eig_vals_s[b]
                .partial_cmp(&eig_vals_s[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select top and bottom n_components
        let n = self.n_components.min(n_ch / 2);
        let selected: Vec<usize> = indices[..n]
            .iter()
            .chain(indices[n_ch - n..].iter())
            .copied()
            .collect();

        // Compute spatial filters: W = eigvecs_S^T * P
        let mut filters = Vec::with_capacity(selected.len());
        let mut eigenvalues = Vec::with_capacity(selected.len());

        for &idx in &selected {
            let mut w = vec![0.0; n_ch];
            for j in 0..n_ch {
                let mut sum = 0.0;
                for k in 0..n_ch {
                    sum += eig_vecs_s[k * n_ch + idx] * p[k * n_ch + j];
                }
                w[j] = sum;
            }
            // Normalize
            let norm: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for v in &mut w {
                    *v /= norm;
                }
            }
            filters.push(w);
            eigenvalues.push(eig_vals_s[idx]);
        }

        self.filters = Some(filters);
        self.eigenvalues = eigenvalues;
    }

    /// Transform a single epoch into CSP features.
    ///
    /// Returns `log(var(W^T X))` for each spatial filter — one value per component.
    /// The epoch shape is `[n_channels][n_samples]`.
    #[must_use]
    pub fn transform(&self, epoch: &[Vec<f64>]) -> Vec<f64> {
        let filters = match &self.filters {
            Some(f) => f,
            None => return Vec::new(),
        };

        let n_ch = epoch.len();
        let n_s = epoch.first().map_or(0, |ch| ch.len());

        filters
            .iter()
            .map(|w| {
                // Project: z[t] = sum_c w[c] * x[c][t]
                let nc = n_ch.min(w.len());
                let projected: Vec<f64> = (0..n_s)
                    .map(|t| (0..nc).map(|c| w[c] * epoch[c][t]).sum::<f64>())
                    .collect();

                let mean = if n_s > 0 {
                    projected.iter().sum::<f64>() / n_s as f64
                } else {
                    0.0
                };
                let var = if n_s > 1 {
                    projected.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / (n_s - 1) as f64
                } else {
                    0.0
                };
                if var > 0.0 {
                    var.ln()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect()
    }

    /// Transform multiple epochs into a feature matrix.
    ///
    /// Returns `[n_epochs][2*n_components]`.
    #[must_use]
    pub fn transform_all(&self, epochs: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
        epochs.iter().map(|e| self.transform(e)).collect()
    }

    /// Number of output features (2 × n_components).
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.filters.as_ref().map_or(0, |f| f.len())
    }

    /// Whether the CSP has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.filters.is_some()
    }

    /// Get the eigenvalues.
    #[must_use]
    pub fn eigenvalues(&self) -> &[f64] {
        &self.eigenvalues
    }
}

// ─── Linear algebra helpers (no BLAS needed) ───────────────────────────────────

/// Compute mean covariance matrix across epochs.
fn mean_covariance(epochs: &[Vec<Vec<f64>>], n_ch: usize) -> Vec<f64> {
    let mut cov = vec![0.0; n_ch * n_ch];
    let n_epochs = epochs.len() as f64;

    for epoch in epochs {
        let nc = epoch.len().min(n_ch);
        let ns = epoch.first().map_or(0, |ch| ch.len());
        if ns < 2 {
            continue;
        }

        // Compute means
        let means: Vec<f64> = (0..nc)
            .map(|c| epoch[c].iter().sum::<f64>() / ns as f64)
            .collect();

        // Accumulate covariance
        for i in 0..nc {
            for j in i..nc {
                let sum: f64 = (0..ns)
                    .map(|t| (epoch[i][t] - means[i]) * (epoch[j][t] - means[j]))
                    .sum();
                let val = sum / (ns - 1) as f64;
                cov[i * n_ch + j] += val / n_epochs;
                if i != j {
                    cov[j * n_ch + i] += val / n_epochs;
                }
            }
        }
    }

    // Normalize by trace
    let trace: f64 = (0..n_ch).map(|i| cov[i * n_ch + i]).sum();
    if trace > 1e-12 {
        for v in &mut cov {
            *v /= trace;
        }
    }

    cov
}

/// Simple symmetric eigendecomposition via Jacobi iteration.
///
/// Returns (eigenvalues, eigenvectors_column_major).
/// Good enough for small matrices (n_channels typically < 128).
fn symmetric_eigen(n: usize, a: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut d = a.to_vec(); // working copy
    let mut v = vec![0.0; n * n]; // eigenvectors (identity initially)
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in i + 1..n {
                let val = d[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-14 {
            break;
        }

        // Compute rotation
        let app = d[p * n + p];
        let aqq = d[q * n + q];
        let apq = d[p * n + q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation to d
        let mut new_d = d.clone();
        new_d[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
        new_d[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
        new_d[p * n + q] = 0.0;
        new_d[q * n + p] = 0.0;

        for i in 0..n {
            if i != p && i != q {
                let dip = c * d[i * n + p] + s * d[i * n + q];
                let diq = -s * d[i * n + p] + c * d[i * n + q];
                new_d[i * n + p] = dip;
                new_d[p * n + i] = dip;
                new_d[i * n + q] = diq;
                new_d[q * n + i] = diq;
            }
        }
        d = new_d;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| d[i * n + i]).collect();
    (eigenvalues, v)
}

/// Matrix multiply: C = A * B (all n×n, row-major).
fn mat_mul(n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik.abs() < 1e-15 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

/// Transpose n×n matrix.
fn transpose(n: usize, a: &[f64]) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_epochs(n_epochs: usize, n_ch: usize, n_s: usize, class: usize) -> Vec<Vec<Vec<f64>>> {
        // Generate deterministic pseudo-random data with class-dependent variance
        (0..n_epochs)
            .map(|epoch_idx| {
                (0..n_ch)
                    .map(|ch| {
                        (0..n_s)
                            .map(|s| {
                                let seed = (epoch_idx * 1000 + ch * 100 + s + class * 50000) as f64;
                                let val =
                                    (seed * 0.1).sin() * (1.0 + class as f64 * ch as f64 * 0.5);
                                val
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_csp_fit_transform() {
        let class1 = make_epochs(20, 4, 100, 0);
        let class2 = make_epochs(20, 4, 100, 1);

        let mut csp = CSP::new(2);
        csp.fit(&class1, &class2);

        assert!(csp.is_fitted());
        assert_eq!(csp.n_features(), 4); // 2 pairs

        let features = csp.transform(&class1[0]);
        assert_eq!(features.len(), 4);
        // Features should be finite
        assert!(features.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn test_csp_transform_all() {
        let class1 = make_epochs(10, 3, 50, 0);
        let class2 = make_epochs(10, 3, 50, 1);

        let mut csp = CSP::new(1);
        csp.fit(&class1, &class2);

        let features1 = csp.transform_all(&class1);
        let features2 = csp.transform_all(&class2);

        assert_eq!(features1.len(), 10);
        assert_eq!(features2.len(), 10);
        assert_eq!(features1[0].len(), 2); // 1 pair = 2 features

        // CSP should separate the classes: mean features should differ
        let mean1: f64 = features1.iter().map(|f| f[0]).sum::<f64>() / 10.0;
        let mean2: f64 = features2.iter().map(|f| f[0]).sum::<f64>() / 10.0;
        assert!(
            (mean1 - mean2).abs() > 1e-6,
            "CSP should separate classes: {mean1} vs {mean2}"
        );
    }

    #[test]
    fn test_csp_not_fitted() {
        let csp = CSP::new(2);
        assert!(!csp.is_fitted());
        assert_eq!(csp.transform(&[vec![1.0, 2.0]]).len(), 0);
    }

    #[test]
    fn test_symmetric_eigen() {
        // 2x2 symmetric matrix
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let (vals, _vecs) = symmetric_eigen(2, &a);
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Eigenvalues of [[2,1],[1,3]] are (5±√5)/2 ≈ 1.382, 3.618
        assert!((sorted[0] - 1.382).abs() < 0.01);
        assert!((sorted[1] - 3.618).abs() < 0.01);
    }
}
