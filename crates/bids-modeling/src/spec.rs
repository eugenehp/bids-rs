//! Statistical model specifications for BIDS-StatsModels.
//!
//! Provides data structures for GLM and meta-analysis model specifications,
//! design matrix construction, VIF computation, and formatted output for
//! inspection and reporting.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single term (predictor column) in a statistical model's design matrix.
///
/// Each term has a name (e.g., `"trial_type.face"`, `"intercept"`) and a
/// vector of numeric values — one per observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Term {
    pub name: String,
    pub values: Vec<f64>,
}

/// General Linear Model (GLM) specification.
///
/// Contains the design matrix terms and the distributional family/link
/// function. Can be constructed from raw data rows and a model specification,
/// and provides methods for extracting the design matrix, computing
/// dimensions, and adding random effects (placeholder for GLMM).
#[derive(Debug, Clone)]
pub struct GlmSpec {
    pub terms: Vec<Term>,
    pub family: String,
    pub link: String,
}

impl GlmSpec {
    pub fn new(terms: Vec<Term>) -> Self {
        Self { terms, family: "gaussian".into(), link: "identity".into() }
    }

    /// Build from data rows and a model specification dict.
    pub fn from_rows(
        data: &[HashMap<String, f64>],
        x_names: &[String],
        model: &serde_json::Value,
    ) -> Self {
        let family = model.get("family")
            .and_then(|v| v.as_str())
            .unwrap_or("gaussian").to_string();
        let link = model.get("link")
            .and_then(|v| v.as_str())
            .unwrap_or("identity").to_string();

        let mut terms = Vec::new();
        for name in x_names {
            if name == "intercept" || name == "1" {
                terms.push(Term { name: "intercept".into(), values: vec![1.0; data.len()] });
            } else {
                let values: Vec<f64> = data.iter()
                    .map(|row| row.get(name).copied().unwrap_or(0.0))
                    .collect();
                terms.push(Term { name: name.clone(), values });
            }
        }

        Self { terms, family, link }
    }

    /// Design matrix as (column_names, column_data).
    pub fn design_matrix(&self) -> (Vec<String>, Vec<Vec<f64>>) {
        let names: Vec<String> = self.terms.iter().map(|t| t.name.clone()).collect();
        let columns: Vec<Vec<f64>> = self.terms.iter().map(|t| t.values.clone()).collect();
        (names, columns)
    }

    pub fn n_obs(&self) -> usize { self.terms.first().map_or(0, |t| t.values.len()) }
    pub fn n_predictors(&self) -> usize { self.terms.len() }

    /// Get the design matrix X as column vectors.
    pub fn x(&self) -> &[Term] { &self.terms }

    /// Add random effects (Z matrix) — placeholder for GLMM.
    pub fn with_random_effects(self, _z_terms: Vec<Term>) -> Self {
        // Random effects support is a placeholder
        self
    }
}

/// Meta-analysis specification.
#[derive(Debug, Clone)]
pub struct MetaAnalysisSpec {
    pub terms: Vec<Term>,
}

impl MetaAnalysisSpec {
    pub fn new(terms: Vec<Term>) -> Self { Self { terms } }

    pub fn from_rows(data: &[HashMap<String, f64>], x_names: &[String]) -> Self {
        let terms: Vec<Term> = x_names.iter().map(|name| {
            let values: Vec<f64> = data.iter()
                .map(|row| row.get(name).copied().unwrap_or(0.0))
                .collect();
            Term { name: name.clone(), values }
        }).collect();
        Self { terms }
    }
}

/// Convert dummy-coded columns to a weight vector for a contrast.
#[must_use]
pub fn dummies_to_vec(condition_list: &[String], all_columns: &[String], weights: &[f64]) -> Vec<f64> {
    let mut vec = vec![0.0; all_columns.len()];
    for (cond, &w) in condition_list.iter().zip(weights) {
        if let Some(idx) = all_columns.iter().position(|c| c == cond) {
            vec[idx] = w;
        }
    }
    vec
}

/// Compute Variance Inflation Factor for each column.
#[must_use]
pub fn compute_vif(columns: &[Vec<f64>]) -> Vec<f64> {
    let n_cols = columns.len();
    if n_cols < 2 { return vec![1.0; n_cols]; }
    let n_rows = columns.first().map_or(0, std::vec::Vec::len);
    if n_rows < 2 { return vec![1.0; n_cols]; }

    // Simple VIF: for each predictor, regress on all others, VIF = 1/(1-R²)
    (0..n_cols).map(|i| {
        let y = &columns[i];
        let x_others: Vec<&Vec<f64>> = columns.iter().enumerate()
            .filter(|(j, _)| *j != i).map(|(_, c)| c).collect();

        // Simple: compute R² using correlation-based approximation
        let y_mean: f64 = y.iter().sum::<f64>() / n_rows as f64;
        let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
        if ss_tot < 1e-15 { return 1.0; }

        // Predicted = mean of correlations * other vars (simplified)
        let mut ss_res = ss_tot;
        for other in &x_others {
            let o_mean: f64 = other.iter().sum::<f64>() / n_rows as f64;
            let cov: f64 = y.iter().zip(other.iter())
                .map(|(a, b)| (a - y_mean) * (b - o_mean)).sum::<f64>() / n_rows as f64;
            let o_var: f64 = other.iter().map(|v| (v - o_mean).powi(2)).sum::<f64>() / n_rows as f64;
            if o_var > 1e-15 {
                let r = cov / (ss_tot / n_rows as f64).sqrt() / o_var.sqrt();
                ss_res -= r.powi(2) * ss_tot;
            }
        }
        let r_sq = 1.0 - ss_res / ss_tot;
        if r_sq >= 1.0 { return f64::INFINITY; }
        1.0 / (1.0 - r_sq)
    }).collect()
}

/// Format a design matrix as an aligned text table.
#[must_use]
pub fn format_design_matrix(names: &[String], columns: &[Vec<f64>], max_rows: usize) -> String {
    let n_rows = columns.first().map_or(0, std::vec::Vec::len);
    let show = n_rows.min(max_rows);
    let mut lines = Vec::new();

    // Header
    let header: String = names.iter().map(|n| format!("{:>10}", &n[..n.len().min(10)])).collect::<Vec<_>>().join(" ");
    lines.push(header);
    lines.push("-".repeat(names.len() * 11));

    for i in 0..show {
        let row: String = columns.iter().map(|col| {
            let v = col.get(i).copied().unwrap_or(0.0);
            if v == v.round() && v.abs() < 1000.0 { format!("{v:>10.0}") }
            else { format!("{v:>10.3}") }
        }).collect::<Vec<_>>().join(" ");
        lines.push(row);
    }

    if n_rows > max_rows {
        lines.push(format!("... ({} more rows)", n_rows - max_rows));
    }
    lines.join("\n")
}

/// Format a correlation matrix as text with Unicode intensity blocks.
#[must_use]
pub fn format_correlation_matrix(names: &[String], columns: &[Vec<f64>]) -> String {
    let n = columns.len();
    let mut corr = vec![vec![0.0f64; n]; n];

    for i in 0..n {
        for j in 0..n {
            corr[i][j] = pearson_r(&columns[i], &columns[j]);
        }
    }

    let blocks = [' ', '░', '▒', '▓', '█'];
    let mut lines = Vec::new();

    // Header
    let header: String = std::iter::once(format!("{:>10}", ""))
        .chain(names.iter().map(|n| format!("{:>5}", &n[..n.len().min(5)])))
        .collect::<Vec<_>>().join("");
    lines.push(header);

    for i in 0..n {
        let row: String = std::iter::once(format!("{:>10}", &names[i][..names[i].len().min(10)]))
            .chain((0..n).map(|j| {
                let r = corr[i][j].abs();
                let idx = (r * 4.0).round().min(4.0) as usize;
                format!("  {:>1}  ", blocks[idx])
            }))
            .collect::<Vec<_>>().join("");
        lines.push(row);
    }
    lines.join("\n")
}

fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let mx: f64 = x.iter().take(n).sum::<f64>() / n as f64;
    let my: f64 = y.iter().take(n).sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-15 { 0.0 } else { num / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummies_to_vec() {
        let cols = vec!["a".into(), "b".into(), "c".into()];
        let conds = vec!["a".into(), "c".into()];
        let weights = vec![1.0, -1.0];
        let result = dummies_to_vec(&conds, &cols, &weights);
        assert_eq!(result, vec![1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_compute_vif() {
        // Independent columns should have VIF near 1
        let c1: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let c2: Vec<f64> = (0..100).map(|i| (i * 7 % 13) as f64).collect();
        let vifs = compute_vif(&[c1, c2]);
        assert!(vifs[0] < 5.0, "VIF should be low for independent vars, got {}", vifs[0]);
    }
}
