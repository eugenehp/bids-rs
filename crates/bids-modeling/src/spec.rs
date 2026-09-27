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
/// vector of numeric values — one per observation. A term may be flagged as
/// categorical and may carry a [`Prior`] for Bayesian estimators.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Term {
    pub name: String,
    pub values: Vec<f64>,
    /// Whether the term encodes a categorical variable.
    #[serde(default)]
    pub categorical: bool,
    /// Optional prior distribution for this term.
    #[serde(default)]
    pub prior: Option<Prior>,
}

impl Term {
    #[must_use]
    pub fn new(name: &str, values: Vec<f64>) -> Self {
        Self {
            name: name.to_string(),
            values,
            categorical: false,
            prior: None,
        }
    }

    /// Mark this term as categorical.
    #[must_use]
    pub fn categorical(mut self) -> Self {
        self.categorical = true;
        self
    }

    /// Attach a prior distribution.
    #[must_use]
    pub fn with_prior(mut self, prior: Prior) -> Self {
        self.prior = Some(prior);
        self
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Abstract specification of a prior distribution on a term.
///
/// There is no controlled vocabulary for distribution names; the convention
/// PyBIDS suggests is to use PyMC names (`"Normal"` parameterised with `mu`
/// and `sd`, and so on).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prior {
    /// Distribution name, e.g. `"Normal"`.
    pub name: String,
    /// Distribution parameters, e.g. `{"mu": 0, "sd": 1}`.
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Prior {
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            parameters: HashMap::new(),
        }
    }

    /// Set one distribution parameter.
    #[must_use]
    pub fn param(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.parameters.insert(key.to_string(), value.into());
        self
    }
}

/// A variance component: the random-effect design block for one grouping factor.
///
/// `columns` is the n_obs x n_levels binary indicator matrix (column-major)
/// identifying which observations belong to which level of the factor.
///
/// Corresponds to PyBIDS' `VarComp` class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarComp {
    pub name: String,
    /// Column-major indicator matrix: `columns[level][observation]`.
    pub columns: Vec<Vec<f64>>,
    /// One name per column of `columns`.
    pub level_names: Vec<String>,
    #[serde(default)]
    pub prior: Option<Prior>,
    /// 1-based level index per observation; 0 where no level applies.
    ///
    /// Estimation backends index into the coefficient vector with this rather
    /// than multiplying by the full indicator matrix.
    pub index_vec: Vec<usize>,
}

impl VarComp {
    /// Build from an explicit indicator matrix.
    #[must_use]
    pub fn new(name: &str, columns: Vec<Vec<f64>>, level_names: Vec<String>) -> Self {
        let index_vec = Self::dummies_to_index(&columns);
        Self {
            name: name.to_string(),
            columns,
            level_names,
            prior: None,
            index_vec,
        }
    }

    /// Dummy-code a categorical factor into a variance component.
    ///
    /// Levels are sorted, matching `pandas.get_dummies`.
    #[must_use]
    pub fn from_levels(name: &str, factor: &[String]) -> Self {
        let mut levels: Vec<String> = factor.to_vec();
        levels.sort();
        levels.dedup();
        let columns: Vec<Vec<f64>> = levels
            .iter()
            .map(|lev| {
                factor
                    .iter()
                    .map(|v| if v == lev { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        Self::new(name, columns, levels)
    }

    /// Collapse an indicator matrix to a vector of 1-based level indices.
    ///
    /// Corresponds to PyBIDS' `VarComp.dummies_to_vec`.
    #[must_use]
    pub fn dummies_to_index(columns: &[Vec<f64>]) -> Vec<usize> {
        let n_obs = columns.first().map_or(0, Vec::len);
        let mut vec = vec![0usize; n_obs];
        for (i, col) in columns.iter().enumerate() {
            for (row, v) in col.iter().enumerate() {
                if (*v - 1.0).abs() < f64::EPSILON {
                    vec[row] = i + 1;
                }
            }
        }
        vec
    }

    #[must_use]
    pub fn n_levels(&self) -> usize {
        self.columns.len()
    }

    #[must_use]
    pub fn n_obs(&self) -> usize {
        self.columns.first().map_or(0, Vec::len)
    }

    /// Attach a prior distribution.
    #[must_use]
    pub fn with_prior(mut self, prior: Prior) -> Self {
        self.prior = Some(prior);
        self
    }
}

/// Generalized Linear Mixed Model (GLMM) specification.
///
/// Holds the fixed-effect design (`X`) as a list of [`Term`]s, the
/// random-effect design (`Z`) as a list of [`VarComp`]s, the distributional
/// family and link function, and optional priors.
///
/// With no variance components this is an ordinary GLM, which is why the type
/// is named `GlmSpec`; [`GlmmSpec`] is an alias for the same type and matches
/// PyBIDS' `GLMMSpec`.
#[derive(Debug, Clone)]
pub struct GlmSpec {
    /// Fixed-effect terms, in design-matrix column order.
    pub terms: Vec<Term>,
    /// Random-effect variance components, in Z-block order.
    pub variance_components: Vec<VarComp>,
    pub family: String,
    pub link: String,
    /// Optional covariance matrix between variance components (v x v).
    pub sigma: Option<Vec<Vec<f64>>>,
    /// Default priors, keyed by `"fixed"` and `"random"`.
    pub priors: HashMap<String, Prior>,
}

/// PyBIDS-compatible alias for [`GlmSpec`].
pub type GlmmSpec = GlmSpec;

impl GlmSpec {
    #[must_use]
    pub fn new(terms: Vec<Term>) -> Self {
        Self {
            terms,
            variance_components: Vec::new(),
            family: "gaussian".into(),
            link: "identity".into(),
            sigma: None,
            priors: HashMap::new(),
        }
    }

    /// Build from data rows and a model specification dict.
    pub fn from_rows(
        data: &[HashMap<String, f64>],
        x_names: &[String],
        model: &serde_json::Value,
    ) -> Self {
        let family = model
            .get("family")
            .and_then(|v| v.as_str())
            .unwrap_or("gaussian")
            .to_string();
        let link = model
            .get("link")
            .and_then(|v| v.as_str())
            .unwrap_or("identity")
            .to_string();

        let mut terms = Vec::new();
        for name in x_names {
            if name == "intercept" || name == "1" {
                terms.push(Term::new("intercept", vec![1.0; data.len()]));
            } else {
                let values: Vec<f64> = data
                    .iter()
                    .map(|row| row.get(name).copied().unwrap_or(0.0))
                    .collect();
                terms.push(Term::new(name, values));
            }
        }

        let mut spec = Self::new(terms);
        spec.family = family;
        spec.link = link;
        spec.build_variance_components_from_model(data, model);
        spec
    }

    /// Design matrix as (column_names, column_data).
    pub fn design_matrix(&self) -> (Vec<String>, Vec<Vec<f64>>) {
        let names: Vec<String> = self.terms.iter().map(|t| t.name.clone()).collect();
        let columns: Vec<Vec<f64>> = self.terms.iter().map(|t| t.values.clone()).collect();
        (names, columns)
    }

    pub fn n_obs(&self) -> usize {
        self.terms.first().map_or(0, |t| t.values.len())
    }
    pub fn n_predictors(&self) -> usize {
        self.terms.len()
    }

    /// Get the design matrix X as column vectors.
    pub fn x(&self) -> &[Term] {
        &self.terms
    }

    /// Add random effects as a single variance component over `z_terms`.
    ///
    /// Each term becomes one column of Z, and all of them share one variance.
    /// For several independent grouping factors, call
    /// [`add_variance_component()`](Self::add_variance_component) once per
    /// factor instead.
    #[must_use]
    pub fn with_random_effects(mut self, z_terms: Vec<Term>) -> Self {
        if z_terms.is_empty() {
            return self;
        }
        let level_names: Vec<String> = z_terms.iter().map(|t| t.name.clone()).collect();
        let columns: Vec<Vec<f64>> = z_terms.into_iter().map(|t| t.values).collect();
        let vc = VarComp::new("VC0", columns, level_names);
        self.variance_components.push(vc);
        self
    }

    /// Add a fixed-effect term.
    ///
    /// # Errors
    ///
    /// Returns an error if a term or variance component of the same name
    /// already exists.
    pub fn add_term(&mut self, term: Term) -> Result<(), String> {
        if self.has_name(&term.name) {
            return Err(format!("term with name '{}' already exists", term.name));
        }
        self.terms.push(term);
        Ok(())
    }

    /// Add a variance component (one block of Z).
    ///
    /// # Errors
    ///
    /// Returns an error if a term or variance component of the same name
    /// already exists.
    pub fn add_variance_component(&mut self, vc: VarComp) -> Result<(), String> {
        if self.has_name(&vc.name) {
            return Err(format!(
                "variance component with name '{}' already exists",
                vc.name
            ));
        }
        self.variance_components.push(vc);
        Ok(())
    }

    fn has_name(&self, name: &str) -> bool {
        self.terms.iter().any(|t| t.name == name)
            || self.variance_components.iter().any(|v| v.name == name)
    }

    /// Build fixed terms from named columns, one [`Term`] per column.
    pub fn build_fixed_terms(&mut self, names: &[String], columns: &[Vec<f64>]) {
        for (name, values) in names.iter().zip(columns) {
            let _ = self.add_term(Term::new(name, values.clone()));
        }
    }

    /// Build variance components from a Z matrix and a group-membership map.
    ///
    /// `z` is column-major (`z[column][observation]`). `groups` is a
    /// `n_z_columns x n_components` binary map saying which Z columns belong
    /// to which variance component; passing `None` puts every column into one
    /// component. `names` supplies component names, defaulting to `VC0`,
    /// `VC1`, ….
    ///
    /// # Errors
    ///
    /// Returns an error if `groups` does not have one row per column of `z`.
    pub fn build_variance_components(
        &mut self,
        z: &[Vec<f64>],
        groups: Option<&[Vec<f64>]>,
        names: Option<&[String]>,
    ) -> Result<(), String> {
        if z.is_empty() {
            return Ok(());
        }
        let single = vec![vec![1.0]; z.len()];
        let groups = groups.unwrap_or(&single);
        if groups.len() != z.len() {
            return Err(format!(
                "'groups' has {} rows but Z has {} columns",
                groups.len(),
                z.len()
            ));
        }
        let n_components = groups.first().map_or(0, Vec::len);

        for c in 0..n_components {
            let member_idx: Vec<usize> = (0..z.len())
                .filter(|&k| groups[k].get(c).is_some_and(|v| *v != 0.0))
                .collect();
            if member_idx.is_empty() {
                continue;
            }
            let name = names
                .and_then(|n| n.get(c).cloned())
                .unwrap_or_else(|| format!("VC{c}"));
            let level_names: Vec<String> =
                member_idx.iter().map(|k| format!("{name}.{k}")).collect();
            let columns: Vec<Vec<f64>> = member_idx.iter().map(|&k| z[k].clone()).collect();
            self.add_variance_component(VarComp::new(&name, columns, level_names))?;
        }
        Ok(())
    }

    /// Read `VarianceComponents` out of a BIDS-StatsModels `Model` section.
    ///
    /// Each entry names a component and identifies its levels either with
    /// `LevelsFrom` (one categorical column, dummy-coded) or `Levels` (an
    /// explicit list of binary columns).
    fn build_variance_components_from_model(
        &mut self,
        data: &[HashMap<String, f64>],
        model: &serde_json::Value,
    ) {
        let Some(vcs) = model
            .get("variance_components")
            .or_else(|| model.get("VarianceComponents"))
            .and_then(|v| v.as_array())
        else {
            return;
        };

        for (i, vc) in vcs.iter().enumerate() {
            let name = vc
                .get("name")
                .or_else(|| vc.get("Name"))
                .and_then(|v| v.as_str())
                .map_or_else(|| format!("VC{i}"), String::from);

            let levels_from = vc
                .get("levels_from")
                .or_else(|| vc.get("LevelsFrom"))
                .and_then(|v| v.as_str());

            if let Some(col) = levels_from {
                // Numeric codes stand in for level labels here, since the
                // caller hands us numeric rows.
                let factor: Vec<String> = data
                    .iter()
                    .map(|row| {
                        row.get(col)
                            .map_or_else(String::new, std::string::ToString::to_string)
                    })
                    .collect();
                let _ = self.add_variance_component(VarComp::from_levels(&name, &factor));
                continue;
            }

            let levels: Vec<String> = vc
                .get("levels")
                .or_else(|| vc.get("Levels"))
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            if levels.is_empty() {
                continue;
            }
            let columns: Vec<Vec<f64>> = levels
                .iter()
                .map(|lev| {
                    data.iter()
                        .map(|row| row.get(lev).copied().unwrap_or(0.0))
                        .collect()
                })
                .collect();
            let _ = self.add_variance_component(VarComp::new(&name, columns, levels));
        }
    }

    /// Assign default priors to terms that do not already have one.
    ///
    /// PyBIDS raises `NotImplementedError` here; this applies the supplied
    /// priors and records them under `"fixed"` and `"random"`.
    pub fn set_priors(&mut self, fixed: Option<Prior>, random: Option<Prior>) {
        if let Some(p) = fixed {
            self.priors.insert("fixed".into(), p.clone());
            for term in &mut self.terms {
                if term.prior.is_none() {
                    term.prior = Some(p.clone());
                }
            }
        }
        if let Some(p) = random {
            self.priors.insert("random".into(), p.clone());
            for vc in &mut self.variance_components {
                if vc.prior.is_none() {
                    vc.prior = Some(p.clone());
                }
            }
        }
    }

    /// The fixed-effect terms.
    #[must_use]
    pub fn fixed_terms(&self) -> &[Term] {
        &self.terms
    }

    /// The variance components making up Z.
    #[must_use]
    pub fn variance_components(&self) -> &[VarComp] {
        &self.variance_components
    }

    /// The random-effects design matrix Z as `(column_names, columns)`.
    ///
    /// Columns from every variance component are concatenated in order and
    /// named `<component>.<index>`, matching PyBIDS.
    #[must_use]
    pub fn z(&self) -> Option<(Vec<String>, Vec<Vec<f64>>)> {
        if self.variance_components.is_empty() {
            return None;
        }
        let mut names = Vec::new();
        let mut columns = Vec::new();
        for vc in &self.variance_components {
            for (i, col) in vc.columns.iter().enumerate() {
                names.push(format!("{}.{i}", vc.name));
                columns.push(col.clone());
            }
        }
        Some((names, columns))
    }

    /// Number of random-effect columns across all variance components.
    #[must_use]
    pub fn n_random(&self) -> usize {
        self.variance_components.iter().map(VarComp::n_levels).sum()
    }

    /// Whether this specification has any random effects.
    #[must_use]
    pub fn is_mixed(&self) -> bool {
        !self.variance_components.is_empty()
    }
}

/// Meta-analysis specification.
#[derive(Debug, Clone)]
pub struct MetaAnalysisSpec {
    pub terms: Vec<Term>,
}

impl MetaAnalysisSpec {
    pub fn new(terms: Vec<Term>) -> Self {
        Self { terms }
    }

    pub fn from_rows(data: &[HashMap<String, f64>], x_names: &[String]) -> Self {
        let terms: Vec<Term> = x_names
            .iter()
            .map(|name| {
                let values: Vec<f64> = data
                    .iter()
                    .map(|row| row.get(name).copied().unwrap_or(0.0))
                    .collect();
                Term::new(name, values)
            })
            .collect();
        Self { terms }
    }

    /// View this meta-analysis as a GLMM specification.
    ///
    /// PyBIDS models `MetaAnalysisSpec` as a subclass of `GLMMSpec`; this is
    /// the Rust equivalent of that upcast.
    #[must_use]
    pub fn to_glmm(&self) -> GlmSpec {
        GlmSpec::new(self.terms.clone())
    }
}

/// Convert dummy-coded columns to a weight vector for a contrast.
#[must_use]
pub fn dummies_to_vec(
    condition_list: &[String],
    all_columns: &[String],
    weights: &[f64],
) -> Vec<f64> {
    let mut vec = vec![0.0; all_columns.len()];
    for (cond, &w) in condition_list.iter().zip(weights) {
        if let Some(idx) = all_columns.iter().position(|c| c == cond) {
            vec[idx] = w;
        }
    }
    vec
}

/// Variance Inflation Factor for each column of a design matrix.
///
/// For column *i*, VIF is `1 / (1 - R²)` where R² comes from regressing
/// column *i* on an intercept plus every other column. Columns that are
/// perfectly predicted by the others return infinity.
///
/// `columns` is column-major: `columns[predictor][observation]`.
#[must_use]
pub fn compute_vif(columns: &[Vec<f64>]) -> Vec<f64> {
    let n_cols = columns.len();
    if n_cols < 2 {
        return vec![1.0; n_cols];
    }
    let n_rows = columns.first().map_or(0, std::vec::Vec::len);
    if n_rows < 3 {
        return vec![1.0; n_cols];
    }

    // Constant columns are absorbed by the intercept we add below. Leaving one
    // in the predictor set would make every other regression rank-deficient and
    // report infinite VIF for the whole design.
    let is_constant = |c: &Vec<f64>| {
        let mean: f64 = c.iter().sum::<f64>() / n_rows as f64;
        c.iter().map(|v| (v - mean).powi(2)).sum::<f64>() < 1e-15
    };

    (0..n_cols)
        .map(|i| {
            let y = &columns[i];
            let y_mean: f64 = y.iter().sum::<f64>() / n_rows as f64;
            let ss_tot: f64 = y.iter().map(|v| (v - y_mean).powi(2)).sum();
            if ss_tot < 1e-15 {
                // A constant column carries no variance to inflate.
                return 1.0;
            }

            // Regress y on an intercept plus every other non-constant predictor.
            let mut design = vec![vec![1.0; n_rows]];
            design.extend(
                columns
                    .iter()
                    .enumerate()
                    .filter(|(j, c)| *j != i && !is_constant(c))
                    .map(|(_, c)| c.clone()),
            );
            if design.len() == 1 {
                // Nothing else to regress on.
                return 1.0;
            }
            let Some(coefs) = ols(&design, y) else {
                // Rank-deficient: the other predictors span this one exactly.
                return f64::INFINITY;
            };

            let ss_res: f64 = (0..n_rows)
                .map(|r| {
                    let fitted: f64 = design.iter().zip(&coefs).map(|(c, b)| c[r] * b).sum();
                    (y[r] - fitted).powi(2)
                })
                .sum();
            let r_sq = 1.0 - ss_res / ss_tot;
            if r_sq >= 1.0 - 1e-12 {
                return f64::INFINITY;
            }
            1.0 / (1.0 - r_sq.max(0.0))
        })
        .collect()
}

/// Solve `A x = b` by Gaussian elimination with partial pivoting.
// The elimination step updates row `r` from row `col` at matching indices, so
// iterating one row alone cannot express it.
#[allow(clippy::needless_range_loop)]
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if a.len() != n || a.iter().any(|row| row.len() != n) {
        return None;
    }
    for col in 0..n {
        let (pivot, _) = (col..n).fold((col, 0.0), |(bi, bv), r| {
            let v = a[r][col].abs();
            if v > bv { (r, v) } else { (bi, bv) }
        });
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for r in (col + 1)..n {
            let factor = a[r][col] / a[col][col];
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                a[r][c] -= factor * a[col][c];
            }
            b[r] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= a[i][j] * x[j];
        }
        x[i] = acc / a[i][i];
    }
    Some(x)
}

/// Ordinary least squares via normal equations. `design` is column-major.
///
/// Returns `None` when the design is rank-deficient.
pub(crate) fn ols(design: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let k = design.len();
    if k == 0 {
        return None;
    }
    let gram: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| design[i].iter().zip(&design[j]).map(|(a, b)| a * b).sum())
                .collect()
        })
        .collect();
    let rhs: Vec<f64> = (0..k)
        .map(|i| design[i].iter().zip(y).map(|(a, b)| a * b).sum())
        .collect();
    solve_linear(gram, rhs)
}

/// Format a design matrix as an aligned text table.
#[must_use]
pub fn format_design_matrix(names: &[String], columns: &[Vec<f64>], max_rows: usize) -> String {
    let n_rows = columns.first().map_or(0, std::vec::Vec::len);
    let show = n_rows.min(max_rows);
    let mut lines = Vec::new();

    // Header
    let header: String = names
        .iter()
        .map(|n| format!("{:>10}", &n[..n.len().min(10)]))
        .collect::<Vec<_>>()
        .join(" ");
    lines.push(header);
    lines.push("-".repeat(names.len() * 11));

    for i in 0..show {
        let row: String = columns
            .iter()
            .map(|col| {
                let v = col.get(i).copied().unwrap_or(0.0);
                if v == v.round() && v.abs() < 1000.0 {
                    format!("{v:>10.0}")
                } else {
                    format!("{v:>10.3}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
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
        .collect::<Vec<_>>()
        .join("");
    lines.push(header);

    for i in 0..n {
        let row: String = std::iter::once(format!("{:>10}", &names[i][..names[i].len().min(10)]))
            .chain((0..n).map(|j| {
                let r = corr[i][j].abs();
                let idx = (r * 4.0).round().min(4.0) as usize;
                format!("  {:>1}  ", blocks[idx])
            }))
            .collect::<Vec<_>>()
            .join("");
        lines.push(row);
    }
    lines.join("\n")
}

pub(crate) fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
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
    fn var_comp_dummy_codes_a_factor() {
        let factor: Vec<String> = ["a", "b", "a", "c"].iter().map(|s| (*s).into()).collect();
        let vc = VarComp::from_levels("subject", &factor);
        assert_eq!(vc.n_levels(), 3);
        assert_eq!(vc.n_obs(), 4);
        assert_eq!(vc.level_names, vec!["a", "b", "c"]);
        // 1-based level index per observation.
        assert_eq!(vc.index_vec, vec![1, 2, 1, 3]);
    }

    #[test]
    fn with_random_effects_actually_stores_z() {
        let spec =
            GlmSpec::new(vec![Term::new("intercept", vec![1.0; 4])]).with_random_effects(vec![
                Term::new("s1", vec![1.0, 1.0, 0.0, 0.0]),
                Term::new("s2", vec![0.0, 0.0, 1.0, 1.0]),
            ]);
        assert!(spec.is_mixed());
        assert_eq!(spec.n_random(), 2);
        let (names, cols) = spec.z().expect("Z should be present");
        assert_eq!(names, vec!["VC0.0", "VC0.1"]);
        assert_eq!(cols[0], vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn plain_glm_has_no_z() {
        let spec = GlmSpec::new(vec![Term::new("x", vec![1.0, 2.0])]);
        assert!(!spec.is_mixed());
        assert!(spec.z().is_none());
    }

    #[test]
    fn build_variance_components_splits_z_by_group() {
        let mut spec = GlmSpec::new(vec![]);
        let z = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        // Columns 0,1 form component "subject"; column 2 forms "site".
        let groups = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let names = vec!["subject".to_string(), "site".into()];
        spec.build_variance_components(&z, Some(&groups), Some(&names))
            .unwrap();
        assert_eq!(spec.variance_components().len(), 2);
        assert_eq!(spec.variance_components()[0].n_levels(), 2);
        assert_eq!(spec.variance_components()[1].n_levels(), 1);
        assert_eq!(spec.n_random(), 3);
    }

    #[test]
    fn build_variance_components_rejects_a_shape_mismatch() {
        let mut spec = GlmSpec::new(vec![]);
        let z = vec![vec![1.0], vec![0.0]];
        let groups = vec![vec![1.0]];
        assert!(
            spec.build_variance_components(&z, Some(&groups), None)
                .is_err()
        );
    }

    #[test]
    fn duplicate_term_names_are_rejected() {
        let mut spec = GlmSpec::new(vec![Term::new("x", vec![1.0])]);
        assert!(spec.add_term(Term::new("x", vec![2.0])).is_err());
        assert!(
            spec.add_variance_component(VarComp::new("x", vec![vec![1.0]], vec!["l".into()]))
                .is_err()
        );
    }

    #[test]
    fn variance_components_parse_from_the_model_json() {
        let data: Vec<HashMap<String, f64>> = (0..4)
            .map(|i| {
                HashMap::from([
                    ("x".to_string(), i as f64),
                    ("sub".to_string(), f64::from(i % 2)),
                ])
            })
            .collect();
        let model = serde_json::json!({
            "type": "glm",
            "variance_components": [{"name": "subject", "levels_from": "sub"}]
        });
        let spec = GlmSpec::from_rows(&data, &["x".to_string()], &model);
        assert_eq!(spec.variance_components().len(), 1);
        assert_eq!(spec.variance_components()[0].name, "subject");
        assert_eq!(spec.variance_components()[0].n_levels(), 2);
    }

    #[test]
    fn set_priors_fills_in_terms_without_one() {
        let mut spec = GlmSpec::new(vec![Term::new("x", vec![1.0, 2.0])])
            .with_random_effects(vec![Term::new("s", vec![1.0, 0.0])]);
        spec.set_priors(
            Some(Prior::new("Normal").param("mu", 0).param("sd", 1)),
            Some(Prior::new("HalfCauchy").param("beta", 1)),
        );
        assert_eq!(spec.terms[0].prior.as_ref().unwrap().name, "Normal");
        assert_eq!(
            spec.variance_components()[0].prior.as_ref().unwrap().name,
            "HalfCauchy"
        );
        assert!(spec.priors.contains_key("fixed"));
    }

    #[test]
    fn vif_is_high_for_collinear_predictors() {
        // x2 = 2*x1 + small noise: VIF must be large, not ~1.
        let x1: Vec<f64> = (0..50).map(f64::from).collect();
        let x2: Vec<f64> = x1
            .iter()
            .enumerate()
            .map(|(i, v)| 2.0 * v + f64::from(i as i32 % 3) * 0.01)
            .collect();
        let vifs = compute_vif(&[x1, x2]);
        assert!(vifs[0] > 100.0, "expected high VIF, got {}", vifs[0]);
    }

    #[test]
    fn vif_detects_exact_collinearity() {
        let a: Vec<f64> = (0..20).map(f64::from).collect();
        let b: Vec<f64> = a.iter().map(|v| v * 3.0).collect();
        let vifs = compute_vif(&[a, b]);
        assert!(
            vifs[0].is_infinite(),
            "expected infinite VIF, got {}",
            vifs[0]
        );
    }

    #[test]
    fn test_compute_vif() {
        // Independent columns should have VIF near 1
        let c1: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let c2: Vec<f64> = (0..100).map(|i| (i * 7 % 13) as f64).collect();
        let vifs = compute_vif(&[c1, c2]);
        assert!(
            vifs[0] < 5.0,
            "VIF should be low for independent vars, got {}",
            vifs[0]
        );
    }
}
