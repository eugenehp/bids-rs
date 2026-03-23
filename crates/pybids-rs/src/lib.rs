//! Python bindings for bids-rs via PyO3.
//!
//! Exposes the core bids-rs functionality to Python as the `pybids_rs` module,
//! providing a high-performance alternative backend for PyBIDS workflows.
//!
//! # Available Classes
//!
//! - `BIDSLayout` — Dataset indexing and querying (wraps `bids_layout::BidsLayout`)
//! - `BIDSFile` — File representation with path, filename, entities, and metadata
//!
//! # Usage from Python
//!
//! ```python
//! from pybids_rs import BIDSLayout
//!
//! layout = BIDSLayout("/path/to/bids/dataset")
//! files = layout.get(suffix="bold", extension=".nii.gz")
//! for f in files:
//!     print(f.path, f.entities)
//!
//! # Multi-value filters (like PyBIDS)
//! files = layout.get(subject=["01", "02"], task="rest")
//!
//! # Metadata with BIDS inheritance
//! meta = layout.get_metadata("/path/to/sub-01_bold.nii.gz")
//! print(meta["RepetitionTime"])  # returns native Python types
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::PathBuf;

// ─────────────────────── BIDSLayout ───────────────────────

/// BIDS dataset layout — indexes and queries a BIDS directory.
///
/// This is the Rust-accelerated equivalent of `bids.BIDSLayout`.
#[pyclass(unsendable)]
struct BIDSLayout {
    inner: bids_layout::BidsLayout,
}

#[pymethods]
impl BIDSLayout {
    #[new]
    #[pyo3(signature = (root, validate=true, derivatives=false, index_metadata=true, database_path=None))]
    fn new(
        root: &str, validate: bool, derivatives: bool,
        index_metadata: bool, database_path: Option<&str>,
    ) -> PyResult<Self> {
        let mut builder = bids_layout::BidsLayout::builder(root)
            .validate(validate)
            .index_metadata(index_metadata);
        if let Some(db) = database_path {
            builder = builder.database_path(db);
        }
        if derivatives {
            let deriv_path = PathBuf::from(root).join("derivatives");
            if deriv_path.exists() {
                builder = builder.add_derivative(deriv_path);
            }
        }
        let inner = builder.build().map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Load a layout from an existing SQLite database.
    #[staticmethod]
    fn load(database_path: &str) -> PyResult<Self> {
        let inner = bids_layout::BidsLayout::load(std::path::Path::new(database_path))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn root(&self) -> String { self.inner.root().to_string_lossy().to_string() }

    fn __repr__(&self) -> String { format!("{}", self.inner) }
    fn __str__(&self) -> String { format!("{}", self.inner) }

    // ── Entity listing ──

    fn get_subjects(&self) -> PyResult<Vec<String>> { self.inner.get_subjects().map_err(to_py_err) }
    fn get_sessions(&self) -> PyResult<Vec<String>> { self.inner.get_sessions().map_err(to_py_err) }
    fn get_tasks(&self) -> PyResult<Vec<String>> { self.inner.get_tasks().map_err(to_py_err) }
    fn get_runs(&self) -> PyResult<Vec<String>> { self.inner.get_runs().map_err(to_py_err) }
    fn get_datatypes(&self) -> PyResult<Vec<String>> { self.inner.get_datatypes().map_err(to_py_err) }
    fn get_suffixes(&self) -> PyResult<Vec<String>> { self.inner.get_suffixes().map_err(to_py_err) }
    fn get_entities(&self) -> PyResult<Vec<String>> { self.inner.get_entities().map_err(to_py_err) }

    /// Get unique values for a specific entity.
    fn get_entity_values(&self, entity: &str) -> PyResult<Vec<String>> {
        self.inner.get_entity_values(entity).map_err(to_py_err)
    }

    // ── Core query ──

    /// Query files matching filters. Supports:
    /// - Single string values: `get(subject="01")`
    /// - List values: `get(subject=["01", "02"])`
    /// - `return_type="file"` for path strings
    /// - `return_type="id"` with `target="subject"` for unique entity values
    /// - `scope="all"`, `scope="raw"`, `scope="derivatives"`
    #[pyo3(signature = (**kwargs))]
    fn get<'py>(&self, py: Python<'py>, kwargs: Option<&Bound<'py, PyDict>>) -> PyResult<PyObject> {
        let mut query = self.inner.get()
            .invalid_filters(bids_layout::InvalidFilters::Drop);

        let mut return_type = "object".to_string();
        let mut target = String::new();

        if let Some(kw) = kwargs {
            for (key, val) in kw.iter() {
                let k: String = key.extract()?;
                match k.as_str() {
                    "return_type" => { return_type = val.extract()?; continue; }
                    "target" => { target = val.extract()?; continue; }
                    "scope" => { let v: String = val.extract()?; query = query.scope(&v); continue; }
                    _ => {}
                }

                // Try extracting as list first, then single string
                let values: Vec<String> = if let Ok(list) = val.extract::<Vec<String>>() {
                    list
                } else if let Ok(s) = val.extract::<String>() {
                    vec![s]
                } else {
                    continue;
                };

                if values.is_empty() { continue; }

                // Map to query builder
                let str_refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
                match k.as_str() {
                    "subject" => query = query.filter_any("subject", &str_refs),
                    "session" => query = query.filter_any("session", &str_refs),
                    "task" => query = query.filter_any("task", &str_refs),
                    "run" => query = query.filter_any("run", &str_refs),
                    "datatype" => query = query.filter_any("datatype", &str_refs),
                    "suffix" => query = query.filter_any("suffix", &str_refs),
                    "acquisition" => query = query.filter_any("acquisition", &str_refs),
                    "space" => query = query.filter_any("space", &str_refs),
                    "recording" => query = query.filter_any("recording", &str_refs),
                    "extension" => {
                        // Normalize extensions to have leading dot
                        let normed: Vec<String> = values.iter().map(|v| {
                            if v.starts_with('.') { v.clone() } else { format!(".{}", v) }
                        }).collect();
                        let refs: Vec<&str> = normed.iter().map(|s| s.as_str()).collect();
                        query = query.filter_any("extension", &refs);
                    }
                    other => {
                        if values.len() == 1 {
                            query = query.filter(other, &values[0]);
                        } else {
                            query = query.filter_any(other, &str_refs);
                        }
                    }
                }
            }
        }

        match return_type.as_str() {
            "file" | "filename" => {
                let paths = query.return_paths().map_err(to_py_err)?;
                let strs: Vec<String> = paths.iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect();
                Ok(strs.into_pyobject(py)?.into_any().unbind())
            }
            "id" => {
                if target.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "return_type='id' requires a 'target' argument"));
                }
                let vals = query.return_unique(&target).map_err(to_py_err)?;
                Ok(vals.into_pyobject(py)?.into_any().unbind())
            }
            "dir" => {
                if target.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "return_type='dir' requires a 'target' argument"));
                }
                let dirs = query.return_directories(&target).map_err(to_py_err)?;
                Ok(dirs.into_pyobject(py)?.into_any().unbind())
            }
            _ => {
                // Default: return BIDSFile objects
                let files = query.collect().map_err(to_py_err)?;
                let pyfiles: Vec<BIDSFile> = files.iter().map(|f| BIDSFile {
                    path: f.path.to_string_lossy().to_string(),
                    filename: f.filename.clone(),
                    entities: f.entities.iter()
                        .map(|(k, v)| (k.clone(), v.as_str_lossy().into_owned()))
                        .collect(),
                }).collect();
                Ok(pyfiles.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Get a single file by path.
    fn get_file(&self, path: &str) -> PyResult<Option<BIDSFile>> {
        let f = self.inner.get_file(path).map_err(to_py_err)?;
        Ok(f.map(|f| BIDSFile {
            path: f.path.to_string_lossy().to_string(),
            filename: f.filename.clone(),
            entities: f.entities.iter()
                .map(|(k, v)| (k.clone(), v.as_str_lossy().into_owned()))
                .collect(),
        }))
    }

    /// Get metadata for a file, with BIDS JSON sidecar inheritance.
    ///
    /// Returns a dict with native Python types (float, int, str, list, dict, bool, None).
    fn get_metadata<'py>(&self, py: Python<'py>, path: &str) -> PyResult<PyObject> {
        let md = self.inner.get_metadata(path).map_err(to_py_err)?;
        let dict = PyDict::new(py);
        for (k, v) in md.iter() {
            dict.set_item(k, json_to_py(py, v)?)?;
        }
        Ok(dict.into_any().unbind())
    }

    fn get_dataset_description<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        Ok(self.inner.description().map(|d| {
            let dict = PyDict::new(py);
            let _ = dict.set_item("Name", &d.name);
            let _ = dict.set_item("BIDSVersion", &d.bids_version);
            if let Some(ref l) = d.license { let _ = dict.set_item("License", l); }
            if let Some(ref dt) = d.dataset_type { let _ = dict.set_item("DatasetType", dt); }
            if let Some(ref a) = d.authors { let _ = dict.set_item("Authors", a); }
            dict.into_any().unbind()
        }))
    }

    /// Get the repetition time for functional data.
    #[pyo3(signature = (**kwargs))]
    fn get_tr(&self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<f64> {
        let mut filters = Vec::new();
        if let Some(kw) = kwargs {
            for (key, val) in kw.iter() {
                let k: String = key.extract()?;
                let v: String = val.extract()?;
                filters.push(bids_layout::QueryFilter::eq(&k, &v));
            }
        }
        self.inner.get_tr(&filters).map_err(to_py_err)
    }

    /// Parse BIDS entities from a filename string.
    fn parse_file_entities(&self, filename: &str) -> HashMap<String, String> {
        self.inner.parse_file_entities(filename)
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str_lossy().into_owned()))
            .collect()
    }

    /// Build a BIDS-compliant path from entities.
    #[pyo3(signature = (entities, path_patterns=None, strict=false))]
    fn build_path(
        &self, entities: HashMap<String, String>,
        path_patterns: Option<Vec<String>>, strict: bool,
    ) -> PyResult<Option<String>> {
        use bids_core::entities::{Entities, EntityValue};
        let ents: Entities = entities.into_iter()
            .map(|(k, v)| (k, EntityValue::Str(v)))
            .collect();
        let patterns_owned = path_patterns.unwrap_or_default();
        let pattern_refs: Vec<&str> = patterns_owned.iter().map(|s| s.as_str()).collect();
        let pats = if pattern_refs.is_empty() { None } else { Some(pattern_refs.as_slice()) };
        match self.inner.build_path(&ents, pats, strict) {
            Ok(p) => Ok(Some(p)),
            Err(_) => Ok(None),
        }
    }

    /// Add a derivatives directory.
    fn add_derivatives(&mut self, path: &str) -> PyResult<()> {
        self.inner.add_derivatives(path).map_err(to_py_err)
    }

    /// Export as a list of (path, entity_name, value) tuples.
    #[pyo3(signature = (metadata=false))]
    fn to_df(&self, metadata: bool) -> PyResult<Vec<(String, String, String)>> {
        self.inner.to_df(metadata).map_err(to_py_err)
    }

    /// Save the database index to a file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(std::path::Path::new(path)).map_err(to_py_err)
    }
}

// ─────────────────────── BIDSFile ───────────────────────

/// A single file in a BIDS dataset.
#[pyclass]
#[derive(Clone)]
struct BIDSFile {
    #[pyo3(get)]
    path: String,
    #[pyo3(get)]
    filename: String,
    #[pyo3(get)]
    entities: HashMap<String, String>,
}

#[pymethods]
impl BIDSFile {
    fn __repr__(&self) -> String { format!("<BIDSFile '{}'>", self.filename) }
    fn __str__(&self) -> String { self.path.clone() }
    fn __fspath__(&self) -> String { self.path.clone() }

    /// Get the path relative to the dataset root.
    #[getter]
    fn relpath(&self) -> String {
        self.path.rsplit_once("/sub-")
            .map(|(_, rest)| format!("sub-{}", rest))
            .unwrap_or_else(|| self.path.clone())
    }

    /// File extension (handles compound extensions like `.nii.gz`).
    #[getter]
    fn extension(&self) -> &str {
        let name = &self.filename;
        if name.ends_with(".nii.gz") { ".nii.gz" }
        else if name.ends_with(".tsv.gz") { ".tsv.gz" }
        else if name.ends_with(".dtseries.nii") { ".dtseries.nii" }
        else { name.rfind('.').map(|i| &name[i..]).unwrap_or("") }
    }

    /// File suffix (the part before the extension, after the last `_`).
    #[getter]
    fn suffix(&self) -> Option<String> {
        let stem = self.filename.split('.').next()?;
        stem.rsplit('_').next().map(|s| s.to_string())
    }

    /// Get the entity value for a key, or None.
    fn get_entity(&self, key: &str) -> Option<String> {
        self.entities.get(key).cloned()
    }
}

// ─────────────────────── NIfTI ───────────────────────

/// Read a NIfTI header without loading voxel data.
#[pyfunction]
fn read_nifti_header(path: &str) -> PyResult<HashMap<String, PyObject>> {
    Python::with_gil(|py| {
        let hdr = bids_nifti::NiftiHeader::from_file(std::path::Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut m: HashMap<String, PyObject> = HashMap::new();
        m.insert("ndim".into(), hdr.ndim.into_pyobject(py)?.into_any().unbind());
        m.insert("shape".into(), hdr.shape().into_pyobject(py)?.into_any().unbind());
        m.insert("n_vols".into(), hdr.n_vols().into_pyobject(py)?.into_any().unbind());
        let (mx, my, mz) = hdr.matrix_size();
        m.insert("matrix_size".into(), (mx, my, mz).into_pyobject(py)?.into_any().unbind());
        let (vx, vy, vz) = hdr.voxel_size();
        m.insert("voxel_size".into(), (vx, vy, vz).into_pyobject(py)?.into_any().unbind());
        m.insert("tr".into(), hdr.tr().into_pyobject(py)?.into_any().unbind());
        m.insert("datatype".into(), (hdr.datatype as i32).into_pyobject(py)?.into_any().unbind());
        m.insert("n_voxels".into(), hdr.n_voxels().into_pyobject(py)?.into_any().unbind());
        Ok(m)
    })
}

// ─────────────────────── Filter ───────────────────────

/// Compute Butterworth lowpass filter coefficients.
#[pyfunction]
#[pyo3(signature = (order, cutoff))]
fn butter_lowpass(order: usize, cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    bids_filter::butter_lowpass(order, cutoff)
}

/// Compute Butterworth highpass filter coefficients.
#[pyfunction]
#[pyo3(signature = (order, cutoff))]
fn butter_highpass(order: usize, cutoff: f64) -> (Vec<f64>, Vec<f64>) {
    bids_filter::butter_highpass(order, cutoff)
}

/// Compute Butterworth bandpass filter coefficients.
#[pyfunction]
#[pyo3(signature = (order, low, high))]
fn butter_bandpass(order: usize, low: f64, high: f64) -> (Vec<f64>, Vec<f64>) {
    bids_filter::butter_bandpass(order, low, high)
}

/// Apply a notch filter to remove a specific frequency (e.g., 50/60 Hz line noise).
#[pyfunction]
#[pyo3(signature = (x, freq_hz, fs, quality=30.0))]
fn notch_filter(x: Vec<f64>, freq_hz: f64, fs: f64, quality: f64) -> Vec<f64> {
    bids_filter::notch_filter(&x, freq_hz, fs, quality)
}

/// Resample a signal with anti-aliasing (like MNE's raw.resample()).
#[pyfunction]
fn resample(x: Vec<f64>, fs_old: f64, fs_new: f64) -> Vec<f64> {
    bids_filter::resample(&x, fs_old, fs_new)
}

/// Zero-phase digital filtering (like scipy.signal.filtfilt).
#[pyfunction]
fn filtfilt(b: Vec<f64>, a: Vec<f64>, x: Vec<f64>) -> Vec<f64> {
    bids_filter::filtfilt(&b, &a, &x)
}

// ─────────────────────── HRF ───────────────────────

/// SPM canonical hemodynamic response function.
#[pyfunction]
#[pyo3(signature = (tr, oversampling=50, time_length=32.0, onset=0.0))]
fn spm_hrf(tr: f64, oversampling: usize, time_length: f64, onset: f64) -> Vec<f64> {
    bids_modeling::spm_hrf(tr, oversampling, time_length, onset)
}

/// Glover canonical hemodynamic response function.
#[pyfunction]
#[pyo3(signature = (tr, oversampling=50, time_length=32.0, onset=0.0))]
fn glover_hrf(tr: f64, oversampling: usize, time_length: f64, onset: f64) -> Vec<f64> {
    bids_modeling::glover_hrf(tr, oversampling, time_length, onset)
}

// ─────────────────────── Formula ───────────────────────

/// Parse a Wilkinson-style formula (e.g., "y ~ a * b + c").
#[pyfunction]
fn parse_formula(formula: &str) -> HashMap<String, PyObject> {
    Python::with_gil(|py| {
        let f = bids_formula::parse_formula(formula);
        let mut m: HashMap<String, PyObject> = HashMap::new();
        m.insert("response".into(), f.response.into_pyobject(py).unwrap().into_any().unbind());
        m.insert("intercept".into(), pyo3::types::PyBool::new(py, f.intercept).to_owned().into_any().unbind());
        let terms: Vec<String> = f.terms.iter().map(|t| t.name()).collect();
        m.insert("terms".into(), terms.into_pyobject(py).unwrap().into_any().unbind());
        m
    })
}

// ─────────────────────── Schema ───────────────────────

/// Check if a path is valid according to the BIDS schema.
#[pyfunction]
fn validate_bids_path(path: &str) -> bool {
    bids_schema::BidsSchema::load().is_valid(path)
}

/// Check if a datatype string is valid (e.g., "anat", "func", "eeg").
#[pyfunction]
fn is_valid_datatype(dt: &str) -> bool {
    bids_schema::BidsSchema::load().is_valid_datatype(dt)
}

// ─────────────────────── Inflect ───────────────────────

/// Convert a plural English word to singular.
#[pyfunction]
fn singularize(word: &str) -> Option<String> { bids_inflect::singularize(word) }

/// Convert a singular English word to plural.
#[pyfunction]
fn pluralize(word: &str) -> String { bids_inflect::pluralize(word) }

// ─────────────────────── Reports ───────────────────────

/// Auto-generate a publication-quality methods section.
#[pyfunction]
fn generate_report(root: &str) -> PyResult<String> {
    let layout = bids_layout::BidsLayout::new(root).map_err(to_py_err)?;
    let report = bids_reports::BidsReport::new(&layout);
    report.generate().map_err(to_py_err)
}

// ─────────────────────── Helpers ───────────────────────

fn to_py_err(e: bids_core::error::BidsError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Convert a serde_json::Value to a native Python object.
fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    match v {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => {
            let obj = pyo3::types::PyBool::new(py, *b);
            Ok(obj.to_owned().into_any().unbind())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(n.to_string().into_pyobject(py)?.into_any().unbind())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr.iter()
                .map(|item| json_to_py(py, item))
                .collect::<PyResult<_>>()?;
            Ok(items.into_pyobject(py)?.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, val) in map {
                dict.set_item(k, json_to_py(py, val)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

// ─────────────────────── Module ───────────────────────

#[pymodule]
fn pybids_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BIDSLayout>()?;
    m.add_class::<BIDSFile>()?;
    m.add_function(wrap_pyfunction!(read_nifti_header, m)?)?;
    m.add_function(wrap_pyfunction!(butter_lowpass, m)?)?;
    m.add_function(wrap_pyfunction!(butter_highpass, m)?)?;
    m.add_function(wrap_pyfunction!(butter_bandpass, m)?)?;
    m.add_function(wrap_pyfunction!(notch_filter, m)?)?;
    m.add_function(wrap_pyfunction!(resample, m)?)?;
    m.add_function(wrap_pyfunction!(filtfilt, m)?)?;
    m.add_function(wrap_pyfunction!(spm_hrf, m)?)?;
    m.add_function(wrap_pyfunction!(glover_hrf, m)?)?;
    m.add_function(wrap_pyfunction!(parse_formula, m)?)?;
    m.add_function(wrap_pyfunction!(validate_bids_path, m)?)?;
    m.add_function(wrap_pyfunction!(is_valid_datatype, m)?)?;
    m.add_function(wrap_pyfunction!(singularize, m)?)?;
    m.add_function(wrap_pyfunction!(pluralize, m)?)?;
    m.add_function(wrap_pyfunction!(generate_report, m)?)?;
    Ok(())
}
