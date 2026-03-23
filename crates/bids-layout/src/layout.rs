//! The main `BidsLayout` type and its builder.
//!
//! [`BidsLayout`] is the primary entry point for working with a BIDS dataset.
//! It indexes the directory tree into a SQLite database and provides fluent
//! query methods, metadata inheritance, derivative support, and path building.

use bids_core::config::Config;
use bids_core::dataset_description::DatasetDescription;
use bids_core::entities::{Entities, EntityValue};
use bids_core::error::{BidsError, Result};
use bids_core::file::BidsFile;
use bids_core::metadata::BidsMetadata;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::db::Database;
use crate::get_builder::GetBuilder;
use crate::indexer::{self, IndexerOptions};
use crate::query::{QueryFilter, Scope};

/// The main entry point for interacting with a BIDS dataset.
///
/// `BidsLayout` indexes a BIDS dataset directory into a SQLite database and
/// provides a fluent query API for finding files by their BIDS entities. It
/// handles JSON sidecar metadata inheritance, file associations, derivative
/// datasets, and path building.
///
/// This is the Rust equivalent of PyBIDS' `BIDSLayout` class.
///
/// # Thread Safety
///
/// `BidsLayout` wraps a `rusqlite::Connection` and is therefore `!Send` and
/// `!Sync`. It **cannot** be shared across threads or sent to async tasks.
///
/// For multi-threaded or async workloads:
///
/// 1. **Save once, load per-thread** — Use [`save()`](Self::save) to persist
///    the index, then [`load()`](Self::load) in each thread/task:
///    ```no_run
///    # use bids_layout::BidsLayout;
///    let layout = BidsLayout::new("/data").unwrap();
///    layout.save(std::path::Path::new("/tmp/index.sqlite")).unwrap();
///
///    // In each thread:
///    let local = BidsLayout::load(std::path::Path::new("/tmp/index.sqlite")).unwrap();
///    ```
///
/// 2. **Create per-thread** — Call `BidsLayout::new()` in each thread. The
///    directory walk is fast for typical datasets (< 100ms for 10k files).
///
/// # Creating a Layout
///
/// ```no_run
/// use bids_layout::BidsLayout;
///
/// // Simple: index with defaults (validation enabled, in-memory database)
/// let layout = BidsLayout::new("/path/to/bids/dataset").unwrap();
///
/// // Builder: customize indexing behavior
/// let layout = BidsLayout::builder("/path/to/dataset")
///     .validate(false)                            // skip BIDS validation
///     .database_path("/tmp/index.sqlite")         // persistent database
///     .index_metadata(true)                       // index JSON sidecars
///     .add_derivative("/path/to/derivatives/fmriprep")
///     .build()
///     .unwrap();
/// ```
///
/// # Querying Files
///
/// ```no_run
/// # use bids_layout::BidsLayout;
/// # let layout = BidsLayout::new("/path").unwrap();
/// // Fluent query API
/// let files = layout.get()
///     .suffix("bold")
///     .extension(".nii.gz")
///     .subject("01")
///     .task("rest")
///     .collect().unwrap();
///
/// // Get unique entity values
/// let subjects = layout.get_subjects().unwrap();
/// let tasks = layout.get_tasks().unwrap();
///
/// // Get metadata with BIDS inheritance
/// let md = layout.get_metadata(&files[0].path).unwrap();
/// let tr = md.get_f64("RepetitionTime");
/// ```
///
/// # Derivatives
///
/// ```no_run
/// # use bids_layout::BidsLayout;
/// # let mut layout = BidsLayout::new("/path").unwrap();
/// layout.add_derivatives("/path/to/derivatives").unwrap();
///
/// // Query across raw + derivatives
/// let all_files = layout.get().scope("all").suffix("bold").collect().unwrap();
///
/// // Query derivatives only
/// let deriv_files = layout.get().scope("derivatives").collect().unwrap();
/// ```
pub struct BidsLayout {
    root: PathBuf,
    db: Database,
    description: Option<DatasetDescription>,
    pub is_derivative: bool,
    pub source_pipeline: Option<String>,
    derivatives: HashMap<String, BidsLayout>,
    configs: Vec<Config>,
    #[allow(dead_code)]
    regex_search: bool,
    /// The BIDS spec version compatibility status for this dataset.
    ///
    /// Set during construction by comparing the dataset's declared
    /// `BIDSVersion` against the library's supported version.
    spec_compatibility: Option<bids_schema::Compatibility>,
}

impl BidsLayout {
    /// Create a new layout with default settings (validation enabled, in-memory DB).
    ///
    /// # Errors
    ///
    /// Returns an error if the root path doesn't exist, `dataset_description.json`
    /// is missing or invalid, or the filesystem walk fails.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        Self::builder(root).build()
    }

    pub fn builder(root: impl AsRef<Path>) -> LayoutBuilder {
        LayoutBuilder::new(root)
    }

    /// Load a layout from an existing database file.
    ///
    /// # Errors
    ///
    /// Returns an error if the database file doesn't exist, can't be opened,
    /// or doesn't contain valid layout info.
    pub fn load(database_path: &Path) -> Result<Self> {
        let db = Database::open(database_path)?;
        let (root_str, config_str) = db.get_layout_info()?
            .ok_or_else(|| BidsError::Database("No layout info in database".into()))?;
        let root = PathBuf::from(&root_str);
        let description = DatasetDescription::from_dir(&root).ok();
        let config_names: Vec<String> = config_str.split(',').map(std::string::ToString::to_string).collect();
        let configs: Vec<Config> = config_names.iter()
            .filter_map(|name| Config::load(name).ok())
            .collect();

        let spec_compatibility = description.as_ref().map(|d| {
            let schema = bids_schema::BidsSchema::load();
            schema.check_dataset_version(&d.bids_version)
        });

        Ok(BidsLayout {
            root,
            db,
            description,
            is_derivative: false,
            source_pipeline: None,
            derivatives: HashMap::new(),
            configs,
            regex_search: false,
            spec_compatibility,
        })
    }

    /// Save the database to a file for later reloading with [`BidsLayout::load`].
    ///
    /// # Errors
    ///
    /// Returns a database error if the backup operation fails.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.db.save_to(path)
    }

    #[must_use] pub fn root(&self) -> &Path { &self.root }
    #[must_use] pub fn description(&self) -> Option<&DatasetDescription> { self.description.as_ref() }
    pub(crate) fn db(&self) -> &Database { &self.db }

    /// The BIDS specification version declared in `dataset_description.json`.
    #[must_use]
    pub fn bids_version(&self) -> Option<&str> {
        self.description.as_ref().map(|d| d.bids_version.as_str())
    }

    /// Compatibility between the dataset's BIDS version and this library.
    ///
    /// Returns `None` if no `dataset_description.json` was found (e.g., for
    /// derivative datasets loaded without validation).
    #[must_use]
    pub fn spec_compatibility(&self) -> Option<&bids_schema::Compatibility> {
        self.spec_compatibility.as_ref()
    }

    /// Start building a query.
    pub fn get(&self) -> GetBuilder<'_> {
        GetBuilder::new(self)
    }

    pub fn get_subjects(&self) -> Result<Vec<String>> { self.db.get_unique_entity_values("subject") }
    pub fn get_sessions(&self) -> Result<Vec<String>> { self.db.get_unique_entity_values("session") }
    pub fn get_tasks(&self) -> Result<Vec<String>> { self.db.get_unique_entity_values("task") }
    pub fn get_runs(&self) -> Result<Vec<String>> { self.db.get_unique_entity_values("run") }
    pub fn get_datatypes(&self) -> Result<Vec<String>> { self.db.get_unique_entity_values("datatype") }
    pub fn get_suffixes(&self) -> Result<Vec<String>> { self.db.get_unique_entity_values("suffix") }
    pub fn get_entities(&self) -> Result<Vec<String>> { self.db.get_entity_names() }
    pub fn get_entity_values(&self, entity: &str) -> Result<Vec<String>> { self.db.get_unique_entity_values(entity) }

    /// Resolve a path (relative to root if not absolute) to an absolute string.
    fn resolve_path(&self, path: impl AsRef<Path>) -> String {
        let p = if path.as_ref().is_absolute() {
            path.as_ref().to_path_buf()
        } else {
            self.root.join(path)
        };
        p.to_string_lossy().into_owned()
    }

    /// Get a specific file by path.
    pub fn get_file(&self, path: impl AsRef<Path>) -> Result<Option<BidsFile>> {
        let path_str = self.resolve_path(path);
        let tags = self.db.get_tags(&path_str)?;
        if tags.is_empty() {
            let all = self.db.all_file_paths()?;
            if !all.contains(&path_str) { return Ok(None); }
        }
        Ok(Some(self.reconstruct_file(&path_str)?))
    }

    /// Get metadata for a file.
    pub fn get_metadata(&self, path: impl AsRef<Path>) -> Result<BidsMetadata> {
        let path_str = self.resolve_path(path);
        let tags = self.db.get_tags(&path_str)?;
        let mut md = BidsMetadata::with_source(&path_str);
        for (name, value, _dtype, is_metadata) in tags {
            if is_metadata {
                let json_val = serde_json::from_str(&value)
                    .unwrap_or(serde_json::Value::String(value));
                md.insert(name, json_val);
            }
        }
        Ok(md)
    }

    /// Get the scanning repetition time (TR) for matching runs.
    pub fn get_tr(&self, filters: &[QueryFilter]) -> Result<f64> {
        let mut all_filters: Vec<(String, Vec<String>, bool)> = QueryFilter::to_tuples(filters);
        all_filters.push(("suffix".to_string(), vec!["bold".to_string()], false));
        all_filters.push(("datatype".to_string(), vec!["func".to_string()], false));

        let paths = self.db.query_files(&all_filters)?;
        if paths.is_empty() {
            return Err(BidsError::NoMatch("No functional images match criteria".into()));
        }

        // Collect unique TRs, rounding to 10µs to avoid float comparison issues.
        let mut trs = std::collections::HashSet::new();
        for path in &paths {
            let md = self.get_metadata(path)?;
            if let Some(tr) = md.get_f64("RepetitionTime") {
                trs.insert((tr * 100_000.0).round() as i64);
            }
        }

        if trs.len() > 1 {
            return Err(BidsError::NoMatch("Multiple unique TRs found".into()));
        }

        trs.into_iter().next()
            .map(|v| v as f64 / 100_000.0)
            .ok_or_else(|| BidsError::NoMatch("No RepetitionTime found in metadata".into()))
    }

    /// Get bvec file for a path.
    pub fn get_bvec(&self, path: impl AsRef<Path>) -> Result<Option<BidsFile>> {
        self.get_nearest(path, &[
            QueryFilter::eq("extension", ".bvec"),
            QueryFilter::eq("suffix", "dwi"),
        ])
    }

    /// Load the gradient table (b-values + b-vectors) for a DWI file.
    ///
    /// Looks up the companion `.bval` and `.bvec` files and parses them
    /// into a [`bids_io::gradient::GradientTable`].
    ///
    /// # Errors
    ///
    /// Returns an error if the companion files aren't found or can't be parsed.
    pub fn get_gradient_table(&self, path: impl AsRef<Path>) -> Result<bids_io::gradient::GradientTable> {
        let bvec_file = self.get_bvec(&path)?
            .ok_or_else(|| BidsError::NoMatch("No .bvec file found".into()))?;
        let bval_file = self.get_bval(&path)?
            .ok_or_else(|| BidsError::NoMatch("No .bval file found".into()))?;
        bids_io::gradient::GradientTable::from_files(&bval_file.path, &bvec_file.path)
    }

    /// Get bval file for a path.
    pub fn get_bval(&self, path: impl AsRef<Path>) -> Result<Option<BidsFile>> {
        self.get_nearest(path, &[
            QueryFilter::eq("extension", ".bval"),
            QueryFilter::eq("suffix", "dwi"),
        ])
    }

    /// Add a derivatives directory.
    pub fn add_derivatives(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let desc_path = path.join("dataset_description.json");

        if desc_path.exists() {
            let pipeline_name = bids_validate::validate_derivative_path(path)?;
            if self.derivatives.contains_key(&pipeline_name) {
                return Err(BidsError::DerivativesValidation(
                    format!("Pipeline '{pipeline_name}' already added")));
            }
            let deriv = LayoutBuilder::new(path)
                .validate(false)
                .is_derivative(true)
                .build()?;
            self.derivatives.insert(pipeline_name, deriv);
        } else if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                if entry.file_type().is_ok_and(|t| t.is_dir()) {
                    let sub_desc = entry.path().join("dataset_description.json");
                    if sub_desc.exists() {
                        self.add_derivatives(entry.path())?;
                    }
                }
            }
        }
        Ok(())
    }

    #[must_use] pub fn get_derivative(&self, name: &str) -> Option<&BidsLayout> { self.derivatives.get(name) }
    #[must_use] pub fn derivatives(&self) -> &HashMap<String, BidsLayout> { &self.derivatives }

    /// Check if this layout is in the specified scope.
    fn in_scope(&self, scope: &Scope) -> bool {
        match scope {
            Scope::All => true,
            Scope::Self_ => true,
            Scope::Raw => !self.is_derivative,
            Scope::Derivatives => self.is_derivative,
            Scope::Pipeline(name) => self.source_pipeline.as_deref() == Some(name.as_str()),
        }
    }

    /// Get all layouts in the specified scope (recursive through derivatives).
    fn get_layouts_in_scope(&self, scope: &Scope) -> Vec<&BidsLayout> {
        if *scope == Scope::Self_ {
            return vec![self];
        }

        let mut layouts = Vec::new();
        if self.in_scope(scope) {
            layouts.push(self);
        }
        for deriv in self.derivatives.values() {
            layouts.extend(deriv.get_layouts_in_scope(scope));
        }
        layouts
    }

    /// Get the nearest file matching filters, walking up the directory tree.
    pub fn get_nearest(
        &self,
        path: impl AsRef<Path>,
        filters: &[QueryFilter],
    ) -> Result<Option<BidsFile>> {
        let path = path.as_ref();

        // Get the suffix from the source file if not in filters
        let has_suffix = filters.iter().any(|f| f.entity == "suffix");
        let mut final_filters = filters.to_vec();
        if !has_suffix
            && let Some(bf) = self.get_file(path)?
                && let Some(EntityValue::Str(s)) = bf.entities.get("suffix") {
                    final_filters.push(QueryFilter::eq("suffix", s));
                }

        // Get the source file's entities for scoring
        let source_entities: HashMap<String, String> = if let Some(bf) = self.get_file(path)? {
            bf.entities.iter().map(|(k, v)| (k.clone(), v.as_str_lossy().into_owned())).collect()
        } else {
            HashMap::new()
        };

        // Get all candidate files
        let filter_tuples: Vec<_> = final_filters.iter()
            .map(|f| (f.entity.clone(), f.values.clone(), f.regex))
            .collect();
        let candidates = self.db.query_files(&filter_tuples)?;

        // Group candidates by directory
        let mut dir_files: HashMap<PathBuf, Vec<String>> = HashMap::new();
        for c in &candidates {
            let p = PathBuf::from(c);
            let dir = p.parent().unwrap_or(Path::new("")).to_path_buf();
            dir_files.entry(dir).or_default().push(c.clone());
        }

        // Walk up from the source file's directory
        let mut dir = path.parent();
        while let Some(current_dir) = dir {
            if let Some(files_in_dir) = dir_files.get(current_dir) {
                // Score candidates by matching entities
                let mut best: Option<(usize, String)> = None;
                for file_path in files_in_dir {
                    let tags = self.db.get_tags(file_path)?;
                    let file_ents: HashMap<String, String> = tags.iter()
                        .filter(|(_, _, _, m)| !m)
                        .map(|(n, v, _, _)| (n.clone(), v.clone()))
                        .collect();

                    let score: usize = source_entities.iter()
                        .filter(|(k, v)| file_ents.get(*k) == Some(v))
                        .count();
                    if best.as_ref().is_none_or(|(s, _)| score > *s) {
                        best = Some((score, file_path.clone()));
                    }
                }

                if let Some((_, best_path)) = best {
                    return Ok(Some(self.reconstruct_file(&best_path)?));
                }
            }

            if current_dir == self.root { break; }
            dir = current_dir.parent();
        }

        Ok(None)
    }

    /// Parse entities from a filename using this layout's config.
    #[must_use]
    pub fn parse_file_entities(&self, filename: &str) -> Entities {
        let all_entities = self.all_entity_defs();
        bids_core::entities::parse_file_entities(filename, &all_entities)
    }

    /// Collect all unique entity definitions from this layout's configs.
    fn all_entity_defs(&self) -> Vec<bids_core::Entity> {
        let mut all = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for config in &self.configs {
            for entity in &config.entities {
                if seen.insert(&entity.name) {
                    all.push(entity.clone());
                }
            }
        }
        all
    }

    /// Build a path from entities using this layout's config patterns.
    pub fn build_path(
        &self,
        source: &Entities,
        path_patterns: Option<&[&str]>,
        strict: bool,
    ) -> Result<String> {
        let default_patterns: Vec<String>;
        let patterns: Vec<&str> = if let Some(p) = path_patterns {
            p.to_vec()
        } else {
            default_patterns = self.configs.iter()
                .filter_map(|c| c.default_path_patterns.as_ref())
                .flat_map(|p| p.iter().cloned())
                .collect();
            default_patterns.iter().map(std::string::String::as_str).collect()
        };

        bids_io::path_builder::build_path(source, &patterns, strict)
            .ok_or_else(|| BidsError::PathBuilding(
                "Unable to construct path from provided entities".into()))
    }

    /// Export file index as rows of (path, entity_name, value).
    pub fn to_df(&self, metadata: bool) -> Result<Vec<(String, String, String)>> {
        let paths = self.db.all_file_paths()?;
        let mut rows = Vec::new();
        for path in &paths {
            let tags = self.db.get_tags(path)?;
            for (name, value, _, is_meta) in &tags {
                if metadata || !is_meta {
                    rows.push((path.clone(), name.clone(), value.clone()));
                }
            }
        }
        Ok(rows)
    }

    /// Deep copy the layout.
    pub fn clone_layout(&self) -> Result<Self> {
        // Re-index from the same root
        Self::builder(&self.root)
            .validate(false)
            .is_derivative(self.is_derivative)
            .build()
    }

    /// Internal: execute a query with filters.
    pub(crate) fn query_files_internal(
        &self,
        filters: &[(String, Vec<String>, bool)],
        scope: &Scope,
    ) -> Result<Vec<String>> {
        let layouts = self.get_layouts_in_scope(scope);
        let mut all_paths = Vec::new();
        for layout in layouts {
            let paths = layout.db.query_files(filters)?;
            all_paths.extend(paths);
        }
        // Deduplicate
        let mut seen = std::collections::HashSet::new();
        all_paths.retain(|p| seen.insert(p.clone()));
        Ok(all_paths)
    }

    /// Internal: reconstruct a BidsFile from its path in the database.
    pub(crate) fn reconstruct_file(&self, path_str: &str) -> Result<BidsFile> {
        let mut bf = BidsFile::new(path_str);
        let tags = self.db.get_tags(path_str)?;
        for (name, value, _dtype, is_metadata) in tags {
            if !is_metadata {
                bf.entities.insert(name, EntityValue::Str(value));
            } else {
                let json_val = serde_json::from_str(&value)
                    .unwrap_or(serde_json::Value::String(value));
                bf.metadata.insert(name, json_val);
            }
        }
        Ok(bf)
    }

    /// Get fieldmap(s) for a specified file path.
    pub fn get_fieldmap(&self, path: impl AsRef<Path>) -> Result<Vec<HashMap<String, String>>> {
        let path = path.as_ref();
        let ents = self.parse_file_entities(&path.to_string_lossy());
        let subject = ents.get("subject").map(|v| v.as_str_lossy()).unwrap_or_default();

        let fmap_files = self.get()
            .subject(&subject)
            .filter_regex("suffix", "(phasediff|magnitude[12]|phase[12]|fieldmap|epi)")
            .filter_any("extension", &[".nii.gz", ".nii"])
            .collect()?;

        let mut fieldmap_set = Vec::new();
        for file in &fmap_files {
            let md = self.get_metadata(&file.path)?;
            let intended = md.get("IntendedFor");
            if intended.is_none() { continue; }

            let intents: Vec<String> = match intended.unwrap() {
                serde_json::Value::String(s) => vec![s.clone()],
                serde_json::Value::Array(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
                _ => continue,
            };

            let path_str = path.to_string_lossy();
            if !intents.iter().any(|i| path_str.ends_with(i)) { continue; }

            let suffix = file.entities.get("suffix").map(|v| v.as_str_lossy()).unwrap_or_default();
            let mut fmap = HashMap::new();
            let fp = file.path.to_string_lossy().to_string();

            match &*suffix {
                "phasediff" => {
                    fmap.insert("phasediff".into(), fp.clone());
                    fmap.insert("magnitude1".into(), fp.replace("phasediff", "magnitude1"));
                    let mag2 = fp.replace("phasediff", "magnitude2");
                    if std::path::Path::new(&mag2).exists() { fmap.insert("magnitude2".into(), mag2); }
                    fmap.insert("suffix".into(), "phasediff".into());
                }
                "phase1" => {
                    fmap.insert("phase1".into(), fp.clone());
                    fmap.insert("magnitude1".into(), fp.replace("phase1", "magnitude1"));
                    fmap.insert("phase2".into(), fp.replace("phase1", "phase2"));
                    fmap.insert("magnitude2".into(), fp.replace("phase1", "magnitude2"));
                    fmap.insert("suffix".into(), "phase".into());
                }
                "epi" => {
                    fmap.insert("epi".into(), fp);
                    fmap.insert("suffix".into(), "epi".into());
                }
                "fieldmap" => {
                    fmap.insert("fieldmap".into(), fp.clone());
                    fmap.insert("magnitude".into(), fp.replace("fieldmap", "magnitude"));
                    fmap.insert("suffix".into(), "fieldmap".into());
                }
                _ => continue,
            }
            fieldmap_set.push(fmap);
        }
        Ok(fieldmap_set)
    }

    /// Copy BIDSFiles to new locations defined by path patterns.
    pub fn copy_files(
        &self,
        path_patterns: &[&str],
        mode: bids_core::file::CopyMode,
        root: Option<&Path>,
        filters: &[QueryFilter],
    ) -> Result<Vec<PathBuf>> {
        let root = root.unwrap_or(&self.root);
        let tuples = QueryFilter::to_tuples(filters);
        let files = self.query_files_internal(&tuples, &Scope::All)?;
        let mut copied = Vec::new();
        for path_str in &files {
            let bf = self.reconstruct_file(path_str)?;
            if let Ok(new_path_str) = self.build_path(&bf.entities, Some(path_patterns), false) {
                let new_path = root.join(&new_path_str);
                bf.copy_to(&new_path, mode)?;
                copied.push(new_path);
            }
        }
        Ok(copied)
    }

    /// Write data to a file defined by entities and path patterns.
    pub fn write_to_file(
        &self,
        entities: &Entities,
        path_patterns: Option<&[&str]>,
        contents: &[u8],
        strict: bool,
    ) -> Result<PathBuf> {
        let path_str = self.build_path(entities, path_patterns, strict)?;
        let full_path = self.root.join(&path_str);
        bids_io::writer::write_to_file(
            &full_path, Some(contents), None, None, None,
            bids_io::writer::ConflictStrategy::Fail,
        )?;
        Ok(full_path)
    }

    /// Auto-convert entity query values to correct dtype.
    pub fn sanitize_query_dtypes(&self, entities: &mut Entities) {
        // Look up entity definitions from config
        for config in &self.configs {
            for ent_def in &config.entities {
                if let Some(val) = entities.get(&ent_def.name) {
                    let val_str = val.as_str_lossy();
                    let coerced = ent_def.coerce_value(&val_str);
                    entities.insert(ent_def.name.clone(), coerced);
                }
            }
        }
    }

    /// Get file associations from the database.
    pub fn get_associations(&self, path: &str, kind: Option<&str>) -> Result<Vec<BidsFile>> {
        let assocs = self.db.get_associations(path, kind)?;
        let mut files = Vec::new();
        for (dst, _kind) in &assocs {
            if let Ok(bf) = self.reconstruct_file(dst) {
                files.push(bf);
            }
        }
        Ok(files)
    }
}

impl std::fmt::Display for BidsLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let root_str = self.root.to_string_lossy();
        let root_display = if root_str.len() > 30 {
            format!("...{}", &root_str[root_str.len() - 30..])
        } else {
            root_str.to_string()
        };
        let n_subjects = self.get_subjects().map(|s| s.len()).unwrap_or(0);
        let n_sessions = self.get_sessions().map(|s| s.len()).unwrap_or(0);
        let n_runs = self.get_runs().map(|s| s.len()).unwrap_or(0);
        write!(f, "BIDS Layout: {root_display} | Subjects: {n_subjects} | Sessions: {n_sessions} | Runs: {n_runs}")
    }
}

// ──────────────────────────────── LayoutBuilder ────────────────────────────────

/// Builder for configuring and constructing a [`BidsLayout`].
///
/// Provides fine-grained control over dataset indexing, including validation,
/// derivative paths, configuration files, database persistence, ignore/force
/// patterns, and metadata indexing.
///
/// # Example
///
/// ```no_run
/// # use bids_layout::BidsLayout;
/// let layout = BidsLayout::builder("/path/to/dataset")
///     .validate(true)
///     .index_metadata(true)
///     .database_path("/tmp/bids_index.sqlite")
///     .add_derivative("/path/to/derivatives/fmriprep")
///     .build()
///     .unwrap();
/// ```
pub struct LayoutBuilder {
    root: PathBuf,
    validate: bool,
    derivatives: Option<Vec<PathBuf>>,
    configs: Vec<String>,
    regex_search: bool,
    database_path: Option<PathBuf>,
    is_derivative: bool,
    index_metadata: bool,
    ignore: Option<Vec<regex::Regex>>,
    force_index: Option<Vec<regex::Regex>>,
    config_filename: String,
}

impl LayoutBuilder {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            validate: true,
            derivatives: None,
            configs: vec!["bids".to_string()],
            regex_search: false,
            database_path: None,
            is_derivative: false,
            index_metadata: true,
            ignore: None,
            force_index: None,
            config_filename: "layout_config.json".to_string(),
        }
    }

    #[must_use] pub fn validate(mut self, v: bool) -> Self { self.validate = v; self }
    #[must_use] pub fn derivatives(mut self, paths: Vec<PathBuf>) -> Self { self.derivatives = Some(paths); self }
    #[must_use] pub fn add_derivative(mut self, path: impl AsRef<Path>) -> Self {
        self.derivatives.get_or_insert_with(Vec::new).push(path.as_ref().to_path_buf());
        self
    }
    #[must_use] pub fn config(mut self, configs: Vec<String>) -> Self { self.configs = configs; self }
    #[must_use] pub fn regex_search(mut self, v: bool) -> Self { self.regex_search = v; self }
    #[must_use] pub fn database_path(mut self, path: impl AsRef<Path>) -> Self {
        self.database_path = Some(path.as_ref().to_path_buf()); self
    }
    #[must_use] pub fn is_derivative(mut self, v: bool) -> Self { self.is_derivative = v; self }
    #[must_use] pub fn index_metadata(mut self, v: bool) -> Self { self.index_metadata = v; self }
    #[must_use] pub fn ignore(mut self, patterns: Vec<regex::Regex>) -> Self { self.ignore = Some(patterns); self }
    #[must_use] pub fn force_index(mut self, patterns: Vec<regex::Regex>) -> Self { self.force_index = Some(patterns); self }
    #[must_use] pub fn config_filename(mut self, name: &str) -> Self { self.config_filename = name.to_string(); self }

    pub fn build(self) -> Result<BidsLayout> {
        let (root, description) = bids_validate::validate_root(&self.root, self.validate)?;

        let is_derivative = self.is_derivative
            || description.as_ref().is_some_and(bids_core::DatasetDescription::is_derivative);
        let source_pipeline = if is_derivative {
            bids_validate::validate_derivative_path(&root).ok()
        } else {
            None
        };

        let default_configs = if is_derivative {
            vec!["bids".to_string(), "derivatives".to_string()]
        } else {
            vec!["bids".to_string()]
        };
        let config_names = if self.configs.is_empty() { default_configs } else { self.configs };
        let configs: Vec<Config> = config_names.iter()
            .filter_map(|name| Config::load(name).ok())
            .collect();

        let (ignore, force_index) = bids_validate::validate_indexing_args(
            self.ignore, self.force_index, &root)?;

        let db = match &self.database_path {
            Some(path) if Database::exists(path) => Database::open(path)?,
            db_path => {
                let db = match db_path {
                    Some(path) => Database::open(path)?,
                    None => Database::in_memory()?,
                };
                let options = IndexerOptions {
                    validate: self.validate && !is_derivative,
                    index_metadata: self.index_metadata,
                    ignore,
                    force_index,
                    config_filename: self.config_filename.clone(),
                };
                indexer::index_dataset(&root, &db, &configs, &options)?;
                db
            }
        };

        db.set_layout_info(&root.to_string_lossy(), &config_names.join(","))?;

        let spec_compatibility = description.as_ref().map(|d| {
            let schema = bids_schema::BidsSchema::load();
            schema.check_dataset_version(&d.bids_version)
        });

        // Warn (via eprintln) if the dataset uses a newer spec than we support.
        // This is non-fatal — the library will still try its best.
        if let Some(compat) = &spec_compatibility {
            if compat.has_warnings() {
                log::warn!("{compat}");
            }
        }

        let mut layout = BidsLayout {
            root, db, description, is_derivative, source_pipeline,
            derivatives: HashMap::new(), configs, regex_search: self.regex_search,
            spec_compatibility,
        };

        if let Some(deriv_paths) = self.derivatives {
            for path in deriv_paths { layout.add_derivatives(path)?; }
        }

        Ok(layout)
    }
}

