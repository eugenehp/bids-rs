//! The BIDS file index: files, entity tags, metadata and file associations,
//! in memory, optionally persisted to a file. Pure Rust (it was DuckDB, C++).
//!
//! Writes follow the SQL the index used to run: files and tags are keyed
//! (`path`; `(file_path, entity_name)`) with last-write-wins, associations
//! (`(src, dst, kind)`) with first-write-wins, and a replaced row keeps its
//! place in table order. Explicit transactions roll back to the tables as they
//! were at `BEGIN`. Reads are answered from a snapshot of hash maps (see
//! [`Snapshot`]) built after the last write.
//!
//! A database opened on a path ([`Database::open`]) writes the whole index to
//! it on every commit and every write outside a transaction: a small,
//! versioned JSON document, replaced atomically (written beside, then
//! renamed). A file from the DuckDB-based bids-rs (0.0.4 and before) is not
//! read: [`Database::exists`] says no, so a layout builder re-indexes into it.

use bids_core::error::{BidsError, Result};
use bids_core::file::BidsFile;
use indexmap::{IndexMap, IndexSet};
use std::cell::{Cell, Ref, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

/// What identifies an index file, and its layout version.
const FORMAT: &str = "bids-rs-index";
const VERSION: u64 = 1;

/// Manages the index backing a [`BidsLayout`](crate::BidsLayout).
///
/// Stores the complete file index for a BIDS dataset, including:
/// - **Files** — Path, filename, directory, file type for every indexed file
/// - **Tags** — Entity-value pairs (subject=01, task=rest, etc.) and metadata
///   key-value pairs from JSON sidecars
/// - **Associations** — Relationships between files (IntendedFor, Metadata
///   inheritance parent/child, InformedBy)
/// - **Layout info** — Root path and config names for database reloading
///
/// The index lives in memory (fast, ephemeral) or is also kept in a file (for
/// caching large dataset indices). Regex filters use the `regex` crate with
/// partial-match semantics (`Regex::is_match`).
///
/// # Tables
///
/// ```text
/// files        (path PRIMARY KEY, filename, dirname, is_dir, file_type)
/// tags         (file_path, entity_name, value, dtype, is_metadata; PRIMARY KEY (file_path, entity_name))
/// associations (src, dst, kind; PRIMARY KEY (src, dst, kind))
/// layout_info  (root PRIMARY KEY, config, derivatives)
/// ```
pub struct Database {
    tables: RefCell<Tables>,
    /// The tables as they were at `begin_transaction`, while one is open.
    rollback: RefCell<Option<Tables>>,
    /// Where the index is kept, for a database opened on a file.
    path: Option<PathBuf>,
    in_tx: Cell<bool>,
    /// Read cache, built on the first read after a write and dropped by
    /// every write.
    snapshot: RefCell<Option<Snapshot>>,
}

/// The four tables, each in table (insertion) order.
#[derive(Default, Clone)]
struct Tables {
    /// path → (filename, dirname, is_dir, file_type)
    files: IndexMap<String, (String, String, bool, String)>,
    /// (file_path, entity_name) → (value, dtype, is_metadata)
    tags: IndexMap<(String, String), (String, String, bool)>,
    /// (src, dst, kind)
    associations: IndexSet<(String, String, String)>,
    /// root → (config, derivatives)
    layout_info: IndexMap<String, (Option<String>, Option<String>)>,
}

/// The index file: a format tag and version, then each table as its rows.
#[derive(serde::Serialize, serde::Deserialize)]
struct IndexFile {
    format: String,
    version: u64,
    /// `[path, filename, dirname, is_dir, file_type]`
    files: Vec<(String, String, String, bool, String)>,
    /// `[file_path, entity_name, value, dtype, is_metadata]`
    tags: Vec<(String, String, String, String, bool)>,
    /// `[src, dst, kind]`
    associations: Vec<(String, String, String)>,
    /// `[root, config, derivatives]`
    layout_info: Vec<(String, Option<String>, Option<String>)>,
}

impl IndexFile {
    fn of(t: &Tables) -> Self {
        IndexFile {
            format: FORMAT.into(),
            version: VERSION,
            files: t
                .files
                .iter()
                .map(|(p, (f, d, is_dir, ty))| {
                    (p.clone(), f.clone(), d.clone(), *is_dir, ty.clone())
                })
                .collect(),
            tags: t
                .tags
                .iter()
                .map(|((p, e), (v, dt, m))| (p.clone(), e.clone(), v.clone(), dt.clone(), *m))
                .collect(),
            associations: t.associations.iter().cloned().collect(),
            layout_info: t
                .layout_info
                .iter()
                .map(|(r, (c, d))| (r.clone(), c.clone(), d.clone()))
                .collect(),
        }
    }

    /// The rows back into tables (a repeated key: the last row wins, at the
    /// first one's place, as the inserts that made them).
    fn into_tables(self) -> Tables {
        let mut t = Tables::default();
        for (p, f, d, is_dir, ty) in self.files {
            t.files.insert(p, (f, d, is_dir, ty));
        }
        for (p, e, v, dt, m) in self.tags {
            t.tags.insert((p, e), (v, dt, m));
        }
        t.associations.extend(self.associations);
        for (r, c, d) in self.layout_info {
            t.layout_info.insert(r, (c, d));
        }
        t
    }
}

/// An in-memory view of the index that answers every read: the tables
/// regrouped by file, plus the per-entity distinct values.
///
/// The snapshot is rebuilt lazily: writes only invalidate it, so indexing
/// (many writes, then bulk reads) builds it once.
#[derive(Default)]
struct Snapshot {
    /// Non-directory file paths, sorted.
    files: Vec<String>,
    /// file path → its tags, in table order.
    tags: HashMap<String, Vec<TagRow>>,
    /// entity name → distinct values, sorted. Keys are the entity names.
    unique_values: BTreeMap<String, Vec<String>>,
    /// src → `(dst, kind)`.
    associations: HashMap<String, Vec<(String, String)>>,
}

impl Database {
    /// Create a new in-memory database.
    pub fn in_memory() -> Result<Self> {
        Ok(Self::wrap(Tables::default(), None))
    }

    /// Open the index kept at `path`, or start one there if there is none
    /// (or only a DuckDB file from an earlier bids-rs, which is replaced).
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created, or the file is an
    /// index of this format that cannot be read.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let existing = Self::exists(path);
        let tables = if existing {
            read_index(path)?
        } else {
            Tables::default()
        };
        let db = Self::wrap(tables, Some(path.to_path_buf()));
        // a new index is written at once (so `exists` sees it); an existing
        // one already holds exactly these tables, so it is not rewritten
        if !existing {
            db.persist()?;
        }
        Ok(db)
    }

    fn wrap(tables: Tables, path: Option<PathBuf>) -> Self {
        Self {
            tables: RefCell::new(tables),
            rollback: RefCell::new(None),
            path,
            in_tx: Cell::new(false),
            snapshot: RefCell::new(None),
        }
    }

    /// Drop the read cache. Every write path calls this.
    fn invalidate(&self) {
        *self.snapshot.borrow_mut() = None;
    }

    /// After a write: drop the read cache, and outside a transaction keep the
    /// file up to date.
    fn written(&self) -> Result<()> {
        self.invalidate();
        if self.in_tx.get() {
            return Ok(());
        }
        self.persist()
    }

    /// Write the committed tables to the database's file, if it has one.
    fn persist(&self) -> Result<()> {
        match &self.path {
            Some(p) => write_index(p, &self.tables.borrow()),
            None => Ok(()),
        }
    }

    /// The read cache, building it from the tables if a write dropped it.
    fn snapshot(&self) -> Result<Ref<'_, Snapshot>> {
        if self.snapshot.borrow().is_none() {
            let built = self.build_snapshot();
            *self.snapshot.borrow_mut() = Some(built);
        }
        Ok(Ref::map(self.snapshot.borrow(), |s| {
            s.as_ref().expect("snapshot was just built")
        }))
    }

    fn scan_file_paths(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .tables
            .borrow()
            .files
            .iter()
            .filter(|(_, (_, _, is_dir, _))| !is_dir)
            .map(|(p, _)| p.clone())
            .collect();
        v.sort_unstable();
        v
    }

    fn scan_tags(&self) -> HashMap<String, Vec<TagRow>> {
        let mut out: HashMap<String, Vec<TagRow>> = HashMap::new();
        for ((file_path, entity), (value, dtype, is_metadata)) in &self.tables.borrow().tags {
            let row = (entity.clone(), value.clone(), dtype.clone(), *is_metadata);
            // the path is cloned once per file, not once per tag
            match out.get_mut(file_path.as_str()) {
                Some(rows) => rows.push(row),
                None => {
                    out.insert(file_path.clone(), vec![row]);
                }
            }
        }
        out
    }

    fn build_snapshot(&self) -> Snapshot {
        let files = self.scan_file_paths();
        let tags = self.scan_tags();

        // borrowed while collecting: one allocation per distinct name and value
        let mut distinct: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for (name, value, _, _) in tags.values().flatten() {
            distinct.entry(name).or_default().insert(value);
        }
        let unique_values = distinct
            .into_iter()
            .map(|(name, values)| {
                (
                    name.to_string(),
                    values.into_iter().map(str::to_string).collect(),
                )
            })
            .collect();

        let mut associations: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for (src, dst, kind) in &self.tables.borrow().associations {
            let row = (dst.clone(), kind.clone());
            match associations.get_mut(src.as_str()) {
                Some(rows) => rows.push(row),
                None => {
                    associations.insert(src.clone(), vec![row]);
                }
            }
        }

        Snapshot {
            files,
            tags,
            unique_values,
            associations,
        }
    }

    /// Whether `path` holds an index this version can read.
    #[must_use]
    pub fn exists(path: &Path) -> bool {
        let Ok(mut f) = std::fs::File::open(path) else {
            return false;
        };
        let mut head = [0u8; 64];
        let n = std::io::Read::read(&mut f, &mut head).unwrap_or(0);
        let head = String::from_utf8_lossy(&head[..n]);
        head.starts_with('{') && head.contains(FORMAT)
    }

    /// The conventional index file inside a database directory.
    ///
    /// Creates `dir` if it does not exist and returns `dir/layout_index.json`.
    /// Passing `None` returns `None`, matching PyBIDS' `get_database_file`.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn get_database_file(dir: Option<&Path>) -> Result<Option<std::path::PathBuf>> {
        let Some(dir) = dir else { return Ok(None) };
        std::fs::create_dir_all(dir)?;
        Ok(Some(dir.join("layout_index.json")))
    }

    /// Empty every table, discarding all indexed data.
    ///
    /// Corresponds to PyBIDS' `ConnectionManager.reset_database`.
    ///
    /// # Errors
    ///
    /// Returns an error if the emptied index cannot be written to its file.
    pub fn reset_database(&self) -> Result<()> {
        *self.tables.borrow_mut() = Tables::default();
        self.written()
    }

    /// Discard any uncommitted work and start clean.
    ///
    /// The analogue of PyBIDS' `ConnectionManager.reset_session`: there is no
    /// session object to replace, so this rolls back an in-flight transaction
    /// if one is open. Succeeds when there is nothing to roll back.
    ///
    /// # Errors
    ///
    /// Never, at present; kept fallible for the API.
    pub fn reset_session(&self) -> Result<()> {
        if self.in_tx.get() {
            self.rollback_transaction()?;
        }
        self.invalidate();
        Ok(())
    }

    /// Begin an explicit transaction for bulk operations: writes until
    /// [`commit_transaction`](Self::commit_transaction) can be undone with
    /// [`rollback_transaction`](Self::rollback_transaction), and reach the
    /// file only at the commit. Reads inside the transaction see them.
    ///
    /// # Errors
    ///
    /// Returns an error if a transaction is already open.
    pub fn begin_transaction(&self) -> Result<()> {
        if self.in_tx.get() {
            return Err(BidsError::Database(
                "cannot start a transaction within a transaction".into(),
            ));
        }
        *self.rollback.borrow_mut() = Some(self.tables.borrow().clone());
        self.in_tx.set(true);
        Ok(())
    }

    /// Commit the current transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is open, or the index cannot be
    /// written to its file (the transaction is then rolled back).
    pub fn commit_transaction(&self) -> Result<()> {
        if !self.in_tx.get() {
            return Err(BidsError::Database(
                "cannot commit - no transaction is active".into(),
            ));
        }
        if let Err(e) = self.persist() {
            let _ = self.rollback_transaction();
            return Err(e);
        }
        self.in_tx.set(false);
        *self.rollback.borrow_mut() = None;
        Ok(())
    }

    /// Roll back the current transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if no transaction is open.
    pub fn rollback_transaction(&self) -> Result<()> {
        let saved = self.rollback.borrow_mut().take();
        self.in_tx.set(false);
        // A snapshot built mid-transaction holds rows the rollback discards.
        self.invalidate();
        match saved {
            Some(t) => {
                *self.tables.borrow_mut() = t;
                Ok(())
            }
            None => Err(BidsError::Database(
                "cannot rollback - no transaction is active".into(),
            )),
        }
    }

    /// Insert a file into the database (replacing the row for its path).
    pub fn insert_file(&self, file: &BidsFile) -> Result<()> {
        self.tables.borrow_mut().files.insert(
            file.path.to_string_lossy().into_owned(),
            (
                file.filename.clone(),
                file.dirname.to_string_lossy().into_owned(),
                file.is_dir,
                format!("{:?}", file.file_type),
            ),
        );
        self.written()
    }

    /// Insert a tag (entity-value pair for a file), replacing the file's
    /// previous value for that entity.
    pub fn insert_tag(
        &self,
        file_path: &str,
        entity_name: &str,
        value: &str,
        dtype: &str,
        is_metadata: bool,
    ) -> Result<()> {
        self.tables.borrow_mut().tags.insert(
            (file_path.to_string(), entity_name.to_string()),
            (value.to_string(), dtype.to_string(), is_metadata),
        );
        self.written()
    }

    /// Insert a file association (a repeat of an existing one is ignored).
    pub fn insert_association(&self, src: &str, dst: &str, kind: &str) -> Result<()> {
        self.tables.borrow_mut().associations.insert((
            src.to_string(),
            dst.to_string(),
            kind.to_string(),
        ));
        self.written()
    }

    /// Store layout info.
    pub fn set_layout_info(&self, root: &str, config: &str) -> Result<()> {
        // `INSERT OR REPLACE` of (root, config): the row's `derivatives` resets
        self.tables
            .borrow_mut()
            .layout_info
            .insert(root.to_string(), (Some(config.to_string()), None));
        self.written()
    }

    /// Get layout info (root, config).
    pub fn get_layout_info(&self) -> Result<Option<(String, String)>> {
        Ok(self
            .tables
            .borrow()
            .layout_info
            .iter()
            .find_map(|(root, (config, _))| config.as_ref().map(|c| (root.clone(), c.clone()))))
    }

    /// Query all file paths, sorted.
    pub fn all_file_paths(&self) -> Result<Vec<String>> {
        if self.snapshot.borrow().is_some() {
            return Ok(self.snapshot()?.files.clone());
        }
        Ok(self.scan_file_paths())
    }

    /// Get tags for a specific file.
    pub fn get_tags(&self, file_path: &str) -> Result<Vec<(String, String, String, bool)>> {
        Ok(self
            .snapshot()?
            .tags
            .get(file_path)
            .cloned()
            .unwrap_or_default())
    }

    /// Tags for every file, keyed by path.
    pub fn all_tags(&self) -> Result<HashMap<String, Vec<TagRow>>> {
        if self.snapshot.borrow().is_some() {
            return Ok(self.snapshot()?.tags.clone());
        }
        Ok(self.scan_tags())
    }

    /// Tags for the given files, keyed by path.
    ///
    /// Files with no tags are absent from the map.
    pub fn get_tags_many(&self, paths: &[String]) -> Result<HashMap<String, Vec<TagRow>>> {
        let snap = self.snapshot()?;
        Ok(paths
            .iter()
            .filter_map(|p| snap.tags.get(p).map(|t| (p.clone(), t.clone())))
            .collect())
    }

    /// Get all unique values for a given entity, sorted.
    pub fn get_unique_entity_values(&self, entity_name: &str) -> Result<Vec<String>> {
        Ok(self
            .snapshot()?
            .unique_values
            .get(entity_name)
            .cloned()
            .unwrap_or_default())
    }

    /// Get all unique entity names, sorted.
    pub fn get_entity_names(&self) -> Result<Vec<String>> {
        Ok(self.snapshot()?.unique_values.keys().cloned().collect())
    }

    /// Query files with advanced filtering including Query::None/Any and regex.
    ///
    /// Filter types:
    /// - Normal: (entity, values, false) — entity must match one of values
    /// - Regex: `(entity, [pattern], true)` — entity value must contain a
    ///   match for the pattern (`Regex::is_match`)
    /// - Query::None: (entity, ["__NONE__"], false) — entity must NOT exist
    /// - Query::Any: (entity, ["__ANY__"], false) — entity must exist (any value)
    ///
    /// Results are sorted by path.
    ///
    /// # Errors
    ///
    /// Returns [`BidsError::InvalidFilter`] if a regex pattern does not compile.
    pub fn query_files(&self, filters: &[(String, Vec<String>, bool)]) -> Result<Vec<String>> {
        enum Cond<'a> {
            Absent,
            Present,
            Regex(regex::Regex),
            Equals(&'a str),
            OneOf(&'a [String]),
        }

        let mut conds: Vec<(&str, Cond<'_>)> = Vec::with_capacity(filters.len());
        for (entity_name, values, is_regex) in filters {
            let cond = match values.as_slice() {
                [v] if v == "__NONE__" => Cond::Absent,
                [v] if v == "__ANY__" => Cond::Present,
                [v] if v == "__OPTIONAL__" => continue,
                [] => continue,
                [pattern, ..] if *is_regex => {
                    Cond::Regex(regex::Regex::new(pattern).map_err(|e| {
                        BidsError::InvalidFilter(format!("invalid regex {pattern:?}: {e}"))
                    })?)
                }
                [v] => Cond::Equals(v),
                vs => Cond::OneOf(vs),
            };
            conds.push((entity_name, cond));
        }

        let snap = self.snapshot()?;
        let no_tags = Vec::new();
        Ok(snap
            .files
            .iter()
            .filter(|path| {
                let tags = snap.tags.get(*path).unwrap_or(&no_tags);
                conds.iter().all(|(entity, cond)| {
                    let value = tags
                        .iter()
                        .find(|(n, _, _, _)| n == entity)
                        .map(|(_, v, _, _)| v.as_str());
                    match (cond, value) {
                        (Cond::Absent, v) => v.is_none(),
                        (_, None) => false,
                        (Cond::Present, Some(_)) => true,
                        (Cond::Regex(re), Some(v)) => re.is_match(v),
                        (Cond::Equals(want), Some(v)) => v == *want,
                        (Cond::OneOf(want), Some(v)) => want.iter().any(|w| w == v),
                    }
                })
            })
            .cloned()
            .collect())
    }

    /// Get distinct directories for files matching filters and having a target entity.
    pub fn query_directories(
        &self,
        target_entity: &str,
        filters: &[(String, Vec<String>, bool)],
    ) -> Result<Vec<String>> {
        let paths = self.query_files(filters)?;
        let snap = self.snapshot()?;
        let mut dirs = BTreeSet::new();
        for path_str in &paths {
            // Check if the file has the target entity
            let has_target = snap
                .tags
                .get(path_str)
                .is_some_and(|tags| tags.iter().any(|(n, _, _, _)| n == target_entity));
            if has_target && let Some(parent) = Path::new(path_str).parent() {
                dirs.insert(parent.to_string_lossy().to_string());
            }
        }
        Ok(dirs.into_iter().collect())
    }

    /// Get associated files for a given source file.
    pub fn get_associations(&self, src: &str, kind: Option<&str>) -> Result<Vec<(String, String)>> {
        let snap = self.snapshot()?;
        Ok(snap
            .associations
            .get(src)
            .into_iter()
            .flatten()
            .filter(|(_, k)| kind.is_none_or(|want| k == want))
            .cloned()
            .collect())
    }

    /// Get the total number of indexed files.
    pub fn file_count(&self) -> Result<usize> {
        if let Some(snap) = self.snapshot.borrow().as_ref() {
            return Ok(snap.files.len());
        }
        Ok(self
            .tables
            .borrow()
            .files
            .values()
            .filter(|(_, _, is_dir, _)| !is_dir)
            .count())
    }

    /// Save the current index to a file (replacing any file at `path`), for
    /// [`Database::open`] / `BidsLayout::load`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        write_index(path, &self.tables.borrow())
    }
}

/// One tag: `(entity_name, value, dtype, is_metadata)`.
pub type TagRow = (String, String, String, bool);

fn read_index(path: &Path) -> Result<Tables> {
    let bytes = std::fs::read(path)?;
    let file: IndexFile = serde_json::from_slice(&bytes).map_err(|e| {
        BidsError::Database(format!("{}: not a readable index: {e}", path.display()))
    })?;
    if file.format != FORMAT || file.version != VERSION {
        return Err(BidsError::Database(format!(
            "{}: index format {:?} version {} (this bids-rs reads {FORMAT:?} version {VERSION})",
            path.display(),
            file.format,
            file.version
        )));
    }
    Ok(file.into_tables())
}

/// Write `tables` to `path` atomically: a sibling temporary file, then a
/// rename over `path`, so a reader never sees half an index.
fn write_index(path: &Path, tables: &Tables) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = serde_json::to_vec(&IndexFile::of(tables))
        .map_err(|e| BidsError::Database(e.to_string()))?;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bids_core::file::BidsFile;

    #[test]
    fn test_database_operations() {
        let db = Database::in_memory().unwrap();
        let file = BidsFile::new("/data/sub-01/eeg/sub-01_task-rest_eeg.edf");
        db.insert_file(&file).unwrap();
        db.insert_tag(
            "/data/sub-01/eeg/sub-01_task-rest_eeg.edf",
            "subject",
            "01",
            "str",
            false,
        )
        .unwrap();
        db.insert_tag(
            "/data/sub-01/eeg/sub-01_task-rest_eeg.edf",
            "task",
            "rest",
            "str",
            false,
        )
        .unwrap();

        let paths = db.all_file_paths().unwrap();
        assert_eq!(paths.len(), 1);

        let subjects = db.get_unique_entity_values("subject").unwrap();
        assert_eq!(subjects, vec!["01"]);

        let filters = vec![("subject".to_string(), vec!["01".to_string()], false)];
        let results = db.query_files(&filters).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_none_any() {
        let db = Database::in_memory().unwrap();
        let f1 = BidsFile::new("/data/sub-01_task-rest_eeg.edf");
        let f2 = BidsFile::new("/data/sub-02_eeg.edf");
        db.insert_file(&f1).unwrap();
        db.insert_file(&f2).unwrap();
        db.insert_tag(
            "/data/sub-01_task-rest_eeg.edf",
            "subject",
            "01",
            "str",
            false,
        )
        .unwrap();
        db.insert_tag(
            "/data/sub-01_task-rest_eeg.edf",
            "task",
            "rest",
            "str",
            false,
        )
        .unwrap();
        db.insert_tag("/data/sub-02_eeg.edf", "subject", "02", "str", false)
            .unwrap();

        // Query::Any — task must exist
        let filters = vec![("task".to_string(), vec!["__ANY__".to_string()], false)];
        let results = db.query_files(&filters).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("sub-01"));

        // Query::None — task must NOT exist
        let filters = vec![("task".to_string(), vec!["__NONE__".to_string()], false)];
        let results = db.query_files(&filters).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].contains("sub-02"));
    }

    #[test]
    fn test_transactions() {
        let db = Database::in_memory().unwrap();
        db.begin_transaction().unwrap();

        for i in 0..100 {
            let path = format!("/data/sub-{:03}_eeg.edf", i);
            let f = BidsFile::new(&path);
            db.insert_file(&f).unwrap();
            db.insert_tag(&path, "subject", &format!("{:03}", i), "str", false)
                .unwrap();
        }

        db.commit_transaction().unwrap();

        assert_eq!(db.file_count().unwrap(), 100);
        let subjects = db.get_unique_entity_values("subject").unwrap();
        assert_eq!(subjects.len(), 100);
    }

    #[test]
    fn test_transaction_rollback() {
        let db = Database::in_memory().unwrap();

        // Insert one file outside transaction
        let f = BidsFile::new("/data/sub-01_eeg.edf");
        db.insert_file(&f).unwrap();
        assert_eq!(db.file_count().unwrap(), 1);

        // Start transaction, insert more, rollback
        db.begin_transaction().unwrap();
        let f2 = BidsFile::new("/data/sub-02_eeg.edf");
        db.insert_file(&f2).unwrap();
        db.rollback_transaction().unwrap();

        // Should only have the first file
        assert_eq!(db.file_count().unwrap(), 1);
    }

    #[test]
    fn test_regexp_query() {
        let db = Database::in_memory().unwrap();
        let f1 = BidsFile::new("/data/sub-01_eeg.edf");
        let f2 = BidsFile::new("/data/sub-02_eeg.edf");
        db.insert_file(&f1).unwrap();
        db.insert_file(&f2).unwrap();
        db.insert_tag("/data/sub-01_eeg.edf", "subject", "01", "str", false)
            .unwrap();
        db.insert_tag("/data/sub-02_eeg.edf", "subject", "02", "str", false)
            .unwrap();

        let filters = vec![("subject".to_string(), vec!["0[12]".to_string()], true)];
        let results = db.query_files(&filters).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_buffered_transaction_semantics() {
        let db = Database::in_memory().unwrap();
        db.insert_file(&BidsFile::new("/d/a.edf")).unwrap();
        db.insert_tag("/d/a.edf", "subject", "01", "str", false)
            .unwrap();

        db.begin_transaction().unwrap();
        db.insert_file(&BidsFile::new("/d/b.edf")).unwrap();
        // Replaces a row committed before the transaction…
        db.insert_tag("/d/a.edf", "subject", "99", "str", false)
            .unwrap();
        // …and the last of two buffered writes to the same key wins.
        db.insert_tag("/d/b.edf", "subject", "02", "str", false)
            .unwrap();
        db.insert_tag("/d/b.edf", "subject", "03", "str", false)
            .unwrap();
        db.insert_association("/d/a.edf", "/d/b.edf", "Child")
            .unwrap();
        db.insert_association("/d/a.edf", "/d/b.edf", "Child")
            .unwrap();
        // Reads inside the transaction see buffered rows.
        assert_eq!(db.file_count().unwrap(), 2);
        db.insert_file(&BidsFile::new("/d/c.edf")).unwrap();
        db.commit_transaction().unwrap();

        assert_eq!(db.file_count().unwrap(), 3);
        assert_eq!(
            db.get_unique_entity_values("subject").unwrap(),
            ["03", "99"]
        );
        assert_eq!(db.get_associations("/d/a.edf", None).unwrap().len(), 1);

        db.begin_transaction().unwrap();
        db.insert_file(&BidsFile::new("/d/z.edf")).unwrap();
        db.rollback_transaction().unwrap();
        assert_eq!(db.file_count().unwrap(), 3);
    }

    #[test]
    fn test_snapshot_sees_later_writes() {
        let db = Database::in_memory().unwrap();
        db.insert_file(&BidsFile::new("/d/a.edf")).unwrap();
        db.insert_tag("/d/a.edf", "subject", "01", "str", false)
            .unwrap();
        // Builds the snapshot…
        assert_eq!(db.get_unique_entity_values("subject").unwrap(), ["01"]);

        // …which every kind of write must drop.
        db.insert_file(&BidsFile::new("/d/b.edf")).unwrap();
        db.insert_tag("/d/b.edf", "subject", "02", "str", false)
            .unwrap();
        db.insert_association("/d/a.edf", "/d/b.edf", "Child")
            .unwrap();
        assert_eq!(db.file_count().unwrap(), 2);
        assert_eq!(
            db.get_unique_entity_values("subject").unwrap(),
            ["01", "02"]
        );
        assert_eq!(
            db.get_associations("/d/a.edf", Some("Child")).unwrap(),
            [("/d/b.edf".to_string(), "Child".to_string())]
        );
        assert!(
            db.get_associations("/d/a.edf", Some("Parent"))
                .unwrap()
                .is_empty()
        );

        // A snapshot built inside a transaction must not survive its rollback.
        db.begin_transaction().unwrap();
        db.insert_tag("/d/a.edf", "task", "rest", "str", false)
            .unwrap();
        assert_eq!(db.get_entity_names().unwrap(), ["subject", "task"]);
        db.rollback_transaction().unwrap();
        assert_eq!(db.get_entity_names().unwrap(), ["subject"]);

        db.reset_database().unwrap();
        assert_eq!(db.file_count().unwrap(), 0);
        assert!(db.get_tags("/d/a.edf").unwrap().is_empty());
    }

    #[test]
    fn test_query_files_regex_and_lists() {
        let db = Database::in_memory().unwrap();
        for (path, sub) in [("/d/c.edf", "10"), ("/d/a.edf", "01"), ("/d/b.edf", "02")] {
            db.insert_file(&BidsFile::new(path)).unwrap();
            db.insert_tag(path, "subject", sub, "str", false).unwrap();
        }
        let q = |entity: &str, values: &[&str], regex: bool| {
            db.query_files(&[(
                entity.to_string(),
                values.iter().map(ToString::to_string).collect(),
                regex,
            )])
        };
        // Partial match, like `re.search`; results sorted by path.
        assert_eq!(
            q("subject", &["0"], true).unwrap(),
            ["/d/a.edf", "/d/b.edf", "/d/c.edf"]
        );
        assert_eq!(
            q("subject", &["^0"], true).unwrap(),
            ["/d/a.edf", "/d/b.edf"]
        );
        assert_eq!(
            q("subject", &["01", "10"], false).unwrap(),
            ["/d/a.edf", "/d/c.edf"]
        );
        assert!(q("subject", &["0"], false).unwrap().is_empty());
        assert_eq!(q("subject", &["__OPTIONAL__"], false).unwrap().len(), 3);
        assert!(matches!(
            q("subject", &["("], true),
            Err(BidsError::InvalidFilter(_))
        ));
    }

    #[test]
    fn test_save_and_reopen() {
        let dir = std::env::temp_dir().join(format!("bids_db_save_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("index.json");
        let db = Database::in_memory().unwrap();
        db.insert_file(&BidsFile::new("/d/a.edf")).unwrap();
        db.insert_tag("/d/a.edf", "subject", "01", "str", false)
            .unwrap();
        db.set_layout_info("/d", "bids").unwrap();
        db.save_to(&path).unwrap();
        // Saving over an existing file replaces it.
        db.save_to(&path).unwrap();
        drop(db);

        let back = Database::open(&path).unwrap();
        assert_eq!(back.file_count().unwrap(), 1);
        assert_eq!(
            back.get_layout_info().unwrap(),
            Some(("/d".into(), "bids".into()))
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Opening an existing index only reads it: one in a read-only place
    /// (a shipped dataset, a shared mount) loads, and its bytes are untouched.
    #[cfg(unix)]
    #[test]
    fn test_open_does_not_rewrite_an_existing_index() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("bids_db_ro_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("index.json");
        let db = Database::in_memory().unwrap();
        db.insert_file(&BidsFile::new("/d/a.edf")).unwrap();
        db.insert_tag("/d/a.edf", "subject", "01", "str", false)
            .unwrap();
        db.set_layout_info("/d", "bids").unwrap();
        db.save_to(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).unwrap();
        let opened = Database::open(&path);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        let back = opened.expect("a read-only index opens");
        assert_eq!(back.file_count().unwrap(), 1);
        assert_eq!(
            back.get_unique_entity_values("subject").unwrap(),
            vec!["01"]
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert!(!dir.join("index.json.tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
