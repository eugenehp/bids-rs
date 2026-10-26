//! DuckDB database backend for the BIDS file index.
//!
//! Stores files, entity tags, metadata, and file associations in a DuckDB
//! database (in-memory or on-disk). Bulk inserts run inside explicit
//! transactions; reads are answered from an in-memory snapshot (see
//! [`Snapshot`]).

use bids_core::error::{BidsError, Result};
use bids_core::file::BidsFile;
use duckdb::{Connection, params};
use indexmap::{IndexMap, IndexSet};
use std::cell::{Cell, Ref, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// Convert a `duckdb::Error` into a `BidsError::Database`.
fn db_err(e: duckdb::Error) -> BidsError {
    BidsError::Database(e.to_string())
}

/// Manages the DuckDB database backing a [`BidsLayout`](crate::BidsLayout) index.
///
/// Stores the complete file index for a BIDS dataset, including:
/// - **Files** — Path, filename, directory, file type for every indexed file
/// - **Tags** — Entity-value pairs (subject=01, task=rest, etc.) and metadata
///   key-value pairs from JSON sidecars
/// - **Associations** — Relationships between files (IntendedFor, Metadata
///   inheritance parent/child, InformedBy)
/// - **Layout info** — Root path and config names for database reloading
///
/// The database supports both in-memory operation (fast, ephemeral) and
/// on-disk persistence (for caching large dataset indices). Regex filters
/// use the `regex` crate with partial-match semantics (`Regex::is_match`).
///
/// # Schema
///
/// ```sql
/// CREATE TABLE files (path TEXT PRIMARY KEY, filename TEXT, dirname TEXT, ...);
/// CREATE TABLE tags (file_path TEXT, entity_name TEXT, value TEXT, ...);
/// CREATE TABLE associations (src TEXT, dst TEXT, kind TEXT, ...);
/// CREATE TABLE layout_info (root TEXT PRIMARY KEY, config TEXT, ...);
/// ```
pub struct Database {
    conn: Connection,
    /// True between `begin_transaction` and `commit`/`rollback`. While set,
    /// inserts are buffered in `pending` instead of hitting the database.
    in_tx: Cell<bool>,
    pending: RefCell<Pending>,
    /// Read cache, built on the first read after a write and dropped by
    /// every write.
    snapshot: RefCell<Option<Snapshot>>,
}

/// An in-memory copy of the index that answers every read.
///
/// DuckDB charges a fixed planning/execution cost per statement — about
/// 50–80 µs for a one-row point lookup and several hundred µs for a
/// `SELECT DISTINCT … ORDER BY`, against ~3 µs for SQLite. Queries such as
/// `get_subjects` or `get_metadata` are a single small lookup, so that floor
/// *is* their cost, and made them 10–80× slower than the SQLite backend.
/// Reading the tables once into hash maps turns them back into lookups.
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

/// Inserts buffered during a transaction.
///
/// DuckDB is a columnar engine: one `INSERT OR REPLACE` per row costs a full
/// statement execution and an index probe, which made indexing a dataset
/// orders of magnitude slower than SQLite. Buffering lets a transaction land
/// as one Appender load plus one set-based merge per table. Keys are
/// deduplicated here with the same last-write-wins (files, tags) and
/// first-write-wins (associations) semantics as the SQL they replace.
#[derive(Default)]
struct Pending {
    /// path → (filename, dirname, is_dir, file_type)
    files: IndexMap<String, (String, String, bool, String)>,
    /// (file_path, entity_name) → (value, dtype, is_metadata)
    tags: IndexMap<(String, String), (String, String, bool)>,
    /// (src, dst, kind)
    associations: IndexSet<(String, String, String)>,
}

impl Pending {
    fn is_empty(&self) -> bool {
        self.files.is_empty() && self.tags.is_empty() && self.associations.is_empty()
    }
}

impl Database {
    /// Create a new in-memory database.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(db_err)?;
        let db = Self::wrap(conn);
        db.create_tables()?;
        Ok(db)
    }

    /// Open or create a database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(db_err)?;
        let db = Self::wrap(conn);
        db.create_tables()?;
        Ok(db)
    }

    fn wrap(conn: Connection) -> Self {
        Self {
            conn,
            in_tx: Cell::new(false),
            pending: RefCell::new(Pending::default()),
            snapshot: RefCell::new(None),
        }
    }

    /// Drop the read cache. Every write path calls this.
    fn invalidate(&self) {
        *self.snapshot.borrow_mut() = None;
    }

    /// The read cache, building it from the tables if a write dropped it.
    fn snapshot(&self) -> Result<Ref<'_, Snapshot>> {
        self.flush_pending()?;
        if self.snapshot.borrow().is_none() {
            let built = self.build_snapshot()?;
            *self.snapshot.borrow_mut() = Some(built);
        }
        Ok(Ref::map(self.snapshot.borrow(), |s| {
            s.as_ref().expect("snapshot was just built")
        }))
    }

    /// True when a snapshot is cached and no buffered writes would drop it.
    fn snapshot_is_warm(&self) -> bool {
        self.snapshot.borrow().is_some() && self.pending.borrow().is_empty()
    }

    fn scan_file_paths(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM files WHERE NOT is_dir ORDER BY path")
            .map_err(db_err)?;
        stmt.query_map([], |row| row.get(0))
            .map_err(db_err)?
            .collect::<std::result::Result<Vec<String>, _>>()
            .map_err(db_err)
    }

    fn scan_tags(&self) -> Result<HashMap<String, Vec<TagRow>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT file_path, entity_name, value, dtype, is_metadata FROM tags")
            .map_err(db_err)?;
        collect_tag_rows(&mut stmt, [])
    }

    fn build_snapshot(&self) -> Result<Snapshot> {
        let files = self.scan_file_paths()?;
        let tags = self.scan_tags()?;

        let mut distinct: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (name, value, _, _) in tags.values().flatten() {
            distinct
                .entry(name.clone())
                .or_default()
                .insert(value.clone());
        }
        let unique_values = distinct
            .into_iter()
            .map(|(name, values)| (name, values.into_iter().collect()))
            .collect();

        let mut stmt = self
            .conn
            .prepare("SELECT src, dst, kind FROM associations")
            .map_err(db_err)?;
        let mut associations: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut rows = stmt.query([]).map_err(db_err)?;
        while let Some(row) = rows.next().map_err(db_err)? {
            associations
                .entry(row.get(0).map_err(db_err)?)
                .or_default()
                .push((row.get(1).map_err(db_err)?, row.get(2).map_err(db_err)?));
        }

        Ok(Snapshot {
            files,
            tags,
            unique_values,
            associations,
        })
    }

    /// Check if a database file exists.
    pub fn exists(path: &Path) -> bool {
        path.exists()
    }

    /// The conventional index file inside a database directory.
    ///
    /// Creates `dir` if it does not exist and returns `dir/layout_index.duckdb`.
    /// Passing `None` returns `None`, matching PyBIDS' `get_database_file`.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn get_database_file(dir: Option<&Path>) -> Result<Option<std::path::PathBuf>> {
        let Some(dir) = dir else { return Ok(None) };
        std::fs::create_dir_all(dir)?;
        Ok(Some(dir.join("layout_index.duckdb")))
    }

    /// Drop every table and recreate the schema, discarding all indexed data.
    ///
    /// Corresponds to PyBIDS' `ConnectionManager.reset_database`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tables cannot be dropped or recreated.
    pub fn reset_database(&self) -> Result<()> {
        self.invalidate();
        for table in ["tags", "associations", "layout_info", "files"] {
            self.conn
                .execute(&format!("DROP TABLE IF EXISTS {table}"), [])
                .map_err(db_err)?;
        }
        self.create_tables()
    }

    /// Discard any uncommitted work and start clean.
    ///
    /// The DuckDB analogue of PyBIDS' `ConnectionManager.reset_session`:
    /// there is no session object to replace, so this rolls back an in-flight
    /// transaction if one is open. Succeeds when there is nothing to roll back.
    ///
    /// # Errors
    ///
    /// Returns an error if the rollback itself fails for a reason other than
    /// there being no active transaction.
    pub fn reset_session(&self) -> Result<()> {
        *self.pending.borrow_mut() = Pending::default();
        self.in_tx.set(false);
        self.invalidate();
        if self.conn.is_autocommit() {
            return Ok(());
        }
        self.conn.execute_batch("ROLLBACK").map_err(db_err)
    }

    /// Create the schema. There are no secondary indexes: reads are served
    /// from the [`Snapshot`], so an index would only slow inserts down. The
    /// primary keys stay — `INSERT OR REPLACE` resolves conflicts on them.
    fn create_tables(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                dirname TEXT NOT NULL,
                is_dir BOOLEAN NOT NULL DEFAULT false,
                file_type TEXT NOT NULL DEFAULT 'Generic'
            );
            CREATE TABLE IF NOT EXISTS tags (
                file_path TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                value TEXT NOT NULL,
                dtype TEXT NOT NULL DEFAULT 'str',
                is_metadata BOOLEAN NOT NULL DEFAULT false,
                PRIMARY KEY (file_path, entity_name)
            );
            CREATE TABLE IF NOT EXISTS associations (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                kind TEXT NOT NULL,
                PRIMARY KEY (src, dst, kind)
            );
            CREATE TABLE IF NOT EXISTS layout_info (
                root TEXT PRIMARY KEY,
                config TEXT,
                derivatives TEXT
            );
            ",
            )
            .map_err(db_err)?;
        Ok(())
    }

    /// Begin an explicit transaction for bulk operations.
    ///
    /// Inserts made until [`commit_transaction`](Self::commit_transaction)
    /// are buffered and written in bulk, which is what makes indexing large
    /// datasets fast on DuckDB. Reads inside the transaction still see them.
    pub fn begin_transaction(&self) -> Result<()> {
        self.conn
            .execute_batch("BEGIN TRANSACTION")
            .map_err(db_err)?;
        self.in_tx.set(true);
        Ok(())
    }

    /// Commit the current transaction.
    pub fn commit_transaction(&self) -> Result<()> {
        if let Err(e) = self.flush_pending() {
            let _ = self.rollback_transaction();
            return Err(e);
        }
        self.in_tx.set(false);
        self.conn.execute_batch("COMMIT").map_err(db_err)?;
        Ok(())
    }

    /// Roll back the current transaction.
    pub fn rollback_transaction(&self) -> Result<()> {
        *self.pending.borrow_mut() = Pending::default();
        self.in_tx.set(false);
        // A snapshot built mid-transaction holds rows the rollback discards.
        self.invalidate();
        self.conn.execute_batch("ROLLBACK").map_err(db_err)?;
        Ok(())
    }

    /// Write buffered inserts, one bulk load per table.
    fn flush_pending(&self) -> Result<()> {
        let pending = std::mem::take(&mut *self.pending.borrow_mut());
        if pending.is_empty() {
            return Ok(());
        }
        self.invalidate();
        if !pending.files.is_empty() {
            self.bulk_load("files", "REPLACE", |app| {
                for (path, (filename, dirname, is_dir, file_type)) in &pending.files {
                    app.append_row(params![path, filename, dirname, is_dir, file_type])?;
                }
                Ok(())
            })?;
        }
        if !pending.tags.is_empty() {
            self.bulk_load("tags", "REPLACE", |app| {
                for ((file_path, entity_name), (value, dtype, is_metadata)) in &pending.tags {
                    app.append_row(params![file_path, entity_name, value, dtype, is_metadata])?;
                }
                Ok(())
            })?;
        }
        if !pending.associations.is_empty() {
            self.bulk_load("associations", "IGNORE", |app| {
                for (src, dst, kind) in &pending.associations {
                    app.append_row(params![src, dst, kind])?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Load rows into `table` through the Appender.
    ///
    /// `Pending` already deduplicates keys, so rows can only conflict with
    /// rows already in the table. An empty table — every first index — takes
    /// the rows directly. Otherwise they go through a temp staging table and
    /// one `INSERT OR <conflict> … SELECT`, which keeps the last-write-wins
    /// (`REPLACE`) or first-write-wins (`IGNORE`) semantics of row inserts.
    fn bulk_load(
        &self,
        table: &str,
        conflict: &str,
        append: impl FnOnce(&mut duckdb::Appender<'_>) -> duckdb::Result<()>,
    ) -> Result<()> {
        let empty: bool = self
            .conn
            .query_row(
                &format!("SELECT NOT EXISTS (SELECT 1 FROM {table})"),
                [],
                |r| r.get(0),
            )
            .map_err(db_err)?;
        let stage = format!("stage_{table}");
        if !empty {
            self.conn
                .execute_batch(&format!(
                    "CREATE TEMP TABLE IF NOT EXISTS {stage} AS SELECT * FROM {table} LIMIT 0;"
                ))
                .map_err(db_err)?;
        }
        let mut app = if empty {
            self.conn.appender(table)
        } else {
            self.conn.appender_to_catalog_and_db(&stage, "temp", "main")
        }
        .map_err(db_err)?;
        append(&mut app).map_err(db_err)?;
        app.flush().map_err(db_err)?;
        drop(app);
        if !empty {
            self.conn
                .execute_batch(&format!(
                    "INSERT OR {conflict} INTO {table} SELECT * FROM temp.{stage};
                     DELETE FROM temp.{stage};"
                ))
                .map_err(db_err)?;
        }
        Ok(())
    }

    /// Insert a file into the database.
    pub fn insert_file(&self, file: &BidsFile) -> Result<()> {
        if self.in_tx.get() {
            self.pending.borrow_mut().files.insert(
                file.path.to_string_lossy().into_owned(),
                (
                    file.filename.clone(),
                    file.dirname.to_string_lossy().into_owned(),
                    file.is_dir,
                    format!("{:?}", file.file_type),
                ),
            );
            return Ok(());
        }
        self.invalidate();
        self.conn
            .execute(
                "INSERT OR REPLACE INTO files (path, filename, dirname, is_dir, file_type)
             VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    file.path.to_string_lossy().as_ref(),
                    file.filename,
                    file.dirname.to_string_lossy().as_ref(),
                    file.is_dir,
                    format!("{:?}", file.file_type),
                ],
            )
            .map_err(db_err)?;
        Ok(())
    }

    /// Insert a tag (entity-value pair for a file).
    pub fn insert_tag(
        &self,
        file_path: &str,
        entity_name: &str,
        value: &str,
        dtype: &str,
        is_metadata: bool,
    ) -> Result<()> {
        if self.in_tx.get() {
            self.pending.borrow_mut().tags.insert(
                (file_path.to_string(), entity_name.to_string()),
                (value.to_string(), dtype.to_string(), is_metadata),
            );
            return Ok(());
        }
        self.invalidate();
        self.conn
            .execute(
                "INSERT OR REPLACE INTO tags (file_path, entity_name, value, dtype, is_metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
                params![file_path, entity_name, value, dtype, is_metadata],
            )
            .map_err(db_err)?;
        Ok(())
    }

    /// Insert a file association.
    pub fn insert_association(&self, src: &str, dst: &str, kind: &str) -> Result<()> {
        if self.in_tx.get() {
            self.pending.borrow_mut().associations.insert((
                src.to_string(),
                dst.to_string(),
                kind.to_string(),
            ));
            return Ok(());
        }
        self.invalidate();
        self.conn
            .execute(
                "INSERT OR IGNORE INTO associations (src, dst, kind) VALUES (?1, ?2, ?3)",
                params![src, dst, kind],
            )
            .map_err(db_err)?;
        Ok(())
    }

    /// Store layout info.
    pub fn set_layout_info(&self, root: &str, config: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO layout_info (root, config) VALUES (?1, ?2)",
                params![root, config],
            )
            .map_err(db_err)?;
        Ok(())
    }

    /// Get layout info (root, config).
    pub fn get_layout_info(&self) -> Result<Option<(String, String)>> {
        self.flush_pending()?;
        let mut stmt = self
            .conn
            .prepare("SELECT root, config FROM layout_info LIMIT 1")
            .map_err(db_err)?;

        let result = stmt
            .query_row([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .ok();
        Ok(result)
    }

    /// Query all file paths, sorted.
    ///
    /// Like [`all_tags`](Self::all_tags), this bulk read scans the table
    /// directly when the snapshot is cold rather than building it: the
    /// indexer reads everything once between writes, and a snapshot built
    /// then would be dropped by its next insert.
    pub fn all_file_paths(&self) -> Result<Vec<String>> {
        if self.snapshot_is_warm() {
            return Ok(self.snapshot()?.files.clone());
        }
        self.flush_pending()?;
        self.scan_file_paths()
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
        if self.snapshot_is_warm() {
            return Ok(self.snapshot()?.tags.clone());
        }
        self.flush_pending()?;
        self.scan_tags()
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
        Ok(self.snapshot()?.files.len())
    }

    /// Save the current database to a file.
    ///
    /// Copies every table into a fresh DuckDB file via `ATTACH` +
    /// `COPY FROM DATABASE`. Any existing file at `path` is replaced.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        self.flush_pending()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        for p in [path.to_path_buf(), wal_path(path)] {
            if p.exists() {
                std::fs::remove_file(&p)?;
            }
        }
        let current: String = self
            .conn
            .query_row("SELECT current_database()", [], |row| row.get(0))
            .map_err(db_err)?;
        let target = path.to_string_lossy().replace('\'', "''");
        self.conn
            .execute_batch(&format!(
                "ATTACH '{target}' AS bids_save_target;
                 COPY FROM DATABASE \"{current}\" TO bids_save_target;
                 DETACH bids_save_target;"
            ))
            .map_err(db_err)?;
        Ok(())
    }
}

/// One tag: `(entity_name, value, dtype, is_metadata)`.
pub type TagRow = (String, String, String, bool);

fn collect_tag_rows(
    stmt: &mut duckdb::Statement<'_>,
    params: impl duckdb::Params,
) -> Result<HashMap<String, Vec<TagRow>>> {
    let mut out: HashMap<String, Vec<TagRow>> = HashMap::new();
    let mut rows = stmt.query(params).map_err(db_err)?;
    while let Some(row) = rows.next().map_err(db_err)? {
        let path: String = row.get(0).map_err(db_err)?;
        out.entry(path).or_default().push((
            row.get(1).map_err(db_err)?,
            row.get(2).map_err(db_err)?,
            row.get(3).map_err(db_err)?,
            row.get(4).map_err(db_err)?,
        ));
    }
    Ok(out)
}

/// DuckDB's write-ahead log sits next to the database as `<file>.wal`.
fn wal_path(path: &Path) -> std::path::PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".wal");
    std::path::PathBuf::from(s)
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
        let path = dir.join("index.duckdb");
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
}
