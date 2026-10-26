//! DuckDB database backend for the BIDS file index.
//!
//! Stores files, entity tags, metadata, and file associations in a DuckDB
//! database (in-memory or on-disk). Regex queries use DuckDB's built-in
//! `regexp_matches`, and bulk inserts run inside explicit transactions.

use bids_core::error::{BidsError, Result};
use bids_core::file::BidsFile;
use duckdb::{Connection, params};
use indexmap::{IndexMap, IndexSet};
use std::cell::{Cell, RefCell};
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
/// are evaluated with DuckDB's native `regexp_matches` (RE2 syntax).
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
        }
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
        if self.conn.is_autocommit() {
            return Ok(());
        }
        self.conn.execute_batch("ROLLBACK").map_err(db_err)
    }

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
            CREATE INDEX IF NOT EXISTS idx_tags_file ON tags(file_path);
            CREATE INDEX IF NOT EXISTS idx_tags_entity ON tags(entity_name);
            CREATE INDEX IF NOT EXISTS idx_tags_value ON tags(value);
            CREATE INDEX IF NOT EXISTS idx_assoc_src ON associations(src);
            CREATE INDEX IF NOT EXISTS idx_assoc_dst ON associations(dst);
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
        self.conn.execute_batch("ROLLBACK").map_err(db_err)?;
        Ok(())
    }

    /// Write buffered inserts: Appender into temp staging tables, then one
    /// `INSERT OR REPLACE … SELECT` (or `OR IGNORE`) per target table.
    fn flush_pending(&self) -> Result<()> {
        let pending = std::mem::take(&mut *self.pending.borrow_mut());
        if pending.is_empty() {
            return Ok(());
        }
        self.conn
            .execute_batch(
                "CREATE TEMP TABLE IF NOT EXISTS stage_files AS SELECT * FROM files LIMIT 0;
                 CREATE TEMP TABLE IF NOT EXISTS stage_tags AS SELECT * FROM tags LIMIT 0;
                 CREATE TEMP TABLE IF NOT EXISTS stage_associations AS SELECT * FROM associations LIMIT 0;",
            )
            .map_err(db_err)?;

        if !pending.files.is_empty() {
            let mut app = self
                .conn
                .appender_to_catalog_and_db("stage_files", "temp", "main")
                .map_err(db_err)?;
            for (path, (filename, dirname, is_dir, file_type)) in &pending.files {
                app.append_row(params![path, filename, dirname, is_dir, file_type])
                    .map_err(db_err)?;
            }
            app.flush().map_err(db_err)?;
            drop(app);
            self.conn
                .execute_batch(
                    "INSERT OR REPLACE INTO files SELECT * FROM temp.stage_files;
                     DELETE FROM temp.stage_files;",
                )
                .map_err(db_err)?;
        }
        if !pending.tags.is_empty() {
            let mut app = self
                .conn
                .appender_to_catalog_and_db("stage_tags", "temp", "main")
                .map_err(db_err)?;
            for ((file_path, entity_name), (value, dtype, is_metadata)) in &pending.tags {
                app.append_row(params![file_path, entity_name, value, dtype, is_metadata])
                    .map_err(db_err)?;
            }
            app.flush().map_err(db_err)?;
            drop(app);
            self.conn
                .execute_batch(
                    "INSERT OR REPLACE INTO tags SELECT * FROM temp.stage_tags;
                     DELETE FROM temp.stage_tags;",
                )
                .map_err(db_err)?;
        }
        if !pending.associations.is_empty() {
            let mut app = self
                .conn
                .appender_to_catalog_and_db("stage_associations", "temp", "main")
                .map_err(db_err)?;
            for (src, dst, kind) in &pending.associations {
                app.append_row(params![src, dst, kind]).map_err(db_err)?;
            }
            app.flush().map_err(db_err)?;
            drop(app);
            self.conn
                .execute_batch(
                    "INSERT OR IGNORE INTO associations SELECT * FROM temp.stage_associations;
                     DELETE FROM temp.stage_associations;",
                )
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

    /// Query all file paths.
    pub fn all_file_paths(&self) -> Result<Vec<String>> {
        self.flush_pending()?;
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM files WHERE NOT is_dir")
            .map_err(db_err)?;
        let paths: std::result::Result<Vec<String>, _> = stmt
            .query_map([], |row| row.get(0))
            .map_err(db_err)?
            .collect();
        paths.map_err(db_err)
    }

    /// Get tags for a specific file.
    pub fn get_tags(&self, file_path: &str) -> Result<Vec<(String, String, String, bool)>> {
        self.flush_pending()?;
        let mut stmt = self
            .conn
            .prepare("SELECT entity_name, value, dtype, is_metadata FROM tags WHERE file_path = ?1")
            .map_err(db_err)?;
        let tags: std::result::Result<Vec<_>, _> = stmt
            .query_map(params![file_path], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, bool>(3)?,
                ))
            })
            .map_err(db_err)?
            .collect();
        tags.map_err(db_err)
    }

    /// Tags for every file, keyed by path, in one scan.
    ///
    /// Prefer this (or [`get_tags_many`](Self::get_tags_many)) over calling
    /// [`get_tags`](Self::get_tags) in a loop: each DuckDB query carries a
    /// fixed planning/execution cost that dominates per-file point lookups.
    pub fn all_tags(&self) -> Result<std::collections::HashMap<String, Vec<TagRow>>> {
        self.flush_pending()?;
        let mut stmt = self
            .conn
            .prepare("SELECT file_path, entity_name, value, dtype, is_metadata FROM tags")
            .map_err(db_err)?;
        collect_tag_rows(&mut stmt, [])
    }

    /// Tags for the given files, keyed by path, in one query.
    ///
    /// Files with no tags are absent from the map.
    pub fn get_tags_many(
        &self,
        paths: &[String],
    ) -> Result<std::collections::HashMap<String, Vec<TagRow>>> {
        self.flush_pending()?;
        if paths.is_empty() {
            return Ok(std::collections::HashMap::new());
        }
        self.conn
            .execute_batch(
                "CREATE TEMP TABLE IF NOT EXISTS lookup_paths (path TEXT);
                 DELETE FROM temp.lookup_paths;",
            )
            .map_err(db_err)?;
        {
            let mut app = self
                .conn
                .appender_to_catalog_and_db("lookup_paths", "temp", "main")
                .map_err(db_err)?;
            for p in paths {
                app.append_row(params![p]).map_err(db_err)?;
            }
            app.flush().map_err(db_err)?;
        }
        let mut stmt = self
            .conn
            .prepare(
                "SELECT t.file_path, t.entity_name, t.value, t.dtype, t.is_metadata
                 FROM tags t SEMI JOIN temp.lookup_paths l ON t.file_path = l.path",
            )
            .map_err(db_err)?;
        let out = collect_tag_rows(&mut stmt, []);
        self.conn
            .execute_batch("DELETE FROM temp.lookup_paths")
            .map_err(db_err)?;
        out
    }

    /// Get all unique values for a given entity.
    pub fn get_unique_entity_values(&self, entity_name: &str) -> Result<Vec<String>> {
        self.flush_pending()?;
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT value FROM tags WHERE entity_name = ?1 ORDER BY value")
            .map_err(db_err)?;
        let values: std::result::Result<Vec<String>, _> = stmt
            .query_map(params![entity_name], |row| row.get(0))
            .map_err(db_err)?
            .collect();
        values.map_err(db_err)
    }

    /// Get all unique entity names.
    pub fn get_entity_names(&self) -> Result<Vec<String>> {
        self.flush_pending()?;
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT entity_name FROM tags ORDER BY entity_name")
            .map_err(db_err)?;
        let names: std::result::Result<Vec<String>, _> = stmt
            .query_map([], |row| row.get(0))
            .map_err(db_err)?
            .collect();
        names.map_err(db_err)
    }

    /// Query files with advanced filtering including Query::None/Any and regex.
    ///
    /// Filter types:
    /// - Normal: (entity, values, false) — entity must match one of values
    /// - Regex: `(entity, [pattern], true)` — entity must match regex
    /// - Query::None: (entity, ["__NONE__"], false) — entity must NOT exist
    /// - Query::Any: (entity, ["__ANY__"], false) — entity must exist (any value)
    pub fn query_files(&self, filters: &[(String, Vec<String>, bool)]) -> Result<Vec<String>> {
        self.flush_pending()?;
        if filters.is_empty() {
            return self.all_file_paths();
        }

        let mut sql = String::from("SELECT f.path FROM files f WHERE NOT f.is_dir");
        let mut bind_values: Vec<String> = Vec::new();

        for (i, (entity_name, values, is_regex)) in filters.iter().enumerate() {
            use std::fmt::Write;

            // Check for special Query types
            if values.len() == 1 {
                match values[0].as_str() {
                    "__NONE__" => {
                        write!(sql, " AND NOT EXISTS (SELECT 1 FROM tags t{i} WHERE t{i}.file_path = f.path AND t{i}.entity_name = ?)").unwrap();
                        bind_values.push(entity_name.clone());
                        continue;
                    }
                    "__ANY__" => {
                        write!(sql, " AND EXISTS (SELECT 1 FROM tags t{i} WHERE t{i}.file_path = f.path AND t{i}.entity_name = ?)").unwrap();
                        bind_values.push(entity_name.clone());
                        continue;
                    }
                    "__OPTIONAL__" => continue,
                    _ => {}
                }
            }

            if *is_regex && !values.is_empty() {
                write!(sql, " AND EXISTS (SELECT 1 FROM tags t{i} WHERE t{i}.file_path = f.path AND t{i}.entity_name = ? AND regexp_matches(t{i}.value, ?))").unwrap();
                bind_values.push(entity_name.clone());
                bind_values.push(values[0].clone());
            } else if values.len() == 1 {
                write!(sql, " AND EXISTS (SELECT 1 FROM tags t{i} WHERE t{i}.file_path = f.path AND t{i}.entity_name = ? AND t{i}.value = ?)").unwrap();
                bind_values.push(entity_name.clone());
                bind_values.push(values[0].clone());
            } else if !values.is_empty() {
                let placeholders = "?,".repeat(values.len());
                let placeholders = &placeholders[..placeholders.len() - 1];
                write!(sql, " AND EXISTS (SELECT 1 FROM tags t{i} WHERE t{i}.file_path = f.path AND t{i}.entity_name = ? AND t{i}.value IN ({placeholders}))").unwrap();
                bind_values.push(entity_name.clone());
                bind_values.extend(values.iter().cloned());
            }
        }

        sql.push_str(" ORDER BY f.path");

        let mut stmt = self.conn.prepare(&sql).map_err(db_err)?;
        let params: Vec<&dyn duckdb::ToSql> = bind_values
            .iter()
            .map(|v| v as &dyn duckdb::ToSql)
            .collect();
        let paths: std::result::Result<Vec<String>, _> = stmt
            .query_map(params.as_slice(), |row| row.get::<_, String>(0))
            .map_err(db_err)?
            .collect();
        paths.map_err(db_err)
    }

    /// Get distinct directories for files matching filters and having a target entity.
    pub fn query_directories(
        &self,
        target_entity: &str,
        filters: &[(String, Vec<String>, bool)],
    ) -> Result<Vec<String>> {
        let paths = self.query_files(filters)?;
        let tags_by_path = self.get_tags_many(&paths)?;
        let mut dirs = std::collections::BTreeSet::new();
        for path_str in &paths {
            // Check if the file has the target entity
            let has_target = tags_by_path
                .get(path_str)
                .is_some_and(|tags| tags.iter().any(|(n, _, _, _)| n == target_entity));
            if has_target && let Some(parent) = std::path::Path::new(path_str).parent() {
                dirs.insert(parent.to_string_lossy().to_string());
            }
        }
        Ok(dirs.into_iter().collect())
    }

    /// Get associated files for a given source file.
    pub fn get_associations(&self, src: &str, kind: Option<&str>) -> Result<Vec<(String, String)>> {
        self.flush_pending()?;
        let (sql, params_vec): (String, Vec<String>) = if let Some(k) = kind {
            (
                "SELECT dst, kind FROM associations WHERE src = ?1 AND kind = ?2".to_string(),
                vec![src.to_string(), k.to_string()],
            )
        } else {
            (
                "SELECT dst, kind FROM associations WHERE src = ?1".to_string(),
                vec![src.to_string()],
            )
        };

        let mut stmt = self.conn.prepare(&sql).map_err(db_err)?;
        let params: Vec<&dyn duckdb::ToSql> =
            params_vec.iter().map(|v| v as &dyn duckdb::ToSql).collect();
        let assocs: std::result::Result<Vec<_>, _> = stmt
            .query_map(params.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(db_err)?
            .collect();
        assocs.map_err(db_err)
    }

    /// Get the total number of indexed files.
    pub fn file_count(&self) -> Result<usize> {
        self.flush_pending()?;
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM files WHERE NOT is_dir")
            .map_err(db_err)?;
        let count: i64 = stmt.query_row([], |row| row.get(0)).map_err(db_err)?;
        Ok(count as usize)
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
) -> Result<std::collections::HashMap<String, Vec<TagRow>>> {
    let mut out: std::collections::HashMap<String, Vec<TagRow>> = std::collections::HashMap::new();
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
