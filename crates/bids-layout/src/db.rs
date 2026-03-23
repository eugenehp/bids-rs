//! SQLite database backend for the BIDS file index.
//!
//! Stores files, entity tags, metadata, and file associations in a SQLite
//! database (in-memory or on-disk). Supports REGEXP queries via a custom
//! function and transactional bulk inserts for fast indexing.

use bids_core::error::{BidsError, Result};
use bids_core::file::BidsFile;
use rusqlite::{Connection, functions::FunctionFlags, params};
use std::path::Path;

/// Convert a `rusqlite::Error` into a `BidsError::Database`.
fn db_err(e: rusqlite::Error) -> BidsError {
    BidsError::Database(e.to_string())
}

/// Manages the SQLite database backing a [`BidsLayout`](crate::BidsLayout) index.
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
/// on-disk persistence (for caching large dataset indices). It registers
/// a custom `REGEXP` function for SQLite to support regex-based queries.
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
}

impl Database {
    /// Create a new in-memory database.
    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(db_err)?;
        let db = Self { conn };
        db.register_regexp()?;
        db.create_tables()?;
        Ok(db)
    }

    /// Open or create a database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path).map_err(db_err)?;
        let db = Self { conn };
        db.register_regexp()?;
        db.create_tables()?;
        Ok(db)
    }

    /// Check if a database file exists.
    pub fn exists(path: &Path) -> bool {
        path.exists()
    }

    /// Register REGEXP function for SQLite.
    ///
    /// Caches compiled regexes so the same pattern is only compiled once per
    /// query, not once per row. Uses SQLite's auxiliary data API for O(1) reuse.
    fn register_regexp(&self) -> Result<()> {
        use std::cell::RefCell;
        use std::collections::HashMap;

        // Thread-local regex cache shared across all REGEXP invocations on this
        // connection. Avoids recompiling the same pattern for every row.
        thread_local! {
            static CACHE: RefCell<HashMap<String, regex::Regex>> = RefCell::new(HashMap::new());
        }

        self.conn
            .create_scalar_function(
                "regexp",
                2,
                FunctionFlags::SQLITE_UTF8 | FunctionFlags::SQLITE_DETERMINISTIC,
                |ctx| {
                    let pattern = ctx
                        .get_raw(0)
                        .as_str()
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                    let value = ctx
                        .get_raw(1)
                        .as_str()
                        .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                    CACHE.with(|cache| {
                        let mut cache = cache.borrow_mut();
                        if let Some(re) = cache.get(pattern) {
                            return Ok(re.is_match(value));
                        }
                        let re = regex::Regex::new(pattern)
                            .map_err(|e| rusqlite::Error::UserFunctionError(Box::new(e)))?;
                        let result = re.is_match(value);
                        cache.insert(pattern.to_string(), re);
                        Ok(result)
                    })
                },
            )
            .map_err(db_err)?;
        Ok(())
    }

    fn create_tables(&self) -> Result<()> {
        self.conn
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                dirname TEXT NOT NULL,
                is_dir INTEGER NOT NULL DEFAULT 0,
                file_type TEXT NOT NULL DEFAULT 'Generic'
            );
            CREATE TABLE IF NOT EXISTS tags (
                file_path TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                value TEXT NOT NULL,
                dtype TEXT NOT NULL DEFAULT 'str',
                is_metadata INTEGER NOT NULL DEFAULT 0,
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
    /// Wrapping many inserts in a single transaction avoids per-statement
    /// fsyncs, giving ~100× better insert throughput on large datasets.
    pub fn begin_transaction(&self) -> Result<()> {
        self.conn
            .execute_batch("BEGIN TRANSACTION")
            .map_err(db_err)?;
        Ok(())
    }

    /// Commit the current transaction.
    pub fn commit_transaction(&self) -> Result<()> {
        self.conn.execute_batch("COMMIT").map_err(db_err)?;
        Ok(())
    }

    /// Roll back the current transaction.
    pub fn rollback_transaction(&self) -> Result<()> {
        self.conn.execute_batch("ROLLBACK").map_err(db_err)?;
        Ok(())
    }

    /// Insert a file into the database.
    pub fn insert_file(&self, file: &BidsFile) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO files (path, filename, dirname, is_dir, file_type)
             VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    file.path.to_string_lossy().as_ref(),
                    file.filename,
                    file.dirname.to_string_lossy().as_ref(),
                    file.is_dir as i32,
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
        self.conn
            .execute(
                "INSERT OR REPLACE INTO tags (file_path, entity_name, value, dtype, is_metadata)
             VALUES (?1, ?2, ?3, ?4, ?5)",
                params![file_path, entity_name, value, dtype, is_metadata as i32],
            )
            .map_err(db_err)?;
        Ok(())
    }

    /// Insert a file association.
    pub fn insert_association(&self, src: &str, dst: &str, kind: &str) -> Result<()> {
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
        let mut stmt = self
            .conn
            .prepare("SELECT path FROM files WHERE is_dir = 0")
            .map_err(db_err)?;
        let paths: std::result::Result<Vec<String>, _> = stmt
            .query_map([], |row| row.get(0))
            .map_err(db_err)?
            .collect();
        paths.map_err(db_err)
    }

    /// Get tags for a specific file.
    pub fn get_tags(&self, file_path: &str) -> Result<Vec<(String, String, String, bool)>> {
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

    /// Get all unique values for a given entity.
    pub fn get_unique_entity_values(&self, entity_name: &str) -> Result<Vec<String>> {
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
        if filters.is_empty() {
            return self.all_file_paths();
        }

        let mut sql = String::from("SELECT f.path FROM files f WHERE f.is_dir = 0");
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
                write!(sql, " AND EXISTS (SELECT 1 FROM tags t{i} WHERE t{i}.file_path = f.path AND t{i}.entity_name = ? AND t{i}.value REGEXP ?)").unwrap();
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
        let params: Vec<&dyn rusqlite::types::ToSql> = bind_values
            .iter()
            .map(|v| v as &dyn rusqlite::types::ToSql)
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
        let mut dirs = std::collections::BTreeSet::new();
        for path_str in &paths {
            // Check if the file has the target entity
            let tags = self.get_tags(path_str)?;
            if tags.iter().any(|(n, _, _, _)| n == target_entity)
                && let Some(parent) = std::path::Path::new(path_str).parent()
            {
                dirs.insert(parent.to_string_lossy().to_string());
            }
        }
        Ok(dirs.into_iter().collect())
    }

    /// Get associated files for a given source file.
    pub fn get_associations(&self, src: &str, kind: Option<&str>) -> Result<Vec<(String, String)>> {
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
        let params: Vec<&dyn rusqlite::types::ToSql> = params_vec
            .iter()
            .map(|v| v as &dyn rusqlite::types::ToSql)
            .collect();
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
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM files WHERE is_dir = 0")
            .map_err(db_err)?;
        let count: i64 = stmt.query_row([], |row| row.get(0)).map_err(db_err)?;
        Ok(count as usize)
    }

    /// Save the current in-memory database to a file.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut backup = Connection::open(path).map_err(db_err)?;
        let b = rusqlite::backup::Backup::new(&self.conn, &mut backup).map_err(db_err)?;
        b.step(-1).map_err(db_err)?;
        Ok(())
    }
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
}
