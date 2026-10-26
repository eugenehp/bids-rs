//! Dataset indexer: walks the filesystem and populates the DuckDB database.
//!
//! Recursively walks a BIDS dataset directory, extracts entities from each
//! file using regex patterns, indexes JSON sidecar metadata following the
//! BIDS inheritance principle, and records file associations.

use bids_core::config::Config;
use bids_core::entities::Entity;
use bids_core::error::Result;
use bids_core::file::BidsFile;
use bids_io::json::read_json_sidecar;
use bids_validate::{should_force_index, should_ignore};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::db::Database;

/// Collect unique entities from configs, preserving first-seen order.
fn collect_unique_entities(configs: &[Config]) -> Vec<Entity> {
    let mut all = Vec::new();
    let mut seen = HashSet::new();
    for config in configs {
        for entity in &config.entities {
            if seen.insert(entity.name.clone()) {
                all.push(entity.clone());
            }
        }
    }
    all
}

/// Extract entities from a path and insert the file + tags into the database.
fn index_single_file(path: &Path, db: &Database, entities: &[Entity]) -> Result<()> {
    let mut bf = BidsFile::new(path);
    let path_str = path.to_string_lossy();
    for entity in entities {
        if let Some(val) = entity.match_path(&path_str) {
            bf.entities.insert(entity.name.clone(), val);
        }
    }
    db.insert_file(&bf)?;
    let file_path_str = path_str.into_owned();
    for (name, val) in &bf.entities {
        db.insert_tag(&file_path_str, name, &val.as_str_lossy(), "str", false)?;
    }
    Ok(())
}

/// Options controlling how a BIDS dataset directory is indexed.
///
/// These options determine which files are included in the index, whether
/// BIDS validation is enforced, and whether JSON sidecar metadata is loaded.
pub struct IndexerOptions {
    pub validate: bool,
    pub ignore: Vec<Regex>,
    pub force_index: Vec<Regex>,
    pub index_metadata: bool,
    pub config_filename: String,
}

impl Default for IndexerOptions {
    fn default() -> Self {
        Self {
            validate: true,
            ignore: bids_validate::DEFAULT_IGNORE.clone(),
            force_index: Vec::new(),
            index_metadata: true,
            config_filename: "layout_config.json".to_string(),
        }
    }
}

/// Index a BIDS dataset directory into the database.
///
/// Walks the dataset directory tree, extracts BIDS entities from each file
/// using the provided configuration, stores files and tags in the database,
/// and optionally indexes JSON sidecar metadata with inheritance resolution
/// and file association tracking.
///
/// Files in the `derivatives/` directory at the root level are excluded
/// (derivatives should be added separately via `BidsLayout::add_derivatives`).
///
/// Bulk inserts use a single DuckDB transaction for dramatically better
/// performance on large datasets (100× faster than autocommit per-file).
pub fn index_dataset(
    root: &Path,
    db: &Database,
    configs: &[Config],
    options: &IndexerOptions,
) -> Result<()> {
    // Collect all entities from configs, deduplicating by name.
    let mut all_entities = collect_unique_entities(configs);

    // Begin a transaction for bulk inserts — avoids per-file fsync.
    db.begin_transaction()?;

    let result = index_files(root, db, &mut all_entities, options);
    if result.is_err() {
        let _ = db.rollback_transaction();
        return result;
    }

    // Index .zarr directories as single files
    index_zarr_dirs(
        root,
        db,
        &all_entities,
        &options.ignore,
        &options.force_index,
    )?;

    db.commit_transaction()?;

    // Index metadata from JSON sidecars (separate transaction)
    if options.index_metadata {
        db.begin_transaction()?;
        let md_result = index_metadata(root, db);
        if md_result.is_err() {
            let _ = db.rollback_transaction();
            return md_result;
        }
        db.commit_transaction()?;
    }

    Ok(())
}

/// Walk and index files (called within a transaction).
fn index_files(
    root: &Path,
    db: &Database,
    all_entities: &mut Vec<Entity>,
    options: &IndexerOptions,
) -> Result<()> {
    // Walk the directory tree
    for entry in WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| {
            // Skip derivatives directory at root level
            if let Ok(rel) = e.path().strip_prefix(root) {
                let rel_str = rel.to_string_lossy();
                if rel_str == "derivatives" || rel_str.starts_with("derivatives/") {
                    return false;
                }
            }
            // Skip ignored directories early
            if e.file_type().is_dir()
                && should_ignore(e.path(), root, &options.ignore)
                && !should_force_index(e.path(), root, &options.force_index)
            {
                return false;
            }
            true
        })
        .filter_map(std::result::Result::ok)
    {
        let path = entry.path();

        // Skip directories themselves
        if entry.file_type().is_dir() {
            // Check for per-directory config files
            let config_file = path.join(&options.config_filename);
            if config_file.exists()
                && let Ok(cfg) = Config::from_file(&config_file)
            {
                for entity in &cfg.entities {
                    if !all_entities.iter().any(|e| e.name == entity.name) {
                        all_entities.push(entity.clone());
                    }
                }
            }
            continue;
        }

        // Skip the config filename itself
        if path
            .file_name()
            .is_some_and(|n| n.to_str() == Some(&options.config_filename))
        {
            continue;
        }

        // Check ignore/force patterns
        let is_ignored = should_ignore(path, root, &options.ignore);
        let is_forced = should_force_index(path, root, &options.force_index);

        if is_ignored && !is_forced {
            continue;
        }

        // Optional BIDS validation
        if !is_forced && options.validate && !is_bids_valid(path, root) {
            continue;
        }

        // Handle symlinks that point to directories (treat as dirs, skip)
        if path.is_dir() {
            continue;
        }

        // Handle .zarr directories as files
        let path_str_raw = path.to_string_lossy();
        if path_str_raw.contains(".zarr/") {
            continue; // Skip files inside .zarr directories
        }

        index_single_file(path, db, all_entities)?;
    }

    Ok(())
}

/// Index .zarr directories as single file entries.
fn index_zarr_dirs(
    root: &Path,
    db: &Database,
    entities: &[Entity],
    _ignore: &[Regex],
    _force: &[Regex],
) -> Result<()> {
    for entry in WalkDir::new(root)
        .follow_links(true)
        .into_iter()
        .filter_map(std::result::Result::ok)
    {
        let path = entry.path();
        if entry.file_type().is_dir()
            && let Some(ext) = path.extension()
            && ext == "zarr"
        {
            index_single_file(path, db, entities)?;
        }
    }
    Ok(())
}

/// Basic BIDS validity check for a file path.
fn is_bids_valid(path: &Path, root: &Path) -> bool {
    let rel = match path.strip_prefix(root) {
        Ok(r) => r,
        Err(_) => return false,
    };
    let rel_str = rel.to_string_lossy();

    // Root-level files are always valid
    if !rel_str.contains('/') && !rel_str.contains('\\') {
        return true;
    }

    // Must be inside a sub-* directory
    let first_component = rel
        .components()
        .next()
        .and_then(|c| c.as_os_str().to_str())
        .unwrap_or("");
    first_component.starts_with("sub-")
}

/// Index metadata from JSON sidecar files.
fn index_metadata(root: &Path, db: &Database) -> Result<()> {
    let all_paths = db.all_file_paths()?;

    // Separate JSON files and data files
    let mut json_files: HashSet<PathBuf> = HashSet::new();
    let mut data_files: Vec<String> = Vec::new();

    for path_str in &all_paths {
        let path = PathBuf::from(path_str);
        if path.extension().is_some_and(|e| e == "json") {
            json_files.insert(path);
        } else {
            data_files.push(path_str.clone());
        }
    }

    // Build existing tags to avoid duplicates and detect conflicts
    // One scan instead of a query per file — per-query overhead dominates
    // point lookups on DuckDB. The metadata pass below only adds tags, so
    // this snapshot is what per-file reads would have returned.
    let tags_by_path = db.all_tags()?;
    let mut existing_tags: HashMap<String, String> = HashMap::new();
    for (path_str, tags) in &tags_by_path {
        for (entity_name, value, _, _) in tags {
            existing_tags.insert(format!("{path_str}_{entity_name}"), value.clone());
        }
    }
    let no_tags = Vec::new();
    let entity_index = EntityIndex::new(&tags_by_path);

    let mut seen_assocs: HashSet<String> = HashSet::new();
    // Sidecars are shared: a top-level task JSON applies to every run in the
    // dataset. Parse each one once rather than per data file.
    let mut sidecar_cache: HashMap<PathBuf, Vec<(String, serde_json::Value)>> = HashMap::new();

    for data_path_str in &data_files {
        let data_path = PathBuf::from(data_path_str);
        let data_tags = tags_by_path.get(data_path_str).unwrap_or(&no_tags);

        let suffix = data_tags
            .iter()
            .find(|(n, _, _, _)| n == "suffix")
            .map(|(_, v, _, _)| v.clone());
        let extension = data_tags
            .iter()
            .find(|(n, _, _, _)| n == "extension")
            .map(|(_, v, _, _)| v.clone());

        let suffix = match suffix {
            Some(s) => s,
            None => continue,
        };

        let data_entities: HashMap<String, String> = data_tags
            .iter()
            .filter(|(n, _, _, _)| n != "suffix" && n != "extension")
            .map(|(n, v, _, _)| (n.clone(), v.clone()))
            .collect();

        // Walk up directory tree finding matching JSON sidecars
        let mut dir = data_path.parent();
        let mut sidecar_stack: Vec<PathBuf> = Vec::new();

        while let Some(current_dir) = dir {
            for json_path in &json_files {
                if json_path.parent() != Some(current_dir) {
                    continue;
                }

                let json_stem = json_path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                let json_suffix = json_stem.rsplit('_').next().unwrap_or("");
                if json_suffix != suffix {
                    continue;
                }

                let json_entities = extract_kv_pairs(json_stem);
                let all_match = json_entities
                    .iter()
                    .all(|(k, v)| data_entities.get(k).is_none_or(|dv| dv == v));

                if all_match {
                    sidecar_stack.push(json_path.clone());

                    let assoc_key =
                        format!("{}#{}#Metadata", json_path.to_string_lossy(), data_path_str);
                    if seen_assocs.insert(assoc_key) {
                        db.insert_association(
                            &json_path.to_string_lossy(),
                            data_path_str,
                            "Metadata",
                        )?;
                    }
                }
            }

            if current_dir == root {
                break;
            }
            dir = current_dir.parent();
        }

        // Create parent/child chain for JSON inheritance
        for i in 0..sidecar_stack.len() {
            if i + 1 < sidecar_stack.len() {
                let src = sidecar_stack[i].to_string_lossy().to_string();
                let dst = sidecar_stack[i + 1].to_string_lossy().to_string();
                let key1 = format!("{src}#{dst}#Child");
                if seen_assocs.insert(key1) {
                    db.insert_association(&src, &dst, "Child")?;
                    db.insert_association(&dst, &src, "Parent")?;
                }
            }
        }

        // Merge sidecars: least specific first
        sidecar_stack.reverse();
        let mut merged_metadata: indexmap::IndexMap<String, serde_json::Value> =
            indexmap::IndexMap::new();
        for sidecar_path in &sidecar_stack {
            let parsed = sidecar_cache
                .entry(sidecar_path.clone())
                .or_insert_with(|| {
                    read_json_sidecar(sidecar_path)
                        .map(|md| md.into_iter().collect())
                        .unwrap_or_default()
                });
            for (k, v) in parsed.iter() {
                merged_metadata.insert(k.clone(), v.clone());
            }
        }

        // Write metadata tags, checking for conflicts
        for (key, value) in &merged_metadata {
            if value.is_null() {
                continue;
            }

            let tag_key = format!("{data_path_str}_{key}");
            // Store the value as plain text so metadata filters compare
            // against it directly, and record its JSON type alongside so
            // get_metadata can restore it faithfully. Storing a string
            // unquoted and re-parsing it as JSON would turn "35119" into the
            // number 35119.
            let (val_str, dtype) = match value {
                serde_json::Value::String(s) => (s.clone(), "str"),
                serde_json::Value::Bool(b) => (b.to_string(), "bool"),
                serde_json::Value::Number(n) => {
                    (n.to_string(), if n.is_f64() { "float" } else { "int" })
                }
                other => (other.to_string(), "json"),
            };

            if let Some(existing_val) = existing_tags.get(&tag_key) {
                if *existing_val != val_str {
                    log::warn!(
                        "conflicting metadata for '{key}' on {data_path_str}: '{existing_val}' vs '{val_str}'"
                    );
                }
                continue;
            }
            db.insert_tag(data_path_str, key, &val_str, dtype, true)?;
        }

        // Handle IntendedFor
        if let Some(intended) = merged_metadata.get("IntendedFor") {
            let subject = data_entities.get("subject").cloned().unwrap_or_default();
            index_intended_for(db, data_path_str, intended, root, &subject)?;
        }

        // Link companion files (events↔bold, bvec/bval↔DWI)
        index_companion_associations(
            db,
            &entity_index,
            data_path_str,
            &suffix,
            extension.as_deref(),
            &data_entities,
        )?;
    }

    Ok(())
}

/// Resolve and record IntendedFor associations from metadata.
fn index_intended_for(
    db: &Database,
    data_path: &str,
    intended: &serde_json::Value,
    root: &Path,
    subject: &str,
) -> Result<()> {
    let intents: Vec<&str> = match intended {
        serde_json::Value::String(s) => vec![s.as_str()],
        serde_json::Value::Array(arr) => arr.iter().filter_map(|v| v.as_str()).collect(),
        _ => vec![],
    };

    for intent in intents {
        if let Some(target) = bids_validate::resolve_intended_for(intent, root, subject) {
            let target_str = target.to_string_lossy();
            db.insert_association(data_path, &target_str, "IntendedFor")?;
            db.insert_association(&target_str, data_path, "InformedBy")?;
        }
    }
    Ok(())
}

/// Link companion files (events/physio/stim/sbref ↔ bold/eeg, bvec/bval ↔ DWI).
fn index_companion_associations(
    db: &Database,
    entity_index: &EntityIndex<'_>,
    data_path: &str,
    suffix: &str,
    extension: Option<&str>,
    data_entities: &HashMap<String, String>,
) -> Result<()> {
    if extension.is_none() {
        return Ok(());
    }

    if matches!(suffix, "events" | "physio" | "stim" | "sbref") {
        let mut filters: Vec<(String, Vec<String>, bool)> = data_entities
            .iter()
            .filter(|(k, _)| matches!(k.as_str(), "subject" | "session" | "task" | "run"))
            .map(|(k, v)| (k.clone(), vec![v.clone()], false))
            .collect();
        filters.push(("suffix".into(), vec!["bold".into(), "eeg".into()], false));

        {
            let images = entity_index.query(&filters);
            for img in &images {
                db.insert_association(data_path, img, "IntendedFor")?;
                db.insert_association(img, data_path, "InformedBy")?;
            }
        }
    }

    if suffix == "dwi" && matches!(extension, Some(".bvec" | ".bval")) {
        let mut filters: Vec<(String, Vec<String>, bool)> = data_entities
            .iter()
            .filter(|(k, _)| matches!(k.as_str(), "subject" | "session" | "run" | "acquisition"))
            .map(|(k, v)| (k.clone(), vec![v.clone()], false))
            .collect();
        filters.push(("suffix".into(), vec!["dwi".into()], false));
        filters.push((
            "extension".into(),
            vec![".nii".into(), ".nii.gz".into()],
            false,
        ));

        {
            let images = entity_index.query(&filters);
            for img in &images {
                db.insert_association(data_path, img, "IntendedFor")?;
                db.insert_association(img, data_path, "InformedBy")?;
            }
        }
    }

    Ok(())
}

/// In-memory inverted index over entity tags: `(entity, value) → paths`.
///
/// The metadata pass looks up companion files once per data file. Answering
/// those from the tag snapshot instead of a database query keeps indexing
/// linear without a per-file round trip (and, on DuckDB, without a buffer
/// flush per lookup). Matches `Database::query_files` for plain value
/// filters: every filter must match, any of its values may.
struct EntityIndex<'a> {
    by_value: HashMap<(&'a str, &'a str), HashSet<&'a str>>,
}

impl<'a> EntityIndex<'a> {
    fn new(tags_by_path: &'a HashMap<String, Vec<crate::db::TagRow>>) -> Self {
        let mut by_value: HashMap<(&str, &str), HashSet<&str>> = HashMap::new();
        for (path, tags) in tags_by_path {
            for (name, value, _, is_metadata) in tags {
                if !is_metadata {
                    by_value
                        .entry((name.as_str(), value.as_str()))
                        .or_default()
                        .insert(path.as_str());
                }
            }
        }
        Self { by_value }
    }

    fn query(&self, filters: &[(String, Vec<String>, bool)]) -> Vec<String> {
        let mut result: Option<HashSet<&str>> = None;
        for (name, values, _) in filters {
            let mut matching: HashSet<&str> = HashSet::new();
            for v in values {
                if let Some(paths) = self.by_value.get(&(name.as_str(), v.as_str())) {
                    matching.extend(paths.iter().copied());
                }
            }
            result = Some(match result {
                None => matching,
                Some(acc) => acc.intersection(&matching).copied().collect(),
            });
            if result.as_ref().is_some_and(HashSet::is_empty) {
                return Vec::new();
            }
        }
        let mut out: Vec<String> = result
            .unwrap_or_default()
            .into_iter()
            .map(String::from)
            .collect();
        out.sort();
        out
    }
}

/// Extract key-value pairs from a BIDS filename stem.
fn extract_kv_pairs(stem: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for part in stem.split('_') {
        if let Some(idx) = part.find('-') {
            let key = &part[..idx];
            let val = &part[idx + 1..];
            let entity_name = match key {
                "sub" => "subject",
                "ses" => "session",
                "acq" => "acquisition",
                "ce" => "ceagent",
                "rec" => "reconstruction",
                "dir" => "direction",
                "mod" => "modality",
                "trc" => "tracer",
                other => other,
            };
            pairs.push((entity_name.to_string(), val.to_string()));
        }
    }
    pairs
}
