//! Package-wide settings.
//!
//! Corresponds to PyBIDS' `bids.config` module plus
//! `bids.layout.utils.add_config_paths`. Settings are process-global and
//! guarded by a lock, so they are safe to read and write from several threads.
//!
//! # Example
//!
//! ```
//! use bids_core::options;
//!
//! options::reset();
//! options::set_option("extension_initial_dot", serde_json::json!(true)).unwrap();
//! assert_eq!(options::get_option("extension_initial_dot").unwrap(), serde_json::json!(true));
//! ```

use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{OnceLock, RwLock};

use crate::error::{BidsError, Result};

/// Filename looked for when loading settings from standard locations.
pub const CONFIG_NAME: &str = "bids_rs_config.json";

/// Environment variable naming an extra settings file.
pub const CONFIG_ENV: &str = "BIDS_RS_CONFIG";

fn defaults() -> HashMap<String, Value> {
    HashMap::from([
        (
            "config_paths".to_string(),
            Value::Object(serde_json::Map::new()),
        ),
        // Retained for PyBIDS compatibility; extensions always carry a leading
        // dot, so setting this to false is rejected.
        ("extension_initial_dot".to_string(), Value::Bool(true)),
    ])
}

fn settings() -> &'static RwLock<HashMap<String, Value>> {
    static SETTINGS: OnceLock<RwLock<HashMap<String, Value>>> = OnceLock::new();
    SETTINGS.get_or_init(|| RwLock::new(defaults()))
}

/// Set a package-wide option.
///
/// # Errors
///
/// Returns an error if `key` is not a known setting, or if
/// `extension_initial_dot` is set to anything but `true`.
pub fn set_option(key: &str, value: Value) -> Result<()> {
    let mut guard = settings()
        .write()
        .map_err(|_| BidsError::Config("settings lock is poisoned".into()))?;
    if !guard.contains_key(key) {
        return Err(BidsError::Config(format!("invalid setting: '{key}'")));
    }
    if key == "extension_initial_dot" && value != Value::Bool(true) {
        return Err(BidsError::Config(
            "'extension_initial_dot' is always true; extensions carry a leading dot".into(),
        ));
    }
    guard.insert(key.to_string(), value);
    Ok(())
}

/// Set several options at once.
///
/// # Errors
///
/// Returns an error on the first key that [`set_option()`] rejects; options
/// applied before that point are kept.
pub fn set_options(values: HashMap<String, Value>) -> Result<()> {
    for (k, v) in values {
        set_option(&k, v)?;
    }
    Ok(())
}

/// Retrieve the current value of a package-wide option.
///
/// # Errors
///
/// Returns an error if `key` is not a known setting.
pub fn get_option(key: &str) -> Result<Value> {
    let guard = settings()
        .read()
        .map_err(|_| BidsError::Config("settings lock is poisoned".into()))?;
    guard
        .get(key)
        .cloned()
        .ok_or_else(|| BidsError::Config(format!("invalid setting: '{key}'")))
}

/// Reset every option to its default.
///
/// With `update_from_file`, settings files in the standard locations are
/// re-applied afterwards.
pub fn reset_options(update_from_file: bool) {
    if let Ok(mut guard) = settings().write() {
        *guard = defaults();
    }
    if update_from_file {
        update_from_standard_locations();
    }
}

/// Reset every option to its default without consulting any files.
pub fn reset() {
    reset_options(false);
}

/// Load settings from one or more JSON files.
///
/// Later files take precedence over earlier ones.
///
/// # Errors
///
/// Returns an error if a file is missing and `error_on_missing` is set, or if
/// a file is not a JSON object.
pub fn from_file(paths: &[impl AsRef<Path>], error_on_missing: bool) -> Result<()> {
    for path in paths {
        let path = path.as_ref();
        if !path.exists() {
            if error_on_missing {
                return Err(BidsError::Config(format!(
                    "config file '{}' does not exist",
                    path.display()
                )));
            }
            continue;
        }
        let text = std::fs::read_to_string(path)?;
        let parsed: Value = serde_json::from_str(&text)?;
        let Value::Object(map) = parsed else {
            return Err(BidsError::Config(format!(
                "config file '{}' is not a JSON object",
                path.display()
            )));
        };
        let mut guard = settings()
            .write()
            .map_err(|_| BidsError::Config("settings lock is poisoned".into()))?;
        for (k, v) in map {
            guard.insert(k, v);
        }
    }
    Ok(())
}

/// Apply settings files found in the standard locations.
///
/// Order is the user's home directory, then `$BIDS_RS_CONFIG`, then the
/// current directory, with later files winning. Missing files are ignored.
pub fn update_from_standard_locations() {
    let mut locs: Vec<PathBuf> = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        locs.push(PathBuf::from(home).join(CONFIG_NAME));
    }
    if let Some(env_path) = std::env::var_os(CONFIG_ENV) {
        locs.push(PathBuf::from(env_path));
    }
    locs.push(PathBuf::from(".").join(CONFIG_NAME));
    let _ = from_file(&locs, false);
}

/// Register additional named layout configuration files.
///
/// Corresponds to PyBIDS' `add_config_paths`. Names must not already be
/// registered, and each path must exist.
///
/// # Errors
///
/// Returns an error if a name is already registered or a path does not exist.
///
/// # Example
///
/// ```no_run
/// use bids_core::options;
/// use std::collections::HashMap;
///
/// options::add_config_paths(HashMap::from([
///     ("my_config".to_string(), "/path/to/config.json".into()),
/// ]))
/// .unwrap();
/// ```
pub fn add_config_paths(paths: HashMap<String, PathBuf>) -> Result<()> {
    let existing = config_paths()?;
    for (name, path) in &paths {
        if !path.exists() {
            return Err(BidsError::Config(format!(
                "configuration file '{name}' does not exist: {}",
                path.display()
            )));
        }
        if existing.contains_key(name) {
            return Err(BidsError::Config(format!(
                "configuration '{name}' already exists"
            )));
        }
    }

    let mut merged = serde_json::Map::new();
    for (name, path) in existing.iter().chain(paths.iter()) {
        merged.insert(
            name.clone(),
            Value::String(path.to_string_lossy().to_string()),
        );
    }
    set_option("config_paths", Value::Object(merged))
}

/// The currently registered named layout configuration files.
///
/// # Errors
///
/// Returns an error if the settings lock is poisoned.
pub fn config_paths() -> Result<HashMap<String, PathBuf>> {
    let value = get_option("config_paths")?;
    Ok(value
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), PathBuf::from(s))))
                .collect()
        })
        .unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The settings are process-global, so these run as one test to keep them
    /// from racing each other.
    #[test]
    fn options_round_trip() {
        reset();

        // Unknown keys are rejected on both read and write.
        assert!(get_option("nope").is_err());
        assert!(set_option("nope", json!(1)).is_err());

        // extension_initial_dot is pinned to true.
        assert_eq!(get_option("extension_initial_dot").unwrap(), json!(true));
        assert!(set_option("extension_initial_dot", json!(false)).is_err());
        assert!(set_option("extension_initial_dot", json!(true)).is_ok());

        // config_paths starts empty and accepts registrations.
        assert!(config_paths().unwrap().is_empty());

        let dir = std::env::temp_dir().join("bids_rs_options_test");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("mine.json");
        std::fs::write(&cfg, "{}").unwrap();

        add_config_paths(HashMap::from([("mine".to_string(), cfg.clone())])).unwrap();
        assert_eq!(config_paths().unwrap().get("mine"), Some(&cfg));

        // Duplicate names and missing files are both errors.
        assert!(add_config_paths(HashMap::from([("mine".to_string(), cfg.clone())])).is_err());
        assert!(
            add_config_paths(HashMap::from([(
                "absent".to_string(),
                dir.join("nope.json")
            )]))
            .is_err()
        );

        // Loading from a file overrides settings.
        let settings_file = dir.join("settings.json");
        std::fs::write(
            &settings_file,
            r#"{"config_paths": {"from_file": "/x.json"}}"#,
        )
        .unwrap();
        from_file(&[&settings_file], true).unwrap();
        assert!(config_paths().unwrap().contains_key("from_file"));

        // A missing file errors only when asked to.
        assert!(from_file(&[dir.join("missing.json")], true).is_err());
        assert!(from_file(&[dir.join("missing.json")], false).is_ok());

        reset();
        assert!(config_paths().unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
