//! Layout configuration: entity patterns and path templates.
//!
//! A [`Config`] defines the entities a layout recognizes and the path patterns
//! used to build BIDS-compliant filenames. Two built-in configs ship with the
//! crate (`bids` and `derivatives`); custom configs can be loaded from JSON.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::entities::Entity;
use crate::error::{BidsError, Result};

/// Configuration for a BIDS layout, defining entities and path patterns.
///
/// Corresponds to PyBIDS' `Config` class. Can be loaded from JSON config files
/// (like `bids.json`) or constructed programmatically.
///
/// # Example
///
/// ```
/// use bids_core::Config;
///
/// let config = Config::bids();
/// assert_eq!(config.name, "bids");
/// assert!(config.entity_count() > 20);
/// assert!(config.get_entity("subject").is_some());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub name: String,
    #[serde(default)]
    pub entities: Vec<Entity>,
    #[serde(default)]
    pub default_path_patterns: Option<Vec<String>>,
}

impl Config {
    /// Load a config from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or contains invalid JSON.
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Load the built-in BIDS config.
    ///
    /// # Panics
    ///
    /// Panics if the embedded `bids.json` is malformed (should never happen).
    #[must_use]
    pub fn bids() -> Self {
        serde_json::from_str(include_str!("configs/bids.json"))
            .expect("Built-in bids.json config should be valid")
    }

    /// Load the built-in derivatives config.
    ///
    /// # Panics
    ///
    /// Panics if the embedded `derivatives.json` is malformed (should never happen).
    #[must_use]
    pub fn derivatives() -> Self {
        serde_json::from_str(include_str!("configs/derivatives.json"))
            .expect("Built-in derivatives.json config should be valid")
    }

    /// Load a named config (`"bids"`, `"derivatives"`) or from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is unrecognized and the path doesn't
    /// exist or contains invalid JSON.
    pub fn load(name_or_path: &str) -> Result<Self> {
        match name_or_path {
            "bids" => Ok(Self::bids()),
            "derivatives" => Ok(Self::derivatives()),
            path => {
                let p = Path::new(path);
                if p.exists() {
                    Self::from_file(p)
                } else {
                    Err(BidsError::Config(format!(
                        "'{name_or_path}' is not a valid config name or path"
                    )))
                }
            }
        }
    }

    /// Get mutable references to all entity definitions.
    pub fn entities_mut(&mut self) -> &mut Vec<Entity> {
        &mut self.entities
    }

    /// Look up an entity definition by name.
    #[must_use]
    pub fn get_entity(&self, name: &str) -> Option<&Entity> {
        self.entities.iter().find(|e| e.name == name)
    }

    /// Returns the number of entity definitions in this config.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

impl std::fmt::Display for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Config('{}', {} entities",
            self.name,
            self.entities.len()
        )?;
        if let Some(patterns) = &self.default_path_patterns {
            write!(f, ", {} patterns", patterns.len())?;
        }
        write!(f, ")")
    }
}
