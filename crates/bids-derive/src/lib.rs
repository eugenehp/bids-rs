#![deny(unsafe_code)]
//! BIDS derivatives dataset management.
//!
//! Provides utilities for working with BIDS derivative datasets — pipeline
//! outputs that are stored alongside raw data. Derivative datasets must
//! contain a `dataset_description.json` with `DatasetType: "derivative"`
//! and a `GeneratedBy` field identifying the producing pipeline.
//!
//! The [`DerivativeManager`] tracks multiple derivative datasets by pipeline
//! name and provides scoped access.

use bids_core::error::{BidsError, Result};
use bids_layout::BidsLayout;
use std::collections::HashMap;
use std::path::Path;

/// Manages derivative datasets attached to a raw BIDS layout.
pub struct DerivativeManager {
    derivatives: HashMap<String, BidsLayout>,
}

impl DerivativeManager {
    pub fn new() -> Self {
        Self { derivatives: HashMap::new() }
    }

    /// Add a derivatives directory.
    pub fn add(&mut self, path: &Path) -> Result<()> {
        let desc_path = path.join("dataset_description.json");
        if desc_path.exists() {
            let name = bids_validate::validate_derivative_path(path)?;
            if self.derivatives.contains_key(&name) {
                return Err(BidsError::DerivativesValidation(
                    format!("Pipeline '{name}' already added")));
            }
            let layout = BidsLayout::builder(path)
                .validate(false)
                .is_derivative(true)
                .config(vec!["bids".into(), "derivatives".into()])
                .build()?;
            self.derivatives.insert(name, layout);
        } else {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    let sub_desc = entry.path().join("dataset_description.json");
                    if sub_desc.exists() {
                        self.add(&entry.path())?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&BidsLayout> { self.derivatives.get(name) }
    pub fn names(&self) -> Vec<&str> { self.derivatives.keys().map(std::string::String::as_str).collect() }
    pub fn all(&self) -> &HashMap<String, BidsLayout> { &self.derivatives }
}

impl Default for DerivativeManager {
    fn default() -> Self { Self::new() }
}
