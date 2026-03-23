//! Fluent query builder for filtering BIDS files.
//!
//! [`GetBuilder`] provides a chainable API accessed via `layout.get()` to
//! filter files by entity values, regex, existence, and scope.

use bids_core::error::{BidsError, Result};
use bids_core::file::BidsFile;
use bids_core::utils::get_close_matches;
use std::path::PathBuf;

use crate::layout::BidsLayout;
use crate::query::{ReturnType, Scope};

/// How to handle unrecognized entity names in query filters.
///
/// By default, [`GetBuilder`] returns an error with "did you mean?"
/// suggestions when a filter references an entity that doesn't exist in
/// the dataset's index.
///
/// # Example
///
/// ```no_run
/// # use bids_layout::{BidsLayout, InvalidFilters};
/// # let layout = BidsLayout::new("/path").unwrap();
/// let files = layout.get()
///     .invalid_filters(InvalidFilters::Drop)  // silently ignore bad filters
///     .filter("suject", "01")                 // typo — will be dropped
///     .collect().unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidFilters {
    /// Return an error with suggestions for close matches (default).
    Error,
    /// Silently drop unrecognized filters.
    Drop,
    /// Pass unrecognized filters through without validation.
    Allow,
}

/// Fluent query builder for [`BidsLayout::get()`](crate::BidsLayout::get).
///
/// Provides a chainable API for filtering BIDS files by entity values,
/// regex patterns, existence checks, and scope. Execute the query with
/// [`collect()`](Self::collect) (for `BidsFile` objects),
/// [`return_paths()`](Self::return_paths) (for `PathBuf`s), or
/// [`return_unique()`](Self::return_unique) (for unique entity values).
///
/// # Examples
///
/// ```no_run
/// # use bids_layout::BidsLayout;
/// # let layout = BidsLayout::new("/path").unwrap();
/// // Basic entity filters
/// let files = layout.get()
///     .suffix("eeg")
///     .extension(".edf")
///     .subject("01")
///     .collect().unwrap();
///
/// // Multi-value and regex filters
/// let files = layout.get()
///     .filter_any("subject", &["01", "02", "03"])
///     .filter_regex("suffix", "(bold|sbref)")
///     .collect().unwrap();
///
/// // Existence checks
/// let files = layout.get()
///     .query_any("session")    // must have a session
///     .query_none("recording") // must NOT have a recording
///     .collect().unwrap();
///
/// // Get unique entity values
/// let tasks = layout.get()
///     .suffix("bold")
///     .return_unique("task").unwrap();
/// ```
pub struct GetBuilder<'a> {
    pub(crate) layout: &'a BidsLayout,
    pub(crate) filters: Vec<(String, Vec<String>, bool)>,
    pub(crate) return_type: ReturnType,
    pub(crate) target: Option<String>,
    pub(crate) scope: Scope,
    pub(crate) invalid_filters: InvalidFilters,
}

impl<'a> GetBuilder<'a> {
    pub(crate) fn new(layout: &'a BidsLayout) -> Self {
        Self {
            layout,
            filters: Vec::new(),
            return_type: ReturnType::Object,
            target: None,
            scope: Scope::All,
            invalid_filters: InvalidFilters::Error,
        }
    }

    // ─── Entity filters ───
    #[must_use]
    pub fn subject(self, v: &str) -> Self {
        self.filter("subject", v)
    }
    #[must_use]
    pub fn session(self, v: &str) -> Self {
        self.filter("session", v)
    }
    #[must_use]
    pub fn task(self, v: &str) -> Self {
        self.filter("task", v)
    }
    #[must_use]
    pub fn run(self, v: &str) -> Self {
        self.filter("run", v)
    }
    #[must_use]
    pub fn datatype(self, v: &str) -> Self {
        self.filter("datatype", v)
    }
    #[must_use]
    pub fn acquisition(self, v: &str) -> Self {
        self.filter("acquisition", v)
    }
    #[must_use]
    pub fn recording(self, v: &str) -> Self {
        self.filter("recording", v)
    }
    #[must_use]
    pub fn space(self, v: &str) -> Self {
        self.filter("space", v)
    }
    #[must_use]
    pub fn suffix(self, v: &str) -> Self {
        self.filter("suffix", v)
    }

    #[must_use]
    pub fn extension(self, value: &str) -> Self {
        let v = if value.starts_with('.') {
            value.to_string()
        } else {
            format!(".{value}")
        };
        self.filter_owned("extension", vec![v])
    }

    /// Set scope: "all", "raw", "derivatives", "self", or a pipeline name.
    #[must_use]
    pub fn scope(mut self, scope: &str) -> Self {
        self.scope = Scope::parse(scope);
        self
    }

    /// Set invalid filter handling.
    #[must_use]
    pub fn invalid_filters(mut self, mode: InvalidFilters) -> Self {
        self.invalid_filters = mode;
        self
    }

    /// Filter by entity name and exact value.
    #[must_use]
    pub fn filter(mut self, entity: &str, value: &str) -> Self {
        self.filters
            .push((entity.into(), vec![value.into()], false));
        self
    }

    /// Filter by entity with multiple allowed values.
    #[must_use]
    pub fn filter_any(mut self, entity: &str, values: &[&str]) -> Self {
        self.filters.push((
            entity.into(),
            values
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            false,
        ));
        self
    }

    /// Filter by entity with regex.
    #[must_use]
    pub fn filter_regex(mut self, entity: &str, pattern: &str) -> Self {
        self.filters
            .push((entity.into(), vec![pattern.into()], true));
        self
    }

    /// Require entity to exist (any value).
    #[must_use]
    pub fn query_any(mut self, entity: &str) -> Self {
        self.filters
            .push((entity.into(), vec!["__ANY__".into()], false));
        self
    }

    /// Require entity to NOT exist.
    #[must_use]
    pub fn query_none(mut self, entity: &str) -> Self {
        self.filters
            .push((entity.into(), vec!["__NONE__".into()], false));
        self
    }

    #[must_use]
    fn filter_owned(mut self, entity: &str, values: Vec<String>) -> Self {
        self.filters.push((entity.into(), values, false));
        self
    }

    #[must_use]
    pub fn return_filenames(mut self) -> Self {
        self.return_type = ReturnType::Filename;
        self
    }
    #[must_use]
    pub fn return_ids(mut self, target: &str) -> Self {
        self.return_type = ReturnType::Id;
        self.target = Some(target.into());
        self
    }
    #[must_use]
    pub fn return_dirs(mut self, target: &str) -> Self {
        self.return_type = ReturnType::Dir;
        self.target = Some(target.into());
        self
    }

    // ─── Validation ───

    fn validate_filters(&self) -> Result<Vec<(String, Vec<String>, bool)>> {
        if self.invalid_filters == InvalidFilters::Allow {
            return Ok(self.filters.clone());
        }
        let entities = self.layout.get_entities()?;
        let entity_set: std::collections::HashSet<&str> =
            entities.iter().map(std::string::String::as_str).collect();
        let mut validated = Vec::new();
        for (name, values, regex) in &self.filters {
            if !entity_set.contains(name.as_str()) {
                match self.invalid_filters {
                    InvalidFilters::Error => {
                        let suggestions = get_close_matches(name, &entities, 3);
                        let mut msg = format!("'{name}' is not a recognized entity.");
                        if !suggestions.is_empty() {
                            msg.push_str(&format!(" Did you mean {suggestions:?}?"));
                        }
                        return Err(BidsError::InvalidFilter(msg));
                    }
                    InvalidFilters::Drop => continue,
                    InvalidFilters::Allow => {}
                }
            }
            validated.push((name.clone(), values.clone(), *regex));
        }
        Ok(validated)
    }

    // ─── Execution ───

    /// Execute query, returning `BidsFile` objects.
    pub fn collect(self) -> Result<Vec<BidsFile>> {
        let filters = self.validate_filters()?;
        let paths = self.layout.query_files_internal(&filters, &self.scope)?;
        let mut files: Vec<BidsFile> = paths
            .iter()
            .filter_map(|p| self.layout.reconstruct_file(p).ok())
            .collect();
        files.sort();
        Ok(files)
    }

    /// Execute query, returning file paths.
    pub fn return_paths(self) -> Result<Vec<PathBuf>> {
        let filters = self.validate_filters()?;
        let paths = self.layout.query_files_internal(&filters, &self.scope)?;
        let mut result: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
        result.sort();
        Ok(result)
    }

    /// Execute query, returning unique values for a target entity.
    pub fn return_unique(self, target: &str) -> Result<Vec<String>> {
        let filters = self.validate_filters()?;
        let paths = self.layout.query_files_internal(&filters, &self.scope)?;
        let mut seen = std::collections::HashSet::new();
        let mut values = Vec::new();
        for path in &paths {
            for (name, value, _, _) in self.layout.db().get_tags(path)? {
                if name == target && seen.insert(value.clone()) {
                    values.push(value);
                }
            }
        }
        values.sort();
        Ok(values)
    }

    /// Execute query, returning directories for a target entity.
    pub fn return_directories(self, target: &str) -> Result<Vec<String>> {
        let filters = self.validate_filters()?;
        self.layout.db().query_directories(target, &filters)
    }

    /// Deprecated: use [`collect()`](Self::collect) instead.
    #[deprecated(since = "0.2.0", note = "renamed to `collect()` for clarity")]
    pub fn returns(self) -> Result<Vec<BidsFile>> {
        self.collect()
    }
}
