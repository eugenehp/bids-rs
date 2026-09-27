//! Variable access directly from a [`BidsLayout`].
//!
//! `bids-layout` cannot depend on this crate — that would be a dependency
//! cycle — so PyBIDS' `BIDSLayout.get_collections()` is provided here as an
//! extension trait. Bring [`LayoutVariables`] into scope and the method is
//! available on any layout.

use bids_core::entities::StringEntities;
use bids_core::error::Result;
use bids_layout::BidsLayout;

use crate::collections::{RunVariableCollection, VariableCollection};
use crate::io::{VariableType, load_variables};

/// Variable collections read straight from a layout.
///
/// Corresponds to PyBIDS' `BIDSLayout.get_collections()`.
///
/// # Example
///
/// ```no_run
/// use bids_layout::BidsLayout;
/// use bids_variables::LayoutVariables;
///
/// let layout = BidsLayout::new("/path/to/dataset").unwrap();
/// let collections = layout.get_collections("run", None, None, false).unwrap();
/// for c in &collections {
///     println!("{:?}", c.names());
/// }
/// ```
pub trait LayoutVariables {
    /// Variable collections at `level` (`"run"`, `"session"`, `"subject"`, or
    /// `"dataset"`).
    ///
    /// `types` restricts which BIDS tabular files are read; `None` selects the
    /// defaults for the level. `filters` restricts which nodes are returned by
    /// entity value. With `merge`, all matching nodes collapse into a single
    /// collection.
    ///
    /// # Errors
    ///
    /// Propagates I/O and parse errors from reading the dataset's tabular files.
    fn get_collections(
        &self,
        level: &str,
        types: Option<&[VariableType]>,
        filters: Option<&StringEntities>,
        merge: bool,
    ) -> Result<Vec<VariableCollection>>;

    /// Run-level collections, keeping sparse and dense variables separate.
    ///
    /// # Errors
    ///
    /// Propagates I/O and parse errors from reading the dataset's tabular files.
    fn get_run_collections(
        &self,
        filters: Option<&StringEntities>,
    ) -> Result<Vec<RunVariableCollection>>;
}

impl LayoutVariables for BidsLayout {
    fn get_collections(
        &self,
        level: &str,
        types: Option<&[VariableType]>,
        filters: Option<&StringEntities>,
        merge: bool,
    ) -> Result<Vec<VariableCollection>> {
        let index = load_variables(self, types, Some(level))?;
        let empty = StringEntities::new();
        Ok(index.get_collections(level, filters.unwrap_or(&empty), merge))
    }

    fn get_run_collections(
        &self,
        filters: Option<&StringEntities>,
    ) -> Result<Vec<RunVariableCollection>> {
        let index = load_variables(self, None, Some("run"))?;
        let empty = StringEntities::new();
        Ok(index.get_run_collections(filters.unwrap_or(&empty)))
    }
}
