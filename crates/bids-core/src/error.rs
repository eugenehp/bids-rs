//! Error types for the BIDS crate ecosystem.
//!
//! All fallible operations across `bids-*` crates return [`Result<T>`], which
//! uses [`BidsError`] as the error type. The error variants cover the full
//! range of failure modes: filesystem I/O, JSON parsing, BIDS validation,
//! entity resolution, query filtering, database operations, and path building.

use thiserror::Error;

/// A specialized `Result` type for BIDS operations.
///
/// This is defined as `std::result::Result<T, BidsError>` and is used
/// throughout the `bids-*` crate ecosystem.
pub type Result<T> = std::result::Result<T, BidsError>;

/// Errors that can occur when working with BIDS datasets.
///
/// This enum covers all failure modes across the crate ecosystem, from
/// low-level I/O errors to high-level BIDS validation failures. It implements
/// `From<std::io::Error>` and `From<serde_json::Error>` for convenient
/// error propagation with `?`.
///
/// # Example
///
/// ```
/// use bids_core::BidsError;
///
/// let err = BidsError::validation("Missing required field");
/// assert!(err.is_validation());
/// assert!(!err.is_io());
/// ```
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum BidsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("BIDS validation error: {0}")]
    Validation(String),

    #[error("BIDS root does not exist: {0}")]
    RootNotFound(String),

    #[error("Missing dataset_description.json in project root")]
    MissingDatasetDescription,

    #[error("Missing mandatory field '{field}' in dataset_description.json")]
    MissingMandatoryField { field: String },

    #[error("Derivatives validation error: {0}")]
    DerivativesValidation(String),

    #[error("Entity error: {0}")]
    Entity(String),

    #[error("Invalid target entity: {0}")]
    InvalidTarget(String),

    #[error("No match found: {0}")]
    NoMatch(String),

    #[error("Invalid filter: {0}")]
    InvalidFilter(String),

    #[error("Conflicting values: {0}")]
    ConflictingValues(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("CSV error: {0}")]
    Csv(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Path building error: {0}")]
    PathBuilding(String),

    #[error("File type error: {0}")]
    FileType(String),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Data format error: {0}")]
    DataFormat(String),

    #[error("Network/download error: {0}")]
    Network(String),

    #[error("API error: {0}")]
    Api(String),
}

impl BidsError {
    /// Create a validation error with a message.
    #[must_use]
    pub fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }

    /// Create an entity error with a message.
    #[must_use]
    pub fn entity(msg: impl Into<String>) -> Self {
        Self::Entity(msg.into())
    }

    /// Create a data format error with a message.
    #[must_use]
    pub fn data_format(msg: impl Into<String>) -> Self {
        Self::DataFormat(msg.into())
    }

    /// Returns `true` if this is an I/O error.
    #[must_use]
    pub fn is_io(&self) -> bool {
        matches!(self, Self::Io(_))
    }

    /// Returns `true` if this is a validation error.
    #[must_use]
    pub fn is_validation(&self) -> bool {
        matches!(self, Self::Validation(_))
    }

    /// Returns `true` if this is a "not found" type error.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        matches!(
            self,
            Self::RootNotFound(_) | Self::MissingDatasetDescription | Self::NoMatch(_)
        )
    }
}
