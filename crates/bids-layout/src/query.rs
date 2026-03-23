//! Query types: filters, scopes, return types, and special query values.
//!
//! Defines the building blocks used by [`GetBuilder`](crate::GetBuilder) to
//! express dataset queries: exact matching, regex, existence checks (`Any`/`None`),
//! scope selection (raw/derivatives/pipeline), and return type (objects/paths/IDs).

/// Special query values for entity filtering.
///
/// These correspond to PyBIDS' `Query.NONE`, `Query.ANY`, and `Query.OPTIONAL`
/// sentinel values that modify how entity filters behave beyond simple value
/// matching.
///
/// # Example
///
/// ```no_run
/// # use bids_layout::{BidsLayout, Query, QueryFilter};
/// # let layout = BidsLayout::new("/path").unwrap();
/// // Find files that have a session entity (any value)
/// let files = layout.get().query_any("session").collect().unwrap();
///
/// // Find files that do NOT have a session entity
/// let files = layout.get().query_none("session").collect().unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Query {
    /// The entity must **not** be defined on the file. Files that have any
    /// value for this entity are excluded.
    None,
    /// The entity must be defined (with any value). Files missing this entity
    /// are excluded.
    Any,
    /// The entity is optional — no filtering is applied regardless of whether
    /// the entity is present or absent.
    Optional,
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::None => write!(f, "Query::None"),
            Query::Any => write!(f, "Query::Any"),
            Query::Optional => write!(f, "Query::Optional"),
        }
    }
}

/// A filter for querying files by entity values.
#[derive(Debug, Clone)]
pub struct QueryFilter {
    pub entity: String,
    pub values: Vec<String>,
    pub regex: bool,
    pub query: Option<Query>,
}

impl QueryFilter {
    /// Convert to the internal tuple representation `(entity, values, is_regex)`.
    #[must_use]
    pub fn to_tuple(&self) -> (String, Vec<String>, bool) {
        (self.entity.clone(), self.values.clone(), self.regex)
    }

    /// Convert a slice of `QueryFilter`s to internal tuple representation.
    pub fn to_tuples(filters: &[QueryFilter]) -> Vec<(String, Vec<String>, bool)> {
        filters.iter().map(QueryFilter::to_tuple).collect()
    }

    #[must_use]
    pub fn eq(entity: &str, value: &str) -> Self {
        Self {
            entity: entity.into(),
            values: vec![value.into()],
            regex: false,
            query: None,
        }
    }
    #[must_use]
    pub fn one_of(entity: &str, values: &[&str]) -> Self {
        Self {
            entity: entity.into(),
            values: values
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            regex: false,
            query: None,
        }
    }
    #[must_use]
    pub fn regex(entity: &str, pattern: &str) -> Self {
        Self {
            entity: entity.into(),
            values: vec![pattern.into()],
            regex: true,
            query: None,
        }
    }
    #[must_use]
    pub fn any(entity: &str) -> Self {
        Self {
            entity: entity.into(),
            values: vec!["__ANY__".into()],
            regex: false,
            query: Some(Query::Any),
        }
    }
    #[must_use]
    pub fn none(entity: &str) -> Self {
        Self {
            entity: entity.into(),
            values: vec!["__NONE__".into()],
            regex: false,
            query: Some(Query::None),
        }
    }
    #[must_use]
    pub fn optional(entity: &str) -> Self {
        Self {
            entity: entity.into(),
            values: vec!["__OPTIONAL__".into()],
            regex: false,
            query: Some(Query::Optional),
        }
    }
}

/// Return type for get() queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnType {
    Object,
    Filename,
    Id,
    Dir,
}

/// Scope for queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    All,
    Raw,
    Derivatives,
    Self_,
    Pipeline(String),
}

impl Scope {
    /// Parse scope from a string. Prefer using `str::parse::<Scope>()`.
    pub fn parse(s: &str) -> Self {
        match s {
            "all" => Scope::All,
            "raw" => Scope::Raw,
            "derivatives" => Scope::Derivatives,
            "self" => Scope::Self_,
            other => Scope::Pipeline(other.to_string()),
        }
    }
}

impl std::str::FromStr for Scope {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(Scope::parse(s))
    }
}

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scope::All => write!(f, "all"),
            Scope::Raw => write!(f, "raw"),
            Scope::Derivatives => write!(f, "derivatives"),
            Scope::Self_ => write!(f, "self"),
            Scope::Pipeline(name) => write!(f, "pipeline:{name}"),
        }
    }
}

impl From<(String, Vec<String>, bool)> for QueryFilter {
    fn from((entity, values, regex): (String, Vec<String>, bool)) -> Self {
        Self {
            entity,
            values,
            regex,
            query: None,
        }
    }
}

impl From<QueryFilter> for (String, Vec<String>, bool) {
    fn from(f: QueryFilter) -> Self {
        (f.entity, f.values, f.regex)
    }
}

impl std::fmt::Display for QueryFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}=", self.entity)?;
        if self.regex {
            write!(f, "/{}/", self.values.join("|"))
        } else if self.values.len() == 1 {
            write!(f, "{}", self.values[0])
        } else {
            write!(f, "[{}]", self.values.join(", "))
        }
    }
}
