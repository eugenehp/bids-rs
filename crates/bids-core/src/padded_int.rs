//! Zero-padded integer type that preserves original formatting.
//!
//! BIDS uses zero-padded integers in entity values (e.g., `sub-01`, `run-002`).
//! [`PaddedInt`] stores both the numeric value and the original string so that
//! comparisons use the number but display preserves the padding.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Integer type that preserves zero-padding.
///
/// Acts like an i64 in comparisons and arithmetic, but string formatting
/// preserves the original zero-padding.
///
/// ```
/// use bids_core::PaddedInt;
///
/// let p = PaddedInt::new("02");
/// assert_eq!(p.value(), 2);
/// assert_eq!(p.to_string(), "02");
/// assert_eq!(p, PaddedInt::from(2));
/// ```
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaddedInt {
    value: i64,
    formatted: String,
}

impl PaddedInt {
    /// Parse a zero-padded integer string (e.g., `"02"`, `"001"`).
    pub fn new(s: &str) -> Self {
        let value = s.parse::<i64>().unwrap_or(0);
        Self {
            value,
            formatted: s.to_string(),
        }
    }

    /// The numeric value (ignoring padding).
    #[must_use]
    pub fn value(&self) -> i64 {
        self.value
    }
}

impl From<i64> for PaddedInt {
    fn from(v: i64) -> Self {
        Self {
            value: v,
            formatted: v.to_string(),
        }
    }
}

impl From<i32> for PaddedInt {
    fn from(v: i32) -> Self {
        Self::from(v as i64)
    }
}

impl fmt::Display for PaddedInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.formatted)
    }
}

impl PartialEq for PaddedInt {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for PaddedInt {}

impl PartialOrd for PaddedInt {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PaddedInt {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl std::hash::Hash for PaddedInt {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialEq<i64> for PaddedInt {
    fn eq(&self, other: &i64) -> bool {
        self.value == *other
    }
}

impl PartialEq<PaddedInt> for i64 {
    fn eq(&self, other: &PaddedInt) -> bool {
        *self == other.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_padded_int() {
        let p = PaddedInt::new("02");
        assert_eq!(p.value(), 2);
        assert_eq!(p.to_string(), "02");
        assert_eq!(p, PaddedInt::from(2));
        assert!(p == 2i64);

        let p1 = PaddedInt::new("001");
        let p2 = PaddedInt::new("01");
        assert_eq!(p1, p2);
        assert_eq!(p1.to_string(), "001");
        assert_eq!(p2.to_string(), "01");
    }
}
