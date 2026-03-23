//! HED (Hierarchical Event Descriptors) tag parsing.
//!
//! BIDS events files may include a `HED` column containing HED tags that
//! describe events in a structured, hierarchical vocabulary. This module
//! provides basic parsing and access to HED tag strings.
//!
//! See: <https://www.hedtags.org/> and
//! <https://bids-specification.readthedocs.io/en/stable/appendices/hed.html>
//!
//! # Example
//!
//! ```
//! use bids_core::hed::{parse_hed_string, HedTag};
//!
//! let tags = parse_hed_string("Sensory-event, Visual-presentation, (Item/Object/Man-made/Vehicle/Car, Color/Red)");
//! assert_eq!(tags.len(), 3);
//! assert_eq!(tags[0].tag, "Sensory-event");
//! assert!(tags[2].is_group());
//! ```

/// A single HED tag or tag group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HedTag {
    /// The tag string (e.g., `"Sensory-event"` or a group like `"(A, B)"`).
    pub tag: String,
    /// If this is a group `(...)`, the child tags within the parentheses.
    pub children: Vec<HedTag>,
}

impl HedTag {
    /// Create a simple (non-group) tag.
    #[must_use]
    pub fn simple(tag: &str) -> Self {
        Self {
            tag: tag.trim().to_string(),
            children: Vec::new(),
        }
    }

    /// Create a group tag containing child tags.
    #[must_use]
    pub fn group(children: Vec<HedTag>) -> Self {
        let tag = format!(
            "({})",
            children
                .iter()
                .map(|c| c.tag.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        Self { tag, children }
    }

    /// Returns `true` if this is a tag group (has children).
    #[must_use]
    pub fn is_group(&self) -> bool {
        !self.children.is_empty()
    }

    /// Get the leaf tag name (last component of a hierarchical path).
    ///
    /// For `"Item/Object/Man-made/Vehicle/Car"` returns `"Car"`.
    #[must_use]
    pub fn leaf(&self) -> &str {
        self.tag.rsplit('/').next().unwrap_or(&self.tag)
    }

    /// Get all path components of a hierarchical tag.
    ///
    /// For `"Item/Object/Vehicle/Car"` returns `["Item", "Object", "Vehicle", "Car"]`.
    #[must_use]
    pub fn components(&self) -> Vec<&str> {
        self.tag.split('/').collect()
    }

    /// Check if this tag starts with a given prefix path.
    #[must_use]
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.tag.starts_with(prefix)
    }
}

impl std::fmt::Display for HedTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.tag)
    }
}

/// Parse a HED annotation string into a list of tags and tag groups.
///
/// HED strings use commas to separate tags, and parentheses to group
/// related tags. Hierarchical levels are separated by `/`.
///
/// # Example
///
/// ```
/// use bids_core::hed::parse_hed_string;
///
/// let tags = parse_hed_string("Sensory-event, (Item/Object/Car, Color/Red)");
/// assert_eq!(tags.len(), 2);
/// assert!(!tags[0].is_group());
/// assert!(tags[1].is_group());
/// assert_eq!(tags[1].children.len(), 2);
/// ```
#[must_use]
pub fn parse_hed_string(hed: &str) -> Vec<HedTag> {
    let hed = hed.trim();
    if hed.is_empty() {
        return Vec::new();
    }

    let mut tags = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;

    for (i, ch) in hed.char_indices() {
        match ch {
            '(' => {
                if depth == 0 {
                    // Flush any pending simple tag before the group
                    let before = hed[start..i].trim();
                    if !before.is_empty() && before != "," {
                        for part in before.split(',') {
                            let part = part.trim();
                            if !part.is_empty() {
                                tags.push(HedTag::simple(part));
                            }
                        }
                    }
                    start = i + 1;
                }
                depth += 1;
            }
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    // Parse the group contents
                    let group_str = &hed[start..i];
                    let children: Vec<HedTag> = group_str
                        .split(',')
                        .map(|s| s.trim())
                        .filter(|s| !s.is_empty())
                        .map(HedTag::simple)
                        .collect();
                    tags.push(HedTag::group(children));
                    start = i + 1;
                }
            }
            _ => {}
        }
    }

    // Handle remaining text after last group/comma
    let remaining = hed[start..].trim();
    if !remaining.is_empty() {
        for part in remaining.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                tags.push(HedTag::simple(part));
            }
        }
    }

    tags
}

/// Extract all HED tags from an events TSV column as a flat list of tag strings.
///
/// This is useful for frequency analysis or filtering events by HED tag.
#[must_use]
pub fn extract_all_tags(hed_strings: &[&str]) -> Vec<String> {
    let mut all = Vec::new();
    for s in hed_strings {
        for tag in parse_hed_string(s) {
            if tag.is_group() {
                for child in &tag.children {
                    all.push(child.tag.clone());
                }
            } else {
                all.push(tag.tag);
            }
        }
    }
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tags() {
        let tags = parse_hed_string("Sensory-event, Visual-presentation");
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].tag, "Sensory-event");
        assert_eq!(tags[1].tag, "Visual-presentation");
        assert!(!tags[0].is_group());
    }

    #[test]
    fn test_group() {
        let tags = parse_hed_string("(Item/Object/Car, Color/Red)");
        assert_eq!(tags.len(), 1);
        assert!(tags[0].is_group());
        assert_eq!(tags[0].children.len(), 2);
        assert_eq!(tags[0].children[0].tag, "Item/Object/Car");
        assert_eq!(tags[0].children[0].leaf(), "Car");
    }

    #[test]
    fn test_mixed() {
        let tags = parse_hed_string("Sensory-event, (Item/Car, Color/Red), Agent-action");
        assert_eq!(tags.len(), 3);
        assert!(!tags[0].is_group());
        assert!(tags[1].is_group());
        assert!(!tags[2].is_group());
    }

    #[test]
    fn test_empty() {
        assert!(parse_hed_string("").is_empty());
        assert!(parse_hed_string("  ").is_empty());
    }

    #[test]
    fn test_hierarchical() {
        let tag = HedTag::simple("Item/Object/Man-made/Vehicle/Car");
        assert_eq!(tag.leaf(), "Car");
        assert_eq!(
            tag.components(),
            vec!["Item", "Object", "Man-made", "Vehicle", "Car"]
        );
        assert!(tag.starts_with("Item/Object"));
    }

    #[test]
    fn test_extract_all() {
        let strings = vec!["Sensory-event, Visual", "(Motor, Hand)"];
        let all = extract_all_tags(&strings);
        assert_eq!(all, vec!["Sensory-event", "Visual", "Motor", "Hand"]);
    }
}
