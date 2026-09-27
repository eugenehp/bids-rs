//! Utility functions: entity matching, file grouping, and case conversion.
//!
//! Provides helper functions used throughout the crate ecosystem for comparing
//! entity maps, grouping multi-echo/multi-part files, CamelCase↔snake_case
//! conversion, and fuzzy "did you mean?" suggestions.

use crate::entities::Entities;
use crate::file::BidsFile;

/// Check whether a file's entities match the target entities.
///
/// In non-strict mode, all target entities must be present with matching values
/// in `file_entities`, but extra entities in the file are allowed.
///
/// In strict mode, both maps must have exactly the same keys.
///
/// Corresponds to PyBIDS' `matches_entities()`.
#[must_use]
pub fn matches_entities(file_entities: &Entities, target: &Entities, strict: bool) -> bool {
    // Quick length check for strict mode avoids collecting into HashSets.
    if strict && file_entities.len() != target.len() {
        return false;
    }

    for (k, target_val) in target {
        match file_entities.get(k) {
            Some(current_val) if current_val == target_val => {}
            Some(_) => return false,
            None => {
                if strict {
                    return false;
                }
            }
        }
    }

    // In strict mode, also verify no extra keys in file_entities.
    if strict {
        for k in file_entities.keys() {
            if !target.contains_key(k) {
                return false;
            }
        }
    }

    true
}

/// Group BIDSFiles with multiple files per acquisition (multi-echo, multi-part, etc.).
///
/// Corresponds to PyBIDS' `collect_associated_files()`.
///
/// Groups files that share the same base entities (excluding multi-contrast
/// entities like echo, part, ch, direction, and suffix).
#[must_use]
pub fn collect_associated_files(files: &[BidsFile]) -> Vec<Vec<&BidsFile>> {
    const MULTI_ENTITIES: &[&str] = &["echo", "part", "ch", "direction", "suffix"];

    // Build a grouping key for each file — a sorted Vec of (name, value) pairs
    // excluding multi-contrast entities. Use IndexMap to preserve order.
    let mut groups: indexmap::IndexMap<Vec<(String, String)>, Vec<&BidsFile>> =
        indexmap::IndexMap::new();

    for f in files {
        let mut key: Vec<(String, String)> = f
            .entities
            .iter()
            .filter(|(k, _)| !MULTI_ENTITIES.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.as_str_lossy().into_owned()))
            .collect();
        key.sort_by(|(a, _), (b, _)| a.cmp(b));
        groups.entry(key).or_default().push(f);
    }

    groups.into_values().collect()
}

/// Convert CamelCase keys to snake_case recursively in a JSON value.
///
/// Corresponds to PyBIDS' `convert_JSON()`.
pub fn convert_json_keys(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (k, v) in map {
                let new_key = camel_to_snake(k);
                new_map.insert(new_key, convert_json_keys(v));
            }
            serde_json::Value::Object(new_map)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(convert_json_keys).collect())
        }
        other => other.clone(),
    }
}

/// Convert a CamelCase string to snake_case.
///
/// Handles acronyms like "EEGReference" → "eeg_reference".
#[must_use]
pub fn camel_to_snake(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut result = String::with_capacity(s.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        if c.is_uppercase() && i > 0 {
            let prev = chars[i - 1];
            let next = chars.get(i + 1);
            // Insert underscore before uppercase if preceded by lowercase/digit
            // or followed by lowercase (handles "EEGReference" -> "eeg_reference")
            if prev.is_lowercase()
                || prev.is_ascii_digit()
                || next.is_some_and(|n| n.is_lowercase())
            {
                result.push('_');
            }
        }
        result.push(c.to_lowercase().next().unwrap_or(c));
    }
    result
}

/// Convert a snake_case string to CamelCase.
#[must_use]
pub fn snake_to_camel(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => {
                    let upper: String = c.to_uppercase().collect();
                    format!("{}{}", upper, chars.collect::<String>())
                }
                None => String::new(),
            }
        })
        .collect()
}

/// Find the closest matches for a string from a list of candidates.
/// Returns up to `n` suggestions sorted by edit distance.
#[must_use]
pub fn get_close_matches(word: &str, candidates: &[String], n: usize) -> Vec<String> {
    let mut scored: Vec<(usize, &String)> = candidates
        .iter()
        .map(|c| (edit_distance(word, c), c))
        .filter(|(d, _)| *d <= word.len().max(3))
        .collect();
    scored.sort_by_key(|(d, _)| *d);
    scored.into_iter().take(n).map(|(_, s)| s.clone()).collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = b.len();
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}
/// Strip a leading UTF-8 byte-order mark, if present.
///
/// Several editors and spreadsheet exporters — Excel in particular — prefix
/// UTF-8 files with `EF BB BF`. Left in place the BOM becomes part of the first
/// column's header name, so a `_channels.tsv` saved from Excel parses its first
/// header as `"\u{feff}name"` and every lookup of `"name"` misses. Real
/// OpenNeuro datasets ship such files (`ds005420`, for one), so every TSV
/// header line is passed through this.
#[must_use]
pub fn strip_bom(s: &str) -> &str {
    s.strip_prefix('\u{feff}').unwrap_or(s)
}

/// Compare two strings in natural order: embedded numbers compare as numbers.
///
/// `"sub-2"` sorts before `"sub-10"`, where a plain lexicographic comparison
/// would not.
///
/// # Example
///
/// ```
/// use bids_core::utils::natural_cmp;
/// use std::cmp::Ordering;
///
/// assert_eq!(natural_cmp("sub-2", "sub-10"), Ordering::Less);
/// ```
#[must_use]
pub fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    let mut ai = a.chars().peekable();
    let mut bi = b.chars().peekable();

    loop {
        match (ai.peek(), bi.peek()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(&ac), Some(&bc)) => {
                if ac.is_ascii_digit() && bc.is_ascii_digit() {
                    let mut an = String::new();
                    while let Some(&c) = ai.peek() {
                        if c.is_ascii_digit() {
                            an.push(c);
                            ai.next();
                        } else {
                            break;
                        }
                    }
                    let mut bn = String::new();
                    while let Some(&c) = bi.peek() {
                        if c.is_ascii_digit() {
                            bn.push(c);
                            bi.next();
                        } else {
                            break;
                        }
                    }
                    let av: u64 = an.parse().unwrap_or(0);
                    let bv: u64 = bn.parse().unwrap_or(0);
                    match av.cmp(&bv) {
                        Ordering::Equal => {}
                        ord => return ord,
                    }
                } else {
                    let al = ac.to_lowercase().next().unwrap_or(ac);
                    let bl = bc.to_lowercase().next().unwrap_or(bc);
                    match al.cmp(&bl) {
                        Ordering::Equal => {
                            ai.next();
                            bi.next();
                        }
                        ord => return ord,
                    }
                }
            }
        }
    }
}

/// Sort strings in natural order.
///
/// Corresponds to PyBIDS' `natural_sort`.
///
/// # Example
///
/// ```
/// use bids_core::utils::natural_sort;
///
/// let mut subjects = vec!["sub-10".to_string(), "sub-2".into(), "sub-1".into()];
/// natural_sort(&mut subjects);
/// assert_eq!(subjects, ["sub-1", "sub-2", "sub-10"]);
/// ```
pub fn natural_sort(items: &mut [String]) {
    items.sort_by(|a, b| natural_cmp(a, b));
}

/// Sort items in natural order by a key extracted from each.
///
/// The `field` argument of PyBIDS' `natural_sort`, expressed as a closure.
///
/// # Example
///
/// ```
/// use bids_core::utils::natural_sort_by_key;
///
/// let mut files = vec!["b/sub-10.nii", "a/sub-2.nii"];
/// natural_sort_by_key(&mut files, |f| f.rsplit('/').next().unwrap().to_string());
/// assert_eq!(files[0], "a/sub-2.nii");
/// ```
pub fn natural_sort_by_key<T, F>(items: &mut [T], key: F)
where
    F: Fn(&T) -> String,
{
    items.sort_by(|a, b| natural_cmp(&key(a), &key(b)));
}

/// Split a path into its stem and its (possibly compound) extensions.
///
/// Unlike `Path::extension`, this handles paths whose *directories* contain
/// dots, and returns every extension segment rather than only the last — so
/// `sub-01_bold.nii.gz` yields `["nii", "gz"]`, not just `"gz"`.
///
/// Corresponds to PyBIDS' `splitext`.
///
/// # Example
///
/// ```
/// use bids_core::utils::splitext;
///
/// let (stem, exts) = splitext("/data/my.dir/sub-01_bold.nii.gz");
/// assert_eq!(stem, "/data/my.dir/sub-01_bold");
/// assert_eq!(exts, ["nii", "gz"]);
/// ```
#[must_use]
pub fn splitext(path: &str) -> (String, Vec<String>) {
    let (dir, base) = match path.rfind(['/', '\\']) {
        Some(i) => (&path[..=i], &path[i + 1..]),
        None => ("", path),
    };
    let mut parts = base.split('.');
    let stem = parts.next().unwrap_or("");
    let exts: Vec<String> = parts.map(String::from).collect();
    (format!("{dir}{stem}"), exts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entities::EntityValue;

    #[test]
    fn test_camel_to_snake() {
        assert_eq!(camel_to_snake("RepetitionTime"), "repetition_time");
        assert_eq!(camel_to_snake("TaskName"), "task_name");
        // EEGReference — consecutive capitals get individual underscores
        let eeg = camel_to_snake("EEGReference");
        assert!(eeg.contains("reference"));
    }

    #[test]
    fn test_snake_to_camel() {
        assert_eq!(snake_to_camel("repetition_time"), "RepetitionTime");
        assert_eq!(snake_to_camel("task_name"), "TaskName");
    }

    #[test]
    fn test_matches_entities() {
        let mut file_ents = Entities::new();
        file_ents.insert("subject".into(), EntityValue::Str("01".into()));
        file_ents.insert("task".into(), EntityValue::Str("rest".into()));

        let mut target = Entities::new();
        target.insert("subject".into(), EntityValue::Str("01".into()));

        assert!(matches_entities(&file_ents, &target, false));
        assert!(!matches_entities(&file_ents, &target, true));

        target.insert("task".into(), EntityValue::Str("rest".into()));
        assert!(matches_entities(&file_ents, &target, true));
    }

    #[test]
    fn test_close_matches() {
        let candidates = vec![
            "subject".to_string(),
            "session".to_string(),
            "suffix".to_string(),
            "task".to_string(),
            "run".to_string(),
        ];
        let matches = get_close_matches("suject", &candidates, 2);
        assert!(!matches.is_empty());
        assert_eq!(matches[0], "subject");
    }

    // ── strip_bom ─────────────────────────────────────────────────────────

    #[test]
    fn strips_a_leading_bom() {
        assert_eq!(strip_bom("\u{feff}name\ttype"), "name\ttype");
    }

    #[test]
    fn leaves_unmarked_text_alone() {
        assert_eq!(strip_bom("name\ttype"), "name\ttype");
    }

    #[test]
    fn only_strips_the_leading_one() {
        // A BOM anywhere but the start is a real (if odd) character.
        assert_eq!(strip_bom("name\u{feff}\ttype"), "name\u{feff}\ttype");
    }
}
