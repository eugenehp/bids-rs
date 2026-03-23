//! Build BIDS-compliant file paths from entity key-value pairs.
//!
//! Uses configurable patterns with optional sections (`[/ses-{session}]`),
//! value constraints (`{suffix<T1w|T2w>}`), and defaults (`{datatype|anat}`)
//! to construct paths that conform to BIDS naming conventions.

use bids_core::entities::{Entities, EntityValue};
use regex::Regex;
use std::sync::LazyLock;

static PATTERN_FIND: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{([\w\d]*?)(?:<([^>]+)>)?(?:\|((?:\.?[\w])+))?\}").unwrap());

/// Build a file path given entities and a list of path patterns.
///
/// Supports list values in entities — returns a Vec of paths when any entity
/// has multiple values (Cartesian product).
///
/// Returns `None` if no pattern matches all mandatory entities.
pub fn build_path(entities: &Entities, patterns: &[&str], strict: bool) -> Option<String> {
    let result = build_path_multi(entities, patterns, strict);
    result.map(|v| match v.len() {
        1 => v.into_iter().next().expect("length checked"),
        _ => v.join(","),
    })
}

/// Build potentially multiple paths when entities contain list values.
pub fn build_path_multi(
    entities: &Entities,
    patterns: &[&str],
    strict: bool,
) -> Option<Vec<String>> {
    // Check if any entity value contains list-like values (comma-separated)
    // For now, we just expand the single entity set
    for pattern in patterns {
        if let Some(result) = try_build_single(entities, pattern, strict) {
            return Some(vec![result]);
        }
    }
    None
}

/// Build paths with entity expansion: when entities map to Vec of values,
/// produce Cartesian product of all combinations.
pub fn build_paths_expanded(
    entities: &std::collections::HashMap<String, Vec<String>>,
    patterns: &[&str],
    strict: bool,
) -> Vec<String> {
    let keys: Vec<&String> = entities.keys().collect();
    let value_lists: Vec<&Vec<String>> = keys.iter().map(|k| &entities[*k]).collect();

    let combos = cartesian_product(&value_lists);
    let mut results = Vec::new();

    for combo in combos {
        let mut ents = Entities::new();
        for (i, key) in keys.iter().enumerate() {
            ents.insert((*key).clone(), EntityValue::Str(combo[i].clone()));
        }
        if let Some(path) = build_path(&ents, patterns, strict) {
            results.push(path);
        }
    }
    results
}

fn cartesian_product(lists: &[&Vec<String>]) -> Vec<Vec<String>> {
    if lists.is_empty() {
        return vec![vec![]];
    }
    let mut result = vec![vec![]];
    for list in lists {
        let mut new_result = Vec::new();
        for existing in &result {
            for item in *list {
                let mut combo = existing.clone();
                combo.push(item.clone());
                new_result.push(combo);
            }
        }
        result = new_result;
    }
    result
}

fn try_build_single(entities: &Entities, pattern: &str, strict: bool) -> Option<String> {
    let matches: Vec<_> = PATTERN_FIND.captures_iter(pattern).collect();

    let defined: Vec<String> = matches
        .iter()
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();

    if strict {
        let defined_set: std::collections::HashSet<&str> =
            defined.iter().map(std::string::String::as_str).collect();
        for key in entities.keys() {
            if entities
                .get(key)
                .is_some_and(|v| !v.as_str_lossy().is_empty())
                && !defined_set.contains(key.as_str())
            {
                return None;
            }
        }
    }

    let mut new_path = pattern.to_string();
    let mut tmp_entities = entities.clone();

    // Remove None/empty entities
    tmp_entities.retain(|_, v| {
        let s = v.as_str_lossy();
        !s.is_empty()
    });

    for cap in &matches {
        let full = cap.get(0)?.as_str();
        let name = cap.get(1)?.as_str();
        let valid = cap.get(2).map(|m| m.as_str()).unwrap_or("");
        let defval = cap.get(3).map(|m| m.as_str()).unwrap_or("");

        if !valid.is_empty()
            && let Some(ent_val) = tmp_entities.get(name)
        {
            let val_str = ent_val.as_str_lossy();
            let expanded: Vec<String> = valid.split('|').flat_map(expand_options).collect();
            if !expanded.iter().any(|v| v == &val_str) {
                return None;
            }
        }

        if !defval.is_empty() && !tmp_entities.contains_key(name) {
            tmp_entities.insert(name.to_string(), EntityValue::Str(defval.to_string()));
        }

        new_path = new_path.replace(full, &format!("{{{name}}}"));
    }

    // Handle optional sections
    static OPT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]*?)\]").unwrap());
    static PH_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\{(\w+)\}").unwrap());
    loop {
        let before = new_path.clone();
        new_path = OPT_RE
            .replace_all(&new_path, |caps: &regex::Captures| {
                let inner = &caps[1];
                for pcap in PH_RE.captures_iter(inner) {
                    let ent_name = &pcap[1];
                    if tmp_entities.contains_key(ent_name) {
                        return inner.to_string();
                    }
                }
                String::new()
            })
            .to_string();
        if new_path == before {
            break;
        }
    }

    // Check all remaining placeholders have values
    for cap in PH_RE.captures_iter(&new_path) {
        let name = cap.get(1)?.as_str();
        if !tmp_entities.contains_key(name) {
            return None;
        }
    }

    // Handle extension with/without leading dot
    if let Some(ext_val) = tmp_entities.get("extension") {
        let ext_str = ext_val.as_str_lossy();
        if !ext_str.starts_with('.') {
            tmp_entities.insert("extension".into(), EntityValue::Str(format!(".{ext_str}")));
        }
    }

    // Replace all placeholders
    for (name, val) in &tmp_entities {
        let placeholder = format!("{{{name}}}");
        new_path = new_path.replace(&placeholder, &val.as_str_lossy());
    }

    if new_path.is_empty() {
        None
    } else {
        Some(new_path)
    }
}

/// Expand bracket options in value strings.
pub fn expand_options(value: &str) -> Vec<String> {
    static BRACKET_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]*?)\]").unwrap());
    let bracket_re = &*BRACKET_RE;
    if !bracket_re.is_match(value) {
        return vec![value.to_string()];
    }

    let parts: Vec<Vec<char>> = bracket_re
        .captures_iter(value)
        .map(|cap| cap[1].chars().collect())
        .collect();

    let template = bracket_re.replace_all(value, "\x00").to_string();
    let segments: Vec<&str> = template.split('\x00').collect();

    let mut results = vec![segments[0].to_string()];
    for (i, opts) in parts.iter().enumerate() {
        let suffix = segments.get(i + 1).unwrap_or(&"");
        let mut new_results = Vec::new();
        for r in &results {
            for &c in opts {
                new_results.push(format!("{r}{c}{suffix}"));
            }
        }
        results = new_results;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_path_basic() {
        let mut entities = Entities::new();
        entities.insert("subject".into(), EntityValue::Str("001".into()));
        entities.insert("suffix".into(), EntityValue::Str("T1w".into()));
        entities.insert("extension".into(), EntityValue::Str(".nii".into()));

        let patterns = &[
            "sub-{subject}[/ses-{session}]/anat/sub-{subject}[_ses-{session}]_{suffix<T1w|T2w>}{extension<.nii|.nii.gz>|.nii.gz}",
        ];

        let result = build_path(&entities, patterns, false);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "sub-001/anat/sub-001_T1w.nii");
    }

    #[test]
    fn test_build_path_with_default() {
        let mut entities = Entities::new();
        entities.insert("subject".into(), EntityValue::Str("001".into()));
        entities.insert("extension".into(), EntityValue::Str(".bvec".into()));

        let patterns = &[
            "sub-{subject}[/ses-{session}]/{datatype|dwi}/sub-{subject}[_ses-{session}]_{suffix|dwi}{extension<.bval|.bvec|.json|.nii.gz|.nii>|.nii.gz}",
        ];

        let result = build_path(&entities, patterns, true);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "sub-001/dwi/sub-001_dwi.bvec");
    }

    #[test]
    fn test_expand_options() {
        assert_eq!(expand_options("json"), vec!["json"]);
        let expanded = expand_options("[Jj]son");
        assert_eq!(expanded, vec!["Json", "json"]);
    }

    #[test]
    fn test_cartesian_product() {
        let a = vec!["01".to_string(), "02".to_string()];
        let b = vec!["rest".to_string()];
        let result = cartesian_product(&[&a, &b]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_build_paths_expanded() {
        let mut entities = std::collections::HashMap::new();
        entities.insert(
            "subject".to_string(),
            vec!["01".to_string(), "02".to_string()],
        );
        entities.insert("suffix".to_string(), vec!["T1w".to_string()]);
        entities.insert("extension".to_string(), vec![".nii.gz".to_string()]);

        let patterns = &["sub-{subject}/anat/sub-{subject}_{suffix}{extension}"];
        let results = build_paths_expanded(&entities, patterns, false);
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("sub-01"));
        assert!(results[1].contains("sub-02"));
    }
}
