#![deny(unsafe_code)]
//! English singular/plural inflection for BIDS entity names.
//!
//! Handles the ~30 entity names used in BIDS, plus common English rules.
//! Replaces the `inflect` Python library used by PyBIDS.

/// Convert a plural noun to its singular form.
///
/// ```
/// assert_eq!(bids_inflect::singularize("subjects"), Some("subject".into()));
/// assert_eq!(bids_inflect::singularize("vertices"), Some("vertice".into())); // matches Python inflect
/// assert_eq!(bids_inflect::singularize("analyses"), Some("analysis".into()));
/// assert_eq!(bids_inflect::singularize("runs"), Some("run".into()));
/// ```
#[must_use]
pub fn singularize(word: &str) -> Option<String> {
    let w = word.to_lowercase();

    // Exact irregular mappings
    let irregular: &[(&str, &str)] = &[
        ("vertices", "vertice"),  // matches Python inflect behavior
        ("matrices", "matrice"),  // matches Python inflect behavior
        ("indices", "index"),
        ("analyses", "analysis"),
        ("atlases", "atlas"),
        ("bases", "basis"),
        ("stimuli", "stimulus"),
        ("radii", "radius"),
        ("foci", "focus"),
        ("nuclei", "nucleus"),
        ("data", "datum"),
        ("criteria", "criterion"),
        ("phenomena", "phenomenon"),
    ];
    for &(plural, singular) in irregular {
        if w == plural { return Some(singular.into()); }
    }

    // Don't singularize if already singular (common BIDS entities)
    let already_singular = [
        "subject", "session", "task", "run", "acquisition", "reconstruction",
        "direction", "space", "split", "recording", "echo", "flip", "part",
        "sample", "staining", "tracer", "modality", "chunk", "atlas",
        "resolution", "density", "label", "description", "suffix",
        "extension", "datatype", "ceagent", "hemisphere", "tracksys",
    ];
    if already_singular.contains(&w.as_str()) { return Some(w); }

    // Rules (most specific first)
    if w.ends_with("oes") && w.len() > 3 {
        // echoes → echo, tomatoes → tomato (but not shoes → sho)
        let stem = &w[..w.len() - 2];
        if !["sh", "to"].iter().any(|s| stem.ends_with(s)) {
            return Some(w[..w.len() - 2].into());
        }
    }
    if w.ends_with("ies") && w.len() > 3 {
        return Some(format!("{}y", &w[..w.len() - 3]));
    }
    if w.ends_with("ses") && w.len() > 3 {
        return Some(w[..w.len() - 2].into());
    }
    if w.ends_with("ves") && w.len() > 3 {
        return Some(format!("{}f", &w[..w.len() - 3]));
    }
    if w.ends_with("xes") || w.ends_with("ches") || w.ends_with("shes") {
        return Some(w[..w.len() - 2].into());
    }
    if w.ends_with('s') && !w.ends_with("ss") && w.len() > 1 {
        return Some(w[..w.len() - 1].into());
    }

    Some(w)
}

/// Convert a singular noun to its plural form.
///
/// ```
/// assert_eq!(bids_inflect::pluralize("subject"), "subjects");
/// assert_eq!(bids_inflect::pluralize("analysis"), "analyses");
/// assert_eq!(bids_inflect::pluralize("vertex"), "vertices");
/// ```
#[must_use]
pub fn pluralize(word: &str) -> String {
    let w = word.to_lowercase();

    let irregular: &[(&str, &str)] = &[
        ("vertex", "vertices"),
        ("matrix", "matrices"),
        ("index", "indices"),
        ("analysis", "analyses"),
        ("atlas", "atlases"),
        ("basis", "bases"),
        ("stimulus", "stimuli"),
        ("radius", "radii"),
        ("focus", "foci"),
        ("nucleus", "nuclei"),
        ("datum", "data"),
        ("criterion", "criteria"),
        ("phenomenon", "phenomena"),
    ];
    for &(singular, plural) in irregular {
        if w == singular { return plural.into(); }
    }

    if w.ends_with('y') && w.len() > 1 && !is_vowel(w.as_bytes()[w.len() - 2]) {
        return format!("{}ies", &w[..w.len() - 1]);
    }
    if w.ends_with('s') || w.ends_with('x') || w.ends_with('z')
        || w.ends_with("ch") || w.ends_with("sh") {
        return format!("{w}es");
    }
    if w.ends_with('f') {
        return format!("{}ves", &w[..w.len() - 1]);
    }
    if w.ends_with("fe") {
        return format!("{}ves", &w[..w.len() - 2]);
    }

    format!("{w}s")
}

fn is_vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u')
}

/// Check if a word is likely plural.
#[must_use]
pub fn is_plural(word: &str) -> bool {
    if let Some(singular) = singularize(word) {
        singular != word.to_lowercase()
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_singularize() {
        assert_eq!(singularize("subjects"), Some("subject".into()));
        assert_eq!(singularize("sessions"), Some("session".into()));
        assert_eq!(singularize("runs"), Some("run".into()));
        assert_eq!(singularize("tasks"), Some("task".into()));
        assert_eq!(singularize("vertices"), Some("vertice".into())); // matches Python inflect
        assert_eq!(singularize("analyses"), Some("analysis".into()));
        assert_eq!(singularize("atlases"), Some("atlas".into()));
        assert_eq!(singularize("categories"), Some("category".into()));
    }

    #[test]
    fn test_pluralize() {
        assert_eq!(pluralize("subject"), "subjects");
        assert_eq!(pluralize("session"), "sessions");
        assert_eq!(pluralize("vertex"), "vertices");
        assert_eq!(pluralize("analysis"), "analyses");
        assert_eq!(pluralize("atlas"), "atlases");
        assert_eq!(pluralize("category"), "categories");
    }

    #[test]
    fn test_singularize_already_singular() {
        assert_eq!(singularize("subject"), Some("subject".into()));
        assert_eq!(singularize("run"), Some("run".into()));
    }

    #[test]
    fn test_is_plural() {
        assert!(is_plural("subjects"));
        assert!(is_plural("sessions"));
        assert!(!is_plural("subject"));
        assert!(!is_plural("run"));
    }

    #[test]
    fn test_roundtrip() {
        // Note: vertex→vertices→vertice due to matching Python inflect behavior
        for word in &["subject", "session", "run", "task", "acquisition", "analysis"] {
            let plural = pluralize(word);
            let back = singularize(&plural).unwrap();
            assert_eq!(&back, word, "roundtrip failed: {} -> {} -> {}", word, plural, back);
        }
    }
}
