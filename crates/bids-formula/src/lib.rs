#![deny(unsafe_code)]
//! Wilkinson formula parser for statistical model design matrices.
//!
//! Supports: `y ~ a + b + a:b - 1`, `a * b` (expands to `a + b + a:b`),
//! `1` (intercept), `0` or `-1` (no intercept).

use std::collections::HashMap;

/// A parsed term in a formula.
#[derive(Debug, Clone, PartialEq)]
pub enum Term {
    /// Intercept column (all 1s).
    Intercept,
    /// A single variable.
    Variable(String),
    /// An interaction: elementwise product of two or more variables.
    Interaction(Vec<String>),
}

impl Term {
    pub fn name(&self) -> String {
        match self {
            Term::Intercept => "intercept".into(),
            Term::Variable(s) => s.clone(),
            Term::Interaction(vs) => vs.join(":"),
        }
    }
}

/// A parsed model formula.
#[derive(Debug, Clone)]
pub struct Formula {
    /// Optional response variable (left of ~).
    pub response: Option<String>,
    /// Terms on the right-hand side.
    pub terms: Vec<Term>,
    /// Whether to include an intercept.
    pub intercept: bool,
}

/// Parse a Wilkinson-style formula string.
///
/// Examples:
/// - `"y ~ a + b"` → response="y", terms=[a, b], intercept=true
/// - `"~ a + b - 1"` → no response, terms=[a, b], intercept=false
/// - `"a * b"` → terms=[a, b, a:b], intercept=true
/// - `"a + a:b"` → terms=[a, a:b], intercept=true
pub fn parse_formula(formula: &str) -> Formula {
    let formula = formula.trim();

    // Split on ~ to get response and RHS
    let (response, rhs) = if let Some(idx) = formula.find('~') {
        let lhs = formula[..idx].trim();
        let rhs = formula[idx + 1..].trim();
        (if lhs.is_empty() { None } else { Some(lhs.to_string()) }, rhs.to_string())
    } else {
        (None, formula.to_string())
    };

    let mut terms = Vec::new();
    let mut intercept = true;

    // Tokenize RHS by + and -, respecting * expansion
    let rhs = rhs.as_str();
    let mut add = true; // true = adding terms, false = removing

    for token in split_terms(rhs) {
        let token = token.trim();
        if token == "+" { add = true; continue; }
        if token == "-" { add = false; continue; }
        if token.is_empty() { continue; }

        if token == "1" {
            if !add { intercept = false; }
            continue;
        }
        if token == "0" {
            intercept = false;
            continue;
        }

        if add {
            // Check for * (full crossing)
            if token.contains('*') {
                let parts: Vec<&str> = token.split('*').map(str::trim).collect();
                // Add main effects
                for &p in &parts {
                    let t = Term::Variable(p.into());
                    if !terms.contains(&t) { terms.push(t); }
                }
                // Add interaction
                let interaction = Term::Interaction(parts.iter().map(std::string::ToString::to_string).collect());
                if !terms.contains(&interaction) { terms.push(interaction); }
            } else if token.contains(':') {
                // Interaction
                let parts: Vec<String> = token.split(':').map(|s| s.trim().to_string()).collect();
                terms.push(Term::Interaction(parts));
            } else {
                terms.push(Term::Variable(token.into()));
            }
        } else {
            // Remove term
            let to_remove = if token.contains(':') {
                let parts: Vec<String> = token.split(':').map(|s| s.trim().to_string()).collect();
                Term::Interaction(parts)
            } else {
                Term::Variable(token.into())
            };
            terms.retain(|t| t != &to_remove);
            add = true; // Reset to adding after a subtraction
        }
    }

    Formula { response, terms, intercept }
}

/// Build a design matrix from a formula and data.
///
/// `data` maps variable names to their column values.
/// Returns `(column_names, columns)`.
pub fn build_design_matrix(
    formula: &Formula,
    data: &HashMap<String, Vec<f64>>,
) -> (Vec<String>, Vec<Vec<f64>>) {
    let n = data.values().next().map_or(0, std::vec::Vec::len);
    let mut names = Vec::new();
    let mut columns = Vec::new();

    if formula.intercept {
        names.push("intercept".into());
        columns.push(vec![1.0; n]);
    }

    for term in &formula.terms {
        match term {
            Term::Intercept => {
                if !names.contains(&"intercept".to_string()) {
                    names.push("intercept".into());
                    columns.push(vec![1.0; n]);
                }
            }
            Term::Variable(name) => {
                if let Some(col) = data.get(name) {
                    names.push(name.clone());
                    columns.push(col.clone());
                }
            }
            Term::Interaction(parts) => {
                let mut interaction = vec![1.0; n];
                let mut valid = true;
                for part in parts {
                    if let Some(col) = data.get(part) {
                        for (i, v) in col.iter().enumerate() {
                            interaction[i] *= v;
                        }
                    } else {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    names.push(term.name());
                    columns.push(interaction);
                }
            }
        }
    }

    (names, columns)
}

fn split_terms(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in s.chars() {
        match ch {
            '+' => {
                if !current.trim().is_empty() { tokens.push(current.trim().to_string()); }
                current.clear();
                tokens.push("+".into());
            }
            '-' => {
                if !current.trim().is_empty() { tokens.push(current.trim().to_string()); }
                current.clear();
                tokens.push("-".into());
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() { tokens.push(current.trim().to_string()); }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_formula() {
        let f = parse_formula("y ~ a + b");
        assert_eq!(f.response, Some("y".into()));
        assert_eq!(f.terms.len(), 2);
        assert!(f.intercept);
    }

    #[test]
    fn test_no_intercept() {
        let f = parse_formula("y ~ a + b - 1");
        assert!(!f.intercept);
        assert_eq!(f.terms.len(), 2);
    }

    #[test]
    fn test_interaction() {
        let f = parse_formula("~ a + b + a:b");
        assert_eq!(f.terms.len(), 3);
        assert_eq!(f.terms[2], Term::Interaction(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn test_star_expansion() {
        let f = parse_formula("~ a * b");
        assert_eq!(f.terms.len(), 3); // a, b, a:b
    }

    #[test]
    fn test_build_design_matrix() {
        let f = parse_formula("~ a + b");
        let mut data = HashMap::new();
        data.insert("a".into(), vec![1.0, 2.0, 3.0]);
        data.insert("b".into(), vec![4.0, 5.0, 6.0]);

        let (names, cols) = build_design_matrix(&f, &data);
        assert_eq!(names, vec!["intercept", "a", "b"]);
        assert_eq!(cols[0], vec![1.0, 1.0, 1.0]); // intercept
        assert_eq!(cols[1], vec![1.0, 2.0, 3.0]); // a
        assert_eq!(cols[2], vec![4.0, 5.0, 6.0]); // b
    }

    #[test]
    fn test_interaction_matrix() {
        let f = parse_formula("~ a * b - 1");
        let mut data = HashMap::new();
        data.insert("a".into(), vec![2.0, 3.0]);
        data.insert("b".into(), vec![4.0, 5.0]);

        let (names, cols) = build_design_matrix(&f, &data);
        assert_eq!(names, vec!["a", "b", "a:b"]);
        assert_eq!(cols[2], vec![8.0, 15.0]); // 2*4, 3*5
    }
}
