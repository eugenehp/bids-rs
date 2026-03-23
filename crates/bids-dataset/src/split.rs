//! Train/val/test splitting with subject-level stratification.

/// Defines how to split subjects into train/val/test sets.
#[derive(Debug, Clone)]
pub struct Split {
    /// Fraction for training (e.g., 0.8).
    pub train: f64,
    /// Fraction for validation (e.g., 0.1).
    pub val: f64,
    /// Fraction for test (e.g., 0.1).
    pub test: f64,
    /// Random seed for reproducible splits.
    pub seed: u64,
}

impl Split {
    /// Create a split with given ratios. They must sum to ≈1.0.
    pub fn ratio(train: f64, val: f64, test: f64) -> Self {
        Self {
            train,
            val,
            test,
            seed: 42,
        }
    }

    /// Set the random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Partition a list of subjects into train/val/test groups.
    ///
    /// Uses a deterministic hash-based assignment (no external RNG needed)
    /// so the split is reproducible across runs.
    pub fn partition(&self, subjects: &[String]) -> (Vec<String>, Vec<String>, Vec<String>) {
        let mut scored: Vec<(u64, &String)> = subjects
            .iter()
            .map(|s| (hash_subject(s, self.seed), s))
            .collect();
        scored.sort_by_key(|(h, _)| *h);

        let n = scored.len();
        let n_train = (n as f64 * self.train).round() as usize;
        let n_val = (n as f64 * self.val).round() as usize;

        let train: Vec<String> = scored[..n_train]
            .iter()
            .map(|(_, s)| (*s).clone())
            .collect();
        let val: Vec<String> = scored[n_train..n_train + n_val]
            .iter()
            .map(|(_, s)| (*s).clone())
            .collect();
        let test: Vec<String> = scored[n_train + n_val..]
            .iter()
            .map(|(_, s)| (*s).clone())
            .collect();

        (train, val, test)
    }
}

impl Default for Split {
    fn default() -> Self {
        Self::ratio(0.8, 0.1, 0.1)
    }
}

/// Deterministic hash for subject assignment. Uses a simple FNV-like hash
/// mixed with the seed so splits are reproducible.
pub fn hash_subject(subject: &str, seed: u64) -> u64 {
    let mut h: u64 = seed ^ 0xcbf29ce484222325;
    for b in subject.bytes() {
        h = h.wrapping_mul(0x100000001b3);
        h ^= b as u64;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_partition() {
        let subjects: Vec<String> = (1..=20).map(|i| format!("sub-{:02}", i)).collect();
        let split = Split::ratio(0.6, 0.2, 0.2);
        let (train, val, test) = split.partition(&subjects);

        assert_eq!(train.len(), 12); // 60%
        assert_eq!(val.len(), 4); // 20%
        assert_eq!(test.len(), 4); // 20%

        // No overlap
        for s in &train {
            assert!(!val.contains(s));
            assert!(!test.contains(s));
        }
        for s in &val {
            assert!(!test.contains(s));
        }

        // Reproducible
        let (t2, v2, te2) = split.partition(&subjects);
        assert_eq!(train, t2);
        assert_eq!(val, v2);
        assert_eq!(test, te2);

        // Different seed → different split
        let split2 = Split::ratio(0.6, 0.2, 0.2).with_seed(123);
        let (t3, _, _) = split2.partition(&subjects);
        assert_ne!(train, t3);
    }

    #[test]
    fn test_split_small() {
        let subjects: Vec<String> = vec!["A".into(), "B".into(), "C".into()];
        let split = Split::ratio(0.7, 0.15, 0.15);
        let (train, val, test) = split.partition(&subjects);
        assert_eq!(train.len() + val.len() + test.len(), 3);
    }
}
