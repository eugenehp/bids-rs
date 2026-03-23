//! BCI evaluation splitters: within-session, cross-session, cross-subject.
//!
//! These produce train/test index pairs that respect the BIDS hierarchy
//! to prevent data leakage. Inspired by MOABB's evaluation system.
//!
//! All splitters work on a metadata table (subject, session, run per sample)
//! rather than raw data, so they can be used with any modality.

use std::collections::{BTreeSet, HashMap};

/// Metadata for one sample/epoch — identifies where it came from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SampleMeta {
    pub subject: String,
    pub session: String,
    pub run: String,
}

/// A train/test index split.
#[derive(Debug, Clone)]
pub struct SplitIndices {
    pub train: Vec<usize>,
    pub test: Vec<usize>,
    /// Description of this fold (e.g., "test_subject=01").
    pub description: String,
}

// ─── Within-Session Splitter ───────────────────────────────────────────────────

/// K-fold CV within each (subject, session) pair.
///
/// For each subject's each session, splits that session's data into k folds.
/// This is the most common BCI evaluation — no cross-session or cross-subject
/// generalization is measured.
///
/// Corresponds to MOABB's `WithinSessionEvaluation`.
pub fn within_session_splits(metadata: &[SampleMeta], k: usize, seed: u64) -> Vec<SplitIndices> {
    let mut results = Vec::new();

    // Group indices by (subject, session)
    let mut groups: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
    for (i, m) in metadata.iter().enumerate() {
        groups
            .entry((m.subject.as_str(), m.session.as_str()))
            .or_default()
            .push(i);
    }

    let mut group_keys: Vec<_> = groups.keys().cloned().collect();
    group_keys.sort();

    for (subj, sess) in group_keys {
        let indices = &groups[&(subj, sess)];
        let n = indices.len();
        if n < k {
            continue;
        }

        // Deterministic shuffle
        let mut shuffled = indices.clone();
        deterministic_shuffle(&mut shuffled, seed ^ hash_str(&format!("{subj}_{sess}")));

        let fold_size = n / k;
        for fold in 0..k {
            let test_start = fold * fold_size;
            let test_end = if fold == k - 1 {
                n
            } else {
                test_start + fold_size
            };

            let test: Vec<usize> = shuffled[test_start..test_end].to_vec();
            let train: Vec<usize> = shuffled[..test_start]
                .iter()
                .chain(shuffled[test_end..].iter())
                .copied()
                .collect();

            results.push(SplitIndices {
                train,
                test,
                description: format!("within_session sub={subj} ses={sess} fold={fold}"),
            });
        }
    }

    results
}

// ─── Cross-Session Splitter ────────────────────────────────────────────────────

/// Leave-one-session-out per subject.
///
/// For each subject, trains on all sessions except one, tests on the held-out
/// session. Measures within-subject cross-session generalization.
/// Requires ≥ 2 sessions per subject.
///
/// Corresponds to MOABB's `CrossSessionEvaluation`.
pub fn cross_session_splits(metadata: &[SampleMeta]) -> Vec<SplitIndices> {
    let mut results = Vec::new();

    // Group by subject
    let mut by_subject: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, m) in metadata.iter().enumerate() {
        by_subject.entry(m.subject.as_str()).or_default().push(i);
    }

    let mut subjects: Vec<_> = by_subject.keys().cloned().collect();
    subjects.sort();

    for &subj in &subjects {
        let subj_indices = &by_subject[subj];

        // Find unique sessions for this subject
        let sessions: BTreeSet<&str> = subj_indices
            .iter()
            .map(|&i| metadata[i].session.as_str())
            .collect();

        if sessions.len() < 2 {
            continue;
        }

        for test_session in &sessions {
            let train: Vec<usize> = subj_indices
                .iter()
                .filter(|&&i| metadata[i].session.as_str() != *test_session)
                .copied()
                .collect();
            let test: Vec<usize> = subj_indices
                .iter()
                .filter(|&&i| metadata[i].session.as_str() == *test_session)
                .copied()
                .collect();

            results.push(SplitIndices {
                train,
                test,
                description: format!("cross_session sub={subj} test_ses={test_session}"),
            });
        }
    }

    results
}

// ─── Cross-Subject Splitter ────────────────────────────────────────────────────

/// Leave-one-subject-out evaluation.
///
/// Trains on all subjects except one, tests on the held-out subject.
/// Measures cross-subject generalization (transfer learning).
/// Requires ≥ 2 subjects.
///
/// Corresponds to MOABB's `CrossSubjectEvaluation`.
pub fn cross_subject_splits(metadata: &[SampleMeta]) -> Vec<SplitIndices> {
    let mut results = Vec::new();

    let mut subjects: BTreeSet<&str> = BTreeSet::new();
    for m in metadata {
        subjects.insert(m.subject.as_str());
    }

    if subjects.len() < 2 {
        return results;
    }

    for &test_subject in &subjects {
        let train: Vec<usize> = metadata
            .iter()
            .enumerate()
            .filter(|(_, m)| m.subject.as_str() != test_subject)
            .map(|(i, _)| i)
            .collect();
        let test: Vec<usize> = metadata
            .iter()
            .enumerate()
            .filter(|(_, m)| m.subject.as_str() == test_subject)
            .map(|(i, _)| i)
            .collect();

        results.push(SplitIndices {
            train,
            test,
            description: format!("cross_subject test_sub={test_subject}"),
        });
    }

    results
}

// ─── Cross-Subject K-Fold ──────────────────────────────────────────────────────

/// K-fold cross-validation at the subject level.
///
/// Like `cross_subject_splits` but groups subjects into k folds instead
/// of leave-one-out (faster for large N).
pub fn cross_subject_kfold_splits(
    metadata: &[SampleMeta],
    k: usize,
    seed: u64,
) -> Vec<SplitIndices> {
    let mut subjects: Vec<String> = metadata
        .iter()
        .map(|m| m.subject.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    if subjects.len() < k {
        return Vec::new();
    }

    deterministic_shuffle(&mut subjects, seed);

    let mut results = Vec::new();
    let fold_size = subjects.len() / k;

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 {
            subjects.len()
        } else {
            test_start + fold_size
        };
        let test_subjects: BTreeSet<&str> = subjects[test_start..test_end]
            .iter()
            .map(|s| s.as_str())
            .collect();

        let train: Vec<usize> = metadata
            .iter()
            .enumerate()
            .filter(|(_, m)| !test_subjects.contains(m.subject.as_str()))
            .map(|(i, _)| i)
            .collect();
        let test: Vec<usize> = metadata
            .iter()
            .enumerate()
            .filter(|(_, m)| test_subjects.contains(m.subject.as_str()))
            .map(|(i, _)| i)
            .collect();

        let test_list: Vec<&str> = test_subjects.into_iter().collect();
        results.push(SplitIndices {
            train,
            test,
            description: format!(
                "cross_subject_kfold fold={fold} test=[{}]",
                test_list.join(",")
            ),
        });
    }

    results
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let n = items.len();
    if n <= 1 {
        return;
    }
    let mut state = seed ^ 0x517cc1b727220a95;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        items.swap(i, j);
    }
}

fn hash_str(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h = h.wrapping_mul(0x100000001b3);
        h ^= b as u64;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metadata() -> Vec<SampleMeta> {
        let mut meta = Vec::new();
        for sub in ["01", "02", "03", "04"] {
            for ses in ["A", "B"] {
                for _ in 0..10 {
                    meta.push(SampleMeta {
                        subject: sub.into(),
                        session: ses.into(),
                        run: "01".into(),
                    });
                }
            }
        }
        meta
    }

    #[test]
    fn test_within_session() {
        let meta = make_metadata();
        let splits = within_session_splits(&meta, 5, 42);
        // 4 subjects × 2 sessions × 5 folds = 40
        assert_eq!(splits.len(), 40);
        for split in &splits {
            assert!(!split.train.is_empty());
            assert!(!split.test.is_empty());
            // No overlap
            for &t in &split.test {
                assert!(!split.train.contains(&t));
            }
        }
    }

    #[test]
    fn test_cross_session() {
        let meta = make_metadata();
        let splits = cross_session_splits(&meta);
        // 4 subjects × 2 sessions = 8
        assert_eq!(splits.len(), 8);
        for split in &splits {
            // Test samples should all be from one session
            let test_sessions: BTreeSet<_> = split
                .test
                .iter()
                .map(|&i| meta[i].session.as_str())
                .collect();
            assert_eq!(test_sessions.len(), 1);

            // Train samples should be from the other session of the same subject
            let test_subject: BTreeSet<_> = split
                .test
                .iter()
                .map(|&i| meta[i].subject.as_str())
                .collect();
            let train_subject: BTreeSet<_> = split
                .train
                .iter()
                .map(|&i| meta[i].subject.as_str())
                .collect();
            assert_eq!(test_subject, train_subject);
        }
    }

    #[test]
    fn test_cross_subject() {
        let meta = make_metadata();
        let splits = cross_subject_splits(&meta);
        assert_eq!(splits.len(), 4); // 4 subjects, leave-one-out
        for split in &splits {
            let test_subjects: BTreeSet<_> = split
                .test
                .iter()
                .map(|&i| meta[i].subject.as_str())
                .collect();
            assert_eq!(test_subjects.len(), 1);

            // No subject overlap
            let train_subjects: BTreeSet<_> = split
                .train
                .iter()
                .map(|&i| meta[i].subject.as_str())
                .collect();
            for ts in &test_subjects {
                assert!(!train_subjects.contains(ts));
            }
        }
    }

    #[test]
    fn test_cross_subject_kfold() {
        let meta = make_metadata();
        let splits = cross_subject_kfold_splits(&meta, 2, 42);
        assert_eq!(splits.len(), 2);
        let total: usize = splits.iter().map(|s| s.test.len()).sum();
        assert_eq!(total, meta.len());
    }
}
