//! ML pipeline utilities: windowing, epoch iterators, and data loaders.
//!
//! Provides the glue between BIDS datasets and ML training loops:
//!
//! - [`EpochSpec`] — defines how to window continuous data into fixed-size samples
//! - [`Sample`] — a single (data, label, metadata) tuple ready for a model
//! - [`DatasetIter`] — a lazy, shuffleable iterator over samples from an [`super::Aggregator`]
//! - [`KFold`] — k-fold cross-validation with subject-level grouping
//! - [`StratifiedSplit`] — label-aware splitting to preserve class ratios
//!
//! All iterators are lazy and zero-copy where possible, so you can stream
//! through datasets that don't fit in memory.

use crate::aggregate::FileEntry;
use crate::split::Split;
use std::collections::HashMap;

/// A (train_subjects, test_subjects) pair for one fold of cross-validation.
pub type FoldPair<'a> = (Vec<&'a String>, Vec<&'a String>);

/// A label-assignment function that maps a file entry to an optional label.
type LabelFn<'a> = Box<dyn Fn(&FileEntry) -> Option<String> + 'a>;

// ─── Sample / Epoch Types ──────────────────────────────────────────────────────

/// Specification for extracting fixed-size epochs from continuous data.
#[derive(Debug, Clone)]
pub struct EpochSpec {
    /// Window length in seconds.
    pub window_sec: f64,
    /// Stride (step) in seconds. If < window_sec, windows overlap.
    pub stride_sec: f64,
    /// Whether to drop the last window if it's shorter than `window_sec`.
    pub drop_last: bool,
}

impl EpochSpec {
    /// Non-overlapping windows.
    #[must_use]
    pub fn non_overlapping(window_sec: f64) -> Self {
        Self { window_sec, stride_sec: window_sec, drop_last: true }
    }

    /// Overlapping windows with a given stride.
    #[must_use]
    pub fn overlapping(window_sec: f64, stride_sec: f64) -> Self {
        Self { window_sec, stride_sec, drop_last: true }
    }

    /// Compute the number of windows that fit in a signal of `duration_sec`.
    #[must_use]
    pub fn n_windows(&self, duration_sec: f64) -> usize {
        if duration_sec < self.window_sec {
            return if self.drop_last { 0 } else { 1 };
        }
        ((duration_sec - self.window_sec) / self.stride_sec).floor() as usize + 1
    }

    /// Compute (start_sec, end_sec) for window index `i`.
    #[must_use]
    pub fn window_bounds(&self, i: usize) -> (f64, f64) {
        let start = i as f64 * self.stride_sec;
        (start, start + self.window_sec)
    }
}

/// A single ML sample: data path + metadata, ready for a data loader to read.
///
/// Deliberately does **not** hold loaded signal data — the loader reads on demand
/// so we can represent million-file datasets without memory pressure.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Path to the data file on disk.
    pub path: std::path::PathBuf,
    /// Subject identifier (global, unique across datasets).
    pub subject: String,
    /// Optional label (from events, participant variables, task name, etc.).
    pub label: Option<String>,
    /// Time window within the file (None = entire file).
    pub window: Option<(f64, f64)>,
    /// Extra metadata (task, session, run, dataset, etc.).
    pub metadata: HashMap<String, String>,
}

// ─── K-Fold Cross-Validation ───────────────────────────────────────────────────

/// K-fold cross-validation with subject-level grouping.
///
/// Ensures all files from a single subject are in the same fold,
/// preventing data leakage in neuroimaging/EEG models.
#[derive(Debug, Clone)]
pub struct KFold {
    /// Number of folds.
    pub k: usize,
    /// Random seed.
    pub seed: u64,
}

impl KFold {
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k, seed: 42 }
    }

    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Partition subjects into k folds. Returns a Vec of k `Vec<String>`,
    /// where each inner vec contains the subjects in that fold.
    #[must_use]
    pub fn split_subjects(&self, subjects: &[String]) -> Vec<Vec<String>> {
        use crate::split::hash_subject;

        let mut scored: Vec<(u64, &String)> = subjects.iter()
            .map(|s| (hash_subject(s, self.seed), s))
            .collect();
        scored.sort_by_key(|(h, _)| *h);

        let mut folds: Vec<Vec<String>> = (0..self.k).map(|_| Vec::new()).collect();
        for (i, (_, subj)) in scored.iter().enumerate() {
            folds[i % self.k].push((*subj).clone());
        }
        folds
    }

    /// Iterate over (train_subjects, test_subjects) for each fold.
    pub fn iter_folds<'a>(&self, subjects: &'a [String]) -> Vec<FoldPair<'a>> {
        let folds = self.split_subjects(subjects);
        let subj_to_fold: HashMap<&str, usize> = folds.iter().enumerate()
            .flat_map(|(fi, fold)| fold.iter().map(move |s| (s.as_str(), fi)))
            .collect();

        (0..self.k).map(|test_fold| {
            let train: Vec<&String> = subjects.iter()
                .filter(|s| subj_to_fold.get(s.as_str()) != Some(&test_fold))
                .collect();
            let test: Vec<&String> = subjects.iter()
                .filter(|s| subj_to_fold.get(s.as_str()) == Some(&test_fold))
                .collect();
            (train, test)
        }).collect()
    }
}

// ─── Stratified Split ──────────────────────────────────────────────────────────

/// Label-balanced train/val/test splitting.
///
/// Ensures each split has approximately the same class distribution as the
/// full dataset. Stratification is done at the subject level.
pub struct StratifiedSplit;

impl StratifiedSplit {
    /// Partition subjects by label, then split each label group proportionally.
    ///
    /// `subject_labels`: maps subject_id → label.
    #[must_use]
    pub fn partition(
        subjects: &[String],
        subject_labels: &HashMap<String, String>,
        split: &Split,
    ) -> (Vec<String>, Vec<String>, Vec<String>) {
        // Group subjects by label
        let mut by_label: HashMap<&str, Vec<String>> = HashMap::new();
        for s in subjects {
            let label = subject_labels.get(s).map(|l| l.as_str()).unwrap_or("_unknown_");
            by_label.entry(label).or_default().push(s.clone());
        }

        let mut train = Vec::new();
        let mut val = Vec::new();
        let mut test = Vec::new();

        for group in by_label.values() {
            let (t, v, te) = split.partition(group);
            train.extend(t);
            val.extend(v);
            test.extend(te);
        }

        (train, val, test)
    }
}

// ─── Dataset Iterator ──────────────────────────────────────────────────────────

/// Lazy iterator that yields [`Sample`]s from an [`super::Aggregator`]'s file entries.
///
/// Supports optional windowing (epoch extraction), label assignment,
/// subject filtering, and deterministic shuffling.
///
/// ```ignore
/// let iter = DatasetIter::new(&aggregator.files)
///     .with_label_fn(|e| Some(e.task.clone().unwrap_or_default()))
///     .with_epochs(EpochSpec::non_overlapping(2.0), 60.0)
///     .with_subjects(&train_subjects)
///     .shuffle(42);
///
/// for sample in iter {
///     // load sample.path, crop to sample.window, use sample.label
/// }
/// ```
pub struct DatasetIter<'a> {
    entries: &'a [FileEntry],
    label_fn: Option<LabelFn<'a>>,
    epoch_spec: Option<EpochSpec>,
    /// Assumed duration per file (used to compute windows when not reading files).
    assumed_duration: f64,
    /// Only include these subjects (if set).
    subject_filter: Option<std::collections::HashSet<String>>,
    /// Shuffle order seed. None = no shuffle.
    shuffle_seed: Option<u64>,
}

impl<'a> DatasetIter<'a> {
    #[must_use]
    pub fn new(entries: &'a [FileEntry]) -> Self {
        Self {
            entries,
            label_fn: None,
            epoch_spec: None,
            assumed_duration: 0.0,
            subject_filter: None,
            shuffle_seed: None,
        }
    }

    /// Assign labels from a closure on each file entry.
    #[must_use]
    pub fn with_label_fn<F>(mut self, f: F) -> Self
    where F: Fn(&FileEntry) -> Option<String> + 'a {
        self.label_fn = Some(Box::new(f));
        self
    }

    /// Window each file into fixed-size epochs.
    /// `assumed_duration` is used to compute the number of windows per file.
    #[must_use]
    pub fn with_epochs(mut self, spec: EpochSpec, assumed_duration: f64) -> Self {
        self.epoch_spec = Some(spec);
        self.assumed_duration = assumed_duration;
        self
    }

    /// Only yield samples from these subjects.
    #[must_use]
    pub fn with_subjects(mut self, subjects: &[String]) -> Self {
        self.subject_filter = Some(subjects.iter().cloned().collect());
        self
    }

    /// Deterministic shuffle (reproducible with the same seed).
    #[must_use]
    pub fn shuffle(mut self, seed: u64) -> Self {
        self.shuffle_seed = Some(seed);
        self
    }

    /// Collect all samples into a Vec.
    ///
    /// For large datasets, prefer iterating lazily with `into_iter()`.
    pub fn collect_samples(&self) -> Vec<Sample> {
        let mut samples = Vec::new();

        for entry in self.entries {
            if let Some(ref filter) = self.subject_filter {
                if !filter.contains(&entry.global_subject) {
                    continue;
                }
            }

            let label = self.label_fn.as_ref().and_then(|f| f(entry));
            let meta = entry_metadata(entry);

            if let Some(ref spec) = self.epoch_spec {
                let n = spec.n_windows(self.assumed_duration);
                for i in 0..n {
                    let (start, end) = spec.window_bounds(i);
                    samples.push(Sample {
                        path: entry.path.clone(),
                        subject: entry.global_subject.clone(),
                        label: label.clone(),
                        window: Some((start, end)),
                        metadata: meta.clone(),
                    });
                }
            } else {
                samples.push(Sample {
                    path: entry.path.clone(),
                    subject: entry.global_subject.clone(),
                    label,
                    window: None,
                    metadata: meta,
                });
            }
        }

        if let Some(seed) = self.shuffle_seed {
            deterministic_shuffle(&mut samples, seed);
        }

        samples
    }
}

fn entry_metadata(entry: &FileEntry) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("dataset".into(), entry.dataset.clone());
    m.insert("datatype".into(), entry.datatype.clone());
    m.insert("suffix".into(), entry.suffix.clone());
    if let Some(ref t) = entry.task { m.insert("task".into(), t.clone()); }
    if let Some(ref s) = entry.session { m.insert("session".into(), s.clone()); }
    if let Some(ref r) = entry.run { m.insert("run".into(), r.clone()); }
    m
}

/// Fisher-Yates shuffle with a simple deterministic LCG.
fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let n = items.len();
    if n <= 1 { return; }
    let mut state = seed ^ 0x517cc1b727220a95;
    for i in (1..n).rev() {
        // LCG step
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        items.swap(i, j);
    }
}

// ─── Summary Stats ─────────────────────────────────────────────────────────────

/// Compute label distribution from a set of samples.
#[must_use]
pub fn label_distribution(samples: &[Sample]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for s in samples {
        let label = s.label.as_deref().unwrap_or("_unlabeled_");
        *counts.entry(label.to_string()).or_insert(0) += 1;
    }
    counts
}

/// Compute per-dataset file counts.
#[must_use]
pub fn dataset_summary(entries: &[FileEntry]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for e in entries {
        *counts.entry(e.dataset.clone()).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries(n_subjects: usize, n_files_per: usize) -> Vec<FileEntry> {
        let mut entries = Vec::new();
        for s in 0..n_subjects {
            for f in 0..n_files_per {
                entries.push(FileEntry {
                    dataset: "ds001".into(),
                    subject: format!("{s:02}"),
                    session: None,
                    task: Some("rest".into()),
                    run: Some(format!("{f:02}")),
                    datatype: "eeg".into(),
                    suffix: "eeg".into(),
                    extension: ".edf".into(),
                    path: format!("/data/ds001/sub-{s:02}/eeg/sub-{s:02}_task-rest_run-{f:02}_eeg.edf").into(),
                    size: 10_000,
                    global_subject: format!("ds001_{s:02}"),
                });
            }
        }
        entries
    }

    #[test]
    fn test_epoch_spec() {
        let spec = EpochSpec::non_overlapping(2.0);
        assert_eq!(spec.n_windows(10.0), 5);
        assert_eq!(spec.n_windows(1.5), 0);
        assert_eq!(spec.window_bounds(2), (4.0, 6.0));

        let spec = EpochSpec::overlapping(4.0, 2.0);
        assert_eq!(spec.n_windows(10.0), 4); // [0-4, 2-6, 4-8, 6-10]
    }

    #[test]
    fn test_kfold() {
        let subjects: Vec<String> = (0..10).map(|i| format!("sub-{i:02}")).collect();
        let kf = KFold::new(5);
        let folds = kf.split_subjects(&subjects);
        assert_eq!(folds.len(), 5);
        let total: usize = folds.iter().map(|f| f.len()).sum();
        assert_eq!(total, 10);

        // No overlap between folds
        let mut all: Vec<String> = folds.into_iter().flatten().collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn test_kfold_iter() {
        let subjects: Vec<String> = (0..10).map(|i| format!("sub-{i:02}")).collect();
        let kf = KFold::new(5);
        let fold_pairs = kf.iter_folds(&subjects);
        assert_eq!(fold_pairs.len(), 5);
        for (train, test) in &fold_pairs {
            assert_eq!(train.len() + test.len(), 10);
            // No overlap
            for t in test {
                assert!(!train.contains(t));
            }
        }
    }

    #[test]
    fn test_dataset_iter_basic() {
        let entries = make_entries(5, 3);
        let iter = DatasetIter::new(&entries);
        let samples = iter.collect_samples();
        assert_eq!(samples.len(), 15);
    }

    #[test]
    fn test_dataset_iter_with_epochs() {
        let entries = make_entries(2, 1);
        let spec = EpochSpec::non_overlapping(2.0);
        let iter = DatasetIter::new(&entries).with_epochs(spec, 10.0);
        let samples = iter.collect_samples();
        assert_eq!(samples.len(), 2 * 5); // 2 files × 5 windows
        assert!(samples[0].window.is_some());
    }

    #[test]
    fn test_dataset_iter_with_subjects() {
        let entries = make_entries(5, 2);
        let subs = vec!["ds001_00".into(), "ds001_02".into()];
        let iter = DatasetIter::new(&entries).with_subjects(&subs);
        let samples = iter.collect_samples();
        assert_eq!(samples.len(), 4); // 2 subjects × 2 files
    }

    #[test]
    fn test_dataset_iter_shuffle() {
        let entries = make_entries(10, 2);
        let s1 = DatasetIter::new(&entries).shuffle(42).collect_samples();
        let s2 = DatasetIter::new(&entries).shuffle(42).collect_samples();
        let s3 = DatasetIter::new(&entries).shuffle(99).collect_samples();

        // Same seed → same order
        let paths1: Vec<_> = s1.iter().map(|s| &s.path).collect();
        let paths2: Vec<_> = s2.iter().map(|s| &s.path).collect();
        assert_eq!(paths1, paths2);

        // Different seed → different order
        let paths3: Vec<_> = s3.iter().map(|s| &s.path).collect();
        assert_ne!(paths1, paths3);
    }

    #[test]
    fn test_stratified_split() {
        let subjects: Vec<String> = (0..20).map(|i| format!("sub-{i:02}")).collect();
        let mut labels = HashMap::new();
        for (i, s) in subjects.iter().enumerate() {
            labels.insert(s.clone(), if i < 10 { "healthy" } else { "patient" }.into());
        }
        let split = Split::ratio(0.6, 0.2, 0.2);
        let (train, val, test) = StratifiedSplit::partition(&subjects, &labels, &split);

        assert_eq!(train.len() + val.len() + test.len(), 20);

        // Each split should have some of each class
        let count_label = |group: &[String], label: &str| -> usize {
            group.iter().filter(|s| labels.get(*s).map(|l| l.as_str()) == Some(label)).count()
        };
        assert!(count_label(&train, "healthy") > 0);
        assert!(count_label(&train, "patient") > 0);
    }

    #[test]
    fn test_label_distribution() {
        let samples = vec![
            Sample { path: "a".into(), subject: "s1".into(), label: Some("A".into()), window: None, metadata: HashMap::new() },
            Sample { path: "b".into(), subject: "s2".into(), label: Some("B".into()), window: None, metadata: HashMap::new() },
            Sample { path: "c".into(), subject: "s3".into(), label: Some("A".into()), window: None, metadata: HashMap::new() },
        ];
        let dist = label_distribution(&samples);
        assert_eq!(dist["A"], 2);
        assert_eq!(dist["B"], 1);
    }
}
