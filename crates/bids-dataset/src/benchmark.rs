//! Benchmark result storage with CSV and Parquet export.
//!
//! Stores evaluation results from BCI/ML pipelines in a structured format
//! that can be exported for analysis. Inspired by MOABB's result tracking.

use serde::{Deserialize, Serialize};
use std::io::Write;

/// A single benchmark evaluation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Dataset identifier (e.g., "ds004362").
    pub dataset: String,
    /// Subject identifier.
    pub subject: String,
    /// Session identifier.
    pub session: String,
    /// Pipeline / model name (e.g., "CSP+LDA").
    pub pipeline: String,
    /// Paradigm name (e.g., "MotorImagery").
    pub paradigm: String,
    /// Evaluation type (e.g., "WithinSession", "CrossSubject").
    pub eval_type: String,
    /// Primary score (accuracy, AUC, etc.).
    pub score: f64,
    /// Name of the scoring metric.
    pub scoring: String,
    /// Training time in seconds.
    pub time_seconds: f64,
    /// Number of training samples.
    pub n_train: usize,
    /// Number of test samples.
    pub n_test: usize,
    /// Number of channels.
    pub n_channels: usize,
    /// Number of classes.
    pub n_classes: usize,
    /// Fold/split index within the evaluation.
    pub fold: usize,
    /// Additional notes / configuration.
    pub notes: String,
}

impl BenchmarkResult {
    /// Create a new result with required fields, defaults for the rest.
    #[must_use]
    pub fn new(dataset: &str, subject: &str, pipeline: &str, score: f64) -> Self {
        Self {
            dataset: dataset.into(),
            subject: subject.into(),
            session: String::new(),
            pipeline: pipeline.into(),
            paradigm: String::new(),
            eval_type: String::new(),
            score,
            scoring: "accuracy".into(),
            time_seconds: 0.0,
            n_train: 0,
            n_test: 0,
            n_channels: 0,
            n_classes: 2,
            fold: 0,
            notes: String::new(),
        }
    }
}

/// Collection of benchmark results with export capabilities.
#[derive(Debug, Default, Clone)]
pub struct BenchmarkResults {
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkResults {
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Add a result.
    pub fn push(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Number of results.
    #[must_use]
    pub fn len(&self) -> usize { self.results.len() }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.results.is_empty() }

    /// Compute mean score per pipeline across all subjects/sessions/folds.
    #[must_use]
    pub fn mean_scores_by_pipeline(&self) -> Vec<(String, f64, usize)> {
        let mut sums: std::collections::HashMap<&str, (f64, usize)> = std::collections::HashMap::new();
        for r in &self.results {
            let e = sums.entry(r.pipeline.as_str()).or_insert((0.0, 0));
            e.0 += r.score;
            e.1 += 1;
        }
        let mut out: Vec<(String, f64, usize)> = sums.into_iter()
            .map(|(name, (sum, count))| (name.to_string(), sum / count as f64, count))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Summary table as a formatted string.
    #[must_use]
    pub fn summary(&self) -> String {
        let means = self.mean_scores_by_pipeline();
        let mut lines = vec![format!("{:<30} {:>10} {:>8}", "Pipeline", "Mean Score", "N")];
        lines.push("-".repeat(50));
        for (name, mean, n) in &means {
            lines.push(format!("{:<30} {:>10.4} {:>8}", name, mean, n));
        }
        lines.join("\n")
    }

    // ── CSV Export ─────────────────────────────────────────────────────

    /// Export results as CSV.
    pub fn to_csv(&self, path: &str) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "{}", Self::csv_header())?;
        for r in &self.results {
            writeln!(f, "{}", Self::csv_row(r))?;
        }
        Ok(())
    }

    /// Append results to an existing CSV (creates file + header if missing).
    pub fn append_csv(&self, path: &str) -> std::io::Result<()> {
        let exists = std::path::Path::new(path).exists();
        let mut f = std::fs::OpenOptions::new()
            .create(true).append(true).open(path)?;
        if !exists {
            writeln!(f, "{}", Self::csv_header())?;
        }
        for r in &self.results {
            writeln!(f, "{}", Self::csv_row(r))?;
        }
        Ok(())
    }

    /// Load results from a CSV file.
    pub fn from_csv(path: &str) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut results = Vec::new();
        let mut lines = content.lines();
        let _header = lines.next(); // skip header

        for line in lines {
            if line.trim().is_empty() { continue; }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 15 { continue; }
            results.push(BenchmarkResult {
                dataset: cols[0].to_string(),
                subject: cols[1].to_string(),
                session: cols[2].to_string(),
                pipeline: cols[3].to_string(),
                paradigm: cols[4].to_string(),
                eval_type: cols[5].to_string(),
                score: cols[6].parse().unwrap_or(0.0),
                scoring: cols[7].to_string(),
                time_seconds: cols[8].parse().unwrap_or(0.0),
                n_train: cols[9].parse().unwrap_or(0),
                n_test: cols[10].parse().unwrap_or(0),
                n_channels: cols[11].parse().unwrap_or(0),
                n_classes: cols[12].parse().unwrap_or(2),
                fold: cols[13].parse().unwrap_or(0),
                notes: cols[14..].join(","),
            });
        }
        Ok(Self { results })
    }

    fn csv_header() -> &'static str {
        "dataset,subject,session,pipeline,paradigm,eval_type,score,scoring,time_seconds,n_train,n_test,n_channels,n_classes,fold,notes"
    }

    fn csv_row(r: &BenchmarkResult) -> String {
        format!("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.dataset, r.subject, r.session, r.pipeline, r.paradigm,
            r.eval_type, r.score, r.scoring, r.time_seconds,
            r.n_train, r.n_test, r.n_channels, r.n_classes, r.fold, r.notes)
    }

    // ── Parquet Export ──────────────────────────────────────────────────

    /// Export results as a Parquet file.
    ///
    /// Requires the `arrow` feature flag.
    #[cfg(feature = "arrow")]
    pub fn to_parquet(&self, path: &str) -> crate::Result<()> {
        use arrow::builder::{Float64Builder, StringBuilder, UInt64Builder};
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;

        let n = self.results.len();
        let fields = vec![
            Field::new("dataset", DataType::Utf8, false),
            Field::new("subject", DataType::Utf8, false),
            Field::new("session", DataType::Utf8, false),
            Field::new("pipeline", DataType::Utf8, false),
            Field::new("paradigm", DataType::Utf8, false),
            Field::new("eval_type", DataType::Utf8, false),
            Field::new("score", DataType::Float64, false),
            Field::new("scoring", DataType::Utf8, false),
            Field::new("time_seconds", DataType::Float64, false),
            Field::new("n_train", DataType::UInt64, false),
            Field::new("n_test", DataType::UInt64, false),
            Field::new("n_channels", DataType::UInt64, false),
            Field::new("n_classes", DataType::UInt64, false),
            Field::new("fold", DataType::UInt64, false),
            Field::new("notes", DataType::Utf8, false),
        ];
        let schema = Arc::new(Schema::new(fields));

        let mut dataset_b = StringBuilder::with_capacity(n, n * 10);
        let mut subject_b = StringBuilder::with_capacity(n, n * 5);
        let mut session_b = StringBuilder::with_capacity(n, n * 5);
        let mut pipeline_b = StringBuilder::with_capacity(n, n * 15);
        let mut paradigm_b = StringBuilder::with_capacity(n, n * 15);
        let mut eval_b = StringBuilder::with_capacity(n, n * 15);
        let mut score_b = Float64Builder::with_capacity(n);
        let mut scoring_b = StringBuilder::with_capacity(n, n * 10);
        let mut time_b = Float64Builder::with_capacity(n);
        let mut n_train_b = UInt64Builder::with_capacity(n);
        let mut n_test_b = UInt64Builder::with_capacity(n);
        let mut n_chan_b = UInt64Builder::with_capacity(n);
        let mut n_class_b = UInt64Builder::with_capacity(n);
        let mut fold_b = UInt64Builder::with_capacity(n);
        let mut notes_b = StringBuilder::with_capacity(n, n * 20);

        for r in &self.results {
            dataset_b.append_value(&r.dataset);
            subject_b.append_value(&r.subject);
            session_b.append_value(&r.session);
            pipeline_b.append_value(&r.pipeline);
            paradigm_b.append_value(&r.paradigm);
            eval_b.append_value(&r.eval_type);
            score_b.append_value(r.score);
            scoring_b.append_value(&r.scoring);
            time_b.append_value(r.time_seconds);
            n_train_b.append_value(r.n_train as u64);
            n_test_b.append_value(r.n_test as u64);
            n_chan_b.append_value(r.n_channels as u64);
            n_class_b.append_value(r.n_classes as u64);
            fold_b.append_value(r.fold as u64);
            notes_b.append_value(&r.notes);
        }

        let batch = arrow::RecordBatch::try_new(schema.clone(), vec![
            Arc::new(dataset_b.finish()),
            Arc::new(subject_b.finish()),
            Arc::new(session_b.finish()),
            Arc::new(pipeline_b.finish()),
            Arc::new(paradigm_b.finish()),
            Arc::new(eval_b.finish()),
            Arc::new(score_b.finish()),
            Arc::new(scoring_b.finish()),
            Arc::new(time_b.finish()),
            Arc::new(n_train_b.finish()),
            Arc::new(n_test_b.finish()),
            Arc::new(n_chan_b.finish()),
            Arc::new(n_class_b.finish()),
            Arc::new(fold_b.finish()),
            Arc::new(notes_b.finish()),
        ]).map_err(|e| crate::Error::Network(e.to_string()))?;

        let file = std::fs::File::create(path)?;
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| crate::Error::Network(e.to_string()))?;
        writer.write(&batch).map_err(|e| crate::Error::Network(e.to_string()))?;
        writer.close().map_err(|e| crate::Error::Network(e.to_string()))?;

        Ok(())
    }
}

impl std::fmt::Display for BenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BenchmarkResults({} results)\n{}", self.results.len(), self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results() -> BenchmarkResults {
        let mut br = BenchmarkResults::new();
        for sub in ["01", "02", "03"] {
            for pipeline in ["CSP+LDA", "Riemann+SVM"] {
                br.push(BenchmarkResult {
                    dataset: "ds001".into(),
                    subject: sub.into(),
                    session: "A".into(),
                    pipeline: pipeline.into(),
                    paradigm: "MotorImagery".into(),
                    eval_type: "WithinSession".into(),
                    score: if pipeline == "CSP+LDA" { 0.75 } else { 0.82 },
                    scoring: "accuracy".into(),
                    time_seconds: 0.5,
                    n_train: 80,
                    n_test: 20,
                    n_channels: 22,
                    n_classes: 2,
                    fold: 0,
                    notes: String::new(),
                });
            }
        }
        br
    }

    #[test]
    fn test_mean_scores() {
        let br = make_results();
        let means = br.mean_scores_by_pipeline();
        assert_eq!(means.len(), 2);
        // Riemann+SVM should be first (higher score)
        assert_eq!(means[0].0, "Riemann+SVM");
        assert!((means[0].1 - 0.82).abs() < 1e-10);
        assert_eq!(means[1].0, "CSP+LDA");
    }

    #[test]
    fn test_csv_roundtrip() {
        let br = make_results();
        let dir = std::env::temp_dir().join("bids_bench_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("results.csv");

        br.to_csv(path.to_str().unwrap()).unwrap();
        let loaded = BenchmarkResults::from_csv(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.len(), br.len());
        assert!((loaded.results[0].score - br.results[0].score).abs() < 1e-10);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_summary() {
        let br = make_results();
        let s = br.summary();
        assert!(s.contains("CSP+LDA"));
        assert!(s.contains("Riemann+SVM"));
    }

    #[test]
    fn test_append_csv() {
        let dir = std::env::temp_dir().join("bids_bench_append");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("results.csv");

        let mut br1 = BenchmarkResults::new();
        br1.push(BenchmarkResult::new("ds001", "01", "CSP", 0.75));
        br1.to_csv(path.to_str().unwrap()).unwrap();

        let mut br2 = BenchmarkResults::new();
        br2.push(BenchmarkResult::new("ds001", "02", "CSP", 0.80));
        br2.append_csv(path.to_str().unwrap()).unwrap();

        let loaded = BenchmarkResults::from_csv(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.len(), 2);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
