//! Aggregate multiple BIDS datasets into a unified file collection.

use crate::filter::{self, DatasetFilter};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

/// A single file entry in the aggregated collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    /// Source dataset identifier (directory name).
    pub dataset: String,
    /// Subject ID (without "sub-" prefix).
    pub subject: String,
    /// Session (if any).
    pub session: Option<String>,
    /// Task (if any).
    pub task: Option<String>,
    /// Run (if any).
    pub run: Option<String>,
    /// Datatype / modality (e.g., "eeg", "anat").
    pub datatype: String,
    /// File suffix (e.g., "eeg", "bold", "T1w").
    pub suffix: String,
    /// File extension (e.g., ".edf", ".nii.gz").
    pub extension: String,
    /// Absolute path to the file on disk.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
    /// Globally unique subject ID: "{dataset}_{subject}" to avoid collisions.
    pub global_subject: String,
}

/// Aggregates files from multiple BIDS datasets into a flat collection.
///
/// Each added dataset gets a namespace prefix on subject IDs to avoid
/// collisions when combining datasets.
pub struct Aggregator {
    /// All collected file entries.
    pub files: Vec<FileEntry>,
    /// Dataset names added so far.
    pub datasets: Vec<String>,
}

impl Aggregator {
    pub fn new() -> Self {
        Self { files: Vec::new(), datasets: Vec::new() }
    }

    /// Add a local BIDS dataset directory, applying the given filter.
    pub fn add_dataset(&mut self, root: &Path, filter: DatasetFilter) -> crate::Result<usize> {
        // Derive dataset name: if path looks like .../datasets/{id}/snapshots/{ver}, use {id}
        let dataset_name = infer_dataset_name(root);

        let matching = filter::filter_local(root, &filter)?;
        let count = matching.len();

        for bf in matching {
            let ent = |key: &str| -> String {
                bf.entities.get(key).map(std::string::ToString::to_string).unwrap_or_default()
            };
            let ent_opt = |key: &str| -> Option<String> {
                bf.entities.get(key).map(std::string::ToString::to_string)
            };
            let subject = ent("subject");
            let entry = FileEntry {
                global_subject: format!("{dataset_name}_{subject}"),
                dataset: dataset_name.clone(),
                subject,
                session: ent_opt("session"),
                task: ent_opt("task"),
                run: ent_opt("run"),
                datatype: ent("datatype"),
                suffix: ent("suffix"),
                extension: bf.extension().to_string(),
                size: std::fs::metadata(&bf.path).map(|m| m.len()).unwrap_or(0),
                path: bf.path.clone(),
            };
            self.files.push(entry);
        }

        if !self.datasets.contains(&dataset_name) {
            self.datasets.push(dataset_name);
        }

        Ok(count)
    }

    /// Number of files in the aggregate.
    pub fn len(&self) -> usize { self.files.len() }

    /// Whether the aggregate is empty.
    pub fn is_empty(&self) -> bool { self.files.is_empty() }

    /// Iterate over entries (zero-copy). Enables streaming pipelines:
    ///
    /// ```ignore
    /// agg.iter()
    ///     .filter(|e| e.datatype == "eeg")
    ///     .for_each(|e| println!("{}", e.path.display()));
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &FileEntry> {
        self.files.iter()
    }



    /// Stream entries matching a predicate directly to a CSV writer.
    /// Never holds more than one entry in memory.
    pub fn stream_to_csv<F>(&self, path: &str, predicate: F) -> crate::Result<usize>
    where F: Fn(&FileEntry) -> bool
    {
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "dataset,global_subject,subject,session,task,run,datatype,suffix,extension,path,size")?;
        let mut count = 0;
        for e in self.files.iter().filter(|e| predicate(e)) {
            writeln!(f, "{},{},{},{},{},{},{},{},{},{},{}",
                e.dataset, e.global_subject, e.subject,
                e.session.as_deref().unwrap_or(""),
                e.task.as_deref().unwrap_or(""),
                e.run.as_deref().unwrap_or(""),
                e.datatype, e.suffix, e.extension,
                e.path.display(), e.size)?;
            count += 1;
        }
        Ok(count)
    }

    /// Get unique global subject IDs.
    pub fn subjects(&self) -> Vec<String> {
        let mut s: Vec<String> = self.files.iter()
            .map(|f| f.global_subject.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        s.sort();
        s
    }

    /// Get unique tasks.
    pub fn tasks(&self) -> Vec<String> {
        let mut t: Vec<String> = self.files.iter()
            .filter_map(|f| f.task.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        t.sort();
        t
    }

    /// Export a CSV manifest of all files.
    ///
    /// Columns: dataset, global_subject, subject, session, task, run, datatype,
    /// suffix, extension, path, size
    pub fn export_manifest(&self, path: &str) -> crate::Result<()> {
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "dataset,global_subject,subject,session,task,run,datatype,suffix,extension,path,size")?;
        for e in &self.files {
            writeln!(f, "{},{},{},{},{},{},{},{},{},{},{}",
                e.dataset, e.global_subject, e.subject,
                e.session.as_deref().unwrap_or(""),
                e.task.as_deref().unwrap_or(""),
                e.run.as_deref().unwrap_or(""),
                e.datatype, e.suffix, e.extension,
                e.path.display(), e.size)?;
        }
        Ok(())
    }

    /// Export train/val/test splits as separate CSV manifests.
    ///
    /// Splits are done at the **subject** level (all files from a subject go
    /// into the same split) to prevent data leakage.
    pub fn export_split(&self, dir: &str, split: crate::Split) -> crate::Result<SplitReport> {
        std::fs::create_dir_all(dir)?;
        let subjects = self.subjects();
        let (train_subs, val_subs, test_subs) = split.partition(&subjects);

        let write_split = |name: &str, subs: &[String]| -> crate::Result<usize> {
            let path = Path::new(dir).join(format!("{name}.csv"));
            let entries: Vec<&FileEntry> = self.files.iter()
                .filter(|f| subs.contains(&f.global_subject))
                .collect();
            let mut f = std::fs::File::create(&path)?;
            writeln!(f, "dataset,global_subject,subject,session,task,run,datatype,suffix,extension,path,size")?;
            for e in &entries {
                writeln!(f, "{},{},{},{},{},{},{},{},{},{},{}",
                    e.dataset, e.global_subject, e.subject,
                    e.session.as_deref().unwrap_or(""),
                    e.task.as_deref().unwrap_or(""),
                    e.run.as_deref().unwrap_or(""),
                    e.datatype, e.suffix, e.extension,
                    e.path.display(), e.size)?;
            }
            Ok(entries.len())
        };

        let n_train = write_split("train", &train_subs)?;
        let n_val = write_split("val", &val_subs)?;
        let n_test = write_split("test", &test_subs)?;

        Ok(SplitReport {
            train_subjects: train_subs.len(),
            val_subjects: val_subs.len(),
            test_subjects: test_subs.len(),
            train_files: n_train,
            val_files: n_val,
            test_files: n_test,
        })
    }

    /// Export the manifest as Apache Arrow IPC (Feather) format.
    ///
    /// Requires the `arrow` feature flag.
    #[cfg(feature = "arrow")]
    pub fn export_arrow(&self, path: &str) -> crate::Result<()> {
        use arrow::builder::StringBuilder;
        use arrow::builder::UInt64Builder;
        use arrow_schema::{Schema, Field, DataType};
        use std::sync::Arc;

        let n = self.files.len();
        let fields = vec![
            Field::new("dataset", DataType::Utf8, false),
            Field::new("global_subject", DataType::Utf8, false),
            Field::new("subject", DataType::Utf8, false),
            Field::new("session", DataType::Utf8, true),
            Field::new("task", DataType::Utf8, true),
            Field::new("run", DataType::Utf8, true),
            Field::new("datatype", DataType::Utf8, false),
            Field::new("suffix", DataType::Utf8, false),
            Field::new("extension", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("size", DataType::UInt64, false),
        ];
        let schema = Arc::new(Schema::new(fields));

        let mut dataset_b = StringBuilder::with_capacity(n, n * 10);
        let mut gsub_b = StringBuilder::with_capacity(n, n * 15);
        let mut sub_b = StringBuilder::with_capacity(n, n * 5);
        let mut ses_b = StringBuilder::with_capacity(n, n * 5);
        let mut task_b = StringBuilder::with_capacity(n, n * 10);
        let mut run_b = StringBuilder::with_capacity(n, n * 3);
        let mut dt_b = StringBuilder::with_capacity(n, n * 5);
        let mut suf_b = StringBuilder::with_capacity(n, n * 5);
        let mut ext_b = StringBuilder::with_capacity(n, n * 5);
        let mut path_b = StringBuilder::with_capacity(n, n * 80);
        let mut size_b = UInt64Builder::with_capacity(n);

        for e in &self.files {
            dataset_b.append_value(&e.dataset);
            gsub_b.append_value(&e.global_subject);
            sub_b.append_value(&e.subject);
            ses_b.append_option(e.session.as_deref());
            task_b.append_option(e.task.as_deref());
            run_b.append_option(e.run.as_deref());
            dt_b.append_value(&e.datatype);
            suf_b.append_value(&e.suffix);
            ext_b.append_value(&e.extension);
            path_b.append_value(e.path.to_string_lossy().as_ref());
            size_b.append_value(e.size);
        }

        let batch = arrow::RecordBatch::try_new(schema.clone(), vec![
            Arc::new(dataset_b.finish()),
            Arc::new(gsub_b.finish()),
            Arc::new(sub_b.finish()),
            Arc::new(ses_b.finish()),
            Arc::new(task_b.finish()),
            Arc::new(run_b.finish()),
            Arc::new(dt_b.finish()),
            Arc::new(suf_b.finish()),
            Arc::new(ext_b.finish()),
            Arc::new(path_b.finish()),
            Arc::new(size_b.finish()),
        ]).map_err(|e| crate::Error::Network(e.to_string()))?;

        let file = std::fs::File::create(path)?;
        let mut writer = arrow_csv::WriterBuilder::new()
            .build(file);
        writer.write(&batch).map_err(|e| crate::Error::Network(e.to_string()))?;

        Ok(())
    }

    /// Export the manifest as Parquet format.
    ///
    /// Requires the `arrow` feature flag.
    #[cfg(feature = "arrow")]
    pub fn export_parquet(&self, path: &str) -> crate::Result<()> {
        use arrow::builder::StringBuilder;
        use arrow::builder::UInt64Builder;
        use arrow_schema::{Schema, Field, DataType};
        use std::sync::Arc;

        let n = self.files.len();
        let fields = vec![
            Field::new("dataset", DataType::Utf8, false),
            Field::new("global_subject", DataType::Utf8, false),
            Field::new("subject", DataType::Utf8, false),
            Field::new("session", DataType::Utf8, true),
            Field::new("task", DataType::Utf8, true),
            Field::new("datatype", DataType::Utf8, false),
            Field::new("suffix", DataType::Utf8, false),
            Field::new("path", DataType::Utf8, false),
            Field::new("size", DataType::UInt64, false),
        ];
        let schema = Arc::new(Schema::new(fields));

        let mut dataset_b = StringBuilder::with_capacity(n, n * 10);
        let mut gsub_b = StringBuilder::with_capacity(n, n * 15);
        let mut sub_b = StringBuilder::with_capacity(n, n * 5);
        let mut ses_b = StringBuilder::with_capacity(n, n * 5);
        let mut task_b = StringBuilder::with_capacity(n, n * 10);
        let mut dt_b = StringBuilder::with_capacity(n, n * 5);
        let mut suf_b = StringBuilder::with_capacity(n, n * 5);
        let mut path_b = StringBuilder::with_capacity(n, n * 80);
        let mut size_b = UInt64Builder::with_capacity(n);

        for e in &self.files {
            dataset_b.append_value(&e.dataset);
            gsub_b.append_value(&e.global_subject);
            sub_b.append_value(&e.subject);
            ses_b.append_option(e.session.as_deref());
            task_b.append_option(e.task.as_deref());
            dt_b.append_value(&e.datatype);
            suf_b.append_value(&e.suffix);
            path_b.append_value(e.path.to_string_lossy().as_ref());
            size_b.append_value(e.size);
        }

        let batch = arrow::RecordBatch::try_new(schema.clone(), vec![
            Arc::new(dataset_b.finish()),
            Arc::new(gsub_b.finish()),
            Arc::new(sub_b.finish()),
            Arc::new(ses_b.finish()),
            Arc::new(task_b.finish()),
            Arc::new(dt_b.finish()),
            Arc::new(suf_b.finish()),
            Arc::new(path_b.finish()),
            Arc::new(size_b.finish()),
        ]).map_err(|e| crate::Error::Network(e.to_string()))?;

        let file = std::fs::File::create(path)?;
        let props = parquet::file::properties::WriterProperties::builder().build();
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, Some(props))
            .map_err(|e| crate::Error::Network(e.to_string()))?;
        writer.write(&batch).map_err(|e| crate::Error::Network(e.to_string()))?;
        writer.close().map_err(|e| crate::Error::Network(e.to_string()))?;

        Ok(())
    }
}

impl Default for Aggregator {
    fn default() -> Self { Self::new() }
}

impl IntoIterator for Aggregator {
    type Item = FileEntry;
    type IntoIter = std::vec::IntoIter<FileEntry>;
    fn into_iter(self) -> Self::IntoIter {
        self.files.into_iter()
    }
}

/// Infer dataset name from path. For cache paths like
/// `.../datasets/ds004362/snapshots/1.0.0`, returns "ds004362".
fn infer_dataset_name(root: &Path) -> String {
    let components: Vec<&str> = root.iter().filter_map(|c| c.to_str()).collect();
    // Look for "datasets/{id}/snapshots" pattern
    for (i, c) in components.iter().enumerate() {
        if *c == "snapshots" && i >= 2 && components[i - 2] == "datasets" {
            return components[i - 1].to_string();
        }
    }
    // Fallback: last non-empty component
    root.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string()
}

/// Summary of a train/val/test split.
#[derive(Debug, Clone)]
pub struct SplitReport {
    pub train_subjects: usize,
    pub val_subjects: usize,
    pub test_subjects: usize,
    pub train_files: usize,
    pub val_files: usize,
    pub test_files: usize,
}

impl std::fmt::Display for SplitReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "train: {} subj / {} files, val: {} subj / {} files, test: {} subj / {} files",
            self.train_subjects, self.train_files,
            self.val_subjects, self.val_files,
            self.test_subjects, self.test_files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_entry_serde() {
        let e = FileEntry {
            dataset: "ds001".into(), subject: "01".into(),
            session: None, task: Some("rest".into()), run: None,
            datatype: "eeg".into(), suffix: "eeg".into(), extension: ".edf".into(),
            path: "/data/ds001/sub-01/eeg/sub-01_task-rest_eeg.edf".into(),
            size: 1000, global_subject: "ds001_01".into(),
        };
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("ds001_01"));
    }
}
