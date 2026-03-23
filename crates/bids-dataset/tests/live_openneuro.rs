//! Live integration tests against the real OpenNeuro API.
//!
//! These tests make actual HTTP requests. Run with:
//!   cargo test -p bids-dataset --test live_openneuro -- --nocapture
//!
//! They are behind `#[ignore]` by default so they don't run in CI without
//! network access. Use `--ignored` to include them.

use bids_dataset::{Aggregator, DatasetFilter, OpenNeuro, Split};
use std::path::Path;

#[test]
#[ignore] // requires network
fn test_search_eeg_datasets() {
    let on = OpenNeuro::new();
    let hits = on
        .search()
        .modality("eeg")
        .species("Human")
        .limit(10)
        .execute()
        .unwrap();

    println!("Found {} EEG datasets:", hits.len());
    for ds in &hits {
        println!(
            "  {} — {} (modalities: {:?}, size: {:?} bytes)",
            ds.id, ds.name, ds.modalities, ds.size_bytes
        );
    }
    assert!(!hits.is_empty(), "Should find at least one EEG dataset");
    assert!(
        hits.iter()
            .all(|d| d.modalities.iter().any(|m| m.eq_ignore_ascii_case("eeg")))
    );
}

#[test]
#[ignore]
fn test_search_mri_datasets() {
    let on = OpenNeuro::new();
    let hits = on.search().modality("mri").limit(5).execute().unwrap();

    println!("Found {} MRI datasets", hits.len());
    assert!(!hits.is_empty());
}

#[test]
#[ignore]
fn test_search_by_keyword() {
    let on = OpenNeuro::new();
    // Keyword search is client-side (OpenNeuro API has no text search).
    // We search EEG datasets and filter by name containing "eeg"
    let hits = on
        .search()
        .modality("eeg")
        .keyword("EEG")
        .limit(20)
        .execute()
        .unwrap();

    println!("Found {} datasets matching 'EEG' keyword:", hits.len());
    for ds in &hits[..hits.len().min(5)] {
        println!("  {} — {}", ds.id, ds.name);
    }
    assert!(!hits.is_empty(), "Should find datasets with 'EEG' in name");
}

#[test]
#[ignore]
fn test_list_files_s3() {
    let on = OpenNeuro::new();
    // ds004362 is a known EEG dataset
    let files = on.list_files("ds004362", Some("sub-001/eeg")).unwrap();

    println!("Files in ds004362/sub-001/eeg: {}", files.len());
    for f in &files[..files.len().min(10)] {
        println!("  {:60} {:>10} bytes", f.path, f.size);
    }
    assert!(
        !files.is_empty(),
        "Should find files in ds004362/sub-001/eeg"
    );
    assert!(files.iter().any(|f| f.path.ends_with(".json")
        || f.path.ends_with(".set")
        || f.path.ends_with(".tsv")));
}

#[test]
#[ignore]
fn test_list_full_dataset() {
    let on = OpenNeuro::new();
    let files = on.list_files("ds004362", None).unwrap();

    println!("Total files in ds004362: {}", files.len());
    let total_mb = files.iter().map(|f| f.size).sum::<u64>() as f64 / 1e6;
    println!("Total size: {:.1} MB", total_mb);

    // Count by extension
    let mut ext_counts = std::collections::HashMap::new();
    for f in &files {
        let ext = f.path.rsplit('.').next().unwrap_or("other");
        *ext_counts.entry(ext.to_string()).or_insert(0u32) += 1;
    }
    for (ext, count) in &ext_counts {
        println!("  .{}: {} files", ext, count);
    }

    assert!(files.len() > 100, "ds004362 should have hundreds of files");
}

#[test]
#[ignore]
fn test_download_small_files() {
    let on = OpenNeuro::new();
    let dir = std::env::temp_dir().join("bids_dataset_test_dl");
    std::fs::create_dir_all(&dir).unwrap();

    // Download only JSON metadata files from one subject (small)
    let report = on
        .download_dataset(
            "ds004362",
            &dir,
            Some(|f: &bids_dataset::RemoteFile| {
                f.path.starts_with("sub-001/eeg/") && f.path.ends_with(".json") && f.size < 10_000
            }),
        )
        .unwrap();

    println!("Download report: {}", report);
    assert!(
        report.downloaded > 0 || report.skipped > 0,
        "Should download or skip some files"
    );
    assert!(
        report.errors.is_empty(),
        "Should have no errors: {:?}",
        report.errors
    );

    // Verify files exist
    let local_root = dir.join("ds004362");
    assert!(local_root.exists());

    // Second run should skip everything (resume)
    let report2 = on
        .download_dataset(
            "ds004362",
            &dir,
            Some(|f: &bids_dataset::RemoteFile| {
                f.path.starts_with("sub-001/eeg/") && f.path.ends_with(".json") && f.size < 10_000
            }),
        )
        .unwrap();
    println!("Second run: {}", report2);
    assert_eq!(report2.downloaded, 0, "Should skip all on re-download");
    assert!(report2.skipped > 0);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore]
fn test_filter_and_aggregate() {
    let on = OpenNeuro::new();
    let dir = std::env::temp_dir().join("bids_dataset_test_agg");
    std::fs::create_dir_all(&dir).unwrap();

    // Download small metadata files from one subject
    let _ = on
        .download_dataset(
            "ds004362",
            &dir,
            Some(|f: &bids_dataset::RemoteFile| {
                (f.path.starts_with("sub-001/eeg/") || f.path == "dataset_description.json")
                    && (f.path.ends_with(".json") || f.path.ends_with(".tsv"))
                    && f.size < 10_000
            }),
        )
        .unwrap();

    // Aggregate with filter
    let mut agg = Aggregator::new();
    let count = agg
        .add_dataset(
            &dir.join("ds004362"),
            DatasetFilter::new().modality("eeg").extension(".json"),
        )
        .unwrap();
    println!("Aggregated {} files", count);
    assert!(count > 0);
    assert!(!agg.subjects().is_empty());

    // Export manifest
    let manifest = dir.join("manifest.csv");
    agg.export_manifest(manifest.to_str().unwrap()).unwrap();
    let content = std::fs::read_to_string(&manifest).unwrap();
    println!("Manifest:\n{}", &content[..content.len().min(500)]);
    assert!(content.contains("dataset,global_subject"));

    // Export splits
    let split_dir = dir.join("splits");
    let report = agg
        .export_split(split_dir.to_str().unwrap(), Split::ratio(0.6, 0.2, 0.2))
        .unwrap();
    println!("Split: {}", report);
    assert!(split_dir.join("train.csv").exists());
    assert!(split_dir.join("val.csv").exists());
    assert!(split_dir.join("test.csv").exists());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
#[ignore]
fn test_end_to_end_eeg_pipeline() {
    let on = OpenNeuro::new();

    // 1. Search
    let hits = on.search().modality("eeg").limit(5).execute().unwrap();
    println!("Step 1 — Found {} EEG motor datasets", hits.len());
    assert!(!hits.is_empty());

    // 2. List files from first hit
    let ds = &hits[0];
    let files = on.list_files(&ds.id, None).unwrap();
    println!("Step 2 — {} has {} files", ds.id, files.len());

    // 3. Filter to EEG data files only
    let filter = DatasetFilter::new().modality("eeg");
    let eeg_files = filter.filter_remote(&files);
    println!("Step 3 — {} EEG files after filter", eeg_files.len());

    // 4. Download metadata only (fast)
    let dir = std::env::temp_dir().join("bids_dataset_test_e2e");
    std::fs::create_dir_all(&dir).unwrap();
    let report = on
        .download_dataset(
            &ds.id,
            &dir,
            Some(|f: &bids_dataset::RemoteFile| f.path.ends_with(".json") && f.size < 5_000),
        )
        .unwrap();
    println!("Step 4 — Downloaded: {}", report);

    std::fs::remove_dir_all(&dir).unwrap();
}
