//! Integration test: create a minimal BIDS dataset on disk and verify
//! round-trip through BidsLayout → query → verify results.

use std::path::Path;

/// Create a minimal valid BIDS dataset in a temp directory.
fn create_fixture(root: &Path) {
    std::fs::create_dir_all(root).unwrap();

    // dataset_description.json
    std::fs::write(
        root.join("dataset_description.json"),
        r#"{"Name": "test-fixture", "BIDSVersion": "1.9.0", "DatasetType": "raw"}"#,
    )
    .unwrap();

    // participants.tsv
    std::fs::write(
        root.join("participants.tsv"),
        "participant_id\tage\tsex\nsub-01\t25\tM\nsub-02\t30\tF\n",
    )
    .unwrap();

    // sub-01
    let sub01_eeg = root.join("sub-01/eeg");
    std::fs::create_dir_all(&sub01_eeg).unwrap();
    std::fs::write(sub01_eeg.join("sub-01_task-rest_eeg.edf"), b"FAKE_EDF").unwrap();
    std::fs::write(
        sub01_eeg.join("sub-01_task-rest_eeg.json"),
        r#"{"SamplingFrequency": 256, "EEGReference": "Cz"}"#,
    )
    .unwrap();
    std::fs::write(
        sub01_eeg.join("sub-01_task-rest_channels.tsv"),
        "name\ttype\tunits\nFp1\tEEG\tuV\nFp2\tEEG\tuV\n",
    )
    .unwrap();
    std::fs::write(
        sub01_eeg.join("sub-01_task-rest_events.tsv"),
        "onset\tduration\ttrial_type\n1.0\t0.5\tstimulus\n3.0\t0.5\tresponse\n",
    )
    .unwrap();

    // sub-01 task-motor
    std::fs::write(sub01_eeg.join("sub-01_task-motor_eeg.edf"), b"FAKE_EDF2").unwrap();
    std::fs::write(
        sub01_eeg.join("sub-01_task-motor_eeg.json"),
        r#"{"SamplingFrequency": 512}"#,
    )
    .unwrap();

    // sub-02
    let sub02_eeg = root.join("sub-02/eeg");
    std::fs::create_dir_all(&sub02_eeg).unwrap();
    std::fs::write(sub02_eeg.join("sub-02_task-rest_eeg.edf"), b"FAKE_EDF3").unwrap();
}

#[test]
fn test_layout_round_trip() {
    let tmp = std::env::temp_dir().join("bids_roundtrip_test");
    let _ = std::fs::remove_dir_all(&tmp);
    create_fixture(&tmp);

    let layout = bids_layout::BidsLayout::new(&tmp).unwrap();

    // Basic info
    let desc = layout.description().unwrap();
    assert_eq!(desc.name, "test-fixture");

    let subjects = layout.get_subjects().unwrap();
    assert_eq!(subjects.len(), 2);
    assert!(subjects.contains(&"01".to_string()));
    assert!(subjects.contains(&"02".to_string()));

    let tasks = layout.get_tasks().unwrap();
    assert!(tasks.contains(&"rest".to_string()));
    assert!(tasks.contains(&"motor".to_string()));

    // Query by subject
    let sub01_files = layout
        .get()
        .subject("01")
        .suffix("eeg")
        .extension(".edf")
        .collect()
        .unwrap();
    assert_eq!(sub01_files.len(), 2); // rest + motor

    // Query by task
    let rest_files = layout
        .get()
        .task("rest")
        .suffix("eeg")
        .extension(".edf")
        .collect()
        .unwrap();
    assert_eq!(rest_files.len(), 2); // sub-01 + sub-02

    // Query with multiple filters
    let specific = layout
        .get()
        .subject("01")
        .task("rest")
        .suffix("eeg")
        .extension(".edf")
        .collect()
        .unwrap();
    assert_eq!(specific.len(), 1);
    assert!(specific[0].filename.contains("sub-01"));
    assert!(specific[0].filename.contains("task-rest"));

    // Metadata inheritance
    let md = layout.get_metadata(&specific[0].path).unwrap();
    assert_eq!(md.get_f64("SamplingFrequency"), Some(256.0));
    assert_eq!(md.get_str("EEGReference"), Some("Cz"));

    // Companion file
    let companion = specific[0].companion("channels", "tsv");
    assert!(companion.exists());

    // Get all entities
    let entities = layout.get_entities().unwrap();
    assert!(entities.contains(&"subject".to_string()));
    assert!(entities.contains(&"task".to_string()));
    assert!(entities.contains(&"suffix".to_string()));

    // TSV reading
    let events_files = layout
        .get()
        .subject("01")
        .task("rest")
        .suffix("events")
        .collect()
        .unwrap();
    assert_eq!(events_files.len(), 1);
    let rows = events_files[0].get_df().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].get("trial_type").unwrap(), "stimulus");

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_metadata_inheritance() {
    // Test the BIDS inheritance principle:
    // root-level JSON < datatype-level < subject-level < file-level
    let tmp = std::env::temp_dir().join("bids_inheritance_test");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    std::fs::write(
        tmp.join("dataset_description.json"),
        r#"{"Name": "inherit-test", "BIDSVersion": "1.9.0"}"#,
    )
    .unwrap();

    // Root-level sidecar: applies to all EEG files
    std::fs::write(
        tmp.join("task-rest_eeg.json"),
        r#"{"EEGReference": "Cz", "PowerLineFrequency": 50, "Manufacturer": "BioSemi"}"#,
    )
    .unwrap();

    // Subject-level sidecar: overrides PowerLineFrequency for sub-01
    let sub01_eeg = tmp.join("sub-01/eeg");
    std::fs::create_dir_all(&sub01_eeg).unwrap();
    std::fs::write(
        sub01_eeg.join("sub-01_task-rest_eeg.json"),
        r#"{"SamplingFrequency": 256, "PowerLineFrequency": 60}"#,
    )
    .unwrap();
    std::fs::write(sub01_eeg.join("sub-01_task-rest_eeg.edf"), b"FAKE").unwrap();

    // sub-02: no file-level sidecar, should only get root-level
    let sub02_eeg = tmp.join("sub-02/eeg");
    std::fs::create_dir_all(&sub02_eeg).unwrap();
    std::fs::write(sub02_eeg.join("sub-02_task-rest_eeg.edf"), b"FAKE").unwrap();

    let layout = bids_layout::BidsLayout::new(&tmp).unwrap();

    // sub-01: file-level PowerLineFrequency=60 should override root-level 50
    let sub01_files = layout
        .get()
        .subject("01")
        .suffix("eeg")
        .extension(".edf")
        .collect()
        .unwrap();
    assert_eq!(sub01_files.len(), 1);
    let md1 = layout.get_metadata(&sub01_files[0].path).unwrap();
    assert_eq!(
        md1.get_f64("SamplingFrequency"),
        Some(256.0),
        "File-level SamplingFrequency"
    );
    assert_eq!(
        md1.get_i64("PowerLineFrequency"),
        Some(60),
        "Overridden PowerLineFrequency"
    );
    assert_eq!(
        md1.get_str("EEGReference"),
        Some("Cz"),
        "Inherited EEGReference from root"
    );
    assert_eq!(
        md1.get_str("Manufacturer"),
        Some("BioSemi"),
        "Inherited Manufacturer from root"
    );

    // sub-02: should get root-level metadata only
    let sub02_files = layout
        .get()
        .subject("02")
        .suffix("eeg")
        .extension(".edf")
        .collect()
        .unwrap();
    assert_eq!(sub02_files.len(), 1);
    let md2 = layout.get_metadata(&sub02_files[0].path).unwrap();
    assert_eq!(
        md2.get_i64("PowerLineFrequency"),
        Some(50),
        "Root-level PowerLineFrequency"
    );
    assert_eq!(
        md2.get_str("EEGReference"),
        Some("Cz"),
        "Root-level EEGReference"
    );
    // sub-02 should NOT have SamplingFrequency (it's only in sub-01's sidecar)
    assert_eq!(
        md2.get_f64("SamplingFrequency"),
        None,
        "No SamplingFrequency for sub-02"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_persistent_database() {
    let tmp = std::env::temp_dir().join("bids_persist_test");
    let _ = std::fs::remove_dir_all(&tmp);
    create_fixture(&tmp);

    let db_path = tmp.join("index.sqlite");

    // Create layout with persistent database
    {
        let layout = bids_layout::BidsLayout::builder(&tmp)
            .database_path(&db_path)
            .build()
            .unwrap();
        let subjects = layout.get_subjects().unwrap();
        assert_eq!(subjects.len(), 2);
    }

    // Reload from database (should be instant, no re-indexing)
    {
        let layout = bids_layout::BidsLayout::load(&db_path).unwrap();
        let subjects = layout.get_subjects().unwrap();
        assert_eq!(subjects.len(), 2);
        let tasks = layout.get_tasks().unwrap();
        assert!(tasks.contains(&"rest".to_string()));
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_query_return_types() {
    let tmp = std::env::temp_dir().join("bids_return_types_test");
    let _ = std::fs::remove_dir_all(&tmp);
    create_fixture(&tmp);

    let layout = bids_layout::BidsLayout::new(&tmp).unwrap();

    // return_unique: get unique subject values for EEG files
    let subjects = layout
        .get()
        .suffix("eeg")
        .extension(".edf")
        .return_unique("subject")
        .unwrap();
    assert_eq!(subjects.len(), 2);
    assert!(subjects.contains(&"01".to_string()));
    assert!(subjects.contains(&"02".to_string()));

    // return_unique: get unique tasks
    let tasks = layout
        .get()
        .subject("01")
        .suffix("eeg")
        .return_unique("task")
        .unwrap();
    assert_eq!(tasks.len(), 2);
    assert!(tasks.contains(&"rest".to_string()));
    assert!(tasks.contains(&"motor".to_string()));

    // return_paths: get file paths
    let paths = layout
        .get()
        .subject("01")
        .task("rest")
        .suffix("eeg")
        .extension(".edf")
        .return_paths()
        .unwrap();
    assert_eq!(paths.len(), 1);
    assert!(paths[0].to_string_lossy().contains("sub-01"));

    // Query::None — files WITHOUT a task entity (e.g., participants.tsv)
    let no_task = layout.get().query_none("task").collect().unwrap();
    // Should exclude all task-* files
    for f in &no_task {
        assert!(
            !f.filename.contains("task-"),
            "File {} should not have task",
            f.filename
        );
    }

    // Query::Any — files WITH any task entity
    let has_task = layout.get().query_any("task").collect().unwrap();
    for f in &has_task {
        assert!(
            f.entities.contains_key("task"),
            "File {} should have task",
            f.filename
        );
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_regex_filter() {
    let tmp = std::env::temp_dir().join("bids_regex_test");
    let _ = std::fs::remove_dir_all(&tmp);
    create_fixture(&tmp);

    let layout = bids_layout::BidsLayout::new(&tmp).unwrap();

    // Regex filter: subject matching "0[12]"
    let files = layout
        .get()
        .filter_regex("subject", "0[12]")
        .suffix("eeg")
        .extension(".edf")
        .collect()
        .unwrap();
    assert!(files.len() >= 2, "Should match both sub-01 and sub-02");

    // Regex filter: task starting with "r"
    let rest_files = layout
        .get()
        .filter_regex("task", "^r")
        .suffix("eeg")
        .extension(".edf")
        .collect()
        .unwrap();
    for f in &rest_files {
        assert!(
            f.entities
                .get("task")
                .unwrap()
                .as_str_lossy()
                .starts_with("r")
        );
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_parse_and_build_path() {
    let tmp = std::env::temp_dir().join("bids_path_test");
    let _ = std::fs::remove_dir_all(&tmp);
    create_fixture(&tmp);

    let layout = bids_layout::BidsLayout::new(&tmp).unwrap();

    // Parse entities from a full relative path (config regexes need path context)
    let entities = layout.parse_file_entities("sub-01/eeg/sub-01_task-rest_eeg.edf");
    // At minimum, subject and task should be parsed
    assert!(
        entities.get("subject").is_some(),
        "Expected 'subject' entity, got: {:?}",
        entities.keys().collect::<Vec<_>>()
    );
    assert_eq!(entities.get("subject").unwrap().as_str_lossy(), "01");

    if let Some(task) = entities.get("task") {
        assert_eq!(task.as_str_lossy(), "rest");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_bidsfile_builder() {
    use bids_core::entities::Entities;
    use bids_core::entities::EntityValue;
    use bids_core::file::BidsFile;

    let mut entities = Entities::new();
    entities.insert("subject".to_string(), EntityValue::Str("01".to_string()));
    entities.insert("task".to_string(), EntityValue::Str("rest".to_string()));

    let bf = BidsFile::new("/data/sub-01_task-rest_eeg.edf").with_entities(entities);

    assert_eq!(bf.entities.get("subject").unwrap().as_str_lossy(), "01");
    assert_eq!(bf.entities.get("task").unwrap().as_str_lossy(), "rest");
}
