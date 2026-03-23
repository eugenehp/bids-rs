//! Integration tests against the official bids-examples repository.
//!
//! Uses [`BidsExamples`](bids_e2e_tests::fixtures::BidsExamples) for automatic
//! fixture management — no manual `git clone` required.
//!
//! ```bash
//! # Just run it — downloads bids-examples on first run:
//! cargo test -p bids-e2e-tests --test bids_examples
//!
//! # Explicit path:
//! BIDS_EXAMPLES_DIR=/path/to/bids-examples cargo test -p bids-e2e-tests --test bids_examples
//!
//! # Offline (skip if not cached):
//! BIDS_EXAMPLES_OFFLINE=1 cargo test -p bids-e2e-tests --test bids_examples
//! ```

use bids_e2e_tests::fixtures::BidsExamples;
use std::path::{Path, PathBuf};

/// Require the bids-examples fixture or skip the test.
macro_rules! require_examples {
    () => {
        match BidsExamples::require() {
            Some(ex) => ex,
            None => {
                eprintln!("SKIP: bids-examples not available");
                return;
            }
        }
    };
}

/// Discover all example datasets that have dataset_description.json.
fn discover_datasets(root: &Path) -> Vec<PathBuf> {
    let mut datasets = Vec::new();
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("dataset_description.json").exists() {
                datasets.push(path);
            }
        }
    }
    datasets.sort();
    datasets
}

/// Detect datatypes present in a dataset.
fn detect_datatypes(root: &Path) -> Vec<String> {
    let known = [
        "anat", "func", "dwi", "fmap", "eeg", "meg", "ieeg",
        "pet", "perf", "nirs", "motion", "mrs", "micr", "beh",
    ];
    let mut found = Vec::new();
    for entry in walkdir::WalkDir::new(root).max_depth(4).into_iter().flatten() {
        if entry.file_type().is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if known.contains(&name.as_str()) && !found.contains(&name) {
                found.push(name);
            }
        }
    }
    found.sort();
    found
}

// ─── Core: Every dataset can be indexed without errors ─────────────────────

#[test]
fn test_all_datasets_index_successfully() {
    let examples = require_examples!();
    let datasets = discover_datasets(examples.root());
    assert!(!datasets.is_empty(), "No datasets found in {}", examples.root().display());

    let mut pass = 0;
    let mut fail = 0;
    let mut errors = Vec::new();

    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        match bids_layout::BidsLayout::builder(ds).validate(false).build() {
            Ok(_) => pass += 1,
            Err(e) => {
                fail += 1;
                errors.push(format!("{name}: {e}"));
            }
        }
    }

    eprintln!("Indexed {pass}/{} datasets ({fail} failed)", datasets.len());
    for e in &errors {
        eprintln!("  FAIL: {e}");
    }
    assert!(
        pass as f64 / datasets.len() as f64 > 0.90,
        "Too many indexing failures: {fail}/{} datasets failed",
        datasets.len()
    );
}

// ─── Subjects: Every dataset has at least one subject ──────────────────────

#[test]
fn test_all_datasets_have_subjects() {
    let examples = require_examples!();
    let datasets = discover_datasets(examples.root());

    let mut no_subjects = Vec::new();
    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        if name.starts_with("atlas-") { continue; }

        if let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() {
            if let Ok(subjects) = layout.get_subjects() {
                if subjects.is_empty() {
                    no_subjects.push(name);
                }
            }
        }
    }

    if !no_subjects.is_empty() {
        eprintln!("Datasets with no subjects detected: {:?}", no_subjects);
    }
    assert!(no_subjects.len() < 5, "Too many datasets without subjects: {no_subjects:?}");
}

// ─── Metadata: JSON sidecar inheritance works on real datasets ─────────────

#[test]
fn test_metadata_inheritance() {
    let examples = require_examples!();
    let ds = match examples.dataset("ds001") {
        Some(p) => p,
        None => return,
    };

    let layout = bids_layout::BidsLayout::builder(&ds)
        .validate(false)
        .index_metadata(true)
        .build()
        .unwrap();

    let bold_files = layout.get().suffix("bold").extension(".nii.gz").collect().unwrap();
    for f in &bold_files {
        let md = layout.get_metadata(&f.path).unwrap();
        if let Some(tr) = md.get_f64("RepetitionTime") {
            assert!(tr > 0.0, "RepetitionTime should be positive, got {tr}");
        }
    }
}

// ─── EEG ───────────────────────────────────────────────────────────────────

#[test]
fn test_eeg_examples() {
    let examples = require_examples!();
    let eeg_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"eeg".to_string()))
        .collect();

    if eeg_datasets.is_empty() { return; }
    eprintln!("Testing {} EEG datasets", eeg_datasets.len());

    for ds in &eeg_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };

        let eeg = bids_eeg::EegLayout::new(&layout);
        let files = eeg.get_eeg_files().unwrap_or_default();
        eprintln!("  {name}: {} EEG files", files.len());

        for f in &files {
            if let Ok(Some(channels)) = eeg.get_channels(f) {
                assert!(!channels.is_empty(), "{name}: empty channels for {}", f.filename);
            }
            let _ = eeg.get_events(f);
            let _ = eeg.get_eeg_metadata(f);
        }
    }
}

// ─── MEG ───────────────────────────────────────────────────────────────────

#[test]
fn test_meg_examples() {
    let examples = require_examples!();
    let meg_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"meg".to_string()))
        .collect();

    if meg_datasets.is_empty() { return; }
    eprintln!("Testing {} MEG datasets", meg_datasets.len());

    for ds in &meg_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };

        let meg = bids_meg::MegLayout::new(&layout);
        let files = meg.get_meg_files().unwrap_or_default();
        eprintln!("  {name}: {} MEG files", files.len());

        for f in &files {
            let _ = meg.get_channels(f);
            let _ = meg.get_events(f);
            let _ = meg.get_metadata(f);
            if let Ok(Some(points)) = meg.get_headshape(f) {
                assert!(!points.is_empty());
            }
        }
    }
}

// ─── iEEG ──────────────────────────────────────────────────────────────────

#[test]
fn test_ieeg_examples() {
    let examples = require_examples!();
    let ieeg_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"ieeg".to_string()))
        .collect();

    if ieeg_datasets.is_empty() { return; }
    eprintln!("Testing {} iEEG datasets", ieeg_datasets.len());

    for ds in &ieeg_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };

        let ieeg = bids_ieeg::IeegLayout::new(&layout);
        let files = ieeg.get_ieeg_files().unwrap_or_default();
        eprintln!("  {name}: {} iEEG files", files.len());

        for f in &files {
            let _ = ieeg.get_channels(f);
            let _ = ieeg.get_electrodes(f);
            let _ = ieeg.get_events(f);
        }
    }
}

// ─── PET ───────────────────────────────────────────────────────────────────

#[test]
fn test_pet_examples() {
    let examples = require_examples!();
    let pet_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"pet".to_string()))
        .collect();

    if pet_datasets.is_empty() { return; }
    eprintln!("Testing {} PET datasets", pet_datasets.len());

    for ds in &pet_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };

        let pet = bids_pet::PetLayout::new(&layout);
        let files = pet.get_pet_files().unwrap_or_default();
        eprintln!("  {name}: {} PET files", files.len());

        for f in &files {
            if let Ok(Some(md)) = pet.get_metadata(f) {
                if let Some(ref t) = md.tracer_name {
                    eprintln!("    tracer: {t}");
                }
            }
            let _ = pet.get_blood(f);
        }
    }
}

// ─── Perfusion (ASL) ──────────────────────────────────────────────────────

#[test]
fn test_perf_examples() {
    let examples = require_examples!();
    let perf_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"perf".to_string()))
        .collect();

    if perf_datasets.is_empty() { return; }
    eprintln!("Testing {} perf/ASL datasets", perf_datasets.len());

    for ds in &perf_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };

        let perf = bids_perf::PerfLayout::new(&layout);
        let files = perf.get_asl_files().unwrap_or_default();
        eprintln!("  {name}: {} ASL files", files.len());

        for f in &files {
            if let Ok(Some(md)) = perf.get_metadata(f) {
                if let Some(ref t) = md.arterial_spin_labeling_type {
                    eprintln!("    ASL type: {t}");
                }
            }
            let _ = perf.get_aslcontext(f);
        }
    }
}

// ─── NIRS ──────────────────────────────────────────────────────────────────

#[test]
fn test_nirs_examples() {
    let examples = require_examples!();
    let nirs_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"nirs".to_string()))
        .collect();

    if nirs_datasets.is_empty() { return; }
    eprintln!("Testing {} NIRS datasets", nirs_datasets.len());

    for ds in &nirs_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };
        let nirs = bids_nirs::NirsLayout::new(&layout);
        let files = nirs.get_nirs_files().unwrap_or_default();
        eprintln!("  {name}: {} NIRS files", files.len());
    }
}

// ─── Motion ────────────────────────────────────────────────────────────────

#[test]
fn test_motion_examples() {
    let examples = require_examples!();
    let motion_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"motion".to_string()))
        .collect();

    if motion_datasets.is_empty() { return; }
    eprintln!("Testing {} motion datasets", motion_datasets.len());

    for ds in &motion_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };
        let motion = bids_motion::MotionLayout::new(&layout);
        let files = motion.get_motion_files().unwrap_or_default();
        eprintln!("  {name}: {} motion files", files.len());
    }
}

// ─── MRS ───────────────────────────────────────────────────────────────────

#[test]
fn test_mrs_examples() {
    let examples = require_examples!();
    let mrs_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"mrs".to_string()))
        .collect();

    if mrs_datasets.is_empty() { return; }
    eprintln!("Testing {} MRS datasets", mrs_datasets.len());

    for ds in &mrs_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };
        let mrs = bids_mrs::MrsLayout::new(&layout);
        let svs = mrs.get_svs_files().unwrap_or_default();
        let mrsi = mrs.get_mrsi_files().unwrap_or_default();
        eprintln!("  {name}: {} SVS, {} MRSI files", svs.len(), mrsi.len());
    }
}

// ─── Microscopy ────────────────────────────────────────────────────────────

#[test]
fn test_micr_examples() {
    let examples = require_examples!();
    let micr_datasets: Vec<_> = discover_datasets(examples.root())
        .into_iter()
        .filter(|ds| detect_datatypes(ds).contains(&"micr".to_string()))
        .collect();

    if micr_datasets.is_empty() { return; }
    eprintln!("Testing {} microscopy datasets", micr_datasets.len());

    for ds in &micr_datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Ok(layout) = bids_layout::BidsLayout::builder(ds).validate(false).build() else { continue };
        let files = layout.get().datatype("micr").collect().unwrap_or_default();
        eprintln!("  {name}: {} micr files", files.len());
    }
}

// ─── Validation ────────────────────────────────────────────────────────────

#[test]
fn test_validate_all_examples() {
    let examples = require_examples!();
    let datasets = discover_datasets(examples.root());

    let mut valid = 0;
    let mut total_errors = 0;
    let mut total_warnings = 0;

    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        match bids_validate::validate_dataset(ds) {
            Ok(result) => {
                if result.is_valid() { valid += 1; }
                total_errors += result.error_count();
                total_warnings += result.warning_count();
                if result.error_count() > 0 {
                    eprintln!("  {name}: {} errors, {} warnings",
                        result.error_count(), result.warning_count());
                }
            }
            Err(e) => eprintln!("  {name}: validation error: {e}"),
        }
    }

    eprintln!(
        "Validated {} datasets: {} valid, {} total errors, {} total warnings",
        datasets.len(), valid, total_errors, total_warnings
    );
    assert!(
        valid as f64 / datasets.len() as f64 > 0.80,
        "Too few valid datasets: {valid}/{}", datasets.len()
    );
}

// ─── Schema ────────────────────────────────────────────────────────────────

#[test]
fn test_schema_validates_example_filenames() {
    let examples = require_examples!();
    let schema = bids_schema::BidsSchema::load();

    for ds_name in &["ds001", "eeg_cbm", "eeg_matchingpennies"] {
        let Some(ds) = examples.dataset(ds_name) else { continue };

        let mut total = 0;
        let mut valid = 0;
        for entry in walkdir::WalkDir::new(&ds).into_iter().flatten() {
            if !entry.file_type().is_file() { continue; }
            let rel = entry.path().strip_prefix(&ds).unwrap().to_string_lossy().to_string();
            if rel.starts_with('.') { continue; }
            total += 1;
            if schema.is_valid(&rel) { valid += 1; }
        }

        let pct = if total > 0 { valid as f64 / total as f64 * 100.0 } else { 100.0 };
        eprintln!("  {ds_name}: {valid}/{total} files pass schema ({pct:.0}%)");
        assert!(pct > 40.0, "{ds_name}: only {pct:.0}% of files pass schema validation");
    }
}

// ─── Entity parsing ───────────────────────────────────────────────────────

#[test]
fn test_entity_parsing_on_examples() {
    let examples = require_examples!();
    let config = bids_core::Config::bids();

    let Some(ds) = examples.dataset("eeg_cbm") else { return };

    for entry in walkdir::WalkDir::new(&ds).into_iter().flatten() {
        if !entry.file_type().is_file() { continue; }
        let path = entry.path().to_string_lossy().to_string();
        if !path.contains("sub-") { continue; }

        let entities = bids_core::entities::parse_file_entities(&path, &config.entities);
        assert!(
            entities.contains_key("subject"),
            "No subject entity found in: {path}"
        );

        let name = entry.file_name().to_string_lossy().to_string();
        if name.contains('_') && name.contains('.') && !name.starts_with('.') {
            assert!(
                entities.contains_key("suffix"),
                "No suffix entity found in: {path}"
            );
        }
    }
}

// ─── Derivatives ──────────────────────────────────────────────────────────

#[test]
fn test_derivative_examples() {
    let examples = require_examples!();
    let Some(ds) = examples.dataset("ds000001-fmriprep") else { return };

    let layout = bids_layout::BidsLayout::builder(&ds)
        .validate(false)
        .is_derivative(true)
        .build()
        .unwrap();

    assert!(layout.is_derivative);
    let subjects = layout.get_subjects().unwrap();
    assert!(!subjects.is_empty(), "fmriprep derivative should have subjects");
    eprintln!("ds000001-fmriprep: {} subjects", subjects.len());
}
