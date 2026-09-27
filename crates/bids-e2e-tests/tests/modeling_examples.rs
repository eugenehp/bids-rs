//! End-to-end tests of the variables and modeling stack against real datasets.
//!
//! [`bids_examples.rs`](./bids_examples.rs) covers indexing, validation and
//! entity parsing. This file covers what happens *after* indexing: loading
//! variables with real run timing, applying the `pybids-transforms-v1`
//! transformations to real `events.tsv` data, building design matrices, and
//! producing model diagnostics.
//!
//! ```bash
//! cargo test -p bids-e2e-tests --test modeling_examples
//! ```

use bids_e2e_tests::fixtures::BidsExamples;
use bids_layout::BidsLayout;
use bids_modeling::report::{est_vif, generate_report, get_all_contrast_vif, node_report};
use bids_modeling::spec::{GlmSpec, Prior, Term, VarComp};
use bids_modeling::transformations::{TransformSpec, apply_transformations};
use bids_variables::collections::VariableCollection;
use bids_variables::{LayoutVariables, load_variables};
use std::path::{Path, PathBuf};

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

/// Datasets that ship at least one `events.tsv`, since those are the ones with
/// run-level variables to model.
fn datasets_with_events(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let path = entry.path();
        if !path.is_dir() || !path.join("dataset_description.json").exists() {
            continue;
        }
        let has_events = walkdir::WalkDir::new(&path)
            .max_depth(4)
            .into_iter()
            .flatten()
            .any(|e| e.file_name().to_string_lossy().ends_with("_events.tsv"));
        if has_events {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn open(ds: &Path) -> Option<BidsLayout> {
    BidsLayout::builder(ds).validate(false).build().ok()
}

fn spec(instructions: Vec<serde_json::Value>) -> TransformSpec {
    TransformSpec {
        transformer: "pybids-transforms-v1".into(),
        instructions,
    }
}

/// Variable names in a collection, sorted.
///
/// `VariableCollection::variables` is a `HashMap`, so iterating it directly
/// picks a different variable on every run. Every "pick one" below goes
/// through this, so a failure is reproducible.
fn sorted_names(coll: &VariableCollection) -> Vec<String> {
    let mut names: Vec<String> = coll.variables.keys().cloned().collect();
    names.sort();
    names
}

/// The first sparse, numeric variable by name — the kind `Convolve` and
/// `ToDense` operate on.
fn first_sparse_numeric(coll: &VariableCollection) -> Option<String> {
    sorted_names(coll)
        .into_iter()
        .find(|k| coll.timing.contains_key(k) && coll.variables[k].is_numeric)
}

/// Whether a variable carries any non-zero, finite amplitude.
///
/// A regressor that is identically zero convolves to zero, which is correct
/// rather than a defect — several example datasets ship such columns.
fn has_signal(coll: &VariableCollection, name: &str) -> bool {
    coll.variables[name]
        .values
        .iter()
        .any(|v| v.is_finite() && *v != 0.0)
}

/// The first run-level collection in `ds` that has at least one variable.
fn first_run_collection(layout: &BidsLayout) -> Option<VariableCollection> {
    layout
        .get_collections("run", None, None, false)
        .ok()?
        .into_iter()
        .find(|c| !c.variables.is_empty())
}

// ─── Variables load with usable run timing ─────────────────────────────────

#[test]
fn run_collections_load_across_all_example_datasets() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());
    assert!(!datasets.is_empty(), "no datasets with events.tsv found");

    let mut with_vars = 0;
    let mut total_vars = 0;
    let mut failures = Vec::new();

    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Some(layout) = open(ds) else { continue };
        match layout.get_collections("run", None, None, false) {
            Ok(colls) => {
                let n: usize = colls.iter().map(|c| c.variables.len()).sum();
                if n > 0 {
                    with_vars += 1;
                    total_vars += n;
                }
            }
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }

    eprintln!(
        "Run collections: {with_vars}/{} datasets yielded variables ({total_vars} total)",
        datasets.len()
    );
    for f in &failures {
        eprintln!("  FAIL: {f}");
    }
    assert!(failures.is_empty(), "{} datasets errored", failures.len());
    assert!(
        with_vars * 2 > datasets.len(),
        "only {with_vars}/{} datasets produced variables",
        datasets.len()
    );
}

#[test]
fn run_collections_carry_event_timing() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    let mut timed = 0;
    let mut checked = 0;

    for ds in &datasets {
        let Some(layout) = open(ds) else { continue };
        let Some(coll) = first_run_collection(&layout) else {
            continue;
        };
        checked += 1;

        // Every sparse variable must have onsets parallel to its values —
        // this is what ToDense, Resample and Convolve depend on.
        for (name, var) in &coll.variables {
            if let Some(t) = coll.timing.get(name) {
                assert_eq!(
                    t.onset.len(),
                    var.len(),
                    "{}: '{name}' has {} onsets for {} values",
                    ds.display(),
                    t.onset.len(),
                    var.len()
                );
                assert_eq!(t.duration.len(), var.len(), "{}: '{name}'", ds.display());
            }
        }
        if coll.run_duration() > 0.0 {
            timed += 1;
        }
    }

    eprintln!("Run timing: {timed}/{checked} collections have a non-zero run duration");
    assert!(checked > 0, "no run collections to check");
    // Without a duration nothing can be densified or convolved, so this must
    // hold for the overwhelming majority of real datasets.
    assert!(
        timed * 10 >= checked * 9,
        "only {timed}/{checked} collections had a run duration"
    );
}

// ─── Transformations on real events data ───────────────────────────────────

#[test]
fn factor_and_scale_run_on_real_events() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    let mut factored = 0;
    let mut scaled = 0;
    let mut errors = Vec::new();

    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Some(layout) = open(ds) else { continue };
        let Some(mut coll) = first_run_collection(&layout) else {
            continue;
        };

        // Factor a categorical column.
        if coll.variables.contains_key("trial_type") {
            let before: Vec<String> = coll.variables.keys().cloned().collect();
            let errs = apply_transformations(
                &mut coll,
                &spec(vec![
                    serde_json::json!({"Name": "Factor", "Input": ["trial_type"]}),
                ]),
            );
            for e in &errs {
                errors.push(format!("{name} Factor: {e}"));
            }
            if errs.is_empty() {
                assert!(
                    !coll.variables.contains_key("trial_type"),
                    "{name}: Factor should consume its input"
                );
                let new: Vec<&String> = coll
                    .variables
                    .keys()
                    .filter(|k| !before.contains(k))
                    .collect();
                assert!(!new.is_empty(), "{name}: Factor produced no levels");
                // Dummy columns are 0/1.
                for k in new {
                    for v in &coll.variables[k].values {
                        assert!(
                            *v == 0.0 || *v == 1.0,
                            "{name}: '{k}' is not dummy-coded ({v})"
                        );
                    }
                }
                factored += 1;
            }
        }

        // Scale a numeric column with more than one distinct value.
        let numeric: Option<String> = sorted_names(&coll).into_iter().find(|k| {
            let v = &coll.variables[k];
            v.is_numeric && {
                let finite: Vec<u64> = v
                    .values
                    .iter()
                    .filter(|x| x.is_finite())
                    .map(|x| x.to_bits())
                    .collect();
                finite.len() > 2
                    && finite
                        .iter()
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        > 1
            }
        });

        if let Some(col) = numeric {
            let errs = apply_transformations(
                &mut coll,
                &spec(vec![serde_json::json!({"Name": "Scale", "Input": [col]})]),
            );
            for e in &errs {
                errors.push(format!("{name} Scale: {e}"));
            }
            if errs.is_empty() {
                let vals: Vec<f64> = coll.variables[&col]
                    .values
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                assert!(
                    mean.abs() < 1e-6,
                    "{name}: scaled '{col}' has mean {mean}, expected ~0"
                );
                scaled += 1;
            }
        }
    }

    eprintln!("Factor ran on {factored} datasets, Scale on {scaled}");
    for e in errors.iter().take(10) {
        eprintln!("  ERROR: {e}");
    }
    assert!(errors.is_empty(), "{} transformation errors", errors.len());
    assert!(factored > 5, "Factor only exercised on {factored} datasets");
    assert!(scaled > 5, "Scale only exercised on {scaled} datasets");
}

#[test]
fn convolve_runs_on_real_events() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    let mut convolved = 0;
    let mut errors = Vec::new();

    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Some(layout) = open(ds) else { continue };
        let Some(mut coll) = first_run_collection(&layout) else {
            continue;
        };
        if coll.run_duration() <= 0.0 {
            continue;
        }
        let Some(target) = first_sparse_numeric(&coll) else {
            continue;
        };
        let had_signal = has_signal(&coll, &target);

        let errs = apply_transformations(
            &mut coll,
            &spec(vec![serde_json::json!({
                "Name": "Convolve", "Input": [target], "Model": "spm"
            })]),
        );
        if !errs.is_empty() {
            for e in &errs {
                errors.push(format!("{name}: {e}"));
            }
            continue;
        }

        let values = &coll.variables[&target].values;
        assert!(
            coll.is_dense(&target),
            "{name}: convolved '{target}' is not dense"
        );
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{name}: convolved '{target}' has non-finite values"
        );
        if had_signal {
            assert!(
                values.iter().any(|v| v.abs() > 1e-12),
                "{name}: '{target}' had non-zero amplitudes but convolved to all zeros"
            );
            convolved += 1;
        }
    }

    eprintln!("Convolve produced a non-zero regressor on {convolved} datasets");
    for e in errors.iter().take(10) {
        eprintln!("  ERROR: {e}");
    }
    assert!(errors.is_empty(), "{} convolution errors", errors.len());
    assert!(
        convolved > 10,
        "Convolve only exercised on {convolved} datasets"
    );
}

#[test]
fn densify_and_resample_round_trip_on_real_events() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    let mut checked = 0;
    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Some(layout) = open(ds) else { continue };
        let Some(mut coll) = first_run_collection(&layout) else {
            continue;
        };
        let duration = coll.run_duration();
        if duration <= 0.0 {
            continue;
        }
        let Some(target) = first_sparse_numeric(&coll) else {
            continue;
        };

        let errs = apply_transformations(
            &mut coll,
            &spec(vec![
                serde_json::json!({"Name": "ToDense", "Input": [target], "SamplingRate": 10}),
                serde_json::json!({"Name": "Resample", "Input": [target], "SamplingRate": 5}),
            ]),
        );
        assert!(errs.is_empty(), "{name}: {errs:?}");

        let n = coll.n_rows(&target);
        let expected = (duration * 5.0).ceil() as usize;
        // Resampling rounds, so allow a sample either way.
        assert!(
            n.abs_diff(expected) <= 2,
            "{name}: '{target}' has {n} samples at 5 Hz, expected ~{expected} for {duration}s"
        );
        checked += 1;
        if checked >= 20 {
            break;
        }
    }

    eprintln!("ToDense + Resample verified on {checked} datasets");
    assert!(checked > 5, "only {checked} datasets exercised");
}

#[test]
fn filter_selects_rows_on_real_events() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    let mut checked = 0;
    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Some(layout) = open(ds) else { continue };
        let Some(mut coll) = first_run_collection(&layout) else {
            continue;
        };
        let Some(var) = coll.variables.get("trial_type") else {
            continue;
        };
        let Some(level) = var.str_values.iter().find(|s| !s.is_empty()).cloned() else {
            continue;
        };
        let expected = var.str_values.iter().filter(|s| **s == level).count();
        let before = coll.n_rows("trial_type");

        let errs = apply_transformations(
            &mut coll,
            &spec(vec![serde_json::json!({
                "Name": "Filter", "Input": ["trial_type"],
                "Query": format!("trial_type == '{level}'")
            })]),
        );
        assert!(errs.is_empty(), "{name}: {errs:?}");
        assert_eq!(
            coll.n_rows("trial_type"),
            expected,
            "{name}: filtering to '{level}' kept the wrong number of rows (from {before})"
        );
        // Timing must be trimmed in step with the values.
        if let Some(t) = coll.timing.get("trial_type") {
            assert_eq!(t.onset.len(), expected, "{name}: onsets not trimmed");
        }
        checked += 1;
        if checked >= 20 {
            break;
        }
    }

    eprintln!("Filter verified on {checked} datasets");
    assert!(checked > 5, "only {checked} datasets exercised");
}

// ─── Design matrices, diagnostics and reports ──────────────────────────────

#[test]
fn design_matrices_and_vif_from_real_events() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    let mut built = 0;
    for ds in &datasets {
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Some(layout) = open(ds) else { continue };
        let Some(mut coll) = first_run_collection(&layout) else {
            continue;
        };
        if !coll.variables.contains_key("trial_type") || coll.run_duration() <= 0.0 {
            continue;
        }

        // A realistic run-level pipeline: dummy-code conditions, convolve each.
        let errs = apply_transformations(
            &mut coll,
            &spec(vec![
                serde_json::json!({"Name": "Factor", "Input": ["trial_type"]}),
            ]),
        );
        assert!(errs.is_empty(), "{name}: {errs:?}");

        let levels: Vec<String> = sorted_names(&coll)
            .into_iter()
            .filter(|k| k.starts_with("trial_type."))
            .collect();
        if levels.len() < 2 {
            continue;
        }
        let errs = apply_transformations(
            &mut coll,
            &spec(vec![serde_json::json!({
                "Name": "Convolve", "Input": levels, "Model": "spm"
            })]),
        );
        assert!(errs.is_empty(), "{name} Convolve: {errs:?}");

        let n_obs = coll.n_rows(&levels[0]);
        if n_obs < 10 {
            continue;
        }
        let mut names = vec!["intercept".to_string()];
        let mut columns = vec![vec![1.0; n_obs]];
        for lev in &levels {
            if coll.n_rows(lev) == n_obs {
                names.push(lev.clone());
                columns.push(coll.variables[lev].values.clone());
            }
        }
        if columns.len() < 3 {
            continue;
        }

        let vifs = est_vif(&names, &columns);
        assert_eq!(
            vifs.len(),
            columns.len() - 1,
            "{name}: intercept not skipped"
        );
        for (regressor, v) in &vifs {
            assert!(
                v.is_nan() || *v >= 1.0 - 1e-6,
                "{name}: VIF for '{regressor}' is {v}, below the 1.0 floor"
            );
        }
        built += 1;
        if built >= 15 {
            break;
        }
    }

    eprintln!("Design matrices + VIF built for {built} datasets");
    assert!(built > 3, "only {built} datasets produced a design matrix");
}

#[test]
fn html_report_renders_for_a_real_dataset() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    for ds in &datasets {
        let Some(layout) = open(ds) else { continue };
        let Some(mut coll) = first_run_collection(&layout) else {
            continue;
        };
        if !coll.variables.contains_key("trial_type") || coll.run_duration() <= 0.0 {
            continue;
        }
        let errs = apply_transformations(
            &mut coll,
            &spec(vec![
                serde_json::json!({"Name": "Factor", "Input": ["trial_type"]}),
            ]),
        );
        if !errs.is_empty() {
            continue;
        }
        let levels: Vec<String> = sorted_names(&coll)
            .into_iter()
            .filter(|k| k.starts_with("trial_type."))
            .collect();
        if levels.len() < 2 {
            continue;
        }
        let errs = apply_transformations(
            &mut coll,
            &spec(vec![serde_json::json!({
                "Name": "Convolve", "Input": levels.clone(), "Model": "spm"
            })]),
        );
        if !errs.is_empty() {
            continue;
        }

        let n_obs = coll.n_rows(&levels[0]);
        let mut names = vec!["intercept".to_string()];
        let mut columns = vec![vec![1.0; n_obs]];
        for lev in &levels {
            if coll.n_rows(lev) == n_obs {
                names.push(lev.clone());
                columns.push(coll.variables[lev].values.clone());
            }
        }
        if columns.len() < 3 || n_obs < 10 {
            continue;
        }

        let contrasts = vec![bids_modeling::ContrastInfo {
            name: format!("{}_gt_{}", names[1], names[2]),
            conditions: vec![names[1].clone(), names[2].clone()],
            weights: vec![1.0, -1.0],
            test: Some("t".into()),
            entities: coll.entities.clone(),
        }];
        let output = bids_modeling::StatsModelsNodeOutput {
            node_name: "run".into(),
            entities: coll.entities.clone(),
            x_variables: names.clone(),
            contrasts,
            design_matrix: Some((names, columns)),
            transform_errors: Vec::new(),
        };

        let report = node_report(&output);
        assert!(report.n_obs > 0);
        assert!(!report.vif.is_empty(), "no regressor VIFs");
        let con_vif = get_all_contrast_vif(&output);
        assert_eq!(con_vif.len(), 1, "contrast VIF not computed");
        assert!(
            con_vif[0].1.is_finite() && con_vif[0].1 >= 1.0 - 1e-6,
            "contrast VIF is {}",
            con_vif[0].1
        );

        let html = generate_report(&[output], "example report");
        assert!(html.starts_with("<!DOCTYPE html>"));
        assert!(html.contains("<svg"), "report has no figures");
        assert!(html.contains("<table>"), "report has no table view");
        eprintln!(
            "Report rendered from {} ({} bytes)",
            ds.file_name().unwrap().to_string_lossy(),
            html.len()
        );
        return;
    }
    panic!("no example dataset produced a usable design matrix");
}

// ─── Mixed models from real participant structure ──────────────────────────

#[test]
fn variance_components_build_from_real_subject_grouping() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    for ds in &datasets {
        let Some(layout) = open(ds) else { continue };
        let Ok(subjects) = layout.get_subjects() else {
            continue;
        };
        if subjects.len() < 3 {
            continue;
        }

        // One observation per subject, grouped by subject: the canonical
        // group-level random intercept.
        let factor: Vec<String> = subjects.clone();
        let vc = VarComp::from_levels("subject", &factor);
        assert_eq!(vc.n_levels(), subjects.len());
        assert_eq!(vc.n_obs(), subjects.len());
        assert!(
            vc.index_vec.iter().all(|i| *i >= 1),
            "every observation should map to a level"
        );

        let mut glmm = GlmSpec::new(vec![Term::new("intercept", vec![1.0; subjects.len()])]);
        glmm.add_variance_component(vc).unwrap();
        glmm.set_priors(Some(Prior::new("Normal")), Some(Prior::new("HalfCauchy")));

        assert!(glmm.is_mixed());
        assert_eq!(glmm.n_random(), subjects.len());
        let (z_names, z_cols) = glmm.z().expect("Z should be present");
        assert_eq!(z_names.len(), subjects.len());
        // Each subject's indicator column has exactly one 1.
        for col in &z_cols {
            assert_eq!(col.iter().filter(|v| **v == 1.0).count(), 1);
        }

        eprintln!(
            "GLMM built from {} ({} subjects)",
            ds.file_name().unwrap().to_string_lossy(),
            subjects.len()
        );
        return;
    }
    panic!("no example dataset had enough subjects");
}

// ─── Non-run levels ────────────────────────────────────────────────────────

#[test]
fn dataset_level_collections_load_participants() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    let mut found = 0;
    for ds in &datasets {
        if !ds.join("participants.tsv").exists() {
            continue;
        }
        let name = ds.file_name().unwrap().to_string_lossy().to_string();
        let Some(layout) = open(ds) else { continue };
        let colls = layout
            .get_collections("dataset", None, None, false)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let n: usize = colls.iter().map(|c| c.variables.len()).sum();
        if n > 0 {
            for c in &colls {
                assert_eq!(c.level, "dataset", "{name}: wrong level");
            }
            found += 1;
        }
    }

    eprintln!("Dataset-level collections loaded for {found} datasets");
    assert!(
        found > 5,
        "only {found} datasets yielded participant variables"
    );
}

#[test]
fn load_variables_matches_the_extension_trait() {
    let examples = require_examples!();
    let datasets = datasets_with_events(examples.root());

    for ds in &datasets {
        let Some(layout) = open(ds) else { continue };
        let Ok(index) = load_variables(&layout, None, Some("run")) else {
            continue;
        };
        let direct = index.get_collections("run", &Default::default(), false);
        let Ok(via_trait) = layout.get_collections("run", None, None, false) else {
            continue;
        };
        if direct.is_empty() {
            continue;
        }
        assert_eq!(
            direct.len(),
            via_trait.len(),
            "{}: NodeIndex and LayoutVariables disagree",
            ds.display()
        );
        return;
    }
    panic!("no dataset produced run collections");
}
