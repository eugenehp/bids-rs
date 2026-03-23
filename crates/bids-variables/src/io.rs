//! Variable loading from BIDS datasets.
//!
//! Reads variables from all standard BIDS tabular files (participants.tsv,
//! sessions.tsv, scans.tsv, events.tsv, physio/stim .tsv.gz, regressors.tsv)
//! and organizes them into a [`NodeIndex`] hierarchy.
//!
//! The [`load_variables()`] function is the main entry point, accepting a
//! `BidsLayout`, optional variable type filters, and an optional level filter.

use bids_core::entities::StringEntities;
use bids_core::error::Result;
use bids_io::tsv::read_tsv;
use bids_layout::BidsLayout;

use crate::node::{NodeIndex, RunInfo};
use crate::variables::{SimpleVariable, SparseRunVariable};

/// The types of variables that can be loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableType {
    Events,
    Physio,
    Stim,
    Scans,
    Participants,
    Sessions,
    Regressors,
}

/// Load variables from a BIDS dataset.
pub fn load_variables(
    layout: &BidsLayout,
    types: Option<&[VariableType]>,
    level: Option<&str>,
) -> Result<NodeIndex> {
    let types = resolve_types(types, level);
    let mut index = NodeIndex::new();
    for vtype in &types {
        match vtype {
            VariableType::Participants => load_participants(layout, &mut index)?,
            VariableType::Sessions => load_sessions(layout, &mut index)?,
            VariableType::Scans => load_scans(layout, &mut index)?,
            VariableType::Events => load_events(layout, &mut index)?,
            VariableType::Physio | VariableType::Stim | VariableType::Regressors => {
                load_time_variables(layout, &mut index, None)?;
            }
        }
    }
    Ok(index)
}

fn resolve_types(types: Option<&[VariableType]>, level: Option<&str>) -> Vec<VariableType> {
    if let Some(t) = types {
        return t.to_vec();
    }
    match level {
        Some("run") => vec![
            VariableType::Events,
            VariableType::Physio,
            VariableType::Stim,
            VariableType::Regressors,
        ],
        Some("session") => vec![VariableType::Scans],
        Some("subject") => vec![VariableType::Sessions, VariableType::Scans],
        Some("dataset") => vec![VariableType::Participants],
        _ => vec![
            VariableType::Events,
            VariableType::Physio,
            VariableType::Stim,
            VariableType::Regressors,
            VariableType::Scans,
            VariableType::Sessions,
            VariableType::Participants,
        ],
    }
}

static SUB_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"sub-([a-zA-Z0-9]+)").unwrap());
static SES_RE: std::sync::LazyLock<regex::Regex> =
    std::sync::LazyLock::new(|| regex::Regex::new(r"ses-([a-zA-Z0-9]+)").unwrap());

fn load_participants(layout: &BidsLayout, index: &mut NodeIndex) -> Result<()> {
    let tsv_path = layout.root().join("participants.tsv");
    if !tsv_path.exists() {
        return Ok(());
    }
    let rows = read_tsv(&tsv_path)?;
    if rows.is_empty() {
        return Ok(());
    }

    let node_idx = index.create_node("dataset", StringEntities::new());
    let columns: Vec<String> = rows[0]
        .keys()
        .filter(|k| k.as_str() != "participant_id")
        .cloned()
        .collect();

    for col_name in &columns {
        let mut values = Vec::new();
        let mut row_index = Vec::new();
        for row in &rows {
            values.push(row.get(col_name).cloned().unwrap_or_default());
            let mut ent = StringEntities::new();
            if let Some(pid) = row.get("participant_id") {
                ent.insert(
                    "subject".into(),
                    pid.strip_prefix("sub-").unwrap_or(pid).into(),
                );
            }
            row_index.push(ent);
        }
        let var = SimpleVariable::new(col_name, "participants", values, row_index);
        if let Some(node) = index.get_node_mut(node_idx) {
            node.add_variable(var);
        }
    }
    Ok(())
}

fn load_sessions(layout: &BidsLayout, index: &mut NodeIndex) -> Result<()> {
    let session_files = layout
        .get()
        .suffix("sessions")
        .extension("tsv")
        .return_paths()?;
    for tsv_path in &session_files {
        let rows = read_tsv(tsv_path)?;
        if rows.is_empty() {
            continue;
        }

        let mut entities = StringEntities::new();
        let path_str = tsv_path.to_string_lossy();
        if let Some(caps) = SUB_RE.captures(&path_str) {
            entities.insert("subject".into(), caps[1].to_string());
        }
        let node_idx = index.create_node("subject", entities);
        let columns: Vec<String> = rows[0]
            .keys()
            .filter(|k| k.as_str() != "session_id")
            .cloned()
            .collect();

        for col_name in &columns {
            let mut values = Vec::new();
            let mut row_index = Vec::new();
            for row in &rows {
                values.push(row.get(col_name).cloned().unwrap_or_default());
                let mut ent = StringEntities::new();
                if let Some(sid) = row.get("session_id") {
                    ent.insert(
                        "session".into(),
                        sid.strip_prefix("ses-").unwrap_or(sid).into(),
                    );
                }
                row_index.push(ent);
            }
            let var = SimpleVariable::new(col_name, "sessions", values, row_index);
            if let Some(node) = index.get_node_mut(node_idx) {
                node.add_variable(var);
            }
        }
    }
    Ok(())
}

fn load_scans(layout: &BidsLayout, index: &mut NodeIndex) -> Result<()> {
    let scans_files = layout
        .get()
        .suffix("scans")
        .extension("tsv")
        .return_paths()?;
    for tsv_path in &scans_files {
        let rows = read_tsv(tsv_path)?;
        if rows.is_empty() {
            continue;
        }

        let mut entities = StringEntities::new();
        let path_str = tsv_path.to_string_lossy();
        if let Some(caps) = SUB_RE.captures(&path_str) {
            entities.insert("subject".into(), caps[1].to_string());
        }
        if let Some(caps) = SES_RE.captures(&path_str) {
            entities.insert("session".into(), caps[1].to_string());
        }
        let node_idx = index.create_node("session", entities);
        let columns: Vec<String> = rows[0]
            .keys()
            .filter(|k| k.as_str() != "filename")
            .cloned()
            .collect();

        for col_name in &columns {
            let mut values = Vec::new();
            let mut row_index = Vec::new();
            for row in &rows {
                values.push(row.get(col_name).cloned().unwrap_or_default());
                row_index.push(StringEntities::new());
            }
            let var = SimpleVariable::new(col_name, "scans", values, row_index);
            if let Some(node) = index.get_node_mut(node_idx) {
                node.add_variable(var);
            }
        }
    }
    Ok(())
}

/// Load time-series variables (physio/stim/regressors) with scan_length fallback.
fn load_time_variables(
    layout: &BidsLayout,
    index: &mut NodeIndex,
    scan_length: Option<f64>,
) -> Result<()> {
    // Look for physio/stim files
    for suffix in &["physio", "stim"] {
        let files = layout.get().suffix(suffix).extension("tsv.gz").collect()?;
        for f in &files {
            let mut entities = StringEntities::new();
            for (k, v) in &f.entities {
                entities.insert(k.clone(), v.as_str_lossy().into_owned());
            }
            let md = layout.get_metadata(&f.path)?;
            let sr = md.get_f64("SamplingFrequency").unwrap_or(1.0);
            let start_time = md.get_f64("StartTime").unwrap_or(0.0);
            let columns: Vec<String> = md
                .get_array("Columns")
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let duration = scan_length.unwrap_or(0.0);
            let node_idx = index.get_or_create_run_node(entities.clone(), None, duration, 0.0, 0);

            if let Ok(rows) = bids_io::tsv::read_tsv_gz(&f.path) {
                for (col_idx, col_name) in columns.iter().enumerate() {
                    let values: Vec<f64> = rows
                        .iter()
                        .filter_map(|row| row.values().nth(col_idx).and_then(|v| v.parse().ok()))
                        .collect();
                    if values.is_empty() {
                        continue;
                    }

                    // Trim/pad to match scan duration
                    let n_expected = (duration * sr).ceil() as usize;
                    let trimmed = if values.len() > n_expected && n_expected > 0 {
                        let skip = if start_time < 0.0 {
                            (-start_time * sr).floor() as usize
                        } else {
                            0
                        };
                        values[skip..].iter().take(n_expected).copied().collect()
                    } else {
                        values
                    };

                    let ri = index
                        .get_run_node_mut(node_idx)
                        .map(|rn| rn.get_info())
                        .unwrap_or(crate::node::RunInfo {
                            entities: entities.clone(),
                            duration,
                            tr: 0.0,
                            image: None,
                            n_vols: 0,
                        });
                    let var = crate::variables::DenseRunVariable::new(
                        col_name,
                        suffix,
                        trimmed,
                        sr,
                        vec![ri],
                    );
                    if let Some(rn) = index.get_run_node_mut(node_idx) {
                        rn.add_dense_variable(var);
                    }
                }
            }
        }
    }

    // Regressors/timeseries TSV files
    let reg_files = layout
        .get()
        .filter_any("suffix", &["regressors", "timeseries"])
        .extension("tsv")
        .collect()?;
    for f in &reg_files {
        let mut entities = StringEntities::new();
        for (k, v) in &f.entities {
            entities.insert(k.clone(), v.as_str_lossy().into_owned());
        }

        let node_idx = index.get_or_create_run_node(
            entities.clone(),
            None,
            scan_length.unwrap_or(0.0),
            0.0,
            0,
        );

        if let Ok(rows) = bids_io::tsv::read_tsv(&f.path) {
            if rows.is_empty() {
                continue;
            }
            let columns: Vec<String> = rows[0].keys().cloned().collect();
            let tr = index
                .get_run_node_mut(node_idx)
                .map(|rn| rn.repetition_time)
                .unwrap_or(1.0);
            let sr = if tr > 0.0 { 1.0 / tr } else { 1.0 };

            for col_name in &columns {
                let values: Vec<f64> = rows
                    .iter()
                    .filter_map(|row| row.get(col_name).and_then(|v| v.parse().ok()))
                    .collect();
                if values.is_empty() {
                    continue;
                }

                let ri = index
                    .get_run_node_mut(node_idx)
                    .map(|rn| rn.get_info())
                    .unwrap_or(crate::node::RunInfo {
                        entities: entities.clone(),
                        duration: 0.0,
                        tr,
                        image: None,
                        n_vols: 0,
                    });
                let var = crate::variables::DenseRunVariable::new(
                    col_name,
                    "regressors",
                    values,
                    sr,
                    vec![ri],
                );
                if let Some(rn) = index.get_run_node_mut(node_idx) {
                    rn.add_dense_variable(var);
                }
            }
        }
    }

    Ok(())
}

fn load_events(layout: &BidsLayout, index: &mut NodeIndex) -> Result<()> {
    let event_files = layout.get().suffix("events").extension("tsv").collect()?;
    for ef in &event_files {
        let rows = read_tsv(&ef.path)?;
        if rows.is_empty() {
            continue;
        }

        let mut entities = StringEntities::new();
        for (k, v) in &ef.entities {
            entities.insert(k.clone(), v.as_str_lossy().into_owned());
        }

        let node_idx = index.get_or_create_run_node(entities.clone(), None, 0.0, 0.0, 0);

        let run_info = index
            .get_run_node_mut(node_idx)
            .map(|rn| rn.get_info())
            .unwrap_or(RunInfo {
                entities: entities.clone(),
                duration: 0.0,
                tr: 0.0,
                image: None,
                n_vols: 0,
            });

        let columns: Vec<String> = rows[0]
            .keys()
            .filter(|k| k.as_str() != "onset" && k.as_str() != "duration")
            .cloned()
            .collect();

        for col_name in &columns {
            let mut onset = Vec::new();
            let mut duration = Vec::new();
            let mut amplitude = Vec::new();
            let mut row_index = Vec::new();

            for row in &rows {
                let o: f64 = row.get("onset").and_then(|v| v.parse().ok()).unwrap_or(0.0);
                let d: f64 = row
                    .get("duration")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.0);
                let a = row.get(col_name).cloned().unwrap_or_default();
                if a.is_empty() {
                    continue;
                }

                onset.push(o);
                duration.push(d);
                amplitude.push(a);
                row_index.push(entities.clone());
            }
            if onset.is_empty() {
                continue;
            }

            let var = SparseRunVariable::new(
                col_name,
                "events",
                onset,
                duration,
                amplitude,
                row_index,
                vec![run_info.clone()],
            );
            if let Some(rn) = index.get_run_node_mut(node_idx) {
                rn.add_sparse_variable(var);
            }
        }
    }
    Ok(())
}
