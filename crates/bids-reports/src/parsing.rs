//! Methods section text generation for individual BIDS datatypes.
//!
//! Each function generates a prose paragraph describing the acquisition
//! parameters for a specific modality (func, anat, dwi, fmap, eeg),
//! using metadata from JSON sidecars.

use crate::parameters;
use bids_core::file::BidsFile;
use bids_core::metadata::BidsMetadata;
use std::collections::HashMap;

/// Generate a description paragraph for functional (BOLD) scans.
pub fn func_info(files: &[BidsFile], metadata: &BidsMetadata) -> String {
    let n_runs = files.len();
    let task = metadata.get_str("TaskName").unwrap_or("unknown");
    let tr = parameters::describe_repetition_time(metadata);
    let te = parameters::describe_echo_times(metadata);
    let fa = parameters::describe_flip_angle(metadata);
    let slice_order = parameters::describe_slice_timing(metadata);
    let mb = parameters::describe_multiband_factor(metadata);

    let run_str = if n_runs == 1 {
        format!("{} run", parameters::num_to_words(n_runs))
    } else {
        format!("{} runs", parameters::num_to_words(n_runs))
    };

    let mut desc = format!(
        "{} of {} task were collected (TR={}, TE={}, flip angle={}, {} slice acquisition",
        capitalize(&run_str),
        task,
        tr,
        te,
        fa,
        slice_order,
    );
    if mb != "n/a" {
        desc.push_str(&format!(", MB={mb}"));
    }
    desc.push_str(").");
    desc
}

/// Generate a description for anatomical scans.
pub fn anat_info(files: &[BidsFile], metadata: &BidsMetadata) -> String {
    let suffix = files
        .first()
        .and_then(|f| f.suffix())
        .unwrap_or("structural");
    let tr = parameters::describe_repetition_time(metadata);
    let te = parameters::describe_echo_times(metadata);
    let fa = parameters::describe_flip_angle(metadata);
    format!("A {suffix} image was collected (TR={tr}, TE={te}, flip angle={fa}).")
}

/// Generate a description for diffusion scans.
pub fn dwi_info(_files: &[BidsFile], metadata: &BidsMetadata) -> String {
    let n_dirs = parameters::describe_dmri_directions(metadata);
    let tr = parameters::describe_repetition_time(metadata);
    let te = parameters::describe_echo_times(metadata);
    let bvals = parameters::describe_bvals(metadata);

    let mut desc = format!("Diffusion-weighted images were collected (TR={tr}, TE={te}");
    if n_dirs != "unknown" {
        desc.push_str(&format!(", {n_dirs} directions"));
    }
    if bvals != "unknown" {
        desc.push_str(&format!(", b-values: {bvals}"));
    }
    desc.push_str(").");
    desc
}

/// Generate a description for fieldmap scans.
pub fn fmap_info(files: &[BidsFile], metadata: &BidsMetadata) -> String {
    let n = files.len();
    let suffix = files.first().and_then(|f| f.suffix()).unwrap_or("fieldmap");
    let te = parameters::describe_echo_times_fmap(metadata);
    let targets = parameters::describe_intendedfor_targets(metadata);

    let mut desc = format!("{n} {suffix} fieldmap image(s) were collected");
    if te != "unknown" {
        desc.push_str(&format!(" (TE={te})"));
    }
    if targets != "none" {
        desc.push_str(&format!(" for distortion correction of {targets}"));
    }
    desc.push('.');
    desc
}

/// Group files by acquisition parameters (suffix + shared metadata).
pub fn parse_files(files: &[BidsFile]) -> Vec<(String, Vec<&BidsFile>)> {
    let mut by_suffix: HashMap<String, Vec<&BidsFile>> = HashMap::new();
    for f in files {
        let suffix = f.suffix().unwrap_or("unknown").to_string();
        by_suffix.entry(suffix).or_default().push(f);
    }
    let mut result: Vec<_> = by_suffix.into_iter().collect();
    result.sort_by_key(|(s, _)| s.clone());
    result
}

/// Generate a per-subject report section.
pub fn report_subject(subject: &str, files: &[BidsFile]) -> String {
    let n_files = files.len();
    let datatypes: Vec<String> = {
        let mut set = std::collections::BTreeSet::new();
        for f in files {
            if let Some(dt) = f.entities.get("datatype") {
                set.insert(dt.as_str_lossy().into_owned());
            }
        }
        set.into_iter().collect()
    };
    format!(
        "Subject {}: {} files across {} datatype(s) ({}).",
        subject,
        n_files,
        datatypes.len(),
        parameters::list_to_str(&datatypes)
    )
}

/// Generate a full methods section for all datatypes found.
pub fn generate_methods(datatypes: &[(String, Vec<BidsFile>, BidsMetadata)]) -> String {
    let mut sections = Vec::new();
    for (datatype, files, md) in datatypes {
        if files.is_empty() {
            continue;
        }
        let section = match datatype.as_str() {
            "func" => func_info(files, md),
            "anat" => anat_info(files, md),
            "dwi" => dwi_info(files, md),
            "fmap" => fmap_info(files, md),
            _ => format!("{} data were collected ({} files).", datatype, files.len()),
        };
        sections.push(section);
    }
    if !sections.is_empty() {
        sections.push(parameters::final_paragraph().to_string());
    }
    sections.join("\n\n")
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}
