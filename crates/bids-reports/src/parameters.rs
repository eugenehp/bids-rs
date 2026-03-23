//! Human-readable parameter descriptions for BIDS metadata values.
//!
//! Converts raw metadata values (TR, TE, flip angle, slice timing, phase
//! encoding direction, etc.) into formatted strings suitable for inclusion
//! in a methods section.

use bids_core::metadata::BidsMetadata;

pub fn describe_sequence(md: &BidsMetadata) -> (String, String) {
    let seq = md.get_str("PulseSequenceType").unwrap_or("unknown").to_string();
    let variant = md.get_str("SequenceVariant").unwrap_or("").to_string();
    (seq, variant)
}

pub fn describe_repetition_time(md: &BidsMetadata) -> String {
    md.get_f64("RepetitionTime").map_or("unknown".into(), |tr| format!("{tr:.2} s"))
}

pub fn describe_echo_times(md: &BidsMetadata) -> String {
    if let Some(te) = md.get_f64("EchoTime") {
        format!("{:.1} ms", te * 1000.0)
    } else if let Some(tes) = md.get_array("EchoTime") {
        let vals: Vec<String> = tes.iter().filter_map(serde_json::Value::as_f64).map(|te| format!("{:.1}", te * 1000.0)).collect();
        format!("{} ms", vals.join("/"))
    } else { "unknown".into() }
}

pub fn describe_echo_times_fmap(md: &BidsMetadata) -> String {
    let te1 = md.get_f64("EchoTime1").map(|t| format!("{:.2}", t * 1000.0));
    let te2 = md.get_f64("EchoTime2").map(|t| format!("{:.2}", t * 1000.0));
    match (te1, te2) {
        (Some(a), Some(b)) => format!("{a}/{b} ms"),
        (Some(a), None) => format!("{a} ms"),
        _ => "unknown".into(),
    }
}

pub fn describe_flip_angle(md: &BidsMetadata) -> String {
    md.get_f64("FlipAngle").map_or("unknown".into(), |fa| format!("{fa}°"))
}

pub fn describe_slice_timing(md: &BidsMetadata) -> String {
    if let Some(st) = md.get_array("SliceTiming") {
        if st.len() < 2 { return "single slice".into(); }
        let times: Vec<f64> = st.iter().filter_map(serde_json::Value::as_f64).collect();
        if times.len() < 2 { return "unknown".into(); }
        let diffs: Vec<f64> = times.windows(2).map(|w| w[1] - w[0]).collect();
        if diffs.iter().all(|d| d.abs() < 0.001) { "sequential ascending".into() }
        else if diffs.iter().all(|d| *d < -0.001) { "sequential descending".into() }
        else { "interleaved".into() }
    } else { "unknown".into() }
}

pub fn describe_duration(n_vols: Option<usize>, tr: Option<f64>) -> String {
    match (n_vols, tr) {
        (Some(n), Some(t)) => {
            let dur = t * n as f64;
            let mins = (dur / 60.0).floor() as u32;
            let secs = dur % 60.0;
            if mins > 0 { format!("{mins}:{secs:05.2}") }
            else { format!("{dur:.1} s") }
        }
        _ => "unknown".into(),
    }
}

pub fn describe_multiband_factor(md: &BidsMetadata) -> String {
    md.get_f64("MultibandAccelerationFactor")
        .map_or("n/a".into(), |f| format!("{}", f as i64))
}

pub fn describe_image_size(md: &BidsMetadata) -> String {
    // From metadata only (no image header)
    if let Some(acq_matrix) = md.get_array("AcquisitionMatrixPE") {
        let dims: Vec<String> = acq_matrix.iter().filter_map(serde_json::Value::as_i64).map(|d| d.to_string()).collect();
        return dims.join("×").to_string();
    }
    "unknown".into()
}

pub fn describe_inplane_accel(md: &BidsMetadata) -> String {
    md.get_f64("ParallelReductionFactorInPlane")
        .map_or("none".into(), |f| format!("{f}"))
}

pub fn describe_dmri_directions(md: &BidsMetadata) -> String {
    md.get_i64("NumberOfDirections")
        .map_or("unknown".into(), |d| format!("{d}"))
}

pub fn describe_bvals(md: &BidsMetadata) -> String {
    if let Some(bvals) = md.get_array("bValues") {
        let vals: Vec<String> = bvals.iter().filter_map(serde_json::Value::as_f64).map(|b| format!("{b:.0}")).collect();
        vals.join(", ")
    } else { "unknown".into() }
}

pub fn describe_pe_direction(md: &BidsMetadata) -> String {
    md.get_str("PhaseEncodingDirection").map_or("unknown".into(), |d| {
        match d { "i" => "left-right", "i-" => "right-left",
                  "j" => "posterior-anterior", "j-" => "anterior-posterior",
                  "k" => "inferior-superior", "k-" => "superior-inferior",
                  _ => d }.into()
    })
}

pub fn describe_intendedfor_targets(md: &BidsMetadata) -> String {
    if let Some(intended) = md.get("IntendedFor") {
        match intended {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(a) => {
                let targets: Vec<String> = a.iter().filter_map(|v| v.as_str().map(String::from)).collect();
                targets.join(", ")
            }
            _ => "unknown".into(),
        }
    } else { "none".into() }
}

pub fn describe_acquisition(md: &BidsMetadata) -> Vec<(String, String)> {
    let mut info = Vec::new();
    if let Some(tr) = md.get_f64("RepetitionTime") { info.push(("TR".into(), format!("{tr:.3} s"))); }
    if let Some(te) = md.get_f64("EchoTime") { info.push(("TE".into(), format!("{:.1} ms", te * 1000.0))); }
    if let Some(fa) = md.get_f64("FlipAngle") { info.push(("Flip Angle".into(), format!("{fa}°"))); }
    if let Some(plf) = md.get_f64("ParallelReductionFactorInPlane") { info.push(("GRAPPA".into(), format!("{plf}"))); }
    if let Some(mb) = md.get_f64("MultibandAccelerationFactor") { info.push(("Multiband".into(), format!("{mb}"))); }
    if let Some(_pe) = md.get_str("PhaseEncodingDirection") { info.push(("PE Direction".into(), describe_pe_direction(md))); }
    info
}

pub fn num_to_words(n: usize) -> String {
    match n {
        0 => "zero", 1 => "one", 2 => "two", 3 => "three", 4 => "four",
        5 => "five", 6 => "six", 7 => "seven", 8 => "eight", 9 => "nine",
        10 => "ten", 11 => "eleven", 12 => "twelve", 13 => "thirteen",
        14 => "fourteen", 15 => "fifteen", 16 => "sixteen", 17 => "seventeen",
        18 => "eighteen", 19 => "nineteen", 20 => "twenty",
        _ => return n.to_string(),
    }.into()
}

pub fn num_to_str(n: usize) -> String { num_to_words(n) }

pub fn final_paragraph() -> &'static str {
    "Note: This report was auto-generated by bids-rs. Please verify all details before publication."
}

pub fn list_to_str(items: &[String]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => {
            let last = items.last().unwrap();
            let rest = &items[..items.len() - 1];
            format!("{}, and {}", rest.join(", "), last)
        }
    }
}

pub fn remove_duplicates(items: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    items.iter().filter(|s| seen.insert((*s).clone())).cloned().collect()
}

pub fn get_size_str(dims: &[usize]) -> String {
    dims.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join("×")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_to_words() {
        assert_eq!(num_to_words(1), "one");
        assert_eq!(num_to_words(3), "three");
        assert_eq!(num_to_words(15), "fifteen");
        assert_eq!(num_to_words(25), "25");
    }

    #[test]
    fn test_list_to_str() {
        assert_eq!(list_to_str(&["a".into(), "b".into()]), "a and b");
        assert_eq!(list_to_str(&["a".into(), "b".into(), "c".into()]), "a, b, and c");
    }
}
