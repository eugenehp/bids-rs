//! Render a sample model report to `/tmp/bidsrs-report/report.html`.
use bids_core::entities::StringEntities;
use bids_modeling::node::{ContrastInfo, StatsModelsNodeOutput};
use bids_modeling::report::generate_report;

fn con(name: &str, conditions: &[&str], weights: &[f64]) -> ContrastInfo {
    ContrastInfo {
        name: name.into(),
        conditions: conditions.iter().map(|s| (*s).to_string()).collect(),
        weights: weights.to_vec(),
        test: Some("t".into()),
        entities: StringEntities::new(),
    }
}

fn main() {
    let n = 240usize;
    let tr = 2.0;
    let frames: Vec<f64> = (0..n).map(|i| i as f64 * tr).collect();

    let mk = |onsets: &[f64]| -> Vec<f64> {
        let durations = vec![2.0; onsets.len()];
        let amps = vec![1.0; onsets.len()];
        bids_modeling::hrf::compute_regressor(
            onsets,
            &durations,
            &amps,
            &bids_modeling::HrfModel::Spm,
            &frames,
            10,
            None,
        )
        .remove(0)
    };
    let face: Vec<f64> = mk(&(0..12).map(|k| f64::from(k) * 40.0).collect::<Vec<_>>());
    let house: Vec<f64> = mk(&(0..12)
        .map(|k| f64::from(k) * 40.0 + 18.0)
        .collect::<Vec<_>>());
    let drift: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.5).collect();
    // Deliberately collinear with `face`, to exercise the VIF path.
    let motion: Vec<f64> = face
        .iter()
        .enumerate()
        .map(|(i, v)| v * 0.9 + f64::from(i as i32 % 5) * 0.01)
        .collect();

    let names = vec![
        "intercept".to_string(),
        "trial_type.face".into(),
        "trial_type.house".into(),
        "drift".into(),
        "motion_x".into(),
    ];
    let columns = vec![vec![1.0; n], face, house, drift, motion];

    let output = StatsModelsNodeOutput {
        node_name: "run_level".into(),
        entities: StringEntities::from([
            ("subject".to_string(), "01".to_string()),
            ("task".to_string(), "faces".to_string()),
            ("run".to_string(), "1".to_string()),
        ]),
        x_variables: names.clone(),
        contrasts: vec![
            con(
                "face_gt_house",
                &["trial_type.face", "trial_type.house"],
                &[1.0, -1.0],
            ),
            con("face", &["trial_type.face"], &[1.0]),
            con("house", &["trial_type.house"], &[1.0]),
        ],
        design_matrix: Some((names, columns)),
        transform_errors: Vec::new(),
    };

    let html = generate_report(&[output], "faces task · model report");
    std::fs::create_dir_all("/tmp/bidsrs-report").unwrap();
    std::fs::write("/tmp/bidsrs-report/report.html", &html).unwrap();
    println!("wrote {} bytes", html.len());
}
