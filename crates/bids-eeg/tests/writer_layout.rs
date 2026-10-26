//! A dataset written with `bids_eeg::writer` indexes through `BidsLayout`
//! and reads back through `EegLayout` with the same samples and sidecars.

use bids_core::DatasetDescription;
use bids_eeg::writer::{
    BrainVisionWriter, PhysioWriter, write_channels_tsv, write_eeg_sidecar, write_events_tsv,
};
use bids_eeg::{Channel, ChannelType, EegEvent, EegLayout, EegMetadata, ReadOptions};
use bids_layout::BidsLayout;

fn ch(name: &str) -> Channel {
    Channel {
        name: name.into(),
        channel_type: ChannelType::EEG,
        units: "µV".into(),
        description: None,
        sampling_frequency: None,
        low_cutoff: None,
        high_cutoff: None,
        notch: None,
        reference: None,
        status: Some("good".into()),
        status_description: None,
    }
}

#[test]
fn written_dataset_reads_back_through_layout() {
    let root = std::env::temp_dir().join(format!("bids_eeg_writer_layout_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();

    let desc: DatasetDescription =
        serde_json::from_str(r#"{"Name":"writer test","BIDSVersion":"1.10.0"}"#).unwrap();
    desc.save_to(&root).unwrap();
    std::fs::write(root.join("README"), "test\n").unwrap();

    let eeg_dir = root.join("sub-01/ses-20261026/eeg");
    let stem = "sub-01_ses-20261026_task-rest_run-1";
    let p = |suffix: &str| eeg_dir.join(format!("{stem}_{suffix}"));
    let channels = [ch("TP9"), ch("AF7"), ch("AF8"), ch("TP10")];

    let mut w = BrainVisionWriter::create(&p("eeg.vhdr"), &channels, 256.0).unwrap();
    for i in 0..1024 {
        let x = (i as f32 * 0.1).sin() * 50.0;
        w.write_frame(&[x, x + 1.0, x + 2.0, x + 3.0]).unwrap();
    }
    w.add_marker(2.0, 1.0, "focus");
    w.finish().unwrap();

    let md: EegMetadata = serde_json::from_str(
        r#"{"TaskName":"rest","SamplingFrequency":256,"EEGReference":"Fpz","EEGChannelCount":4}"#,
    )
    .unwrap();
    write_eeg_sidecar(&p("eeg.json"), &md, None).unwrap();
    write_channels_tsv(&p("channels.tsv"), &channels).unwrap();
    write_events_tsv(
        &p("events.tsv"),
        &[EegEvent {
            onset: 2.0,
            duration: 1.0,
            trial_type: Some("focus".into()),
            value: None,
            sample: Some(512),
            response_time: None,
            extra: Default::default(),
        }],
    )
    .unwrap();
    let mut physio = PhysioWriter::create(&p("physio.tsv.gz"), &["red", "infrared"], 64.0).unwrap();
    for i in 0..256 {
        physio
            .write_row(&[Some(i as f64), Some(-(i as f64))])
            .unwrap();
    }
    physio.finish(0.0, None).unwrap();

    let validation = bids_validate::validate_dataset(&root).unwrap();
    assert_eq!(validation.error_count(), 0, "{:?}", validation.issues);

    let layout = BidsLayout::new(&root).unwrap();
    let eeg = EegLayout::new(&layout);
    let files = eeg.get_eeg_files().unwrap();
    let vhdr = files
        .iter()
        .find(|f| f.filename.ends_with(".vhdr"))
        .expect("vhdr indexed");

    let data = eeg.read_data(vhdr, &ReadOptions::default()).unwrap();
    assert_eq!(data.channel_labels, ["TP9", "AF7", "AF8", "TP10"]);
    assert_eq!(data.data[0].len(), 1024);
    assert!((data.data[3][100] - data.data[0][100] - 3.0).abs() < 1e-4);

    let md_back = eeg.get_eeg_metadata(vhdr).unwrap().expect("eeg.json");
    assert_eq!(md_back.sampling_frequency, 256.0);
    assert_eq!(md_back.eeg_reference.as_deref(), Some("Fpz"));

    let chans = eeg.get_channels(vhdr).unwrap().expect("channels.tsv");
    assert_eq!(chans.len(), 4);
    let events = eeg.get_events(vhdr).unwrap().expect("events.tsv");
    assert_eq!(events[0].trial_type.as_deref(), Some("focus"));

    let physio = eeg.get_physio(vhdr).unwrap().expect("physio");
    assert_eq!(
        physio.rows.len(),
        256,
        "first row must not be eaten as a header"
    );
    assert_eq!(physio.rows[0]["red"], "0");
    assert_eq!(physio.rows[255]["infrared"], "-255");
    assert_eq!(physio.sampling_frequency, 64.0);

    std::fs::remove_dir_all(&root).unwrap();
}
