//! SNIRF reading against a file libhdf5 wrote (`data/make_minimal_snirf.py`).
#![cfg(feature = "snirf")]

use std::path::Path;

use bids_core::error::BidsError;
use bids_nirs::read_snirf;

fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/minimal.snirf")
}

#[test]
fn reads_a_libhdf5_snirf_file() {
    let d = read_snirf(&fixture()).unwrap();
    assert_eq!((d.n_channels, d.n_samples), (4, 100));
    // channels × samples, transposed from the file's samples × channels
    for (c, channel) in d.data.iter().enumerate() {
        for (s, &v) in channel.iter().enumerate() {
            assert_eq!(v, (s * 4 + c) as f64 / 7.0, "sample {s} channel {c}");
        }
    }
    for (s, &t) in d.time.iter().enumerate() {
        assert_eq!(t, s as f64 / 10.0);
    }
    assert_eq!(
        d.measurement_list,
        [(1, 11, 2.0), (2, 12, 1.0), (3, 13, 2.0), (4, 14, 1.0)]
    );
    assert!((d.sfreq() - 10.0).abs() < 1e-9);
}

#[test]
fn a_file_that_is_not_hdf5_is_an_error() {
    let path =
        std::env::temp_dir().join(format!("bids_nirs_not_hdf5_{}.snirf", std::process::id()));
    std::fs::write(&path, b"not an HDF5 file").unwrap();
    let err = read_snirf(&path).unwrap_err();
    std::fs::remove_file(&path).unwrap();
    assert!(matches!(err, BidsError::Io(_)), "{err:?}");
}
