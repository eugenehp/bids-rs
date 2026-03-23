//! Performance regression tests for EEG data reading.
//!
//! These tests generate realistic EDF, BDF, and BrainVision files, read them,
//! and verify both correctness and performance against thresholds derived from
//! the Python MNE benchmark (tests/golden/eeg_read_benchmark.json).
//!
//! The thresholds are set conservatively (2× the measured Rust time) to avoid
//! flaky CI failures while still catching real regressions.
//!
//! Run with: cargo test -p bids-eeg --release --test bench_regression

use bids_eeg::{read_edf, read_brainvision, ReadOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// ─── Test file generators ──────────────────────────────────────────────────────

fn generate_edf(path: &Path, n_ch: usize, n_rec: usize, spr: usize) {
    let mut file = std::fs::File::create(path).unwrap();
    let mut hdr = [b' '; 256];
    hdr[0..1].copy_from_slice(b"0");
    hdr[168..176].copy_from_slice(b"01.01.01");
    hdr[176..184].copy_from_slice(b"00.00.00");
    let hs = format!("{:<8}", 256 + n_ch * 256);
    hdr[184..192].copy_from_slice(hs.as_bytes());
    let nr = format!("{:<8}", n_rec);
    hdr[236..244].copy_from_slice(nr.as_bytes());
    hdr[244..252].copy_from_slice(b"1       ");
    let nc = format!("{:<4}", n_ch);
    hdr[252..256].copy_from_slice(nc.as_bytes());
    file.write_all(&hdr).unwrap();

    let mut ext = vec![b' '; n_ch * 256];
    for i in 0..n_ch {
        let label = format!("{:<16}", format!("EEG{}", i + 1));
        ext[i * 16..i * 16 + 16].copy_from_slice(label.as_bytes());
        ext[n_ch * 96 + i * 8..n_ch * 96 + i * 8 + 2].copy_from_slice(b"uV");
        let vals = [("-3200", 104), ("3200", 112), ("-32768", 120), ("32767", 128)];
        for (val, base) in &vals {
            let s = format!("{:<8}", val);
            ext[n_ch * base + i * 8..n_ch * base + i * 8 + 8].copy_from_slice(s.as_bytes());
        }
        let s = format!("{:<8}", spr);
        ext[n_ch * 216 + i * 8..n_ch * 216 + i * 8 + 8].copy_from_slice(s.as_bytes());
    }
    file.write_all(&ext).unwrap();

    let rec_bytes = n_ch * spr * 2;
    let mut buf = vec![0u8; rec_bytes];
    for rec in 0..n_rec {
        for ch in 0..n_ch {
            for s in 0..spr {
                let t = rec as f64 + s as f64 / spr as f64;
                let value =
                    (1000.0 * (2.0 * std::f64::consts::PI * (ch as f64 + 1.0) * t).sin()) as i16;
                let off = (ch * spr + s) * 2;
                buf[off..off + 2].copy_from_slice(&value.to_le_bytes());
            }
        }
        file.write_all(&buf).unwrap();
    }
}

fn generate_bdf(path: &Path, n_ch: usize, n_rec: usize, spr: usize) {
    let mut file = std::fs::File::create(path).unwrap();
    let mut hdr = [b' '; 256];
    hdr[0..1].copy_from_slice(b"\xff");
    hdr[1..8].copy_from_slice(b"BIOSEMI");
    hdr[168..176].copy_from_slice(b"01.01.01");
    hdr[176..184].copy_from_slice(b"00.00.00");
    let hs = format!("{:<8}", 256 + n_ch * 256);
    hdr[184..192].copy_from_slice(hs.as_bytes());
    hdr[192..197].copy_from_slice(b"BDF+C");
    let nr = format!("{:<8}", n_rec);
    hdr[236..244].copy_from_slice(nr.as_bytes());
    hdr[244..252].copy_from_slice(b"1       ");
    let nc = format!("{:<4}", n_ch);
    hdr[252..256].copy_from_slice(nc.as_bytes());
    file.write_all(&hdr).unwrap();

    let mut ext = vec![b' '; n_ch * 256];
    for i in 0..n_ch {
        let label = format!("{:<16}", format!("EEG{}", i + 1));
        ext[i * 16..i * 16 + 16].copy_from_slice(label.as_bytes());
        ext[n_ch * 96 + i * 8..n_ch * 96 + i * 8 + 2].copy_from_slice(b"uV");
        let vals = [("-3200", 104), ("3200", 112), ("-8388608", 120), ("8388607", 128)];
        for (val, base) in &vals {
            let s = format!("{:<8}", val);
            ext[n_ch * base + i * 8..n_ch * base + i * 8 + 8].copy_from_slice(s.as_bytes());
        }
        let s = format!("{:<8}", spr);
        ext[n_ch * 216 + i * 8..n_ch * 216 + i * 8 + 8].copy_from_slice(s.as_bytes());
    }
    file.write_all(&ext).unwrap();

    let rec_bytes = n_ch * spr * 3;
    let mut buf = vec![0u8; rec_bytes];
    for rec in 0..n_rec {
        for ch in 0..n_ch {
            for s in 0..spr {
                let t = rec as f64 + s as f64 / spr as f64;
                let value = (100000.0
                    * (2.0 * std::f64::consts::PI * (ch as f64 + 1.0) * t).sin())
                    as i32;
                let value = value.clamp(-8388608, 8388607);
                let off = (ch * spr + s) * 3;
                buf[off] = (value & 0xFF) as u8;
                buf[off + 1] = ((value >> 8) & 0xFF) as u8;
                buf[off + 2] = ((value >> 16) & 0xFF) as u8;
            }
        }
        file.write_all(&buf).unwrap();
    }
}

fn generate_brainvision(dir: &Path, n_ch: usize, n_samples: usize, spr: usize) {
    let us = 1_000_000.0 / spr as f64;
    let mut vhdr = format!(
        "Brain Vision Data Exchange Header File Version 1.0\n\n\
         [Common Infos]\nDataFile=test.eeg\nMarkerFile=test.vmrk\n\
         DataFormat=BINARY\nDataOrientation=MULTIPLEXED\n\
         NumberOfChannels={}\nSamplingInterval={}\n\n\
         [Binary Infos]\nBinaryFormat=INT_16\n\n[Channel Infos]\n",
        n_ch, us
    );
    for i in 0..n_ch {
        vhdr.push_str(&format!("Ch{}=EEG{},,0.1\n", i + 1, i + 1));
    }
    std::fs::write(dir.join("test.vhdr"), &vhdr).unwrap();

    let vmrk = "Brain Vision Data Exchange Marker File Version 1.0\n\n\
        [Marker Infos]\nMk1=Stimulus,S1,512,1,0\nMk2=Stimulus,S2,2048,1,0\n";
    std::fs::write(dir.join("test.vmrk"), vmrk).unwrap();

    let total_bytes = n_samples * n_ch * 2;
    let mut buf = vec![0u8; total_bytes];
    for s in 0..n_samples {
        for ch in 0..n_ch {
            let t = s as f64 / spr as f64;
            let v =
                (1000.0 * (2.0 * std::f64::consts::PI * (ch as f64 + 1.0) * t).sin()) as i16;
            let off = (s * n_ch + ch) * 2;
            buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(dir.join("test.eeg"), &buf).unwrap();
}

// ─── Timing helper ─────────────────────────────────────────────────────────────

fn bench<F: Fn() -> R, R>(f: F, runs: usize) -> (R, f64) {
    // Warm up
    let _ = f();

    let mut best_ms = f64::MAX;
    let mut result = None;
    for _ in 0..runs {
        let t = Instant::now();
        let r = f();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
            result = Some(r);
        }
    }
    (result.unwrap(), best_ms)
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

/// Only assert performance in release mode. Debug builds are 5-10× slower
/// due to bounds checks and no optimizations, making timing comparisons
/// against Python baselines meaningless.
const MUST_BE_FASTER_THAN_PYTHON: bool = cfg!(not(debug_assertions));

/// Python MNE timings from golden benchmark (ms). Conservative thresholds.
struct PythonBaseline {
    python_ms: f64,
}

#[test]
fn regression_edf_64ch_60s() {
    let dir = std::env::temp_dir().join("bids_eeg_regr_edf60");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.edf");
    generate_edf(&path, 64, 60, 2048);

    let (data, ms) = bench(|| read_edf(&path, &ReadOptions::default()).unwrap(), 5);

    assert_eq!(data.n_channels(), 64);
    assert_eq!(data.n_samples(0), 60 * 2048);
    assert!((data.sampling_rates[0] - 2048.0).abs() < 0.01);
    // Data in physical units: should be in [-3200, 3200] µV
    assert!(data.data[0].iter().all(|v| v.abs() <= 3200.1));

    // Performance: must beat Python (~12ms for this size)
    let baseline = PythonBaseline { python_ms: 15.0 };
    if MUST_BE_FASTER_THAN_PYTHON {
        assert!(
            ms < baseline.python_ms,
            "EDF 64ch×60s: {:.1}ms > Python baseline {:.1}ms",
            ms, baseline.python_ms
        );
    }
    eprintln!("  EDF 64ch×60s: {:.1}ms (Python baseline: {:.1}ms)", ms, baseline.python_ms);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn regression_edf_2ch_select() {
    let dir = std::env::temp_dir().join("bids_eeg_regr_edf2ch");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.edf");
    generate_edf(&path, 64, 60, 2048);

    let opts = ReadOptions::new().with_channels(vec!["EEG1".into(), "EEG32".into()]);
    let (data, ms) = bench(|| read_edf(&path, &opts).unwrap(), 5);

    assert_eq!(data.n_channels(), 2);
    assert_eq!(data.channel_labels, vec!["EEG1", "EEG32"]);

    // Python: ~12ms (loads everything then picks)
    let baseline = PythonBaseline { python_ms: 15.0 };
    if MUST_BE_FASTER_THAN_PYTHON {
        assert!(
            ms < baseline.python_ms,
            "EDF 2ch select: {:.1}ms > Python baseline {:.1}ms",
            ms, baseline.python_ms
        );
    }
    eprintln!("  EDF 2ch select: {:.1}ms (Python baseline: {:.1}ms)", ms, baseline.python_ms);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn regression_edf_time_window() {
    let dir = std::env::temp_dir().join("bids_eeg_regr_edftw");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.edf");
    generate_edf(&path, 64, 60, 2048);

    let opts = ReadOptions::new().with_time_range(20.0, 30.0);
    let (data, ms) = bench(|| read_edf(&path, &opts).unwrap(), 5);

    assert_eq!(data.n_channels(), 64);
    assert_eq!(data.n_samples(0), 10 * 2048);

    let baseline = PythonBaseline { python_ms: 15.0 };
    if MUST_BE_FASTER_THAN_PYTHON {
        assert!(
            ms < baseline.python_ms,
            "EDF 10s window: {:.1}ms > Python baseline {:.1}ms",
            ms, baseline.python_ms
        );
    }
    eprintln!("  EDF 10s window: {:.1}ms (Python baseline: {:.1}ms)", ms, baseline.python_ms);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn regression_bdf_32ch_30s() {
    let dir = std::env::temp_dir().join("bids_eeg_regr_bdf");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.bdf");
    generate_bdf(&path, 32, 30, 2048);

    let (data, ms) = bench(|| read_edf(&path, &ReadOptions::default()).unwrap(), 5);

    assert_eq!(data.n_channels(), 32);
    assert_eq!(data.n_samples(0), 30 * 2048);

    let baseline = PythonBaseline { python_ms: 12.0 };
    if MUST_BE_FASTER_THAN_PYTHON {
        assert!(
            ms < baseline.python_ms,
            "BDF 32ch×30s: {:.1}ms > Python baseline {:.1}ms",
            ms, baseline.python_ms
        );
    }
    eprintln!("  BDF 32ch×30s: {:.1}ms (Python baseline: {:.1}ms)", ms, baseline.python_ms);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn regression_brainvision_64ch_60s() {
    let dir = std::env::temp_dir().join("bids_eeg_regr_bv");
    std::fs::create_dir_all(&dir).unwrap();
    generate_brainvision(&dir, 64, 60 * 2048, 2048);
    let vhdr = dir.join("test.vhdr");

    let (data, ms) = bench(|| read_brainvision(&vhdr, &ReadOptions::default()).unwrap(), 5);

    assert_eq!(data.n_channels(), 64);
    assert_eq!(data.n_samples(0), 60 * 2048);
    // Should have 2 markers from .vmrk
    assert_eq!(data.annotations.len(), 2);
    assert_eq!(data.annotations[0].description, "S1");

    let baseline = PythonBaseline { python_ms: 25.0 };
    if MUST_BE_FASTER_THAN_PYTHON {
        assert!(
            ms < baseline.python_ms,
            "BV 64ch×60s: {:.1}ms > Python baseline {:.1}ms",
            ms, baseline.python_ms
        );
    }
    eprintln!(
        "  BV 64ch×60s: {:.1}ms (Python baseline: {:.1}ms)",
        ms, baseline.python_ms
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn regression_edf_large_150mb() {
    let dir = std::env::temp_dir().join("bids_eeg_regr_large");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("large.edf");
    generate_edf(&path, 64, 600, 2048);

    let (data, ms) = bench(|| read_edf(&path, &ReadOptions::default()).unwrap(), 3);

    assert_eq!(data.n_channels(), 64);
    assert_eq!(data.n_samples(0), 600 * 2048);

    // Python: ~104ms
    let baseline = PythonBaseline { python_ms: 130.0 };
    if MUST_BE_FASTER_THAN_PYTHON {
        assert!(
            ms < baseline.python_ms,
            "EDF 150MB: {:.1}ms > Python baseline {:.1}ms",
            ms, baseline.python_ms
        );
    }
    eprintln!(
        "  EDF 150MB: {:.1}ms (Python baseline: {:.1}ms)",
        ms, baseline.python_ms
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Verify numeric correctness: the sine wave signal should have predictable values.
#[test]
fn regression_numeric_correctness() {
    let dir = std::env::temp_dir().join("bids_eeg_regr_correct");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.edf");
    generate_edf(&path, 2, 2, 256);

    let data = read_edf(&path, &ReadOptions::default()).unwrap();
    assert_eq!(data.n_channels(), 2);
    assert_eq!(data.n_samples(0), 512);

    // Channel 0: sin(2π × 1 × t), scaled by 1000 digital → physical
    // At t=0: sin(0) = 0, digital=0, physical should be near 0
    assert!(data.data[0][0].abs() < 1.0, "First sample ch0={}", data.data[0][0]);

    // At t=0.25s (sample 64): sin(π/2) = 1, digital=1000
    // Physical = 1000 * (3200-(-3200)) / (32767-(-32768)) + ... ≈ 97.66 µV
    let gain = 6400.0 / 65535.0; // (phys_max - phys_min) / (dig_max - dig_min)
    let expected = 1000.0 * gain; // ≈ 97.66
    let actual = data.data[0][64];
    let err = (actual - expected).abs() / expected;
    assert!(
        err < 0.01,
        "ch0 sample 64: expected {:.4} got {:.4} err={:.4}",
        expected, actual, err
    );

    // Channel 1: sin(2π × 2 × t), at t=0.125 (sample 32): sin(π/2) = 1
    let actual1 = data.data[1][32];
    assert!(
        (actual1 - expected).abs() / expected < 0.01,
        "ch1 sample 32: expected {:.4} got {:.4}",
        expected, actual1
    );

    std::fs::remove_dir_all(&dir).unwrap();
}
