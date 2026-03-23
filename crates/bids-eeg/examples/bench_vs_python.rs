//! Benchmark binary that reads EEG files and outputs JSON for comparison with Python MNE.
//!
//! Usage: bench_vs_python <path> [--channels ch1,ch2] [--tmin T] [--tmax T]
//!
//! Outputs JSON to stdout:
//! {
//!   "n_channels": 64,
//!   "n_samples": 1228800,
//!   "channel_labels": ["EEG1", ...],
//!   "sampling_rates": [2048.0, ...],
//!   "duration": 600.0,
//!   "read_ms": 48.3,
//!   "first_samples": [[ch0_s0, ch0_s1, ...], [ch1_s0, ...]],  // first 10 per channel
//!   "last_samples": [[ch0_sN-9, ...], ...],  // last 10 per channel
//!   "n_annotations": 0,
//!   "stim_indices": [],
//!   "checksum": 123456.789  // sum of all samples (for correctness check)
//! }

use bids_eeg::{ReadOptions, read_eeg_data};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bench_vs_python <path> [--channels ch1,ch2] [--tmin T] [--tmax T]");
        std::process::exit(1);
    }

    let path = std::path::Path::new(&args[1]);
    let mut opts = ReadOptions::new();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--channels" => {
                i += 1;
                opts.channels = Some(args[i].split(',').map(|s| s.to_string()).collect());
            }
            "--tmin" => {
                i += 1;
                let tmin: f64 = args[i].parse().unwrap();
                let tmax = opts.end_time.unwrap_or(f64::MAX);
                opts = opts.with_time_range(tmin, tmax);
            }
            "--tmax" => {
                i += 1;
                let tmax: f64 = args[i].parse().unwrap();
                let tmin = opts.start_time.unwrap_or(0.0);
                opts = opts.with_time_range(tmin, tmax);
            }
            "--exclude" => {
                i += 1;
                opts.exclude = Some(args[i].split(',').map(|s| s.to_string()).collect());
            }
            _ => {}
        }
        i += 1;
    }

    // Warm up filesystem cache
    let _ = read_eeg_data(path, &opts);

    // Timed run (best of 3)
    let mut best_ms = f64::MAX;
    let mut data = None;
    for _ in 0..3 {
        let t = Instant::now();
        let d = read_eeg_data(path, &opts).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if ms < best_ms {
            best_ms = ms;
            data = Some(d);
        }
    }
    let data = data.unwrap();

    // Extract first/last 10 samples per channel for correctness check
    let n_peek = 10;
    let first_samples: Vec<Vec<f64>> = data.data.iter()
        .map(|ch| ch.iter().take(n_peek).copied().collect())
        .collect();
    let last_samples: Vec<Vec<f64>> = data.data.iter()
        .map(|ch| {
            let start = ch.len().saturating_sub(n_peek);
            ch[start..].to_vec()
        })
        .collect();

    // Checksum: sum of all values (for quick correctness check)
    let checksum: f64 = data.data.iter()
        .flat_map(|ch| ch.iter())
        .sum();

    // Output JSON
    println!("{{");
    println!("  \"n_channels\": {},", data.n_channels());
    println!("  \"n_samples\": {},", data.n_samples(0));
    println!("  \"channel_labels\": {:?},", data.channel_labels);
    println!("  \"sampling_rates\": {:?},", data.sampling_rates);
    println!("  \"duration\": {},", data.duration);
    println!("  \"read_ms\": {:.3},", best_ms);
    println!("  \"first_samples\": {:?},", first_samples);
    println!("  \"last_samples\": {:?},", last_samples);
    println!("  \"n_annotations\": {},", data.annotations.len());
    println!("  \"stim_indices\": {:?},", data.stim_channel_indices);
    println!("  \"checksum\": {:.6}", checksum);
    println!("}}");
}
