use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn create_test_edf(path: &Path, n_ch: usize, n_rec: usize, spr: usize) {
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
        let s = format!("{:<8}", "-3200");
        ext[n_ch * 104 + i * 8..n_ch * 104 + i * 8 + 8].copy_from_slice(s.as_bytes());
        let s = format!("{:<8}", "3200");
        ext[n_ch * 112 + i * 8..n_ch * 112 + i * 8 + 8].copy_from_slice(s.as_bytes());
        let s = format!("{:<8}", "-32768");
        ext[n_ch * 120 + i * 8..n_ch * 120 + i * 8 + 8].copy_from_slice(s.as_bytes());
        let s = format!("{:<8}", "32767");
        ext[n_ch * 128 + i * 8..n_ch * 128 + i * 8 + 8].copy_from_slice(s.as_bytes());
        let s = format!("{:<8}", spr);
        ext[n_ch * 216 + i * 8..n_ch * 216 + i * 8 + 8].copy_from_slice(s.as_bytes());
    }
    file.write_all(&ext).unwrap();

    // Write data in bulk per record
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

fn create_test_bv(dir: &Path, n_ch: usize, n_samples: usize) {
    // Write .vhdr
    let vhdr_path = dir.join("bench.vhdr");
    let mut vhdr = String::new();
    vhdr.push_str("Brain Vision Data Exchange Header File Version 1.0\n\n");
    vhdr.push_str("[Common Infos]\nDataFile=bench.eeg\nDataOrientation=MULTIPLEXED\n");
    vhdr.push_str("SamplingInterval=488.28125\n\n"); // 2048 Hz
    vhdr.push_str("[Binary Infos]\nBinaryFormat=INT_16\n\n[Channel Infos]\n");
    for i in 0..n_ch {
        vhdr.push_str(&format!("Ch{}=EEG{},,0.1\n", i + 1, i + 1));
    }
    std::fs::write(&vhdr_path, &vhdr).unwrap();

    // Write binary data (multiplexed int16)
    let eeg_path = dir.join("bench.eeg");
    let total_bytes = n_samples * n_ch * 2;
    let mut buf = vec![0u8; total_bytes];
    for s in 0..n_samples {
        for ch in 0..n_ch {
            let t = s as f64 / 2048.0;
            let v = (1000.0 * (2.0 * std::f64::consts::PI * (ch as f64 + 1.0) * t).sin()) as i16;
            let off = (s * n_ch + ch) * 2;
            buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(&eeg_path, &buf).unwrap();
}

fn main() {
    let dir = std::env::temp_dir().join("bids_eeg_bench");
    std::fs::create_dir_all(&dir).unwrap();

    let n_ch = 64;
    let n_rec = 600;
    let spr = 2048;

    // ═══ EDF Benchmark ═══
    let path = dir.join("bench.edf");
    println!("Creating test EDF: {}ch × {}s × {}Hz...", n_ch, n_rec, spr);
    let t = Instant::now();
    create_test_edf(&path, n_ch, n_rec, spr);
    let file_mb = std::fs::metadata(&path).unwrap().len() as f64 / 1e6;
    println!(
        "  Created in {:.2}s ({:.0} MB)\n",
        t.elapsed().as_secs_f64(),
        file_mb
    );

    println!("EDF Read All Channels ({} ch × {} samples):", n_ch, n_rec * spr);
    for i in 0..3 {
        let t = Instant::now();
        let data = bids_eeg::read_edf(&path, &bids_eeg::ReadOptions::default()).unwrap();
        let e = t.elapsed();
        println!(
            "  run {}: {:.3}s ({:.0} MB/s) → {}ch × {} samples",
            i + 1,
            e.as_secs_f64(),
            file_mb / e.as_secs_f64(),
            data.n_channels(),
            data.n_samples(0)
        );
    }

    println!("\nEDF Read 2 Channels:");
    for i in 0..3 {
        let t = Instant::now();
        let data = bids_eeg::read_edf(
            &path,
            &bids_eeg::ReadOptions::new()
                .with_channels(vec!["EEG1".into(), "EEG32".into()]),
        )
        .unwrap();
        let e = t.elapsed();
        println!(
            "  run {}: {:.3}s → {}ch × {} samples",
            i + 1,
            e.as_secs_f64(),
            data.n_channels(),
            data.n_samples(0)
        );
    }

    println!("\nEDF Read 10s Window (all ch):");
    for i in 0..3 {
        let t = Instant::now();
        let data = bids_eeg::read_edf(
            &path,
            &bids_eeg::ReadOptions::new().with_time_range(100.0, 110.0),
        )
        .unwrap();
        let e = t.elapsed();
        println!(
            "  run {}: {:.3}s → {}ch × {} samples",
            i + 1,
            e.as_secs_f64(),
            data.n_channels(),
            data.n_samples(0)
        );
    }

    // ═══ BrainVision Benchmark ═══
    let n_bv_samples = n_rec * spr; // same total
    println!(
        "\nCreating test BrainVision: {}ch × {} samples...",
        n_ch, n_bv_samples
    );
    let t = Instant::now();
    create_test_bv(&dir, n_ch, n_bv_samples);
    let bv_path = dir.join("bench.vhdr");
    let bv_mb =
        std::fs::metadata(dir.join("bench.eeg")).unwrap().len() as f64 / 1e6;
    println!(
        "  Created in {:.2}s ({:.0} MB)\n",
        t.elapsed().as_secs_f64(),
        bv_mb
    );

    println!(
        "BrainVision Read All Channels ({} ch × {} samples):",
        n_ch, n_bv_samples
    );
    for i in 0..3 {
        let t = Instant::now();
        let data =
            bids_eeg::read_brainvision(&bv_path, &bids_eeg::ReadOptions::default()).unwrap();
        let e = t.elapsed();
        println!(
            "  run {}: {:.3}s ({:.0} MB/s) → {}ch × {} samples",
            i + 1,
            e.as_secs_f64(),
            bv_mb / e.as_secs_f64(),
            data.n_channels(),
            data.n_samples(0)
        );
    }

    std::fs::remove_dir_all(&dir).unwrap();
    println!("\nDone.");
}
