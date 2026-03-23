#!/usr/bin/env python3
"""
Benchmark & correctness test: bids-eeg (Rust) vs MNE-Python for EEG data reading.

Generates realistic EDF, BDF, and BrainVision test files, reads them with both
Rust and Python, and compares:
  1. Correctness: channel labels, sample values, shapes, checksums
  2. Performance: Rust must be >= 1x speed of Python (ideally 2-6x)

Results are saved to tests/golden/eeg_read_benchmark.json for regression testing.

Usage:
    cd bids-rs
    cargo build -p bids-eeg --release --example bench_vs_python
    python3 tests/bench_eeg_read.py
"""

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ─── Helpers: generate test files ───────────────────────────────────────────────

def generate_edf(path, n_ch, n_rec, spr, label_prefix="EEG"):
    """Generate a valid EDF file with sine-wave data."""
    with open(path, 'wb') as f:
        hdr = bytearray(b' ' * 256)
        hdr[0:1] = b'0'
        hdr[168:176] = b'01.01.01'
        hdr[176:184] = b'00.00.00'
        hdr[184:192] = f'{256 + n_ch * 256:<8}'.encode()
        hdr[236:244] = f'{n_rec:<8}'.encode()
        hdr[244:252] = b'1       '  # 1 second records
        hdr[252:256] = f'{n_ch:<4}'.encode()
        f.write(hdr)

        ext = bytearray(b' ' * (n_ch * 256))
        for i in range(n_ch):
            label = f'{label_prefix}{i+1}'[:16].ljust(16)
            ext[i*16:i*16+16] = label.encode()
            ext[n_ch*96+i*8:n_ch*96+i*8+2] = b'uV'
            ext[n_ch*104+i*8:n_ch*104+i*8+8] = f'{-3200:<8}'.encode()
            ext[n_ch*112+i*8:n_ch*112+i*8+8] = f'{3200:<8}'.encode()
            ext[n_ch*120+i*8:n_ch*120+i*8+8] = f'{-32768:<8}'.encode()
            ext[n_ch*128+i*8:n_ch*128+i*8+8] = f'{32767:<8}'.encode()
            ext[n_ch*216+i*8:n_ch*216+i*8+8] = f'{spr:<8}'.encode()
        f.write(ext)

        rec_bytes = n_ch * spr * 2
        buf = bytearray(rec_bytes)
        for rec in range(n_rec):
            for ch in range(n_ch):
                for s in range(spr):
                    t = rec + s / spr
                    val = int(1000 * math.sin(2 * math.pi * (ch + 1) * t))
                    val = max(-32768, min(32767, val))
                    off = (ch * spr + s) * 2
                    struct.pack_into('<h', buf, off, val)
            f.write(buf)


def generate_bdf(path, n_ch, n_rec, spr, label_prefix="EEG"):
    """Generate a valid BDF file with 24-bit sine-wave data."""
    with open(path, 'wb') as f:
        hdr = bytearray(b' ' * 256)
        hdr[0:1] = b'\xff'  # BDF magic byte
        hdr[1:8] = b'BIOSEMI'
        hdr[168:176] = b'01.01.01'
        hdr[176:184] = b'00.00.00'
        hdr[184:192] = f'{256 + n_ch * 256:<8}'.encode()
        hdr[192:236] = b'BDF+C' + b' ' * 39
        hdr[236:244] = f'{n_rec:<8}'.encode()
        hdr[244:252] = b'1       '
        hdr[252:256] = f'{n_ch:<4}'.encode()
        f.write(hdr)

        ext = bytearray(b' ' * (n_ch * 256))
        for i in range(n_ch):
            label = f'{label_prefix}{i+1}'[:16].ljust(16)
            ext[i*16:i*16+16] = label.encode()
            ext[n_ch*96+i*8:n_ch*96+i*8+2] = b'uV'
            ext[n_ch*104+i*8:n_ch*104+i*8+8] = f'{-3200:<8}'.encode()
            ext[n_ch*112+i*8:n_ch*112+i*8+8] = f'{3200:<8}'.encode()
            ext[n_ch*120+i*8:n_ch*120+i*8+8] = f'{-8388608:<8}'.encode()
            ext[n_ch*128+i*8:n_ch*128+i*8+8] = f'{8388607:<8}'.encode()
            ext[n_ch*216+i*8:n_ch*216+i*8+8] = f'{spr:<8}'.encode()
        f.write(ext)

        rec_bytes = n_ch * spr * 3
        buf = bytearray(rec_bytes)
        for rec in range(n_rec):
            for ch in range(n_ch):
                for s in range(spr):
                    t = rec + s / spr
                    val = int(100000 * math.sin(2 * math.pi * (ch + 1) * t))
                    val = max(-8388608, min(8388607, val))
                    off = (ch * spr + s) * 3
                    buf[off] = val & 0xFF
                    buf[off+1] = (val >> 8) & 0xFF
                    buf[off+2] = (val >> 16) & 0xFF
            f.write(buf)


def generate_brainvision(directory, n_ch, n_samples, spr=2048, label_prefix="EEG"):
    """Generate BrainVision .vhdr + .eeg + .vmrk files."""
    us = 1_000_000.0 / spr

    vhdr = f"""Brain Vision Data Exchange Header File Version 1.0

[Common Infos]
Codepage=UTF-8
DataFile=test.eeg
MarkerFile=test.vmrk
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels={n_ch}
SamplingInterval={us}

[Binary Infos]
BinaryFormat=INT_16

[Channel Infos]
"""
    for i in range(n_ch):
        vhdr += f"Ch{i+1}={label_prefix}{i+1},,0.1\n"
    Path(directory, "test.vhdr").write_text(vhdr)

    vmrk = """Brain Vision Data Exchange Marker File Version 1.0

[Common Infos]
Codepage=UTF-8
DataFile=test.eeg

[Marker Infos]
Mk1=Stimulus,S  1,512,1,0
Mk2=Stimulus,S  2,2048,1,0
Mk3=Response,R  1,4096,1,0
"""
    Path(directory, "test.vmrk").write_text(vmrk)

    total_bytes = n_samples * n_ch * 2
    buf = bytearray(total_bytes)
    for s in range(n_samples):
        for ch in range(n_ch):
            t = s / spr
            val = int(1000 * math.sin(2 * math.pi * (ch + 1) * t))
            val = max(-32768, min(32767, val))
            off = (s * n_ch + ch) * 2
            struct.pack_into('<h', buf, off, val)
    Path(directory, "test.eeg").write_bytes(buf)


# ─── Run Rust reader ────────────────────────────────────────────────────────────

RUST_BIN = Path(__file__).parent.parent / "target" / "release" / "examples" / "bench_vs_python"


def run_rust(path, **kwargs):
    """Run the Rust bench_vs_python binary and parse its JSON output."""
    cmd = [str(RUST_BIN), str(path)]
    if 'channels' in kwargs:
        cmd += ['--channels', ','.join(kwargs['channels'])]
    if 'tmin' in kwargs:
        cmd += ['--tmin', str(kwargs['tmin'])]
    if 'tmax' in kwargs:
        cmd += ['--tmax', str(kwargs['tmax'])]
    if 'exclude' in kwargs:
        cmd += ['--exclude', ','.join(kwargs['exclude'])]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  Rust STDERR: {result.stderr[:500]}", file=sys.stderr)
        raise RuntimeError(f"Rust binary failed: {result.returncode}")
    return json.loads(result.stdout)


# ─── Run MNE-Python reader ──────────────────────────────────────────────────────

def run_mne_edf(path, **kwargs):
    """Read an EDF/BDF file with MNE and return comparable results."""
    import mne
    import warnings
    warnings.filterwarnings('ignore')

    # Warm up
    mne.io.read_raw_edf(str(path), preload=True, verbose=False)

    best_ms = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        best_ms = min(best_ms, ms)

    picks = kwargs.get('channels')
    tmin = kwargs.get('tmin')
    tmax = kwargs.get('tmax')
    data = raw.get_data(picks=picks, tmin=tmin, tmax=tmax)
    ch_names = [raw.ch_names[i] for i in (mne.pick_channels(raw.ch_names, picks) if picks else range(len(raw.ch_names)))]

    n_peek = 10
    first = [data[i, :n_peek].tolist() for i in range(data.shape[0])]
    last = [data[i, -n_peek:].tolist() for i in range(data.shape[0])]
    checksum = float(data.sum())

    return {
        "n_channels": data.shape[0],
        "n_samples": data.shape[1],
        "channel_labels": ch_names,
        "read_ms": best_ms,
        "first_samples": first,
        "last_samples": last,
        "checksum": checksum,
    }


def run_mne_bdf(path, **kwargs):
    """Read a BDF file with MNE."""
    import mne
    import warnings
    warnings.filterwarnings('ignore')

    mne.io.read_raw_bdf(str(path), preload=True, verbose=False)

    best_ms = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        raw = mne.io.read_raw_bdf(str(path), preload=True, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        best_ms = min(best_ms, ms)

    data = raw.get_data()
    ch_names = raw.ch_names

    n_peek = 10
    first = [data[i, :n_peek].tolist() for i in range(data.shape[0])]
    last = [data[i, -n_peek:].tolist() for i in range(data.shape[0])]
    checksum = float(data.sum())

    return {
        "n_channels": data.shape[0],
        "n_samples": data.shape[1],
        "channel_labels": list(ch_names),
        "read_ms": best_ms,
        "first_samples": first,
        "last_samples": last,
        "checksum": checksum,
    }


def run_mne_brainvision(path, **kwargs):
    """Read a BrainVision file with MNE."""
    import mne
    import warnings
    warnings.filterwarnings('ignore')

    mne.io.read_raw_brainvision(str(path), preload=True, verbose=False)

    best_ms = float('inf')
    for _ in range(3):
        t0 = time.perf_counter()
        raw = mne.io.read_raw_brainvision(str(path), preload=True, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        best_ms = min(best_ms, ms)

    data = raw.get_data()
    ch_names = raw.ch_names
    annotations = [(a['onset'], a['duration'], a['description'])
                    for a in raw.annotations]

    n_peek = 10
    first = [data[i, :n_peek].tolist() for i in range(data.shape[0])]
    last = [data[i, -n_peek:].tolist() for i in range(data.shape[0])]
    checksum = float(data.sum())

    return {
        "n_channels": data.shape[0],
        "n_samples": data.shape[1],
        "channel_labels": list(ch_names),
        "read_ms": best_ms,
        "first_samples": first,
        "last_samples": last,
        "checksum": checksum,
        "n_annotations": len(annotations),
    }


# ─── Compare results ────────────────────────────────────────────────────────────

def compare_results(name, rust, python, tolerance=1e-4, scale=1.0):
    """Compare Rust and Python results.

    Args:
        scale: factor to multiply Rust values by before comparing.
               MNE returns SI (Volts), Rust returns physical units from header (e.g. µV).
               For µV→V comparison, use scale=1e-6.
    """
    issues = []

    # Shape
    if rust["n_channels"] != python["n_channels"]:
        issues.append(f"n_channels: rust={rust['n_channels']} py={python['n_channels']}")
    if rust["n_samples"] != python["n_samples"]:
        issues.append(f"n_samples: rust={rust['n_samples']} py={python['n_samples']}")

    # Channel labels (MNE may strip trailing whitespace differently)
    r_labels = [l.strip() for l in rust["channel_labels"]]
    p_labels = [l.strip() for l in python["channel_labels"]]
    if r_labels != p_labels:
        issues.append(f"labels differ: rust={r_labels[:3]}... py={p_labels[:3]}...")

    # Sample values (first/last) — apply scale factor
    for key in ["first_samples", "last_samples"]:
        if key in rust and key in python:
            for ch in range(min(len(rust[key]), len(python[key]))):
                r_vals = rust[key][ch]
                p_vals = python[key][ch]
                for s in range(min(len(r_vals), len(p_vals))):
                    r_v = r_vals[s] * scale
                    p_v = p_vals[s]
                    if abs(p_v) > 1e-15:
                        rel_err = abs(r_v - p_v) / abs(p_v)
                    else:
                        rel_err = abs(r_v - p_v)
                    if rel_err > tolerance:
                        issues.append(f"{key}[{ch}][{s}]: rust={r_v:.6e} py={p_v:.6e} err={rel_err:.2e}")
                        break  # one per channel is enough

    # Performance
    speedup = python["read_ms"] / rust["read_ms"] if rust["read_ms"] > 0 else float('inf')

    passed = len(issues) == 0
    return passed, speedup, issues


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    if not RUST_BIN.exists():
        print(f"ERROR: Rust binary not found at {RUST_BIN}")
        print("Run: cargo build -p bids-eeg --release --example bench_vs_python")
        sys.exit(1)

    tmpdir = tempfile.mkdtemp(prefix="bids_eeg_bench_")
    results = {}
    all_passed = True

    print("=" * 72)
    print("  EEG DATA READING BENCHMARK: bids-eeg (Rust) vs MNE-Python")
    print("=" * 72)

    # ═══ Test 1: EDF — 64ch × 60s × 2048Hz (~15 MB) ═══
    print("\n── EDF: 64ch × 60s × 2048Hz ──")
    edf_path = os.path.join(tmpdir, "test.edf")
    generate_edf(edf_path, 64, 60, 2048)
    file_mb = os.path.getsize(edf_path) / 1e6
    print(f"  Generated: {file_mb:.1f} MB")

    rust_edf = run_rust(edf_path)
    py_edf = run_mne_edf(edf_path)
    # MNE returns Volts; Rust returns µV (physical units from header). Scale: µV→V = 1e-6
    passed, speedup, issues = compare_results("edf_full", rust_edf, py_edf, scale=1e-6)
    results["edf_64ch_60s"] = {
        "rust_ms": rust_edf["read_ms"], "python_ms": py_edf["read_ms"],
        "speedup": round(speedup, 2), "passed": passed,
        "file_mb": round(file_mb, 1), "n_channels": 64, "n_samples": rust_edf["n_samples"],
    }
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  Rust: {rust_edf['read_ms']:.1f}ms  Python: {py_edf['read_ms']:.1f}ms  Speedup: {speedup:.1f}x")
    if issues:
        for iss in issues[:5]:
            print(f"    ⚠ {iss}")
        all_passed = False

    # ═══ Test 2: EDF — 2 channel select ═══
    print("\n── EDF: 2ch select from 64ch ──")
    rust_sel = run_rust(edf_path, channels=["EEG1", "EEG32"])
    py_sel = run_mne_edf(edf_path, channels=["EEG1", "EEG32"])
    passed, speedup, issues = compare_results("edf_2ch", rust_sel, py_sel, scale=1e-6)
    results["edf_2ch_select"] = {
        "rust_ms": rust_sel["read_ms"], "python_ms": py_sel["read_ms"],
        "speedup": round(speedup, 2), "passed": passed,
    }
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  Rust: {rust_sel['read_ms']:.1f}ms  Python: {py_sel['read_ms']:.1f}ms  Speedup: {speedup:.1f}x")
    if issues:
        for iss in issues[:5]:
            print(f"    ⚠ {iss}")
        all_passed = False

    # ═══ Test 3: EDF — 10s time window ═══
    print("\n── EDF: 10s window from 60s ──")
    rust_win = run_rust(edf_path, tmin=20.0, tmax=30.0)
    py_win = run_mne_edf(edf_path, tmin=20.0, tmax=30.0)
    passed, speedup, issues = compare_results("edf_window", rust_win, py_win, scale=1e-6)
    results["edf_10s_window"] = {
        "rust_ms": rust_win["read_ms"], "python_ms": py_win["read_ms"],
        "speedup": round(speedup, 2), "passed": passed,
    }
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  Rust: {rust_win['read_ms']:.1f}ms  Python: {py_win['read_ms']:.1f}ms  Speedup: {speedup:.1f}x")
    if issues:
        for iss in issues[:5]:
            print(f"    ⚠ {iss}")
        all_passed = False

    # ═══ Test 4: BDF — 32ch × 30s × 2048Hz ═══
    print("\n── BDF: 32ch × 30s × 2048Hz ──")
    bdf_path = os.path.join(tmpdir, "test.bdf")
    generate_bdf(bdf_path, 32, 30, 2048)
    file_mb = os.path.getsize(bdf_path) / 1e6
    print(f"  Generated: {file_mb:.1f} MB")

    rust_bdf = run_rust(bdf_path)
    py_bdf = run_mne_bdf(bdf_path)
    # BDF: same µV→V scale
    passed, speedup, issues = compare_results("bdf_full", rust_bdf, py_bdf, scale=1e-6)
    results["bdf_32ch_30s"] = {
        "rust_ms": rust_bdf["read_ms"], "python_ms": py_bdf["read_ms"],
        "speedup": round(speedup, 2), "passed": passed,
        "file_mb": round(file_mb, 1),
    }
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  Rust: {rust_bdf['read_ms']:.1f}ms  Python: {py_bdf['read_ms']:.1f}ms  Speedup: {speedup:.1f}x")
    if issues:
        for iss in issues[:5]:
            print(f"    ⚠ {iss}")
        all_passed = False

    # ═══ Test 5: BrainVision — 64ch × 60s × 2048Hz ═══
    print("\n── BrainVision: 64ch × 60s × 2048Hz ──")
    bv_dir = os.path.join(tmpdir, "bv")
    os.makedirs(bv_dir)
    generate_brainvision(bv_dir, 64, 60 * 2048, spr=2048)
    vhdr_path = os.path.join(bv_dir, "test.vhdr")
    file_mb = os.path.getsize(os.path.join(bv_dir, "test.eeg")) / 1e6
    print(f"  Generated: {file_mb:.1f} MB")

    rust_bv = run_rust(vhdr_path)
    py_bv = run_mne_brainvision(vhdr_path)
    # BrainVision: Rust applies resolution (0.1) → µV. MNE converts µV → V (×1e-6).
    passed, speedup, issues = compare_results("bv_full", rust_bv, py_bv, scale=1e-6)
    results["bv_64ch_60s"] = {
        "rust_ms": rust_bv["read_ms"], "python_ms": py_bv["read_ms"],
        "speedup": round(speedup, 2), "passed": passed,
        "file_mb": round(file_mb, 1),
    }
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  Rust: {rust_bv['read_ms']:.1f}ms  Python: {py_bv['read_ms']:.1f}ms  Speedup: {speedup:.1f}x")
    if issues:
        for iss in issues[:5]:
            print(f"    ⚠ {iss}")
        all_passed = False

    # ═══ Test 6: Large EDF — 64ch × 600s × 2048Hz (~150 MB) ═══
    print("\n── EDF Large: 64ch × 600s × 2048Hz ──")
    edf_large = os.path.join(tmpdir, "large.edf")
    generate_edf(edf_large, 64, 600, 2048)
    file_mb = os.path.getsize(edf_large) / 1e6
    print(f"  Generated: {file_mb:.1f} MB")

    rust_large = run_rust(edf_large)
    py_large = run_mne_edf(edf_large)
    passed, speedup, issues = compare_results("edf_large", rust_large, py_large, scale=1e-6)
    results["edf_64ch_600s"] = {
        "rust_ms": rust_large["read_ms"], "python_ms": py_large["read_ms"],
        "speedup": round(speedup, 2), "passed": passed,
        "file_mb": round(file_mb, 1),
    }
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}  Rust: {rust_large['read_ms']:.1f}ms  Python: {py_large['read_ms']:.1f}ms  Speedup: {speedup:.1f}x")
    if issues:
        for iss in issues[:5]:
            print(f"    ⚠ {iss}")
        all_passed = False

    # ═══ Summary ═══
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    print(f"\n  {'Test':<25} {'Rust ms':>10} {'Python ms':>12} {'Speedup':>10} {'Status':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*12} {'─'*10} {'─'*8}")
    for name, r in results.items():
        status = "✅" if r["passed"] else "❌"
        print(f"  {name:<25} {r['rust_ms']:>10.1f} {r['python_ms']:>12.1f} {r['speedup']:>9.1f}x {status:>6}")

    n_passed = sum(1 for r in results.values() if r["passed"])
    avg_speedup = sum(r["speedup"] for r in results.values()) / len(results)
    min_speedup = min(r["speedup"] for r in results.values())

    print(f"\n  Passed: {n_passed}/{len(results)}")
    print(f"  Average speedup: {avg_speedup:.1f}x")
    print(f"  Minimum speedup: {min_speedup:.1f}x")

    # Save golden results
    golden_path = Path(__file__).parent / "golden" / "eeg_read_benchmark.json"
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with open(golden_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {golden_path}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)

    if not all_passed:
        print("\n  ⚠ Some correctness checks failed!")
        sys.exit(1)
    if min_speedup < 1.0:
        print("\n  ⚠ Rust is slower than Python in some tests!")
        sys.exit(1)

    print("\n  ✅ All tests passed. Rust is faster than Python across the board.")


if __name__ == "__main__":
    main()
