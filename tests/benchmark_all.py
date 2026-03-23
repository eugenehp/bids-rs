#!/usr/bin/env python3
"""
Comprehensive benchmarks: pybids_rs (Rust) vs PyBIDS (Python)
across all functions and modalities.

Usage:
    source /tmp/pybids_venv/bin/activate
    cd bids-rs/crates/pybids-rs && maturin develop --release
    cd ../.. && python tests/benchmark_all.py
"""

import time
import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from collections import defaultdict

EXAMPLES = Path(__file__).parent.parent.parent / "pybids" / "bids-examples"

results = defaultdict(dict)

def bench(name, rust_fn, python_fn, n=10, warmup=1):
    """Benchmark two functions, return (rust_ms, python_ms, speedup, match)."""
    # Warmup
    for _ in range(warmup):
        r_result = rust_fn()
        p_result = python_fn()

    # Rust
    t0 = time.perf_counter()
    for _ in range(n):
        r_result = rust_fn()
    t_rust = (time.perf_counter() - t0) / n * 1000

    # Python
    t0 = time.perf_counter()
    for _ in range(n):
        p_result = python_fn()
    t_python = (time.perf_counter() - t0) / n * 1000

    speedup = t_python / t_rust if t_rust > 0 else 0
    results[name] = {"rust_ms": round(t_rust, 2), "python_ms": round(t_python, 2), "speedup": round(speedup, 1)}
    return r_result, p_result


def compare(name, r_val, p_val):
    """Compare two values for equality."""
    if isinstance(r_val, (list, tuple)) and isinstance(p_val, (list, tuple)):
        match = sorted(map(str, r_val)) == sorted(map(str, p_val))
    elif isinstance(r_val, dict) and isinstance(p_val, dict):
        match = set(r_val.keys()) == set(p_val.keys())
    elif isinstance(r_val, (int, float)) and isinstance(p_val, (int, float)):
        match = abs(r_val - p_val) < max(abs(r_val), abs(p_val)) * 1e-6 + 1e-15
    else:
        match = str(r_val) == str(p_val)
    results[name]["match"] = match
    return match


# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("  COMPREHENSIVE BENCHMARK: pybids_rs vs PyBIDS")
print("=" * 70)

import pybids_rs
from bids import BIDSLayout
from bids.modeling.hrf import spm_hrf as py_spm_hrf, glover_hrf as py_glover_hrf
from scipy.signal import butter as py_butter, filtfilt as py_filtfilt
from bids.external import inflect

# ─── 1. Layout Indexing ───────────────────────────────────
print("\n── Layout Indexing ──")

eeg_datasets = {}
for ds in ["eeg_cbm", "eeg_rishikesh"]:
    ds_path = str(EXAMPLES / ds)
    if not os.path.exists(ds_path):
        continue

    r, p = bench(
        f"index_{ds}",
        lambda p=ds_path: pybids_rs.BIDSLayout(p),
        lambda p=ds_path: BIDSLayout(p),
        n=5
    )
    eeg_datasets[ds] = (r, p)
    compare(f"index_{ds}", r.get_subjects(), p.get_subjects())
    m = "✅" if results[f"index_{ds}"]["match"] else "❌"
    print(f"  {ds}: Rust {results[f'index_{ds}']['rust_ms']:.1f}ms vs Python {results[f'index_{ds}']['python_ms']:.1f}ms "
          f"→ {results[f'index_{ds}']['speedup']:.1f}x {m}")

# ─── 2. Entity Queries ───────────────────────────────────
print("\n── Entity Queries ──")

if eeg_datasets:
    ds = list(eeg_datasets.keys())[0]
    ds_path = str(EXAMPLES / ds)
    r_layout = pybids_rs.BIDSLayout(ds_path)
    p_layout = BIDSLayout(ds_path)

    for entity_fn in ["get_subjects", "get_sessions", "get_tasks", "get_runs", "get_datatypes"]:
        name = f"query_{entity_fn}"
        r, p = bench(name,
            lambda fn=entity_fn: getattr(r_layout, fn)(),
            lambda fn=entity_fn: getattr(p_layout, fn)(),
            n=100)
        compare(name, r, p)
        m = "✅" if results[name]["match"] else "❌"
        print(f"  {entity_fn}: Rust {results[name]['rust_ms']:.2f}ms vs Python {results[name]['python_ms']:.2f}ms "
              f"→ {results[name]['speedup']:.1f}x {m}")

# ─── 3. File Queries ──────────────────────────────────────
print("\n── File Queries ──")

if eeg_datasets:
    # Query EEG files
    r, p = bench("query_eeg_files",
        lambda: r_layout.get(suffix="eeg"),
        lambda: p_layout.get(suffix="eeg"),
        n=50)
    compare("query_eeg_files", len(r), len(p))
    m = "✅" if results["query_eeg_files"]["match"] else "❌"
    print(f"  get(suffix='eeg'): Rust {results['query_eeg_files']['rust_ms']:.2f}ms vs Python {results['query_eeg_files']['python_ms']:.2f}ms "
          f"→ {results['query_eeg_files']['speedup']:.1f}x (n={len(r)}) {m}")

    # Query with multiple filters
    first_sub = r_layout.get_subjects()[0]
    r, p = bench("query_filtered",
        lambda: r_layout.get(suffix="eeg", subject=first_sub),
        lambda: p_layout.get(suffix="eeg", subject=first_sub),
        n=100)
    compare("query_filtered", len(r), len(p))
    m = "✅" if results["query_filtered"]["match"] else "❌"
    print(f"  get(suffix='eeg', subject='{first_sub}'): Rust {results['query_filtered']['rust_ms']:.2f}ms vs Python {results['query_filtered']['python_ms']:.2f}ms "
          f"→ {results['query_filtered']['speedup']:.1f}x (n={len(r)}) {m}")

    # Events query
    r, p = bench("query_events",
        lambda: r_layout.get(suffix="events", extension=".tsv"),
        lambda: p_layout.get(suffix="events", extension=".tsv"),
        n=50)
    compare("query_events", len(r), len(p))
    m = "✅" if results["query_events"]["match"] else "❌"
    print(f"  get(suffix='events'): Rust {results['query_events']['rust_ms']:.2f}ms vs Python {results['query_events']['python_ms']:.2f}ms "
          f"→ {results['query_events']['speedup']:.1f}x (n={len(r)}) {m}")

# ─── 4. Metadata ─────────────────────────────────────────
print("\n── Metadata ──")

if eeg_datasets:
    eeg_files_r = r_layout.get(suffix="eeg")
    eeg_files_p = p_layout.get(suffix="eeg")
    if eeg_files_r:
        first_path_r = eeg_files_r[0].path
        first_path_p = eeg_files_p[0].path

        r, p = bench("get_metadata",
            lambda: r_layout.get_metadata(first_path_r),
            lambda: p_layout.get_metadata(first_path_p),
            n=100)
        # Compare key sets
        compare("get_metadata", set(r.keys()), set(p.keys()))
        m = "✅" if results["get_metadata"]["match"] else "❌"
        print(f"  get_metadata: Rust {results['get_metadata']['rust_ms']:.2f}ms vs Python {results['get_metadata']['python_ms']:.2f}ms "
              f"→ {results['get_metadata']['speedup']:.1f}x ({len(r)} keys) {m}")

# ─── 5. HRF ──────────────────────────────────────────────
print("\n── HRF ──")

for hrf_name, r_fn, p_fn in [
    ("spm_hrf", lambda: pybids_rs.spm_hrf(2.0), lambda: py_spm_hrf(2.0, oversampling=50, time_length=32.0).tolist()),
    ("glover_hrf", lambda: pybids_rs.glover_hrf(2.0), lambda: py_glover_hrf(2.0, oversampling=50, time_length=32.0).tolist()),
]:
    r, p = bench(hrf_name, r_fn, p_fn, n=1000)
    # Compare peak index
    r_peak = r.index(max(r))
    p_peak = p.index(max(p))
    compare(hrf_name, r_peak, p_peak)
    m = "✅" if results[hrf_name]["match"] else "❌"
    print(f"  {hrf_name}: Rust {results[hrf_name]['rust_ms']:.3f}ms vs Python {results[hrf_name]['python_ms']:.3f}ms "
          f"→ {results[hrf_name]['speedup']:.1f}x (peak={r_peak}) {m}")

# ─── 6. Butterworth Filter ───────────────────────────────
print("\n── Butterworth Filter ──")

for order, cutoff in [(1, 0.5), (3, 0.3), (5, 0.2)]:
    name = f"butter_o{order}_c{cutoff}"
    r, p = bench(name,
        lambda o=order, c=cutoff: pybids_rs.butter_lowpass(o, c),
        lambda o=order, c=cutoff: (py_butter(o, c, btype='low')[0].tolist(), py_butter(o, c, btype='low')[1].tolist()),
        n=10000)
    r_b, r_a = r
    p_b, p_a = p
    max_err = max(abs(a - b) for a, b in zip(r_b, p_b))
    results[name]["max_err"] = f"{max_err:.2e}"
    compare(name, len(r_b), len(p_b))
    m = "✅" if max_err < 1e-10 else "❌"
    print(f"  butter({order}, {cutoff}): Rust {results[name]['rust_ms']:.4f}ms vs Python {results[name]['python_ms']:.4f}ms "
          f"→ {results[name]['speedup']:.0f}x (err={max_err:.1e}) {m}")

# ─── 7. filtfilt ─────────────────────────────────────────
print("\n── filtfilt ──")

signal = [np.sin(2 * np.pi * 5 * t / 100) + np.sin(2 * np.pi * 40 * t / 100) for t in range(200)]
b_py, a_py = py_butter(5, 0.2)
b_rs, a_rs = pybids_rs.butter_lowpass(5, 0.2)

r, p = bench("filtfilt",
    lambda: pybids_rs.filtfilt(list(b_rs), list(a_rs), signal),
    lambda: py_filtfilt(b_py, a_py, signal).tolist(),
    n=1000)
r_rms = (sum(v**2 for v in r) / len(r)) ** 0.5
p_rms = (sum(v**2 for v in p) / len(p)) ** 0.5
rms_match = abs(r_rms - p_rms) / max(p_rms, 1e-15) < 0.05
results["filtfilt"]["rms_match"] = rms_match
m = "✅" if rms_match else "❌"
print(f"  filtfilt(200 samples): Rust {results['filtfilt']['rust_ms']:.3f}ms vs Python {results['filtfilt']['python_ms']:.3f}ms "
      f"→ {results['filtfilt']['speedup']:.1f}x (RMS rust={r_rms:.4f} py={p_rms:.4f}) {m}")

# ─── 8. Schema Validation ────────────────────────────────
print("\n── Schema Validation ──")

test_paths = [
    "participants.tsv",
    "dataset_description.json",
    "sub-01/eeg/sub-01_task-rest_eeg.edf",
    "sub-01/func/sub-01_task-rest_bold.nii.gz",
    "sub-01/anat/sub-01_T1w.nii.gz",
    "sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.edf",
]
r, p = bench("schema_validate",
    lambda: [pybids_rs.validate_bids_path(p) for p in test_paths],
    lambda: [True for _ in test_paths],  # PyBIDS doesn't have this standalone
    n=10000)
print(f"  validate {len(test_paths)} paths: Rust {results['schema_validate']['rust_ms']:.4f}ms "
      f"(all valid: {all(r)})")

# ─── 9. Inflect ──────────────────────────────────────────
print("\n── Inflect ──")

p_engine = inflect.engine()
words = ["subjects", "sessions", "runs", "tasks", "vertices", "analyses", "atlases", "categories", "echoes"]

r, p = bench("inflect_singular",
    lambda: [pybids_rs.singularize(w) for w in words],
    lambda: [p_engine.singular_noun(w) or w for w in words],
    n=10000)
match_all = all(str(a) == str(b) for a, b in zip(r, p))
results["inflect_singular"]["match"] = match_all
m = "✅" if match_all else "❌"
print(f"  singularize {len(words)} words: Rust {results['inflect_singular']['rust_ms']:.4f}ms vs Python {results['inflect_singular']['python_ms']:.4f}ms "
      f"→ {results['inflect_singular']['speedup']:.0f}x {m}")
if not match_all:
    for w, rv, pv in zip(words, r, p):
        if str(rv) != str(pv):
            print(f"    DIFF: {w} → rust={rv} py={pv}")

# ─── 10. NIfTI Header ────────────────────────────────────
print("\n── NIfTI Header ──")

try:
    import nibabel as nib
    # Create temp NIfTI
    data = np.zeros((64, 64, 32, 100), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header['pixdim'][4] = 2.0
    nii_path = os.path.join(tempfile.gettempdir(), "bench_test.nii.gz")
    nib.save(img, nii_path)

    r, p = bench("nifti_header",
        lambda: pybids_rs.read_nifti_header(nii_path),
        lambda: {
            "n_vols": int(nib.load(nii_path).header['dim'][4]),
            "tr": float(nib.load(nii_path).header['pixdim'][4]),
        },
        n=100)
    nvols_match = r["n_vols"] == p["n_vols"]
    results["nifti_header"]["match"] = nvols_match
    m = "✅" if nvols_match else "❌"
    print(f"  read header (.nii.gz): Rust {results['nifti_header']['rust_ms']:.2f}ms vs Python {results['nifti_header']['python_ms']:.2f}ms "
          f"→ {results['nifti_header']['speedup']:.1f}x (n_vols={r['n_vols']}) {m}")
except ImportError:
    print("  skipped (nibabel not available)")

# ─── 11. Report Generation ───────────────────────────────
print("\n── Report Generation ──")

if eeg_datasets:
    ds_path = str(EXAMPLES / list(eeg_datasets.keys())[0])
    r, p = bench("generate_report",
        lambda: pybids_rs.generate_report(ds_path),
        lambda: "N/A",  # PyBIDS report requires different API
        n=3)
    print(f"  generate_report: Rust {results['generate_report']['rust_ms']:.1f}ms ({len(r)} chars)")

# ─── 12. Modality-specific datasets ──────────────────────
print("\n── Modality End-to-End ──")

modality_datasets = {
    "eeg": ["eeg_cbm", "eeg_rishikesh", "eeg_face13"],
    "ieeg": ["ieeg_epilepsy", "ieeg_epilepsyNWB", "ieeg_visual_multimodal"],
    "meg": ["ds000246", "ds000247", "ds000248"],
    "pet": ["pet001", "pet006"],
    "perf": ["asl001", "asl002", "asl003", "asl004", "asl005"],
    "func": ["ds001", "ds002", "ds003", "ds005", "ds051"],
    "anat": ["ds004332", "qmri_mtsat"],
    "dwi": ["ds114"],
    "motion": ["motion_dualtask"],
    "micr": ["eeg_ds003645s_hed_demo"],
}

for modality, datasets in modality_datasets.items():
    for ds in datasets:
        ds_path = EXAMPLES / ds
        if not ds_path.exists():
            continue

        name = f"e2e_{modality}_{ds}"
        try:
            r, p = bench(name,
                lambda p=str(ds_path): pybids_rs.BIDSLayout(p, validate=False),
                lambda p=str(ds_path): BIDSLayout(p, validate=False),
                n=3, warmup=0)
            r_subs = r.get_subjects()
            p_subs = p.get_subjects()
            sub_match = sorted(r_subs) == sorted(p_subs)

            r_dts = r.get_datatypes()
            p_dts = p.get_datatypes()
            dt_match = sorted(r_dts) == sorted(p_dts)

            match = sub_match and dt_match
            results[name]["match"] = match
            m = "✅" if match else "❌"
            print(f"  [{modality:>6}] {ds}: Rust {results[name]['rust_ms']:.0f}ms vs Python {results[name]['python_ms']:.0f}ms "
                  f"→ {results[name]['speedup']:.1f}x subs={len(r_subs)} dts={r_dts} {m}")
            if not sub_match:
                print(f"         DIFF subjects: rust={sorted(r_subs)[:3]}... py={sorted(p_subs)[:3]}...")
            if not dt_match:
                print(f"         DIFF datatypes: rust={sorted(r_dts)} py={sorted(p_dts)}")
        except Exception as e:
            print(f"  [{modality:>6}] {ds}: ERROR {e}")

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

total = len(results)
matched = sum(1 for v in results.values() if v.get("match", True))
avg_speedup = sum(v.get("speedup", 1) for v in results.values() if "speedup" in v) / max(1, sum(1 for v in results.values() if "speedup" in v))

print(f"\n  Total benchmarks: {total}")
print(f"  Results matching: {matched}/{total}")
print(f"  Average speedup:  {avg_speedup:.1f}x")
print()

# Breakdown by category
categories = defaultdict(list)
for name, data in results.items():
    cat = name.split("_")[0]
    categories[cat].append(data)

print(f"  {'Category':<20} {'Avg Speedup':>12} {'Match':>8}")
print(f"  {'─' * 20} {'─' * 12} {'─' * 8}")
for cat, items in sorted(categories.items()):
    speeds = [d["speedup"] for d in items if "speedup" in d]
    matches = [d.get("match", True) for d in items]
    avg = sum(speeds) / len(speeds) if speeds else 0
    all_match = all(matches)
    print(f"  {cat:<20} {avg:>10.1f}x  {'✅' if all_match else '❌':>6}")

# Save results
out_path = Path(__file__).parent / "golden" / "benchmark_results.json"
with open(out_path, "w") as f:
    json.dump(dict(results), f, indent=2)
print(f"\n  Results saved to {out_path}")
