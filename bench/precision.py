#!/usr/bin/env python3
"""Compare bids-rs against PyBIDS/SciPy numerically, not just structurally.

`bench.py` measures speed and checks that results are equal. This measures
*how* equal: for every numeric surface both implementations share, it reports
the maximum absolute error, the maximum relative error, and the worst distance
in ULPs (units in the last place — 0 means bit-identical).

Real PyBIDS and `pybids_rs` are imported side by side here; the shim is not
involved, so nothing can mask a difference.

    python precision.py            # all checks
    python precision.py --json out.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pybids_rs
import scipy.signal as sps
from bids.modeling import hrf as py_hrf

EXAMPLES = Path.home() / '.cache' / 'bids-rs' / 'bids-examples'


def ulp_distance(a, b):
    """Distance between two floats in units in the last place."""
    if a == b:
        return 0
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return math.inf
    ia = np.float64(a).view(np.int64).item()
    ib = np.float64(b).view(np.int64).item()
    # Map to a monotonic ordering across the sign boundary.
    if ia < 0:
        ia = np.int64(np.uint64(0x8000000000000000)).item() - ia
    if ib < 0:
        ib = np.int64(np.uint64(0x8000000000000000)).item() - ib
    return abs(ia - ib)


def compare(name, reference, actual, results):
    """Record the error between a reference array and ours."""
    ref = np.asarray(reference, dtype=np.float64).ravel()
    act = np.asarray(actual, dtype=np.float64).ravel()
    if ref.shape != act.shape:
        results.append(
            {'check': name, 'status': 'SHAPE', 'detail': f'{ref.shape} vs {act.shape}'}
        )
        return

    abs_err = np.abs(ref - act)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = np.where(ref != 0, abs_err / np.abs(ref), 0.0)
    ulps = [ulp_distance(float(r), float(a)) for r, a in zip(ref, act)]
    finite_ulps = [u for u in ulps if math.isfinite(u)]

    results.append(
        {
            'check': name,
            'n': int(ref.size),
            'max_abs': float(abs_err.max()) if ref.size else 0.0,
            'max_rel': float(np.nanmax(rel_err)) if ref.size else 0.0,
            'max_ulp': max(finite_ulps) if finite_ulps else 0,
            'exact': bool(np.array_equal(ref, act)),
            'within_tol': bool(np.allclose(ref, act, rtol=1e-9, atol=1e-12)),
        }
    )


def check_hrf(results):
    """SPM and Glover HRFs against PyBIDS' own implementations."""
    for tr in (0.5, 1.0, 2.0, 3.0):
        for ov in (16, 50):
            compare(
                f'spm_hrf(tr={tr}, ov={ov})',
                py_hrf.spm_hrf(tr, oversampling=ov, time_length=32.0),
                pybids_rs.spm_hrf(tr, ov, 32.0, 0.0),
                results,
            )
            compare(
                f'glover_hrf(tr={tr}, ov={ov})',
                py_hrf.glover_hrf(tr, oversampling=ov, time_length=32.0),
                pybids_rs.glover_hrf(tr, ov, 32.0, 0.0),
                results,
            )


def check_filters(results):
    """Butterworth design and filtering against SciPy."""
    for order in (1, 2, 3, 4, 5):
        for cutoff in (0.1, 0.2, 0.3, 0.5, 0.8):
            b_ref, a_ref = sps.butter(order, cutoff, btype='low')
            b, a = pybids_rs.butter_lowpass(order, cutoff)
            compare(f'butter_lowpass(o={order}, c={cutoff}) b', b_ref, b, results)
            compare(f'butter_lowpass(o={order}, c={cutoff}) a', a_ref, a, results)

            b_ref, a_ref = sps.butter(order, cutoff, btype='high')
            b, a = pybids_rs.butter_highpass(order, cutoff)
            compare(f'butter_highpass(o={order}, c={cutoff}) b', b_ref, b, results)
            compare(f'butter_highpass(o={order}, c={cutoff}) a', a_ref, a, results)

    for order in (2, 3, 4):
        lo, hi = 0.1, 0.4
        b_ref, a_ref = sps.butter(order, [lo, hi], btype='band')
        b, a = pybids_rs.butter_bandpass(order, lo, hi)
        compare(f'butter_bandpass(o={order}) b', b_ref, b, results)
        compare(f'butter_bandpass(o={order}) a', a_ref, a, results)

    rng = np.random.default_rng(0)
    sig = rng.standard_normal(500)
    for order, cutoff in ((3, 0.3), (5, 0.2)):
        b, a = sps.butter(order, cutoff)
        compare(
            f'filtfilt(o={order}, c={cutoff})',
            sps.filtfilt(b, a, sig),
            pybids_rs.filtfilt(list(b), list(a), list(sig)),
            results,
        )


def check_metadata(results, limit=40):
    """Sidecar float values, read by both implementations from the same files.

    Any difference here is a JSON parsing difference, so the bar is bit
    equality rather than a tolerance.
    """
    from bids import BIDSLayout as PyLayout

    datasets = sorted(
        p.name
        for p in EXAMPLES.iterdir()
        if (p / 'dataset_description.json').exists()
    )[:limit]

    total = mismatched = 0
    worst = []
    type_diffs = []

    for name in datasets:
        root = str(EXAMPLES / name)
        try:
            py = PyLayout(root, validate=False)
            rs = pybids_rs.BIDSLayout(root, validate=False)
        except Exception:  # noqa: BLE001
            continue

        files = sorted(str(f) for f in py.get(return_type='filename'))[:40]
        for f in files:
            try:
                md_py = dict(py.get_metadata(f))
                md_rs = dict(rs.get_metadata(f))
            except Exception:  # noqa: BLE001
                continue
            for key in set(md_py) & set(md_rs):
                a, b = md_py[key], md_rs[key]
                if type(a) is not type(b) and not (
                    isinstance(a, (int, float)) and isinstance(b, (int, float))
                ):
                    type_diffs.append(f'{name}/{key}: {type(a).__name__} vs {type(b).__name__}')
                    continue
                for va, vb in _pairs(a, b):
                    total += 1
                    if va != vb:
                        mismatched += 1
                        worst.append((ulp_distance(va, vb), f'{name}/{key}', va, vb))

    worst.sort(reverse=True)
    results.append(
        {
            'check': 'sidecar float round-trip',
            'n': total,
            'max_abs': 0.0 if not worst else abs(worst[0][2] - worst[0][3]),
            'max_rel': 0.0,
            'max_ulp': 0 if not worst else worst[0][0],
            'exact': mismatched == 0 and not type_diffs,
            'detail': (
                f'{mismatched}/{total} values differ'
                + (f'; {len(type_diffs)} type differences' if type_diffs else '')
            ),
            'examples': [f'{w[1]}: {w[2]!r} vs {w[3]!r}' for w in worst[:5]] + type_diffs[:5],
            'datasets': len(datasets),
        }
    )


def _pairs(a, b):
    """Yield comparable float pairs from two metadata values."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        yield float(a), float(b)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for x, y in zip(a, b):
            yield from _pairs(x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json')
    ap.add_argument('--skip-metadata', action='store_true')
    args = ap.parse_args()

    results = []
    check_hrf(results)
    check_filters(results)
    if not args.skip_metadata:
        check_metadata(results)

    print(f'{"check":<42} {"n":>6} {"max abs":>12} {"max rel":>12} {"ulp":>6}  verdict')
    print('-' * 92)
    worst_ulp = 0
    failures = 0
    for r in results:
        if r.get('status') == 'SHAPE':
            print(f'{r["check"]:<42} {"—":>6} {"SHAPE MISMATCH: " + r["detail"]}')
            failures += 1
            continue
        # ULP distance explodes across values that straddle zero, so the
        # verdict rests on absolute and relative error; ULP stays informative.
        # numpy's allclose convention: a value agrees if
        # |err| <= atol + rtol*|ref|. A pure relative test is meaningless where
        # the reference is zero, which is exactly where coefficients cancel.
        if r['exact']:
            verdict = 'exact'
        elif r.get('within_tol'):
            verdict = 'close'
        else:
            verdict = 'DIFF'
        if verdict == 'DIFF':
            failures += 1
        worst_ulp = max(worst_ulp, r['max_ulp'] if math.isfinite(r['max_ulp']) else 0)
        print(
            f'{r["check"]:<42} {r["n"]:>6} {r["max_abs"]:>12.3e} '
            f'{r["max_rel"]:>12.3e} {r["max_ulp"]:>6}  {verdict}'
        )
        if r.get('detail'):
            print(f'{"":<42} {r["detail"]}')
        for ex in r.get('examples', [])[:5]:
            print(f'{"":<44} {ex}')

    print()
    exact = sum(1 for r in results if r.get('exact'))
    close = sum(1 for r in results if not r.get('exact') and r.get('within_tol'))
    print(f'{exact} bit-exact, {close} within 1e-12 abs / 1e-9 rel, '
          f'{failures} beyond tolerance, of {len(results)} checks')

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f'Written to {args.json}')


if __name__ == '__main__':
    main()
