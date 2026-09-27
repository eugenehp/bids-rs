#!/usr/bin/env python3
"""Benchmark the bids-rs shim against real PyBIDS.

Runs `runner.py` twice per dataset in separate subprocesses — once with real
PyBIDS on `sys.path`, once with the shim ahead of it — then compares the
results for equivalence before reporting timings. A speedup on a wrong answer
is not a speedup, so correctness is checked first and disagreements are
reported per operation.

    python bench.py                      # default dataset set
    python bench.py ds001 ds002          # named datasets
    python bench.py --all                # every example dataset
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIM = HERE / 'shim'
PYTHON = HERE / '.venv' / 'bin' / 'python'
EXAMPLES = Path.home() / '.cache' / 'bids-rs' / 'bids-examples'

#: Datasets chosen to span modalities and sizes.
DEFAULT_DATASETS = [
    'ds001', 'ds002', 'ds005', 'ds006', 'ds008', 'ds011', 'ds051',
    'ds101', 'ds102', 'ds105', 'ds107', 'ds109', 'ds114', 'ds116',
    'eeg_ds000117', 'ieeg_visual', 'pet005', 'synthetic', 'ds000117',
]


def run(dataset: Path, use_shim: bool, repeats: int) -> dict:
    """Run the workload in a subprocess against one backend."""
    env = dict(os.environ)
    env['PYTHONPATH'] = str(SHIM) if use_shim else ''
    proc = subprocess.run(
        [str(PYTHON), str(HERE / 'runner.py'), str(dataset), '--repeats', str(repeats)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        return {'fatal': proc.stderr.strip()[-600:]}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {'fatal': f'unparseable output: {proc.stdout[:300]}'}


def compare(a, b):
    """Compare two captured results, returning None if equal or a diff string."""
    if a == b:
        return None
    if isinstance(a, list) and isinstance(b, list):
        only_py = [x for x in a if x not in b]
        only_rs = [x for x in b if x not in a]
        parts = []
        if len(a) != len(b):
            parts.append(f'{len(a)} vs {len(b)}')
        if only_py or only_rs:
            # Report the first differing key rather than dumping whole records.
            parts.append(_first_key_diff(only_py, only_rs))
        return '; '.join(p for p in parts if p) or 'ordering differs'
    return f'{str(a)[:80]} vs {str(b)[:80]}'


def _first_key_diff(only_py, only_rs):
    """Name the first key whose value differs, rather than dumping records."""
    for pa, pb in zip(only_py, only_rs):
        if isinstance(pa, dict) and isinstance(pb, dict):
            for k in sorted(set(pa) | set(pb)):
                va, vb = pa.get(k, '<missing>'), pb.get(k, '<missing>')
                if va != vb:
                    return (f'{k}: {type(va).__name__} {str(va)[:40]!r} vs '
                            f'{type(vb).__name__} {str(vb)[:40]!r}')
    return f'{str(only_py[:1])[:60]} vs {str(only_rs[:1])[:60]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('datasets', nargs='*', default=None)
    ap.add_argument('--all', action='store_true', help='every example dataset')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--json', help='write the full result set here')
    args = ap.parse_args()

    if not EXAMPLES.is_dir():
        sys.exit(f'bids-examples not found at {EXAMPLES}')

    if args.all:
        names = sorted(
            p.name for p in EXAMPLES.iterdir()
            if (p / 'dataset_description.json').exists()
        )
    else:
        names = args.datasets or DEFAULT_DATASETS

    all_results = []
    mismatches = []
    op_totals = {}

    print(f'{"dataset":<18} {"op":<26} {"pybids":>10} {"bids-rs":>10} {"speedup":>9}  check')
    print('-' * 88)

    for name in names:
        ds = EXAMPLES / name
        if not (ds / 'dataset_description.json').exists():
            print(f'{name:<18} SKIP (not found)')
            continue

        py = run(ds, use_shim=False, repeats=args.repeats)
        rs = run(ds, use_shim=True, repeats=args.repeats)
        all_results.append({'dataset': name, 'pybids': py, 'bids_rs': rs})

        if 'fatal' in py or 'fatal' in rs:
            who = 'pybids' if 'fatal' in py else 'bids-rs'
            print(f'{name:<18} FATAL ({who}): {(py.get("fatal") or rs.get("fatal"))[:50]}')
            continue

        for op in py.get('timings', {}):
            if op not in rs.get('timings', {}):
                print(f'{name:<18} {op:<26} {"—":>10} {"missing":>10}')
                continue
            t_py = py['timings'][op]['best']
            t_rs = rs['timings'][op]['best']
            speedup = t_py / t_rs if t_rs > 0 else float('inf')

            diff = None
            if op in py.get('results', {}) and op in rs.get('results', {}):
                diff = compare(py['results'][op], rs['results'][op])
            mark = 'ok' if diff is None else 'DIFF'
            if diff is not None:
                mismatches.append((name, op, diff))

            op_totals.setdefault(op, []).append(speedup)
            print(
                f'{name:<18} {op:<26} {t_py * 1000:>9.1f}m {t_rs * 1000:>9.1f}m '
                f'{speedup:>8.1f}x  {mark}'
            )

        for op, err in {**py.get('errors', {}), **rs.get('errors', {})}.items():
            last = err.strip().splitlines()[-1][:60]
            print(f'{name:<18} {op:<26} ERROR: {last}')

    # ── Summary ──
    print()
    print('=' * 88)
    print(f'{"operation":<30} {"datasets":>9} {"median":>10} {"min":>9} {"max":>9}')
    print('-' * 88)
    import statistics

    for op, speeds in sorted(op_totals.items(), key=lambda kv: -statistics.median(kv[1])):
        print(
            f'{op:<30} {len(speeds):>9} {statistics.median(speeds):>9.1f}x '
            f'{min(speeds):>8.1f}x {max(speeds):>8.1f}x'
        )

    print()
    if mismatches:
        print(f'RESULT MISMATCHES: {len(mismatches)}')
        for name, op, diff in mismatches[:25]:
            print(f'  {name} / {op}: {diff}')
    else:
        print('All compared results identical between backends.')

    if args.json:
        Path(args.json).write_text(json.dumps(all_results, indent=2))
        print(f'\nFull results written to {args.json}')


if __name__ == '__main__':
    main()
