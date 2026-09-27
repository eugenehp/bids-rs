#!/usr/bin/env python3
"""Run one benchmark workload against whichever ``bids`` package is importable.

Invoked as a subprocess by ``bench.py`` with ``sys.path`` arranged so that
``import bids`` resolves either to real PyBIDS or to the bids-rs shim. Emits a
single JSON object on stdout so the driver can compare both runs.
"""

import argparse
import gc
import json
import statistics
import sys
import time
import traceback

import bids
from bids import BIDSLayout

BACKEND = getattr(bids, 'BACKEND', 'pybids')


def timed(fn, repeats):
    """Run ``fn`` ``repeats`` times, returning (best, median, result)."""
    times = []
    result = None
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--repeats', type=int, default=3)
    args = ap.parse_args()

    out = {'backend': BACKEND, 'version': getattr(bids, '__version__', '?'),
           'dataset': args.dataset, 'timings': {}, 'results': {}, 'errors': {}}

    def record(name, fn, repeats=None, capture=None):
        """Time an operation; record its result fingerprint and any error."""
        reps = args.repeats if repeats is None else repeats
        try:
            best, median, value = timed(fn, reps)
            out['timings'][name] = {'best': best, 'median': median, 'repeats': reps}
            if capture is not None:
                out['results'][name] = capture(value)
            return value
        except Exception:  # noqa: BLE001
            out['errors'][name] = traceback.format_exc(limit=3)
            return None

    # ── Indexing ──
    layout = record(
        'index',
        lambda: BIDSLayout(args.dataset, validate=False),
        capture=lambda lay: len(lay.get_files()),
    )
    record(
        'index_no_metadata',
        lambda: BIDSLayout(args.dataset, validate=False,
                           indexer=bids.BIDSLayoutIndexer(validate=False, index_metadata=False)),
        capture=lambda lay: len(lay.get_files()),
    )

    if layout is None:
        json.dump(out, sys.stdout)
        return

    # ── Queries ──
    record('get_subjects', layout.get_subjects, capture=sorted)
    record('get_tasks', layout.get_tasks, capture=sorted)
    record('get_sessions', layout.get_sessions, capture=sorted)
    record('get_datatypes', layout.get_datatypes, capture=sorted)
    record('get_suffixes', layout.get_suffixes, capture=sorted)

    import os

    def relative(paths):
        root = os.path.abspath(args.dataset)
        return sorted(os.path.relpath(str(p), root) for p in paths)

    record(
        'get_all_files',
        lambda: layout.get(return_type='filename'),
        capture=relative,
    )
    record(
        'get_bold',
        lambda: layout.get(suffix='bold', extension='.nii.gz', return_type='filename'),
        capture=relative,
    )
    record(
        'get_events',
        lambda: layout.get(suffix='events', extension='.tsv', return_type='filename'),
        capture=relative,
    )
    record(
        'get_objects',
        lambda: layout.get(suffix='bold'),
        capture=len,
    )
    record(
        'get_by_subject',
        lambda: layout.get(subject=layout.get_subjects()[:1], return_type='filename')
        if layout.get_subjects() else [],
        capture=relative,
    )
    record('get_entities', lambda: list(layout.get_entities().keys()), capture=sorted)

    # ── Metadata ──
    # Sort before slicing: the two backends return files in different orders,
    # so an unsorted slice would compare different files against each other.
    sample = sorted(str(f) for f in
                    layout.get(suffix='bold', extension='.nii.gz', return_type='filename'))[:25]
    if sample:
        record(
            'get_metadata_x25',
            lambda: [layout.get_metadata(f) for f in sample],
            capture=lambda mds: [
                {k: _norm(v) for k, v in sorted(md.items())} for md in mds
            ],
        )

    # ── Entity parsing ──
    all_files = sorted(str(f) for f in layout.get(return_type='filename'))[:200]
    if all_files:
        record(
            'parse_file_entities_x200',
            lambda: [layout.parse_file_entities(f) for f in all_files],
            capture=lambda ents: [
                {k: str(v) for k, v in sorted(e.items())} for e in ents
            ],
        )

    json.dump(out, sys.stdout)


def _norm(v):
    """Normalise a metadata value for cross-backend comparison."""
    if isinstance(v, float) and v == int(v):
        return int(v)
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


if __name__ == '__main__':
    main()
