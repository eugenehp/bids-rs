#!/usr/bin/env python3
"""Render the PyBIDS speedup comparison as SVG for the README.

Emits a light and a dark file so the README can pair them in a `<picture>`;
the dark version is its own set of steps against the dark surface, not an
inverted copy.

    python make_chart.py /tmp/bench-final2.json ../figures
"""

import argparse
import json
import math
import statistics
from pathlib import Path

# Which group each operation belongs to. The grouping is the story: queries are
# answered from the index, indexing pays for building it.
GROUPS = {
    'get_subjects': 'Entity queries',
    'get_sessions': 'Entity queries',
    'get_tasks': 'Entity queries',
    'get_datatypes': 'Entity queries',
    'get_suffixes': 'Entity queries',
    'get_entities': 'Entity queries',
    'get_by_subject': 'File queries',
    'get_all_files': 'File queries',
    'get_bold': 'File queries',
    'get_events': 'File queries',
    'get_objects': 'File queries',
    'get_metadata_x25': 'Indexing & metadata',
    'parse_file_entities_x200': 'Indexing & metadata',
    'index': 'Indexing & metadata',
    'index_no_metadata': 'Indexing & metadata',
}

LABELS = {
    'get_metadata_x25': 'get_metadata (×25)',
    'parse_file_entities_x200': 'parse_file_entities (×200)',
    'get_by_subject': 'get(subject=…)',
    'get_all_files': 'get(return_type="filename")',
    'get_bold': 'get(suffix="bold")',
    'get_events': 'get(suffix="events")',
    'get_objects': 'get() → BIDSFile objects',
    'index': 'BIDSLayout(…)',
    'index_no_metadata': 'BIDSLayout(index_metadata=False)',
}

ORDER = ['Entity queries', 'File queries', 'Indexing & metadata']

THEMES = {
    'light': {
        'surface': '#fcfcfb',
        'text': '#0b0b0b',
        'secondary': '#52514e',
        'muted': '#898781',
        'grid': '#e1e0d9',
        'axis': '#c3c2b7',
        'series': {'Entity queries': '#2a78d6', 'File queries': '#eb6834',
                   'Indexing & metadata': '#1baf7a'},
    },
    'dark': {
        'surface': '#1a1a19',
        'text': '#ffffff',
        'secondary': '#c3c2b7',
        'muted': '#898781',
        'grid': '#2c2c2a',
        'axis': '#383835',
        'series': {'Entity queries': '#3987e5', 'File queries': '#d95926',
                   'Indexing & metadata': '#199e70'},
    },
}


def collect(path):
    """Speedup statistics per operation, from a bench.py result file."""
    data = json.loads(Path(path).read_text())
    per_op = {}
    datasets = set()
    for entry in data:
        py, rs = entry.get('pybids', {}), entry.get('bids_rs', {})
        if 'timings' not in py or 'timings' not in rs:
            continue
        datasets.add(entry['dataset'])
        for op, t in py['timings'].items():
            if op not in rs['timings'] or op not in GROUPS:
                continue
            best_rs = rs['timings'][op]['best']
            if best_rs <= 0:
                continue
            per_op.setdefault(op, []).append(t['best'] / best_rs)

    rows = [
        {
            'op': op,
            'label': LABELS.get(op, op),
            'group': GROUPS[op],
            'median': statistics.median(s),
            'lo': min(s),
            'hi': max(s),
            'n': len(s),
        }
        for op, s in per_op.items()
    ]
    rows.sort(key=lambda r: -r['median'])
    return rows, len(datasets)


def fmt(v):
    """Format a speedup for a direct label."""
    if v >= 1_000_000:
        return f'{v / 1_000_000:g}M×'
    if v >= 100_000:
        return f'{v / 1000:.0f}k×'
    if v >= 1000:
        return f'{v / 1000:.1f}k×'
    if v >= 100:
        return f'{v:.0f}×'
    if v >= 10:
        return f'{v:.0f}×'
    return f'{v:.1f}×'


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def render(rows, n_datasets, theme_name):
    """Build the SVG for one theme."""
    t = THEMES[theme_name]

    pad_left = 232.0
    pad_top = 106.0
    plot_w = 560.0
    row_h = 26.0
    plot_h = row_h * len(rows)
    pad_right = 78.0
    width = pad_left + plot_w + pad_right
    height = pad_top + plot_h + 56.0

    hi_max = max(r['hi'] for r in rows)
    lo_min = min(r['lo'] for r in rows)
    decades = max(2, math.ceil(math.log10(hi_max)))
    # Extend half a decade below parity when anything dips under it, so a
    # slower-than-PyBIDS minimum shows as such instead of being clamped.
    lo_exp = -0.5 if lo_min < 1.0 else 0.0
    span = decades - lo_exp

    def x_of(v):
        v = max(v, 10.0 ** lo_exp)
        return pad_left + (math.log10(v) - lo_exp) / span * plot_w

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="bids-rs speedup over PyBIDS by operation, log scale">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="{t["surface"]}"/>',
        f'<text x="24" y="34" font-family="system-ui,-apple-system,Segoe UI,sans-serif" '
        f'font-size="17" font-weight="600" fill="{t["text"]}">'
        f'bids-rs vs PyBIDS — speedup by operation</text>',
        f'<text x="24" y="55" font-family="system-ui,-apple-system,Segoe UI,sans-serif" '
        f'font-size="12" fill="{t["secondary"]}">'
        f'Best of 3 runs on {n_datasets} bids-examples datasets. Dot is the median across '
        f'datasets, bar spans min to max. Log scale — higher is faster.</text>',
    ]

    # Legend: identity never rests on colour alone, but name the groups anyway.
    lx = 24.0
    for group in ORDER:
        if not any(r['group'] == group for r in rows):
            continue
        out.append(
            f'<circle cx="{lx + 5:.1f}" cy="72" r="5" fill="{t["series"][group]}"/>'
            f'<text x="{lx + 16:.1f}" y="76" font-family="system-ui,-apple-system,Segoe UI,sans-serif" '
            f'font-size="11" fill="{t["secondary"]}">{esc(group)}</text>'
        )
        lx += 30 + len(group) * 6.1

    # Gridlines, one per decade.
    for d in range(0, decades + 1):
        v = 10 ** d
        x = x_of(v)
        out.append(
            f'<line x1="{x:.1f}" y1="{pad_top:.1f}" x2="{x:.1f}" y2="{pad_top + plot_h:.1f}" '
            f'stroke="{t["grid"]}" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 18:.1f}" text-anchor="middle" '
            f'font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="10" '
            f'fill="{t["muted"]}">{fmt(v) if v > 1 else "1×"}</text>'
        )

    # Parity reference: left of this line bids-rs would be slower.
    out.append(
        f'<line x1="{x_of(1):.1f}" y1="{pad_top - 6:.1f}" x2="{x_of(1):.1f}" '
        f'y2="{pad_top + plot_h + 4:.1f}" stroke="{t["axis"]}" stroke-width="1.5"/>'
        f'<text x="{x_of(1) + 6:.1f}" y="{pad_top - 9:.1f}" '
        f'font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="10" '
        f'fill="{t["muted"]}">parity — left of this line bids-rs is slower</text>'
    )

    for i, r in enumerate(rows):
        cy = pad_top + i * row_h + row_h / 2
        colour = t['series'][r['group']]
        x_lo, x_hi, x_med = x_of(r['lo']), x_of(r['hi']), x_of(r['median'])

        out.append(
            f'<text x="{pad_left - 14:.1f}" y="{cy:.1f}" text-anchor="end" '
            f'dominant-baseline="middle" font-family="ui-monospace,SFMono-Regular,Menlo,monospace" '
            f'font-size="11" fill="{t["secondary"]}">{esc(r["label"])}</text>'
        )
        # Range across datasets, then the median on top.
        out.append(
            f'<line x1="{x_lo:.1f}" y1="{cy:.1f}" x2="{x_hi:.1f}" y2="{cy:.1f}" '
            f'stroke="{colour}" stroke-width="2" stroke-linecap="round" opacity="0.45"/>'
            f'<circle cx="{x_med:.1f}" cy="{cy:.1f}" r="5" fill="{colour}" '
            f'stroke="{t["surface"]}" stroke-width="1.5"/>'
        )
        out.append(
            f'<text x="{pad_left + plot_w + 10:.1f}" y="{cy:.1f}" dominant-baseline="middle" '
            f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" '
            f'font-weight="600" fill="{t["text"]}">{fmt(r["median"])}</text>'
        )

    out.append(
        f'<text x="24" y="{height - 14:.0f}" '
        f'font-family="system-ui,-apple-system,Segoe UI,sans-serif" font-size="10" '
        f'fill="{t["muted"]}">Every operation compared for equality first; all results '
        f'identical between backends.</text>'
    )
    out.append('</svg>')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results')
    ap.add_argument('outdir')
    args = ap.parse_args()

    rows, n = collect(args.results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = outdir / f'pybids-speedup-{theme}.svg'
        path.write_text(render(rows, n, theme))
        print(f'wrote {path}')
    for r in rows:
        print(f'  {r["label"]:<34} {r["median"]:>9.1f}x  ({r["lo"]:.1f}–{r["hi"]:.1f})')


if __name__ == '__main__':
    main()
