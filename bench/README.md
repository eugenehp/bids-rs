# Benchmarking bids-rs against PyBIDS

A drop-in `bids` package backed by the Rust engine, and two harnesses that
compare it with real PyBIDS: one for speed, one for numerical precision.

## Layout

| Path | What it is |
|------|------------|
| `shim/bids/` | A `bids` package mirroring PyBIDS' module structure one-to-one, backed by `pybids_rs`. Ahead of PyBIDS on `sys.path`, `import bids` resolves here. |
| `runner.py` | One benchmark workload. Runs against whichever `bids` is importable and prints JSON. |
| `bench.py` | Runs `runner.py` twice per dataset in separate subprocesses — real PyBIDS, then the shim — compares results, then reports timings. |
| `precision.py` | Numerical agreement with PyBIDS and SciPy: max absolute error, max relative error, worst ULP distance. |

The shim exposes `bids.BACKEND` so a run can say which implementation it used.

## Setup

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install pybids
VIRTUAL_ENV=$PWD/.venv maturin build --release -m ../crates/pybids-rs/Cargo.toml --out /tmp/wheel
VIRTUAL_ENV=$PWD/.venv uv pip install --reinstall /tmp/wheel/*.whl
```

Datasets come from `bids-examples`, which the Rust e2e fixtures download to
`~/.cache/bids-rs/bids-examples`.

## Running

```bash
./.venv/bin/python bench.py                 # default dataset set
./.venv/bin/python bench.py ds001 ds005     # named datasets
./.venv/bin/python bench.py --all           # every example dataset
./.venv/bin/python precision.py             # numerical agreement
```

## Reading the output

`bench.py` marks every operation `ok` or `DIFF`. **A speedup on a `DIFF` row is
not a speedup** — correctness is checked first, and the run ends with either
`All compared results identical between backends` or a list of mismatches.

Two things the harness itself got wrong at first, both fixed, both worth
remembering when extending it:

- The two backends return files in different orders, so anything sliced
  (`[:200]`) must be sorted first or it compares different files.
- ULP distance explodes across values that straddle zero, so `precision.py`
  judges agreement on `|err| <= atol + rtol*|ref|` and reports ULPs as
  information only.
