# Specification and PyBIDS parity

Two audits live here:

1. **[BIDS specification coverage](#bids-specification-coverage)** — what
   `bids-rs` supports across the 14 BIDS datatypes, against BIDS v1.9.0.
2. **[PyBIDS parity](#pybids-parity)** — module-by-module correspondence with
   [PyBIDS](https://github.com/bids-standard/pybids), including the places
   where the two deliberately differ.

Target spec version is declared in code as
`bids_schema::SUPPORTED_BIDS_VERSION` (1.9.0), with datasets back to
`MIN_COMPATIBLE_VERSION` (1.4.0) accepted.

## BIDS specification coverage

All 14 datatypes in the built-in schema (`bids_schema::BidsSchema::built_in`)
are indexable and queryable through `BidsLayout`; the table records what each
datatype gets *beyond* indexing.

| Datatype | Crate | Typed metadata | Sidecar tables | Raw data reading |
|----------|-------|----------------|----------------|------------------|
| `anat` | `bids-nifti` | — | — | NIfTI-1/2 (`.nii`, `.nii.gz`), CIFTI, GIFTI |
| `func` | `bids-nifti` | — | — | NIfTI-1/2, incl. volume count and TR |
| `dwi` | `bids-io` | — | `.bval`, `.bvec` via `GradientTable` | NIfTI-1/2 |
| `fmap` | `bids-layout` | — | — | NIfTI-1/2; `IntendedFor` resolution via `get_fieldmap` |
| `perf` | `bids-perf` | `PerfMetadata` | `aslcontext`, `m0scan` | NIfTI-1/2 |
| `pet` | `bids-pet` | `PetMetadata` | blood `.tsv` | NIfTI-1/2 |
| `mrs` | `bids-mrs` | `MrsMetadata` | — | NIfTI-1/2 (NIfTI-MRS) |
| `eeg` | `bids-eeg` | `EegMetadata` | `channels`, `electrodes`, `events`, `coordsystem` | EDF, BDF, BrainVision, EEGLAB `.set` |
| `ieeg` | `bids-ieeg` | `IeegMetadata` | `channels`, `electrodes`, `events` | EDF, BrainVision |
| `meg` | `bids-meg` | `MegMetadata` | `channels`, `headshape` | FIFF, CTF headers |
| `nirs` | `bids-nirs` | `NirsMetadata` | `channels`, `optodes` | SNIRF |
| `motion` | `bids-motion` | `MotionMetadata` | `channels` | `.tsv` time series |
| `micr` | `bids-micr` | OME metadata | — | TIFF, TIFF stacks |
| `beh` | `bids-beh` | — | `events`, `beh` | `.tsv` via `BehLayout` |

Cross-cutting spec features:

| Feature | Status | Where |
|---------|--------|-------|
| Entity parsing from filenames | Supported | `bids_core::entities::parse_file_entities` |
| JSON sidecar inheritance | Supported | `bids_layout::BidsLayout::get_metadata` |
| Path construction from entities | Supported | `bids_io::build_path` |
| `dataset_description.json` | Supported | `bids_core::DatasetDescription` |
| Derivatives, `GeneratedBy`, pipeline scoping | Supported | `bids_layout::Scope`, `bids_core::PipelineDescription` |
| `participants.tsv` / `sessions.tsv` / `scans.tsv` | Supported | `bids_variables::load_variables` |
| `physio` / `stim` `.tsv.gz` | Supported | `bids_variables::load_variables` |
| HED tags | Parsed | `bids_core::hed` |
| Genetic descriptor | Parsed | `bids_core::genetic` |
| BIDS-StatsModels | Supported | `bids-modeling` |
| Schema validation, version compatibility | Supported | `bids-schema`, `bids-validate` |

## PyBIDS parity

Audited against the vendored PyBIDS checkout in `pybids/`.

### Module mapping

| PyBIDS | bids-rs | Status |
|--------|---------|--------|
| `layout/layout.py` (`BIDSLayout`) | `bids-layout` | Complete |
| `layout/index.py` (`BIDSLayoutIndexer`) | `bids_layout::LayoutBuilder` | Complete |
| `layout/db.py` (`ConnectionManager`) | `bids_layout::Database` | Complete |
| `layout/models.py` | `bids-core` (`BidsFile`, `Entity`, `Config`, `PaddedInt`) | Complete |
| `layout/utils.py` | `bids_core::utils`, `bids_core::options::add_config_paths` | Complete |
| `layout/validation.py` | `bids-validate` | Complete |
| `layout/writing.py` | `bids_io::build_path`, `bids_io::write_to_file` | Complete |
| `variables/*` | `bids-variables` | Complete |
| `modeling/hrf.py` | `bids_modeling::hrf` | Complete (all 7 functions) |
| `modeling/statsmodels.py` | `bids_modeling::{graph, node}` | Complete |
| `modeling/auto_model.py` | `bids_modeling::auto_model` | Complete |
| `modeling/transformations/` | `bids_modeling::transformations` | Complete (24/24) |
| `modeling/model_spec.py` | `bids_modeling::spec` | Complete |
| `modeling/report/` | `bids_modeling::report` | Complete; SVG figures instead of Altair |
| `reports/*` | `bids-reports` | Complete |
| `config.py` | `bids_core::options` | Complete |
| `utils.py` | `bids_core::utils` (incl. `natural_sort`, `splitext`) | Complete |
| `layout/db.py` lifecycle | `Database::{get_database_file, reset_database, reset_session}` | Complete |
| `exceptions.py` | `bids_core::BidsError` | Complete |
| `external/inflect.py` | `bids-inflect` | Complete |
| `cli.py` | `bids-cli` | Superset |

### Transformations

All 24 PyBIDS transformations are implemented and dispatched; see
`bids_modeling::transformations::SUPPORTED_TRANSFORMATIONS`.

`Assign`, `Copy`, `Delete`, `DropNA`, `Factor`, `Filter`, `Group`, `Rename`,
`Replace`, `Select`, `Split`, `ToDense`, `Resample`, `Convolve`, `Demean`,
`Orthogonalize`, `Product`, `Scale`, `Sum`, `Threshold`, `And`, `Or`, `Not`,
`Lag` — plus two non-PyBIDS extensions, `Concatenate` and `Power`.

An instruction that cannot be applied produces a `TransformError` rather than
being skipped, and those errors are carried on `StatsModelsNodeOutput`. A model
that asks for a transformation the pipeline cannot perform therefore fails
visibly instead of silently producing an untransformed design matrix.

### Naming map

`bids-rs` drops PyBIDS' `BIDS` prefix and Python-isms. These are renames, not
gaps:

| PyBIDS | bids-rs |
|--------|---------|
| `BIDSStatsModelsGraph` / `Node` / `NodeOutput` | `StatsModelsGraph` / `StatsModelsNode` / `StatsModelsNodeOutput` |
| `BIDSVariableCollection`, `BIDSRunVariableCollection` | `VariableCollection`, `RunVariableCollection` |
| `BIDSVariable` | `SimpleVariable` / `SparseRunVariable` / `DenseRunVariable` |
| `ConnectionManager` | `Database` |
| `BIDSLayoutIndexer` | `LayoutBuilder` |
| `BIDSDataFile` / `BIDSImageFile` / `BIDSJSONFile` | `BidsFile` + `FileType`; images via the `BidsImageFile` trait |
| `DerivativeDatasets.get_pipeline` | `BidsLayout::get_derivative` |
| `Entity.unique` / `.count` | `BidsLayout::get_entity_values` |
| `bids_sort`, `entity_indices` | `sort_entities` |
| `make_bidsfile` | `FileType` |
| `convert_JSON` | `convert_json_keys` |
| `merge_variables` | `merge_simple`, `merge_sparse` |
| `BIDSVariable.clone` / `.apply` | `clone_with` / `apply_grouped` |
| `ModelSpec.from_df`, `GLMMSpec.from_df` | `GlmSpec::from_rows` |
| `validate_model`, `run_graph` | `StatsModelsGraph::validate`, `::run` |
| 24 transformation classes | 24 `Instruction` variants + dispatch arms |
| 9 exception classes | `BidsError` variants |
| `cli.py` commands | `bids-cli` subcommands |

Not ported, by design: `listify`, `hashablefy`, `frozendict`,
`validate_multiple` (the type system covers these); `Sentinel` /
`RemovedOption` (PyBIDS deprecation machinery); `LayoutInfo`, `Tag`,
`FileAssociation` (SQLAlchemy ORM row types — `rusqlite` is used directly);
`Transformation` / `TransformationOutput` (Python ABCs; dispatch is a `match`);
`melt_dm` (a pandas reshape for Altair, irrelevant to the SVG figures here).

### Deliberate differences

| Area | PyBIDS | bids-rs | Why |
|------|--------|---------|-----|
| `And`/`Or`/`Not` on NaN | NaN is truthy (pandas) | NaN is false | Pandas' truthiness silently flips a conjunction over missing data |
| `Lag` interpolation | Cubic spline (`scipy.ndimage.shift`, order 3) | Exact for integer shifts, linear for fractional | Integer shifts are the common case and are exact either way |
| `Convolve` output names | Suffix taken by splitting on the last `_` | `<name>_derivative`, `<name>_dispersion` | PyBIDS' rule mangles variable names that already contain `_` |
| `GLMMSpec.set_priors` | Raises `NotImplementedError` | Implemented | No reason to ship the stub |
| `node_report` | Run-level nodes only | Every level | A collinear group-level design is worth catching too |
| Design matrix figure | One global colour scale | Per-regressor scale, plus small-multiple time courses | An intercept beside an HRF regressor on one scale renders the interesting columns as noise |
| Report rendering | Altair/Vega, matplotlib | Self-contained inline SVG | No runtime dependency; the report is one file |
| Variables | pandas `DataFrame` | `SimpleVariable` / `SparseRunVariable` / `DenseRunVariable` | Typed, no pandas dependency |

### Python bindings

`pybids-rs` exposes `BIDSLayout` and `BIDSFile` to Python via PyO3. The layout
covers the PyBIDS public surface: `get`, `get_file`, `get_files`,
`get_metadata`, `get_dataset_description`, `get_associations`, `get_bval`,
`get_bvec`, `get_fieldmap`, `get_nearest`, `get_tr`, `get_collections`,
`get_entities`, `get_entity_values`, `get_subjects`/`get_sessions`/`get_tasks`/
`get_runs`/`get_datatypes`/`get_suffixes`, `parse_file_entities`, `build_path`,
`copy_files`, `write_to_file`, `add_derivatives`, `to_df`, `save`, `load`,
`clone_layout`, `__len__`, and the `root`, `files`, `entities`, `session` and
`derivatives` properties — 40 members in all.

The one PyBIDS member not bridged is the `config` property, which exposes the
layout's entity-configuration objects; `bids_core::Config` covers it on the
Rust side but `BidsLayout` does not currently hand its configs back out.

### Extensibility

`TransformerManager::register` takes a boxed handler, so a custom
transformation can be added without touching this crate. Registered handlers
are consulted before the built-in table, which also lets them override a
built-in — the same precedence PyBIDS gives its registry.

### Schema documents

`bids_schema::collect_schema` reads a published BIDS schema from a local path,
a URL, or a version string (`"1.9.0"`, `"latest"`, `"stable"`). Network access
is behind the crate's `http` feature, off by default, so the dependency is
opt-in; local paths always work.

### Verification

Numerical agreement with SciPy and PyBIDS is checked against committed golden
data in `tests/golden/precision.json` (regenerate with
`tests/generate_golden.py`, which requires `pybids` and `scipy`). Coverage is
Butterworth filter design, `lfilter`, `filtfilt`, and the SPM/Glover HRFs and
their derivatives, to <1e-10 relative error.

There is no shared dataset fixture between the two implementations — the
`pybids/bids-examples` submodule is unpopulated — so layout, variable and
transformation behaviour is verified by the Rust test suite rather than by
differential testing against PyBIDS.
