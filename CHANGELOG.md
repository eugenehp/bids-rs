# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3]

Full PyBIDS parity. Every crate in the workspace now shares one version — `bids`
had gone out at 0.0.2 while the other 27 sat at 0.0.1 — and several numerical
results changed, so read **Fixed** before upgrading.

### Fixed

Four of these produce different numbers than 0.0.x did. If you have results
computed with an earlier version, they are affected.

- **`glover_hrf` was materially wrong** — its peak was ~17% low. The internal
  `gammaln` used a rational approximation that was only correct at integer
  arguments, which the exact factorial path already short-circuits. Glover's
  dispersion of 0.9 gives a shape of 6/0.9, so every Glover HRF went through
  the broken branch. Replaced with a Lanczos approximation. `spm_hrf`, whose
  shape parameters are integers, was and remains correct.
- **`butter_highpass` returned the filter for `1 - cutoff`.** It applied
  spectral inversion to a low-pass of the same cutoff, which is a high-pass at
  the mirrored frequency. Replaced with the proper low-pass-to-high-pass
  transform; coefficients now match `scipy.signal.butter(..., btype='high')` to
  machine precision.
- **`butter_bandpass` was built by cascading a high-pass and a low-pass.** That
  gives the right filter order but different poles. Replaced with the
  low-pass-to-band-pass transform, matching SciPy.
- **JSON floats were parsed up to 1 ULP off.** `serde_json`'s fast float parser
  is approximate; the `float_roundtrip` feature is now enabled workspace-wide,
  so sidecar values such as `SliceTiming` round-trip bit-exactly.
- **Metadata string values that looked numeric came back as numbers.** A
  sidecar `"DeviceSerialNumber": "35119"` was stored unquoted and re-parsed as
  JSON, yielding the integer `35119`. The value's type is now recorded with it.
- **`Scale` defaulted to doing nothing.** `Demean` and `Rescale` defaulted to
  `false`; PyBIDS defaults both to `true`, so `Scale` is a z-score. It also
  used the population standard deviation where `pandas.Series.std()` uses the
  sample (ddof=1) one.
- **`Orthogonalize` omitted the intercept**, leaving a bias whenever the
  regressors were not mean-centred.
- **`Factor` kept its input variable**; PyBIDS consumes it.
- **Run duration was never read**, so run-level variables had a duration of
  zero and could not be made dense or convolved. It is now derived from the
  associated image's `RepetitionTime` and volume count.
- **Variance inflation factors** used a correlation-sum approximation that is
  only correct when the other predictors are mutually orthogonal. Replaced
  with a proper OLS R².

### Added

- **All 24 PyBIDS transformations.** `Convolve`, `Demean`, `Product`, `Sum`,
  `And`, `Or`, `Not`, `Filter`, `Group`, `Resample`, `ToDense` and `Assign`
  join the twelve that already worked. Five of those previously parsed
  successfully and did nothing.
- **Transformation errors are reported, not swallowed.** `apply_transformations`
  returns `Vec<TransformError>` and `StatsModelsNodeOutput` carries them, so a
  model asking for a transformation the pipeline cannot perform fails visibly
  instead of yielding an untransformed design matrix.
- **Custom transformations** via `TransformerManager::register`.
- **Mixed models** — `VarComp`, `Prior`, `GlmSpec::z()`, `build_variance_components`,
  `set_priors`, and `GlmmSpec` as a PyBIDS-compatible alias. `with_random_effects`
  was previously a no-op that returned `self`.
- **Model diagnostics and reports** (`bids_modeling::report`): contrast
  matrices, per-regressor and per-contrast VIF via the Smith et al. effective
  regressor construction, and a self-contained HTML report with inline SVG
  design-matrix, correlation, timecourse and VIF figures.
- **`bids_core::options`** — package-wide settings mirroring `bids.config`,
  plus `add_config_paths`.
- **Preludes** on 14 crates, and a much broader `bids::prelude`.
- **`LayoutVariables`** extension trait, providing `get_collections` on a layout.
- **`BidsImageFile`** extension trait, providing `get_image`/`get_header` on a
  `BidsFile`.
- **`bids_schema::collect_schema`** for reading published BIDS schema documents;
  network fetching is behind the crate's `http` feature, off by default.
- `BidsLayout::all_files`, `file_count`, `configured_entities`,
  `derivative_names`; `Database::get_database_file`, `reset_database`,
  `reset_session`; `NodeIndex::get_nodes`, `levels`; `bids_core::utils::natural_sort`,
  `splitext`; report helpers `deroot`, `displayify`, `to_alphanum`; and the
  `bids-reports` functions `describe_func_duration`, `get_slice_info`,
  `general_acquisition_info`, `reminder`.
- **Python bindings** grew from 21 to 40 `BIDSLayout` members: `get_nearest`,
  `get_bval`, `get_bvec`, `get_fieldmap`, `get_collections`, `get_files`,
  `get_associations`, `copy_files`, `write_to_file`, `clone_layout`, `__len__`,
  and the `root`, `files`, `entities`, `session` and `derivatives` properties.
- **`SPEC_PARITY.md`** — the specification and PyBIDS parity audit the README
  has always linked to.
- **`bench/`** — a drop-in `bids` package backed by `bids-rs` plus speed and
  precision harnesses that compare it against real PyBIDS.

### Changed

- Every crate now shares the workspace version. `bids` had drifted to 0.0.2
  while the rest sat at 0.0.1, because `bids` and `pybids-rs` carried literal
  versions; both now inherit from the workspace, as do `pybids-rs`'s internal
  dependency pins. 0.0.2 was skipped for the workspace because `bids` 0.0.2 is
  already published and crates.io does not allow a version to be reused.
- `apply_transformations` returns `Vec<TransformError>` instead of `()`.
- `StatsModelsNodeOutput` gained a `transform_errors` field.
- `Term` gained `categorical` and `prior`; use `Term::new` rather than a struct
  literal.
- `BidsLayout::get_entities()` still returns the entities present in the
  dataset; `configured_entities()` returns the full vocabulary, which is what
  PyBIDS' `get_entities()` reports.
- MSRV is declared as 1.85 (required by edition 2024).

### Performance

- **Indexing is up to 4.4× faster on metadata-heavy datasets.** Sidecars were
  re-read and re-parsed once per data file, so a shared task-level JSON was
  parsed hundreds of times; they are now cached per index run. On `ds000117`
  this took indexing from 3.0 s to 0.68 s — from 3× slower than PyBIDS to 4.4×
  faster.
- **`parse_file_entities` is ~85× faster.** Entity definitions were cloned on
  every call, discarding each entity's compiled-regex cache and recompiling
  every pattern. They are now built once per layout.

### Notes

Measured against PyBIDS 0.22.0 on 19 `bids-examples` datasets, with every
result compared for equality before timing: entity queries 1,000–3,000×,
filtered file queries 326×, metadata access 22×, indexing 4.2×. Numerical
agreement is within 1e-12 absolute / 1e-9 relative across 125 checks, with
sidecar floats bit-exact. See [`bench/`](bench/) to reproduce.

## [0.0.2]

Published for the `bids` umbrella crate only; the rest of the workspace stayed
at 0.0.1.

- Added the `bids` umbrella crate README and a benchmark suite with charts.

## [0.0.1]

- Initial release.

[0.0.3]: https://github.com/eugenehp/bids-rs/releases/tag/v0.0.3
[0.0.2]: https://github.com/eugenehp/bids-rs/releases/tag/v0.0.2
[0.0.1]: https://github.com/eugenehp/bids-rs/releases/tag/v0.0.1
