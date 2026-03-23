# Updating bids-rs for a New BIDS Specification Version

When a new version of the [BIDS specification](https://bids-specification.readthedocs.io/)
is released, follow this checklist to update `bids-rs`.

## Checklist

### 1. Update the version constants

**File:** `crates/bids-schema/src/version.rs`

- [ ] Update `SUPPORTED_BIDS_VERSION` to the new version.
- [ ] If the minimum readable version changed, update `MIN_COMPATIBLE_VERSION`.
- [ ] Add a new `SpecChange` entry to `CHANGELOG` describing what changed
      (new entities, datatypes, suffixes, deprecations, breaking changes).

```rust
// Example: bumping from 1.9.0 to 1.10.0
pub const SUPPORTED_BIDS_VERSION: BidsVersion = BidsVersion::new(1, 10, 0);
```

### 2. Update the schema

**File:** `crates/bids-schema/src/lib.rs`

- [ ] Add new entities to the `entities` Vec in `BidsSchema::built_in()`.
- [ ] Add new datatypes to the `datatypes` HashSet.
- [ ] Add new suffixes to the `suffixes` HashSet.
- [ ] Add new extensions to the `extensions` HashSet.
- [ ] Update/add filename validation patterns in `file_patterns` if needed.

### 3. Update entity config files

**Files:**
- `crates/bids-core/src/configs/bids.json`
- `crates/bids-core/src/configs/derivatives.json`

- [ ] Add regex patterns for any new entities.
- [ ] Add path-building patterns for any new file types.
- [ ] Update existing patterns if entity constraints changed.

### 4. Update the entity ordering

**File:** `crates/bids-core/src/entities.rs`

- [ ] Add new entities to `ENTITY_ORDER` in the correct canonical position.

### 5. Update modality crates (if applicable)

If the new spec version adds or changes modality-specific rules:

- [ ] `bids-eeg` — new channel types, metadata fields, file formats
- [ ] `bids-meg` — new channel types, metadata fields
- [ ] `bids-ieeg` — new electrode types, metadata fields
- [ ] `bids-pet` — new tracer metadata, blood recording fields
- [ ] `bids-perf` — new ASL parameters
- [ ] `bids-nirs` — new optode/channel types
- [ ] `bids-motion` — new tracking system metadata
- [ ] `bids-mrs` — new spectroscopy parameters
- [ ] `bids-micr` — new microscopy modalities/suffixes
- [ ] `bids-beh` — new behavioral data fields

### 6. Update the CLI upgrade command

**File:** `crates/bids-cli/src/cmd/upgrade.rs`

- [ ] Add migration logic for any deprecated fields that need conversion.
- [ ] The version constants are already used dynamically — no hardcoded
      versions to update.

### 7. Run tests

```bash
# All tests must pass
cargo test --workspace --exclude pybids-rs

# Specifically verify the version tracking tests
cargo test -p bids-schema

# The following test will FAIL if you forgot to update CHANGELOG:
# test_supported_version_matches_last_changelog
```

### 8. Update documentation

- [ ] Update the README if the new version adds major features.
- [ ] Update the `bids-schema` crate-level doc comment if the version number
      is mentioned.

## Architecture: How Spec Knowledge Flows

```
                    BIDS Specification (e.g., 1.10.0)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
            bids-schema              bids-core/configs/
          (entities, datatypes,    (regex patterns,
           suffixes, extensions,    path templates)
           validation patterns,     bids.json
           version tracking)        derivatives.json
                    │                    │
                    ├────────┬───────────┤
                    ▼        ▼           ▼
              bids-layout  bids-io    bids-dataset
              (indexing,   (path      (datatype
               querying)   building)   detection)
                    │
                    ▼
              Domain crates (bids-eeg, bids-meg, ...)
              read metadata keys, channel types, etc.
```

**Key principle:** No crate outside `bids-schema` and `bids-core/configs/`
should hardcode entity names, datatype lists, or suffix lists. They should
always reference these two sources of truth.

## Guard Rails

The test suite includes several safeguards:

1. **`test_supported_version_matches_last_changelog`** — Fails if
   `SUPPORTED_BIDS_VERSION` doesn't match the last `CHANGELOG` entry.
   This catches forgetting to add a changelog entry when bumping the version.

2. **`test_changelog_is_sorted`** — Ensures the changelog entries are in
   ascending version order.

3. **`BidsLayout` prints a warning** when a dataset declares a newer BIDS
   version than the library supports, so users know they may be missing
   features.

4. **`BidsLayout::spec_compatibility()`** — Programmatic access to the
   compatibility status for downstream tools that want to handle it
   differently (e.g., fail hard, or log and continue).
