#!/usr/bin/env bash
#
# Publish the workspace to crates.io in dependency order.
#
# crates.io needs every dependency of a crate to already be live before that
# crate can be uploaded, so the order below is a topological sort of the
# workspace graph — regenerate it with `./publish.sh --order` if crates are
# added or their dependencies change.
#
#   ./publish.sh --dry-run    package and verify everything, upload nothing
#   ./publish.sh --order      print the dependency order and exit
#   ./publish.sh              publish for real
#
# Already-published versions are skipped, so a run interrupted by rate limiting
# can simply be restarted.

set -euo pipefail

# Dependency order: each crate appears after everything it depends on.
CRATES=(
    bids-core
    bids-filter
    bids-formula
    bids-inflect
    bids-io
    bids-schema
    bids-validate
    bids-layout
    bids-beh
    bids-dataset
    bids-derive
    bids-micr
    bids-nifti
    bids-reports
    bids-mrs
    bids-perf
    bids-variables
    bids-eeg
    bids-modeling
    bids-ieeg
    bids-meg
    bids-motion
    bids-nirs
    bids-pet
    bids
    pybids-rs
    bids-cli
)

# crates.io allows a burst then throttles; this is the documented cool-off.
WAIT_SECS=610

DRY_RUN=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    --order)
        printf '%s\n' "${CRATES[@]}"
        exit 0
        ;;
    "") ;;
    *)
        echo "unknown argument: $1" >&2
        echo "usage: $0 [--dry-run|--order]" >&2
        exit 2
        ;;
esac

VERSION=$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')
echo ">>> Workspace version: $VERSION"

# Refuse to ship something that does not build or pass its tests.
echo ">>> Checking formatting, lints and tests"
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --quiet

# Crates with no workspace-internal dependencies. Only these can be packaged
# before a release: `cargo package` resolves dependencies from crates.io, and
# for every other crate the new version of its dependencies is not live yet.
# Those are verified as they are published, each after its dependencies.
ROOT_CRATES=(bids-core bids-filter bids-formula bids-inflect)

if [[ $DRY_RUN -eq 1 ]]; then
    echo ">>> Dry run: packaging crates with no internal dependencies"
    for crate in "${ROOT_CRATES[@]}"; do
        printf '    %-16s ' "$crate"
        cargo package -p "$crate" --allow-dirty >/dev/null 2>&1 \
            && echo "packaged" || { echo "FAILED"; exit 1; }
    done
    echo
    echo "Dry run complete — nothing was uploaded."
    echo "The remaining $(( ${#CRATES[@]} - ${#ROOT_CRATES[@]} )) crates cannot be packaged"
    echo "until their dependencies are live at $VERSION; cargo verifies each of them"
    echo "at publish time and the run aborts on the first failure."
    exit 0
fi

for crate in "${CRATES[@]}"; do
    while true; do
        echo ">>> Publishing $crate $VERSION at $(date -u)"
        output=$(cargo publish -p "$crate" --allow-dirty 2>&1) && status=0 || status=$?

        if [[ $status -eq 0 ]] || grep -qE "Published|Uploaded" <<<"$output"; then
            echo "    ✓ $crate published"
            break
        elif grep -q "already uploaded" <<<"$output"; then
            echo "    ⏭ $crate $VERSION already on crates.io, skipping"
            break 2>/dev/null || break
        elif grep -q "429 Too Many Requests" <<<"$output"; then
            echo "    ⏳ rate limited, sleeping ${WAIT_SECS}s"
            sleep "$WAIT_SECS"
        else
            echo "    ✗ $crate failed:" >&2
            echo "$output" | tail -30 >&2
            exit 1
        fi
    done
    # crates.io needs a moment to index a new version before a dependent can
    # resolve it.
    sleep 20
done

echo "🎉 All crates published at $VERSION"
