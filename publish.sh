#!/bin/bash

CRATES=(
    pybids-rs
    bids-e2e-tests
)

WAIT_SECS=610

for crate in "${CRATES[@]}"; do
    while true; do
        echo ">>> Publishing $crate at $(date -u)"
        output=$(cargo publish -p "$crate" --allow-dirty 2>&1)
        echo "$output" | tail -5

        if echo "$output" | grep -q "Published\|Uploaded"; then
            echo "✓ $crate done"
            break
        elif echo "$output" | grep -q "already uploaded"; then
            echo "⏭ $crate already published, skipping"
            break
        elif echo "$output" | grep -q "429 Too Many Requests"; then
            echo "⏳ Rate limited. Sleeping ${WAIT_SECS}s..."
            sleep "$WAIT_SECS"
        else
            echo "✗ Failed for another reason."
            echo "$output"
            exit 1
        fi
    done
    echo "--- Waiting ${WAIT_SECS}s before next publish ---"
    sleep "$WAIT_SECS"
done

echo "🎉 All crates published!"
