#!/usr/bin/env bash
# =============================================================================
# Parameter Golf — 3-Seed Validation Run
# =============================================================================
# Runs the SOTA baseline config with 3 seeds for submission-ready statistics.
# Records require p < 0.01, typically 3 seeds suffice.
#
# Usage:
#   bash experiments/run_3seed_validation.sh
# =============================================================================

set -euo pipefail

echo "=== 3-Seed Validation ==="
echo "Starting at: $(date)"
echo ""

for SEED in 1337 42 2025; do
    echo "--- Seed ${SEED} ---"
    SEED=${SEED} bash experiments/run_sota_baseline.sh
    echo ""
done

echo "=== All 3 seeds complete ==="
echo "Grep results:"
grep "final_sliding_window_exact\|ttt_exact\|final_int8_zlib_roundtrip_exact" logs/sota_baseline_s*.txt 2>/dev/null || echo "(check logs manually)"
