ABOUTME: This file records benchmark results for each phase of competitive improvements.
ABOUTME: It is used by Claude Code to track metrics, deltas, and keep/revert decisions.

# Phase Results

## Phase 0: Clean Neural Baseline
- Status: COMPLETE
- Baseline: 1487 lines (HEAD 403ec72)
- New: 1307 lines (-180 lines, 193 lines of budget available)
- Delta: -183 deletions, +3 insertions
- Removed: n-gram cache (scoring, tables, primes, constants), phrase cache, frozen prefill, distillation
- Kept: neural_temp (used in sliding eval), eval_val_sliding (pure neural CE), all architecture code
- Neural eval: unchanged (cache was default-off, NGRAM_CACHE=0)
- Artifact size: unchanged (cache was eval-only, never serialized)
- Decision: **KEEP**
