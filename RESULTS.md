ABOUTME: This file records benchmark results for each phase of the Muon batching optimization.
ABOUTME: It is used by Claude Code to track metrics, deltas, and keep/revert decisions.

# Muon Batching Results

## Baseline (Phase 0)
- Architecture: 11L/512d/3xMLP, 26.5M params
- NS calls per step: 66 (1 GPU) or ~8 (8 GPUs, round-robin)
- Shape buckets: 4 unique shapes
- Target: reduce NS calls to 4 per step

### H100 Run Metrics (v2, WD=0.04, 8xH100)
- Steps completed: 5644 in 600s
- Step avg: 106ms
- val_bpb: 1.2107 (sliding window)
- Artifact: 12.8MB

### H100 Run Metrics (v4, WD=0.02, 8xH100)
- Steps completed: 4581 in 600s
- Step avg: ~131ms (600000/4581)
- val_bpb: 1.2099 (sliding window)
- Artifact: 15.1MB
