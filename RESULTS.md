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

## Batched Muon Implementation (Phases 1-3)
- NS calls per step: reduced from 66 (1GPU) / ~8 (8GPU) to ~4 (shape buckets)
- Implementation: shape-bucket collection + torch.bmm batched NS + torch.compile
- Lines added: +34 (from 1427 to 1461)
- Phase 4 (correctness) and Phase 5 (throughput): pending GPU validation
- Test file: tests/test_batched_ns.py (needs torch to run)
- Code uploaded to S3 network volume
