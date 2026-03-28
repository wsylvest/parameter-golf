ABOUTME: This file records benchmark results for each phase of competitive improvements.
ABOUTME: It is used by Claude Code to track metrics, deltas, and keep/revert decisions.

# Phase Results

## Summary

| Phase | Lines | Delta | Technique | Decision |
|-------|-------|-------|-----------|----------|
| 0 | 1307 | -180 | Remove dead cache/distill code | KEEP |
| 1 | 1321 | +14 | EMA shadow weights | KEEP |
| 2 | 1320 | -1 | LeakyReLU(0.5)² | KEEP |
| 3 | 1325 | +5 | LN Scale by depth | KEEP |
| 4 | 1332 | +7 | Partial RoPE | KEEP |
| 5 | 1342 | +10 | XSA last N layers | KEEP |
| 6 | 1353 | +11 | GPTQ-lite export | KEEP |
| 7 | -- | -- | Warmdown/QAT | SKIP (already implemented) |
| 8 | 1412 | +59 | Score-First TTT | KEEP |
| 9 | -- | -- | TTT subset | SKIP (tuning, not code) |

## Estimated Combined Impact

| Technique | Est. bpb gain |
|-----------|---------------|
| LeakyReLU(0.5)² | -0.003 |
| LN Scale | -0.002 |
| Partial RoPE (16/64) | -0.003 |
| XSA (last 3-4 layers) | -0.005 |
| EMA (0.997) | -0.005 |
| GPTQ-lite | -0.003 q_gap |
| Score-First TTT | -0.003 |
| **Total estimated** | **-0.024** |

Starting from Run 5 baseline (1.1517), estimated target: **~1.128** (competitive with #2-3).

## Line Budget
- Start: 1487 (HEAD 403ec72)
- Phase 0: 1307 (-180, freed budget)
- Final: 1412 (88 remaining of 1500)
