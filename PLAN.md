ABOUTME: This file tracks the current implementation plan for parameter-golf competitive improvements.
ABOUTME: It is used by Claude Code to maintain phase-by-phase execution state.

# Phase 1: Add EMA

## Verified Facts
- SWA already implemented (accumulates checkpoints during warmdown, averages at export)
- Top submissions use EMA(0.997) every step, which provides smoother weight averaging than SWA
- SWA and EMA can coexist — EMA runs every step, SWA samples during warmdown
- EMA state must be on CPU to avoid doubling GPU memory
- 1307 lines currently, 193 available

## Assumptions
- EMA(0.997) decay is the right starting point (matches top submissions)
- EMA weights should be preferred over SWA when both are available (EMA is every-step, SWA is periodic)
- EMA update cost is negligible (CPU copy every step)

## Plan
1. Add EMA_DECAY hyperparameter (default 0.0 = disabled)
2. Initialize EMA shadow dict after model setup
3. Update EMA every step after optimizer.step()
4. At export time: prefer EMA weights over raw weights (SWA still available as alternative)
5. ~15 lines added

## Files to Change
- train_gpt.py — Hyperparameters, training loop, export selection

## Acceptance Criteria
- EMA_DECAY=0 produces identical behavior to current code
- EMA_DECAY=0.997 accumulates shadow weights and uses them at export
- Syntax check passes
- Line count <= 1325

---

# Phase 0: Clean Neural Baseline (COMPLETE)

## Verified Facts
- train_gpt.py at HEAD (403ec72): 1487 lines (13 remaining of 1500 limit)
- Working tree: 1532 lines (32 over limit) from uncommitted eval_only/eval_diag code
- N-gram cache invalidated by competition organizers (improper probability normalization)
- A/B test: phrase cache harmful (-0.29 bpb), order 2 harmful (-1.664 bpb)
- Best H100 run: val_bpb=1.1517 (Run 5, 17L, pure neural)
- Competition SOTA: 1.1194 (LeakyReLU² + TTT + Parallel Muon)

## Assumptions
- Removing cache code does not change neural-only eval results
- Freed line budget needed for phases 1-8 (~8 technique additions)

## Hypothesis
Removing ~130 lines of dead n-gram/phrase/distillation code will bring train_gpt.py under 1400 lines, leaving ~100 lines of budget for 8 competitive technique additions.

## Files to Change
- train_gpt.py — remove cache paths, distillation, prefill, phrase cache; keep eval_only/eval_diag gated

## Tests to Run
- python3 -c "import py_compile; py_compile.compile('train_gpt.py', doraise=True)"
- Verify all env var parsing still works
- Verify eval_val_sliding still callable with NGRAM_CACHE=0

## Acceptance Criteria
- Line count <= 1400
- Syntax check passes
- No cache state allocation in submission path
- eval_only mode still functional

## Rollback Criteria
- Neural eval path broken
- Non-cache functionality accidentally removed
