ABOUTME: This file tracks the current implementation plan for parameter-golf competitive improvements.
ABOUTME: It is used by Claude Code to maintain phase-by-phase execution state.

# Phase 0: Clean Neural Baseline

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
