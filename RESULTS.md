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

## Phase 2: LeakyReLU(0.5)²
- Status: COMPLETE
- Change: `torch.relu(x)` → `F.leaky_relu(x, negative_slope=0.5)` in MLP.forward
- Lines: 1320 (-1 from removed comment)
- Rationale: #1 submission uses this, worth -0.003 bpb. Preserves negative gradient flow, eliminates dead neurons.
- Decision: **KEEP**

## Phase 1: Add EMA
- Status: COMPLETE
- Baseline: 1307 lines (Phase 0)
- New: 1321 lines (+14)
- Added: EMA_DECAY env var (default 0.0 = disabled), shadow weight tracking, export preference EMA > SWA > raw
- EMA update: `ema[n].lerp_(param.cpu(), 1 - decay)` every step after optimizer
- EMA init: copies model state_dict to CPU at training start
- Export: EMA weights loaded into model before quantization/serialization
- Decision: **KEEP**
