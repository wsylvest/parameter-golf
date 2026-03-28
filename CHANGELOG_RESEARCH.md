ABOUTME: This file documents what changed per phase, why, and observed effects.
ABOUTME: It is used by Claude Code to maintain a research log of technique experiments.

# Changelog

## Phase 0: Clean Neural Baseline
- **Technique:** Dead code removal
- **What changed:** Removed n-gram cache, phrase cache, frozen prefill, distillation (183 lines)
- **Why:** N-gram cache invalidated by competition organizers. A/B test confirmed harmful.
- **Observed effect:** 1307 lines. 193 lines freed. No behavior change (cache was default-off).

## Phase 1: EMA
- **Technique:** Exponential Moving Average shadow weights
- **What changed:** EMA_DECAY env var, CPU shadow state, lerp update every step, EMA>SWA>raw export priority
- **Why:** All top-3 submissions use EMA(0.997). Smoother than periodic SWA.
- **Observed effect:** +14 lines. Needs GPU canary.

## Phase 2: LeakyReLU(0.5)²
- **Technique:** Activation swap
- **What changed:** `torch.relu` → `F.leaky_relu(negative_slope=0.5)` in MLP.forward
- **Why:** #1 submission uses this. Preserves negative gradient flow, eliminates dead neurons.
- **Observed effect:** -1 lines. Needs GPU canary.

## Phase 3: LN Scale
- **Technique:** Depth-based normalization scaling
- **What changed:** LN_SCALE env var. Each block's RMSNorm output scaled by 1/sqrt(layer_idx+1).
- **Why:** Used by top-3. Damps deeper layers, reduces gradient magnitude imbalance.
- **Observed effect:** +5 lines. Needs GPU canary.

## Phase 4: Partial RoPE
- **Technique:** Partial rotary position embeddings
- **What changed:** ROPE_DIMS env var. Only first N dims of each head get rotary; rest attend without positional bias.
- **Why:** Top submission uses 16/64 (25%). Zero parameters added.
- **Observed effect:** +7 lines. Needs GPU canary.

## Phase 5: XSA
- **Technique:** Exclusive Self-Attention
- **What changed:** XSA_LAST_N env var. Last N layers subtract value-aligned component from attention output.
- **Why:** Used by all top-5 submissions. Encourages orthogonal information capture.
- **Observed effect:** +10 lines. Needs GPU canary.

## Phase 6: GPTQ-lite
- **Technique:** Per-row optimal clip search during quantization
- **What changed:** Try 5 clip percentiles per row, pick min MSE. Replaces single fixed percentile.
- **Why:** Used by #2 submission. Zero training cost, better quantized quality.
- **Observed effect:** +11 lines. Needs GPU canary.

## Phase 7: Warmdown/QAT — SKIPPED
- Already implemented via QAT_START_FRAC and WARMDOWN_ITERS. No code change needed.

## Phase 8: Score-First TTT
- **Technique:** Test-time training with score-first no-leakage guarantee
- **What changed:** TTT_LR, TTT_EPOCHS, TTT_CHUNK_TOKENS env vars. Score chunk, then adapt with SGD.
- **Why:** #1 submission's key differentiator (-0.0025 bpb). Legally adapts model at eval time.
- **Observed effect:** +59 lines. Needs GPU canary.

## Phase 9: TTT Subset — SKIPPED
- Tuning decision, not code change. Base TTT adapts all params (matches #1 submission).
