ABOUTME: This file logs issues discovered during phase execution but not fixed in that phase.
ABOUTME: It is used by Claude Code to track technical debt and deferred work.

# Issues Found

## From A/B Testing (pre-Phase 0)
- Order 2 (bigrams) hurts by -1.664 bpb due to concentration=50 overwhelming neural prior
- All n-gram cache "improvements" were artifacts of hash collision density, not prediction quality
- ADAM_WD env var exists but WD should only apply to Muon matrix params

## Fixed in Audit (post-Phase 8)
- **CRASH**: INT8_CLIP_Q undefined after Phase 6 removed INT8_CLIP_PERCENTILE. QAT would NameError. Fixed: added INT8_CLIP_Q = 0.9999.
- **PRECISION**: EMA accumulated in bf16 instead of float32, causing rounding drift over thousands of steps. Fixed: init and update in float32.
- **CORRECTNESS**: GPTQ-lite used per-row percentile selection; winning submissions use per-tensor. Fixed: matched winning approach.
- **TTT**: Missing cosine LR decay and gradient clipping vs winning submission. Fixed: added both.

## Fixed in Second Audit (post-TTT rewrite)
- **DESIGN**: TTT adapted cumulatively across chunks. Winning submission resets per chunk. Fixed: deepcopy + restore per chunk.
- **DESIGN**: TTT adapted ALL params with SGD. Winning submission freezes weight matrices. Fixed: freeze ndim>=2 with min_dim>=64.
- **PERF**: TTT processed 1 seq at a time (batch=1). Fixed: batched all seqs per chunk (~32x throughput).
- **RISK**: No divisibility guard on chunk_tokens % train_seq_len. Fixed: assert added.
- **PRECISION**: SWA accumulated in bf16. Fixed: float32 init and update.
- **NOTE**: neural_temp=0.85 in sliding eval but not standard eval — metrics not directly comparable. Not fixed (by design).
- **NOTE**: All 8 GPUs run identical redundant TTT work. Not fixed (TTT is sequential by nature — each chunk depends on prior adaptation). Winning submission does the same.
