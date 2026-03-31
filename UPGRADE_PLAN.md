# Upgrade Plan: v12 Competitive Stack

## Goal
Upgrade train_gpt.py from ~1.21 BPB to competitive ~1.11 BPB by implementing the proven frontier stack, inspired by PR #1089 but as our own custom implementation.

## Reference
- PR #1089 human source: /tmp/pr1089_human.py (2940 lines)
- Our current: /Volumes/2TB STORAGE/Workspace/parameter-golf/train_gpt.py (1470 lines)
- Hard limit: 1500 lines for train_gpt.py

## Changes Required (Priority Order)

### 1. Turbo-Muon with AOL + Polar Express (CRITICAL)
**Current:** Basic Newton-Schulz with fixed coefficients (a=3.4445, b=-4.7750, c=2.0315), 3 steps
**Target:** AOL preconditioning (Gershgorin scaling) + Polar Express coefficients (Amsel et al.), 4 steps, row_col post-normalization

Replace `zeropower_via_newtonschulz5` (lines 127-140) AND `zeropower_via_newtonschulz5_batched` (lines 142-156):
- Add `_POLAR_COEFFS_FULL` and `_AOL_POLAR_COEFFS` tables
- Left Gram AOL: compute A = X @ X.T, Gershgorin scaling s = 1/(A.abs().sum(dim=1).sqrt() + eps)
- Apply s to both X and A, then iterate with Polar Express coefficients
- Support both 2D and 3D (batched) paths
- Handle transpose when rows > cols (always work in smaller dimension)
- Add `_post_ns_normalize(X, mode)` function with "row_col" mode

### 2. LeakyReLU² Activation (CRITICAL, easy)
**Current:** Standard MLP activation (likely GELU or ReLU²)
**Target:** `F.leaky_relu(x, negative_slope=0.3).square()` in MLP forward

In MLP.forward, change activation to:
```python
x = F.leaky_relu(F.linear(x, w_up), negative_slope=0.3)
return F.linear(x.square(), w_down)
```

### 3. EngramLite (replaces BigramHash) (HIGH)
**Current:** BigramHashEmbedding with single hash function
**Target:** Multi-head prime-based hash with bigram + trigram coverage + learned sigmoid gate

New class EngramLite(nn.Module):
- num_buckets=8192, num_heads=2, num_orders=2, dim_per_head=32
- 2 bigram hash heads (prime-based: 1009, 2719/3137)
- 2 trigram hash heads (prime-based: 36313/27191/4903, 7919/4391/6151)
- Unified lookup, concat, project through CastedLinear
- Learned sigmoid gate per model_dim

Replace BigramHashEmbedding entirely. Update GPT.__init__ and forward.

### 4. Full Hessian GPTQ (CRITICAL)
**Current:** Percentile-search clip quantization
**Target:** Hessian-calibrated GPTQ with Cholesky error compensation

Add:
- `collect_hessians(model, loader, ...)` — hook-based H = X^T X collection on CastedLinear modules
- `quantize_int6_gptq(weight, hessian, ...)` — Full GPTQ with:
  - Cholesky factorization of damped Hessian
  - Column reordering by descending Hessian diagonal
  - Block-column sweep with error compensation
  - Multi-percentile scale search per GPTQ pass
  - Fallback to percentile search on Cholesky failure
- `_gptq_block_sweep(W, Hinv, sf, qmin, qmax, block_size)` helper
- Reserve ~9s from training wallclock for Hessian collection

### 5. Mixed-Precision Bit Allocation (HIGH)
**Current:** Uniform quant_bits for all layers
**Target:** Hessian-sensitivity-based int5/int6/int7 allocation

Add `_allocate_bits_mixed(hessian_map, state_dict, target_bytes, code_bytes)`:
- Group tensors by (layer, attn/mlp)
- Compute per-group sensitivity from Hessian trace
- Greedy promotion: most sensitive → int7, next → int6, rest → int5
- Stay within 16MB budget minus pruning headroom

### 6. Parallel Muon Optimizer (HIGH)
**Current:** Gather-scatter Muon with sequential NS per matrix
**Target:** Reduce-scatter → local NS → all-gather pipeline

Refactor Muon class:
- `launch_reduce_scatters()` — async reduce-scatter for all banks (biggest first)
- `step()` — wait for RS, local NS, launch async all-gather, overlap with next bank
- No DDP needed for bank params
- Add post_norm parameter ("row_col")
- Add weight_decay parameter

### 7. MLP 3.5x Width (MEDIUM)
**Current:** MLP_MULT=2 (int)
**Target:** MLP_MULT=3.5 (float), fits within 16MB via mixed-precision quantization

Change `mlp_mult` from int to float in Hyperparameters.

### 8. U-Net Skip Connections (MEDIUM)
**Current:** No skip connections
**Target:** Encoder/decoder with sigmoid-gated skip connections

In GPT:
- Compute encoder_layers = num_layers // 2, decoder_layers = num_layers - encoder_layers
- Add skip_weights and skip_gates parameters (per decoder layer, per model_dim)
- In forward: store encoder outputs, apply `gate * skip_weight * encoder_output + (1-gate) * decoder_input`

### 9. ValueEmbedding (MEDIUM)
**Current:** Not present
**Target:** Reinject token identity into attention values at layers 9, 10

New class ValueEmbedding:
- Embedding(vocab_size, ve_dim) + optional projection to kv_dim
- Learned scale parameter (init 0.1)
- Add v_embed to attention output at specified layers

### 10. Soft-Round QAT (MEDIUM)
**Current:** Standard STE fake quantization
**Target:** Soft-round with sigmoid alpha ramping 1→16

Add `_apply_qat_soft_round(w_cast, w_fp32, bits, alpha)`:
- Sigmoid-based differentiable rounding
- Alpha ramps from 1 to 16 over QAT phase

### 11. Brotli + Byte-Shuffle Compression (LOW)
**Current:** lzma/zlib/brotli candidates
**Target:** Byte-shuffle preprocessing + brotli-11 as primary

Add `_byte_shuffle(data, stride=2)` and `_byte_unshuffle(data)`:
- Reorder bytes by significance position before compression
- Magic header "BSHF" for auto-detection on decompress

### 12. Hyperparameter Updates (LOW, easy)
- MUON_MOMENTUM: 0.95 → 0.99
- MUON_BACKEND_STEPS: 3 → 4
- MLP_MULT: 2 → 3.5
- WARMDOWN_ITERS: 1200 → 3500
- XSA_LAST_N: 0 → 11 (all layers)
- NUM_LAYERS: 9 → 11
- TRAIN_SEQ_LEN: 1024 → 2048
- QAT_START_FRAC: 0.0 → 0.15
- EMA_DECAY: 0.0 → 0.997
- MUON_WD: 0.0 → 0.04
- MATRIX_LR: 0.04 → 0.025
- Add LR_FLOOR=0.05 (warmdown doesn't reach zero)
- NEURAL_TEMP: 0.85 → 0.90
- ROPE_DIMS: 0 → 16
- LN_SCALE: 0 → 1

## Line Budget Strategy
Current: 1470 lines. Limit: 1500.
- We CANNOT fit all of the above in 1500 lines
- Solution: This is for our records/ submission, not the base script
- Create records/track_10min_16mb/our_submission/train_gpt.py (can be longer than 1500)
- Actually: The 1500 limit is for the BASE scripts only, not records/
- Records submissions include their own train_gpt.py with no line limit
- We can also use a shrink.py to compress for code bytes

## Implementation Order
1. Branch: `strategy/v3-frontier`
2. Copy train_gpt.py to records dir as working copy
3. Implement changes 1-12 in order
4. Test locally with MLX script first (smoke test)
5. Deploy to RunPod for H100 benchmarking
