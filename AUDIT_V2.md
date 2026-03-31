# Parameter Golf v3 Frontier — Independent Audit V2

**Date:** 2026-03-31 04:45 EDT  
**Auditor:** Merlin  
**Branch:** `strategy/v3-frontier` @ `dc907b9`  
**File:** `train_gpt.py` (2009 lines)  
**Reference:** `experiments/pr1089_reference.py` (PR #1089, 2940 lines)  

---

## 0. Executive Summary

The code is well-structured and implements a competitive Parameter Golf stack. However, this independent audit found **3 bugs** (1 critical, 2 minor), **1 algorithmic deviation** from PR #1089 that likely costs ~0.005 BPB, **1 missing coefficient**, and several refactoring opportunities. The branch is committed and pushed to `myfork/strategy/v3-frontier` in sync — no uncommitted work.

| Category | Count | Severity |
|---|---|---|
| 🔴 Critical bug | 1 | U-Net skip formula wrong |
| 🟡 Missing coefficient | 1 | Iter 6 Polar Express approximated |
| 🟡 Duplicate definition | 1 | `INT8_KEEP_FLOAT_MAX_NUMEL` defined twice |
| 🟡 Duplicate line | 1 | `quant_raw_bytes` assigned twice |
| ⚠️ SDP backend config | 1 | mem_efficient+math disabled vs ref |
| ⚠️ Line count violation | 1 | 2009 > 1500 limit |
| 💡 Design improvements | 5 | See §7 |

---

## 1. Branch & Commit Integrity ✅

```
strategy/v3-frontier  dc907b9  (local == myfork, pushed)
strategy/v2-competitive  87c836a  (frozen)
main  50390d6  (upstream)
pr-1089  7a9aa1b  (reference fetch)
```

Working tree clean. No uncommitted changes. No divergence between local and remote.

---

## 2. 🔴 CRITICAL: U-Net Skip Connection Formula Mismatch

### What PR #1089 does (line 1115-1117):
```python
g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
x = torch.lerp(scaled_skip, x, g)
# Result: (1-g)*scaled_skip + g*x
# At init (g=0.5, sw=1.0): x = 0.5*skip + 0.5*x (interpolation)
```

### What our code does (line 1009-1011):
```python
g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
sw = self.skip_weights[i].to(dtype=x.dtype)[None, None, :]
x = x + g * sw * skip
# Result: x + g*sw*skip (pure additive)
# At init (g=0.5, sw=1.0): x = x + 0.5*skip (additive injection)
```

### Impact Analysis

These are **fundamentally different operations**:

| Property | PR #1089 (lerp) | Ours (additive) |
|---|---|---|
| At init | `0.5*x + 0.5*skip` | `x + 0.5*skip` |
| Residual magnitude | Bounded (convex combo) | Unbounded (additive) |
| Gradient flow to x | Scaled by `g` | Full + skip contrib |
| Skip contribution | Replaces part of x | Adds to x |

The lerp formulation is standard in U-Net skip architectures because it:
1. Prevents activation explosion from additive skip injection
2. Creates a smooth interpolation where the model learns how much encoder info to retain
3. At init, the decoder stream is an equal blend (not amplified)

Our additive formula at init doubles the effective activation scale in the first skip connections, which could cause training instability and is almost certainly sub-optimal.

**Estimated BPB cost: -0.003 to -0.008 from using wrong formula.**

### Fix
```python
# In _run_blocks(), replace:
x = x + g * sw * skips.pop()
# With:
skip = skips.pop()
scaled_skip = sw * skip
x = torch.lerp(scaled_skip, x, g)
```

---

## 3. 🟡 Missing Polar Express Coefficient (Iter 6)

### Current (5 entries):
```python
_AOL_POLAR_COEFFS = [
    (4.107059, -2.947850, 0.544843),   # iter 2
    (3.948691, -2.908902, 0.551819),   # iter 3
    (3.318420, -2.488488, 0.510049),   # iter 4
    (2.300652, -1.668904, 0.418807),   # iter 5
    (1.875,    -1.25,     0.375),      # iter 6+: converged fixed point
]
```

### PR #1089 (6 entries before fixed point):
```python
_POLAR_COEFFS_FULL = [
    (8.28721..., ...),                                        # iter 1 (skipped with AOL)
    (4.107059111542203,  -2.9478499167379106,  0.5448431...),  # iter 2
    (3.9486908534822946, -2.908902115962949,   0.5518191...),  # iter 3
    (3.3184196573706015, -2.488488024314874,   0.51004894...), # iter 4
    (2.300652019954817,  -1.6689039845747493,  0.4188073...),  # iter 5
    (1.891301407787398,  -1.2679958271945868,  0.3768040...),  # iter 6
    (1.875, -1.25, 0.375),                                     # iter 7+: fixed point
]
```

We're **missing iter 6** `(1.8913, -1.2680, 0.3768)` and jumping to the converged fixed point. Since default NS steps = 4, the coefficients used are iters 2-5 (indices 0-3), so this **doesn't affect the default 4-step path**. But if `MUON_BACKEND_STEPS` is set to 5+, iter 6 will use the fixed point instead of the proper coefficient:

| Coeff | Iter 6 (ref) | Fixed point (ours) | Relative error |
|---|---|---|---|
| a | 1.8913 | 1.875 | 0.86% |
| b | -1.2680 | -1.250 | 1.42% |
| c | 0.3768 | 0.375 | 0.48% |

**Impact for 4-step (default): NONE. Impact for 5+ steps: minor convergence degradation.**

### Fix
Add the missing coefficient:
```python
_AOL_POLAR_COEFFS = [
    (4.107059111542203,  -2.9478499167379106,  0.5448431082926601),
    (3.9486908534822946, -2.908902115962949,   0.5518191394370137),
    (3.3184196573706015, -2.488488024314874,   0.51004894012372),
    (2.300652019954817,  -1.6689039845747493,  0.4188073119525673),
    (1.891301407787398,  -1.2679958271945868,  0.37680408948524835),
    (1.875, -1.25, 0.375),  # converged fixed point
]
```

Also use full precision (15+ digits) instead of truncated 6-digit values. While the truncation only causes ~1e-7 relative error (negligible in bf16), there's no reason not to match exactly.

---

## 4. Code Quality Issues

### 4.1 `INT8_KEEP_FLOAT_MAX_NUMEL` Defined Twice
- Line 519: `INT8_KEEP_FLOAT_MAX_NUMEL = 65_536` (in quantization section)
- Line 726: `INT8_KEEP_FLOAT_MAX_NUMEL = 65_536` (before CastedLinear)

Both are the same value, but the second definition is redundant and could mask future changes to the first. Remove line 726.

### 4.2 `quant_raw_bytes` Assigned Twice
Lines 1818-1819:
```python
quant_raw_bytes = len(quant_raw)
quant_raw_bytes = len(quant_raw)  # duplicate
```

Remove the duplicate.

### 4.3 Line Count Violation
The repo's own docstring states: *"Hard stop: train_gpt.py must never be longer than 1500 lines."*

Our file is 2009 lines (509 over limit). This is a **submission blocker**. The `shrink.py` code compressor will reduce it, but the source should still be kept as lean as possible. Major opportunities:
- TTT code (lines 1880-2000): ~120 lines, likely not used in competition run
- Profile/diagnostic code: ~40 lines removable
- Comments & docstrings: ~100 lines compressible
- Sliding window eval: ~40 lines, only used with EVAL_SEQ_LEN > 0

### 4.4 SDP Backend Configuration Divergence

| Backend | PR #1089 | Ours |
|---|---|---|
| cudnn_sdp | ✅ True | ✅ True |
| flash_sdp | ✅ True | ✅ True |
| mem_efficient_sdp | ✅ True | ❌ False |
| math_sdp | ✅ True | ❌ False |

PR #1089 enables all SDP backends and lets PyTorch pick the fastest. We disable mem_efficient and math. On H100, flash_sdp is likely selected regardless, but disabling fallbacks could cause issues on non-standard sequence lengths or when flash attention can't handle GQA edge cases.

**Fix:** Enable all SDP backends like PR #1089.

---

## 5. Algorithm Comparison — Verified Matches ✅

### 5.1 Newton-Schulz (Turbo-Muon) ✅
- AOL left-Gram Gershgorin preconditioning: match
- Coefficients: match for 4-step default (see §3 for 5+ step divergence)
- Post-NS row_col normalization: match
- 2D/3D dispatch: match
- Transpose handling: match

### 5.2 Parallel Muon Communication ✅
- No DDP: match (fixed in dc907b9)
- Async reduce-scatter after backward: match
- Size-descending sort: match
- Local NS on shard: match
- Async all-gather with pipeline: match
- Manual coalesced all-reduce for non-Muon params: match

### 5.3 Architecture ✅
- 11 layers, d=512, 8/4 GQA heads: match
- MLP 3.5× with LeakyReLU²(0.3): match
- Partial RoPE (16/64 dims): match
- LN scale 1/√(i+1): match
- XSA all 11 layers: match
- Logit softcap 30.0: match
- Tied embeddings: match

### 5.4 EngramLite ✅
- 8192 buckets, 2 heads, bi+tri: match
- All hash primes: match
- Learned sigmoid gate: match
- Projection zero-init: match

### 5.5 ValueEmbedding ✅
- Layers 9, 10: match
- ve_dim=128, scale=0.1: match
- Projection (ve_dim→kv_dim): match

### 5.6 U-Net Skip Connections ❌
- See §2 — formula is wrong (additive vs lerp)

### 5.7 SmearGate ✅
- `(1-g)*x + g*x_prev` ≡ `torch.lerp(x, x_prev, g)`: match

### 5.8 Quantization ✅
- Full Hessian GPTQ with Cholesky: match
- Column reordering by descending Hessian diagonal: match
- Multi-percentile search (5 levels): match
- Cholesky fallback: match
- Mixed-precision int5/6/7: match
- Soft-round QAT (alpha 1→16): match
- Selective pruning: match (sequential vs vectorized, see §6.1)

### 5.9 Compression ✅
- Byte-shuffle (stride=2, BSHF magic): match
- Multi-compressor best-of: match (ours is actually better — includes zstd)

### 5.10 Training Hyperparameters ✅
All match: momentum 0.99, wd 0.04, matrix_lr 0.025, warmdown 3500, QAT start 0.15, EMA 0.997, seq_len 2048, LR floor 0.05, NS steps 4.

---

## 6. Design Differences (Intentional)

### 6.1 No Parameter Banking
PR #1089 uses 3D bank tensors for all linear weights. We use per-layer CastedLinear modules. This is simpler but means more kernel launches on multi-GPU. Estimated throughput cost: 2-5%.

**Verdict:** Acceptable for now. Parameter banking is a larger refactor that could be done before the 8×H100 benchmark run.

### 6.2 No Mimetic V-O Initialization
PR #1089 initializes output projections as `O_h = -α * V_h` (α=0.05). We use standard orthogonal init with `_zero_init` on proj layers.

**Impact:** The mimetic init ensures the first attention step produces near-zero output (since OV ≈ -αI), allowing clean gradient flow at init. Our zero-init on proj achieves the same zero-at-init property, but the mimetic form gives the O matrix a head start in learning the right direction.

**Estimated BPB cost: -0.001 to -0.002.**

### 6.3 No Code Shrinking
Critical for the 16MB budget. PR #1089's `shrink.py` reduces code from ~120KB to ~24KB, freeing ~96KB for model weights. Our 2009-line file will be even larger.

**Status: Blocker for submission.**

### 6.4 Selective Pruning: Sequential vs Vectorized
Our pruning uses sequential entry-by-entry application. PR #1089 uses `prune_groups` with vectorized scatter. Functionally identical, ours is slower for large prune counts (doesn't matter — pruning runs once at export).

---

## 7. Proposed New Algorithms

### 7.1 🆕 Adaptive NS Early Termination (Novel)

**Idea:** Monitor convergence during NS iterations. If the relative change in X falls below threshold, skip remaining iterations.

```python
def zeropower_adaptive(G, max_steps=4, tol=1e-3):
    # ... AOL setup ...
    for i in range(max_steps):
        X_prev = X.clone()
        # ... NS step ...
        if i > 0:
            rel_change = (X - X_prev).norm() / (X.norm() + 1e-8)
            if rel_change < tol:
                break
    return X
```

**Why it helps:** For well-conditioned matrices (common in later training), fewer NS steps suffice. This trades ~5% compute savings for the `clone()` overhead. Net benefit depends on the distribution of convergence rates across layers.

**Risk:** Extra memory for `X_prev.clone()`, potential torch.compile graph breaks.

### 7.2 🆕 Cosine-Scheduled Momentum Warmup (Refinement)

Current momentum warmup is linear: `0.92 → 0.99` over 1500 steps. PR #1089 uses the same linear ramp.

**Proposal:** Use cosine schedule: slower ramp at the start (when gradients are noisiest) and faster convergence to target:

```python
frac = 0.5 * (1 - math.cos(math.pi * step / warmup_steps))
momentum = (1 - frac) * start + frac * target
```

**Expected impact:** Marginal (-0.0005 BPB). Low risk.

### 7.3 🆕 Hessian-Weighted QAT (From Previous Audit, Refined)

During QAT, scale the STE gradient by normalized Hessian diagonal:

```python
def _hessian_weighted_soft_round(w, bits, alpha, hessian_diag):
    # Standard soft-round quantization
    q = _soft_round_quantize(w, bits, alpha)
    # Weight gradient by importance
    h_norm = hessian_diag / hessian_diag.max()
    grad_scale = 1.0 + alpha * h_norm
    return w + (q - w).detach() * grad_scale
```

**Why:** Aligns QAT training with GPTQ export — the model learns to be robust on high-Hessian weights. **Expected: -0.001 to -0.003 BPB.**

### 7.4 🆕 Gradient Stochastic Depth (Novel)

During training, randomly skip entire block gradients with probability proportional to warmdown progress:

```python
if training and random.random() < skip_prob * warmdown_frac:
    return x  # identity — no gradient through this block
```

This acts as a regularizer, reduces compute during warmdown, and prevents overfitting to the training distribution. Skip rate increases as LR decreases.

**Expected: -0.001 to -0.002 BPB via regularization. Needs ablation.**

### 7.5 🆕 Multi-Resolution Byte-Shuffle (Novel)

Current byte-shuffle uses fixed stride=2. For mixed data (int8 quant + float16 scales + various metadata), different stride values may compress better for different sections.

**Proposal:** Partition the serialized blob by tensor type and apply optimal stride per partition. Small implementation cost, expected -1 to -3% compression improvement.

---

## 8. Priority Action Items

| Priority | Action | Impact |
|---|---|---|
| **P0** | Fix skip connection formula (§2) | ~-0.005 BPB |
| **P0** | Implement code shrinking (§6.3) | Submission blocker |
| **P1** | Add missing iter 6 coefficient (§3) | Correctness for 5+ steps |
| **P1** | Use full-precision coefficients (§3) | Exact match to reference |
| **P1** | Enable all SDP backends (§4.4) | Robustness |
| **P1** | Remove duplicate definitions (§4.1-4.2) | Code quality |
| **P2** | Add mimetic V-O init (§6.2) | ~-0.001 BPB |
| **P2** | Implement coprime-stride loading (from prev audit) | ~-0.003 BPB |
| **P2** | Hessian-weighted QAT (§7.3) | ~-0.002 BPB |
| **P3** | Reduce file to ≤1500 lines | Repo compliance |
| **P3** | Parameter banking refactor (§6.1) | ~3% throughput |

---

## 9. Estimated BPB Improvement Budget

If all P0-P2 items are implemented:

| Improvement | Estimated BPB Δ |
|---|---|
| Fix skip connection (P0) | -0.003 to -0.008 |
| Mimetic V-O init (P2) | -0.001 to -0.002 |
| Coprime-stride loading (P2) | -0.002 to -0.005 |
| Hessian-weighted QAT (P2) | -0.001 to -0.003 |
| **Total estimated** | **-0.007 to -0.018** |

Current best: 1.2099 BPB → after fixes: ~1.19-1.20 BPB  
Target (PR #1089): 1.1086 BPB  
Gap remaining: ~0.08-0.09 BPB (likely from actual training run on 8×H100)

---

*Audit completed 2026-03-31 04:45 EDT*
