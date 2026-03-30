# Parameter Golf — TODO & Implementation Roadmap

## Current State
- **v8:** 1.2111 BPB (sliding, stride=64) — ~17th place
- **v11:** Queued but unrun — all #549 techniques enabled
- **Branch:** `strategy/v2-competitive`

---

## 🔴 IMMEDIATE FIXES (v12 — Expected: ~1.13-1.14)

### Fix 1: QAT Start Fraction [CRITICAL]
```
QAT_START_FRAC=0.15   # was 0.85
```
- #549 uses `LATE_QAT_THRESHOLD=0.15` = QAT runs for 85% of training
- Our 0.85 = QAT runs for only 15% of training
- Model needs much more time to adapt to quantization noise
- **Expected gain: -0.005 to -0.010 BPB**

### Fix 2: Neural Temperature
```
NEURAL_TEMP=0.90   # was 0.85
```
- Ciprian found T=0.90 optimal for relu²/LeakyReLU² via 5-point grid search
- T=0.85 may be suboptimal for our activation function
- **Expected gain: -0.001 to -0.003 BPB**
- TODO: Sweep [0.80, 0.85, 0.90, 0.95, 1.00] on a single run

### Fix 3: Brotli Compression
- Add `brotli` to compression candidates (level 11)
- PR #1105 saved 581KB vs LZMA-9
- 581KB freed = room for more parameters or higher precision
- **Expected gain: -0.001 to -0.003 BPB (indirect, via smaller artifact)**

### Fix 4: EMA vs SWA Conflict
- v11 config has BOTH `EMA_DECAY=0.997` AND `SWA_FRAC=0.3`
- Current code: if EMA is ready, it takes priority over SWA for export
- But both running wastes memory. Pick one.
- #549 uses EMA(0.997) + Tight SWA(every 50) together — check if code handles this correctly
- Recommendation: Keep EMA, disable SWA (`SWA_FRAC=0.0`) unless proven beneficial

---

## 🟡 MEDIUM-TERM IMPROVEMENTS (v13-v15 — Expected: ~1.11-1.12)

### Improvement 1: LoRA TTT (Replace Full-Weight SGD)
- Current: SGD on ALL params, 3 epochs, ~410s eval time
- Target: LoRA rank-8 on Q/V projections, AdamW + Polyak averaging
- Rascal (#1120): 53s eval, 6x better BPB gain per second
- Implementation:
  - Add `LoRALinear` wrapper for CausalSelfAttention.c_q and c_v
  - TTT phase: inject LoRA adapters, adapt with AdamW(lr=3e-4)
  - Polyak averaging (decay=0.998) for stability
  - Score with adapted model, then remove LoRA for next chunk
- **Expected gain: -0.003 to -0.005 BPB + 350s faster eval**

### Improvement 2: FlashAttention-3
- Replace `F.scaled_dot_product_attention` with direct `flash_attn` import
- FA3 uses Hopper-native kernels, ~9% faster per step
- More steps in 10 minutes = lower BPB
- Implementation:
  ```python
  try:
      from flash_attn import flash_attn_func
      USE_FA3 = True
  except ImportError:
      USE_FA3 = False
  ```
- **Expected gain: -0.005 to -0.010 BPB (via ~380 more training steps)**

### Improvement 3: Full GPTQ (Not Just Clip Search)
- Current: per-row int6 with 5-percentile clip search (GPTQ-lite)
- Target: Full GPTQ with Hessian-based calibration on training data
- PR #1099 and #1105 both use Full GPTQ for significant gains
- **Expected gain: -0.003 to -0.005 BPB**

### Improvement 4: Triton Fused MLP Kernel
- PR #1105 fuses gate+up+activation+down into a single Triton kernel
- Eliminates intermediate tensor materialization
- ~4-6ms/step saving = ~180 extra training steps
- Complex to implement but high impact
- **Expected gain: -0.003 to -0.005 BPB (via more steps)**

### Improvement 5: Brotli + Memmap Data Pipeline
- Brotli-11 for model compression (see Fix 3)
- Memmap for zero-copy data loading (PR #1105, PR #726)
- Reduces memory pressure and IO overhead
- **Expected gain: -0.001 to -0.002 BPB**

---

## 🟢 NOVEL TECHNIQUES (v16+ — Target: <1.10 or Notable Non-Record)

### Track A: Optimize MDLM Diffusion (Biggest Research Impact)
- agalimova's #1106 reached 1.1465 — first diffusion to beat AR baseline
- Nobody has applied competition-grade tricks to diffusion yet
- Steps:
  1. Reproduce MDLM at 1.1465
  2. Apply Muon optimizer (replaces AdamW)
  3. Scale to 13L, 640d
  4. Int6 QAT on diffusion weights
  5. Better ELBO eval (discrete, not MC)
- **Target: 1.10-1.12 BPB — first competitive diffusion model**
- **Impact: 🔥🔥🔥 OpenAI explicitly asked for this**

### Track B: H-Net + AR Stack Hybrid
- PR #1121 claims 0.6846 BPB with token merge/unmerge
- Process encoder at half resolution (4x less attention compute)
- If real, this is the path to winning outright
- Steps:
  1. Validate #1121's numbers
  2. Implement token merge/unmerge layers
  3. Stack with all SOTA techniques
- **Target: Sub-1.0 BPB if H-Net efficiency is real**
- **Risk: High — need to verify the numbers**

### Track C: Adaptive Per-Layer Quantization (Novel, Easy to Implement)
- Nobody has tried different quant bits per layer
- Idea: int8 for layers 0,1,10 (critical), int5 for layers 2-9
- Same total artifact size but better quality on sensitive layers
- Implementation: Add `per_layer_bits` dict to quantization function
- **Expected gain: -0.002 to -0.005 BPB**
- **Impact: Novel technique, easy to ablate and document**

### Track D: Mixture of Tiny Experts MLP
- Replace single 3x MLP with 4×1x MLPs + tiny router
- Same parameter count, more expressive
- Each expert specializes on different token patterns
- Router: `softmax(Linear(dim, 4))` = 2K params per layer
- **Expected gain: Unknown but theoretically promising**
- **Impact: Novel architecture, interesting non-record even if not SOTA**

### Track E: Sparse + Dense Attention Hybrid
- First 7 layers: local attention (window=128) — O(n) cost
- Last 4 layers: full causal attention — O(n²)
- Saves compute → more training steps in 10 minutes
- Requires Triton kernel for efficient local attention
- **Expected gain: -0.005 to -0.010 BPB (via more steps)**

### Track F: Progressive Growing
- Train 6 layers for 5 minutes
- Expand to 11 layers (duplicate/interpolate weights)
- Train 5 more minutes at full depth
- Gets more useful gradient updates early
- **Expected gain: Unknown, needs testing**
- **Impact: Novel training technique**

---

## EXECUTION PRIORITY

### This Week (v12-v13): Close the Known Gap
1. ✅ Run v11 (already configured)
2. Fix QAT_START_FRAC=0.15 → v12
3. Sweep NEURAL_TEMP → pick best
4. Add Brotli compression
5. Implement LoRA TTT
6. Add FA3 support
7. **Target: 1.12-1.13 BPB**

### Next Week (v14-v15): Systems Optimization
8. Full GPTQ
9. Triton fused MLP
10. Memmap data loading
11. **Target: 1.11-1.12 BPB**

### Weeks 3-4 (v16+): Novel Techniques
12. Pick best novel track (A, B, C, D, E, or F)
13. Implement and ablate
14. 3-seed validation
15. Submit PR
16. **Target: <1.10 BPB or notable non-record submission**

---

## WISHLIST ITEM STATUS (for OpenAI attention)

| Item | Our Plan | Priority |
|------|----------|----------|
| JEPA | Track A alt — after diffusion | P3 |
| Text Diffusion | **Track A — primary novel track** | P1 |
| H-Net Tokenization | Track B — validate first | P2 |
| Universal Transformer | Skip (proven dead end at 10min) | — |
| Megakernels | Triton fused MLP (Improvement 4) | P2 |
| State-Space Models | Could combine with Track E | P3 |
| Random Linear Maps | Interesting but not competitive yet | P4 |

---

## SUBMISSION STRATEGY

### Option 1: Record Submission (if we reach ~1.11)
- Must beat merged SOTA by ≥0.005 nats
- 3-seed validation at p < 0.01
- Full README, submission.json, train log
- Focus: "gathered-scatter Muon + LoRA TTT + [novel technique]"

### Option 2: Notable Non-Record (more likely, higher impact)
- Submit MDLM diffusion at competitive BPB
- Or submit H-Net hybrid
- Or submit adaptive per-layer quantization
- These get OpenAI attention even without beating SOTA
- "First [technique] to reach X BPB" is valuable

### Option 3: Both
- Record submission from standard stack optimization
- Non-record from novel technique
- Maximum visibility

---

*Last updated: 2026-03-30 02:14 EDT*
*Branch: strategy/v2-competitive*
*31 days remaining (deadline: April 30, 2026)*
