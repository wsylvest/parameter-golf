# Parameter Golf — Research Plan v2 (Updated with PR Analysis)

## Current Landscape (as of March 30, 2026)

### SOTA Leaderboard
| PR | BPB | Key Innovation |
|----|-----|----------------|
| #1120 "Rascal" | **1.1099** | Oracle alpha n-gram + LoRA TTT + neural/n-gram mixing |
| #1105 | **1.1138** | Fused MLP (Triton+CUTLASS EVT) + Brotli compression |
| #1099 | **1.1133** | Coprime-Stride Loader + Full GPTQ + XSA-all |
| #1122 | **1.1146** | EngramLite + Gated Skips + Full GPTQ + FA3 |
| #549 (merged) | 1.1194 | LeakyReLU² + Legal TTT + Parallel Muon |

### OpenAI Wishlist — All Now Have First Attempts
| Item | Best PR | BPB | Quality | Opportunity |
|------|---------|-----|---------|-------------|
| **H-Net Tokenization** | #1121 (gowtham) | 0.6846 | 🔥 Extraordinary | Validate, improve |
| | #1104 (DariusFeher) | 1.4116 | Solid research | Better methodology |
| **Text Diffusion** | #1106 (agalimova) | 1.1465 | 🔥 Beat AR baseline! | Close the 0.027 gap to SOTA |
| | #1119 (gowtham) | 1.4584 | Basic | — |
| **JEPA** | #1116 (gowtham) | 1.4447 | Early | Major improvement room |
| **Random Linear Maps** | #1113 (gowtham) | 1.3705 | Good proof of concept | **5.19MB artifact — 32% of limit!** |
| **Universal Transformer** | #1110 (gowtham) | 1.2249 | Near baseline | Known dead end at 10min |
| **SSM/Mamba** | #1107 (mradassaad) | 1.5633 | Very early | Massive improvement room |
| **Megakernels** | #1105 (abaybektursun) | 1.1138 | 🔥 SOTA-tier! | Already competitive |

---

## Key Insights from PR Analysis

### 1. H-Net (#1121): 0.6846 BPB is Suspicious but Potentially Groundbreaking
- **Token Merge/Unmerge:** Groups 2 adjacent tokens → 1 super-token, processes at half resolution
- Encoder (5 layers) at T/2 tokens = 4x less attention compute
- Decoder (6 layers) at full T tokens
- Only 27M params, 14MB artifact, 555s training
- **The 0.6846 BPB needs scrutiny** — this would be a massive leap. But the architecture is sound.
- **Our angle:** Combine H-Net with the current SOTA tricks (TTT, n-gram eval, better quantization)

### 2. Diffusion (#1106): First to Beat AR Baseline — THIS IS THE BIG ONE
- **MDLM with discrete absorbing-mask ELBO** eval (not Monte Carlo!)
- Key finding: **Eval method matters enormously** — MC ELBO = 2.41, discrete ELBO = 1.15 (same model!)
- Masking eps=0.1 >> 0.001 (biggest single hyperparameter)
- AR tricks that DON'T transfer: LeakyReLU², BigramHash
- Only 33M params, developed on an NVIDIA GB10 (Project DIGITS)
- **Gap to AR SOTA: only 0.027 BPB** — this is closeable
- **Our angle:** Scale up (more layers, bigger dim), better optimizer (Muon), QAT, longer training

### 3. Random Linear Maps (#1113): Proof That 5MB Artifacts Work
- **30M effective params but only 5.19MB stored** (32% of 16MB limit!)
- LoRA rank-32 on frozen random orthogonal weights
- BPB: 1.3705 — not competitive yet, but the artifact efficiency is incredible
- **Our angle:** Use the 11MB headroom for MORE adapters, higher rank, or hybrid with stored weights
- Could combine: Random maps for some layers + learned weights for critical layers

### 4. Mamba/SSM (#1107): Very Early, Huge Room
- First Mamba-3 SISO hybrid: 1.5633 BPB
- Key finding: QAT must cover Mamba in_proj/out_proj
- Quantization gap still +32 mBPB even after fix
- **Our angle:** This is wide open. A proper Mamba-2 hybrid with all the SOTA tricks could be competitive

### 5. "Rascal" (#1120): The New SOTA Uses N-gram Mixing
- **Oracle alpha** = sigmoid(8 × log(ngram_p/model_p)) × 0.95
- Compare actual model probability vs n-gram probability per token
- When n-gram is more confident: trust it. When model is: trust model.
- **LoRA TTT** (rank-8, Q/V projections, AdamW, Polyak averaging) — 53s vs 410s for old TTT
- This is essentially a **neural + n-gram ensemble**

---

## Revised Strategy: Three Tracks

### Track A: Take Diffusion to SOTA (Highest Impact)

**Why:** Agalimova's PR #1106 proved MDLM can reach 1.1465 — only 0.027 from AR SOTA. No one has optimized it with the full competition toolkit yet. If we close that gap, we have the **first diffusion model to match autoregressive SOTA in parameter golf**. OpenAI would notice.

**Steps:**
1. **A1:** Reproduce #1106's 1.1465 on RunPod (2xH100 or 8xH100)
2. **A2:** Apply Muon optimizer (replaces AdamW — should help significantly)
3. **A3:** Scale architecture: 11L→13L, 512d→640d, add GQA
4. **A4:** Int6 QAT on diffusion weights
5. **A5:** Sliding window eval at stride 16 (already standard)
6. **A6:** Try adding n-gram mixing (Rascal-style) — does it work with diffusion?
7. **A7:** TTT adaptation — can we do test-time training on a diffusion model?

**Target:** 1.115 BPB or better — competitive with AR models
**Risk:** Medium — the base result already works
**Impact:** 🔥🔥🔥 First diffusion to match AR would be a landmark

### Track B: H-Net + AR Stack Hybrid (Highest Potential BPB)

**Why:** H-Net's hierarchical processing at half resolution saves massive compute, enabling more training steps or deeper/wider models. If #1121's 0.6846 is real, this is the path to winning outright.

**Steps:**
1. **B1:** Study #1121 and #1104 architectures in detail
2. **B2:** Reproduce the H-Net token merge/unmerge mechanism
3. **B3:** Stack H-Net with the SOTA 11L techniques:
   - LeakyReLU²
   - XSA on decoder layers
   - BigramHash
   - Muon optimizer
   - EMA + SWA
   - Int6 GPTQ
4. **B4:** Add TTT and n-gram eval
5. **B5:** Experiment with merge_factor=3 or 4 (more compression)
6. **B6:** Byte-level H-Net (from #1104) — learns word boundaries from raw bytes!

**Target:** Sub-1.0 BPB if the H-Net efficiency is real
**Risk:** High — need to validate #1121's numbers first
**Impact:** 🔥🔥🔥🔥 Could win the competition outright

### Track C: Random Maps + SSM Hybrid (Most Novel)

**Why:** Random linear maps give 5MB artifacts. SSMs give parameter-efficient layers. Combine them: random frozen SSM state matrices + learned adapters = extremely compact model.

**Steps:**
1. **C1:** Reproduce #1113's random maps approach
2. **C2:** Build Mamba-2 layers with frozen random A/B/C matrices + LoRA adapters
3. **C3:** Hybrid: Random SSM layers (first 7) + learned attention layers (last 4)
4. **C4:** Budget: 5MB for random+adapter layers, 11MB for learned layers
5. **C5:** Apply all SOTA tricks to the learned portion

**Target:** 1.15-1.20 BPB with a fundamentally novel architecture
**Risk:** High — two untested ideas combined
**Impact:** 🔥🔥🔥 Most "interesting to OpenAI researchers" submission

---

## Execution Plan (31 Days Remaining)

### Week 1 (Mar 30 - Apr 5): Foundation + Track A
- **Day 1-2:** Set up RunPod, reproduce AR baseline (1.22) and current SOTA techniques
- **Day 3-4:** Reproduce MDLM diffusion (#1106) at 1.1465
- **Day 5-7:** Apply Muon, scale up, QAT → target 1.12 BPB with diffusion

### Week 2 (Apr 6 - Apr 12): Track B
- **Day 8-9:** Reproduce H-Net (#1121), validate the 0.6846 number
- **Day 10-12:** H-Net + SOTA AR tricks
- **Day 13-14:** Submit H-Net result, begin Track C prototyping

### Week 3 (Apr 13 - Apr 19): Track C + Refinement
- **Day 15-17:** Random maps + SSM hybrid
- **Day 18-19:** Best combination across all tracks

### Week 4 (Apr 20 - Apr 30): Polish + Submit
- **Day 20-25:** Optimize best-performing track
- **Day 26-28:** 3-seed validation, documentation, PR write-up
- **Day 29-30:** Final submissions

---

## What We Need

### Compute
- **RunPod account** with SSH key configured
- Apply for **OpenAI's $1M compute grant** (free credits)
- Budget: ~$500-1000 for H100 time if no grant

### Local Dev
- ✅ Repo forked and cloned at `~/Workspace/parameter-golf/`
- ✅ PyTorch env on external drive `/Volumes/2TB STORAGE/parameter-golf/`
- Architecture code: develop locally, test shapes on CPU
- Training: RunPod only

### Key Files to Study
- `train_gpt.py` — main training script (1126 lines)
- PR #1106's training script — MDLM diffusion
- PR #1121's training script — H-Net modifications
- PR #1113's training script — Random linear maps

---

## Competition Meta-Notes

1. **Claude is everywhere.** Multiple top submissions co-authored with Claude Opus/Sonnet. This is normal and accepted.
2. **N-gram eval is the new meta.** "Rascal" at 1.1099 uses oracle alpha mixing of neural + n-gram predictions. This isn't just a model — it's a hybrid neural/statistical system.
3. **The real SOTA might be > PR numbers.** Some PRs are still being validated (1-seed results, pending 3-seed).
4. **gowtham0992 submitted 5 wishlist items but they're all mediocre.** There's room to do each one significantly better.
5. **The diffusion result is underappreciated.** 1.1465 from agalimova is a genuine breakthrough — first non-AR to beat baseline — but nobody has tried to optimize it with competition-grade tricks yet.

---

*Updated: 2026-03-30 01:48 EDT | Competition ends: 2026-04-30 | 31 days remaining*
