# Parameter Golf — Research Plan

## Objective
Build submissions targeting OpenAI's **unchecked wishlist items** — the things nobody has successfully implemented yet. These are what OpenAI wants to see and what will get noticed, regardless of whether they immediately top the leaderboard.

**Current SOTA:** 1.1194 BPB (LeakyReLU² + TTT + Parallel Muon)  
**Baseline:** 1.2244 BPB  
**Deadline:** April 30, 2026  

---

## OpenAI's Wishlist — Status

| Item | Status | Difficulty | Our Priority |
|------|--------|-----------|-------------|
| ✅ 1-bit quantization | Done (Ciprian, 1.1239 non-record) | — | Study only |
| ✅ Ternary quantization | Done (Ciprian, 1.1570 record) | — | Study only |
| ❌ **JEPA** | **UNCLAIMED** | High | **P1** |
| ❌ **Text diffusion** | **UNCLAIMED** | High | **P1** |
| ❌ **H-net tokenization** | **UNCLAIMED** | Medium | **P2** |
| ❌ **Universal transformer** | Attempted, failed (+0.025 gap) | High | **P3** (risky) |
| ❌ **Megakernels** | **UNCLAIMED** | Very High | **P4** |
| ❌ **State-space models / E2E TTT / super long context** | **UNCLAIMED** | Medium-High | **P1** |
| ❌ **Learning adapters on random linear maps** | **UNCLAIMED** | Medium | **P2** |

---

## Pillar 1: JEPA (Joint-Embedding Predictive Architecture)

### What It Is
Yann LeCun's architecture where a model predicts representations in latent space rather than predicting exact tokens. Instead of "predict the next token," you "predict the next abstract representation."

### Why OpenAI Wants It
JEPA is fundamentally different from autoregressive LMs. If someone can make it work for text compression under 16MB, that's a genuine research contribution — it would show JEPA can compete with autoregressive models at the bits-per-byte metric.

### Research Tasks
- [ ] **R1.1** — Literature review: Read I-JEPA (Assran et al. 2023), V-JEPA, and any text-domain JEPA work
- [ ] **R1.2** — Architecture design: How to make JEPA output a probability distribution over bytes/tokens for BPB scoring
- [ ] **R1.3** — The scoring problem: JEPA predicts embeddings, not tokens. We need P(token) for BPB. Options:
  - Train a lightweight prediction head that maps JEPA embeddings → token logits
  - Use the JEPA loss as a proxy, convert via variational bound
  - Hybrid: JEPA encoder + autoregressive decoder head
- [ ] **R1.4** — Prototype: Minimal JEPA that produces token-level BPB on FineWeb
- [ ] **R1.5** — Fit to 16MB: Quantization strategy for JEPA weights

### Key Risk
JEPA was designed for self-supervised representation learning, not text compression. Getting a valid BPB score requires an autoregressive or probabilistic output layer, which partially defeats the purpose. The "hybrid JEPA" approach (JEPA backbone + AR head) is likely the practical path.

### Estimated Effort
2-3 days research + 2-3 days implementation + RunPod experiments

---

## Pillar 2: Text Diffusion

### What It Is
Instead of generating text left-to-right (autoregressive), generate it by iteratively denoising — start from random noise and refine toward coherent text. Diffusion models dominate image generation; applying them to text is an active research frontier.

### Why OpenAI Wants It
Diffusion models for text could be more parameter-efficient than transformers because they can iteratively refine using the same weights (like depth recurrence, but for generation). If it works, it's a fundamentally different paradigm.

### Research Tasks
- [ ] **R2.1** — Literature review: MDLM (Sahoo et al. 2024), SEDD (Lou et al. 2024), Plaid (Gulrajani & Hashimoto 2024), Score Entropy Discrete Diffusion
- [ ] **R2.2** — BPB scoring: Diffusion models produce a variational lower bound (ELBO) on log-likelihood. This converts directly to BPB. Need to verify the eval pipeline accepts this.
- [ ] **R2.3** — Architecture: Discrete diffusion on token space (MDLM/SEDD) vs continuous diffusion on embedding space
  - MDLM: masked diffusion — progressively unmask tokens. Simple, well-studied.
  - SEDD: score-based diffusion on discrete state spaces. More principled but complex.
- [ ] **R2.4** — The 10-minute problem: Diffusion models typically need many denoising steps at eval time. With a 10-minute training budget + eval time, this is tight. Need efficient sampling.
- [ ] **R2.5** — Prototype: MDLM-style masked diffusion LM, measure BPB on FineWeb
- [ ] **R2.6** — Quantization: Same int6/ternary strategies should apply

### Key Risk
Diffusion models for text are still worse than autoregressive models at perplexity. But we don't need to win — we need to demonstrate signs of life under the 16MB constraint.

### Estimated Effort
3-4 days research + 3-4 days implementation

---

## Pillar 3: State-Space Models (SSM) / Mamba

### What It Is
SSMs (S4, Mamba, Mamba-2) replace attention with structured recurrences that run in O(n) time vs O(n²). They excel at long sequences and are parameter-efficient.

### Why OpenAI Wants It
SSMs could be dramatically more parameter-efficient than attention at this scale. If an SSM beats transformer-based entries in 16MB, that's a significant finding about the right inductive bias for small models.

### Research Tasks
- [ ] **R3.1** — Literature review: Mamba (Gu & Dao 2023), Mamba-2, Jamba (hybrid SSM+attention)
- [ ] **R3.2** — Pure SSM baseline: Mamba-only model at 512d, 10-12 layers, measure BPB
- [ ] **R3.3** — Hybrid SSM-Attention: SSM layers for local patterns (first 7-8 layers) + attention for global patterns (last 3-4 layers). This is the Jamba approach.
- [ ] **R3.4** — Parameter budget: SSM layers have ~3x fewer params than attention layers at same dim. Budget the savings into more layers or wider dim.
- [ ] **R3.5** — Quantization: Can SSM state matrices (A, B, C, D) be quantized as aggressively as attention weights? Need to test int6 and ternary on SSM-specific params.
- [ ] **R3.6** — Long context for eval: SSMs handle long sequences natively. Train at 4K+, eval with sliding window at 8K+. The long-context eval might give a BPB edge.
- [ ] **R3.7** — E2E TTT with SSM: Test-time training on SSMs — adapt the SSM state matrices on validation data. Potentially more stable than TTT on attention because SSMs have fewer parameters to corrupt.

### Key Risk
Mamba requires custom CUDA kernels for efficiency. On CPU (local dev), we can only validate shapes/logic. Real perf testing needs RunPod. Also, at 512d, SSMs might not have enough capacity — they shine at long sequences, but FineWeb eval uses relatively short contexts.

### Estimated Effort
2-3 days research + 3-4 days implementation

---

## Pillar 4: H-Net Tokenization

### What It Is
Hierarchical tokenization where the model operates on multiple granularity levels simultaneously — bytes, characters, subwords, and words. Instead of a flat BPE vocabulary, the model builds representations bottom-up through a hierarchy.

### Why OpenAI Wants It
Tokenization is a bottleneck. BPE is lossy and the vocab size directly impacts model size (embedding table). H-net tokenization could allow the model to "see" text at multiple resolutions, potentially improving BPB with a smaller or eliminated vocabulary.

### Research Tasks
- [ ] **R4.1** — Literature review: "Hierarchical Transformers Are More Efficient Language Models" (Nawrot et al. 2022), MegaByte (Yu et al. 2023), byte-level models
- [ ] **R4.2** — Architecture design:
  - **Option A:** Byte-level input → local encoder (small CNN/SSM) → subword-level transformer → byte-level decoder
  - **Option B:** Multi-scale: parallel byte and subword streams with cross-attention
  - **Option C:** MegaByte-style: patch bytes into groups, run a global model on patches, local model on bytes within patches
- [ ] **R4.3** — The 16MB advantage: No large embedding table needed if working at byte level (vocab=256, embedding=256×512=128K params vs 1024×512=512K for BPE). Savings go into more model capacity.
- [ ] **R4.4** — Prototype: MegaByte-lite with byte-level input, patch size 4-8
- [ ] **R4.5** — Integration: Can H-net tokenization stack with BigramHash, XSA, etc.?

### Key Risk
Byte-level models need more layers/compute per character than subword models. In 10 minutes, training throughput in bytes-per-second matters. The hierarchy adds complexity.

### Estimated Effort
2-3 days research + 2-3 days implementation

---

## Pillar 5: Learning Adapters on Random Linear Maps

### What It Is
Instead of learning full weight matrices, fix random (untrained) linear projections and learn only lightweight adapters on top. The random projections serve as a "basis" and the adapters learn to combine/modify them.

### Why OpenAI Wants It
This is related to random feature theory and compressed sensing. If a model can achieve good performance by only learning small adapters (LoRA-like) on random matrices, the artifact size shrinks dramatically because random matrices can be regenerated from a seed.

### Research Tasks
- [ ] **R5.1** — Literature review: Random features (Rahimi & Recht 2007), lottery ticket hypothesis, LoRA, Intrinsic Dimensionality of Objectives (Li et al. 2018)
- [ ] **R5.2** — Architecture:
  - Fix all weight matrices as deterministic pseudo-random (seeded) projections
  - Learn only: (a) per-layer LoRA adapters (rank 4-16), (b) layer norms, (c) embedding table
  - At inference: regenerate random matrices from seed (not stored in artifact), apply learned adapters
- [ ] **R5.3** — Artifact size analysis:
  - Random matrices: 0 bytes (regenerated from seed)
  - LoRA rank-8 adapters for 11 layers × 4 matrices: 11 × 4 × (512×8 + 8×512) × 2 bytes = ~720KB
  - LayerNorms + embeddings: ~100KB
  - Total: potentially **under 1MB** for the learned parameters
- [ ] **R5.4** — The quality question: Can random projections + adapters match trained dense matrices? Literature suggests ~90% of performance is achievable with intrinsic dimension << full parameter count.
- [ ] **R5.5** — Prototype: Standard 11L transformer with frozen random weights + LoRA adapters
- [ ] **R5.6** — Scaling: If artifact is only 1MB, we can go wider/deeper. 20 layers with random weights + adapters in 16MB?

### Key Risk
Random projections lose expressiveness. The adapters may not have enough capacity to compensate, especially for text modeling where precise weight configurations matter. But even a partial result (e.g., 1.15 BPB with a 2MB artifact) would be noteworthy.

### Estimated Effort
1-2 days research + 2-3 days implementation (simplest pillar to prototype)

---

## Pillar 6: Megakernels

### What It Is
Fusing the entire transformer forward pass (or large portions of it) into a single custom CUDA kernel, eliminating memory bandwidth bottlenecks from kernel launch overhead and intermediate tensor materialization.

### Why OpenAI Wants It
Speed = more training steps in 10 minutes = lower BPB. If a megakernel gets 2x throughput, that's 2x the steps. The current leaders already optimize step time aggressively (83ms/step with FlashAttention-3). Megakernels could push that further.

### Research Tasks
- [ ] **R6.1** — Literature review: ThunderKittens (Stanford), FlashAttention-3 internals, Triton fused kernels
- [ ] **R6.2** — Profile baseline: Where does time go in the current 83ms/step? Memory-bound vs compute-bound.
- [ ] **R6.3** — Fusion targets:
  - LayerNorm + QKV projection + RoPE → single kernel
  - Attention output + MLP gate+up + activation + down → single kernel
  - Full transformer block as one kernel
- [ ] **R6.4** — Triton prototype: Write fused kernels in Triton (easier than raw CUDA)
- [ ] **R6.5** — Benchmark: ms/step improvement on H100

### Key Risk
This is deep systems programming. Writing correct, fast CUDA/Triton kernels for the full transformer block is extremely hard. Also, this only helps with speed (more steps), not with architecture quality. It's an optimization on the existing stack, not a new approach.

### Estimated Effort
5+ days, requires strong CUDA/Triton skills. **Deprioritized unless speed becomes the bottleneck.**

---

## Pillar 7: E2E Test-Time Training + Super Long Context

### What It Is
Current TTT (the #1 submission) adapts model weights on validation chunks using SGD. "E2E TTT" means end-to-end test-time training with backprop through the full eval — potentially adapting not just MLP weights but learning entirely new behavior at eval time. Super long context means evaluating at 16K-128K tokens with sliding windows.

### Research Tasks
- [ ] **R7.1** — Study current TTT (PR #461, PR #549): Score-first, SGD, 3 epochs per chunk, -0.0025 BPB
- [ ] **R7.2** — Multi-scale TTT: Run adaptation at multiple chunk sizes (8K, 16K, 32K), ensemble predictions
- [ ] **R7.3** — Meta-learned TTT: Store a tiny network that predicts optimal TTT hyperparameters per chunk
- [ ] **R7.4** — E2E TTT: Can we backprop through the scoring itself? This would let the model learn to adapt.
- [ ] **R7.5** — Super long context eval: Train at 4K, eval with sliding window at 32K+ stride 16. How much does BPB improve with more context?
- [ ] **R7.6** — SSM + TTT: State-space models might be more amenable to TTT because their "state" is more compact and interpretable

### Key Risk
TTT is already the biggest BPB win after the base model. Pushing it further has diminishing returns. Also, TTT time counts toward the eval budget (current #1 uses ~410s for TTT out of a 600s eval window).

### Estimated Effort
2-3 days research + 2-3 days implementation

---

## Execution Schedule

### Phase 1: Quick Wins (Days 1-3)
**Goal:** Get a working submission on the leaderboard using existing techniques + one novel element.

1. **Day 1:** Set up RunPod, reproduce baseline (1.2244), reproduce ~1.15 with current techniques
2. **Day 2:** Implement **Pillar 5 (Random Linear Maps + Adapters)** — simplest novel idea, fastest to prototype
3. **Day 3:** Submit random-maps result (even if BPB is mediocre, it's unclaimed territory)

### Phase 2: Core Research (Days 4-10)
**Goal:** Implement the two highest-impact unclaimed items.

4. **Days 4-5:** **Pillar 3 (SSM/Mamba)** — hybrid SSM-attention model
5. **Days 6-7:** **Pillar 4 (H-Net Tokenization)** — byte-level hierarchical model
6. **Days 8-9:** **Pillar 7 (E2E TTT + Long Context)** — push TTT further
7. **Day 10:** Best combination of Pillars 3+4+7, submit

### Phase 3: Ambitious Bets (Days 11-20)
**Goal:** Tackle the hardest unclaimed items for maximum OpenAI attention.

8. **Days 11-14:** **Pillar 1 (JEPA)** — fundamentally different architecture
9. **Days 15-18:** **Pillar 2 (Text Diffusion)** — MDLM-style discrete diffusion
10. **Days 19-20:** Polish, ablate, submit best results

### Phase 4: Systems Optimization (Days 21-30)
**Goal:** If any novel architecture shows promise, optimize it with megakernels and training tricks.

11. **Days 21-25:** **Pillar 6 (Megakernels)** — only if an architecture is speed-bottlenecked
12. **Days 26-30:** Final submissions, ablations, documentation

---

## Key Lessons from Existing Submissions

### From Evangeline's Depth Recurrence Report (PR #363):
- **Quantization compounds through shared weights** — 900x error amplification through 3 cycles
- **Noisy QAT** calibrated to export precision collapses quant gap from 0.37 to 0.002 BPB
- **3x3 > 2x5** — more unique blocks with fewer repeats always wins
- **Always use all 80 training shards** — 1 shard vs 80 = 0.1 BPB difference
- **Validate single-GPU findings on target hardware** — results don't always transfer
- **Step time matters enormously** — 22% fewer steps = brutal in 10-minute budget

### From Ciprian's Ternary Report (PR #640):
- **Width > depth for ternary** — 768d/10L beats 512d/25L (faster steps)
- **relu² is strictly better than relu** — -0.024 BPB at zero cost
- **4x MLP > 3x MLP** for ternary
- **NeoMuon with 3 Newton-Schulz steps** is optimal
- **Temperature scaling (T=0.90)** is architecture-dependent (relu² specific)
- **Sliding window stride 16** gives ~0.025 BPB over chunked eval
- **Base-3 packing for ternary weights** — 5 trits/byte, 39% compression reduction

### From the Current #1 (PR #549):
- **Score-first TTT is legal** — inference_mode() guarantees no data leakage
- **TTT worth ~0.0025 BPB** — significant but not transformative
- **LeakyReLU(0.5)²** — eliminates dead neurons, -0.003 BPB from one line change
- **Parameter Banking** — batched Newton-Schulz via torch.bmm saves step time

---

## Local Dev Workflow

```bash
# Activate environment
cd "/Volumes/2TB STORAGE/parameter-golf"
source .venv/bin/activate
cd repo/

# Local: CPU-only shape/logic validation
python3 our_model.py --test-shapes

# RunPod: Real training
# SSH to pod, clone fork, run training
```

**Rule:** All architecture code developed and committed locally. Only training runs happen on RunPod.

---

## Files to Create

- `models/jepa.py` — JEPA architecture
- `models/diffusion.py` — Text diffusion (MDLM)
- `models/ssm_hybrid.py` — SSM-Attention hybrid
- `models/hnet_tokenizer.py` — Hierarchical tokenization
- `models/random_maps.py` — Random projections + learned adapters
- `kernels/` — Triton megakernels (Phase 4)
- `experiments/` — RunPod training scripts per pillar
- `results/` — Our results, ablations, writeups

---

*Created: 2026-03-30 | Competition ends: 2026-04-30 | 31 days remaining*
