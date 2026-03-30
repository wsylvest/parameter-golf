# Parameter Golf — Official Rules Summary

## Hard Constraints (Leaderboard Track)

| Constraint | Limit |
|-----------|-------|
| **Artifact size** | ≤ 16,000,000 bytes (16 MB decimal, NOT 16 MiB) |
| **Artifact contents** | Code bytes (`train_gpt.py`) + compressed model bytes |
| **Training time** | ≤ 10 minutes on 8×H100 SXM |
| **Evaluation time** | ≤ 10 minutes on 8×H100 |
| **Hardware** | 8×H100 SXM (specifically SXM variant) |
| **Metric** | Bits per byte (BPB) on FineWeb validation set |
| **Tokenizer** | Bring your own (BPB is tokenizer-agnostic) |
| **Eval sequence length** | Any (unlimited) |

## What's Allowed

- ✅ Evaluation at any sequence length
- ✅ Sliding window evaluation
- ✅ Test-time training (backward-looking, score-first — already-evaluated tokens only)
- ✅ Any architecture (transformers, SSMs, diffusion, JEPA, hybrids)
- ✅ Any tokenizer (but scrutinized carefully for bugs)
- ✅ Any quantization scheme
- ✅ Any optimizer
- ✅ Any compression method
- ✅ Hyperparameter tuning across many runs
- ✅ N-gram statistics from already-evaluated validation tokens
- ✅ Aggressive eval methods ("push the bounds of evaluation methods as aggressively as with training methods")
- ✅ Non-record submissions for novel/interesting approaches even if not SOTA
- ✅ Unlimited compute track (non-record, must note in README)

## What's NOT Allowed

- ❌ No external downloads during evaluation
- ❌ No training dataset access during evaluation (unless bits stored in <16MB artifact)
- ❌ No network calls during evaluation
- ❌ Artifact must be fully self-contained and reproducible
- ❌ No brute-forcing seeds
- ❌ No sneaking in additional compute unfairly
- ❌ Results must be reproducible (non-reproducible = disqualified)

## Submission Requirements (Record Track)

1. Beat existing SOTA by ≥ 0.005 nats
2. Statistical significance: p < 0.01 (typically 3-seed average)
3. Reproducibly train in under 10 minutes on 8×H100s
4. If tokenizer is modified: prove BPB calculation is correct with certainty
5. PR must include:
   - `README.md` with detailed explanation
   - `submission.json` with name, GitHub ID, val_bpb, metadata
   - Training log
   - `train_gpt.py` + dependencies (must compile and run within records folder)

## Non-Record Submissions

- Must satisfy 16MB artifact limit
- Open to unique/interesting approaches, in-progress solutions, negative results
- Unlimited compute track accepted (note in README)
- High bar still maintained — justify ideas and results in detail

## Key Clarifications

- **External compute for tuning:** Tuning hyperparameters across runs is fine. Brute-forcing is not.
- **OpenAI verification:** Top entries verified over time. Not every submission auto-verified.
- **Challenge period:** March 18 – April 30, 2026
- **Code size:** Counted as part of 16MB artifact (code bytes + model bytes)
- **train_gpt.py limit:** Must never exceed 1500 lines
