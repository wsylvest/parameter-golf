ABOUTME: This file tracks the current implementation plan for Muon optimizer throughput optimization.
ABOUTME: It is used by Claude Code to maintain phase-by-phase execution state.

# Shape-Bucket Batched Newton-Schulz for Muon

## Phase 0: Inspect and Baseline

### Verified Facts
- train_gpt.py at HEAD (8849f3f): 1427 lines (73 remaining)
- Muon optimizer: lines 136-195
- Newton-Schulz helper: lines 120-133, called once per matrix param per step
- zeropower_via_newtonschulz5 is torch.compiled at line 887
- Communication batching exists: updates_flat + single all_reduce (line 183-184)
- Compute batching does NOT exist: NS called sequentially per matrix in loop (line 177)
- Matrix params: 6 per block × 11 blocks = 66 matrices (+ optional bigram.proj)
- Shape buckets (11L/512d/3xMLP): 4 unique shapes
  - (512, 512): 22 matrices (c_q, attn.proj)
  - (256, 512): 22 matrices (c_k, c_v)
  - (1536, 512): 11 matrices (mlp.fc)
  - (512, 1536): 11 matrices (mlp.proj)
- NS steps per call: 5 (backend_steps env var default)
- NS is called only on rank-local params (i % world_size == rank, line 168)
- On 8 GPUs: each rank processes ~66/8 ≈ 8-9 matrices per step
- On 1 GPU: each rank processes all 66 matrices per step
- Existing params distributed round-robin across ranks

### Assumptions
- The primary bottleneck is per-matrix NS call overhead (kernel launch + synchronization)
- Batching into 4 shape buckets (from 66 calls) will reduce launch overhead
- torch.bmm for batched matmul should be faster than sequential mm
- Existing communication path (updates_flat) need not change

### Hypothesis
Batching NS compute by shape bucket will reduce Muon step latency by 15-30% on GPU, translating to 5-15% more steps in 600s.

### Files to Change
- train_gpt.py — add batched NS helper, refactor Muon.step

### Acceptance Criteria per Phase
- Phase 0: baseline metrics captured
- Phase 1: instrumentation produces timing/count data
- Phase 2: batched NS helper passes equivalence tests
- Phase 3: Muon.step uses shape buckets, NS calls drop to ~4
- Phase 4: loss trajectory matches sequential path
- Phase 5: throughput improves measurably
- Phase 6: decision on L13 follow-up

### Rollback Criteria
- Loss trajectory diverges from sequential baseline
- Throughput does not improve
- Memory usage increases unacceptably
