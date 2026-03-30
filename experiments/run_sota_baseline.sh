#!/usr/bin/env bash
# =============================================================================
# Parameter Golf — SOTA Baseline Run
# =============================================================================
# This enables all known competitive techniques on our gather-scatter-muon branch.
# Expected BPB: ~1.13-1.15 (from 1.22 baseline)
# Target hardware: 8×H100 SXM on RunPod
#
# Usage:
#   bash experiments/run_sota_baseline.sh
#   SEED=42 bash experiments/run_sota_baseline.sh
# =============================================================================

set -euo pipefail

# Allow seed override
SEED="${SEED:-1337}"

echo "=== Parameter Golf SOTA Baseline ==="
echo "Seed: ${SEED}"
echo "Starting at: $(date)"
echo ""

# --- Architecture ---
export NUM_LAYERS=11              # 9→11 (all top submissions)
export MODEL_DIM=512              # standard
export NUM_HEADS=8                # standard
export NUM_KV_HEADS=4             # GQA
export MLP_MULT=3                 # 2→3 (bigger MLP, ~-0.005 BPB)
export VOCAB_SIZE=1024            # standard BPE
export TIE_EMBEDDINGS=1           # tied embeddings

# --- Attention Enhancements ---
export XSA_LAST_N=4               # Cross-Sequence Attention on last 4 layers (~-0.01)
export ROPE_DIMS=16               # Partial RoPE: only 16/64 dims (~-0.002)
export LN_SCALE=1                 # LayerNorm scaling 1/√(layer+1) (~-0.002)
export QK_GAIN_INIT=1.5           # standard

# --- BigramHash ---
export BIGRAM_VOCAB_SIZE=1536     # Enable bigram hashing (~-0.01)
export BIGRAM_DIM=128             # standard

# --- Quantization ---
export QUANT_BITS=6               # int6 QAT (~-0.015, fits more params in 16MB)
export QAT_START_FRAC=0.85        # Enable QAT in last 15% of training

# --- Optimizer ---
export MATRIX_LR=0.025            # Muon LR for matrix params
export SCALAR_LR=0.025            # Adam LR for scalar params
export TIED_EMBED_LR=0.035        # Embedding LR
export MUON_MOMENTUM=0.99         # Higher momentum
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export MUON_BACKEND_STEPS=3       # Newton-Schulz steps
export MUON_WD=0.04               # Weight decay for Muon (~-0.002)
export ADAM_WD=0.04               # Weight decay for Adam

# --- Training Schedule ---
export TRAIN_BATCH_TOKENS=524288  # ~512K tokens/step
export TRAIN_SEQ_LEN=1024         # standard
export ITERATIONS=20000           # max iterations (wallclock will stop earlier)
export MAX_WALLCLOCK_SECONDS=600  # 10 minute hard cap
export WARMDOWN_ITERS=3500        # Longer warmdown (~-0.003)
export WARMUP_STEPS=20            # Compile warmup

# --- Weight Averaging ---
export EMA_DECAY=0.997            # EMA for export (~-0.005)
export SWA_FRAC=0.0               # Disable SWA (using EMA instead)

# --- Eval ---
export EVAL_SEQ_LEN=1024          # Enable sliding window eval
export EVAL_STRIDE=64             # Stride for sliding window (~-0.020)
export EVAL_BATCH_SEQS=32         # Batch size for eval
export NEURAL_TEMP=0.85           # Temperature scaling

# --- TTT (Test-Time Training) ---
export TTT_LR=0.002               # Score-first TTT (~-0.003)
export TTT_EPOCHS=3               # Adaptation epochs per chunk
export TTT_CHUNK_TOKENS=32768     # 32K tokens per chunk

# --- Misc ---
export SEED="${SEED}"
export VAL_LOSS_EVERY=500
export TRAIN_LOG_EVERY=100
export COMPILE_MODE=default
export LOGIT_SOFTCAP=30.0

# --- Data (RunPod paths) ---
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="sota_baseline_s${SEED}_$(date +%Y%m%d_%H%M%S)"

echo "Run ID: ${RUN_ID}"
echo "Config: 11L 512d 3xMLP int6 XSA4 RoPE16 LN_Scale BigramHash1536"
echo "        EMA0.997 MuonWD0.04 Warmdown3500 SlidingEval64 TTT"
echo ""

torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=== Run complete: ${RUN_ID} ==="
echo "Check logs/${RUN_ID}.txt for results"
