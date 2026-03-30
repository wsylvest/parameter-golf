#!/usr/bin/env bash
# =============================================================================
# Parameter Golf — RunPod Setup Script
# =============================================================================
# Run this once after SSH-ing into a fresh RunPod instance.
# Uses the official Parameter Golf template (deps pre-installed).
#
# Usage:
#   cd /workspace
#   git clone https://github.com/wsylvest/parameter-golf.git
#   cd parameter-golf
#   git checkout gather-scatter-muon
#   bash experiments/setup_runpod.sh
# =============================================================================

set -euo pipefail

echo "=== RunPod Setup ==="

# Install zstandard if missing (needed for best compression)
pip install zstandard 2>/dev/null || echo "zstandard already installed"

# Download dataset (full 80 shards + val)
echo "Downloading FineWeb dataset (sp1024, 80 shards)..."
python3 data/cached_challenge_fineweb.py --variant sp1024

# Verify
echo ""
echo "=== Verification ==="
TRAIN_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/datasets/fineweb10B_sp1024/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Train shards: ${TRAIN_COUNT} (should be 80)"
echo "Val shards: ${VAL_COUNT} (should be ≥1)"

if [ "$TRAIN_COUNT" -lt 80 ]; then
    echo "WARNING: Only ${TRAIN_COUNT} train shards! Use all 80 for best results."
    echo "Re-run: python3 data/cached_challenge_fineweb.py --variant sp1024"
fi

# Check Flash Attention
python3 -c "
try:
    import flash_attn
    print(f'FlashAttention: {flash_attn.__version__}')
except ImportError:
    print('FlashAttention: NOT FOUND (using SDPA fallback)')
" 2>/dev/null

# Check GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem // 1024**3} GB per GPU')
"

echo ""
echo "=== Setup complete ==="
echo "Run: bash experiments/run_sota_baseline.sh"
