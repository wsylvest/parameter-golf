"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: `train_gpt.py` and `train_gpt_mlx.py` must never be longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
import lzma
import zlib
from pathlib import Path

try:
    import zstandard
except ImportError:
    zstandard = None
try:
    import brotli
except ImportError:
    brotli = None

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
# DDP not used: Parallel Muon handles gradient sync for matrix params;
# non-Muon params get manual coalesced all-reduce.

# --- BYTE-SHUFFLE COMPRESSION ---
# Reorder bytes by significance position before compression.
# For float16 scale data, grouping same-position bytes together creates
# runs of similar values -> better entropy coding. Lossless and fast (<1s).
_BSHF_MAGIC = b"BSHF"

def _byte_shuffle(data: bytes, stride: int = 2) -> bytes:
    """Interleave-transpose byte stream by stride. Prepends magic + stride header."""
    if stride <= 1 or len(data) < stride:
        return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src); out = np.empty(n, dtype=np.uint8); off = 0
    for pos in range(stride):
        chunk = src[pos::stride]; out[off:off + len(chunk)] = chunk; off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data: bytes) -> bytes:
    """Inverse of _byte_shuffle; auto-detects BSHF magic header."""
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]; payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload); out = np.empty(n, dtype=np.uint8); off = 0
    for pos in range(stride):
        clen = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:clen] = payload[off:off + clen]; off += clen
    return out.tobytes()

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3500))
    lr_floor = float(os.environ.get("LR_FLOOR", 0.05))  # minimum LR as fraction of peak
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape. mlp_mult supports float (e.g. 3.5).
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = float(os.environ.get("MLP_MULT", 3.5))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    quant_bits = int(os.environ.get("QUANT_BITS", 8))
    # EngramLite n-gram embedding (replaces BigramHashEmbedding).
    ngram_buckets = int(os.environ.get("NGRAM_BUCKETS", 8192))
    ngram_heads = int(os.environ.get("NGRAM_HEADS", 2))
    ngram_orders = int(os.environ.get("NGRAM_ORDERS", 2))
    ngram_dim_per_head = int(os.environ.get("NGRAM_DIM_PER_HEAD", 32))
    # ValueEmbedding: reinject token identity into attention values at deep layers.
    ve_enabled = bool(int(os.environ.get("VE_ENABLED", "1")))
    ve_dim = int(os.environ.get("VE_DIM", 128))
    ve_layers = os.environ.get("VE_LAYERS", "9,10")
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 0))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.025))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.025))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 4))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    muon_post_norm = os.environ.get("MUON_POST_NORM", "row_col")
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    muon_wd = float(os.environ.get("MUON_WD", 0.04))
    adam_wd = float(os.environ.get("ADAM_WD", 0.0))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.15))
    soft_round_qat = bool(int(os.environ.get("SOFT_ROUND_QAT", "1")))
    swa_frac = float(os.environ.get("SWA_FRAC", 0.0))
    swa_every = int(os.environ.get("SWA_EVERY", 100))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    neural_temp = float(os.environ.get("NEURAL_TEMP", 0.90))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))
    ttt_lr = float(os.environ.get("TTT_LR", 0.0))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))
    ttt_mode = os.environ.get("TTT_MODE", "sgd")  # "sgd" (full-weight) or "lora"
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_polyak_decay = float(os.environ.get("TTT_POLYAK_DECAY", 0.998))
    # Adaptive per-layer quantization: comma-separated bits per layer, e.g. "8,6,6,6,6,6,6,6,6,6,8"
    adaptive_quant = os.environ.get("ADAPTIVE_QUANT", "")
    profile_step_every = int(os.environ.get("PROFILE_STEP_EVERY", 0))
    compile_mode = os.environ.get("COMPILE_MODE", "default")

# --- TURBO-MUON OPTIMIZER ---
# Newton-Schulz with AOL preconditioning + Polar Express coefficients (Amsel et al., 2505.16932).
# AOL (Adaptive Orthogonal Learning): left-Gram Gershgorin scaling contracts singular value
# range before iteration, so we skip the first Frobenius-init step and get better convergence.
# Handles both 2D (single matrix) and 3D (batched) inputs in one unified function.

_AOL_POLAR_COEFFS = [
    (4.107059111542203,  -2.9478499167379106,  0.5448431082926601),   # iter 2 (AOL skips iter 1)
    (3.9486908534822946, -2.908902115962949,   0.5518191394370137),   # iter 3
    (3.3184196573706015, -2.488488024314874,   0.51004894012372),     # iter 4
    (2.300652019954817,  -1.6689039845747493,  0.4188073119525673),   # iter 5
    (1.891301407787398,  -1.2679958271945868,  0.37680408948524835),  # iter 6
    (1.875, -1.25, 0.375),                                            # iter 7+: converged fixed point
]

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 4, eps: float = 1e-7) -> Tensor:
    """Turbo-Muon: AOL left-Gram preconditioning + Polar Express coefficients.
    Supports 2D (M, N) and batched 3D (B, M, N) input. Always works in smaller dimension."""
    X = G.bfloat16()
    if X.ndim == 3:
        transposed = X.size(-2) > X.size(-1)
        if transposed: X = X.mT
        A = X @ X.mT
        s = 1.0 / (A.abs().sum(dim=-1).sqrt() + eps)
        X = s.unsqueeze(-1) * X; A = s.unsqueeze(-2) * A * s.unsqueeze(-1)
        for i in range(steps):
            a, b, c = _AOL_POLAR_COEFFS[min(i, len(_AOL_POLAR_COEFFS) - 1)]
            if i > 0: A = X @ X.mT
            X = a * X + (b * A + c * A @ A) @ X
        return X.mT if transposed else X
    else:
        transposed = X.size(0) > X.size(1)
        if transposed: X = X.T
        A = X @ X.T
        s = 1.0 / (A.abs().sum(dim=1).sqrt() + eps)
        X = s.unsqueeze(1) * X; A = s.unsqueeze(0) * A * s.unsqueeze(1)
        for i in range(steps):
            a, b, c = _AOL_POLAR_COEFFS[min(i, len(_AOL_POLAR_COEFFS) - 1)]
            if i > 0: A = X @ X.T
            X = a * X + (b * A + c * A @ A) @ X
        return X.T if transposed else X

def _post_ns_normalize(X: Tensor, mode: str) -> Tensor:
    """Post-NS normalization: equalize per-neuron update magnitudes.
    Modes: 'none' (passthrough), 'row', 'col', 'row_col'. Supports 2D and 3D."""
    if mode == "none": return X
    if mode in ("row", "row_col"):
        X = X / (X.float().norm(dim=-1, keepdim=True).to(X.dtype) + 1e-7)
    if mode in ("col", "row_col"):
        X = X / (X.float().norm(dim=-2, keepdim=True).to(X.dtype) + 1e-7)
    return X

class Muon(torch.optim.Optimizer):
    """Parallel Muon: reduce-scatter → local NS → all-gather pipeline.

    On multi-GPU: after backward, launches async reduce-scatter for each param
    (biggest first), runs NS on the local shard, then all-gathers the result.
    Each all-gather overlaps with the next param's NS. On single-GPU: falls back
    to batched NS on shape-bucketed banks (same as before).
    """
    ns_calls: int = 0

    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, post_norm: str = "none"):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                     nesterov=nesterov, post_norm=post_norm))
        self._built = False

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self._built = False

    def _build(self) -> None:
        self._distributed = dist.is_available() and dist.is_initialized()
        self._world_size = dist.get_world_size() if self._distributed else 1
        self._rank = dist.get_rank() if self._distributed else 0
        ws = self._world_size
        # Build per-param metadata for distributed path
        self._param_meta: list[dict] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2: continue
                B = p.shape[0]; tail = p.shape[1:]
                padded_B = ((B + ws - 1) // ws) * ws; shard_B = padded_B // ws
                dev = p.device
                self._param_meta.append({
                    "p": p, "B": B,
                    "padded_grad": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "shard_mom": torch.zeros(shard_B, *tail, device=dev, dtype=torch.bfloat16),
                    "full_update": torch.zeros(padded_B, *tail, device=dev, dtype=torch.bfloat16),
                    "scale": max(1, p.shape[0] / p.shape[1]) ** 0.5,
                })
        # Sort biggest first for optimal overlap
        self._param_meta.sort(key=lambda m: -m["p"].numel())
        # Build shape-bucketed banks for single-GPU fallback
        for group in self.param_groups:
            buckets: dict[tuple[int, int], list[int]] = {}
            params = group["params"]
            for i, p in enumerate(params):
                if p.ndim == 2:
                    buckets.setdefault((p.size(0), p.size(1)), []).append(i)
            banks = {s: torch.zeros(len(idxs), s[0], s[1], dtype=torch.bfloat16, device=params[0].device)
                     for s, idxs in buckets.items()}
            group["_buckets"] = buckets
            group["_banks"] = banks
            group["_scales"] = {s: max(1, s[0] / s[1]) ** 0.5 for s in buckets}
        self._built = True

    def launch_reduce_scatters(self) -> None:
        """Phase 1: launch async reduce-scatter for all params. Call right after backward."""
        if not self._built: self._build()
        if not self._distributed: return
        self._rs_futures = []
        for m in self._param_meta:
            p = m["p"]
            if p.grad is None:
                self._rs_futures.append(None); continue
            pg = m["padded_grad"]; pg[:m["B"]].copy_(p.grad.bfloat16())
            if pg.shape[0] > m["B"]: pg[m["B"]:].zero_()
            fut = dist.reduce_scatter_tensor(m["shard"], pg, op=dist.ReduceOp.AVG, async_op=True)
            self._rs_futures.append(fut)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        if not self._built: self._build()

        for group in self.param_groups:
            lr, mom, bs = group["lr"], group["momentum"], group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("wd", 0.0)
            post_norm = group.get("post_norm", "none")

            if self._distributed and hasattr(self, "_rs_futures"):
                # --- Parallel path: wait RS → local NS → async AG ---
                prev_ag, prev_m = None, None
                for i, m in enumerate(self._param_meta):
                    p = m["p"]
                    if p.grad is None: continue
                    # Finalize previous all-gather
                    if prev_ag is not None:
                        prev_ag.wait()
                        pp = prev_m["p"]; upd = prev_m["full_update"][:prev_m["B"]]
                        pp.add_(upd.to(pp.dtype), alpha=-lr * prev_m["scale"])
                        if wd > 0: pp.data.mul_(1.0 - lr * wd)
                    # Wait for this param's reduce-scatter
                    if self._rs_futures[i] is not None:
                        self._rs_futures[i].wait()
                    # Momentum on shard
                    g = m["shard"]; buf = m["shard_mom"]
                    buf.mul_(mom).add_(g)
                    update = g.add(buf, alpha=mom) if nesterov else buf
                    # NS on local shard (2D path)
                    update = zeropower_via_newtonschulz5(update, steps=bs)
                    update = _post_ns_normalize(update, post_norm)
                    Muon.ns_calls += 1
                    # Launch async all-gather
                    prev_ag = dist.all_gather_into_tensor(m["full_update"], update, async_op=True)
                    prev_m = m
                # Finalize last all-gather
                if prev_ag is not None:
                    prev_ag.wait()
                    pp = prev_m["p"]; upd = prev_m["full_update"][:prev_m["B"]]
                    pp.add_(upd.to(pp.dtype), alpha=-lr * prev_m["scale"])
                    if wd > 0: pp.data.mul_(1.0 - lr * wd)
                if hasattr(self, "_rs_futures"): del self._rs_futures
            else:
                # --- Single-GPU path: batched NS on shape-bucketed banks ---
                params = group["params"]
                for i, p in enumerate(params):
                    if p.grad is None or p.ndim != 2: continue
                    g = p.grad; state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]; buf.mul_(mom).add_(g)
                    self.state[p]["_ns_grad"] = g.add(buf, alpha=mom) if nesterov else buf.clone()
                for shape, idxs in group["_buckets"].items():
                    bank = group["_banks"][shape]; active = []
                    for slot, pi in enumerate(idxs):
                        ns_g = self.state[params[pi]].get("_ns_grad")
                        if ns_g is not None: bank[slot].copy_(ns_g); active.append((slot, pi))
                        else: bank[slot].zero_()
                    if not active: continue
                    ortho = zeropower_via_newtonschulz5(bank, steps=bs)  # 3D batched path
                    ortho = _post_ns_normalize(ortho, post_norm)
                    ortho *= group["_scales"][shape]
                    Muon.ns_calls += 1
                    for slot, pi in active:
                        p = params[pi]
                        p.add_(ortho[slot].to(p.dtype), alpha=-lr)
                        if wd > 0: p.data.mul_(1.0 - lr * wd)
                        self.state[p].pop("_ns_grad", None)
        return loss

# --- TOKENIZER-AGNOSTIC EVALUATION (BPB metric) ---

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]

def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

def eval_val_sliding(
    args, model: nn.Module, base_model: nn.Module, rank: int, world_size: int,
    device: torch.device, val_tokens: Tensor, base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    eval_seq = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    stride, batch_seqs = args.eval_stride, args.eval_batch_seqs
    total_tokens = val_tokens.numel() - 1
    starts = list(range(0, total_tokens - eval_seq + 1, stride))
    rank_starts = starts[rank::world_size]
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    neural_temp = args.neural_temp
    model.eval()
    with torch.inference_mode():
        for batch_i in range(0, len(rank_starts), batch_seqs):
            batch_starts = rank_starts[batch_i:batch_i + batch_seqs]
            B = len(batch_starts)
            raw = torch.stack([val_tokens[s:s + eval_seq + 1].to(device=device, dtype=torch.int64) for s in batch_starts])
            x, y = raw[:, :-1], raw[:, 1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = base_model.forward_logits(x)
            seg_logits = logits[:, -stride:, :].float()
            seg_targets = y[:, -stride:]
            seg_nll = F.cross_entropy((seg_logits / neural_temp).reshape(-1, seg_logits.size(-1)),
                                      seg_targets.reshape(-1), reduction="none").to(torch.float64).reshape(B, stride)
            val_loss_sum += seg_nll.sum()
            val_token_count += float(B * stride)
            prev_ids, tgt_ids = x[:, -stride:].reshape(-1), y[:, -stride:].reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            val_byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        for t in [val_loss_sum, val_token_count, val_byte_count]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    model.train()
    return float(val_loss.item()), float(bits_per_token * val_token_count.item() / val_byte_count.item())

# --- POST-TRAINING QUANTIZATION ---

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,"
        "skip_weight,skip_weights,skip_gates,ngram_gate,ve.scale",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
GPTQ_CLIP_PERCENTILES = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
INT8_CLIP_Q = 0.9999  # fixed quantile for QAT fake-quantize (training-time only)

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor, bits: int = 8) -> tuple[Tensor, Tensor]:
    max_val = 2 ** (bits - 1) - 1
    t32 = t.float()
    if t32.ndim == 2 and t32.numel() > 0:
        best_q, best_scale, best_mse = None, None, float("inf")
        for pct in GPTQ_CLIP_PERCENTILES:
            clip_abs = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
            clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
            q = torch.clamp(torch.round(clipped / scale[:, None]), -max_val, max_val).to(torch.int8)
            recon = q.float() * scale[:, None]
            mse = (t32 - recon).pow(2).mean().item()
            if mse < best_mse:
                best_q, best_scale, best_mse = q, scale, mse
        return best_q.contiguous(), best_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()
    if t32.ndim == 2:
        return torch.zeros_like(t32, dtype=torch.int8), torch.empty((t32.shape[0],), dtype=INT8_PER_ROW_SCALE_DTYPE)
    clip_abs = float(t32.abs().max().item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / max_val if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_val, max_val).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor], quant_bits: int = 8, layer_bits: dict[int, int] | None = None):
    # Quantization with optional per-layer adaptive bits (layer_bits maps layer_idx → bits)
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        bits = 8 if "tok_emb" in name else quant_bits
        # Adaptive per-layer quantization: override bits for specific layers
        if layer_bits:
            import re
            m = re.search(r'blocks\.(\d+)\.', name)
            if m:
                bits = layer_bits.get(int(m.group(1)), bits)
        q, s = quantize_float_tensor(t, bits=bits)
        meta: dict[str, object] = {}
        if s.ndim > 0:
            meta["scheme"] = "per_row"
            meta["axis"] = 0
        if bits != 8:
            meta["bits"] = bits
        if meta:
            qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out

# --- DATA LOADING ---

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# --- TRANSFORMER MODULES ---

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    qat_enabled: bool = False
    qat_soft_round: bool = False   # if True, use differentiable soft-round instead of STE
    qat_alpha: float = 1.0         # soft-round sharpness: ramped 1→16 during QAT phase

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.qat_enabled and self.training and w.ndim == 2 and w.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            if self.qat_soft_round:
                w = _soft_round_quantize(w, _QUANT_BITS_QAT, CastedLinear.qat_alpha)
            else:
                w = _fake_quantize(w)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)

_QUANT_BITS_QAT = 8

def _fake_quantize(w: Tensor) -> Tensor:
    max_val = 2 ** (_QUANT_BITS_QAT - 1) - 1
    w32 = w.float()
    clip_abs = torch.quantile(w32.abs(), INT8_CLIP_Q, dim=1)
    scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
    w_clipped = torch.clamp(w32, -clip_abs[:, None], clip_abs[:, None])
    w_q = torch.round(w_clipped / scale[:, None]) * scale[:, None]
    return w + (w_q - w).detach()

def _soft_round_quantize(w: Tensor, bits: int, alpha: float) -> Tensor:
    """Differentiable quantization via soft-round (sigmoid-based).
    Alpha ramps 1→16 during QAT: low alpha = smooth gradient, high = near-hard rounding."""
    max_val = 2 ** (bits - 1) - 1
    w32 = w.float()
    clip_abs = torch.quantile(w32.abs(), INT8_CLIP_Q, dim=1)
    scale = (clip_abs / max_val).clamp_min(1.0 / max_val)
    w_scaled = (w32 / scale[:, None]).clamp(-max_val, max_val)
    w_floor = w_scaled.detach().floor()
    frac = w_scaled - w_floor  # fractional part; grad flows through w_scaled
    soft = w_floor + torch.sigmoid(alpha * (frac - 0.5))
    return (soft * scale[:, None]).to(w.dtype)

def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()

class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, rope_dims: int = 0, use_xsa: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.use_xsa = use_xsa
        self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dims, base=rope_base)

    def forward(self, x: Tensor, v_embed: "Tensor | None" = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_raw = self.c_v(x)
        if v_embed is not None:
            v_raw = v_raw + v_embed  # inject token identity into values
        v = v_raw.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        rd = self.rope_dims
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = torch.cat((apply_rotary_emb(q[..., :rd], cos, sin), q[..., rd:]), dim=-1)
        k = torch.cat((apply_rotary_emb(k[..., :rd], cos, sin), k[..., rd:]), dim=-1)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True,
                                           enable_gqa=(self.num_kv_heads != self.num_heads))
        if self.use_xsa:
            v_exp = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1) if self.num_kv_heads != self.num_heads else v
            v_n = F.normalize(v_exp, dim=-1)
            y = y - (y * v_n).sum(dim=-1, keepdim=True) * v_n
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: float):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        # LeakyReLU² activation: negative_slope=0.3 (better than 0.5 empirically)
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.3).square())

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        return (1 - g) * x + g * x_prev

class EngramLite(nn.Module):
    """Multi-head hash-based n-gram embedding with learned gating (replaces BigramHashEmbedding).

    Uses multiple independent prime-based hash functions for bigram + trigram coverage,
    reducing collision probability vs a single hash. A learned sigmoid gate per model_dim
    suppresses n-gram signal when the Transformer's own context is more informative.
    """
    def __init__(self, num_buckets: int, num_heads: int, num_orders: int, dim_per_head: int, model_dim: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.num_orders = num_orders
        total_slots = num_orders * num_heads * num_buckets
        concat_dim = num_orders * num_heads * dim_per_head
        self.embed = nn.Embedding(total_slots, dim_per_head)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(concat_dim, model_dim, bias=False)
        self.proj._zero_init = True
        self.ngram_gate = nn.Parameter(torch.zeros(model_dim, dtype=torch.float32))

    def forward(self, input_ids: Tensor) -> Tensor:
        B = self.num_buckets
        prev = F.pad(input_ids[:, :-1], (1, 0), value=0)
        # Bigram heads: two independent prime-based hashes
        bi0 = (prev * 1009 + input_ids) % B
        bi1 = ((prev * 2719 + 314159) ^ (input_ids * 3137)) % B
        indices = [bi0, bi1 + B]
        # Trigram heads (if enabled)
        if self.num_orders >= 2:
            pp = F.pad(prev[:, :-1], (1, 0), value=0)
            tri0 = ((pp * 36313) ^ (prev * 27191) ^ (input_ids * 4903)) % B
            tri1 = ((pp * 7919) ^ (prev * 4391) ^ (input_ids * 6151)) % B
            off = 2 * B
            indices.extend([tri0 + off, tri1 + off + B])
        all_idx = torch.stack(indices, dim=-1)
        all_emb = self.embed(all_idx)
        out = self.proj(all_emb.reshape(*input_ids.shape, -1))
        return out * torch.sigmoid(self.ngram_gate.to(dtype=out.dtype))[None, None, :]

class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers.

    Helps shallow attention heads leverage token identity directly without
    relying solely on the residual stream. Applied at deep layers where
    attention has sufficient context to benefit from this signal.
    """
    def __init__(self, vocab_size: int, ve_dim: int, kv_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, kv_dim, bias=False) if ve_dim != kv_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: float,
                 rope_base: float, qk_gain_init: float, ln_scale: float = 1.0,
                 rope_dims: int = 0, use_xsa: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        rope_dims=rope_dims, use_xsa=use_xsa)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale = ln_scale

    def forward(self, x: Tensor, x0: Tensor, v_embed: "Tensor | None" = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.ln_scale
        attn_out = self.attn(self.attn_norm(x) * s, v_embed=v_embed)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * s)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: float, tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 ngram_buckets: int = 0, ngram_heads: int = 2, ngram_orders: int = 2,
                 ngram_dim_per_head: int = 32, ln_scale: bool = False, rope_dims: int = 0,
                 xsa_last_n: int = 0, ve_enabled: bool = False, ve_dim: int = 128,
                 ve_layers: "list[int] | None" = None):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        # U-Net skip connections: learned per-dim weight + sigmoid gate
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                  ln_scale=1.0 / (i + 1) ** 0.5 if ln_scale else 1.0, rope_dims=rope_dims,
                  use_xsa=(i >= num_layers - xsa_last_n))
            for i in range(num_layers)
        ])
        self.smear_gate = SmearGate(model_dim)
        # EngramLite n-gram embedding (replaces BigramHash)
        kv_dim = model_dim // num_heads * num_kv_heads
        self.engram = EngramLite(ngram_buckets, ngram_heads, ngram_orders, ngram_dim_per_head, model_dim) if ngram_buckets > 0 else None
        # ValueEmbedding: inject token identity at deep attention layers
        self.ve_layers: list[int] = ve_layers or []
        self.ve_modules = nn.ModuleList([ValueEmbedding(vocab_size, ve_dim, kv_dim) for _ in self.ve_layers]) if (ve_enabled and self.ve_layers) else nn.ModuleList()
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.ndim == 2:
                if min(module.weight.shape) >= 64 and not getattr(module, "_zero_init", False):
                    nn.init.orthogonal_(module.weight, gain=1.0)
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_blocks(self, x: Tensor, x0: Tensor, input_ids: Tensor) -> Tensor:
        ve_map = {li: ve for li, ve in zip(self.ve_layers, self.ve_modules)}
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            v_emb = ve_map[i](input_ids) if i in ve_map else None
            x = self.blocks[i](x, x0, v_embed=v_emb)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                skip = skips.pop()
                g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                scaled_skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
                x = torch.lerp(scaled_skip, x, g)
            li = self.num_encoder_layers + i
            v_emb = ve_map[li](input_ids) if li in ve_map else None
            x = self.blocks[li](x, x0, v_embed=v_emb)
        return x

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.engram is not None:
            x = x + self.engram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear_gate(x)
        x0 = x
        x = self._run_blocks(x, x0, input_ids)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.engram is not None:
            x = x + self.engram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear_gate(x)
        x0 = x
        x = self._run_blocks(x, x0, input_ids)
        x = self.final_norm(x)
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

# --- FULL HESSIAN GPTQ ---
# Hessian-calibrated post-training quantization with Cholesky error compensation.
# Significantly better than percentile-only clip search, especially for sensitive layers.

def _gptq_block_sweep(W: Tensor, Hinv: Tensor, sf: Tensor, qmin: int, qmax: int, bs: int = 128) -> Tensor:
    """GPTQ block-column sweep with per-row fixed scales and Hessian error propagation."""
    rows, cols = W.shape
    Q = torch.zeros_like(W, dtype=torch.int8)
    W_work = W.clone()
    for i1 in range(0, cols, bs):
        i2 = min(i1 + bs, cols)
        W1 = W_work[:, i1:i2].clone()
        Q1 = torch.zeros(rows, i2 - i1, dtype=torch.int8, device=W.device)
        Err1 = torch.zeros(rows, i2 - i1, device=W.device)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]; d = Hinv1[i, i]
            q = torch.clamp(torch.round(w / sf), qmin, qmax).to(torch.int8)
            Q1[:, i] = q; err = (w - q.float() * sf) / d
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0); Err1[:, i] = err
        Q[:, i1:i2] = Q1
        if i2 < cols:
            W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q

def quantize_int_gptq(weight: Tensor, hessian: "Tensor | None" = None,
                      qmax: int = 31, damp: float = 0.01) -> "tuple[Tensor, Tensor]":
    """Full Hessian GPTQ for a 2D weight matrix. Falls back to percentile search if no hessian."""
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        # Fallback: percentile clip search
        best_q, best_s, best_err = None, None, float("inf")
        for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
            rc = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
            s = (rc / qmax).clamp_min(1.0 / qmax).to(torch.float16)
            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -qmax, qmax).to(torch.int8)
            err = (t32 - q.float() * s.float()[:, None]).pow(2).mean().item()
            if err < best_err: best_q, best_s, best_err = q, s, err
        return best_q.cpu(), best_s.cpu()
    H = hessian.float().clone().to(t32.device)
    diag = torch.diag(H); dead = diag == 0
    damp_val = damp * (diag[~dead].mean() if not dead.all() else torch.tensor(1.0, device=H.device))
    idx = torch.arange(H.shape[0], device=H.device)
    H[idx, idx] += damp_val; H[dead, dead] = float(damp_val)
    perm = torch.argsort(torch.diag(H), descending=True); inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone(); W[:, dead[perm]] = 0; H = H[perm][:, perm]
    try:
        Hinv = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(Hinv)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except torch.linalg.LinAlgError:
        return quantize_int_gptq(weight, hessian=None, qmax=qmax)
    best_q, best_s, best_err = None, None, float("inf")
    for pct in [0.999, 0.9995, 0.9999, 0.99999, 1.0]:
        rc = torch.quantile(W.float().abs(), pct, dim=1) if pct < 1.0 else W.abs().amax(dim=1)
        s = (rc / qmax).clamp_min(1.0 / qmax).to(torch.float16)
        Q = _gptq_block_sweep(W, Hinv, s.float(), -qmax, qmax)
        err = (W - Q.float() * s.float()[:, None]).pow(2).mean().item()
        if err < best_err: best_q, best_s, best_err = Q, s, err
    return best_q[:, inv_perm].cpu(), best_s.cpu()

def collect_hessians(model: nn.Module, loader: "DistributedTokenLoader",
                     args: "Hyperparameters", device: torch.device,
                     grad_accum: int, num_batches: int = 64) -> "dict[str, Tensor]":
    """Run calibration forward passes and collect H = X^T X for each CastedLinear layer."""
    hessians: dict[str, Tensor] = {}
    hooks: list = []
    for name, mod in model.named_modules():
        if isinstance(mod, CastedLinear) and mod.weight.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            cols = mod.weight.shape[1]
            H = torch.zeros(cols, cols, dtype=torch.float32, device=device)
            hessians[name + ".weight"] = H
            def _make_hook(h):
                def fn(m, inp, out):
                    x = inp[0].detach()
                    if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
                    h.add_((x.float().T @ x.float()))
                return fn
            hooks.append(mod.register_forward_hook(_make_hook(H)))
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y = loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum)
            model(x, y)
    for h in hooks: h.remove()
    # All-reduce across ranks if distributed
    if dist.is_available() and dist.is_initialized():
        ws = dist.get_world_size()
        for H in hessians.values():
            dist.all_reduce(H, op=dist.ReduceOp.SUM); H.div_(num_batches * ws)
    else:
        for H in hessians.values(): H.div_(num_batches)
    model.train()
    return {k: v.cpu() for k, v in hessians.items()}

# --- MIXED-PRECISION BIT ALLOCATION ---
# Allocate int5/6/7 per tensor group based on Hessian sensitivity and byte budget.
# Most sensitive layers get int7, then int6, rest int5. Greedy fill within 16MB.

_MP_BYTES_PP_INT5 = 0.46     # estimated compressed bytes per param at int5
_MP_COST_PER_BIT = 0.24      # additional compressed bytes per extra bit above int5
_MP_NON_WEIGHT_RATIO = 0.55  # compression ratio for non-quantized tensors
_MP_PRUNE_HEADROOM = 0.02    # reserve 2% of byte budget for selective pruning

def allocate_bits_mixed(hessians: "dict[str, Tensor]", state_dict: "dict[str, Tensor]",
                        target_bytes: int = 16_000_000, code_bytes: int = 0
                        ) -> "tuple[dict[str, int], dict[str, float]]":
    """Allocate per-tensor bit widths based on Hessian sensitivity. Returns (name→bits, info)."""
    group_traces: dict[str, list[float]] = {}
    group_numel: dict[str, int] = {}
    tensor_group: dict[str, str] = {}
    for name, H in hessians.items():
        tr = float(torch.trace(H).item()) / H.shape[0]
        if not name.startswith("blocks."): continue
        dot2 = name.index(".", 7)
        li = int(name[7:dot2])
        gtype = "attn" if ".attn." in name else "mlp" if ".mlp." in name else "other"
        gkey = f"L{li}.{gtype}"
        group_traces.setdefault(gkey, []).append(tr)
        tensor_group[name] = gkey
        w = state_dict.get(name)
        if w is not None:
            group_numel[gkey] = group_numel.get(gkey, 0) + w.numel()
    gsens = {k: sum(v) / len(v) for k, v in group_traces.items()}
    ranked = sorted(gsens.items(), key=lambda x: x[1], reverse=True)
    # Estimate baseline (all int5)
    total_qn = sum(group_numel.values())
    nw_raw = sum(t.numel() * t.element_size() for n, t in state_dict.items() if n not in hessians)
    base = code_bytes + int(nw_raw * _MP_NON_WEIGHT_RATIO) + int(total_qn * _MP_BYTES_PP_INT5)
    budget = int(target_bytes * (1 - _MP_PRUNE_HEADROOM)) - base
    gbits: dict[str, int] = {g: 5 for g, _ in ranked}
    extra = 0
    if budget > 0 and ranked:
        # Top group → int7
        top = ranked[0][0]; tn = group_numel.get(top, 0)
        c7 = int(tn * _MP_COST_PER_BIT * 2)
        c6 = int(tn * _MP_COST_PER_BIT)
        if c7 <= budget: gbits[top] = 7; extra += c7
        elif c6 <= budget: gbits[top] = 6; extra += c6
        # Rest → int6 if fits
        for g, _ in ranked:
            if gbits[g] > 5: continue
            n = group_numel.get(g, 0)
            if n == 0: continue
            c = int(n * _MP_COST_PER_BIT)
            if extra + c <= budget: gbits[g] = 6; extra += c
    bits_map = {tn: gbits[g] for tn, g in tensor_group.items()}
    info = {"base_mb": base / 1e6, "extra_mb": extra / 1e6, "budget_mb": target_bytes / 1e6}
    return bits_map, info


def selective_prune(quant_data: "dict[str, Tensor]", target_bytes: int, code_bytes: int,
                    log_fn: "callable" = print) -> None:
    """Prune small quantized values (|q|<=2) to fit compressed artifact under target_bytes.
    Uses binary search with fast (zlib-1) probing, verified with real compressor."""
    # Collect pruneable entries: values with |q| in {1, 2} sorted by abs value
    entries: list[tuple[str, int]] = []  # (tensor_name, flat_index)
    for name, t in quant_data.items():
        if t.dtype != torch.int8 or t.ndim < 2: continue
        flat = t.reshape(-1)
        mask = (flat.abs() >= 1) & (flat.abs() <= 2)
        idxs = mask.nonzero(as_tuple=True)[0]
        vals = flat[idxs].abs().float()
        # Sort by ascending abs value (prune smallest first)
        order = vals.argsort()
        for oi in order:
            entries.append((name, int(idxs[oi].item())))
    if not entries:
        return

    def _compress_size(fast: bool = False) -> int:
        buf = io.BytesIO(); torch.save(quant_data, buf); raw = buf.getvalue()
        if fast: return len(zlib.compress(raw, 1)) + code_bytes
        raw_s = _byte_shuffle(raw, stride=2)
        blob = brotli.compress(raw_s, quality=11) if brotli else lzma.compress(raw, preset=6)
        return len(blob) + code_bytes

    no_prune_sz = _compress_size()
    log_fn(f"selective_prune: {len(entries)} candidates, unpruned={no_prune_sz/1e6:.2f}MB target={target_bytes/1e6:.2f}MB")
    if no_prune_sz <= target_bytes:
        log_fn("selective_prune: already fits"); return

    def _apply(n: int) -> None:
        for i in range(min(n, len(entries))):
            tn, fi = entries[i]
            quant_data[tn].reshape(-1)[fi] = 0

    def _trial(n: int, fast: bool = False) -> int:
        # Save, prune, measure, restore
        saved = {tn: quant_data[tn].clone() for tn in {e[0] for e in entries[:n]}}
        _apply(n)
        sz = _compress_size(fast=fast)
        for tn, t in saved.items(): quant_data[tn].copy_(t)
        return sz

    # Binary search with fast compressor
    full_prune_fast = _trial(len(entries), fast=True)
    no_prune_fast = _trial(0, fast=True)
    full_prune_real = _trial(len(entries))
    if full_prune_real > target_bytes:
        log_fn("selective_prune: even full prune not enough, applying all")
        _apply(len(entries)); return
    # Calibrate fast/real ratio
    fd = no_prune_fast - full_prune_fast; rd = no_prune_sz - full_prune_real
    ratio = rd / max(fd, 1)
    fast_target = no_prune_fast - int((no_prune_sz - target_bytes) / max(ratio, 0.01))
    lo, hi = 0, len(entries)
    while lo < hi:
        mid = (lo + hi) // 2
        if _trial(mid, fast=True) <= fast_target: hi = mid
        else: lo = mid + 1
    # Verify with real compressor
    real_sz = _trial(lo)
    while lo < len(entries) and real_sz > target_bytes:
        lo += max(1, len(entries) // 200); lo = min(lo, len(entries))
        real_sz = _trial(lo)
    log_fn(f"selective_prune: pruning {lo}/{len(entries)} ({100*lo/max(len(entries),1):.1f}%) to fit")
    _apply(lo)


# --- TRAINING ---

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    # Inductor compiler optimizations
    torch._inductor.config.fx_graph_cache = True
    torch._inductor.config.coordinate_descent_tuning = True
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(True)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(True)
    enable_math_sdp(True)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    ve_layers_list = [int(x) for x in args.ve_layers.split(",") if x.strip()] if args.ve_enabled else []
    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        ngram_buckets=args.ngram_buckets, ngram_heads=args.ngram_heads, ngram_orders=args.ngram_orders,
        ngram_dim_per_head=args.ngram_dim_per_head, ln_scale=args.ln_scale,
        rope_dims=args.rope_dims, xsa_last_n=args.xsa_last_n,
        ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=ve_layers_list,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compile_kwargs = dict(dynamic=False, fullgraph=True)
    if args.compile_mode != "default":
        compile_kwargs["mode"] = args.compile_mode
    log0(f"compile: mode={args.compile_mode} grad_accum={grad_accum_steps}")
    compiled_model = torch.compile(base_model, **compile_kwargs)
    # No DDP: Parallel Muon handles matrix param communication via reduce-scatter/all-gather.
    # Non-Muon params (scalars, embeddings) get manual coalesced all-reduce before Adam steps.
    model: nn.Module = compiled_model

    # Optimizer split: matrix weights via Muon (gather-scatter), scalars via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params
                     if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.engram is not None:
        matrix_params.append(base_model.engram.proj.weight)
        scalar_params.append(base_model.engram.embed.weight)
        scalar_params.append(base_model.engram.ngram_gate)
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
        scalar_params.append(base_model.skip_gates)
    scalar_params.append(base_model.smear_gate.gate)
    for ve in base_model.ve_modules:
        scalar_params.append(ve.embed.weight)
        scalar_params.append(ve.scale)
        if ve.proj is not None:
            matrix_params.append(ve.proj.weight)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        post_norm=args.muon_post_norm,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
        group["wd"] = args.muon_wd
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.adam_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_wd,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    # Build replicated param list for manual all-reduce (non-Muon params need gradient sync)
    replicated_params: list[nn.Parameter] = []
    replicated_params.append(base_model.tok_emb.weight)
    replicated_params.extend(scalar_params)
    if base_model.lm_head is not None:
        replicated_params.append(base_model.lm_head.weight)
    _repl_numels = [p.numel() for p in replicated_params]
    _repl_grad_buf = torch.zeros(sum(_repl_numels), device=device, dtype=torch.bfloat16) if distributed else None

    global _QUANT_BITS_QAT
    _QUANT_BITS_QAT = args.quant_bits
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    ema_state: dict[str, Tensor] | None = None
    ema_names: list[tuple[str, nn.Parameter]] | None = None
    ema_alpha = 1.0 - args.ema_decay
    if args.ema_decay > 0:
        ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}
        ema_names = [(n, p) for n, p in list(base_model.named_parameters()) + list(base_model.named_buffers()) if n in ema_state]

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params} swa_frac:{args.swa_frac} swa_every:{args.swa_every} quant_bits:{args.quant_bits}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    # Reserve Hessian collection time from training budget if using GPTQ
    gptq_reserve_ms = float(os.environ.get("GPTQ_RESERVE_MS", "9000"))
    effective_wallclock = args.max_wallclock_seconds - gptq_reserve_ms / 1000.0 if args.max_wallclock_seconds > 0 else 0.0
    if effective_wallclock <= 0 and args.max_wallclock_seconds > 0:
        effective_wallclock = args.max_wallclock_seconds

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * effective_wallclock if effective_wallclock > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        floor = args.lr_floor
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            raw = max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        else:
            step_ms = elapsed_ms / max(step, 1)
            warmdown_ms = args.warmdown_iters * step_ms
            remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
            raw = remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
        return max(raw, floor)

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            # Pre-warm both QAT-off and QAT-on compiled graphs during warmup
            if args.qat_start_frac > 0 and warmup_step == args.warmup_steps - 1:
                CastedLinear.qat_enabled = True
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        CastedLinear.qat_enabled = False
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # QAT: enable fake quantization late in training (one-way latch)
        if args.qat_start_frac > 0 and not CastedLinear.qat_enabled:
            qat_frac = elapsed_ms / max_wallclock_ms if max_wallclock_ms else step / max(args.iterations, 1)
            if qat_frac >= args.qat_start_frac:
                CastedLinear.qat_enabled = True
                if args.soft_round_qat:
                    CastedLinear.qat_soft_round = True
                    CastedLinear.qat_alpha = 1.0  # start smooth, ramp to 16
        # Ramp soft-round alpha 1→16 during QAT phase
        if CastedLinear.qat_soft_round and max_wallclock_ms is not None:
            qat_frac = elapsed_ms / max_wallclock_ms
            if qat_frac >= args.qat_start_frac:
                t = min((qat_frac - args.qat_start_frac) / max(1.0 - args.qat_start_frac, 1e-6), 1.0)
                CastedLinear.qat_alpha = 1.0 + 15.0 * t

        do_profile = args.profile_step_every > 0 and step > 0 and step % args.profile_step_every == 0
        if do_profile:
            _pev = {k: torch.cuda.Event(enable_timing=True) for k in ("s", "fb", "opt", "ema", "e")}
            _pev["s"].record()

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # Phase 1: Launch async reduce-scatters for Muon matrix params
        if distributed:
            optimizer_muon.launch_reduce_scatters()

        # Phase 2: Manual coalesced all-reduce for non-Muon replicated params
        if distributed and _repl_grad_buf is not None:
            off = 0
            for p, n in zip(replicated_params, _repl_numels):
                if p.grad is not None:
                    _repl_grad_buf[off:off + n].copy_(p.grad.reshape(-1).bfloat16())
                else:
                    _repl_grad_buf[off:off + n].zero_()
                off += n
            dist.all_reduce(_repl_grad_buf, op=dist.ReduceOp.AVG)
            off = 0
            for p, n in zip(replicated_params, _repl_numels):
                if p.grad is not None:
                    p.grad.copy_(_repl_grad_buf[off:off + n].reshape_as(p.grad))
                off += n

        if do_profile:
            _pev["fb"].record()

        if args.muon_momentum_warmup_steps > 0 and step <= args.muon_momentum_warmup_steps:
            frac = step / args.muon_momentum_warmup_steps
            muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
            for group in optimizer_muon.param_groups:
                group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        # Step Adam optimizers first (overlaps with Muon reduce-scatter in flight)
        for opt in optimizers:
            if opt is not optimizer_muon:
                opt.step()
        # Phase 3: Muon step — waits for RS, runs local NS, all-gathers
        optimizer_muon.step()
        zero_grad_all()

        if do_profile:
            _pev["opt"].record()

        if ema_state is not None:
            with torch.no_grad():
                for n, t in ema_names:
                    ema_state[n].lerp_(t.float(), ema_alpha)

        if do_profile:
            _pev["ema"].record(); _pev["e"].record(); torch.cuda.synchronize()
            fb = _pev["s"].elapsed_time(_pev["fb"])
            opt = _pev["fb"].elapsed_time(_pev["opt"])
            ema = _pev["opt"].elapsed_time(_pev["ema"])
            total = _pev["s"].elapsed_time(_pev["e"])
            misc = total - fb - opt - ema
            if distributed:
                vals = torch.tensor([fb, opt, ema, total, misc], device=device)
                gathered = [torch.zeros_like(vals) for _ in range(world_size)]
                dist.all_gather(gathered, vals)
                if master_process:
                    all_v = torch.stack(gathered)
                    mx = all_v.max(dim=0).values; mn = all_v.mean(dim=0)
                    log0(f"profile step={step} fwd+bwd: mean={mn[0]:.1f} max={mx[0]:.1f}  "
                         f"opt: mean={mn[1]:.1f} max={mx[1]:.1f}  ema: mean={mn[2]:.1f} max={mx[2]:.1f}  "
                         f"total: mean={mn[3]:.1f} max={mx[3]:.1f}  misc: mean={mn[4]:.1f} max={mx[4]:.1f}")
            else:
                log0(f"profile step={step} fwd+bwd:{fb:.1f} opt:{opt:.1f} ema:{ema:.1f} total:{total:.1f} misc:{misc:.1f}")

        # Norm diagnostics every 500 steps
        if step % 500 == 0 and step > 0:
            with torch.no_grad():
                norms = []
                for bi, blk in enumerate(base_model.blocks):
                    w_rms = sum(p.float().pow(2).mean().item() for p in blk.parameters()) ** 0.5
                    norms.append(f"L{bi}:{w_rms:.4f}")
                emb_rms = base_model.tok_emb.weight.float().pow(2).mean().item() ** 0.5
                near_zero = sum((p.abs() < 1e-6).sum().item() for p in base_model.parameters()) / n_params
                log0(f"norms emb:{emb_rms:.4f} blocks:[{','.join(norms)}] near_zero:{near_zero:.4f}")
                log0(f"muon ns_calls_total:{Muon.ns_calls} ns_calls_per_step:{Muon.ns_calls / step:.1f}")

        # SWA: accumulate checkpoints during warmdown
        if args.swa_frac > 0 and scale < args.swa_frac and step % args.swa_every == 0:
            with torch.no_grad():
                if swa_state is None:
                    swa_state = {n: t.detach().cpu().float().clone() for n, t in base_model.state_dict().items()}
                    swa_count = 1
                else:
                    for n, t in base_model.state_dict().items():
                        swa_state[n] += t.detach().cpu().float()
                    swa_count += 1

        step += 1
        approx_training_time_ms = elapsed_ms  # reuse value from LR schedule computation
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Sync wallclock cap only when a rank actually hits it (avoids all_reduce every step)
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None and reached_cap:
            reached_cap_tensor = torch.tensor(1, device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save raw checkpoint first
    if master_process:
        torch.save(base_model.state_dict(), "final_model_noswa.pt")
        log0(f"saved raw checkpoint: {os.path.getsize('final_model_noswa.pt')} bytes")
    ema_ready = ema_state is not None and step >= int(1.0 / (1.0 - args.ema_decay))
    if ema_ready:
        ema_sd = {n: t.to(dtype=base_model.state_dict()[n].dtype, device=base_model.state_dict()[n].device) for n, t in ema_state.items()}
        base_model.load_state_dict(ema_sd, strict=True)
        log0(f"using EMA weights for export (decay={args.ema_decay}, steps={step})")
    elif swa_count > 1:
        avg = {n: (t / swa_count).to(base_model.state_dict()[n].dtype) for n, t in swa_state.items()}
        base_model.load_state_dict(avg, strict=True)
        log0(f"using SWA weights for export (averaged {swa_count} checkpoints)")

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    # --- Hessian GPTQ Export with Mixed-Precision Allocation ---
    log0("gptq: collecting hessians...")
    t_gptq = time.perf_counter()
    calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    hessians = collect_hessians(base_model, calib_loader, args, device, grad_accum_steps, num_batches=64)
    log0(f"gptq: collected {len(hessians)} hessians in {time.perf_counter() - t_gptq:.1f}s")

    sd = base_model.state_dict()
    code_bytes = len(code.encode("utf-8"))

    # Mixed-precision bit allocation: int5/6/7 per layer based on Hessian sensitivity
    bits_map, mp_info = allocate_bits_mixed(hessians, sd, target_bytes=16_000_000, code_bytes=code_bytes)
    log0(f"mixed_precision: base={mp_info['base_mb']:.2f}MB extra={mp_info['extra_mb']:.2f}MB budget={mp_info['budget_mb']:.2f}MB")
    for name, bits in sorted(bits_map.items()):
        if bits != 6: log0(f"  {name}: int{bits}")

    # Apply Hessian GPTQ with per-tensor bit widths
    quant_result: dict[str, Tensor] = {}
    quant_scales: dict[str, Tensor] = {}
    quant_dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    for name, tensor in sd.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or name == "tok_emb.weight":
            passthrough[name] = t.to(torch.float16) if t.is_floating_point() else t
            continue
        if any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
            passthrough[name] = t.float()
            continue
        H = hessians.get(name)
        bits = bits_map.get(name, 6)
        qmax = (1 << (bits - 1)) - 1  # int5→15, int6→31, int7→63
        q, s = quantize_int_gptq(t, hessian=H, qmax=qmax)
        quant_result[name] = q
        quant_scales[name] = s
        quant_dtypes[name] = str(t.dtype).removeprefix("torch.")

    quant_obj = {"quantized": quant_result, "scales": quant_scales, "dtypes": quant_dtypes,
                 "passthrough": passthrough, "bits_map": {k: v for k, v in bits_map.items()},
                 "__quant_format__": "hessian_gptq_mixed_v1"}

    # Selective pruning to fit under 16MB
    selective_prune(quant_result, target_bytes=16_000_000, code_bytes=code_bytes, log_fn=log0)

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    # Byte-shuffle + best compression
    quant_shuffled = _byte_shuffle(quant_raw, stride=2)
    quant_blob_brotli = brotli.compress(quant_shuffled, quality=11) if brotli else None
    quant_blob_lzma = lzma.compress(quant_raw, preset=6)
    quant_blob_zstd = zstandard.ZstdCompressor(level=22).compress(quant_raw) if zstandard else None
    quant_blob_zlib = zlib.compress(quant_raw, level=9)
    candidates = [("zlib", quant_blob_zlib), ("lzma", quant_blob_lzma)]
    if quant_blob_zstd: candidates.append(("zstd", quant_blob_zstd))
    if quant_blob_brotli: candidates.append(("brotli+shuffle", quant_blob_brotli))
    compress_method, quant_blob = min(candidates, key=lambda x: len(x[1]))
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        log0(
            f"Serialized model quant+{compress_method}: {quant_file_bytes} bytes "
            f"(raw_torch:{quant_raw_bytes} zlib:{len(quant_blob_zlib)})"
        )
        log0(f"Total submission size quant+{compress_method}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    # Decompress: try all formats, then un-shuffle if needed
    decompressed = None
    for decompress_fn in ([brotli.decompress] if brotli else []) + [lzma.decompress, zlib.decompress] + ([zstandard.ZstdDecompressor().decompress] if zstandard else []):
        try:
            decompressed = decompress_fn(quant_blob_disk); break
        except Exception: continue
    if decompressed is None:
        raise RuntimeError("Failed to decompress quantized model")
    decompressed = _byte_unshuffle(decompressed)  # auto-detects BSHF header
    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")

    # Dequantize: handle both legacy int8 format and new Hessian GPTQ mixed format
    if quant_state.get("__quant_format__", "").startswith("hessian_gptq"):
        dequant_sd: dict[str, Tensor] = {}
        bm = quant_state.get("bits_map", {})
        for name, q in quant_state["quantized"].items():
            dtype = getattr(torch, quant_state["dtypes"][name])
            s = quant_state["scales"][name]
            if s.ndim > 0:
                dequant_sd[name] = (q.float() * s.float().view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype).contiguous()
            else:
                dequant_sd[name] = (q.float() * float(s.item())).to(dtype).contiguous()
        for name, t in quant_state["passthrough"].items():
            dequant_sd[name] = t
        base_model.load_state_dict(dequant_sd, strict=True)
    else:
        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if args.eval_seq_len > 0:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = eval_val_sliding(
            args, model, base_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(
            f"final_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms "
            f"eval_seq_len:{args.eval_seq_len} eval_stride:{args.eval_stride}"
        )
        log0(f"final_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    # Score-First TTT: score chunk, adapt on it, repeat. Supports SGD (full-weight) or LoRA mode.
    if args.ttt_lr > 0:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        chunk_seq = args.train_seq_len
        chunk_tokens = args.ttt_chunk_tokens
        assert chunk_tokens % chunk_seq == 0, f"TTT_CHUNK_TOKENS ({chunk_tokens}) must be divisible by TRAIN_SEQ_LEN ({chunk_seq})"
        n_seqs = chunk_tokens // chunk_seq
        total_val = val_tokens.numel() - 1
        n_chunks = total_val // chunk_tokens
        ttt_init_sd = copy.deepcopy(base_model.state_dict())
        use_lora = args.ttt_mode == "lora"
        # LoRA TTT: inject lightweight adapters on Q/V projections
        lora_modules: list[tuple[nn.Module, str, nn.Module]] = []
        if use_lora:
            rank = args.ttt_lora_rank
            for blk in base_model.blocks:
                for attr in ["c_q", "c_v"]:
                    orig = getattr(blk.attn, attr)
                    out_f, in_f = orig.weight.shape
                    lora_A = nn.Linear(in_f, rank, bias=False, device=device, dtype=torch.float32)
                    lora_B = nn.Linear(rank, out_f, bias=False, device=device, dtype=torch.float32)
                    nn.init.normal_(lora_A.weight, std=1.0 / in_f**0.5)
                    nn.init.zeros_(lora_B.weight)
                    lora_mod = nn.Sequential(lora_A, lora_B)
                    lora_modules.append((blk.attn, attr, orig))
                    # Monkey-patch: wrap original forward to add LoRA output
                    orig._lora = lora_mod
                    orig._orig_forward = orig.forward
                    def _make_lora_fwd(o):
                        def fwd(x):
                            return o._orig_forward(x) + o._lora(x)
                        return fwd
                    orig.forward = _make_lora_fwd(orig)
            ttt_params = [p for _, _, o in lora_modules for p in o._lora.parameters()]
            ttt_opt = torch.optim.AdamW(ttt_params, lr=args.ttt_lr, weight_decay=0.01)
            # Polyak averaging state
            polyak_state = {id(p): p.data.clone() for p in ttt_params}
            polyak_decay = args.ttt_polyak_decay
            # Freeze non-LoRA params
            for p in base_model.parameters():
                p.requires_grad_(False)
            for p in ttt_params:
                p.requires_grad_(True)
            log0(f"ttt_lora: rank={rank} params={sum(p.numel() for p in ttt_params)} polyak={polyak_decay}")
        else:
            ttt_params = list(base_model.parameters())
            ttt_opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=0.9)
        ttt_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        ttt_token_count = torch.zeros((), device=device, dtype=torch.float64)
        ttt_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        log0(f"ttt: mode={args.ttt_mode} {n_chunks} chunks, {n_seqs} seqs/chunk, {args.ttt_epochs} ep, lr={args.ttt_lr}")
        for ci in range(n_chunks):
            cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(n_chunks - 1, 1)))
            for pg in ttt_opt.param_groups:
                pg["lr"] = cos_lr
            start = ci * chunk_tokens
            chunk = val_tokens[start:start + chunk_tokens + 1].to(device=device, dtype=torch.int64)
            x_all = chunk[:-1].reshape(n_seqs, chunk_seq)
            y_all = chunk[1:].reshape(n_seqs, chunk_seq)
            # Score entire chunk (no gradient, inference_mode guarantees no weight mutation)
            base_model.eval()
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits_all = base_model.forward_logits(x_all)
                loss_all = F.cross_entropy(logits_all.float().reshape(-1, logits_all.size(-1)),
                                           y_all.reshape(-1), reduction="none").to(torch.float64)
                ttt_loss_sum += loss_all.sum()
                ttt_token_count += float(y_all.numel())
                prev_flat, tgt_flat = x_all.reshape(-1), y_all.reshape(-1)
                tb = base_bytes_lut[tgt_flat].to(torch.float64)
                tb += (has_leading_space_lut[tgt_flat] & ~is_boundary_token_lut[prev_flat]).to(torch.float64)
                ttt_byte_count += tb.sum()
            # Adapt on scored chunk; adaptation accumulates across chunks
            base_model.train()
            for _ in range(args.ttt_epochs):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    adapt_loss = base_model(x_all, y_all)
                ttt_opt.zero_grad()
                adapt_loss.backward()
                torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                ttt_opt.step()
                # Polyak averaging for LoRA
                if use_lora:
                    with torch.no_grad():
                        for p in ttt_params:
                            polyak_state[id(p)].mul_(polyak_decay).add_(p.data, alpha=1 - polyak_decay)
            if ci % 100 == 0 or ci == n_chunks - 1:
                el = time.perf_counter() - t_ttt
                eta = el / (ci + 1) * (n_chunks - ci - 1)
                log0(f"ttt {ci+1}/{n_chunks} time:{el:.0f}s eta:{eta:.0f}s lr:{cos_lr:.5f}")
        # Clean up: restore original weights, remove LoRA patches
        if use_lora:
            for parent, attr, orig in lora_modules:
                orig.forward = orig._orig_forward
                del orig._lora, orig._orig_forward
        base_model.load_state_dict(ttt_init_sd, strict=True)
        for p in base_model.parameters():
            p.requires_grad_(True)
        ttt_val_loss = (ttt_loss_sum / ttt_token_count).item()
        ttt_bpt = ttt_val_loss / math.log(2.0)
        ttt_val_bpb = ttt_bpt * ttt_token_count.item() / ttt_byte_count.item()
        torch.cuda.synchronize()
        log0(f"ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms "
             f"mode:{args.ttt_mode} chunks:{n_chunks} epochs:{args.ttt_epochs}")
        log0(f"ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
