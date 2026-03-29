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
import zlib
from pathlib import Path

try:
    import zstandard
except ImportError:
    zstandard = None

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

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
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    quant_bits = int(os.environ.get("QUANT_BITS", 8))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))
    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 0))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 32))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 3))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    muon_wd = float(os.environ.get("MUON_WD", 0.0))
    adam_wd = float(os.environ.get("ADAM_WD", 0.0))
    qat_start_frac = float(os.environ.get("QAT_START_FRAC", 0.0))
    swa_frac = float(os.environ.get("SWA_FRAC", 0.0))
    swa_every = int(os.environ.get("SWA_EVERY", 100))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.0))
    neural_temp = float(os.environ.get("NEURAL_TEMP", 0.85))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "0")))
    rope_dims = int(os.environ.get("ROPE_DIMS", 0))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", 0))
    ttt_lr = float(os.environ.get("TTT_LR", 0.0))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", 32768))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def zeropower_via_newtonschulz5_batched(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Batched Newton-Schulz for [B, M, N] tensors. Uses torch.bmm.
    assert G.ndim == 3, f"Expected 3D tensor, got {G.ndim}D"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    norms = X.norm(dim=(1, 2), keepdim=True)
    X = X / (norms + eps)
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = torch.bmm(X, X.transpose(1, 2))
        B = b * A + c * torch.bmm(A, A)
        X = a * X + torch.bmm(B, X)
    return X.transpose(1, 2) if transposed else X


class Muon(torch.optim.Optimizer):
    ns_calls: int = 0

    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom, bs = group["lr"], group["momentum"], group["backend_steps"]
            nesterov, wd = group["nesterov"], group.get("wd", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nesterov else buf
                if g.ndim == 3:
                    u = zeropower_via_newtonschulz5_batched(g, steps=bs)
                    u *= max(1, g.size(1) / g.size(2)) ** 0.5
                elif g.ndim == 2:
                    u = zeropower_via_newtonschulz5(g, steps=bs)
                    u *= max(1, g.size(0) / g.size(1)) ** 0.5
                else:
                    u = g
                Muon.ns_calls += 1
                p.add_(u.to(p.dtype), alpha=-lr)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

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

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
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

def quantize_state_dict_int8(state_dict: dict[str, Tensor], quant_bits: int = 8):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
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


# -----------------------------
# DATA LOADING 
# -----------------------------

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

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


INT8_KEEP_FLOAT_MAX_NUMEL = 65_536

class CastedLinear(nn.Linear):
    qat_enabled: bool = False
    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self.qat_enabled and self.training and w.ndim == 2 and w.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
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


def _fq(w: Tensor) -> Tensor:
    # Apply fake quantization for QAT if enabled, otherwise passthrough.
    if CastedLinear.qat_enabled and w.requires_grad and w.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
        return _fake_quantize(w)
    return w

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, rope_dims: int = 0, use_xsa: bool = False):
        super().__init__()
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = dim // num_heads
        self.use_xsa = use_xsa
        self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.rope_dims, base=rope_base)

    def forward(self, x: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor, w_proj: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = F.linear(x, _fq(w_q)).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = F.linear(x, _fq(w_k)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = F.linear(x, _fq(w_v)).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
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
        return F.linear(y, _fq(w_proj))


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.mod = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        self.proj._zero_init = True
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        t = token_ids.to(torch.int32)
        h = torch.zeros_like(t, dtype=torch.long)
        h[:, 1:] = (36313 * t[:, 1:] ^ 27191 * t[:, :-1]) % self.mod
        return self.proj(self.embed(h)) * self.scale


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float,
                 qk_gain_init: float, ln_scale: float = 1.0, rope_dims: int = 0, use_xsa: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                                        rope_dims=rope_dims, use_xsa=use_xsa)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale = ln_scale

    def forward(self, x: Tensor, x0: Tensor, w_q: Tensor, w_k: Tensor, w_v: Tensor,
                w_aproj: Tensor, w_fc: Tensor, w_mproj: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        s = self.ln_scale
        attn_out = self.attn(self.attn_norm(x) * s, w_q, w_k, w_v, w_aproj)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_h = F.leaky_relu(F.linear(self.mlp_norm(x) * s, _fq(w_fc)), negative_slope=0.5)
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * F.linear(mlp_h.square(), _fq(w_mproj))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, num_heads: int,
                 num_kv_heads: int, mlp_mult: int, tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 bigram_vocab_size: int = 0, bigram_dim: int = 128,
                 ln_scale: bool = False, rope_dims: int = 0, xsa_last_n: int = 0):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        kv_dim = (num_kv_heads * model_dim) // num_heads
        hidden = mlp_mult * model_dim
        # Parameter banks: contiguous 3D tensors for batched Muon Newton-Schulz
        self.bank_sq = nn.Parameter(torch.empty(2 * num_layers, model_dim, model_dim))
        self.bank_kv = nn.Parameter(torch.empty(2 * num_layers, kv_dim, model_dim))
        self.bank_fc = nn.Parameter(torch.empty(num_layers, hidden, model_dim))
        self.bank_pr = nn.Parameter(torch.empty(num_layers, model_dim, hidden))
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
                  ln_scale=1.0 / (i + 1) ** 0.5 if ln_scale else 1.0, rope_dims=rope_dims,
                  use_xsa=(i >= num_layers - xsa_last_n))
            for i in range(num_layers)
        ])
        self.smear_gate = SmearGate(model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()
        # Pre-slice bank views for compiler-friendly access (no dynamic indexing in forward)
        self._w_q = [self.bank_sq[2 * i] for i in range(num_layers)]
        self._w_ap = [self.bank_sq[2 * i + 1] for i in range(num_layers)]
        self._w_k = [self.bank_kv[2 * i] for i in range(num_layers)]
        self._w_v = [self.bank_kv[2 * i + 1] for i in range(num_layers)]
        self._w_fc = [self.bank_fc[i] for i in range(num_layers)]
        self._w_mp = [self.bank_pr[i] for i in range(num_layers)]

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for i in range(self.num_layers):
            nn.init.orthogonal_(self.bank_sq.data[2 * i], gain=1.0)      # c_q
            nn.init.zeros_(self.bank_sq.data[2 * i + 1])                   # attn.proj (zero_init)
            nn.init.orthogonal_(self.bank_kv.data[2 * i], gain=1.0)      # c_k
            nn.init.orthogonal_(self.bank_kv.data[2 * i + 1], gain=1.0)  # c_v
            nn.init.orthogonal_(self.bank_fc.data[i], gain=1.0)           # mlp.fc
            nn.init.zeros_(self.bank_pr.data[i])                           # mlp.proj (zero_init)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.weight.ndim == 2:
                if min(module.weight.shape) >= 64 and not getattr(module, "_zero_init", False):
                    nn.init.orthogonal_(module.weight, gain=1.0)
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_blocks(self, x: Tensor) -> Tensor:
        x = self.smear_gate(x)
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0, self._w_q[i], self._w_k[i], self._w_v[i],
                               self._w_ap[i], self._w_fc[i], self._w_mp[i])
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            j = self.num_encoder_layers + i
            x = self.blocks[j](x, x0, self._w_q[j], self._w_k[j], self._w_v[j],
                               self._w_ap[j], self._w_fc[j], self._w_mp[j])
        return self.final_norm(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self._run_blocks(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self._run_blocks(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def export_state_dict(self) -> dict[str, Tensor]:
        sd: dict[str, Tensor] = {}
        for i in range(self.num_layers):
            p = f"blocks.{i}"
            sd[f"{p}.attn.c_q.weight"] = self.bank_sq[2 * i].detach()
            sd[f"{p}.attn.proj.weight"] = self.bank_sq[2 * i + 1].detach()
            sd[f"{p}.attn.c_k.weight"] = self.bank_kv[2 * i].detach()
            sd[f"{p}.attn.c_v.weight"] = self.bank_kv[2 * i + 1].detach()
            sd[f"{p}.mlp.fc.weight"] = self.bank_fc[i].detach()
            sd[f"{p}.mlp.proj.weight"] = self.bank_pr[i].detach()
        for n, t in self.state_dict().items():
            if not n.startswith("bank_"):
                sd[n] = t
        return sd

    def load_export_state_dict(self, sd: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for i in range(self.num_layers):
                p = f"blocks.{i}"
                self.bank_sq.data[2 * i].copy_(sd[f"{p}.attn.c_q.weight"])
                self.bank_sq.data[2 * i + 1].copy_(sd[f"{p}.attn.proj.weight"])
                self.bank_kv.data[2 * i].copy_(sd[f"{p}.attn.c_k.weight"])
                self.bank_kv.data[2 * i + 1].copy_(sd[f"{p}.attn.c_v.weight"])
                self.bank_fc.data[i].copy_(sd[f"{p}.mlp.fc.weight"])
                self.bank_pr.data[i].copy_(sd[f"{p}.mlp.proj.weight"])
            own = self.state_dict()
            for k, v in sd.items():
                if k in own and not k.startswith("bank_"):
                    own[k].copy_(v)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5, zeropower_via_newtonschulz5_batched

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    zeropower_via_newtonschulz5_batched = torch.compile(zeropower_via_newtonschulz5_batched)

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
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

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

    base_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        ln_scale=args.ln_scale, rope_dims=args.rope_dims, xsa_last_n=args.xsa_last_n,
    ).to(device).bfloat16()
    # Promote weight matrices to float32 (banks + any remaining CastedLinear like bigram.proj, lm_head)
    for bank in [base_model.bank_sq, base_model.bank_kv, base_model.bank_fc, base_model.bank_pr]:
        bank.data = bank.data.float()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False,
                           gradient_as_bucket_view=True) if distributed else compiled_model

    # Optimizer split:
    # Optimizer split: banks via Muon, scalars/vectors via Adam
    bank_params: list[Tensor] = [base_model.bank_sq, base_model.bank_kv, base_model.bank_fc, base_model.bank_pr]
    if base_model.bigram is not None:
        bank_params.append(base_model.bigram.proj.weight)
    scalar_params = list(p for n, p in base_model.blocks.named_parameters())
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear_gate.gate)
    if base_model.bigram is not None:
        scalar_params.append(base_model.bigram.embed.weight)
        scalar_params.append(base_model.bigram.scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        bank_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
        group["wd"] = args.muon_wd
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

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

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    bigram_lut = None

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
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
            CastedLinear.qat_enabled = qat_frac >= args.qat_start_frac

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

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
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        if ema_state is not None:
            with torch.no_grad():
                for n, t in ema_names:
                    ema_state[n].lerp_(t.float(), ema_alpha)

        # Norm diagnostics every 500 steps
        if step % 500 == 0 and step > 0:
            with torch.no_grad():
                norms = []
                for bi, blk in enumerate(base_model.blocks):
                    w_rms = sum(p.float().pow(2).mean().item() for p in blk.parameters()) ** 0.5
                    norms.append(f"L{bi}:{w_rms:.4f}")
                emb_rms = base_model.tok_emb.weight.float().pow(2).mean().item() ** 0.5
                near_zero = sum((p.abs() < 1e-6).sum().item() for p in base_model.parameters()) / n_params
                bank_rms = sum(b.float().pow(2).mean().item() for b in [base_model.bank_sq, base_model.bank_kv, base_model.bank_fc, base_model.bank_pr]) ** 0.5
                log0(f"norms emb:{emb_rms:.4f} banks:{bank_rms:.4f} blocks:[{','.join(norms)}] near_zero:{near_zero:.4f}")
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

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.export_state_dict(), quant_bits=args.quant_bits)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob_zlib = zlib.compress(quant_raw, level=9)
    quant_blob_zstd = zstandard.ZstdCompressor(level=22).compress(quant_raw) if zstandard else quant_blob_zlib
    quant_blob = quant_blob_zstd if len(quant_blob_zstd) < len(quant_blob_zlib) else quant_blob_zlib
    compress_method = "zstd" if len(quant_blob_zstd) < len(quant_blob_zlib) else "zlib"
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model quant+{compress_method}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x "
            f"zstd:{len(quant_blob_zstd)} zlib:{len(quant_blob_zlib)})"
        )
        log0(f"Total submission size quant+{compress_method}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    try:
        decompressed = zstandard.ZstdDecompressor().decompress(quant_blob_disk) if zstandard else zlib.decompress(quant_blob_disk)
    except Exception:
        decompressed = zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
    base_model.load_export_state_dict(dequantize_state_dict_int8(quant_state))
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

    # Score-First TTT: score chunk, adapt on it, restore weights, repeat
    if args.ttt_lr > 0:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        chunk_seq = args.train_seq_len
        chunk_tokens = args.ttt_chunk_tokens
        assert chunk_tokens % chunk_seq == 0, f"TTT_CHUNK_TOKENS ({chunk_tokens}) must be divisible by TRAIN_SEQ_LEN ({chunk_seq})"
        n_seqs = chunk_tokens // chunk_seq
        total_val = val_tokens.numel() - 1
        n_chunks = total_val // chunk_tokens
        # Save initial state for per-chunk reset
        ttt_init_sd = copy.deepcopy(base_model.state_dict())
        # Freeze matrix params (weight matrices) — only adapt embeds, scalars, gates, norms
        ttt_params = []
        for n, p in base_model.named_parameters():
            if p.ndim >= 2 and min(p.shape) >= 64:
                p.requires_grad_(False)
            else:
                ttt_params.append(p)
        ttt_opt = torch.optim.SGD(ttt_params, lr=args.ttt_lr, momentum=0.9)
        ttt_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        ttt_token_count = torch.zeros((), device=device, dtype=torch.float64)
        ttt_byte_count = torch.zeros((), device=device, dtype=torch.float64)
        log0(f"ttt: {n_chunks} chunks, {n_seqs} seqs/chunk, {args.ttt_epochs} ep, lr={args.ttt_lr}, adapted_params={len(ttt_params)}")
        for ci in range(n_chunks):
            # Reset model to initial state each chunk
            base_model.load_state_dict(ttt_init_sd, strict=True)
            ttt_opt.state.clear()
            cos_lr = args.ttt_lr * 0.5 * (1.0 + math.cos(math.pi * ci / max(n_chunks - 1, 1)))
            for pg in ttt_opt.param_groups:
                pg["lr"] = cos_lr
            start = ci * chunk_tokens
            chunk = val_tokens[start:start + chunk_tokens + 1].to(device=device, dtype=torch.int64)
            x_all = chunk[:-1].reshape(n_seqs, chunk_seq)
            y_all = chunk[1:].reshape(n_seqs, chunk_seq)
            # Score entire chunk in one batched forward (no gradient)
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
            # Adapt on scored chunk (batched), then discard adaptation
            base_model.train()
            for _ in range(args.ttt_epochs):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    adapt_loss = base_model(x_all, y_all)
                ttt_opt.zero_grad()
                adapt_loss.backward()
                torch.nn.utils.clip_grad_norm_(ttt_params, 1.0)
                ttt_opt.step()
            if ci % 100 == 0 or ci == n_chunks - 1:
                el = time.perf_counter() - t_ttt
                eta = el / (ci + 1) * (n_chunks - ci - 1)
                tok_s = float(ttt_token_count) / max(el, 1e-9)
                log0(f"ttt {ci+1}/{n_chunks} time:{el:.0f}s eta:{eta:.0f}s tok/s:{tok_s:.0f} lr:{cos_lr:.5f}")
        # Restore initial weights (TTT should not permanently modify model)
        base_model.load_state_dict(ttt_init_sd, strict=True)
        for p in base_model.parameters():
            p.requires_grad_(True)
        ttt_val_loss = (ttt_loss_sum / ttt_token_count).item()
        ttt_bpt = ttt_val_loss / math.log(2.0)
        ttt_val_bpb = ttt_bpt * ttt_token_count.item() / ttt_byte_count.item()
        torch.cuda.synchronize()
        log0(
            f"ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
            f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms "
            f"chunks:{n_chunks} epochs:{args.ttt_epochs} lr:{args.ttt_lr}"
        )
        log0(f"ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
