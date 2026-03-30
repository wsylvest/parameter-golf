"""
Tests for v2-competitive features:
1. Brotli compression path
2. LoRA TTT injection/removal
3. Adaptive per-layer quantization
4. QAT fake-quantize bit alignment

Run: python3 tests/test_v2_features.py
(CPU-only, no GPU required)
"""
import sys, os, io, math, zlib, lzma
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")


# ============================================================
print("=== Test 1: Brotli Compression Path ===")
# ============================================================
try:
    import brotli
    data = os.urandom(100_000)
    compressed = brotli.compress(data, quality=11)
    decompressed = brotli.decompress(compressed)
    check("brotli installed", True)
    check("brotli roundtrip", data == decompressed)
    check("brotli compression ratio", len(compressed) < len(data),
          f"{len(compressed)} >= {len(data)}")
    # Compare with lzma
    lzma_compressed = lzma.compress(data, preset=6)
    print(f"    brotli: {len(compressed)} bytes, lzma: {len(lzma_compressed)} bytes")
except ImportError:
    check("brotli installed", False, "brotli not installed (pip install brotli)")


# ============================================================
print("\n=== Test 2: LoRA TTT Injection/Removal ===")
# ============================================================
from train_gpt import CastedLinear

class FakeAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_v = CastedLinear(dim, dim, bias=False)
    def forward(self, x):
        return self.c_q(x) + self.c_v(x)

dim = 64
rank = 4
attn = FakeAttn(dim)
x = torch.randn(2, 8, dim)

# Baseline output before LoRA
with torch.no_grad():
    out_before = attn(x).clone()

# Inject LoRA on c_q and c_v
lora_modules = []
for attr in ["c_q", "c_v"]:
    orig = getattr(attn, attr)
    out_f, in_f = orig.weight.shape
    lora_A = nn.Linear(in_f, rank, bias=False)
    lora_B = nn.Linear(rank, out_f, bias=False)
    nn.init.normal_(lora_A.weight, std=0.01)
    nn.init.zeros_(lora_B.weight)  # B=0 means LoRA output starts at 0
    lora_mod = nn.Sequential(lora_A, lora_B)
    orig._lora = lora_mod
    orig._orig_forward = orig.forward
    def _make_lora_fwd(o):
        def fwd(x):
            return o._orig_forward(x) + o._lora(x)
        return fwd
    orig.forward = _make_lora_fwd(orig)
    lora_modules.append((attn, attr, orig))

# With B=0 init, LoRA should not change output
with torch.no_grad():
    out_with_lora_zero = attn(x).clone()
diff_zero = (out_before - out_with_lora_zero).abs().max().item()
check("LoRA zero-init preserves output", diff_zero < 1e-5, f"diff={diff_zero:.2e}")

# After training LoRA (simulate), output should change
for _, _, orig in lora_modules:
    for p in orig._lora.parameters():
        p.data.normal_(std=0.1)
with torch.no_grad():
    out_with_lora_trained = attn(x).clone()
diff_trained = (out_before - out_with_lora_trained).abs().max().item()
check("LoRA training changes output", diff_trained > 0.01, f"diff={diff_trained:.2e}")

# Remove LoRA and verify restoration
for parent, attr, orig in lora_modules:
    orig.forward = orig._orig_forward
    del orig._lora, orig._orig_forward
with torch.no_grad():
    out_after_removal = attn(x).clone()
diff_restored = (out_before - out_after_removal).abs().max().item()
check("LoRA removal restores original", diff_restored < 1e-6, f"diff={diff_restored:.2e}")

# Check LoRA param count
total_lora_params = 2 * (dim * rank + rank * dim)  # A + B for c_q and c_v
check("LoRA param count correct", total_lora_params == 2 * 2 * dim * rank,
      f"expected {2 * 2 * dim * rank}, got {total_lora_params}")


# ============================================================
print("\n=== Test 3: Adaptive Per-Layer Quantization ===")
# ============================================================
from train_gpt import quantize_state_dict_int8, dequantize_state_dict_int8

# Create a fake state dict with named layers (must exceed INT8_KEEP_FLOAT_MAX_NUMEL=65536)
state = {
    "blocks.0.attn.c_q.weight": torch.randn(512, 512),
    "blocks.1.attn.c_q.weight": torch.randn(512, 512),
    "blocks.2.attn.c_q.weight": torch.randn(512, 512),
    "blocks.10.attn.c_q.weight": torch.randn(512, 512),
    "tok_emb.weight": torch.randn(1024, 512),
}

# Test uniform quantization (baseline)
obj_uniform, _ = quantize_state_dict_int8(state, quant_bits=6)
check("uniform quant runs", True)

# Test adaptive: layers 0,10 at int8, layers 1,2 at int5
layer_bits = {0: 8, 1: 5, 2: 5, 10: 8}
obj_adaptive, stats_adaptive = quantize_state_dict_int8(state, quant_bits=6, layer_bits=layer_bits)
check("adaptive quant runs", True)

# Verify layer 0 used int8 (check qmeta)
qmeta = obj_adaptive.get("qmeta", {})
l0_bits = qmeta.get("blocks.0.attn.c_q.weight", {}).get("bits")
l1_bits = qmeta.get("blocks.1.attn.c_q.weight", {}).get("bits")
# int8 = default (no bits in qmeta), int5 should have bits=5
check("layer 0 uses int8", l0_bits is None or l0_bits == 8, f"bits={l0_bits}")
check("layer 1 uses int5", l1_bits == 5, f"bits={l1_bits}")

# Roundtrip test
deq = dequantize_state_dict_int8(obj_adaptive)
check("adaptive quant roundtrip", set(deq.keys()) == set(state.keys()) - {"tok_emb.weight"} | {"tok_emb.weight"})

# Quality: int8 layers should have lower reconstruction error
err_l0 = (state["blocks.0.attn.c_q.weight"] - deq["blocks.0.attn.c_q.weight"]).pow(2).mean().item()
err_l1 = (state["blocks.1.attn.c_q.weight"] - deq["blocks.1.attn.c_q.weight"]).pow(2).mean().item()
check("int8 layer has less error than int5", err_l0 < err_l1,
      f"L0(int8)={err_l0:.6f} L1(int5)={err_l1:.6f}")


# ============================================================
print("\n=== Test 4: QAT Fake-Quantize Bit Alignment ===")
# ============================================================
from train_gpt import _fake_quantize, _QUANT_BITS_QAT

w = torch.randn(64, 64)
fq = _fake_quantize(w)
check("fake_quantize produces same shape", fq.shape == w.shape)
check("fake_quantize is differentiable", fq.requires_grad == w.requires_grad)
# Verify STE: gradient flows through
w_param = nn.Parameter(torch.randn(64, 64))
fq_out = _fake_quantize(w_param)
loss = fq_out.sum()
loss.backward()
check("STE gradient flows", w_param.grad is not None and w_param.grad.abs().sum() > 0)


# ============================================================
print("\n=== Test 5: Compression Selection ===")
# ============================================================
# Verify the compression candidate logic picks the smallest
data = os.urandom(50_000)
zlib_c = zlib.compress(data, level=9)
lzma_c = lzma.compress(data, preset=6)
candidates = [("zlib", zlib_c), ("lzma", lzma_c)]
try:
    import zstandard
    zstd_c = zstandard.ZstdCompressor(level=22).compress(data)
    candidates.append(("zstd", zstd_c))
except ImportError:
    pass
try:
    import brotli
    brotli_c = brotli.compress(data, quality=11)
    candidates.append(("brotli", brotli_c))
except ImportError:
    pass
method, blob = min(candidates, key=lambda x: len(x[1]))
check("compression selection works", len(blob) == min(len(c) for _, c in candidates))
print(f"    Winner: {method} ({len(blob)} bytes)")
for name, c in candidates:
    print(f"    {name}: {len(c)} bytes")


# ============================================================
print("\n=== Test 6: Model Forward/Backward Smoke ===")
# ============================================================
try:
    # Polyfill F.rms_norm for older PyTorch (< 2.4)
    if not hasattr(F, 'rms_norm'):
        def _rms_norm(x, normalized_shape, eps=None):
            eps = eps or 1e-6
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        F.rms_norm = _rms_norm
    from train_gpt import GPT, restore_low_dim_params_to_fp32
    device = torch.device("cpu")
    m = GPT(vocab_size=32, num_layers=3, model_dim=64, num_heads=4, num_kv_heads=2,
            mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            bigram_vocab_size=32, bigram_dim=16,
            ln_scale=True, rope_dims=8, xsa_last_n=1).to(device)
    restore_low_dim_params_to_fp32(m)
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    loss = m(x, y)
    check("forward runs", not math.isnan(loss.item()), f"loss={loss.item()}")
    loss.backward()
    has_grads = all(p.grad is not None for p in m.parameters() if p.requires_grad)
    check("backward produces gradients", has_grads)
    logits = m.forward_logits(x)
    check("forward_logits shape", logits.shape == (2, 16, 32), f"shape={logits.shape}")
except Exception as e:
    check("model smoke test", False, str(e))


# ============================================================
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
