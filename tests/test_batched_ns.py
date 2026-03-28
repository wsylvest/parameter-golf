# ABOUTME: Tests for batched Newton-Schulz helper vs sequential single-matrix helper.
# ABOUTME: Verifies numerical equivalence within bf16 tolerance.

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import without triggering full module load (no sentencepiece/CUDA needed)
import importlib.util
spec = importlib.util.spec_from_file_location("train_gpt", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_gpt.py"))

# We can't import the full module (needs torch.distributed, sentencepiece, etc.)
# So we directly define the functions for testing.

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
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


def zeropower_via_newtonschulz5_batched(G, steps=10, eps=1e-7):
    assert G.ndim == 3
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


BF16_TOL = 0.02  # bf16 has ~0.4% relative error; 5 NS iterations compound this

def test_single_item_batch():
    """Batched helper with B=1 should match sequential helper within bf16 noise."""
    torch.manual_seed(42)
    G = torch.randn(512, 512)
    seq_result = zeropower_via_newtonschulz5(G, steps=5)
    batch_result = zeropower_via_newtonschulz5_batched(G.unsqueeze(0), steps=5).squeeze(0)
    diff = (seq_result.float() - batch_result.float()).abs().max().item()
    print(f"Single item batch: max diff = {diff:.2e}")
    assert diff < BF16_TOL, f"Too large: {diff}"


def test_multi_item_batch():
    """Batched result should match stacked sequential results."""
    torch.manual_seed(42)
    B = 8
    Gs = [torch.randn(512, 512) for _ in range(B)]
    seq_results = torch.stack([zeropower_via_newtonschulz5(g, steps=5) for g in Gs])
    batch_result = zeropower_via_newtonschulz5_batched(torch.stack(Gs), steps=5)
    diff = (seq_results.float() - batch_result.float()).abs().max().item()
    print(f"Multi item batch (B={B}): max diff = {diff:.2e}")
    assert diff < BF16_TOL, f"Too large: {diff}"


def test_wide_matrix():
    """Wide matrices (rows < cols) should not need transpose."""
    torch.manual_seed(42)
    G = torch.randn(256, 512)
    seq_result = zeropower_via_newtonschulz5(G, steps=5)
    batch_result = zeropower_via_newtonschulz5_batched(G.unsqueeze(0), steps=5).squeeze(0)
    diff = (seq_result.float() - batch_result.float()).abs().max().item()
    print(f"Wide matrix (256x512): max diff = {diff:.2e}")
    assert diff < BF16_TOL, f"Too large: {diff}"


def test_tall_matrix():
    """Tall matrices (rows > cols) trigger transpose path."""
    torch.manual_seed(42)
    G = torch.randn(1536, 512)
    seq_result = zeropower_via_newtonschulz5(G, steps=5)
    batch_result = zeropower_via_newtonschulz5_batched(G.unsqueeze(0), steps=5).squeeze(0)
    diff = (seq_result.float() - batch_result.float()).abs().max().item()
    print(f"Tall matrix (1536x512): max diff = {diff:.2e}")
    assert diff < BF16_TOL, f"Too large: {diff}"


def test_all_shape_buckets():
    """Test all 4 shape buckets from the 11L model."""
    shapes = [(512, 512), (256, 512), (1536, 512), (512, 1536)]
    torch.manual_seed(42)
    for M, N in shapes:
        B = 11
        Gs = [torch.randn(M, N) for _ in range(B)]
        seq_results = torch.stack([zeropower_via_newtonschulz5(g, steps=5) for g in Gs])
        batch_result = zeropower_via_newtonschulz5_batched(torch.stack(Gs), steps=5)
        diff = (seq_results.float() - batch_result.float()).abs().max().item()
        print(f"Shape ({M}x{N}) B={B}: max diff = {diff:.2e}")
        assert diff < BF16_TOL, f"Too large for ({M}x{N}): {diff}"


if __name__ == "__main__":
    test_single_item_batch()
    test_multi_item_batch()
    test_wide_matrix()
    test_tall_matrix()
    test_all_shape_buckets()
    print("\nAll tests passed!")
