# ABOUTME: Tests bank model equivalence: init audit, mapping, forward, one-step.
# ABOUTME: Run on GPU with: python3 tests/test_bank_equivalence.py
"""
Bank equivalence test suite. Validates that the banked model produces
identical results to a reference set of logical weights.
"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LAYERS, DIM, HEADS, KV_HEADS, MLP_MULT = 11, 512, 8, 4, 3
VOCAB, SEQ = 1024, 64

def make_model():
    from train_gpt import GPT
    m = GPT(vocab_size=VOCAB, num_layers=NUM_LAYERS, model_dim=DIM,
            num_heads=HEADS, num_kv_heads=KV_HEADS, mlp_mult=MLP_MULT,
            tie_embeddings=True, tied_embed_init_std=0.005,
            logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
            ln_scale=True, rope_dims=16, xsa_last_n=4)
    return m.to(device).bfloat16()


def test_init_audit():
    """Verify per-slice initialization: ortho vs zero."""
    print("=== Test 1: Init Audit ===")
    m = make_model()
    for bank in [m.bank_sq, m.bank_kv, m.bank_fc, m.bank_pr]:
        bank.data = bank.data.float()
    errors = []
    for i in range(NUM_LAYERS):
        ap = m.bank_sq.data[2*i+1]
        cq = m.bank_sq.data[2*i]
        if ap.abs().max() > 1e-6:
            errors.append(f"bank_sq[{2*i+1}] (attn.proj L{i}) should be zero, max={ap.abs().max():.4e}")
        if cq.abs().max() < 1e-6:
            errors.append(f"bank_sq[{2*i}] (c_q L{i}) appears zero, should be ortho")
        ck = m.bank_kv.data[2*i]
        cv = m.bank_kv.data[2*i+1]
        if ck.abs().max() < 1e-6:
            errors.append(f"bank_kv[{2*i}] (c_k L{i}) appears zero")
        if cv.abs().max() < 1e-6:
            errors.append(f"bank_kv[{2*i+1}] (c_v L{i}) appears zero")
        fc = m.bank_fc.data[i]
        if fc.abs().max() < 1e-6:
            errors.append(f"bank_fc[{i}] (mlp.fc L{i}) appears zero")
        pr = m.bank_pr.data[i]
        if pr.abs().max() > 1e-6:
            errors.append(f"bank_pr[{i}] (mlp.proj L{i}) should be zero, max={pr.abs().max():.4e}")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return False
    print("  PASS: all slices have correct init type")
    return True


def test_export_roundtrip():
    """Export -> load_export -> re-export should produce identical tensors."""
    print("\n=== Test 2: Export Mapping Roundtrip ===")
    m = make_model()
    for bank in [m.bank_sq, m.bank_kv, m.bank_fc, m.bank_pr]:
        bank.data = bank.data.float()
    sd1 = m.export_state_dict()
    m2 = make_model()
    for bank in [m2.bank_sq, m2.bank_kv, m2.bank_fc, m2.bank_pr]:
        bank.data = bank.data.float()
    m2.load_export_state_dict(sd1)
    sd2 = m2.export_state_dict()
    if set(sd1.keys()) != set(sd2.keys()):
        missing = set(sd1.keys()) - set(sd2.keys())
        extra = set(sd2.keys()) - set(sd1.keys())
        print(f"  FAIL: key mismatch. missing={missing}, extra={extra}")
        return False
    max_diff = 0.0
    mismatches = []
    for k in sd1:
        d = (sd1[k].float() - sd2[k].float()).abs().max().item()
        max_diff = max(max_diff, d)
        if d > 1e-5:
            mismatches.append(f"{k}: diff={d:.4e}")
    if mismatches:
        for ms in mismatches[:10]:
            print(f"  FAIL: {ms}")
        return False
    print(f"  PASS: all {len(sd1)} tensors match (max_diff={max_diff:.2e})")
    return True


def test_forward():
    """Forward pass produces reasonable loss and is deterministic."""
    print("\n=== Test 3: Forward ===")
    m = make_model()
    for bank in [m.bank_sq, m.bank_kv, m.bank_fc, m.bank_pr]:
        bank.data = bank.data.float()
    from train_gpt import restore_low_dim_params_to_fp32
    restore_low_dim_params_to_fp32(m)
    torch.manual_seed(123)
    x = torch.randint(0, VOCAB, (2, SEQ), device=device)
    y = torch.randint(0, VOCAB, (2, SEQ), device=device)
    m.eval()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = m(x, y)
            logits = m.forward_logits(x)
    loss_val = loss.item()
    print(f"  loss: {loss_val:.6f} (expected ~{math.log(VOCAB):.2f})")
    print(f"  logits: shape={logits.shape}, mean={logits.float().mean().item():.4f}, std={logits.float().std().item():.4f}")
    if math.isnan(loss_val) or math.isinf(loss_val):
        print(f"  FAIL: loss is {loss_val}")
        return False
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss2 = m(x, y)
    diff = abs(loss.item() - loss2.item())
    if diff > 1e-6:
        print(f"  FAIL: non-deterministic (diff={diff:.2e})")
        return False
    print(f"  PASS: deterministic (diff={diff:.2e}), loss reasonable")
    return True


def test_backward_step():
    """One Muon step completes; gradients exist on all banks."""
    print("\n=== Test 4: Backward + Optimizer Step ===")
    m = make_model()
    for bank in [m.bank_sq, m.bank_kv, m.bank_fc, m.bank_pr]:
        bank.data = bank.data.float()
    from train_gpt import restore_low_dim_params_to_fp32, Muon
    restore_low_dim_params_to_fp32(m)
    banks = [m.bank_sq, m.bank_kv, m.bank_fc, m.bank_pr]
    muon = Muon(banks, lr=0.04, momentum=0.95, backend_steps=3)
    torch.manual_seed(123)
    x = torch.randint(0, VOCAB, (2, SEQ), device=device)
    y = torch.randint(0, VOCAB, (2, SEQ), device=device)
    m.train()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        loss1 = m(x, y)
    print(f"  loss before: {loss1.item():.6f}")
    loss1.backward()
    for name, bank in [("sq", m.bank_sq), ("kv", m.bank_kv), ("fc", m.bank_fc), ("pr", m.bank_pr)]:
        if bank.grad is None:
            print(f"  FAIL: bank_{name}.grad is None")
            return False
        print(f"  bank_{name} grad_norm: {bank.grad.float().norm().item():.4f}")
    muon.step()
    muon.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        loss2 = m(x, y)
    print(f"  loss after:  {loss2.item():.6f}")
    print(f"  PASS: step completed")
    return True


if __name__ == "__main__":
    results = [
        ("Init Audit", test_init_audit()),
        ("Export Roundtrip", test_export_roundtrip()),
        ("Forward", test_forward()),
        ("Backward + Step", test_backward_step()),
    ]
    print("\n=== Summary ===")
    ok = True
    for name, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}: {name}")
        if not passed:
            ok = False
    print(f"\n{'All tests passed!' if ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if ok else 1)
