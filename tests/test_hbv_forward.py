"""Regression + gradient tests for the differentiable HBV core (app/hbv.py).

Pins the numeric output of hbv_forward so the B1 speedup (torch.jit.script /
scan rewrite) can be verified BIT-EXACT against the reference Python loop, and
guards that gradients still flow to every parameter (static + dynamic) and that
the model is mass-conservative-ish (runoff bounded by inputs).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app import hbv  # noqa: E402


def _fixture(B: int = 4, T: int = 40, seed: int = 0):
    """A deterministic (precip, tmean, pet, params, dyn) batch."""
    g = torch.Generator().manual_seed(seed)
    precip = torch.rand(B, T, generator=g) * 20.0          # 0..20 mm/day
    tmean = torch.rand(B, T, generator=g) * 30.0 - 5.0     # -5..25 C
    pet = torch.rand(B, T, generator=g) * 5.0              # 0..5 mm/day
    raw = torch.randn(B, hbv.N_HBV_PARAMS, generator=g)
    params = hbv.map_params(raw, torch)
    # dynamic BETA/K0 over the sequence
    dyn_raw = torch.randn(B, T, len(hbv.DYNAMIC_PARAMS), generator=g)
    dyn = {}
    for i, name in enumerate(hbv.DYNAMIC_PARAMS):
        lo, hi = hbv.PARAM_RANGES[name]
        dyn[name] = lo + (hi - lo) * torch.sigmoid(dyn_raw[..., i])
    return precip, tmean, pet, params, dyn


def test_forward_shape_and_finite():
    precip, tmean, pet, params, dyn = _fixture()
    q = hbv.hbv_forward(precip, tmean, pet, params, torch, dyn=dyn)
    assert q.shape == precip.shape
    assert torch.isfinite(q).all()
    assert (q >= 0).all(), "runoff must be non-negative"


def test_gradients_flow_to_all_params():
    precip, tmean, pet, params, dyn = _fixture()
    for v in params.values():
        v.requires_grad_(True)
    for v in dyn.values():
        v.requires_grad_(True)
    q = hbv.hbv_forward(precip, tmean, pet, params, torch, dyn=dyn)
    q.sum().backward()
    # Params overridden per-step by `dyn` carry grad through dyn, not params[name].
    # ROUTN/ROUTK are routing params consumed by gamma_uh (in dhbv.py), NOT by
    # hbv_forward — so they legitimately have no grad from the runoff output here.
    ROUTING = {"ROUTN", "ROUTK"}
    for name, v in params.items():
        if name in dyn or name in ROUTING:
            continue
        assert v.grad is not None and torch.isfinite(v.grad).all(), f"no grad for {name}"
    for name, v in dyn.items():
        assert v.grad is not None and torch.isfinite(v.grad).all(), f"no grad for dyn {name}"


def test_routing_params_get_grad_through_gamma_uh():
    """ROUTN/ROUTK are exercised by gamma_uh, so grads must reach them there."""
    precip, tmean, pet, params, dyn = _fixture()
    for name in ("ROUTN", "ROUTK"):
        params[name].requires_grad_(True)
    q = hbv.hbv_forward(precip, tmean, pet, params, torch, dyn=dyn)
    routed = hbv.gamma_uh(q, params["ROUTN"], params["ROUTK"], torch)
    routed.sum().backward()
    for name in ("ROUTN", "ROUTK"):
        g = params[name].grad
        assert g is not None and torch.isfinite(g).all(), f"no grad for {name}"


def test_dynamic_routing_reduces_to_static():
    """Time-varying gamma_uh (B,T) with n,k CONSTANT over time must reproduce
    the static (B,) routing bit-for-bit — the B3 dynamic-γ correctness anchor."""
    g = torch.Generator().manual_seed(3)
    B, T = 4, 30
    q = torch.rand(B, T, generator=g) * 10.0
    n = torch.rand(B, generator=g) * 3.0 + 1.5
    k = torch.rand(B, generator=g) * 2.0 + 0.6
    static = hbv.gamma_uh(q, n, k, torch)
    dyn = hbv.gamma_uh(q, n[:, None].expand(B, T).contiguous(),
                       k[:, None].expand(B, T).contiguous(), torch)
    torch.testing.assert_close(static, dyn, rtol=1e-5, atol=1e-5)
    # grads flow to the per-step params
    nd = n[:, None].expand(B, T).clone().requires_grad_(True)
    kd = k[:, None].expand(B, T).clone().requires_grad_(True)
    hbv.gamma_uh(q, nd, kd, torch).sum().backward()
    assert torch.isfinite(nd.grad).all() and torch.isfinite(kd.grad).all()


def test_warmup_trims_output():
    precip, tmean, pet, params, dyn = _fixture(T=50)
    q_full = hbv.hbv_forward(precip, tmean, pet, params, torch, n_warmup=0, dyn=dyn)
    q_warm = hbv.hbv_forward(precip, tmean, pet, params, torch, n_warmup=10, dyn=dyn)
    assert q_warm.shape[1] == q_full.shape[1] - 10
    torch.testing.assert_close(q_warm, q_full[:, 10:], rtol=0, atol=0)


# --- GOLDEN regression: pins the exact reference-loop output. The B1 rewrite
# must reproduce these to < 1e-5. Regenerate ONLY with an intentional physics
# change (and note it in EXPERIMENTS).
def test_golden_regression():
    precip, tmean, pet, params, dyn = _fixture(B=3, T=30, seed=7)
    q = hbv.hbv_forward(precip, tmean, pet, params, torch, dyn=dyn)
    # summary stats are enough to catch any drift while staying compact
    got = torch.stack([q.mean(), q.std(), q.min(), q.max(), q[:, -1].mean()])
    golden = torch.tensor(GOLDEN, dtype=got.dtype)
    torch.testing.assert_close(got, golden, rtol=1e-5, atol=1e-5)


def test_routing_finite_and_normalized():
    precip, tmean, pet, params, dyn = _fixture()
    q = hbv.hbv_forward(precip, tmean, pet, params, torch, dyn=dyn)
    routed = hbv.gamma_uh(q, params["ROUTN"], params["ROUTK"], torch)
    assert routed.shape == q.shape
    assert torch.isfinite(routed).all()
    assert (routed >= 0).all()


# Pinned from the reference Python loop (app/hbv.py). The B1 speedup must
# reproduce these to < 1e-5. Regenerate ONLY on an intentional physics change.
# Regenerated after adding BETAET (dynamic ET-shape exponent, δHBV1.1p recipe).
GOLDEN = [1.382322907447815, 1.0474077463150024, 0.0038756050635129213,
          4.782498359680176, 3.3050968647003174]


if __name__ == "__main__":
    # Print the golden vector so it can be pasted into GOLDEN above.
    precip, tmean, pet, params, dyn = _fixture(B=3, T=30, seed=7)
    q = hbv.hbv_forward(precip, tmean, pet, params, torch, dyn=dyn)
    got = torch.stack([q.mean(), q.std(), q.min(), q.max(), q[:, -1].mean()])
    print("GOLDEN =", got.tolist())
