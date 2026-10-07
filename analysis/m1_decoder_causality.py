#!/usr/bin/env python3
"""Is the decoder CAUSAL -- can a decoder input at lead k affect the lead-1 prediction?

This decides the design of the real-forecast decay measurement. _parse_forcing_file drops any
station lacking all 14 leads, so a forecast file must carry 14. The registered question is day-1
only, so the plan is: lead 1 = real GFS, leads 2-14 = the observed values. That is sound only if the
lead-1 OUTPUT cannot depend on decoder inputs at leads 2-14.

A plain forward LSTM cannot, but "it's an LSTM so it must be causal" is exactly the kind of claim
that has been wrong 22 times in this campaign. So: perturb the decoder input at one lead and see
which output leads move.
"""
import sys
import numpy as np, torch
sys.path.insert(0, ".")
from app.mblstm import build_model

CK = "data/mblstm/gpu_ckpts/m1_gridmetm9_s501.pt"
sd = torch.load(CK, map_location="cpu", weights_only=False)
cfg = sd["cfg"]
m = build_model(cfg)
m.load_state_dict(sd["state_dict"])
m.eval()
print(f"model built: head={cfg['head']} hidden={cfg['hidden']} horizon={cfg['horizon']}")

H = cfg["horizon"]
n_enc = m.encoder.weight_ih_l0.shape[1]
n_dec = m.decoder.weight_ih_l0.shape[1]
print(f"encoder input width {n_enc}, decoder input width {n_dec}")

torch.manual_seed(0)
xe = torch.randn(2, cfg["context"], n_enc)
xd = torch.randn(2, H, n_dec)
with torch.no_grad():
    base = m(xe, xd)
base = base.detach().clone()
print(f"output shape {tuple(base.shape)}")

print(f"\nperturb the decoder input at ONE lead; which output leads move?\n")
print(f"  {'perturbed lead':<18}" + "".join(f"{f'out{j+1}':>8}" for j in range(min(H, 8))))
causal = True
for k in range(H):
    xd2 = xd.clone()
    xd2[:, k, :] += 5.0
    with torch.no_grad():
        out = m(xe, xd2)
    d = (out - base).abs()
    # collapse over batch and quantile dims, keep the lead axis
    lead_axis = [i for i, s in enumerate(d.shape) if s == H]
    assert lead_axis, f"cannot find the lead axis in {tuple(d.shape)}"
    ax = lead_axis[0]
    perlead = d.amax(dim=tuple(i for i in range(d.ndim) if i != ax)).numpy()
    row = "".join(f"{perlead[j]:8.4f}" for j in range(min(H, 8)))
    moved_before = [j + 1 for j in range(k) if perlead[j] > 1e-9]
    flag = "" if not moved_before else f"   <- also moved EARLIER leads {moved_before}"
    if moved_before: causal = False
    print(f"  lead {k+1:<13}{row}{flag}")

print()
if causal:
    print("✅ DECODER IS CAUSAL: a decoder input at lead k moves only leads >= k. So filling")
    print("   leads 2-14 with observed forcing cannot affect the LEAD-1 number, and a forecast")
    print("   file carrying real GFS at lead 1 and observations after it isolates day-1 decay.")
else:
    print("⛔ NOT CAUSAL: later decoder inputs move the lead-1 output. Leads 2-14 must then also")
    print("   carry real forecasts, so the build must fetch all 14 leads.")
