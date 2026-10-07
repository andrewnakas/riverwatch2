#!/usr/bin/env python3
"""Verify BOTH branches of the new enc_lead guard against real checkpoints.

A gate that has only been seen to pass is not a gate. Three cases:
  1. no flag + a pre-patch checkpoint (no enc_lead in cfg)  -> must PASS
  2. --enc-lead 1 + a checkpoint whose cfg says 0            -> must ABORT
  3. --enc-lead 1 + a cfg that says 1 (simulated)            -> must PASS
"""
import os, re, torch

CK = "data/mblstm/gpu_ckpts/m1_gridmetm9fix2_s501.pt"
cfg_real = torch.load(CK, map_location="cpu")["cfg"]
print(f"checkpoint cfg has enc_lead? {'enc_lead' in cfg_real} "
      f"(value {cfg_real.get('enc_lead', 'absent')})")

def guard(cfg, extra):
    m = re.search(r"--enc-lead\s+(\d+)", extra)
    want = int(m.group(1)) if m else 0
    assert int(cfg.get("enc_lead", 0)) == want, \
        f"enc_lead {cfg.get('enc_lead')} != requested {want}"
    return want

cases = [
    ("1. no flag, pre-patch ckpt", dict(cfg_real), "", True),
    ("2. --enc-lead 1, ckpt says 0", dict(cfg_real), "--enc-lead 1", False),
    ("3. --enc-lead 1, ckpt says 1", {**cfg_real, "enc_lead": 1}, "--enc-lead 1", True),
    ("4. no flag, ckpt says 1", {**cfg_real, "enc_lead": 1}, "", False),
]
ok = True
for name, cfg, extra, should_pass in cases:
    try:
        w = guard(cfg, extra)
        got = True; msg = f"passed (want={w})"
    except AssertionError as e:
        got = False; msg = f"aborted: {e}"
    verdict = "✅" if got == should_pass else "⛔ WRONG"
    if got != should_pass: ok = False
    print(f"  {name:<32} expect {'PASS' if should_pass else 'ABORT':<5} -> {msg:<42}{verdict}")
print("\n✅ both branches behave" if ok else "\n⛔ the guard does not discriminate -- fix before use")
