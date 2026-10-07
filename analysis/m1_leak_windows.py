#!/usr/bin/env python3
"""LEAK AUDIT — windows read back from the CHECKPOINTS, and encoder reach."""
import glob, json, os
import numpy as np, torch, pandas as pd

TEST0, TEST1 = pd.Timestamp("2018-10-01"), pd.Timestamp("2025-09-30")
VAL0, VAL1 = pd.Timestamp("2014-10-01"), pd.Timestamp("2018-09-30")
CONTEXT = 365
FINAL = {"withq": ["gridmetm9", "fused4md49", "fused4m9", "nldasm9"],
         "noq": ["lstm_fused4md4"]}
print("\n=== D. TRAIN/VAL/TEST SEPARATION, from the CHECKPOINT (never argv) ===")
bad = []
for arm, members in FINAL.items():
    for m in members:
        for s in (501, 502, 503, 504, 505):
            p = f"data/mblstm/gpu_ckpts/m1_{m}_s{s}.pt"
            if not os.path.exists(p):
                print(f"  {arm:6s} {m:<16} s{s}  MISSING CHECKPOINT"); bad.append((m, s, "missing")); continue
            cfg = torch.load(p, map_location="cpu")["cfg"]
            ts, te = pd.Timestamp(cfg["train_start"]), pd.Timestamp(cfg["train_end"])
            v0, v1 = (pd.Timestamp(x) for x in cfg["val_range"])
            ov_test = not (te < TEST0 or ts > TEST1)
            ov_val = not (te < VAL0 or ts > VAL1)
            # encoder reach: earliest input row used by the first scored test target
            reach = TEST0 - pd.Timedelta(days=CONTEXT)
            touches_train = reach <= te
            flag = ""
            if ov_test: flag += " ⛔TRAIN∩TEST"; bad.append((m, s, "train∩test"))
            if ov_val: flag += " ⚠️TRAIN∩VAL"; bad.append((m, s, "train∩val"))
            if touches_train: flag += f" ⚠️ENCODER REACH {reach.date()} <= train_end"; bad.append((m, s, "reach"))
            print(f"  {arm:6s} {m:<16} s{s}  train {ts.date()}..{te.date()}  val {v0.date()}..{v1.date()}"
                  f"  no_q={cfg.get('no_q_input')} dec_q={cfg.get('dec_q')}{flag or '  ✅'}")
print(f"\n  violations: {len(bad)}")
print(f"  encoder reach for the FIRST scored test day: {(TEST0 - pd.Timedelta(days=CONTEXT)).date()}"
      f"  (train ends 2014-09-30 for 9-year members => gap "
      f"{(TEST0 - pd.Timedelta(days=CONTEXT) - pd.Timestamp('2014-09-30')).days} days) ✅")

print("\n=== E. STATIC ATTRIBUTES: what period are they computed over? ===")
a = json.load(open("data/camels_attrs.json"))
k0 = sorted(a)[0]
clim = [x for x in a[k0] if x in ("p_mean", "pet_mean", "aridity", "p_seasonality",
                                  "frac_snow", "high_prec_freq", "low_prec_freq")]
print(f"  climate-derived statics present: {clim}")
print("  ⚠️ CAMELS (Addor et al. 2017) computes these over WY1989-2009 — entirely BEFORE")
print("     the 2018-2025 test window, so they cannot carry test-period information.")
print("     They DO overlap the 9-year training window (2005-2014) by ~4 years, which is")
print("     not leakage: training-period information in a training input is expected.")
