#!/usr/bin/env python3
"""LEDGER 56 side read (zero GPU) — zero-parameter combiners across the frozen 8.
The record is the equal-weight MEAN of 8 members in physical cfs. Alternatives with
NO fitted parameter: per-row median, trimmed mean (drop min & max), and the mean
in sqrt space. Paired vs the mean on val1 only (selection frame). Reported, not
shipped, unless the paired CI excludes zero on val1.
"""
import importlib.util, json, os, sys
import numpy as np
spec = importlib.util.spec_from_file_location("l51", os.path.join(os.path.dirname(os.path.abspath(__file__)), "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)
ALL8 = ["fused3h1", "daymeth1", "nldash1", "maurerh1", "aorch1", "nhar", "nhar0", "nhar0h256"]
frame = sys.argv[1] if len(sys.argv) > 1 else "val1"
m = l51.join(ALL8, frame)
P = m[ALL8].to_numpy(float); y = m.truth.to_numpy(float)
idx = m.groupby("station_id").indices
def score(pred):
    out = {}
    for s, ii in idx.items():
        yy, pp = y[ii], pred[ii]; k = np.isfinite(yy) & np.isfinite(pp); yy, pp = yy[k], pp[k]
        if len(yy) < 20 or np.var(yy) < 1e-9: continue
        out[s] = 1 - np.mean((yy - pp) ** 2) / np.var(yy)
    import pandas as pd; return pd.Series(out)
combs = {"mean": P.mean(1), "median": np.median(P, 1),
         "trimmed": (P.sum(1) - P.max(1) - P.min(1)) / (P.shape[1] - 2),
         "sqrt_mean": np.sqrt(np.clip(P, 0, None)).mean(1) ** 2}
base = score(combs["mean"]); res = {"frame": frame, "rows": int(len(m)), "mean": float(base.median())}
print(f"{frame}: rows {len(m):,}  mean-combiner median NSE {base.median():.6f}")
for k, v in combs.items():
    if k == "mean": continue
    s = score(v); d, lo, hi, br, nb = l51.paired_ci(s, base)
    res[k] = {"median": float(s.median()), "paired": d, "ci95": [lo, hi], "breadth": br}
    print(f"  {k:10s} {s.median():.6f}  paired {d:+.6f} CI [{lo:+.6f},{hi:+.6f}] breadth {br:.3f}")
os.makedirs("benchmarks/l56", exist_ok=True)
json.dump(res, open(f"benchmarks/l56/COMBINER_{frame}.json", "w"), indent=1)
