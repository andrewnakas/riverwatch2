#!/usr/bin/env python3
"""Isolate the 0.8298 vs 0.8305 discrepancy: is it purely lstm_multi seed depth?"""
import contextlib, io, os, sys
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from gate_eval import build_merged, L
from phase_c import load_seed_avg

f = io.StringIO()
with contextlib.redirect_stderr(f):
    m, cols = build_merged(train=False)


def pbn(df, col):
    o = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float); p = g[col].to_numpy(float)
        k = np.isfinite(y) & np.isfinite(p); y, p = y[k], p[k]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        o[sid] = 1.0 - np.mean((y - p) ** 2) / np.var(y)
    return pd.Series(o)


m3 = load_seed_avg([L / f"camels531ls_multi_nhlstm_{s}.csv.gz"
                    for s in ("s111", "s222", "s3334")])
m3 = m3[["station_id", "date", "pred"]].rename(columns={"pred": "multi3seed"})
mm = m.merge(m3, on=["station_id", "date"], how="inner").copy()

c5 = list(cols)
c3 = [c for c in cols if c != "lstm_multi"] + ["multi3seed"]
mm["e5"] = mm[c5].mean(axis=1)
mm["e3"] = mm[c3].mean(axis=1)

s5 = pbn(mm, "e5").median()
s3 = pbn(mm, "e3").median()
print(f"rows {len(mm):,}  basins {mm.station_id.nunique()}")
print(f"  lstm_multi = 3 seeds  -> {s3:.4f}   (phase_c canonical, expect 0.8298)")
print(f"  lstm_multi = 5 seeds  -> {s5:.4f}   (gate_eval glob)")
print(f"  delta from seed depth  = {s5 - s3:+.4f}")
