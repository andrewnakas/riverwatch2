#!/usr/bin/env python3
"""Recompute near-median cohort saturation with the CORRECTED ceiling form."""
import contextlib, io, os, sys
os.chdir(os.path.expanduser("~/riverwatch2")); sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from gate_eval import build_merged, L
from phase_c import load_seed_avg
from target_feasibility import sigma_series

with contextlib.redirect_stderr(io.StringIO()):
    tr, cols = build_merged(train=True)
for nm, sd in (("lstm_multi5", ("s111","s222","s333","s444","s555")),
               ("lstm_multi6", ("s111","s222","s333"))):
    k = nm.split("_")[1]
    d = load_seed_avg([L / f"camels531ls_{k}_nhlstm_TRAIN_{s}.csv.gz" for s in sd])
    tr = tr.merge(d[["station_id","date","pred"]].rename(columns={"pred": nm}),
                  on=["station_id","date"], how="inner")
cols = cols + ["lstm_multi5", "lstm_multi6"]
tr["_dt"] = pd.to_datetime(tr.date)
fit = tr[tr._dt <= "1990-09-30"]; val = tr[tr._dt >= "1990-10-01"].copy()
y = fit.truth.to_numpy(float)
mse = np.array([np.nanmean((fit[c].to_numpy(float) - y) ** 2) for c in cols])
raw = mse ** (-4.0); raw /= raw.sum()
w = 0.25 * np.ones(len(cols)) / len(cols) + 0.75 * raw; w /= w.sum()
val = val.assign(ens=(val[cols].to_numpy(float) * w).sum(1))

def pbn(df, c):
    o = {}
    for s, g in df.groupby("station_id"):
        yy = g.truth.to_numpy(float); pp = g[c].to_numpy(float)
        k = np.isfinite(yy) & np.isfinite(pp); yy, pp = yy[k], pp[k]
        if len(yy) < 20 or np.var(yy) < 1e-9: continue
        o[s] = 1 - np.mean((yy - pp) ** 2) / np.var(yy)
    return pd.Series(o)

s = pbn(val, "ens").sort_values(); n = len(s); m0 = s.iloc[n // 2]
near = list(s.index[[i for i in range(n) if abs(s.iloc[i] - m0) < 0.01]])
fl = val.groupby("station_id")["truth"].apply(
    lambda x: np.percentile(x[x > 0], 99) / np.median(x[x > 0]) if (x > 0).sum() > 20 else np.nan)
th = fl.median()
print(f"median {m0:.4f}, near-median cohort n={len(near)}\n")
print(f"{'scenario':<13}{'OLD sat':>10}{'NEW sat':>10}{'OLD hr':>10}{'NEW hr':>10}{'NEW all->c':>12}")
print("-" * 65)
for lab, (lo, hi, flv) in (("central", (0.30, 0.18, 0.35)),
                           ("optimistic", (0.25, 0.13, 0.13))):
    o = {}; nw = {}
    for sid, g in val.groupby("station_id"):
        q = g.truth.to_numpy(float); ok = np.isfinite(q)
        if ok.sum() < 50: continue
        qq = q[ok]; V = ((qq - qq.mean()) ** 2).mean()
        if V <= 0: continue
        M = (qq ** 2).mean(); sh = flv if fl.get(sid, 0) > th else hi
        sg = sigma_series(qq, lo, sh)
        o[sid] = 1 - (M / V) * (np.exp(np.nanmean(sg ** 2)) - 1)
        num = np.mean(qq ** 2 * (np.exp(sg ** 2) - 1))
        Vo = np.mean(qq ** 2 * np.exp(sg ** 2)) - (np.mean(qq)) ** 2
        nw[sid] = 1 - num / Vo
    o = pd.Series(o); nw = pd.Series(nw)
    i = [x for x in near if x in nw.index]
    ho = (o.reindex(i) - s.reindex(i)).dropna(); hn = (nw.reindex(i) - s.reindex(i)).dropna()
    j = sorted(set(nw.index) & set(s.index))
    cap = float(np.median(np.maximum(s.reindex(j), np.minimum(nw.reindex(j), 1.0))))
    print(f"{lab:<13}{(ho<=0).sum():>6}/{len(ho)}{(hn<=0).sum():>7}/{len(hn)}"
          f"{ho.median():>+10.4f}{hn.median():>+10.4f}{cap:>12.4f}")
