#!/usr/bin/env python3
"""Sanity-verify the 0.8363 held-out result. Checks that would each catch a bug."""
import contextlib, io, json, os, sys
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from gate_eval import build_merged, L
from phase_c import load_seed_avg

r = json.load(open("noq_test_result.json"))
w = r["weights"]; cols = r["streams"]

f = io.StringIO()
with contextlib.redirect_stderr(f):
    m, base_cols = build_merged(train=False)
for name, seeds in (("lstm_multi5", ("s111","s222","s333","s444","s555")),
                    ("lstm_multi6", ("s111","s222","s333"))):
    d = load_seed_avg([L / f"camels531ls_{name.split('_')[1]}_nhlstm_{s}.csv.gz" for s in seeds])
    m = m.merge(d[["station_id","date","pred"]].rename(columns={"pred":name}),
                on=["station_id","date"], how="inner")

print("CHECK 1 — weights sum to 1 and none is degenerate")
wv = np.array([w[c] for c in cols])
print(f"  sum={wv.sum():.6f}  min={wv.min():.4f}  max={wv.max():.4f}  std={wv.std():.4f}")
assert abs(wv.sum()-1) < 1e-9

print("\nCHECK 2 — TEST truth is identical across streams (no misalignment)")
# truth came from the first stream; verify against an independently loaded dump
ind = load_seed_avg([L/"camels531ls_multi5_nhlstm_s111.csv.gz"])
j = m[["station_id","date","truth"]].merge(
        ind[["station_id","date","truth"]].rename(columns={"truth":"truth2"}),
        on=["station_id","date"], how="inner")
d = (j.truth - j.truth2).abs()
print(f"  rows compared {len(j):,}   max|diff| = {d.max():.10f}")
assert d.max() < 1e-6, "TRUTH MISMATCH — streams are misaligned"

print("\nCHECK 3 — no test row leaked into the weight-fitting frame")
te = pd.to_datetime(m.date)
with contextlib.redirect_stderr(io.StringIO()):
    tr, _ = build_merged(train=True)
trd = pd.to_datetime(tr.date)
print(f"  TRAIN window {trd.min().date()} -> {trd.max().date()}")
print(f"  TEST  window {te.min().date()} -> {te.max().date()}")
overlap = set(zip(tr.station_id, tr.date)) & set(zip(m.station_id, m.date))
print(f"  overlapping (basin,date) keys: {len(overlap)}")
assert len(overlap) == 0, "LEAKAGE: train and test frames share rows"

print("\nCHECK 4 — recompute the headline independently (fresh NSE code path)")
ens = (m[cols].to_numpy(float) * wv).sum(axis=1)
m = m.assign(ens=ens)
nses = []
for sid, g in m.groupby("station_id"):
    y = g.truth.to_numpy(float); p = g.ens.to_numpy(float)
    k = np.isfinite(y) & np.isfinite(p); y, p = y[k], p[k]
    if len(y) < 20 or np.var(y) < 1e-9: continue
    nses.append(1 - ((y-p)**2).sum()/((y-y.mean())**2).sum())   # sum-form, not mean/var
print(f"  sum-form median NSE = {np.median(nses):.4f}   (reported {r['median_nse']:.4f})")
assert abs(np.median(nses) - r["median_nse"]) < 5e-4

print("\nCHECK 5 — bootstrap CI over basins")
a = np.array(nses); rng = np.random.default_rng(0)
bs = [np.median(a[rng.choice(len(a), len(a), True)]) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"  median {np.median(a):.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

print("\nCHECK 6 — how many basins beat the Li/Song published 0.8294?")
print(f"  basins with NSE > 0.8294 : {(a > 0.8294).sum()}/{len(a)}")
print("\nALL CHECKS PASSED")
