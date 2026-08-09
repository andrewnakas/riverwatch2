#!/usr/bin/env python3
"""Song et al. WRR 2026: dHBV1.1p beats LSTM on events with return period >= 5yr
(+0.06 median NSE on those events, better peak errors in 80% of cases).

Our dHBV members ARE dHBV1.1p (nmul=16, 3 dynamic params incl. BETAET,
combined loss). So the claim is directly testable on dumps we already have --
zero GPU. If it replicates, a dHBV member deserves MORE weight on peak days
than inverse-MSE gives it, and a peak-day-specialised weighting is a free lever.

Tests, on the no-q TEST window (531 basins):
  T1. peak-day NSE of each stream (top-10% observed flow days, pooled)
  T2. high-return-period days: the top-N events per basin by observed flow
  T3. does dHBV beat the LSTM streams THERE, even though it loses overall?
  T4. would a peak-day-upweighted dHBV improve the ENSEMBLE? (the only
      question that matters -- member skill has repeatedly misled us)
"""
import contextlib, io, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg

f = io.StringIO()
with contextlib.redirect_stderr(f):
    m, cols = build_merged(train=False)
for name, seeds in (("lstm_multi5", ("s111","s222","s333","s444","s555")),
                    ("lstm_multi6", ("s111","s222","s333"))):
    key = name.split("_")[1]
    d = load_seed_avg([L / f"camels531ls_{key}_nhlstm_{s}.csv.gz" for s in seeds])
    m = m.merge(d[["station_id","date","pred"]].rename(columns={"pred":name}),
                on=["station_id","date"], how="inner")
cols = cols + ["lstm_multi5","lstm_multi6"]
print(f"rows {len(m):,}  basins {m.station_id.nunique()}  streams {len(cols)}\n")

# ---- flag peak days per basin ----
m = m.copy()
m["_thr90"] = m.groupby("station_id")["truth"].transform(lambda s: s.quantile(0.90))
m["_thr99"] = m.groupby("station_id")["truth"].transform(lambda s: s.quantile(0.99))
pk90 = m.truth >= m._thr90
pk99 = m.truth >= m._thr99

def pooled_nse(df, col, mask=None):
    d = df[mask] if mask is not None else df
    y = d.truth.to_numpy(float); p = d[col].to_numpy(float)
    k = np.isfinite(y) & np.isfinite(p); y, p = y[k], p[k]
    # pooled, variance about the FULL-record basin means would be ideal;
    # use overall variance of the subset (consistent across streams)
    return 1 - ((y-p)**2).sum() / ((y-y.mean())**2).sum()

print("=== T1/T2. pooled NSE by flow regime (higher = better) ===")
print(f"{'stream':<16}{'ALL':>9}{'top10%':>9}{'top1%':>9}")
print("-"*43)
rows=[]
for c in cols:
    a, b, d = pooled_nse(m,c), pooled_nse(m,c,pk90), pooled_nse(m,c,pk99)
    rows.append((c,a,b,d))
    print(f"{c:<16}{a:>9.4f}{b:>9.4f}{d:>9.4f}")
best_all = max(rows, key=lambda r: r[1])
best_pk  = max(rows, key=lambda r: r[3])
print(f"\n  best OVERALL : {best_all[0]} ({best_all[1]:.4f})")
print(f"  best TOP-1%  : {best_pk[0]} ({best_pk[3]:.4f})")

dh = [c for c in cols if c.startswith("dhbv")]
ls = [c for c in cols if c.startswith("lstm")]
print(f"\n=== T3. dHBV vs LSTM family on extremes ===")
for lab, mask in (("ALL",None),("top10%",pk90),("top1%",pk99)):
    md = np.mean([pooled_nse(m,c,mask) for c in dh])
    ml = np.mean([pooled_nse(m,c,mask) for c in ls])
    print(f"  {lab:<8} dHBV mean {md:>8.4f}   LSTM mean {ml:>8.4f}   diff {md-ml:+.4f}")
print("  (Song 2026 predicts dHBV closes or reverses the gap on extremes)")

print("\n=== T4. THE DECIDING TEST: peak-day-upweighted dHBV in the ENSEMBLE ===")
# fit inverse-MSE weights on TRAIN (as production does), then test whether
# boosting dHBV weight ON PEAK DAYS ONLY helps the TEST ensemble.
with contextlib.redirect_stderr(io.StringIO()):
    tr, _ = build_merged(train=True)
for name, seeds in (("lstm_multi5", ("s111","s222","s333","s444","s555")),
                    ("lstm_multi6", ("s111","s222","s333"))):
    key = name.split("_")[1]
    d = load_seed_avg([L / f"camels531ls_{key}_nhlstm_TRAIN_{s}.csv.gz" for s in seeds])
    tr = tr.merge(d[["station_id","date","pred"]].rename(columns={"pred":name}),
                  on=["station_id","date"], how="inner")
ty = tr.truth.to_numpy(float)
mse = np.array([np.nanmean((tr[c].to_numpy(float)-ty)**2) for c in cols])
raw = mse**(-4.0); raw/=raw.sum()
w = 0.25*np.ones(len(cols))/len(cols) + 0.75*raw; w/=w.sum()

def med_nse(df, ens):
    df = df.assign(_e=ens); o=[]
    for s,g in df.groupby("station_id"):
        y=g.truth.to_numpy(float); p=g._e.to_numpy(float)
        k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
        if len(y)<20 or np.var(y)<1e-9: continue
        o.append(1-np.mean((y-p)**2)/np.var(y))
    return float(np.median(o))

X = m[cols].to_numpy(float)
base = med_nse(m, (X*w).sum(1))
print(f"  production inverse-MSE ensemble        {base:.4f}")
di = [cols.index(c) for c in dh]
for boost in (1.5, 2.0, 3.0):
    w2 = w.copy(); w2[di] *= boost; w2 /= w2.sum()
    ens = (X*w).sum(1).copy()
    pm = pk90.to_numpy()
    ens[pm] = (X[pm]*w2).sum(1)
    print(f"  dHBV weight x{boost} on top-10% days   {med_nse(m, ens):.4f}"
          f"   delta {med_nse(m, ens)-base:+.4f}")
print("\n  (a positive delta would mean dHBV carries real extreme-event skill")
print("   that uniform weighting throws away; negative closes the lever)")
