#!/usr/bin/env python3
"""LEVER HUNT 1 — cheap combination levers, ALL fit and judged TRAIN-side.

The no-q TEST query is SPENT, so nothing here may touch test. Every rule is
fit on TRAIN-fit (1981-1990) and scored on TRAIN-val (1990-1995). A rule that
wins on TRAIN-val is a candidate; it does NOT get a test number without a new
pre-registered query.

Levers tested (none previously tried in this form):
  L1. readout on the NO-Q track: ymean vs ymed. NEVER TESTED on no-q dumps,
      and on with-q the analogous choice was worth +0.0026.
  L2. flow-regime-conditional weights: separate inverse-MSE weights for
      peak days vs ordinary days (global weights are a compromise between
      two regimes with different best members).
  L3. per-basin-COHORT weights (arid/humid, flashy/steady) rather than global
      or fully per-basin (per-basin overfits; global is too coarse).
  L4. rank/robust combination: median of streams instead of mean (a median is
      robust to one member blowing up on a peak day -- directly targets scatter).
  L5. trimmed mean (drop min+max stream per row).
"""
import contextlib, io, json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg

FIT_END, VAL_START = "1990-09-30", "1990-10-01"

def add_extras(df, train):
    for name, seeds in (("lstm_multi5", ("s111","s222","s333","s444","s555")),
                        ("lstm_multi6", ("s111","s222","s333"))):
        key = name.split("_")[1]
        pat = (f"camels531ls_{key}_nhlstm_TRAIN_{{}}.csv.gz" if train
               else f"camels531ls_{key}_nhlstm_{{}}.csv.gz")
        d = load_seed_avg([L / pat.format(s) for s in seeds])
        df = df.merge(d[["station_id","date","pred"]].rename(columns={"pred":name}),
                      on=["station_id","date"], how="inner")
    return df

with contextlib.redirect_stderr(io.StringIO()):
    tr, cols = build_merged(train=True)
tr = add_extras(tr, True)
cols = cols + ["lstm_multi5","lstm_multi6"]
tr["_dt"] = pd.to_datetime(tr.date)
fit = tr[tr._dt <= FIT_END].copy()
val = tr[tr._dt >= VAL_START].copy()
print(f"TRAIN-fit rows {len(fit):,}  TRAIN-val rows {len(val):,}  streams {len(cols)}\n")

def med_nse(df, ens):
    d = df.assign(_e=ens); o=[]
    for s,g in d.groupby("station_id"):
        y=g.truth.to_numpy(float); p=g._e.to_numpy(float)
        k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
        if len(y)<20 or np.var(y)<1e-9: continue
        o.append(1-np.mean((y-p)**2)/np.var(y))
    return float(np.median(o))

def invmse_w(df, cols, theta=4.0, lam=0.25, mask=None):
    d = df[mask] if mask is not None else df
    y = d.truth.to_numpy(float)
    mse = np.array([np.nanmean((d[c].to_numpy(float)-y)**2) for c in cols])
    raw = mse**(-theta); raw = raw/raw.sum()
    w = lam*np.ones(len(cols))/len(cols) + (1-lam)*raw
    return w/w.sum()

Xv = val[cols].to_numpy(float)
w_glob = invmse_w(fit, cols)
base = med_nse(val, (Xv*w_glob).sum(1))
print(f"BASELINE (global inverse-MSE, fit on TRAIN-fit) TRAIN-val = {base:.4f}\n")

print("=== L4/L5. robust combination (targets SCATTER directly) ===")
print(f"  mean of streams          {med_nse(val, Xv.mean(1)):.4f}")
print(f"  MEDIAN of streams        {med_nse(val, np.median(Xv,1)):.4f}")
Xs = np.sort(Xv,1)
print(f"  trimmed mean (drop hi+lo){med_nse(val, Xs[:,1:-1].mean(1)):.4f}")
# weighted median
def wmed(X, w):
    idx = np.argsort(X,1); Xs_=np.take_along_axis(X,idx,1)
    W = np.take_along_axis(np.broadcast_to(w,X.shape),idx,1)
    cw = np.cumsum(W,1); out=np.empty(len(X))
    for i in range(len(X)):
        out[i] = Xs_[i, np.searchsorted(cw[i], 0.5)]
    return out
print(f"  weighted median          {med_nse(val, wmed(Xv,w_glob)):.4f}")

print("\n=== L2. flow-regime-conditional weights ===")
fit["_thr"] = fit.groupby("station_id")["truth"].transform(lambda s: s.quantile(0.90))
val["_thr"] = val.groupby("station_id")["truth"].transform(lambda s: s.quantile(0.90))
fpk = (fit.truth >= fit._thr).to_numpy()
vpk = (val.truth >= val._thr).to_numpy()
w_pk = invmse_w(fit, cols, mask=fpk)
w_lo = invmse_w(fit, cols, mask=~fpk)
ens = np.where(vpk, (Xv*w_pk).sum(1), (Xv*w_lo).sum(1))
print(f"  peak/ordinary split weights {med_nse(val, ens):.4f}  delta {med_nse(val,ens)-base:+.4f}")
print(f"  (peak-day weights)   {dict(zip(cols, np.round(w_pk,3)))}")
print(f"  (ordinary weights)   {dict(zip(cols, np.round(w_lo,3)))}")

print("\n=== L3. cohort-conditional weights (arid/humid) ===")
try:
    a = pd.read_csv("gpu1080/nh_data_multi/attributes/attributes.csv")
    idc = [c for c in a.columns if c.lower() in ("gauge_id","basin_id","station_id")][0]
    a[idc] = a[idc].astype(str).str.zfill(8); a = a.set_index(idc)
    arid = set(a.index[a.aridity > 1.0])
    fa = fit.station_id.isin(arid).to_numpy(); va = val.station_id.isin(arid).to_numpy()
    w_a = invmse_w(fit, cols, mask=fa); w_h = invmse_w(fit, cols, mask=~fa)
    ens = np.where(va, (Xv*w_a).sum(1), (Xv*w_h).sum(1))
    print(f"  arid/humid split weights   {med_nse(val, ens):.4f}  delta {med_nse(val,ens)-base:+.4f}")
except Exception as e:
    print(f"  (skipped: {e})")

print("\n=== L1. NO-Q READOUT: ymean vs ymed (never tested on no-q) ===")
# rebuild one stream with ymean to see if the column exists and differs
p = L/"camels531ls_multi5_nhlstm_TRAIN_s111.csv.gz"
d = pd.read_csv(p, usecols=lambda c: c in ("station_id","t0","h","truth","ymed","ymean"), nrows=200000)
if "ymean" in d.columns:
    d = d[d.h==1]
    ok = np.isfinite(d.ymed) & np.isfinite(d.ymean)
    print(f"  ymean present. corr(ymed,ymean)={np.corrcoef(d.ymed[ok],d.ymean[ok])[0,1]:.5f}"
          f"  mean diff {(d.ymean-d.ymed)[ok].mean():+.4f}")
    print("  -> worth a full ymean rebuild of every stream if corr < 0.999")
else:
    print("  ymean NOT in no-q dumps -> readout lever unavailable without re-dumping")
