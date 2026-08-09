#!/usr/bin/env python3
"""LEVER HUNT 2 — structural levers. All TRAIN-side (fit 1981-90, score 1990-95).

L6. NONLINEAR STACKING: a ridge/linear stacker on stream predictions per flow
    regime. Combination *rules* are closed, but a learned stacker on the
    predictions themselves (not just weights) has not been tried at this scale.
L7. LOG-SPACE / TRANSFORMED combination: average in asinh space instead of
    physical space. On right-skewed peaks, arithmetic averaging in physical
    space is dominated by whichever member is highest; averaging in a
    compressed space is a different estimator. Never tried.
L8. THE SATURATION QUESTION: 29% of no-q basins are at/above their gauge
    ceiling. Can we IDENTIFY them TRAIN-side and does anything change if we
    treat them differently? (They cannot be improved, but the MEDIAN is a rank
    statistic -- what matters is basins near the median.)
L9. MEDIAN-TARGETED effort: which basins actually SET the median? The median of
    531 is basin #266 by rank. Improving a basin at rank 500 does nothing.
    Quantify how concentrated the median's sensitivity is.
"""
import contextlib, io, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg

FIT_END, VAL_START = "1990-09-30", "1990-10-01"
with contextlib.redirect_stderr(io.StringIO()):
    tr, cols = build_merged(train=True)
for name, seeds in (("lstm_multi5",("s111","s222","s333","s444","s555")),
                    ("lstm_multi6",("s111","s222","s333"))):
    key=name.split("_")[1]
    d=load_seed_avg([L/f"camels531ls_{key}_nhlstm_TRAIN_{s}.csv.gz" for s in seeds])
    tr=tr.merge(d[["station_id","date","pred"]].rename(columns={"pred":name}),
                on=["station_id","date"],how="inner")
cols=cols+["lstm_multi5","lstm_multi6"]
tr["_dt"]=pd.to_datetime(tr.date)
fit=tr[tr._dt<=FIT_END].copy(); val=tr[tr._dt>=VAL_START].copy()

def per_basin(df, ens):
    d=df.assign(_e=ens); o={}
    for s,g in d.groupby("station_id"):
        y=g.truth.to_numpy(float); p=g._e.to_numpy(float)
        k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
        if len(y)<20 or np.var(y)<1e-9: continue
        o[s]=1-np.mean((y-p)**2)/np.var(y)
    return pd.Series(o)
def med(df,ens): return float(per_basin(df,ens).median())

def invmse_w(df, theta=4.0, lam=0.25):
    y=df.truth.to_numpy(float)
    mse=np.array([np.nanmean((df[c].to_numpy(float)-y)**2) for c in cols])
    raw=mse**(-theta); raw/=raw.sum()
    w=lam*np.ones(len(cols))/len(cols)+(1-lam)*raw
    return w/w.sum()
w=invmse_w(fit)
Xf=fit[cols].to_numpy(float); Xv=val[cols].to_numpy(float)
base=med(val,(Xv*w).sum(1))
print(f"BASELINE TRAIN-val {base:.4f}\n")

print("=== L7. combination in ASINH space (targets right-skew) ===")
def a(x): return np.arcsinh(x)
def ai(x): return np.sinh(x)
print(f"  physical-space weighted mean  {base:.4f}")
print(f"  asinh-space weighted mean     {med(val, ai((a(Xv)*w).sum(1))):.4f}")
print(f"  geometric-ish (log1p) mean    {med(val, np.expm1((np.log1p(np.clip(Xv,0,None))*w).sum(1))):.4f}")

print("\n=== L6. RIDGE STACKER on stream predictions (fit TRAIN-fit) ===")
yf=fit.truth.to_numpy(float); ok=np.isfinite(yf)&np.isfinite(Xf).all(1)
A=Xf[ok]; b=yf[ok]
for lam_r in (1e-3, 1.0, 100.0):
    # global stacker with intercept
    A1=np.hstack([A,np.ones((len(A),1))])
    coef=np.linalg.solve(A1.T@A1+lam_r*np.eye(A1.shape[1]), A1.T@b)
    pv=np.hstack([Xv,np.ones((len(Xv),1))])@coef
    print(f"  ridge lam={lam_r:<7} {med(val,pv):.4f}  coef={np.round(coef[:-1],3)}")
# per-flow-regime stacker
fit_thr=fit.groupby("station_id")["truth"].transform(lambda s:s.quantile(0.9)).to_numpy()
val_thr=val.groupby("station_id")["truth"].transform(lambda s:s.quantile(0.9)).to_numpy()
fpk=(yf>=fit_thr); vpk=(val.truth.to_numpy(float)>=val_thr)
pv=np.empty(len(Xv))
for sel_f, sel_v in ((fpk,vpk),(~fpk,~vpk)):
    m=sel_f&ok
    A1=np.hstack([Xf[m],np.ones((m.sum(),1))])
    coef=np.linalg.solve(A1.T@A1+1.0*np.eye(A1.shape[1]), A1.T@yf[m])
    pv[sel_v]=np.hstack([Xv[sel_v],np.ones((sel_v.sum(),1))])@coef
print(f"  ridge per-regime      {med(val,pv):.4f}  delta {med(val,pv)-base:+.4f}")

print("\n=== L9. WHICH BASINS SET THE MEDIAN? (sensitivity of a rank statistic) ===")
s=per_basin(val,(Xv*w).sum(1)).sort_values()
n=len(s); mid=n//2
print(f"  {n} basins; median is rank {mid}, NSE {s.iloc[mid]:.4f}")
for k in (5,10,25,50):
    lo,hi=s.iloc[mid-k],s.iloc[mid+k]
    print(f"    ranks {mid-k}..{mid+k}: NSE {lo:.4f}..{hi:.4f}  (spread {hi-lo:.4f})")
print("  => to move the MEDIAN you must move basins near rank 266, not the worst ones.")
print(f"  worst 50 basins median NSE {s.iloc[:50].median():.4f} -- improving them moves nothing")
print(f"  basins within +/-0.01 NSE of the median: {((s-s.iloc[mid]).abs()<0.01).sum()}")
print(f"  basins within +/-0.02 NSE of the median: {((s-s.iloc[mid]).abs()<0.02).sum()}")
