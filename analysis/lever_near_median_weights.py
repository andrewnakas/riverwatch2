#!/usr/bin/env python3
"""LEVER: fit ensemble weights to maximise the MEDIAN, not the mean MSE.

Discovery: on the ~78 near-median basins the ENSEMBLE is BEATEN by lstm_multi5
alone (-0.0007). Global inverse-MSE weights optimise pooled squared error over
ALL basins -- including 416 saturated ones and 53 hopeless ones -- which is not
the objective we report.

Rules tested (ALL fit on TRAIN-fit 1981-90, scored TRAIN-val 1990-95, so the
selection is honest; the no-q test query is SPENT and is NOT touched):
  R0. current production: global inverse-MSE          [baseline]
  R1. inverse-MSE fit ONLY on near-median basins (identified on TRAIN-fit)
  R2. drop dHBV streams entirely
  R3. multi5-heavy: weights from near-median per-stream NSE ranking
  R4. direct median-maximising weight search (coordinate ascent on TRAIN-fit
      median, evaluated on TRAIN-val -- the objective we actually report)

The transfer question is the real one: near-median membership is defined by a
RANK, and ranks move between windows. R1/R4 are only real if the TRAIN-fit
ranking predicts the TRAIN-val ranking.
"""
import contextlib, io, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg

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
fit=tr[tr._dt<="1990-09-30"].copy(); val=tr[tr._dt>="1990-10-01"].copy()

def pbn(df, ens):
    d=df.assign(_e=ens); o={}
    for s,g in d.groupby("station_id"):
        y=g.truth.to_numpy(float); p=g._e.to_numpy(float)
        k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
        if len(y)<20 or np.var(y)<1e-9: continue
        o[s]=1-np.mean((y-p)**2)/np.var(y)
    return pd.Series(o)
def med(df,ens): return float(pbn(df,ens).median())

def invmse(df, theta=4.0, lam=0.25, ids=None):
    d = df[df.station_id.isin(ids)] if ids is not None else df
    y=d.truth.to_numpy(float)
    mse=np.array([np.nanmean((d[c].to_numpy(float)-y)**2) for c in cols])
    raw=mse**(-theta); raw/=raw.sum()
    w=lam*np.ones(len(cols))/len(cols)+(1-lam)*raw
    return w/w.sum()

Xf=fit[cols].to_numpy(float); Xv=val[cols].to_numpy(float)
w0=invmse(fit)
base_fit=med(fit,(Xf*w0).sum(1)); base=med(val,(Xv*w0).sum(1))
print(f"R0 production global inverse-MSE : TRAIN-fit {base_fit:.4f}  TRAIN-val {base:.4f}\n")

# identify near-median cohort ON TRAIN-FIT ONLY (honest)
sf=pbn(fit,(Xf*w0).sum(1)).sort_values(); nf=len(sf); mf=sf.iloc[nf//2]
near_fit=set(sf.index[[i for i in range(nf) if abs(sf.iloc[i]-mf)<0.01]])
sv=pbn(val,(Xv*w0).sum(1)).sort_values(); nv=len(sv); mv=sv.iloc[nv//2]
near_val=set(sv.index[[i for i in range(nv) if abs(sv.iloc[i]-mv)<0.01]])
print(f"near-median cohort: TRAIN-fit n={len(near_fit)}, TRAIN-val n={len(near_val)}")
print(f"  RANK TRANSFER: overlap {len(near_fit&near_val)} "
      f"({100*len(near_fit&near_val)/max(len(near_val),1):.0f}% of val cohort)")
print("  (low overlap => a near-median specialist cannot be targeted in advance)\n")

print("=== weight rules (all fit on TRAIN-fit) ===")
res={}
res["R0 global inverse-MSE"]=base
w1=invmse(fit, ids=near_fit); res["R1 invMSE on near-median"]=med(val,(Xv*w1).sum(1))
nod=[c for c in cols if not c.startswith("dhbv")]; idx=[cols.index(c) for c in nod]
w2=invmse(fit)[idx]; w2=w2/w2.sum(); res["R2 drop dHBV"]=med(val,(val[nod].to_numpy(float)*w2).sum(1))
# R3: rank-based on near-median NSE
nm=fit[fit.station_id.isin(near_fit)]
sc=np.array([pbn(nm,nm[c].to_numpy(float)).median() for c in cols])
w3=np.maximum(sc-sc.min(),1e-6)**3; w3/=w3.sum(); res["R3 near-median NSE^3"]=med(val,(Xv*w3).sum(1))
# R4: coordinate ascent maximising TRAIN-fit MEDIAN directly
w4=w0.copy(); cur=med(fit,(Xf*w4).sum(1))
for _ in range(3):
    for j in range(len(cols)):
        for mult in (0.5,0.75,1.5,2.0):
            t=w4.copy(); t[j]*=mult; t/=t.sum()
            v=med(fit,(Xf*t).sum(1))
            if v>cur: cur, w4 = v, t
res["R4 median-maximising"]=med(val,(Xv*w4).sum(1))
for k,v in res.items():
    print(f"  {k:<28}{v:.4f}   delta {v-base:+.5f}")
print(f"\n  R4 weights: {dict(zip(cols,np.round(w4,3)))}")
print(f"  R4 achieved TRAIN-fit median {cur:.4f} (vs {base_fit:.4f}) "
      f"-> TRAIN-val {res['R4 median-maximising']:.4f}")
print("\n  (if R4 gains on fit but not val, median-maximising OVERFITS the rank)")
