#!/usr/bin/env python3
"""Is multi7 (GLDAS land-surface state) worth a multi-day extraction?

The pre-registered gate predicted +0.001..+0.004 and justified it as helping
ARID runoff generation. But [[median-leverage]] shows arid specialists cannot
move the median. So the real question is: does a land-surface-state channel
plausibly deliver a BROAD gain?

We cannot test GLDAS without the data. But we CAN bound it, because multi6 is
exactly this experiment already run: multi6 = multi3 + 3 Livneh SOIL MOISTURE
layers, i.e. land-surface state, the same physical information class GLDAS
supplies.

B1. multi6's BREADTH: what fraction of basins did it improve? (breadth is what
    moves a median)
B2. multi6's gain on the NEAR-MEDIAN cohort specifically -- the only cohort
    that matters.
B3. Does multi6's information look SATURATED? i.e. is its LOO value already
    small, implying a second land-surface channel adds even less?
B4. The decisive framing: if multi6 (soil moisture, already built, 3 seeds)
    delivers ~0 on the near-median basins, GLDAS -- a DIFFERENT MODEL'S version
    of the same state variables -- is very unlikely to do better.
"""
import contextlib, io, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg

with contextlib.redirect_stderr(io.StringIO()):
    tr, base_cols = build_merged(train=True)
d5=load_seed_avg([L/f"camels531ls_multi5_nhlstm_TRAIN_{s}.csv.gz" for s in ("s111","s222","s333","s444","s555")])
tr=tr.merge(d5[["station_id","date","pred"]].rename(columns={"pred":"lstm_multi5"}),on=["station_id","date"],how="inner")
d6=load_seed_avg([L/f"camels531ls_multi6_nhlstm_TRAIN_{s}.csv.gz" for s in ("s111","s222","s333")])
tr=tr.merge(d6[["station_id","date","pred"]].rename(columns={"pred":"lstm_multi6"}),on=["station_id","date"],how="inner")
tr["_dt"]=pd.to_datetime(tr.date)
fit=tr[tr._dt<="1990-09-30"]; val=tr[tr._dt>="1990-10-01"].copy()
def wts(df,cols):
    y=df.truth.to_numpy(float)
    mse=np.array([np.nanmean((df[c].to_numpy(float)-y)**2) for c in cols])
    raw=mse**(-4.0); raw/=raw.sum()
    w=0.25*np.ones(len(cols))/len(cols)+0.75*raw
    return w/w.sum()
def pbn(df,ens):
    d=df.assign(_e=ens); o={}
    for s,g in d.groupby("station_id"):
        y=g.truth.to_numpy(float); p=g._e.to_numpy(float)
        k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
        if len(y)<20 or np.var(y)<1e-9: continue
        o[s]=1-np.mean((y-p)**2)/np.var(y)
    return pd.Series(o)

c9=base_cols+["lstm_multi5","lstm_multi6"]
c8=base_cols+["lstm_multi5"]
w9=wts(fit,c9); w8=wts(fit,c8)
s9=pbn(val,(val[c9].to_numpy(float)*w9).sum(1))
s8=pbn(val,(val[c8].to_numpy(float)*w8).sum(1))
ids=sorted(set(s9.index)&set(s8.index))
d=(s9.reindex(ids)-s8.reindex(ids)).dropna()
print("=== B1. multi6's BREADTH (adding land-surface state to the ensemble) ===")
print(f"  median delta        {s9.reindex(ids).median()-s8.reindex(ids).median():+.5f}")
print(f"  basins improved     {(d>0).sum()}/{len(d)} = {(d>0).mean():.1%}")
print(f"  mean delta          {d.mean():+.5f}")
print("  (multi5 for comparison improved 93.6% of basins -- THAT is breadth)")

srt=s8.sort_values(); m0=srt.iloc[len(srt)//2]
near=set(srt.index[(srt-m0).abs()<0.01])
worst=set(srt.index[:53])
print("\n=== B2. WHERE does multi6 help? ===")
for lab,sel in (("near-median (moves metric)",near),("worst 53 (moves nothing)",worst),("ALL",set(ids))):
    i=[x for x in ids if x in sel]
    dd=d.reindex(i).dropna()
    print(f"  {lab:<28} n={len(dd):>3}  median {dd.median():+.5f}  improved {(dd>0).mean():.1%}")

print("\n=== B3/B4. VERDICT on multi7 ===")
nm=d.reindex([x for x in ids if x in near]).dropna()
print(f"  multi6 (soil moisture) on the near-median cohort: {nm.median():+.5f}")
print(f"  multi6 LOO value overall:                        {s9.reindex(ids).median()-s8.reindex(ids).median():+.5f}")
print("\n  multi7 = GLDAS land-surface state = the SAME information class as")
print("  multi6 (Livneh soil moisture), from a different land-surface model.")
print("  If multi6's near-median contribution is ~0, multi7's expected value on")
print("  the REPORTED METRIC is ~0 too -- for a multi-day, 90,584-request extraction.")
