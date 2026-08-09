#!/usr/bin/env python3
"""I claimed seed depth is the best remaining GPU spend. Test it before
recommending it, TRAIN-side (the no-q test query is spent).

Measures, for each stream, the marginal ensemble value of going from k to k+1
seeds -- and critically the BREADTH of that gain, since only broad gains move
a median ([[median-leverage-the-targeting-error]]).

Q1. Which streams are seed-starved? (how many seeds each currently has)
Q2. Marginal ensemble delta per added seed, per stream.
Q3. Is the gain BROAD (fraction of basins improved)?
Q4. Extrapolate: what would filling every stream to 5 seeds be worth?
"""
import contextlib, io, os, re, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2")); sys.path.insert(0,os.getcwd())
from gate_eval import build_merged, L, H, _clean, _MULTI_WEAK_SEEDS
from phase_c import load_seed_avg

def seeds_for(stream):
    """All TRAIN seed dumps backing a stream, honouring the exclusion rules."""
    if stream.startswith("dhbv_"):
        f=stream.split("_",1)[1]
        return _clean(sorted(H.glob(f"camels531ls_{f}_dhbv_TRAIN_s*.csv.gz")))
    if stream=="lstm_multi":
        ps=_clean(sorted(L.glob("camels531ls_multi_nhlstm_TRAIN_s*.csv.gz")))
        return [p for p in ps if not any(p.name.endswith(f"_{s}.csv.gz") for s in _MULTI_WEAK_SEEDS)]
    if stream in ("lstm_multi5","lstm_multi6"):
        k=stream.replace("lstm_","")
        return sorted(L.glob(f"camels531ls_{k}_nhlstm_TRAIN_s*.csv.gz"))
    f=stream.split("_",1)[1]
    ps=_clean(sorted(L.glob(f"camels531ls_{f}_nhlstm_TRAIN_s*.csv.gz")))
    if any(p.name.endswith("_TRAIN_s1111.csv.gz") for p in ps):
        ps=[p for p in ps if not p.name.endswith("_TRAIN_s111.csv.gz")]
    return ps

with contextlib.redirect_stderr(io.StringIO()):
    base,cols=build_merged(train=True)
for nm,sd in (("lstm_multi5",("s111","s222","s333","s444","s555")),
              ("lstm_multi6",("s111","s222","s333"))):
    k=nm.split("_")[1]
    d=load_seed_avg([L/f"camels531ls_{k}_nhlstm_TRAIN_{s}.csv.gz" for s in sd])
    base=base.merge(d[["station_id","date","pred"]].rename(columns={"pred":nm}),
                    on=["station_id","date"],how="inner")
cols=cols+["lstm_multi5","lstm_multi6"]

print("=== Q1. seeds currently backing each stream ===")
inv={}
for c in cols:
    inv[c]=seeds_for(c)
    print(f"  {c:<16} {len(inv[c])} seeds")

def pbn(df,ens):
    d=df.assign(_e=ens); o={}
    for s,g in d.groupby("station_id"):
        y=g.truth.to_numpy(float); p=g._e.to_numpy(float)
        k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
        if len(y)<20 or np.var(y)<1e-9: continue
        o[s]=1-np.mean((y-p)**2)/np.var(y)
    return pd.Series(o)
def w_of(df,cs):
    y=df.truth.to_numpy(float)
    mse=np.array([np.nanmean((df[c].to_numpy(float)-y)**2) for c in cs])
    raw=mse**(-4.0); raw/=raw.sum()
    w=0.25*np.ones(len(cs))/len(cs)+0.75*raw
    return w/w.sum()
w=w_of(base,cols)
ref=pbn(base,(base[cols].to_numpy(float)*w).sum(1))
print(f"\nfull ensemble (all seeds) TRAIN medNSE {ref.median():.4f}")

print("\n=== Q2/Q3. marginal value of the LAST seed of each stream ===")
print(f"{'stream':<16}{'seeds':>6}{'drop-1-seed':>13}{'marginal':>11}{'breadth':>10}")
print("-"*58)
rows=[]
for c in cols:
    ps=inv[c]
    if len(ps)<2: continue
    d=load_seed_avg(ps[:-1], shift_days=1 if c.startswith("dhbv_") else 0)[["station_id","date","pred"]].rename(columns={"pred":"_x"})
    t=base.drop(columns=[c]).merge(d,on=["station_id","date"],how="inner").rename(columns={"_x":c})
    ww=w_of(t,cols)
    s=pbn(t,(t[cols].to_numpy(float)*ww).sum(1))
    i=sorted(set(s.index)&set(ref.index))
    marg=ref.reindex(i).median()-s.reindex(i).median()
    dd=(ref.reindex(i)-s.reindex(i)).dropna()
    rows.append((c,len(ps),marg,(dd>0).mean()))
    print(f"{c:<16}{len(ps):>6}{s.reindex(i).median():>13.4f}{marg:>+11.5f}{(dd>0).mean():>9.1%}")

print("\n=== Q4. read-out ===")
if rows:
    pos=[r for r in rows if r[2]>0]
    print(f"  streams where the last seed HELPED: {len(pos)}/{len(rows)}")
    print(f"  median marginal value of one seed : {np.median([r[2] for r in rows]):+.5f}")
    print(f"  median breadth                    : {np.median([r[3] for r in rows]):.1%}")
    thin=[r for r in rows if r[1]<=3]
    if thin:
        print(f"  thin streams (<=3 seeds): {[r[0] for r in thin]}")
        print(f"    their median marginal value: {np.median([r[2] for r in thin]):+.5f}")
print("\n  (a seed is worth funding only if marginal>0 AND breadth is high;")
print("   breadth near 50% means it is noise, not a gain)")
