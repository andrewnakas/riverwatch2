#!/usr/bin/env python3
"""Recompute the NO-Q ceiling numbers with the corrected form, on the TEST
frame (the held-out 0.8363 config). Replaces the 0.8890 all-at-ceiling figure
and the '29% saturated' claim.
"""
import contextlib, io, json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2")); sys.path.insert(0,os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg
from target_feasibility import sigma_series

with contextlib.redirect_stderr(io.StringIO()):
    m, cols = build_merged(train=False)
for name, seeds in (("lstm_multi5",("s111","s222","s333","s444","s555")),
                    ("lstm_multi6",("s111","s222","s333"))):
    key=name.split("_")[1]
    d=load_seed_avg([L/f"camels531ls_{key}_nhlstm_{s}.csv.gz" for s in seeds])
    m=m.merge(d[["station_id","date","pred"]].rename(columns={"pred":name}),
              on=["station_id","date"],how="inner")
cols=cols+["lstm_multi5","lstm_multi6"]
r=json.load(open("noq_test_result.json")); w=np.array([r["weights"][c] for c in cols])
m=m.assign(ens=(m[cols].to_numpy(float)*w).sum(1))
flash=m.groupby("station_id")["truth"].apply(
    lambda s: np.percentile(s[s>0],99)/np.median(s[s>0]) if (s>0).sum()>20 else np.nan)
thr=flash.median()
print(f"no-q TEST frame: {m.station_id.nunique()} basins, {len(m):,} rows\n")
print(f"{'scenario':<14}{'current':>9}{'OLD ceil':>10}{'NEW ceil':>10}"
      f"{'OLD all->c':>12}{'NEW all->c':>12}{'OLD sat':>10}{'NEW sat':>10}")
print("-"*87)
for lab,(lo,hi,fl) in (("central",(0.30,0.18,0.35)),("optimistic",(0.25,0.13,0.13))):
    o={};nw={};cu={}
    for sid,g in m.groupby("station_id"):
        q=g.truth.to_numpy(float); ok=np.isfinite(q)
        if ok.sum()<50: continue
        qq=q[ok]; V=((qq-qq.mean())**2).mean()
        if V<=0: continue
        M=(qq**2).mean()
        sh=fl if flash.get(sid,0)>thr else hi
        sg=sigma_series(qq,lo,sh)
        o[sid]=1-(M/V)*(np.exp(np.nanmean(sg**2))-1)
        num=np.mean(qq**2*(np.exp(sg**2)-1))
        Vobs=np.mean(qq**2*np.exp(sg**2))-(np.mean(qq))**2
        nw[sid]=1-num/Vobs
        p=g.ens.to_numpy(float)[ok]
        cu[sid]=1-np.mean((qq-p)**2)/np.var(qq)
    o=pd.Series(o);nw=pd.Series(nw);cu=pd.Series(cu)
    i=sorted(set(o.index)&set(cu.index)); o,nw,cu=o.reindex(i),nw.reindex(i),cu.reindex(i)
    co=float(np.median(np.maximum(cu,np.minimum(o,1.0))))
    cn=float(np.median(np.maximum(cu,np.minimum(nw,1.0))))
    print(f"{lab:<14}{cu.median():>9.4f}{o.median():>10.4f}{nw.median():>10.4f}"
          f"{co:>12.4f}{cn:>12.4f}{(o<=cu).sum():>7}/{len(o)}{(nw<=cu).sum():>7}/{len(nw)}")
    if lab=="optimistic":
        need=(0.845-cu.median())/(cn-cu.median())*100 if cn>cu.median() else float("nan")
        print(f"\n  0.845 needs {need:.0f}% of CORRECTED recoverable headroom (optimistic)")
