#!/usr/bin/env python3
"""Before rewriting the paper's central claim, verify the corrected form
against Monte Carlo ON THE ACTUAL WITH-Q SERIES, per basin, both scenarios.

The correction is only trustworthy if MC agrees basin-by-basin, not just at
the median. Also re-check the no-q track.
"""
import json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2")); sys.path.insert(0,os.getcwd())
from target_feasibility import sigma_series
rng=np.random.default_rng(7)
ids=set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
d=pd.read_csv("data/mblstm/gpu_dumps_s14/camels531_daymet_withq5_full531.csv.gz",
              usecols=["station_id","t0","h","truth"])
d=d[d.h==1].copy(); d["station_id"]=d.station_id.astype(str).str.zfill(8)
d=d[d.station_id.isin(ids)]
flash=d.groupby("station_id")["truth"].apply(
    lambda s: np.percentile(s[s>0],99)/np.median(s[s>0]) if (s>0).sum()>20 else np.nan)
thr=flash.median()

for lab,(lo,hi,fl) in (("central",(0.30,0.18,0.35)),("optimistic",(0.25,0.13,0.13))):
    old=[];new=[];mc=[]
    for sid,g in list(d.groupby("station_id"))[:200]:
        q=g.truth.to_numpy(float); q=q[np.isfinite(q)]
        if len(q)<50 or q.max()<=0: continue
        V=((q-q.mean())**2).mean(); M=(q**2).mean()
        if V<=0: continue
        sh=fl if flash.get(sid,0)>thr else hi
        sg=sigma_series(q,lo,sh)
        old.append(1-(M/V)*(np.exp(np.nanmean(sg**2))-1))
        num=np.mean(q**2*(np.exp(sg**2)-1))
        Vobs=np.mean(q**2*np.exp(sg**2))-(np.mean(q))**2
        new.append(1-num/Vobs)
        acc=[]
        for _ in range(25):
            obs=q*np.exp(rng.normal(0,1,len(q))*sg-sg**2/2)
            acc.append(1-np.mean((q-obs)**2)/np.var(obs))
        mc.append(np.mean(acc))
    old=np.array(old);new=np.array(new);mc=np.array(mc)
    print(f"=== {lab} (n={len(mc)} basins, 25 MC reps each) ===")
    print(f"  OLD form: median {np.median(old):.4f}   vs MC {np.median(mc):.4f}"
          f"   median|err| {np.median(np.abs(old-mc)):.4f}")
    print(f"  NEW form: median {np.median(new):.4f}   vs MC {np.median(mc):.4f}"
          f"   median|err| {np.median(np.abs(new-mc)):.4f}")
    print(f"  NEW within 0.01 of MC on {100*np.mean(np.abs(new-mc)<0.01):.0f}% of basins"
          f"  (OLD: {100*np.mean(np.abs(old-mc)<0.01):.0f}%)\n")
