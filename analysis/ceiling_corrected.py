#!/usr/bin/env python3
"""Recompute EVERY headline ceiling number with the CORRECTED closed form.

WRONG (used everywhere so far):  1 - (M/V)(e^{mean(s^2)} - 1)
RIGHT (matches Monte Carlo to 0.0005 under flow-dependent sigma):
    numerator   = E[y^2 (e^{s(y)^2} - 1)]        <- flow-weighted, exact
    denominator = Var(obs) = E[y^2 e^{s^2}] - (E y)^2
    ceiling     = 1 - numerator/denominator

Both differences matter and push the same way (ceiling too LOW):
 - plain mean(s^2) overstates error because s is LOW where y^2 is LARGE
 - Var(truth) < Var(obs), and NSE divides by the variance of what you score against
"""
import json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2")); sys.path.insert(0,os.getcwd())
from target_feasibility import sigma_series

ids=set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
P="data/mblstm/gpu_dumps_s14/camels531_{}_withq5_full531.csv.gz"
F=("daymet","maurer","nldas","aorc")
parts={}
for f in F:
    d=pd.read_csv(P.format(f),usecols=["station_id","t0","h","truth","ylo","yhi"])
    d=d[d.h==1].copy(); d["station_id"]=d.station_id.astype(str).str.zfill(8)
    d=d[d.station_id.isin(ids)]; d[f]=(d.ylo+d.yhi)/2
    parts[f]=d[["station_id","t0","truth",f]]
m=None
for f,d in parts.items():
    m=d if m is None else m.merge(d[["station_id","t0",f]],on=["station_id","t0"])
m["ens"]=m[list(F)].to_numpy(float).mean(1)
flash=m.groupby("station_id")["truth"].apply(
    lambda s: np.percentile(s[s>0],99)/np.median(s[s>0]) if (s>0).sum()>20 else np.nan)
thr=flash.median()

def ceilings(scen):
    lo,hi,fl=scen
    old={};new={};cur={}
    for sid,g in m.groupby("station_id"):
        q=g.truth.to_numpy(float); ok=np.isfinite(q)
        if ok.sum()<50: continue
        qq=q[ok]; V=((qq-qq.mean())**2).mean()
        if V<=0: continue
        M=(qq**2).mean()
        sh=fl if flash.get(sid,0)>thr else hi
        sg=sigma_series(qq,lo,sh)
        old[sid]=1-(M/V)*(np.exp(np.nanmean(sg**2))-1)
        num=np.mean(qq**2*(np.exp(sg**2)-1))
        Vobs=np.mean(qq**2*np.exp(sg**2))-(np.mean(qq))**2
        new[sid]=1-num/Vobs
        p=g.ens.to_numpy(float)[ok]
        cur[sid]=1-np.mean((qq-p)**2)/np.var(qq)
    return pd.Series(old),pd.Series(new),pd.Series(cur)

print("WITH-Q (record config, readout (ylo+yhi)/2)\n")
print(f"{'scenario':<14}{'current':>9}{'OLD ceiling':>13}{'NEW ceiling':>13}{'shift':>8}"
      f"{'OLD sat':>9}{'NEW sat':>9}")
print("-"*76)
for lab,scen in (("central",(0.30,0.18,0.35)),("optimistic",(0.25,0.13,0.13))):
    o,nw,cu=ceilings(scen)
    i=sorted(set(o.index)&set(cu.index))
    o,nw,cu=o.reindex(i),nw.reindex(i),cu.reindex(i)
    print(f"{lab:<14}{cu.median():>9.4f}{o.median():>13.4f}{nw.median():>13.4f}"
          f"{nw.median()-o.median():>+8.4f}{(o<=cu).sum():>6}/{len(o)}{(nw<=cu).sum():>6}/{len(nw)}")
    capped_o=np.maximum(cu,np.minimum(o,1.0)); capped_n=np.maximum(cu,np.minimum(nw,1.0))
    print(f"{'':14}{'':9}{'  all->ceiling:':>13}{float(np.median(capped_o)):.4f} (old)"
          f"   {float(np.median(capped_n)):.4f} (new)")
    if lab=="central":
        print(f"{'':14}   => RECORD 0.9203 vs NEW central ceiling {nw.median():.4f}"
              f"  gap {nw.median()-cu.median():+.4f}")
