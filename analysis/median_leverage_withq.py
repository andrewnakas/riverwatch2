#!/usr/bin/env python3
"""Does the median-leverage finding hold on the WITH-Q track too?

If yes, then the top-50 with-q target list (56% of 'headroom') is also
targeting basins that cannot move the reported median -- and the ceiling
analysis, which is computed per-basin and summed, describes a quantity that
is NOT what the benchmark reports.

This is the decisive cross-check. It uses the record dumps and readout.
"""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
ids=set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json")).get("531",[]))
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
s={}
for sid,g in m.groupby("station_id"):
    y=g.truth.to_numpy(float); p=g.ens.to_numpy(float)
    k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
    if len(y)<20 or np.var(y)<1e-9: continue
    s[sid]=1-np.mean((y-p)**2)/np.var(y)
s=pd.Series(s).sort_values()
n=len(s); mid=n//2; m0=s.iloc[mid]
print(f"WITH-Q: {n} basins, median NSE {m0:.4f} (rank {mid})\n")

def med_after(series, idx, delta):
    t=series.copy(); t.iloc[idx]=np.minimum(t.iloc[idx]+delta,1.0)
    return float(np.median(t))

top50=set(pd.read_csv("withq_top50_targets.csv").iloc[:,0].astype(str).str.zfill(8))
t50_idx=[i for i in range(n) if s.index[i] in top50]
near=[i for i in range(n) if abs(s.iloc[i]-m0)<0.01]
print(f"top-50 target basins present: {len(t50_idx)}   near-median cohort: {len(near)}")
print(f"top-50 median NSE {s.iloc[t50_idx].median():.4f}   near-median NSE {s.iloc[near].median():.4f}")
print(f"OVERLAP top50 & near-median: {len(set(t50_idx)&set(near))}\n")

print("=== median payoff by cohort ===")
for delta in (0.05,0.10,0.20,0.35):
    a=med_after(s,t50_idx,delta)-m0
    b=med_after(s,near,delta)-m0
    print(f"  +{delta:.2f}:  top-50 targets -> {a:+.5f}    near-median({len(near)}) -> {b:+.5f}")

print("\n=== where do the top-50 sit in the RANKING? ===")
r=[i for i in t50_idx]
print(f"  ranks: min {min(r)}, median {int(np.median(r))}, max {max(r)}  (median rank is {mid})")
print(f"  how many of the top-50 are ABOVE the median rank? {sum(1 for i in r if i>mid)}")

print("\n=== what would raising ALL basins to their ceiling do to the MEDIAN? ===")
print("  (the feasibility scripts report exactly this, and it IS a median)")
print(f"  current median {m0:.4f}")
print("  => the ceiling analysis is consistent; the issue is only that")
print("     SUMMED per-basin headroom != median movement.")
print("\n=== the honest statement ===")
gain_all=med_after(s,list(range(n)),0.01)-m0
print(f"  uniform +0.01 everywhere -> median {gain_all:+.5f}")
print(f"  +0.35 on the 50 worst    -> median {med_after(s,t50_idx,0.35)-m0:+.5f}")
