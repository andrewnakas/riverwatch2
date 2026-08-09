#!/usr/bin/env python3
"""The corrected ceiling is FLOW-WEIGHTED, so it is more sensitive to the SHAPE
of sigma(q) than the old form was. That shape is an assumption we never tested:
sigma_series interpolates linearly in LOG-FLOW between the 5th and 95th
percentiles, from s_low (at low flow) to s_high (at high flow).

Does the shape matter, or only the endpoints? Test alternatives that all share
the SAME endpoints (0.30 low -> 0.18 high, central) but differ in between:

  S1 log-linear (current)
  S2 linear in flow (not log) -- sigma stays high until very large flows
  S3 step at the median -- crude two-regime
  S4 constant at the flow-weighted mean of S1 -- tests whether variation matters
  S5 sigma constant = s_high everywhere (best case for high flows)
  S6 sigma constant = s_low everywhere (worst case)

If S1..S4 agree, the ceiling is robust to shape and only the endpoints matter --
which is the defensible claim for the paper. If they diverge, the shape is an
unstated free parameter and must be reported as such.
"""
import json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2")); sys.path.insert(0,os.getcwd())
from target_feasibility import sigma_series
ids=set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
d=pd.read_csv("data/mblstm/gpu_dumps_s14/camels531_daymet_withq5_full531.csv.gz",
              usecols=["station_id","t0","h","truth"])
d=d[d.h==1].copy(); d["station_id"]=d.station_id.astype(str).str.zfill(8)
d=d[d.station_id.isin(ids)]
LO,HI=0.30,0.18   # central: low-flow sigma, high-flow sigma

def ceil_new(q,sg):
    num=np.mean(q**2*(np.exp(sg**2)-1))
    Vo=np.mean(q**2*np.exp(sg**2))-(np.mean(q))**2
    return 1-num/Vo

def shapes(q):
    out={}
    out["S1 log-linear (current)"]=sigma_series(q,LO,HI)
    rng_=q.max()-q.min()
    f=np.clip((q-q.min())/rng_,0,1) if rng_>0 else np.full(q.shape,0.5)
    out["S2 linear in flow"]=LO+f*(HI-LO)
    med=np.median(q)
    out["S3 step at median"]=np.where(q>=med,HI,LO)
    s1=out["S1 log-linear (current)"]
    wmean=np.sqrt(np.average(s1**2,weights=q**2))   # flow-weighted rms
    out["S4 const = flow-wtd rms of S1"]=np.full(q.shape,wmean)
    out["S5 const = s_high (0.18)"]=np.full(q.shape,HI)
    out["S6 const = s_low (0.30)"]=np.full(q.shape,LO)
    return out

acc={}
for sid,g in list(d.groupby("station_id")):
    q=g.truth.to_numpy(float); q=q[np.isfinite(q)]
    if len(q)<50 or q.max()<=0: continue
    for k,sg in shapes(q).items():
        acc.setdefault(k,[]).append(ceil_new(q,sg))
print(f"n basins = {len(next(iter(acc.values())))}, central endpoints "
      f"s_low={LO} (low flow) -> s_high={HI} (high flow)\n")
print(f"{'sigma shape':<34}{'median ceiling':>16}{'vs current':>13}")
print("-"*63)
base=np.median(acc["S1 log-linear (current)"])
for k,v in acc.items():
    m=np.median(v)
    print(f"{k:<34}{m:>16.4f}{m-base:>+13.4f}")
print("\n=> S1-S4 close  => shape does not matter, only endpoints (defensible)")
print("=> S1-S4 spread => shape is a hidden free parameter (must be reported)")
print(f"\nfor scale, the S5-S6 span (all-high vs all-low sigma) is "
      f"{np.median(acc['S5 const = s_high (0.18)'])-np.median(acc['S6 const = s_low (0.30)']):+.4f}")
