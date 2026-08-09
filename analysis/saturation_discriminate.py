#!/usr/bin/env python3
"""Is the top-50 under-reach ARCHITECTURAL SATURATION or just ordinary error?

Three discriminating tests. Saturation (Baste) predicts a HARD CAP:
  D1. The cap should be a property of the MODEL, not the basin. Normalise by
      basin scale: if saturation is real, sim_max/obs_max should fall as
      obs_max grows *in normalised units the model actually sees*.
  D2. A hard cap means the model's largest outputs CLUSTER at a ceiling —
      sim_max should be compressed relative to how variable obs_max is.
  D3. Ensemble averaging alone shrinks maxima. Compare the BEST SINGLE MEMBER's
      max to the ensemble max. If single members also cap out, it is the model;
      if singles reach much higher, it is averaging.
  D4. If it is a cap, under-reach should NOT correlate with how well the basin
      is predicted overall (a cap is indifferent to skill). If under-reach
      tracks NSE tightly, it is just "bad prediction".
"""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))

ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json")).get("531",[]))
P = "data/mblstm/gpu_dumps_s14/camels531_{}_withq5_full531.csv.gz"
F = ("daymet","maurer","nldas","aorc")
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
top50=set(pd.read_csv("withq_top50_targets.csv").iloc[:,0].astype(str).str.zfill(8))

r=[]
for sid,g in m.groupby("station_id"):
    y=g.truth.to_numpy(float); e=g.ens.to_numpy(float)
    k=np.isfinite(y)&np.isfinite(e); y,e=y[k],e[k]
    if len(y)<20 or np.var(y)<1e-9 or y.max()<=0: continue
    singles={f:g[f].to_numpy(float)[k] for f in F}
    best_single_max=max(np.nanmax(v) for v in singles.values())
    r.append(dict(sid=sid, nse=1-np.mean((y-e)**2)/np.var(y),
        obs_max=y.max(), ens_max=e.max(), best_single_max=best_single_max,
        ratio_ens=e.max()/y.max(), ratio_single=best_single_max/y.max(),
        obs_mean=y.mean(), sim_mean=e.mean(),
        obs_max_over_mean=y.max()/y.mean() if y.mean()>0 else np.nan,
        sim_max_over_mean=e.max()/e.mean() if e.mean()>0 else np.nan,
        top50=sid in top50))
df=pd.DataFrame(r)

print("=== D3. Ensemble averaging vs the model itself ===")
for lab,s in (("top-50",df[df.top50]),("others",df[~df.top50])):
    print(f"  {lab:<9} ens_max/obs_max {s.ratio_ens.median():.3f}   "
          f"BEST-SINGLE_max/obs_max {s.ratio_single.median():.3f}   "
          f"lift {s.ratio_single.median()-s.ratio_ens.median():+.3f}")
print("  => if lift is small, averaging is NOT the explanation\n")

print("=== D2. Dynamic range: does the model compress peak/mean? ===")
for lab,s in (("top-50",df[df.top50]),("others",df[~df.top50])):
    print(f"  {lab:<9} obs max/mean {s.obs_max_over_mean.median():>8.1f}   "
          f"sim max/mean {s.sim_max_over_mean.median():>8.1f}   "
          f"compression {s.sim_max_over_mean.median()/s.obs_max_over_mean.median():.2f}x")
print("  => a hard output cap compresses dynamic range on spiky basins\n")

print("=== D4. Does under-reach track SKILL (ordinary error) or stand alone? ===")
d2=df.dropna(subset=["ratio_ens","nse"])
print(f"  corr(ratio_ens, NSE) all      = {np.corrcoef(d2.ratio_ens,d2.nse)[0,1]:+.3f}")
o=d2[~d2.top50]
print(f"  corr(ratio_ens, NSE) others   = {np.corrcoef(o.ratio_ens,o.nse)[0,1]:+.3f}")
t=d2[d2.top50]
print(f"  corr(ratio_ens, NSE) top-50   = {np.corrcoef(t.ratio_ens,t.nse)[0,1]:+.3f}")
print("  => strong positive corr means 'under-reach' is just 'bad prediction'\n")

print("=== D1. Is under-reach explained by basin spikiness? ===")
d2=d2.assign(spike=d2.obs_max_over_mean)
print(f"  corr(ratio_ens, log obs_max/mean) = "
      f"{np.corrcoef(d2.ratio_ens,np.log(d2.spike.clip(lower=1e-9)))[0,1]:+.3f}")
print(f"  corr(ratio_ens, log obs_max)      = "
      f"{np.corrcoef(d2.ratio_ens,np.log(d2.obs_max.clip(lower=1e-9)))[0,1]:+.3f}")
print("  (obs_max corr near 0 argues AGAINST an absolute cap in physical units)")
