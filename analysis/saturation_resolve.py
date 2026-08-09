#!/usr/bin/env python3
"""D4 came out NEGATIVE (basins that under-reach MORE have HIGHER NSE), which
contradicts 'under-reach = bad prediction'. Resolve it.

Likely confound: obs_max/obs_mean (spikiness) drives BOTH. Very spiky basins
have low NSE AND low ratio. Within-cohort and partial correlations separate it.

Also the decisive saturation test: SATURATION is a claim about the model's
NORMALISED output. Our models train on asinh/z-scored targets, so a cap lives
in normalised space. Test: does sim_max, expressed in each basin's OWN
normalised units, cluster at a common value across basins? A hard cap => tight
clustering. Ordinary error => wide spread.
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
top50=set(pd.read_csv("withq_top50_targets.csv").iloc[:,0].astype(str).str.zfill(8))

r=[]
for sid,g in m.groupby("station_id"):
    y=g.truth.to_numpy(float); e=g.ens.to_numpy(float)
    k=np.isfinite(y)&np.isfinite(e); y,e=y[k],e[k]
    if len(y)<20 or np.var(y)<1e-9 or y.max()<=0 or y.mean()<=0: continue
    sd=y.std()
    r.append(dict(sid=sid,nse=1-np.mean((y-e)**2)/np.var(y),
        ratio=e.max()/y.max(), spike=y.max()/y.mean(),
        # normalised-space maxima: how many basin-SDs above basin-mean
        obs_z=(y.max()-y.mean())/sd, sim_z=(e.max()-y.mean())/sd,
        top50=sid in top50))
df=pd.DataFrame(r)

print("=== Resolving D4: is spikiness the confound? ===")
def pc(a,b,c):
    """partial corr(a,b | c)"""
    A=np.asarray(a,float);B=np.asarray(b,float);C=np.asarray(c,float)
    ra=A-np.polyval(np.polyfit(C,A,1),C); rb=B-np.polyval(np.polyfit(C,B,1),C)
    return np.corrcoef(ra,rb)[0,1]
d=df.dropna()
ls=np.log(d.spike.clip(lower=1e-9))
print(f"  corr(ratio, NSE)                    = {np.corrcoef(d.ratio,d.nse)[0,1]:+.3f}")
print(f"  corr(ratio, log spikiness)          = {np.corrcoef(d.ratio,ls)[0,1]:+.3f}")
print(f"  corr(NSE,   log spikiness)          = {np.corrcoef(d.nse,ls)[0,1]:+.3f}")
print(f"  PARTIAL corr(ratio, NSE | spike)    = {pc(d.ratio,d.nse,ls):+.3f}")
print("\n  within spikiness quartiles:")
d=d.assign(q=pd.qcut(ls,4,labels=["low","q2","q3","high"]))
for q,s in d.groupby("q",observed=True):
    print(f"    {str(q):<5} n={len(s):>3}  corr(ratio,NSE) {np.corrcoef(s.ratio,s.nse)[0,1]:+.3f}"
          f"   med ratio {s.ratio.median():.3f}  med NSE {s.nse.median():.3f}")

print("\n=== DECISIVE: do simulated maxima cluster at a cap in NORMALISED units? ===")
print("   (obs_z / sim_z = basin-SDs above the basin mean)")
for lab,s in (("top-50",df[df.top50]),("others",df[~df.top50])):
    print(f"  {lab:<8} obs_z med {s.obs_z.median():>6.2f} [p10 {s.obs_z.quantile(.1):>5.2f}, "
          f"p90 {s.obs_z.quantile(.9):>6.2f}]   "
          f"sim_z med {s.sim_z.median():>6.2f} [p10 {s.sim_z.quantile(.1):>5.2f}, "
          f"p90 {s.sim_z.quantile(.9):>6.2f}]")
    print(f"           spread ratio (sim IQR / obs IQR) = "
          f"{(s.sim_z.quantile(.75)-s.sim_z.quantile(.25))/(s.obs_z.quantile(.75)-s.obs_z.quantile(.25)):.2f}")
print("\n  A HARD CAP => sim_z tightly clustered across basins (low spread).")
print("  ORDINARY ERROR => sim_z tracks obs_z with wide spread.")
print(f"\n  corr(sim_z, obs_z) all    = {np.corrcoef(df.sim_z,df.obs_z)[0,1]:+.3f}")
print(f"  corr(sim_z, obs_z) top-50 = "
      f"{np.corrcoef(df[df.top50].sim_z,df[df.top50].obs_z)[0,1]:+.3f}")
