#!/usr/bin/env python3
"""THE LAST UNSTUDIED INPUT LEVER, and it is free to test.

Nobody has used concurrent NON-NESTED neighbour gauge observations as inputs
for gauged-basin CAMELS prediction. If our residual at basin A on a peak day
is predictable from the CONCURRENT residual at a nearby basin B, then B's
observed discharge carries information about A that our model is not using --
and that is new INFORMATION, the only thing that beats a conditional-mean
bound (arXiv:2606.04342).

Test (no retraining):
  1. per-basin daily residual (obs - ensemble) on the no-q TRAIN window
  2. pair basins by distance using CAMELS lat/lon (gauge_lat, gauge_lon)
  3. correlate residuals for near pairs vs far pairs, ALL days vs PEAK days
  4. the decisive number: partial correlation of residuals GIVEN that both
     basins share forcing error (they may just be seeing the same storm) --
     approximated by comparing near-pair residual corr to near-pair TRUTH corr

Kill criterion: if near-pair residual correlation on peak days is not clearly
above far-pair, the lever is dead for zero cost.
"""
import contextlib, io, json, os, sys
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
y=tr.truth.to_numpy(float)
mse=np.array([np.nanmean((tr[c].to_numpy(float)-y)**2) for c in cols])
raw=mse**(-4.0); raw/=raw.sum(); w=0.25*np.ones(len(cols))/len(cols)+0.75*raw; w/=w.sum()
tr=tr.assign(ens=(tr[cols].to_numpy(float)*w).sum(1))
tr["resid"]=tr.truth-tr.ens
# normalise residual per basin so big basins don't dominate correlations
g=tr.groupby("station_id")["resid"]
tr["rz"]=(tr.resid-g.transform("mean"))/g.transform("std").replace(0,np.nan)
tr["_thr"]=tr.groupby("station_id")["truth"].transform(lambda s:s.quantile(0.90))
tr["_pk"]=tr.truth>=tr._thr

# locations
import json as _json
_st = _json.load(open("data/camels_station_meta.json"))["stations"]
loc = pd.DataFrame([{"id": str(x["id"]).zfill(8), "lat": x["lat"], "lon": x["lon"],
                     "area_gages2": x.get("drain_area_sqmi", float("nan"))}
                    for x in _st if x.get("lat") is not None]).set_index("id")
if loc is None or len(loc) == 0:
    sys.exit("no lat/lon available")
print(f"locations for {len(loc)} basins\n")

P=tr.pivot_table(index="date",columns="station_id",values="rz")
PK=tr.pivot_table(index="date",columns="station_id",values="_pk")
T=tr.pivot_table(index="date",columns="station_id",values="truth")
ids=[c for c in P.columns if c in loc.index]
P=P[ids]; PK=PK[ids]; T=T[ids]
lat=loc.loc[ids,"lat"].to_numpy(float); lon=loc.loc[ids,"lon"].to_numpy(float)
n=len(ids); print(f"{n} basins x {len(P)} days")

R=np.radians
dlat=R(lat[:,None]-lat[None,:]); dlon=R(lon[:,None]-lon[None,:])
aa=np.sin(dlat/2)**2+np.cos(R(lat))[:,None]*np.cos(R(lat))[None,:]*np.sin(dlon/2)**2
D=6371*2*np.arcsin(np.sqrt(np.clip(aa,0,1)))

Rz=P.to_numpy(float); Tz=T.to_numpy(float); Pk=PK.to_numpy(float)==1
def corrmat(X):
    Xc=X-np.nanmean(X,0); s=np.nanstd(X,0)
    Xn=Xc/np.where(s>0,s,np.nan)
    M=np.ma.masked_invalid(Xn)
    C=np.ma.dot(M.T,M)/np.ma.dot((~M.mask).astype(float).T,(~M.mask).astype(float))
    return np.ma.filled(C,np.nan)
Call=corrmat(Rz)
Rpk=np.where(Pk,Rz,np.nan)
Cpk=corrmat(Rpk)
Ctruth=corrmat(Tz)

iu=np.triu_indices(n,1)
d=D[iu]; ca=Call[iu]; cp=Cpk[iu]; ct=Ctruth[iu]
print("\n=== residual correlation vs distance ===")
print(f"{'band (km)':<14}{'pairs':>8}{'resid corr ALL':>16}{'resid corr PEAK':>17}{'truth corr':>12}")
print("-"*68)
for lo_,hi_ in ((0,50),(50,100),(100,200),(200,500),(500,1000),(1000,99999)):
    m=(d>=lo_)&(d<hi_)&np.isfinite(ca)
    mp=(d>=lo_)&(d<hi_)&np.isfinite(cp)
    if m.sum()<20: continue
    print(f"{lo_}-{hi_ if hi_<9999 else '+':<10}{m.sum():>8}"
          f"{np.nanmedian(ca[m]):>16.4f}{np.nanmedian(cp[mp]):>17.4f}"
          f"{np.nanmedian(ct[m]):>12.4f}")

near=(d<100)&np.isfinite(ca); far=(d>500)&np.isfinite(ca)
nearp=(d<100)&np.isfinite(cp); farp=(d>500)&np.isfinite(cp)
print(f"\n=== VERDICT ===")
print(f"  near(<100km) resid corr ALL  {np.nanmedian(ca[near]):.4f}   "
      f"far(>500km) {np.nanmedian(ca[far]):.4f}   lift {np.nanmedian(ca[near])-np.nanmedian(ca[far]):+.4f}")
print(f"  near(<100km) resid corr PEAK {np.nanmedian(cp[nearp]):.4f}   "
      f"far(>500km) {np.nanmedian(cp[farp]):.4f}   lift {np.nanmedian(cp[nearp])-np.nanmedian(cp[farp]):+.4f}")
print(f"  (for scale, near-pair TRUTH corr {np.nanmedian(ct[near]):.4f})")
lift=np.nanmedian(cp[nearp])-np.nanmedian(cp[farp])
print(f"\n  {'LEVER ALIVE: neighbour residuals carry signal' if lift>0.10 else 'LEVER DEAD: near-pair residual correlation is not materially above far-pair'}")
