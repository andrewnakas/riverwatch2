#!/usr/bin/env python3
"""PRE-GPU SCREEN for the neighbour member: the NESTING AUDIT.

The single biggest failure risk. If basin B is upstream of basin A on the same
river, B's discharge is literally part of A's discharge and the member would
"work" spectacularly for a meaningless reason. Distance+area heuristics are not
enough -- two gauges on the same river 60km apart with a 3x area ratio pass the
current filter and are still nested.

Uses HUC codes (hydrologic unit) from the station registry: gauges sharing an
8-digit HUC are in the same sub-basin and are prime nesting suspects. We require
neighbours to differ at HUC8, and separately report how many pairs that removes.

Also reports final availability: how many of the 531 have >=3 clean neighbours.
"""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))

ids531=set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
meta=json.load(open("data/camels_station_meta.json"))["stations"]
loc=pd.DataFrame([{"id":str(x["id"]).zfill(8),"lat":x["lat"],"lon":x["lon"],
                   "area":x.get("drain_area_sqmi",np.nan)} for x in meta
                  if x.get("lat") is not None]).set_index("id")
# HUC from the big registry
v15=json.load(open("data/stations_v15.json"))
st=v15["stations"] if isinstance(v15,dict) else v15
huc={}
for x in st:
    sid=str(x.get("site_no") or x.get("id") or "").zfill(8)
    h=x.get("huc_cd")
    if sid and h: huc[sid]=str(h).zfill(8)
print(f"HUC codes found for {len(set(huc)&ids531)}/{len(ids531)} benchmark basins")

ids=[i for i in loc.index if i in ids531]
lat=loc.loc[ids,"lat"].to_numpy(float); lon=loc.loc[ids,"lon"].to_numpy(float)
area=loc.loc[ids,"area"].to_numpy(float)
R=np.radians
dl=R(lat[:,None]-lat[None,:]); dn=R(lon[:,None]-lon[None,:])
aa=np.sin(dl/2)**2+np.cos(R(lat))[:,None]*np.cos(R(lat))[None,:]*np.sin(dn/2)**2
D=6371*2*np.arcsin(np.sqrt(np.clip(aa,0,1)))
n=len(ids); np.fill_diagonal(D,np.inf)
ar=np.where(np.isfinite(area),area,np.nan)
ratio=np.maximum(ar[:,None]/ar[None,:],ar[None,:]/ar[:,None])

H=np.array([huc.get(i,"") for i in ids])
same_huc8=(H[:,None]==H[None,:]) & (H[:,None]!="")
same_huc4=np.array([[a[:4]==b[:4] and a!="" for b in H] for a in H])

old_filter=(D<25)&(ratio>5)
print(f"\npairs (upper triangle): {n*(n-1)//2:,}")
print(f"  flagged by OLD filter (<25km & >5x area): {int(old_filter.sum()/2):,}")
print(f"  share an 8-digit HUC:                     {int((same_huc8.sum()-n)/2):,}")
print(f"  share a 4-digit HUC (same basin region):  {int((same_huc4.sum()-n)/2):,}")
both=old_filter|same_huc8
print(f"  UNION (old filter OR same HUC8):          {int(both.sum()/2):,}")
print(f"  caught by HUC8 but NOT by the old filter: {int(((same_huc8&~old_filter).sum()-n)/2):,}  <-- leaks the old screen missed")

for lab,excl in (("old filter only",old_filter),
                 ("old + HUC8",old_filter|same_huc8),
                 ("old + HUC4 (strictest)",old_filter|same_huc4)):
    Dx=D.copy(); Dx[excl]=np.inf
    k3=np.sort(Dx,1)[:,:3]
    ok=np.isfinite(k3).all(1)
    within200=ok&(k3.max(1)<=200)
    print(f"\n  [{lab}]")
    print(f"    basins with >=3 clean neighbours:        {ok.sum()}/{n}")
    print(f"    ... and all 3 within 200km:              {within200.sum()}/{n}")
    print(f"    median distance to 3rd neighbour:        {np.nanmedian(k3[ok][:,2]):.1f} km")

print("\n=== SCREEN VERDICT (pre-registered: >=3 neighbours <200km for >=400 basins) ===")
Dx=D.copy(); Dx[old_filter|same_huc8]=np.inf
k3=np.sort(Dx,1)[:,:3]; ok=np.isfinite(k3).all(1)&(np.sort(Dx,1)[:,:3].max(1)<=200)
print(f"  with the HUC8-strengthened filter: {ok.sum()}/{n} basins qualify"
      f"  -> {'PASS' if ok.sum()>=400 else 'FAIL'}")
