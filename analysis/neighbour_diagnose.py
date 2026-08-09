#!/usr/bin/env python3
"""Median +0.0028 but mean -0.047 and only 52.7% improved. Diagnose before
believing anything. A median gain with a catastrophic mean means a few basins
are being destroyed -- and 52.7% is indistinguishable from a coin flip.

Q1. Where does the mean collapse come from? (a few basins made much worse)
Q2. Is the median gain robust to SHRINKING the correction (ridge)?
Q3. Does it survive requiring the neighbour relation to be STABLE
    (fit on 1981-85, validated 1985-90, applied 1990-95)?
Q4. Is the gain concentrated in the near-median basins (which is what would
    actually move the metric) or scattered?
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
tr["resid"]=tr.truth-tr.ens; tr["_dt"]=pd.to_datetime(tr.date)
st=json.load(open("data/camels_station_meta.json"))["stations"]
loc=pd.DataFrame([{"id":str(x["id"]).zfill(8),"lat":x["lat"],"lon":x["lon"],
                   "area":x.get("drain_area_sqmi",np.nan)} for x in st
                  if x.get("lat") is not None]).set_index("id")
E=tr.pivot_table(index="_dt",columns="station_id",values="resid")
O=tr.pivot_table(index="_dt",columns="station_id",values="truth")
A=tr.pivot_table(index="_dt",columns="station_id",values="ens")
ids=[c for c in E.columns if c in loc.index]; E,O,A=E[ids],O[ids],A[ids]
lat=loc.loc[ids,"lat"].to_numpy(float); lon=loc.loc[ids,"lon"].to_numpy(float)
area=loc.loc[ids,"area"].to_numpy(float)
Rr=np.radians
dlat=Rr(lat[:,None]-lat[None,:]); dlon=Rr(lon[:,None]-lon[None,:])
aa=np.sin(dlat/2)**2+np.cos(Rr(lat))[:,None]*np.cos(Rr(lat))[None,:]*np.sin(dlon/2)**2
D=6371*2*np.arcsin(np.sqrt(np.clip(aa,0,1)))
ar=np.where(np.isfinite(area),area,np.nan)
ratio=np.maximum(ar[:,None]/ar[None,:], ar[None,:]/ar[:,None])
np.fill_diagonal(D,np.inf); D[(D<25)&(ratio>5)]=np.inf
n=len(ids)
Ev=E.to_numpy(float); Ov=O.to_numpy(float); Av=A.to_numpy(float)
fit=E.index<=pd.Timestamp("1990-09-30"); val=~fit
def nse(y,p):
    k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
    if len(y)<20 or np.var(y)<1e-9: return np.nan
    return 1-np.mean((y-p)**2)/np.var(y)

def run(K=3, ridge=1e-6, maxdist=np.inf, shrink=1.0):
    out=[]
    for j in range(n):
        nb=np.argsort(D[j])[:K]
        if not np.isfinite(D[j,nb]).all() or D[j,nb].max()>maxdist: continue
        Xf=Ev[np.ix_(fit,nb)]; yf=Ev[fit,j]
        m=np.isfinite(Xf).all(1)&np.isfinite(yf)
        if m.sum()<200: continue
        X1=np.hstack([Xf[m],np.ones((m.sum(),1))])
        try: c=np.linalg.solve(X1.T@X1+ridge*np.eye(K+1), X1.T@yf[m])
        except Exception: continue
        Xv=Ev[np.ix_(val,nb)]; mv=np.isfinite(Xv).all(1)
        adj=np.zeros(val.sum())
        adj[mv]=np.hstack([Xv[mv],np.ones((mv.sum(),1))])@c
        b=nse(Ov[val,j],Av[val,j]); a=nse(Ov[val,j],Av[val,j]+shrink*adj)
        if np.isfinite(b) and np.isfinite(a): out.append((ids[j],b,a))
    return pd.DataFrame(out,columns=["id","base","corr"])

r=run()
print(f"=== Q1. where does the mean collapse come from? (K=3, no shrink) ===")
r["d"]=r.corr_ if False else r["corr"]-r["base"]
print(f"  basins {len(r)}  median {r['corr'].median()-r['base'].median():+.5f}  "
      f"mean {r['corr'].mean()-r['base'].mean():+.5f}")
w5=r.nsmallest(5,"d")
print(f"  5 worst-damaged basins: {[f'{x.id}:{x.base:.2f}->{x.corr:.2f}' for x in w5.itertuples()]}")
print(f"  basins made worse by >0.05: {(r.d<-0.05).sum()}   by >0.20: {(r.d<-0.20).sum()}")

print(f"\n=== Q2. does SHRINKING the correction fix it? ===")
print(f"{'shrink':<10}{'median delta':>14}{'mean delta':>13}{'% improved':>12}{'worse>0.05':>12}")
for s in (1.0,0.5,0.25,0.1):
    rr=run(shrink=s); d=rr["corr"]-rr["base"]
    print(f"{s:<10}{rr['corr'].median()-rr['base'].median():>+14.5f}"
          f"{rr['corr'].mean()-rr['base'].mean():>+13.5f}{(d>0).mean():>11.1%}{(d<-0.05).sum():>12}")

print(f"\n=== Q3. stricter ridge + closer neighbours only ===")
for K,ridge,md in ((3,1.0,100),(3,10.0,100),(1,1.0,50),(3,100.0,np.inf)):
    rr=run(K=K,ridge=ridge,maxdist=md); d=rr["corr"]-rr["base"]
    lab=f"K={K} ridge={ridge} <{md}km"
    print(f"  {lab:<26} n={len(rr):>3} median {rr['corr'].median()-rr['base'].median():+.5f}"
          f"  mean {rr['corr'].mean()-rr['base'].mean():+.5f}  improved {(d>0).mean():.1%}")
