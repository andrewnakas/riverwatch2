#!/usr/bin/env python3
"""Does the neighbour signal SURVIVE removing the 163 same-HUC8 (likely nested)
pairs the distance/area filter missed? If the gain came from nesting, it dies
here. This is the honest version of the result.
"""
import contextlib, io, json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg
rng=np.random.default_rng(0)

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

meta=json.load(open("data/camels_station_meta.json"))["stations"]
loc=pd.DataFrame([{"id":str(x["id"]).zfill(8),"lat":x["lat"],"lon":x["lon"],
                   "area":x.get("drain_area_sqmi",np.nan)} for x in meta
                  if x.get("lat") is not None]).set_index("id")
v15=json.load(open("data/stations_v15.json"))
st=v15["stations"] if isinstance(v15,dict) else v15
huc={}
for x in st:
    sid=str(x.get("site_no") or x.get("id") or "").zfill(8)
    if sid and x.get("huc_cd"): huc[sid]=str(x["huc_cd"]).zfill(8)

E=tr.pivot_table(index="_dt",columns="station_id",values="resid")
O=tr.pivot_table(index="_dt",columns="station_id",values="truth")
A=tr.pivot_table(index="_dt",columns="station_id",values="ens")
ids=[c for c in E.columns if c in loc.index]; E,O,A=E[ids],O[ids],A[ids]
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
same8=(H[:,None]==H[None,:])&(H[:,None]!="")
same4=np.array([[a[:4]==b[:4] and a!="" for b in H] for a in H])

Ev=E.to_numpy(float);Ov=O.to_numpy(float);Av=A.to_numpy(float)
fit=E.index<=pd.Timestamp("1990-09-30"); val=~fit
def nse(y,p):
    k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
    if len(y)<20 or np.var(y)<1e-9: return np.nan
    return 1-np.mean((y-p)**2)/np.var(y)
def run(Dx,K=3,ridge=1.0,shrink=0.25,maxd=200.0,shuffle=False):
    o=[]
    order=rng.permutation(n) if shuffle else None
    for j in range(n):
        nb=np.argsort(Dx[j])[:K]
        if not np.isfinite(Dx[j,nb]).all() or Dx[j,nb].max()>maxd: continue
        src = order[nb] if shuffle else nb
        X=Ev[:,src]
        Xf=X[fit]; yf=Ev[fit,j]
        m=np.isfinite(Xf).all(1)&np.isfinite(yf)
        if m.sum()<150: continue
        X1=np.hstack([Xf[m],np.ones((m.sum(),1))])
        try: c=np.linalg.solve(X1.T@X1+ridge*np.eye(K+1),X1.T@yf[m])
        except Exception: continue
        Xa=X[val]; ma=np.isfinite(Xa).all(1)
        adj=np.zeros(val.sum())
        adj[ma]=np.hstack([Xa[ma],np.ones((ma.sum(),1))])@c
        b=nse(Ov[val,j],Av[val,j]); a=nse(Ov[val,j],Av[val,j]+shrink*adj)
        if np.isfinite(b) and np.isfinite(a): o.append((b,a))
    o=np.array(o)
    if o.ndim!=2 or len(o)==0: return None
    d=o[:,1]-o[:,0]
    bs=[np.median(o[:,1][rng.choice(len(o),len(o),True)])-np.median(o[:,0][rng.choice(len(o),len(o),True)]) for _ in range(1000)]
    return (np.median(o[:,1])-np.median(o[:,0]), d.mean(), (d>0).mean(), len(o),
            np.percentile(bs,2.5), np.percentile(bs,97.5))

print(f"{'filter':<34}{'median':>10}{'mean':>10}{'improved':>10}{'n':>6}{'CI95':>22}")
print("-"*92)
for lab,excl in (("distance/area only",(D<25)&(ratio>5)),
                 ("+ same-HUC8 excluded",((D<25)&(ratio>5))|same8),
                 ("+ same-HUC4 excluded (strict)",((D<25)&(ratio>5))|same4)):
    Dx=D.copy(); Dx[excl]=np.inf
    r=run(Dx)
    if r: print(f"{lab:<34}{r[0]:>+10.5f}{r[1]:>+10.5f}{r[2]:>9.1%}{r[3]:>6}   [{r[4]:+.5f},{r[5]:+.5f}]")

print("\n=== CONTROL A (pre-registered): SHUFFLED neighbours ===")
Dx=D.copy(); Dx[((D<25)&(ratio>5))|same8]=np.inf
r=run(Dx,shuffle=True)
if r: print(f"{'shuffled (random distant basins)':<34}{r[0]:>+10.5f}{r[1]:>+10.5f}{r[2]:>9.1%}{r[3]:>6}   [{r[4]:+.5f},{r[5]:+.5f}]")
print("  (must be ~0 -- otherwise the gain is 'an extra channel', not neighbour info)")
