#!/usr/bin/env python3
"""The shrunk neighbour correction gives median +0.0035, mean +0.0027, 73%
improved on TRAIN-val. Before this is called a lever it needs the same
robustness bar every member faced.

R1. Is shrink=0.25 selected ON the validation window? -> nested split:
    fit 1981-1987, SELECT shrink on 1987-1990, SCORE on 1990-1995.
R2. Bootstrap CI over basins.
R3. Does it help the NEAR-MEDIAN basins (the only ones that move the metric)?
R4. Is it just persistence/autocorrelation in disguise? Compare against using
    the basin's OWN lagged residual (which a no-q model also cannot see, but
    which is NOT new spatial information).
R5. Does the gain survive on PEAK days specifically?
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
ratio=np.maximum(ar[:,None]/ar[None,:],ar[None,:]/ar[:,None])
np.fill_diagonal(D,np.inf); D[(D<25)&(ratio>5)]=np.inf
n=len(ids)
Ev=E.to_numpy(float);Ov=O.to_numpy(float);Av=A.to_numpy(float);idx=E.index
def nse(y,p,m=None):
    if m is not None: y,p=y[m],p[m]
    k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
    if len(y)<20 or np.var(y)<1e-9: return np.nan
    return 1-np.mean((y-p)**2)/np.var(y)

def build(fit_m, app_m, K=3, ridge=1.0, shrink=0.25, own_lag=False):
    out=[]
    for j in range(n):
        if own_lag:
            src=np.roll(Ev[:,j],1); src[0]=np.nan; X=src[:,None]; kk=1
        else:
            nb=np.argsort(D[j])[:K]
            if not np.isfinite(D[j,nb]).all(): continue
            X=Ev[:,nb]; kk=K
        Xf=X[fit_m]; yf=Ev[fit_m,j]
        m=np.isfinite(Xf).all(1)&np.isfinite(yf)
        if m.sum()<150: continue
        X1=np.hstack([Xf[m],np.ones((m.sum(),1))])
        try: c=np.linalg.solve(X1.T@X1+ridge*np.eye(kk+1),X1.T@yf[m])
        except Exception: continue
        Xa=X[app_m]; ma=np.isfinite(Xa).all(1)
        adj=np.zeros(app_m.sum())
        adj[ma]=np.hstack([Xa[ma],np.ones((ma.sum(),1))])@c
        out.append((ids[j], nse(Ov[app_m,j],Av[app_m,j]),
                    nse(Ov[app_m,j],Av[app_m,j]+shrink*adj)))
    return pd.DataFrame(out,columns=["id","base","corr"]).dropna()

f1=idx<=pd.Timestamp("1987-09-30")
sel=(idx>pd.Timestamp("1987-09-30"))&(idx<=pd.Timestamp("1990-09-30"))
sc=idx>pd.Timestamp("1990-09-30")
print("=== R1. NESTED selection (fit 81-87, select shrink 87-90, score 90-95) ===")
best=None
for s in (0.1,0.25,0.5,1.0):
    r=build(f1,sel,shrink=s); d=(r["corr"]-r["base"]).median()
    print(f"  shrink {s}: selection-window median delta {d:+.5f}")
    if best is None or d>best[1]: best=(s,d)
print(f"  -> selected shrink={best[0]} on the SELECTION window")
r=build(f1,sc,shrink=best[0]); d=r["corr"]-r["base"]
print(f"  SCORED on 1990-95: median {d.median():+.5f}  mean {d.mean():+.5f}  "
      f"improved {(d>0).mean():.1%}  n={len(r)}")
bs=[np.median(d.to_numpy()[rng.choice(len(d),len(d),True)]) for _ in range(2000)]
lo,hi=np.percentile(bs,[2.5,97.5])
print(f"  bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}]  "
      f"{'EXCLUDES ZERO' if lo>0 else 'STRADDLES ZERO'}")

print("\n=== R4. is it just the basin's OWN lagged residual? (not spatial) ===")
ro=build(f1,sc,shrink=best[0],own_lag=True); do=ro["corr"]-ro["base"]
print(f"  own-lag-1 residual: median {do.median():+.5f}  improved {(do>0).mean():.1%}")
print(f"  neighbour        : median {d.median():+.5f}  improved {(d>0).mean():.1%}")
print("  (if own-lag is comparable, the 'spatial' story is really autocorrelation)")

print("\n=== R3. does it help the NEAR-MEDIAN basins? ===")
b=r.set_index("id")["base"]; s0=b.sort_values(); m0=s0.iloc[len(s0)//2]
near=set(s0.index[(s0-m0).abs()<0.01])
rn=r[r.id.isin(near)]; dn=rn["corr"]-rn["base"]
print(f"  near-median cohort n={len(rn)}: median delta {dn.median():+.5f}  "
      f"improved {(dn>0).mean():.1%}")
print(f"  => median of the WHOLE set moves {r['corr'].median()-r['base'].median():+.5f}")
