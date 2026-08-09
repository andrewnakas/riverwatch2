#!/usr/bin/env python3
"""How much is the neighbour lever WORTH in principle?

The correction gave +0.0014 with 3 neighbours and heavy shrinkage. But that is
a linear, per-basin, same-day regression -- the crudest possible use of the
information. The real question for the paper: what is the UPPER BOUND on what
neighbour information could deliver?

U1. Scaling with K: does more neighbours help, or saturate immediately?
U2. Oracle-ish bound: fit the neighbour regression IN-SAMPLE on the scoring
    window. That is cheating, but it bounds what ANY neighbour-based method
    could extract with this feature set. If the in-sample bound is small, the
    lever is small no matter how cleverly it is modelled.
U3. Does adding neighbour LAGS (t-1, t+1) help? Flood waves propagate; a
    same-day-only feature set may be missing the actual signal.
U4. Which basins benefit? (dense-gauge regions vs isolated)
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
dl=Rr(lat[:,None]-lat[None,:]); dn=Rr(lon[:,None]-lon[None,:])
aa=np.sin(dl/2)**2+np.cos(Rr(lat))[:,None]*np.cos(Rr(lat))[None,:]*np.sin(dn/2)**2
D=6371*2*np.arcsin(np.sqrt(np.clip(aa,0,1)))
ar=np.where(np.isfinite(area),area,np.nan)
rt=np.maximum(ar[:,None]/ar[None,:],ar[None,:]/ar[:,None])
np.fill_diagonal(D,np.inf); D[(D<25)&(rt>5)]=np.inf
n=len(ids); Ev=E.to_numpy(float);Ov=O.to_numpy(float);Av=A.to_numpy(float)
fit=E.index<=pd.Timestamp("1990-09-30"); val=~fit
def nse(y,p):
    k=np.isfinite(y)&np.isfinite(p); y,p=y[k],p[k]
    if len(y)<20 or np.var(y)<1e-9: return np.nan
    return 1-np.mean((y-p)**2)/np.var(y)

def feats(j,K,lags):
    nb=np.argsort(D[j])[:K]
    if not np.isfinite(D[j,nb]).all(): return None
    out=[]
    for L_ in lags:
        X=Ev[:,nb]
        if L_!=0:
            X=np.roll(X,L_,axis=0)
            if L_>0: X[:L_]=np.nan
            else: X[L_:]=np.nan
        out.append(X)
    return np.hstack(out)

def run(K,lags,ridge=1.0,shrink=0.25,insample=False):
    o=[]
    for j in range(n):
        X=feats(j,K,lags)
        if X is None: continue
        fm = val if insample else fit
        Xf=X[fm]; yf=Ev[fm,j]
        m=np.isfinite(Xf).all(1)&np.isfinite(yf)
        if m.sum()<60: continue
        X1=np.hstack([Xf[m],np.ones((m.sum(),1))])
        try: c=np.linalg.solve(X1.T@X1+ridge*np.eye(X1.shape[1]),X1.T@yf[m])
        except Exception: continue
        Xa=X[val]; ma=np.isfinite(Xa).all(1)
        adj=np.zeros(val.sum())
        adj[ma]=np.hstack([Xa[ma],np.ones((ma.sum(),1))])@c
        b=nse(Ov[val,j],Av[val,j]); a=nse(Ov[val,j],Av[val,j]+shrink*adj)
        if np.isfinite(b) and np.isfinite(a): o.append((b,a))
    o=np.array(o)
    if o.ndim!=2 or len(o)==0: return float("nan"), float("nan"), 0
    return o[:,1].mean()-o[:,0].mean(), np.median(o[:,1])-np.median(o[:,0]), len(o)

print("=== U1. scaling with K (honest, shrink=0.25) ===")
for K in (1,2,3,5,8):
    mn,md,c=run(K,[0])
    print(f"  K={K}: median {md:+.5f}  mean {mn:+.5f}  n={c}")

print("\n=== U3. adding neighbour LAGS (flood-wave propagation) ===")
for lags,lab in (([0],"same-day"),([0,1],"t,t-1"),([-1,0,1],"t-1,t,t+1"),([0,1,2],"t,t-1,t-2")):
    mn,md,c=run(3,lags)
    print(f"  {lab:<12} median {md:+.5f}  mean {mn:+.5f}  n={c}")

print("\n=== U2. IN-SAMPLE upper bound (cheating -- bounds ANY method) ===")
for K,lags,lab in ((3,[0],"K=3 same-day"),(8,[-1,0,1],"K=8 with lags")):
    mn,md,c=run(K,lags,shrink=1.0,insample=True)
    print(f"  {lab:<16} median {md:+.5f}  mean {mn:+.5f}  n={c}")
print("  (this is the ceiling on neighbour-based correction with these features)")
