#!/usr/bin/env python3
"""Is 'the near-median basins are all at their ceiling' robust to the sigma
scenario? If it only holds under the CENTRAL scenario it is weak; the
extrapolation measurement (7.2%) says OPTIMISTIC is the better description.

Also: the honest counter-question. If near-median basins are saturated under
optimistic sigma too, the no-q median genuinely cannot move and 0.845 is
unreachable -- regardless of how many members we build. That is a strong claim
and needs the strongest test.
"""
import contextlib, io, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg
from target_feasibility import sigma_series

with contextlib.redirect_stderr(io.StringIO()):
    tr, cols = build_merged(train=True)
for name, seeds in (("lstm_multi5",("s111","s222","s333","s444","s555")),
                    ("lstm_multi6",("s111","s222","s333"))):
    key=name.split("_")[1]
    d=load_seed_avg([L/f"camels531ls_{key}_nhlstm_TRAIN_{s}.csv.gz" for s in seeds])
    tr=tr.merge(d[["station_id","date","pred"]].rename(columns={"pred":name}),
                on=["station_id","date"],how="inner")
cols=cols+["lstm_multi5","lstm_multi6"]
tr["_dt"]=pd.to_datetime(tr.date)
fit=tr[tr._dt<="1990-09-30"]; val=tr[tr._dt>="1990-10-01"].copy()
y=fit.truth.to_numpy(float)
mse=np.array([np.nanmean((fit[c].to_numpy(float)-y)**2) for c in cols])
raw=mse**(-4.0); raw/=raw.sum(); w=0.25*np.ones(len(cols))/len(cols)+0.75*raw; w/=w.sum()
val=val.assign(ens=(val[cols].to_numpy(float)*w).sum(1))
def pbn(df,col):
    o={}
    for s,g in df.groupby("station_id"):
        yy=g.truth.to_numpy(float); pp=g[col].to_numpy(float)
        k=np.isfinite(yy)&np.isfinite(pp); yy,pp=yy[k],pp[k]
        if len(yy)<20 or np.var(yy)<1e-9: continue
        o[s]=1-np.mean((yy-pp)**2)/np.var(yy)
    return pd.Series(o)
s=pbn(val,"ens").sort_values(); n=len(s); mid=n//2; m0=s.iloc[mid]
near=list(s.index[[i for i in range(n) if abs(s.iloc[i]-m0)<0.01]])
flash=val.groupby("station_id")["truth"].apply(
    lambda x: np.percentile(x[x>0],99)/np.median(x[x>0]) if (x>0).sum()>20 else np.nan)
fthr=flash.median()

SC={"central":(0.30,0.18,0.35),"optimistic":(0.25,0.13,0.13),
    "well-gauged (Coxon 13%)":(0.20,0.13,0.13),"very optimistic":(0.15,0.10,0.10)}
print(f"median {m0:.4f}, near-median cohort n={len(near)}\n")
print(f"{'scenario':<26}{'near-med sat':>14}{'ALL sat':>10}{'near-med headroom':>20}{'median if all->ceiling':>24}")
print("-"*94)
for lab,(lo,hi,fl) in SC.items():
    rows=[]
    for sid,g in val.groupby("station_id"):
        q=g.truth.to_numpy(float); ok=np.isfinite(q)
        if ok.sum()<50: continue
        qq=q[ok]; V=((qq-qq.mean())**2).mean()
        if V<=0: continue
        M=(qq**2).mean()
        sh=fl if flash.get(sid,0)>fthr else hi
        s2=np.nanmean(sigma_series(q,lo,sh)**2)
        rows.append((sid,1-(M/V)*(np.exp(s2)-1)))
    ceil=pd.Series(dict(rows))
    i=[x for x in near if x in ceil.index]
    hr=(ceil.reindex(i)-s.reindex(i)).dropna()
    allh=(ceil-s.reindex(ceil.index)).dropna()
    capped=np.maximum(s.reindex(ceil.index),np.minimum(ceil,1.0))
    print(f"{lab:<26}{(hr<=0).sum():>7}/{len(hr):<6}"
          f"{(allh<=0).sum():>6}/{len(allh):<4}{hr.median():>+19.4f}{float(np.median(capped)):>24.4f}")
print("\n=> 'median if all basins reach their ceiling' is the honest upper bound")
print("   on what ANY amount of modelling can achieve on this metric.")
