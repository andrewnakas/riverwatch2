#!/usr/bin/env python3
"""GIVEN the median-leverage finding, the ONLY question that matters for any
new member is: DOES IT HELP THE NEAR-MEDIAN BASINS?

This asks what is actually wrong with the ~78 near-median basins, so a member
can be designed for THEM rather than for the worst basins. Completely different
target from everything the campaign has built.

Q1. Are near-median basins peak-error-limited like the worst ones, or do they
    fail differently? (If they fail differently, the whole peak-error framing
    applies to basins that cannot move the metric!)
Q2. What is their headroom to the gauge ceiling? (if they are SATURATED, the
    median genuinely cannot move and the result is final)
Q3. Which existing stream is best on THEM? Is the ensemble already optimal there?
Q4. Do multi5/multi6 help them? (multi5 worked -- was it via these basins?)
"""
import contextlib, io, json, os, sys
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
Xv=val[cols].to_numpy(float)
val=val.assign(ens=(Xv*w).sum(1))

def pbn(df,col):
    o={}
    for s,g in df.groupby("station_id"):
        yy=g.truth.to_numpy(float); pp=g[col].to_numpy(float)
        k=np.isfinite(yy)&np.isfinite(pp); yy,pp=yy[k],pp[k]
        if len(yy)<20 or np.var(yy)<1e-9: continue
        o[s]=1-np.mean((yy-pp)**2)/np.var(yy)
    return pd.Series(o)
s=pbn(val,"ens").sort_values(); n=len(s); mid=n//2; m0=s.iloc[mid]
near=set(s.index[[i for i in range(n) if abs(s.iloc[i]-m0)<0.01]])
worst=set(s.index[:53])
print(f"median {m0:.4f}; near-median cohort n={len(near)}, worst53 n={len(worst)}\n")

print("=== Q1. HOW does each cohort fail? (share of squared error on peak days) ===")
val["_thr"]=val.groupby("station_id")["truth"].transform(lambda x:x.quantile(0.90))
val["_se"]=(val.truth-val.ens)**2
val["_pk"]=val.truth>=val._thr
for lab,ids in (("near-median",near),("worst53",worst),("ALL",set(s.index))):
    d=val[val.station_id.isin(ids)]
    shr=d.loc[d._pk,"_se"].sum()/d._se.sum()
    # relative bias on peaks
    pk=d[d._pk]
    rb=(pk.ens.sum()-pk.truth.sum())/pk.truth.sum()
    print(f"  {lab:<12} peak-day share of SE {shr:.3f}   peak rel-bias {rb:+.3f}")

print("\n=== Q2. GAUGE-CEILING HEADROOM of the near-median cohort ===")
flash=val.groupby("station_id")["truth"].apply(
    lambda x: np.percentile(x[x>0],99)/np.median(x[x>0]) if (x>0).sum()>20 else np.nan)
thr=flash.median()
rows=[]
for sid,g in val.groupby("station_id"):
    q=g.truth.to_numpy(float); ok=np.isfinite(q)
    if ok.sum()<50: continue
    qq=q[ok]; V=((qq-qq.mean())**2).mean()
    if V<=0: continue
    M=(qq**2).mean()
    s_hi=0.35 if flash.get(sid,0)>thr else 0.18
    s2=np.nanmean(sigma_series(q,0.30,s_hi)**2)
    rows.append((sid,1-(M/V)*(np.exp(s2)-1)))
ceil=pd.Series(dict(rows))
for lab,ids in (("near-median",near),("worst53",worst),("ALL",set(s.index))):
    i=[x for x in ids if x in ceil.index]
    hr=(ceil.reindex(i)-s.reindex(i)).dropna()
    at=(hr<=0).sum()
    print(f"  {lab:<12} median headroom {hr.median():+.4f}   at/above ceiling {at}/{len(hr)} ({100*at/max(len(hr),1):.0f}%)")

print("\n=== Q3. best stream ON THE NEAR-MEDIAN COHORT ===")
sub=val[val.station_id.isin(near)]
res={c: pbn(sub,c).median() for c in cols}
res["ENSEMBLE"]=pbn(sub,"ens").median()
for k,v in sorted(res.items(), key=lambda t:-t[1]):
    mark=" <-- ensemble" if k=="ENSEMBLE" else ""
    print(f"  {k:<16}{v:.4f}{mark}")
best=max((k for k in res if k!="ENSEMBLE"), key=lambda k:res[k])
print(f"  ensemble beats best single member by {res['ENSEMBLE']-res[best]:+.4f}")

print("\n=== Q4. do multi5/multi6 help the NEAR-MEDIAN basins specifically? ===")
for drop in ("lstm_multi5","lstm_multi6"):
    c2=[c for c in cols if c!=drop]
    i2=[cols.index(c) for c in c2]
    w2=w[i2]/w[i2].sum()
    e2=(sub[c2].to_numpy(float)*w2).sum(1)
    d=pbn(sub.assign(_x=e2),"_x").median()
    print(f"  without {drop:<14} {d:.4f}   (LOO value {res['ENSEMBLE']-d:+.5f})")
