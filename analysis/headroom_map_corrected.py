#!/usr/bin/env python3
"""Redraw the headroom map with the CORRECTED ceiling, on the no-q TEST frame.

The old map ([[headroom-where-effort-pays]]) said 29% of basins were saturated
and headroom lived in ~104 flashy basins. Both were computed with the buggy
form. And [[median-leverage]] says only near-median-RANK basins can move the
reported metric anyway.

So the operational question is now: WHICH basins are BOTH (a) below their
corrected ceiling AND (b) near enough the median rank to matter?
That intersection -- if non-empty -- is the only place effort can pay.
"""
import contextlib, io, json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2")); sys.path.insert(0,os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg
from target_feasibility import sigma_series

with contextlib.redirect_stderr(io.StringIO()):
    m, cols = build_merged(train=False)
for nm,sd in (("lstm_multi5",("s111","s222","s333","s444","s555")),
              ("lstm_multi6",("s111","s222","s333"))):
    k=nm.split("_")[1]
    d=load_seed_avg([L/f"camels531ls_{k}_nhlstm_{s}.csv.gz" for s in sd])
    m=m.merge(d[["station_id","date","pred"]].rename(columns={"pred":nm}),
              on=["station_id","date"],how="inner")
cols=cols+["lstm_multi5","lstm_multi6"]
r=json.load(open("noq_test_result.json")); w=np.array([r["weights"][c] for c in cols])
m=m.assign(ens=(m[cols].to_numpy(float)*w).sum(1))
fl=m.groupby("station_id")["truth"].apply(
    lambda s: np.percentile(s[s>0],99)/np.median(s[s>0]) if (s>0).sum()>20 else np.nan)
th=fl.median()

rows=[]
for sid,g in m.groupby("station_id"):
    q=g.truth.to_numpy(float); ok=np.isfinite(q)
    if ok.sum()<50: continue
    qq=q[ok]; V=((qq-qq.mean())**2).mean()
    if V<=0: continue
    p=g.ens.to_numpy(float)[ok]
    cur=1-np.mean((qq-p)**2)/np.var(qq)
    rec={}
    for lab,(lo,hi,flv) in (("cen",(0.30,0.18,0.35)),("opt",(0.25,0.13,0.13))):
        sh=flv if fl.get(sid,0)>th else hi
        sg=sigma_series(qq,lo,sh)
        num=np.mean(qq**2*(np.exp(sg**2)-1))
        Vo=np.mean(qq**2*np.exp(sg**2))-(np.mean(qq))**2
        rec[lab]=1-num/Vo
    rows.append(dict(sid=sid,nse=cur,cen=rec["cen"],opt=rec["opt"],flash=fl.get(sid,np.nan)))
df=pd.DataFrame(rows).dropna(subset=["nse"])
df=df.sort_values("nse").reset_index(drop=True)
n=len(df); mid=n//2; m0=df.nse.iloc[mid]
df["rank"]=np.arange(n)
df["near"]=(df.nse-m0).abs()<0.01
print(f"no-q TEST: {n} basins, median {m0:.4f} at rank {mid}\n")

for lab in ("cen","opt"):
    df["hr"]=df[lab]-df.nse
    below=df.hr>0
    both=below&df.near
    name={"cen":"CENTRAL","opt":"OPTIMISTIC"}[lab]
    print(f"=== {name} scenario ===")
    print(f"  below ceiling            {below.sum():>4}/{n}")
    print(f"  near median rank         {df.near.sum():>4}/{n}")
    print(f"  BOTH (where effort pays) {both.sum():>4}/{n}"
          f"   median headroom there {df.loc[both,'hr'].median():+.4f}")
    if both.sum():
        # how much median gain if ONLY those basins reach their ceiling?
        t=df.nse.copy(); t[both]=np.minimum(df.loc[both,lab],1.0)
        print(f"  -> raising ONLY those to ceiling moves median "
              f"{float(np.median(t))-m0:+.5f}")
        sub=df[both]
        print(f"  -> their profile: median flashiness {sub.flash.median():.1f} "
              f"(all basins {df.flash.median():.1f})")
    t=df.nse.copy(); sel=df.hr>0
    t[sel]=np.minimum(df.loc[sel,lab],1.0)
    print(f"  -> raising ALL below-ceiling basins moves median "
          f"{float(np.median(t))-m0:+.5f}\n")
print("=> the 'BOTH' row is the only cohort where modelling effort can move")
print("   the reported metric AND the observations can reward it.")
