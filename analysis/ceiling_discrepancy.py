#!/usr/bin/env python3
"""The analytic ceiling and Monte Carlo disagree by 0.028-0.071 under
FLOW-DEPENDENT sigma, though memory records agreement to ~0.005 (which was
measured at CONSTANT sigma). Find the cause.

H1. The closed form uses mean(sigma^2) but the true MSE is E[y^2 * (e^{s(y)^2}-1)]
    -- a FLOW-WEIGHTED average, not a plain average. With flow-dependent sigma
    that are ANTI-correlated with flow (sigma is LOWER at high flow in our
    parameterisation), plain-averaging OVERSTATES the error at high flows,
    which are the ones that dominate M. => analytic too LOW. Testable.
H2. The denominator: NSE divides by Var(obs) not Var(truth); obs has inflated
    variance, which RAISES NSE. The closed form uses V = Var(truth).
Both push the same way. Quantify each.
"""
import json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2")); sys.path.insert(0,os.getcwd())
from target_feasibility import sigma_series
rng=np.random.default_rng(0)
ids=set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
d=pd.read_csv("data/mblstm/gpu_dumps_s14/camels531_daymet_withq5_full531.csv.gz",
              usecols=["station_id","t0","h","truth"])
d=d[d.h==1].copy(); d["station_id"]=d.station_id.astype(str).str.zfill(8)
d=d[d.station_id.isin(ids)]

print("=== constant sigma: does the closed form agree there? (memory says ~0.005) ===")
print(f"{'sigma':<8}{'analytic':>11}{'MonteCarlo':>12}{'diff':>9}")
for s in (0.13,0.25,0.35):
    an=[];mc=[]
    for sid,g in list(d.groupby("station_id"))[:120]:
        q=g.truth.to_numpy(float); q=q[np.isfinite(q)]
        if len(q)<50: continue
        V=((q-q.mean())**2).mean(); M=(q**2).mean()
        if V<=0: continue
        an.append(1-(M/V)*(np.exp(s**2)-1))
        acc=[]
        for _ in range(12):
            obs=q*np.exp(rng.normal(-s**2/2,s,len(q)))
            acc.append(1-np.mean((q-obs)**2)/np.var(obs))
        mc.append(np.mean(acc))
    print(f"{s:<8}{np.median(an):>11.4f}{np.median(mc):>12.4f}{np.median(an)-np.median(mc):>+9.4f}")

print("\n=== decompose the flow-dependent gap ===")
print(f"{'variant':<44}{'median':>10}")
rows={}
for lab,(lo,hi) in (("central",(0.30,0.18)),):
    a1=[];a2=[];a3=[];mc=[]
    for sid,g in list(d.groupby("station_id"))[:120]:
        q=g.truth.to_numpy(float); q=q[np.isfinite(q)]
        if len(q)<50 or q.max()<=0: continue
        V=((q-q.mean())**2).mean(); M=(q**2).mean()
        if V<=0: continue
        sg=sigma_series(q,lo,hi)
        # A: plain mean(sigma^2)  [what the scripts do]
        a1.append(1-(M/V)*(np.exp(np.nanmean(sg**2))-1))
        # B: flow-weighted, exact numerator E[y^2(e^{s^2}-1)]
        num=np.mean(q**2*(np.exp(sg**2)-1))
        a2.append(1-num/V)
        # C: exact numerator AND denominator Var(obs)=E[y^2 e^{s^2}]-(E[y])^2
        Vobs=np.mean(q**2*np.exp(sg**2))-(np.mean(q))**2
        a3.append(1-num/Vobs)
        acc=[]
        for _ in range(12):
            obs=q*np.exp(rng.normal(0,1,len(q))*sg-sg**2/2)
            acc.append(1-np.mean((q-obs)**2)/np.var(obs))
        mc.append(np.mean(acc))
    rows["A: 1-(M/V)(e^{mean s^2}-1)  [current]"]=np.median(a1)
    rows["B: exact numerator, V=Var(truth)"]=np.median(a2)
    rows["C: exact numerator AND V=Var(obs)"]=np.median(a3)
    rows["MONTE CARLO (ground truth)"]=np.median(mc)
for k,v in rows.items(): print(f"{k:<44}{v:>10.4f}")
print("\n  => whichever variant matches MC is the correct closed form.")
print("  => sigma is LOWER at high flow in our parameterisation, and high flows")
print("     dominate M, so plain-averaging sigma^2 OVERSTATES error => ceiling too low.")
