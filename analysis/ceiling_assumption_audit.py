#!/usr/bin/env python3
"""AUDIT THE CEILING DERIVATION'S ASSUMPTIONS.

The paper's central claim now rests entirely on NSE_ceiling = 1 - (M/V)(e^{s^2}-1).
A reviewer will attack the assumptions, and the literature agent flagged one
specifically: if gauge error is IID daily, the effective sample size is
overstated and the ceiling's CI is too tight. Rating curves shift on
multi-month timescales, so errors are PERSISTENT within a shift epoch.

A1. Sensitivity: how much does the ceiling move per unit sigma? (is the
    scenario choice doing all the work?)
A2. The IID question: our derivation uses E[(y*eps - y)^2] = M*(e^{s^2}-1),
    which is an EXPECTATION and does NOT require independence across days.
    But the VARIANCE of the estimate does. Quantify: with persistent error,
    how much does the realised ceiling VARY run to run?
A3. Does the derivation hold when sigma is FLOW-DEPENDENT (our sigma_series)?
    Verify the analytic form against Monte Carlo on our actual truth series.
A4. Asymmetry: multiplicative lognormal noise is median-unbiased but MEAN-biased
    by e^{s^2/2}. Does that bias inflate our ceiling?
"""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
import sys; sys.path.insert(0,os.getcwd())
from target_feasibility import sigma_series
rng=np.random.default_rng(0)

ids=set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])
d=pd.read_csv("data/mblstm/gpu_dumps_s14/camels531_daymet_withq5_full531.csv.gz",
              usecols=["station_id","t0","h","truth"])
d=d[d.h==1].copy(); d["station_id"]=d.station_id.astype(str).str.zfill(8)
d=d[d.station_id.isin(ids)]
print(f"{d.station_id.nunique()} basins, {len(d):,} rows\n")

print("=== A1. SENSITIVITY: ceiling vs sigma (is scenario choice doing the work?) ===")
print(f"{'sigma':<10}{'median ceiling':>16}{'d(ceiling)/d(sigma)':>22}")
prev=None
for s in (0.10,0.13,0.15,0.20,0.25,0.30,0.35,0.41):
    cs=[]
    for sid,g in d.groupby("station_id"):
        q=g.truth.to_numpy(float); q=q[np.isfinite(q)]
        if len(q)<50: continue
        V=((q-q.mean())**2).mean(); M=(q**2).mean()
        if V<=0: continue
        cs.append(1-(M/V)*(np.exp(s**2)-1))
    m=float(np.median(cs))
    slope="" if prev is None else f"{(m-prev[1])/(s-prev[0]):>22.3f}"
    print(f"{s:<10}{m:>16.4f}{slope}")
    prev=(s,m)
print("  => the ceiling is STEEPLY sigma-dependent; the scenario IS the result.")

print("\n=== A3/A4. MONTE CARLO check of the analytic form on OUR truth series ===")
print("  (flow-dependent sigma, as the feasibility scripts use)")
print(f"{'scenario':<16}{'analytic':>12}{'MonteCarlo':>13}{'diff':>9}{'MC mean-bias':>14}")
for lab,(lo,hi) in (("optimistic",(0.25,0.13)),("central",(0.30,0.18)),("pessimistic",(0.41,0.30))):
    an=[];mc=[];bias=[]
    for sid,g in list(d.groupby("station_id"))[:120]:
        q=g.truth.to_numpy(float); q=q[np.isfinite(q)]
        if len(q)<50 or q.max()<=0: continue
        V=((q-q.mean())**2).mean(); M=(q**2).mean()
        if V<=0: continue
        sg=sigma_series(q,lo,hi)
        an.append(1-(M/V)*(np.exp(np.nanmean(sg**2))-1))
        # simulate: observed = truth * lognormal(median-unbiased)
        acc=[];bb=[]
        for _ in range(12):
            eps=np.exp(rng.normal(-sg**2/2,sg))
            obs=q*eps
            acc.append(1-np.mean((q-obs)**2)/np.var(obs))
            bb.append(obs.mean()/q.mean()-1)
        mc.append(np.mean(acc)); bias.append(np.mean(bb))
    print(f"{lab:<16}{np.median(an):>12.4f}{np.median(mc):>13.4f}"
          f"{np.median(an)-np.median(mc):>+9.4f}{np.median(bias):>+14.4f}")
print("  (diff within ~0.005 validates the closed form; mean-bias ~0 validates")
print("   the median-unbiased parameterisation)")

print("\n=== A2. PERSISTENCE: does correlated gauge error change the ceiling? ===")
print("  Rating shifts persist for months; simulate AR(1) error with varying rho.")
print(f"{'rho (daily)':<14}{'median ceiling':>16}{'spread (p10-p90)':>20}")
for rho in (0.0,0.5,0.9,0.98):
    cs=[]
    for sid,g in list(d.groupby("station_id"))[:120]:
        q=g.truth.to_numpy(float); q=q[np.isfinite(q)]
        if len(q)<50: continue
        sg=sigma_series(q,0.30,0.18)
        per=[]
        for _ in range(12):
            z=np.zeros(len(q)); e=rng.normal(0,1,len(q))
            for t in range(1,len(q)): z[t]=rho*z[t-1]+np.sqrt(1-rho**2)*e[t]
            obs=q*np.exp(sg*z-sg**2/2)
            per.append(1-np.mean((q-obs)**2)/np.var(obs))
        cs.append(np.mean(per))
    cs=np.array(cs)
    print(f"{rho:<14}{np.median(cs):>16.4f}{np.percentile(cs,90)-np.percentile(cs,10):>20.4f}")
print("  => if the median is stable across rho, the IID assumption is SAFE for the")
print("     point estimate (persistence widens the CI, not the centre).")
