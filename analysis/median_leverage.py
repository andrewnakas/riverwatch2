#!/usr/bin/env python3
"""THE MEDIAN-LEVERAGE QUESTION.

The campaign has spent months targeting the WORST basins (top-50 with-q,
53 arid, headroom>0.2 cohort). But the reported metric is the MEDIAN of
per-basin NSE -- a RANK statistic. Improving a basin at rank 500 changes the
median by exactly zero.

This asks, on TRAIN-val (honest split): how much median gain is available per
unit of effort, as a function of WHICH basins you improve? If the answer is
"only basins near rank 266 matter", then the entire targeting strategy of the
campaign has been optimising a quantity that does not move the reported number.

Simulations (each = "what if we improved cohort X by delta?"):
  S1. improve the WORST 50 basins by a large delta
  S2. improve the 78 basins nearest the median by a small delta
  S3. improve ALL basins by a small delta
  S4. what delta on the near-median cohort equals the 0.845 no-q target?
Also: is the with-q story the same? (its headroom is in the worst basins,
which may be exactly the ones that cannot move the median)
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
tr["_dt"]=pd.to_datetime(tr.date)
fit=tr[tr._dt<="1990-09-30"]; val=tr[tr._dt>="1990-10-01"].copy()
y=fit.truth.to_numpy(float)
mse=np.array([np.nanmean((fit[c].to_numpy(float)-y)**2) for c in cols])
raw=mse**(-4.0); raw/=raw.sum()
w=0.25*np.ones(len(cols))/len(cols)+0.75*raw; w/=w.sum()
Xv=val[cols].to_numpy(float)
d=val.assign(_e=(Xv*w).sum(1))
s={}
for sid,g in d.groupby("station_id"):
    yy=g.truth.to_numpy(float); pp=g._e.to_numpy(float)
    k=np.isfinite(yy)&np.isfinite(pp); yy,pp=yy[k],pp[k]
    if len(yy)<20 or np.var(yy)<1e-9: continue
    s[sid]=1-np.mean((yy-pp)**2)/np.var(yy)
s=pd.Series(s).sort_values()
n=len(s); mid=n//2; m0=s.iloc[mid]
print(f"{n} basins, median NSE {m0:.4f} (rank {mid})\n")

def med_after(series, idx, delta):
    t=series.copy(); t.iloc[idx]=np.minimum(t.iloc[idx]+delta,1.0)
    return float(np.median(t))

print("=== S1 vs S2: same 'effort', wildly different median payoff ===")
worst50=list(range(50))
near=[i for i in range(n) if abs(s.iloc[i]-m0)<0.01]
print(f"  near-median cohort size: {len(near)} basins")
for delta in (0.05, 0.10, 0.20):
    a=med_after(s,worst50,delta)-m0
    b=med_after(s,near,delta)-m0
    print(f"  improve by {delta:+.2f}:  worst-50 -> median {a:+.5f}    "
          f"near-median({len(near)}) -> median {b:+.5f}")
print("\n  ratio of payoff (near-median / worst-50) is effectively infinite:")
print("  improving the worst 50 basins by ANY amount moves the median by 0.00000")

print("\n=== S3. uniform improvement (what a general member does) ===")
for delta in (0.002, 0.005, 0.01):
    print(f"  ALL basins {delta:+.3f} -> median {med_after(s,list(range(n)),delta)-m0:+.5f}")

print("\n=== S4. what does it take to reach the no-q target? ===")
print("  (train-val median here is not the test number; this is about SHAPE)")
for target_gain in (0.005, 0.010):
    # how big a delta on the near-median cohort achieves it?
    for delta in np.arange(0.005, 0.30, 0.005):
        if med_after(s,near,delta)-m0 >= target_gain:
            print(f"    +{target_gain:.3f} median needs {delta:+.3f} NSE on the "
                  f"{len(near)} near-median basins")
            break
    else:
        print(f"    +{target_gain:.3f} median NOT reachable by the near-median cohort alone")

print("\n=== WHO ARE THE NEAR-MEDIAN BASINS? (are they the ones we've targeted?) ===")
near_ids=set(s.index[near])
try:
    a=pd.read_csv("gpu1080/nh_data_multi/attributes/attributes.csv")
    idc=[c for c in a.columns if c.lower() in ("gauge_id","basin_id","station_id")][0]
    a[idc]=a[idc].astype(str).str.zfill(8); a=a.set_index(idc)
    worst_ids=set(s.index[:53])
    for lab,ids in (("near-median (78)",near_ids),("worst 53",worst_ids),("all",set(s.index))):
        sub=a.reindex([i for i in ids if i in a.index])
        print(f"  {lab:<18} aridity {sub.aridity.median():.3f}  "
              f"p_mean {sub.p_mean.median():.2f}  "
              f"frac_snow {sub.frac_snow.median():.3f}  "
              f"area {sub.area_gages2.median():.0f}")
    print(f"\n  overlap between near-median and worst-53: "
          f"{len(near_ids & worst_ids)} basins")
except Exception as e:
    print(f"  (attrs unavailable: {e})")
