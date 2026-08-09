#!/usr/bin/env python3
"""Baste et al. 2025 (HESS 29:5871) output-saturation diagnostic.

Claim: LSTMs have an ARCHITECTURAL output ceiling well below the training max —
gating saturates, so the model literally cannot emit large enough values.
If true for our members, the top-50 flashy basins fail for a reason no amount
of new input information can fix, and hidden-size/output-scaling is a NEW lever.

Falsifier: if simulated peaks reach or exceed observed peaks (ratio ~>=1) the
model is NOT output-limited, and this is a citable closure instead.
"""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))

ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json")).get("531", []))
P = "data/mblstm/gpu_dumps_s14/camels531_{}_withq5_full531.csv.gz"
F = ("daymet", "maurer", "nldas", "aorc")

parts = {}
for f in F:
    d = pd.read_csv(P.format(f), usecols=["station_id","t0","h","truth","ylo","yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(ids)]
    d[f] = (d.ylo + d.yhi)/2
    parts[f] = d[["station_id","t0","truth",f]]
m = None
for f, d in parts.items():
    m = d if m is None else m.merge(d[["station_id","t0",f]], on=["station_id","t0"])
m["ens"] = m[list(F)].to_numpy(float).mean(1)

try:
    top50 = set(pd.read_csv("withq_top50_targets.csv").iloc[:,0].astype(str).str.zfill(8))
except Exception as e:
    top50 = set(); print(f"(top50 list unavailable: {e})")

rows = []
for sid, g in m.groupby("station_id"):
    y = g.truth.to_numpy(float); p = g.ens.to_numpy(float)
    k = np.isfinite(y) & np.isfinite(p); y, p = y[k], p[k]
    if len(y) < 20 or np.var(y) < 1e-9: continue
    n = 1 - np.mean((y-p)**2)/np.var(y)
    # peak-day subset: observed top 10%
    thr = np.percentile(y, 90)
    pk = y >= thr
    rows.append(dict(sid=sid, nse=n,
                     obs_max=y.max(), sim_max=p.max(),
                     ratio_max=p.max()/y.max() if y.max() > 0 else np.nan,
                     obs_p99=np.percentile(y,99), sim_p99=np.percentile(p,99),
                     peak_bias=(p[pk].mean()-y[pk].mean())/y[pk].mean() if pk.sum()>3 and y[pk].mean()>0 else np.nan,
                     spike=np.percentile(y[y>0],99)/np.median(y[y>0]) if (y>0).sum()>20 else np.nan,
                     top50=sid in top50))
df = pd.DataFrame(rows)
print(f"basins {len(df)}   top-50 flagged {df.top50.sum()}\n")

def blk(lab, s):
    if len(s)==0: return
    print(f"{lab:<26} n={len(s):>3}  medNSE {s.nse.median():>7.3f}   "
          f"sim_max/obs_max  med {s.ratio_max.median():.3f}  "
          f"p25 {s.ratio_max.quantile(.25):.3f}  "
          f"frac<0.9 {(s.ratio_max<0.9).mean():.2f}   "
          f"peak rel-bias {s.peak_bias.median():+.3f}")

print("=== SATURATION: can the model REACH the observed maximum? ===")
blk("ALL", df)
blk("top-50 targets", df[df.top50])
blk("others", df[~df.top50])
q = df.spike.quantile(.9)
blk("spikiest decile", df[df.spike>=q])
blk("worst-NSE decile", df.nsmallest(53,"nse"))

print("\n=== per-member (is any single member less saturated?) ===")
for f in F:
    r = []
    for sid, g in m.groupby("station_id"):
        y = g.truth.to_numpy(float); p = g[f].to_numpy(float)
        k = np.isfinite(y)&np.isfinite(p); y,p = y[k],p[k]
        if len(y)<20 or y.max()<=0: continue
        r.append(p.max()/y.max())
    r = np.array(r)
    print(f"  {f:<8} med sim_max/obs_max {np.median(r):.3f}   frac<0.9 {(r<0.9).mean():.2f}")

print("\n=== VERDICT ===")
t = df[df.top50] if df.top50.any() else df.nsmallest(50,"nse")
med = t.ratio_max.median()
if med < 0.75:
    print(f"  SATURATION LIKELY: target basins reach only {med:.2f} of observed max")
    print("  => hidden-size / output-scaling is a NEW mechanism-backed lever")
else:
    print(f"  NOT output-saturated: target basins reach {med:.2f} of observed max")
    print("  => Baste's architectural-ceiling mechanism does NOT explain our failure;")
    print("     citable closure. The error is magnitude SCATTER, not an output cap.")
