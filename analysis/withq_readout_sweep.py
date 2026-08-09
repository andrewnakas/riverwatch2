#!/usr/bin/env python3
"""FREE LEVER TEST: is (ylo+yhi)/2 optimal, or does a different blend of the
predictive quantiles score higher? w*ymed + (1-w)*(ylo+yhi)/2, plus asymmetric
blends. Selection must be TRAIN-side honest, so we also report a temporal split.
"""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))

ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json")).get("531", []))
P = "data/mblstm/gpu_dumps_s14/camels531_{}_withq5_full531.csv.gz"
F = ("daymet", "maurer", "nldas", "aorc")

parts = {}
for f in F:
    d = pd.read_csv(P.format(f), usecols=["station_id","t0","h","truth","ylo","ymed","yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(ids)]
    parts[f] = d.rename(columns={"ylo":f"lo_{f}","ymed":f"md_{f}","yhi":f"hi_{f}"})[
        ["station_id","t0","truth",f"lo_{f}",f"md_{f}",f"hi_{f}"]]
m = None
for f, d in parts.items():
    m = d if m is None else m.merge(d.drop(columns=["truth"]), on=["station_id","t0"])
m["_dt"] = pd.to_datetime(m.t0)
print(f"rows {len(m):,}  basins {m.station_id.nunique()}  "
      f"{m._dt.min().date()} -> {m._dt.max().date()}\n")


def score(df, build):
    ens = np.mean([build(df, f) for f in F], axis=0)
    df = df.assign(_e=ens)
    o = []
    for s, g in df.groupby("station_id"):
        y = g.truth.to_numpy(float); p = g._e.to_numpy(float)
        k = np.isfinite(y) & np.isfinite(p); y, p = y[k], p[k]
        if len(y) < 20 or np.var(y) < 1e-9: continue
        o.append(1 - np.mean((y-p)**2)/np.var(y))
    return float(np.median(o))


cands = {
    "ymed                      ": lambda d,f: d[f"md_{f}"].to_numpy(float),
    "(ylo+yhi)/2   [RECORD]    ": lambda d,f: (d[f"lo_{f}"]+d[f"hi_{f}"]).to_numpy(float)/2,
    "(ylo+ymed+yhi)/3          ": lambda d,f: (d[f"lo_{f}"]+d[f"md_{f}"]+d[f"hi_{f}"]).to_numpy(float)/3,
    "0.5*ymed+0.5*mid          ": lambda d,f: 0.5*d[f"md_{f}"].to_numpy(float)+0.25*(d[f"lo_{f}"]+d[f"hi_{f}"]).to_numpy(float),
    "0.25*ylo+0.75*yhi         ": lambda d,f: (0.25*d[f"lo_{f}"]+0.75*d[f"hi_{f}"]).to_numpy(float),
    "0.75*ylo+0.25*yhi         ": lambda d,f: (0.75*d[f"lo_{f}"]+0.25*d[f"hi_{f}"]).to_numpy(float),
    "yhi (p90)                 ": lambda d,f: d[f"hi_{f}"].to_numpy(float),
    "ylo (p10)                 ": lambda d,f: d[f"lo_{f}"].to_numpy(float),
}
SPLIT = "1994-10-01"
early, late = m[m._dt < SPLIT], m[m._dt >= SPLIT]
print(f"{'readout':<28}{'ALL':>9}{'early':>9}{'late':>9}")
print("-"*55)
res = {}
for lab, fn in cands.items():
    a, e, l = score(m, fn), score(early, fn), score(late, fn)
    res[lab] = a
    print(f"{lab:<28}{a:>9.4f}{e:>9.4f}{l:>9.4f}")
best = max(res, key=res.get)
print(f"\nbest overall: {best.strip()} = {res[best]:.4f}")
print(f"record       : {res['(ylo+yhi)/2   [RECORD]    ']:.4f}")
print(f"delta        : {res[best]-res['(ylo+yhi)/2   [RECORD]    ']:+.4f}")
