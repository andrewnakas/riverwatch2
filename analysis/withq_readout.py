#!/usr/bin/env python3
"""Why is the with-q record 0.9203 but target_feasibility says 0.9177?
Hypothesis: readout column. record_withq5.py uses (ylo+yhi)/2; feasibility uses ymed.
Test all readouts on IDENTICAL frames, 3-forcing and 4-forcing."""
import json, os
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))

ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json")).get("531", []))
P = "data/mblstm/gpu_dumps_s14/camels531_{}_withq5_full531.csv.gz"


def load(f):
    d = pd.read_csv(P.format(f), usecols=["station_id","t0","h","truth","ylo","ymed","yhi","ymean"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    return d[d.station_id.isin(ids)]


def med_nse(m, cols):
    m = m.copy()
    m["ens"] = m[cols].to_numpy(float).mean(1)
    out = []
    for s, g in m.groupby("station_id"):
        y = g.truth.to_numpy(float); p = g.ens.to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p); y, p = y[ok], p[ok]
        if len(y) < 20 or np.var(y) < 1e-9: continue
        out.append(1 - np.mean((y-p)**2)/np.var(y))
    return float(np.median(out)), len(out)


for forcings in (("daymet","maurer","nldas"), ("daymet","maurer","nldas","aorc")):
    frames = {}
    for f in forcings:
        d = load(f)
        d = d.assign(**{f"mid_{f}": (d.ylo + d.yhi)/2,
                        f"med_{f}": d.ymed, f"mean_{f}": d.ymean})
        frames[f] = d[["station_id","t0","truth",f"mid_{f}",f"med_{f}",f"mean_{f}"]]
    m = None
    for f, d in frames.items():
        m = d if m is None else m.merge(d.drop(columns=["truth"]), on=["station_id","t0"])
    tag = f"{len(forcings)}-forcing ({','.join(forcings)})"
    print(f"\n=== {tag} | rows {len(m):,} basins {m.station_id.nunique()} ===")
    for lab, pre in (("(ylo+yhi)/2  [record script]","mid"),
                     ("ymed         [feasibility]","med"),
                     ("ymean","mean")):
        s, n = med_nse(m, [f"{pre}_{f}" for f in forcings])
        print(f"  {lab:<30} {s:.4f}  (basins={n})")
