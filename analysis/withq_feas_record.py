#!/usr/bin/env python3
"""with-q feasibility computed on the RECORD config and RECORD readout (ylo+yhi)/2.
Mirrors target_feasibility.py's ceiling math exactly; only the ensemble differs."""
import json, os, sys
import numpy as np, pandas as pd
os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from target_feasibility import sigma_series, nse

ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json")).get("531", []))
P = "data/mblstm/gpu_dumps_s14/camels531_{}_withq5_full531.csv.gz"
FORCINGS = ("daymet", "maurer", "nldas", "aorc")

frames = {}
for f in FORCINGS:
    d = pd.read_csv(P.format(f), usecols=["station_id","t0","h","truth","ylo","yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(ids)]
    d[f] = (d.ylo + d.yhi) / 2
    frames[f] = d[["station_id","t0","truth",f]]
m = None
for f, d in frames.items():
    m = d if m is None else m.merge(d[["station_id","t0",f]], on=["station_id","t0"])
m["ens"] = m[list(FORCINGS)].to_numpy(float).mean(1)
print(f"RECORD config: 4 forcings x 5 seeds, readout (ylo+yhi)/2")
print(f"rows {len(m):,}  basins {m.station_id.nunique()}\n")

flash = m.groupby("station_id")["truth"].apply(
    lambda s: np.percentile(s[s>0], 99)/np.median(s[s>0]) if (s>0).sum() > 20 else np.nan)
thr = flash.median()

SCEN = {"central": (0.30, 0.18, 0.35), "optimistic": (0.25, 0.13, 0.13)}
for name, (s_low, s_high, s_flashy) in SCEN.items():
    rows = []
    for sid, g in m.groupby("station_id"):
        q = g["truth"].to_numpy(float)
        ok = np.isfinite(q)
        if ok.sum() < 50: continue
        qq = q[ok]
        V = ((qq - qq.mean())**2).mean()
        if V <= 0: continue
        M = (qq**2).mean()
        s_hi = s_flashy if flash.get(sid, 0) > thr else s_high
        s2 = np.nanmean(sigma_series(q, s_low, s_hi)**2)
        ceil = 1 - (M/V)*(np.exp(s2) - 1)
        n = nse(g["truth"], g["ens"])
        if np.isfinite(n):
            rows.append((sid, n, ceil, flash.get(sid, np.nan)))
    df = pd.DataFrame(rows, columns=["sid","nse","ceiling","flash"]).dropna()
    df["cohort"] = np.where(df.flash > thr, "flashy", "steady")
    cur = df.nse.median()
    df["capped"] = np.maximum(df.nse, np.minimum(df.ceiling, 1.0))
    best = df.capped.median()
    at = (df.ceiling <= df.nse).sum()
    print(f"--- {name.upper()} scenario (sigma {s_low}->{s_high}, flashy {s_flashy}) ---")
    print(f"  current (RECORD)  {cur:.4f}")
    print(f"  all at ceiling    {best:.4f}   headroom {best-cur:+.4f}")
    print(f"  gap to 0.95       {0.95-cur:+.4f}")
    if best >= 0.95:
        print(f"  => 0.95 REACHABLE, needs {100*(0.95-cur)/(best-cur):.0f}% of headroom")
    else:
        print(f"  => 0.95 NOT REACHABLE (ceiling {best:.4f} < 0.95)")
    print(f"  saturated basins  {at}/{len(df)} ({100*at/len(df):.0f}%)")
    tot = (df.capped - df.nse).sum()
    for c in ("steady","flashy"):
        s = df[df.cohort==c]
        print(f"    {c:<8} n={len(s)}  medNSE {s.nse.median():.3f}  "
              f"share of gain {100*(s.capped-s.nse).sum()/tot:.0f}%")
    print()
