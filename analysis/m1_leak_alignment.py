#!/usr/bin/env python3
"""Decisive direction test: align each product's precip against DAYMET's, same basin/dates.

If gridMET[t] ~= daymet[t+1], gridMET's row t carries rain that falls one day LATER, and a
decoder reading row T (the target day) sees future rain. If gridMET[t] ~= daymet[t-1] it is
merely stale. This compares products to each other, so it cannot be confused by basin
routing time -- routing is identical for both series.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:80]
V3 = "data/modern_corpus/v3"
REF = "daymet"
prods = ["gridmet", "nclimgrid", "nldas"]
LAGS = range(-3, 4)
print(f"\ncorr( product precip[t] , {REF} precip[t+k] ), median over basins, 2019-2025\n")
hdr = "product      n   " + "".join(f"{k:>8}" for k in LAGS) + "   best k"
print(hdr); print("-" * len(hdr))
res = {}
for p in prods:
    rows = []
    for g in ids:
        fa = f"{V3}/camels_corpus_{p}_m3/{g}.csv.gz"
        fb = f"{V3}/camels_corpus_{REF}_m3/{g}.csv.gz"
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A = pd.read_csv(fa, dtype={"date": str}).set_index("date")["precipitation_sum"]
        B = pd.read_csv(fb, dtype={"date": str}).set_index("date")["precipitation_sum"]
        idx = A.index.intersection(B.index)
        idx = [d for d in idx if "2019-01-01" <= d <= "2025-12-31"]
        if len(idx) < 400:
            continue
        a = A.loc[idx].to_numpy(float); b = B.loc[idx].to_numpy(float)
        row = {}
        for k in LAGS:
            bb = np.roll(b, -k)
            m = np.isfinite(a) & np.isfinite(bb)
            row[k] = float(np.corrcoef(a[m], bb[m])[0, 1]) if m.sum() > 400 else np.nan
        rows.append(row)
    if not rows:
        print(f"{p:<12} no data"); continue
    med = {k: float(np.nanmedian([r[k] for r in rows])) for k in LAGS}
    best = max(LAGS, key=lambda k: med[k])
    res[p] = {"median_r_by_lag": med, "best_k": best, "n_basins": len(rows)}
    print(f"{p:<12}{len(rows):>3} " + "".join(f"{med[k]:>8.3f}" for k in LAGS) + f"   {best:>5}")
print()
for p, r in res.items():
    k = r["best_k"]
    if k == 0:
        print(f"  {p}: aligned with {REF} (k=0) -- same convention.")
    elif k > 0:
        print(f"  ⛔ {p}: {p}[t] matches {REF}[t+{k}] -- {p} row t carries rain from {k} day(s) LATER.")
        print(f"     A decoder reading row T (the target day) therefore sees FUTURE rain. LEAK.")
    else:
        print(f"  ⚠️ {p}: {p}[t] matches {REF}[t{k}] -- {p} is STALE by {-k} day(s). Conservative, not a leak.")
json.dump(res, open("benchmarks/m1_leakaudit_alignment.json", "w"), indent=1)
print("\n-> benchmarks/m1_leakaudit_alignment.json")
