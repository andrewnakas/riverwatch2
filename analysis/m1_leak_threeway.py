#!/usr/bin/env python3
"""Three-way per-variable alignment against the ONLY unambiguous reference.

m1_leak_allvars.py measured every gridMET variable against DAYMET and found MIXED offsets.
That test cannot settle anything, because Daymet is not a reference: its precipitation was
itself shifted +1 by our own build (daymet-v4-vs-camels-ONE-DAY-PRECIP-OFFSET), and its
tmin/tmax conventions are inherited from GHCN station reporting, not derived.

NLDAS-2 is HOURLY-NATIVE. Our builder aggregates its hours into calendar days itself
(build_nldas2_areal.py), so its daily convention is DERIVED, not inherited -- it is the only
product here whose calendar-day labelling we can vouch for. So: measure gridMET AND daymet AND
nclimgrid each against NLDAS-2, per variable. A variable offset in gridMET-vs-nldas is really
offset; one that is offset vs daymet but NOT vs nldas means DAYMET is the odd one and gridMET
was fine.
"""
import json, os
import numpy as np, pandas as pd

V3 = "data/modern_corpus/v3"
REF = "camels_corpus_nldas_m3"
TEST = ["camels_corpus_gridmet_m3", "camels_corpus_daymet_m3", "camels_corpus_nclimgrid_m3"]
VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "shortwave_radiation_sum", "vapor_pressure"]
KS = (-2, -1, 0, 1, 2)
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:60]

def load(corp, g):
    f = f"{V3}/{corp}/{g}.csv.gz"
    if not os.path.exists(f):
        f = f"{V3}/{corp}_cohort/{g}.csv.gz"
    if not os.path.exists(f):
        return None
    return pd.read_csv(f, dtype={"date": str}).set_index("date")

out = {}
for corp in TEST:
    print(f"\n=== {corp}  vs  {REF} (hourly-native reference) ===")
    print(f"{'variable':<26}" + "".join(f"{k:>8}" for k in KS) + "   best k   n")
    out[corp] = {}
    for v in VARS:
        per_basin_best, acc = [], {k: [] for k in KS}
        for g in ids:
            A, B = load(corp, g), load(REF, g)
            if A is None or B is None or v not in A.columns or v not in B.columns:
                continue
            idx = [d for d in A.index.intersection(B.index) if "2019-01-01" <= d <= "2025-12-31"]
            if len(idx) < 400:
                continue
            a = A.loc[idx, v].to_numpy(float); b = B.loc[idx, v].to_numpy(float)
            r = {}
            for k in KS:
                bb = np.roll(b, -k)
                m = np.isfinite(a) & np.isfinite(bb)
                if m.sum() < 300 or np.std(a[m]) == 0 or np.std(bb[m]) == 0:
                    r[k] = np.nan; continue
                r[k] = float(np.corrcoef(a[m], bb[m])[0, 1])
                acc[k].append(r[k])
            fin = {k: x for k, x in r.items() if np.isfinite(x)}
            if fin:
                per_basin_best.append(max(fin, key=fin.get))
        if not per_basin_best:
            print(f"  {v:<24} (no overlap)"); continue
        med = {k: (float(np.median(acc[k])) if acc[k] else np.nan) for k in KS}
        mode = int(pd.Series(per_basin_best).mode().iloc[0])
        frac = per_basin_best.count(mode) / len(per_basin_best)
        row = "".join(f"{med[k]:8.3f}" for k in KS)
        flag = "" if mode == 0 else "  <- OFFSET"
        print(f"  {v:<24}{row}{mode:>8}  {len(per_basin_best):>3} ({frac:.0%}){flag}")
        out[corp][v] = {"best_k": mode, "agree_frac": frac, "median_r": med}

print("\n" + "=" * 78)
print("RECONCILIATION -- best k against the hourly-native reference:")
print(f"{'variable':<26}{'gridmet':>10}{'daymet':>10}{'nclimgrid':>12}   verdict")
verdict = {}
for v in VARS:
    g = out["camels_corpus_gridmet_m3"].get(v, {}).get("best_k")
    d = out["camels_corpus_daymet_m3"].get(v, {}).get("best_k")
    n = out["camels_corpus_nclimgrid_m3"].get(v, {}).get("best_k")
    if g == 0:
        t = "gridMET ALIGNED" + ("" if d == 0 else f" (daymet is the offset one: k={d})")
    elif g == d == n:
        t = f"ALL THREE at k={g} -- our NLDAS aggregation is the offset one"
    else:
        t = f"gridMET GENUINELY OFFSET by {g}"
    verdict[v] = {"gridmet": g, "daymet": d, "nclimgrid": n, "text": t}
    print(f"  {v:<24}{str(g):>10}{str(d):>10}{str(n):>12}   {t}")
really = [v for v, x in verdict.items() if x["gridmet"] not in (0, None)
          and not (x["gridmet"] == x["daymet"] == x["nclimgrid"])]
print(f"\n⇒ gridMET columns genuinely needing a +shift: {really or 'NONE'}")
json.dump({"reference": REF, "per_product": out, "verdict": verdict, "shift_needed": really},
          open("benchmarks/m1_leakaudit_threeway.json", "w"), indent=2)
print("wrote benchmarks/m1_leakaudit_threeway.json")
