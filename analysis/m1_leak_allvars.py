#!/usr/bin/env python3
"""Is gridMET's WHOLE daily record one day ahead, or only precipitation?

If every gridMET variable aligns with daymet at k=+1, the offset is a property of the
product's daily aggregation window, and a precip-ONLY shift de-synchronises precip from
temperature inside gridMET -- which would degrade the member for reasons unrelated to
leakage, confounding the correction test.
"""
import json, os
import numpy as np, pandas as pd

V3 = "data/modern_corpus/v3"
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:60]
VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "shortwave_radiation_sum", "vapor_pressure"]
LAGS = range(-2, 3)
print("\ncorr( gridmet VAR[t] , daymet VAR[t+k] ), median over basins, 2019-2025\n")
hdr = f"{'variable':<26}" + "".join(f"{k:>8}" for k in LAGS) + "   best k"
print(hdr); print("-" * len(hdr))
out = {}
for v in VARS:
    rows = []
    for g in ids:
        fa = f"{V3}/camels_corpus_gridmet_m3/{g}.csv.gz"
        fb = f"{V3}/camels_corpus_daymet_m3/{g}.csv.gz"
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A = pd.read_csv(fa, dtype={"date": str}).set_index("date")
        B = pd.read_csv(fb, dtype={"date": str}).set_index("date")
        if v not in A.columns or v not in B.columns:
            continue
        idx = [d for d in A.index.intersection(B.index) if "2019-01-01" <= d <= "2025-12-31"]
        if len(idx) < 400:
            continue
        a = A.loc[idx, v].to_numpy(float); b = B.loc[idx, v].to_numpy(float)
        r = {}
        for k in LAGS:
            bb = np.roll(b, -k); m = np.isfinite(a) & np.isfinite(bb)
            r[k] = float(np.corrcoef(a[m], bb[m])[0, 1]) if m.sum() > 400 else np.nan
        rows.append(r)
    if not rows:
        print(f"{v:<26} no data"); continue
    med = {k: float(np.nanmedian([x[k] for x in rows])) for k in LAGS}
    best = max(LAGS, key=lambda k: med[k])
    out[v] = {"median_r_by_lag": med, "best_k": best, "n": len(rows)}
    print(f"{v:<26}" + "".join(f"{med[k]:>8.3f}" for k in LAGS) + f"   {best:>5}")
ks = {v: out[v]["best_k"] for v in out}
print()
if len(set(ks.values())) == 1 and list(ks.values())[0] == 1:
    print("⛔ THE WHOLE gridMET RECORD is one day ahead — every variable peaks at k=+1.")
    print("   ⇒ A precip-ONLY shift de-synchronises precip from temperature INSIDE gridMET.")
    print("   ⇒ The gridmetm9fix test is CONFOUNDED and cannot answer the leak question.")
    print("   ⇒ The correct fix shifts EVERY gridMET weather column by +1.")
elif ks.get("precipitation_sum") == 1 and all(v == 0 for k, v in ks.items() if k != "precipitation_sum"):
    print("✅ ONLY precipitation is offset; the other variables align at k=0.")
    print("   ⇒ The precip-only shift was the right correction, and the test stands.")
else:
    print(f"⚠️ MIXED offsets: {ks}")
    print("   ⇒ Interpret the correction test only after reconciling these.")
json.dump(out, open("benchmarks/m1_leakaudit_gridmet_allvars.json", "w"), indent=1)
print("\n-> benchmarks/m1_leakaudit_gridmet_allvars.json")
