#!/usr/bin/env python3
"""Are srad / vapor_pressure SHARED columns across the four 'independent' products?

The three-way scan returned r=0.999 (srad) and 0.990 (vp) between gridmet and nldas, and
IDENTICAL medians for nclimgrid. Near-identity between products that are supposed to be
independently derived would mean the four-product ensemble has less forcing diversity than its
design assumes -- and forcing diversity is the only diversity this campaign is measurably paid
for. Exact equality is the test; correlation is not.
"""
import json, os
import numpy as np, pandas as pd

V3 = "data/modern_corpus/v3"
P = {"daymet": "camels_corpus_daymet_m3", "nldas": "camels_corpus_nldas_m3",
     "gridmet": "camels_corpus_gridmet_m3", "nclimgrid": "camels_corpus_nclimgrid_m3"}
VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "shortwave_radiation_sum", "vapor_pressure"]
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:30]

D = {}
for name, c in P.items():
    rows = {}
    for g in ids:
        f = f"{V3}/{c}/{g}.csv.gz"
        if os.path.exists(f):
            rows[g] = pd.read_csv(f, dtype={"date": str}).set_index("date")
    D[name] = rows

names = list(P)
print("EXACT-EQUALITY matrix (fraction of overlapping values identical to 1e-9):\n")
shared = []
for v in VARS:
    print(f"  {v}")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_n, b_n = names[i], names[j]
            fr = []
            for g in ids:
                A, B = D[a_n].get(g), D[b_n].get(g)
                if A is None or B is None or v not in A.columns or v not in B.columns:
                    continue
                idx = A.index.intersection(B.index)
                if len(idx) < 400: continue
                a = A.loc[idx, v].to_numpy(float); b = B.loc[idx, v].to_numpy(float)
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < 300: continue
                fr.append(float(np.mean(np.abs(a[m] - b[m]) < 1e-9)))
            if not fr: continue
            f = float(np.median(fr))
            tag = ""
            if f > 0.99:
                tag = "   <<< SHARED COLUMN"; shared.append((v, a_n, b_n, f))
            elif f > 0.5:
                tag = "   <<< partially shared"; shared.append((v, a_n, b_n, f))
            if f > 0.01 or tag:
                print(f"      {a_n:<10} vs {b_n:<10} identical {f:6.1%}{tag}")
    print()

print("=" * 72)
if shared:
    print("⚠️ SHARED COLUMNS FOUND -- these products are not independent for these variables:")
    for v, a, b, f in shared:
        print(f"   {v:<26} {a} == {b}  ({f:.1%} of values)")
else:
    print("✅ no shared columns: every product's every variable is independently derived.")
json.dump([{"var": v, "a": a, "b": b, "frac_identical": f} for v, a, b, f in shared],
          open("benchmarks/m1_leakaudit_shared_columns.json", "w"), indent=2)
