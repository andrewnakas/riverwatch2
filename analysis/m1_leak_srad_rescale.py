#!/usr/bin/env python3
"""Is gridMET srad a RESCALED copy of NLDAS srad?

r=0.999 between the two, but 0% of values exactly equal. For two independently derived solar
radiation products that is implausible -- daymet-vs-nldas reads 0.861 on the same basins/dates.
A near-perfect correlation with no exact matches is the signature of y = a*x + b, i.e. a unit
conversion of the SAME underlying field. Fit it and look.
"""
import json, os
import numpy as np, pandas as pd
V3 = "data/modern_corpus/v3"
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:25]
PAIRS = [("gridmet", "nldas"), ("daymet", "nldas"), ("gridmet", "daymet")]
C = {"daymet": "camels_corpus_daymet_m3", "nldas": "camels_corpus_nldas_m3",
     "gridmet": "camels_corpus_gridmet_m3"}
for v in ["shortwave_radiation_sum", "vapor_pressure"]:
    print(f"\n=== {v} : least-squares  A = a*B + b ===")
    print(f"{'pair':<22}{'slope a':>10}{'intercept b':>13}{'r':>8}{'resid sd':>10}{'rel resid':>11}")
    for an, bn in PAIRS:
        A_, B_ = [], []
        for g in ids:
            fa, fb = f"{V3}/{C[an]}/{g}.csv.gz", f"{V3}/{C[bn]}/{g}.csv.gz"
            if not (os.path.exists(fa) and os.path.exists(fb)): continue
            A = pd.read_csv(fa, dtype={"date": str}).set_index("date")
            B = pd.read_csv(fb, dtype={"date": str}).set_index("date")
            idx = A.index.intersection(B.index)
            if len(idx) < 400 or v not in A.columns or v not in B.columns: continue
            a, b = A.loc[idx, v].to_numpy(float), B.loc[idx, v].to_numpy(float)
            m = np.isfinite(a) & np.isfinite(b)
            A_.append(a[m]); B_.append(b[m])
        if not A_: continue
        a = np.concatenate(A_); b = np.concatenate(B_)
        sl, ic = np.polyfit(b, a, 1)
        r = float(np.corrcoef(a, b)[0, 1])
        res = a - (sl * b + ic)
        print(f"  {an+'/'+bn:<20}{sl:10.5f}{ic:13.5f}{r:8.4f}{np.std(res):10.4f}"
              f"{np.std(res)/max(np.std(a),1e-9):11.4f}")
    print(f"  (ranges)  " + "  ".join(
        f"{n}: {np.nanmin(pd.read_csv(f'{V3}/{C[n]}/{ids[0]}.csv.gz')[v]):.2f}"
        f"..{np.nanmax(pd.read_csv(f'{V3}/{C[n]}/{ids[0]}.csv.gz')[v]):.2f}" for n in C))
