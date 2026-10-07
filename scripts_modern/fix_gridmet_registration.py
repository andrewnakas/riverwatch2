#!/usr/bin/env python3
"""Build a REGISTRATION-CORRECTED gridMET corpus and the member to test it.

FINDING (benchmarks/m1_leakaudit_alignment.json): corr(gridmet[t], daymet[t+1]) = 0.957
against 0.103 at k=0, so gridMET row t carries rain(t+1) while Daymet/nClimGrid/NLDAS row t
carries rain(t). Daymet got a +1-day precip shift at build time; gridMET did not.

CORRECTION: row t must hold rain(t) = the value currently at row t-1 => shift the precip
column forward by one row, exactly the fix applied to Daymet.

DIRECTION OF THE EFFECT (to be settled by the retrain, not by argument): the model predicting
Q_T reads corpus row T. Being handed rain(T+1) instead of rain(T) REMOVES the causal driver
and substitutes a non-causal one, which should DEPRESS skill rather than inflate it. If the
corrected member scores HIGHER, the misregistration was a cost and the record is understated,
not false. If it scores LOWER, the uncorrected version was exploiting future information.
"""
import glob, os, shutil, sys
import pandas as pd

SRC = "data/modern_corpus/v3/camels_corpus_gridmet_m3"
DST = "data/modern_corpus/v3/camels_corpus_gridmetfix_m3"
COL = "precipitation_sum"
os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, "*.csv.gz")))
print(f"shifting {COL} +1 row in {len(files)} basins -> {DST}")
n = 0
for f in files:
    d = pd.read_csv(f, dtype={"date": str})
    assert COL in d.columns, f
    d[COL] = d[COL].shift(1)          # row t takes the value from row t-1 = rain(t)
    d = d.iloc[1:]                     # drop the now-NaN first row
    d.to_csv(os.path.join(DST, os.path.basename(f)), index=False, compression="gzip")
    n += 1
print(f"wrote {n} basins")
# verify the correction landed: the fixed product must now align with daymet at k=0
import numpy as np, json
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:40]
best = []
for g in ids:
    fa, fb = f"{DST}/{g}.csv.gz", f"data/modern_corpus/v3/camels_corpus_daymet_m3/{g}.csv.gz"
    if not (os.path.exists(fa) and os.path.exists(fb)):
        continue
    A = pd.read_csv(fa, dtype={"date": str}).set_index("date")[COL]
    B = pd.read_csv(fb, dtype={"date": str}).set_index("date")[COL]
    idx = [d for d in A.index.intersection(B.index) if "2019-01-01" <= d <= "2025-12-31"]
    if len(idx) < 400:
        continue
    a, b = A.loc[idx].to_numpy(float), B.loc[idx].to_numpy(float)
    r = {}
    for k in (-1, 0, 1):
        bb = np.roll(b, -k); m = np.isfinite(a) & np.isfinite(bb)
        r[k] = float(np.corrcoef(a[m], bb[m])[0, 1])
    best.append(max(r, key=r.get))
print(f"VERIFY: modal best lag vs daymet after the fix = {pd.Series(best).mode().iloc[0]}  "
      f"(0 = corrected)   n={len(best)}")
assert pd.Series(best).mode().iloc[0] == 0, "correction did NOT land at k=0"
print("✅ correction verified at k=0")
