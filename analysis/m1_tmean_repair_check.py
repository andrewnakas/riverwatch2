#!/usr/bin/env python3
"""Discriminate the two candidate tmean repairs -- the lag scan cannot.

tmax[t]=Tmax(t), tmin[t]=Tmin(t+1). Two candidate repairs:
  A (recompute, what fix3 now does): shift tmin, then tmean = (tmax[t]+tmin_shifted[t])/2
                                     = (Tmax(t)+Tmin(t))/2   <- the true calendar-day mean
  B (shift the column, what the first fix3 run did): tmean[t-1] = (Tmax(t-1)+Tmin(t))/2
                                     <- half a day stale
Both peak at k=0 against NLDAS, so m1_leak_threeway-style lag scanning passes BOTH. The
discriminating test is the RESIDUAL against the hourly-native NLDAS daily mean on the same rows:
the true mean must sit closer than the stale one. If it does not, the recompute premise is wrong.
"""
import json, os
import numpy as np, pandas as pd
V3 = "data/modern_corpus/v3"
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:60]
rows = []
for g in ids:
    fo, ff, fn = (f"{V3}/camels_corpus_gridmet_m3/{g}.csv.gz",
                  f"{V3}/camels_corpus_gridmetfix3_m3/{g}.csv.gz",
                  f"{V3}/camels_corpus_nldas_m3/{g}.csv.gz")
    if not all(os.path.exists(x) for x in (fo, ff, fn)): continue
    O = pd.read_csv(fo, dtype={"date": str}).set_index("date")
    F = pd.read_csv(ff, dtype={"date": str}).set_index("date")
    N = pd.read_csv(fn, dtype={"date": str}).set_index("date")
    idx = [d for d in F.index.intersection(N.index).intersection(O.index)
           if "2019-01-01" <= d <= "2025-12-31"]
    if len(idx) < 400: continue
    ref = N.loc[idx, "temperature_2m_mean"].to_numpy(float)
    A = F.loc[idx, "temperature_2m_mean"].to_numpy(float)          # recompute (built)
    B = O["temperature_2m_mean"].shift(1).loc[idx].to_numpy(float)  # stale column shift
    U = O.loc[idx, "temperature_2m_mean"].to_numpy(float)           # unrepaired
    m = np.isfinite(ref) & np.isfinite(A) & np.isfinite(B) & np.isfinite(U)
    if m.sum() < 300: continue
    rows.append([np.sqrt(np.mean((A[m]-ref[m])**2)), np.sqrt(np.mean((B[m]-ref[m])**2)),
                 np.sqrt(np.mean((U[m]-ref[m])**2))])
R = np.array(rows)
print(f"RMSE of each candidate tmean against the NLDAS daily mean  (n={len(R)} basins)\n")
print(f"  {'candidate':<34}{'median RMSE':>13}   wins")
names = ["A recompute (tmax + shifted tmin)/2", "B shift the tmean column (stale)",
         "U unrepaired gridmet tmean"]
med = np.median(R, axis=0)
best = int(np.argmin(med))
for i, n in enumerate(names):
    w = float(np.mean(np.argmin(R, axis=1) == i))
    print(f"  {n:<34}{med[i]:13.4f}   {w:5.1%}")
print()
if best == 0:
    print("✅ A wins: the recompute is the correct repair, and B -- which the lag scan ALSO")
    print("   passed -- is measurably worse. The tolerance guard was the only thing catching it.")
else:
    print(f"⚠️ {names[best]} wins -- the recompute premise is WRONG; do not ship fix3 as built.")
json.dump({"median_rmse": {n: float(med[i]) for i, n in enumerate(names)},
           "winner": names[best]}, open("benchmarks/m1_tmean_repair_check.json", "w"), indent=2)
