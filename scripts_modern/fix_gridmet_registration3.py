#!/usr/bin/env python3
"""gridMET registration fix, TAKE 3 -- the version the three-way arbiter supports.

Evidence: benchmarks/m1_leakaudit_threeway.json. gridMET/daymet/nclimgrid were each lag-scanned
per variable against camels_corpus_nldas_m3, the only HOURLY-NATIVE product here (our builder
aggregates its hours into calendar days itself, so its day labelling is derived, not inherited).

  variable                  vs daymet   vs NLDAS(ref)   truth
  precipitation_sum            +1          +1 (67%)     OFFSET  [also: streamflow referee +1]
  temperature_2m_mean           0          +1 (80%)     offset, but DERIVED -- see below
  temperature_2m_min           +1          +1 (97%)     OFFSET
  vapor_pressure               +1           0 (100%)    ALIGNED -- take 2 shifted it wrongly
  temperature_2m_max            0           0 (100%)    aligned
  shortwave_radiation_sum       0           0 (100%)    aligned

Take 1 shifted precipitation alone (-0.011137, confounded: de-synchronised precip from tmin
inside gridMET). Take 2 shifted precip+tmin+vapor_pressure -- wrong on vp, and it missed tmean.

tmean is NOT shifted as a column. build_gridmet_areal.py computes it as (tmmx+tmmn)/2, so its
apparent +1 is INHERITED from tmin: tmax sits at k=0 and tmin at k=+1, so the average leans +1.
Shifting the tmean column would drag its tmax half to k=-1. The physically correct repair is to
shift tmin and RECOMPUTE tmean from the corrected halves, which lands tmean at k=0 for free.
This script asserts that identity holds in the source before relying on it.
"""
import glob, json, os
import numpy as np, pandas as pd

SRC = "data/modern_corpus/v3/camels_corpus_gridmet_m3"
DST = "data/modern_corpus/v3/camels_corpus_gridmetfix3_m3"
SHIFT = ["precipitation_sum", "temperature_2m_min"]
os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, "*.csv.gz")))
assert files, SRC

# Verify the derivation we are about to rely on, before relying on it.
probe = pd.read_csv(files[0], dtype={"date": str})
resid = (probe["temperature_2m_mean"]
         - 0.5 * (probe["temperature_2m_max"] + probe["temperature_2m_min"])).abs()
mx = float(np.nanmax(resid))
print(f"tmean == (tmax+tmin)/2 ?  max|residual| = {mx:.6g}")
# Tolerance must match the STORED precision, not float precision. tmean is written at 3
# decimals, so an exact (tmax+tmin)/2 identity shows up as max|resid| = 5e-4, not 1e-6. A 1e-6
# threshold took the wrong branch on the first run: it declared tmean independent and shifted the
# COLUMN, which yields (Tmax(t-1)+Tmin(t))/2 -- half a day stale -- instead of recomputing
# (Tmax(t)+Tmin(t))/2. The lag scan CANNOT catch that (the stale mean is a half-day smooth of the
# right one and still peaks at k=0), so the tolerance is the only thing standing between us and a
# silently wrong build.
DERIVED = mx < 2e-3
assert DERIVED, "tmean is not (tmax+tmin)/2 to stored precision: max resid %.6g" % mx
if DERIVED:
    print("  ✅ derived -> shift tmin, then RECOMPUTE tmean")
else:
    print(f"  ⚠️ NOT derived (max resid {mx:.4g}) -> tmean is independent; shifting it directly")
    SHIFT.append("temperature_2m_mean")

print(f"\nshifting {SHIFT} +1 row in {len(files)} basins -> {DST}")
for f in files:
    d = pd.read_csv(f, dtype={"date": str})
    for c in SHIFT:
        assert c in d.columns, (f, c)
        d[c] = d[c].shift(1)
    if DERIVED:
        d["temperature_2m_mean"] = 0.5 * (d["temperature_2m_max"] + d["temperature_2m_min"])
    d = d.iloc[1:]
    assert d[["precipitation_sum", "temperature_2m_min", "temperature_2m_mean"]].notna().all().all(), f
    d.to_csv(os.path.join(DST, os.path.basename(f)), index=False, compression="gzip")
print(f"wrote {len(files)} basins")

# Self-verify against the SAME hourly-native reference used to diagnose it.
VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "shortwave_radiation_sum", "vapor_pressure"]
ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:60]
print("\nVERIFY vs the NLDAS reference (every variable must now read k=0):")
ok = True
for v in VARS:
    best = []
    for g in ids:
        fa = f"{DST}/{g}.csv.gz"
        fb = f"data/modern_corpus/v3/camels_corpus_nldas_m3/{g}.csv.gz"
        if not (os.path.exists(fa) and os.path.exists(fb)):
            continue
        A = pd.read_csv(fa, dtype={"date": str}).set_index("date")
        B = pd.read_csv(fb, dtype={"date": str}).set_index("date")
        idx = [d for d in A.index.intersection(B.index) if "2019-01-01" <= d <= "2025-12-31"]
        if len(idx) < 400: continue
        a, b = A.loc[idx, v].to_numpy(float), B.loc[idx, v].to_numpy(float)
        r = {}
        for k in (-1, 0, 1):
            bb = np.roll(b, -k); m = np.isfinite(a) & np.isfinite(bb)
            if m.sum() < 300: r[k] = np.nan; continue
            r[k] = float(np.corrcoef(a[m], bb[m])[0, 1])
        fin = {k: x for k, x in r.items() if np.isfinite(x)}
        if fin: best.append(max(fin, key=fin.get))
    mode = int(pd.Series(best).mode().iloc[0]); frac = best.count(mode) / len(best)
    print(f"  {v:<26} k={mode:>2} ({frac:.0%})" + ("" if mode == 0 else "   <- STILL OFFSET"))
    ok = ok and mode == 0
print("✅ all six variables at k=0" if ok else "⚠️ residual offset remains")

dst_c = DST + "_cohort"
os.system(f"rm -rf {dst_c}"); os.makedirs(dst_c, exist_ok=True)
n = 0
for g in json.load(open("data/m1_cohort.json"))["cohort"]:
    s = os.path.abspath(os.path.join(DST, f"{g}.csv.gz"))
    if os.path.exists(s):
        os.symlink(s, os.path.join(dst_c, f"{g}.csv.gz")); n += 1
print(f"cohort tree: {n} symlinks"); assert n == 490, n
json.dump({"shifted": SHIFT, "tmean_recomputed": DERIVED, "all_k0": ok,
           "arbiter": "benchmarks/m1_leakaudit_threeway.json"},
          open("benchmarks/m1_gridmetfix3_build.json", "w"), indent=2)
