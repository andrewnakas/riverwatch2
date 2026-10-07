#!/usr/bin/env python3
"""gridMET registration fix, TAKE 2 -- shift every variable empirically at k=+1.

Take 1 shifted precipitation ONLY and scored -0.011137. That test was CONFOUNDED:
benchmarks/m1_leakaudit_gridmet_allvars.json shows precipitation (+1, r 0.957 vs 0.090),
temperature_2m_min (+1, 0.992 vs 0.934) and vapor_pressure (+1, 0.968 vs 0.956) are offset
while temperature_2m_max (0.995) and shortwave_radiation_sum (0.860) sit at k=0 -- the
documented PRISM/GHCN pattern (12Z-12Z precip accumulation; overnight minima attributed to
the following morning; daytime max and radiation on the calendar day). Shifting precip alone
de-synchronised it from tmin/vp INSIDE gridMET, so the loss could be that penalty rather
than a removed leak.

This shifts all three +1 variables together, leaving tmax and srad, so every gridMET column
lands on the calendar-day convention the other three products use.
"""
import glob, json, os
import numpy as np, pandas as pd

SRC = "data/modern_corpus/v3/camels_corpus_gridmet_m3"
DST = "data/modern_corpus/v3/camels_corpus_gridmetfix2_m3"
SHIFT = ["precipitation_sum", "temperature_2m_min", "vapor_pressure"]
os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(os.path.join(SRC, "*.csv.gz")))
print(f"shifting {SHIFT} +1 row in {len(files)} basins -> {DST}")
for f in files:
    d = pd.read_csv(f, dtype={"date": str})
    for c in SHIFT:
        assert c in d.columns, (f, c)
        d[c] = d[c].shift(1)
    d = d.iloc[1:]
    d.to_csv(os.path.join(DST, os.path.basename(f)), index=False, compression="gzip")
print(f"wrote {len(files)} basins")

ids = [str(x).zfill(8) for x in json.load(open("data/m1_cohort.json"))["cohort"]][:40]
VARS = ["precipitation_sum", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "shortwave_radiation_sum", "vapor_pressure"]
print("\nVERIFY vs daymet after the fix (best k should be 0 for every variable):")
ok = True
for v in VARS:
    best = []
    for g in ids:
        fa, fb = f"{DST}/{g}.csv.gz", f"data/modern_corpus/v3/camels_corpus_daymet_m3/{g}.csv.gz"
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
            r[k] = float(np.corrcoef(a[m], bb[m])[0, 1])
        best.append(max(r, key=r.get))
    mode = int(pd.Series(best).mode().iloc[0])
    print(f"  {v:<26} best k = {mode}" + ("" if mode == 0 else "   <- still offset"))
    ok = ok and mode == 0
print("✅ all variables at k=0" if ok else "⚠️ some variables still offset")

dst_c = DST + "_cohort"
os.system(f"rm -rf {dst_c}"); os.makedirs(dst_c, exist_ok=True)
n = 0
for g in json.load(open("data/m1_cohort.json"))["cohort"]:
    s = os.path.abspath(os.path.join(DST, f"{g}.csv.gz"))
    if os.path.exists(s):
        os.symlink(s, os.path.join(dst_c, f"{g}.csv.gz")); n += 1
print(f"cohort tree: {n} symlinks"); assert n == 490, n

p = "gpu1080/queue_m1d_withq.sh"
s = open(p).read()
old = "  gridmetm9fix)"
new = """  gridmetm9fix2)             CORP="$V3/camels_corpus_gridmetfix2_m3_cohort"; ENC="camels1f"; TRS=2005-10-01 ;;
  gridmetm9fix)"""
assert old in s
open(p, "w").write(s.replace(old, new, 1))
print("added gridmetm9fix2")
