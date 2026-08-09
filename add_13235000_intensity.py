#!/usr/bin/env python3
"""Add the reconstructed basin to the INTENSITY corpus as its 531st benchmark row.

build_aorc_intensity.py reads the HydroShare tarballs, which omit 13235000. But
we already extracted that basin's raw HOURLY series from the official NOAA zarr
(data/aorc_extra/13235000_WY*.csv, 31 files), so the same within-day features
can be computed from it directly — no new download, no new extraction.

Reuses daily_from_hourly() from the builder verbatim, so the new row is computed
by exactly the same code path as the other 530 and nothing can drift between
them.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from build_aorc_intensity import COLS, daily_from_hourly

BASIN = "13235000"
RAW = Path("data/aorc_extra")
CORPUS = Path("data/local_corpora/camels_corpus_aorcx_v2")
QSRC = Path("data/local_corpora/camels_corpus_daymet_v2")
REF = Path("data/local_corpora/camels_corpus_aorc_v2") / f"{BASIN}.csv.gz"

files = sorted(RAW.glob(f"{BASIN}_WY*.csv"))
print(f"hourly water-year files: {len(files)}/31")
if len(files) < 31:
    sys.exit("incomplete: 1980 supplies the 1980-10-01..12-31 head, so 31 are needed")

parts = []
for f in files:
    d = pd.read_csv(f)
    d["time"] = pd.to_datetime(d["time"])
    parts.append(d)
h = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
print(f"hourly rows: {len(h)}  {h.time.min()} .. {h.time.max()}")

d = daily_from_hourly(h)
q = pd.read_csv(QSRC / f"{BASIN}.csv.gz", usecols=["date", "q_cfs"])
d = d.merge(q, on="date", how="left")

# trim to exactly the window the other basins cover
ref = pd.read_csv(CORPUS / f"{sorted(p.name for p in CORPUS.glob('*.csv.gz'))[0]}",
                  usecols=["date"])
lo, hi = ref.date.min(), ref.date.max()
d = d[(d.date >= lo) & (d.date <= hi)].reset_index(drop=True)

print(f"\ndaily rows: {len(d)}  {d.date.min()} .. {d.date.max()}  (others: {lo} .. {hi})")
w = d[d.precipitation_sum > 1]
print(f"  wet days: {len(w)}")
print(f"  p_max_1h   mean {w.p_max_1h.mean():.3f}  max {w.p_max_1h.max():.2f}")
print(f"  p_hours    mean {w.p_hours.mean():.1f}")
print(f"  p_cv       mean {w.p_cv.mean():.3f}")
print(f"  p_centroid mean {w.p_centroid.mean():.1f}")

# same guards as the main builder, plus consistency with the daily-only corpus
if len(d) != len(ref):
    sys.exit(f"FATAL: {len(d)} rows vs {len(ref)} in the reference basin")
if w.p_max_1h.std() < 1e-6:
    sys.exit("FATAL: intensity features constant")
if (w.p_max_1h > w.precipitation_sum + 1e-6).any():
    sys.exit("FATAL: peak hour exceeds daily total")

if REF.exists():
    old = pd.read_csv(REF, usecols=["date", "precipitation_sum"])
    j = d.merge(old, on="date", suffixes=("_new", "_old"))
    diff = (j.precipitation_sum_new - j.precipitation_sum_old).abs().max()
    print(f"\n  max |precip difference| vs the daily-only corpus: {diff:.6f} mm")
    if diff > 0.01:
        sys.exit("FATAL: daily totals disagree with the existing corpus row")
    print("  -> daily totals match, so only the new columns differ")

out = CORPUS / f"{BASIN}.csv.gz"
d[COLS].to_csv(out, index=False, compression="gzip")
print(f"\nwrote {out}")
print(f"corpus now {len(list(CORPUS.glob('*.csv.gz')))} basins")
