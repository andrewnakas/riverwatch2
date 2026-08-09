#!/usr/bin/env python3
"""Are the NO-Q dumps continuous? If so the phase test needs no new GPU time.

The with-q dumps are stride-14, which makes any lag test meaningless: one row is
fourteen days, so shifting by a row collapses NSE from 0.9196 to -0.38. That is
why a stride-1 probe was queued.

But the no-q streams in gpu1080/dumps come from neuralhydrology, which evaluates
the whole test period rather than sampling windows. If those rows are
consecutive days, the phase hypothesis can be tested on the no-q ensemble
immediately -- and the diagnosis should carry, since it is a property of the
LSTM family rather than of the discharge input.
"""
import glob
import os

import pandas as pd

pats = ["gpu1080/dumps/camels531ls_daymet_nhlstm_s111.csv.gz",
        "gpu1080/dumps/camels531ls_nldas_nhlstm_s222.csv.gz",
        "dhbv_dumps/camels531ls_daymet_dhbv_s111.csv.gz"]

for p in pats:
    if not os.path.exists(p):
        print(f"{os.path.basename(p):45s} (absent)")
        continue
    d = pd.read_csv(p, usecols=["station_id", "t0", "h"], nrows=200000)
    d = d[d.h == 1]
    d["t0"] = pd.to_datetime(d.t0)
    d = d.sort_values(["station_id", "t0"])
    g = d.groupby("station_id").t0.diff().dt.days.dropna()
    if len(g) == 0:
        print(f"{os.path.basename(p):45s} (single row per station)")
        continue
    print(f"{os.path.basename(p):45s} median spacing {g.median():.0f} d, "
          f"{(g == 1).mean()*100:5.1f}% consecutive, "
          f"{d.station_id.nunique()} basins in sample")

print()
print("If consecutive is ~100%, the no-q dumps ARE continuous and the phase")
print("test can run on them now, with no GPU cost and no waiting for the probe.")
