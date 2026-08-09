#!/usr/bin/env python3
"""Pre-register a prediction for the no-q AORC gain, from measurable decorrelation.

The projection of ~+0.003 came from scaling the with-q +0.0062 by member
dilution (1-of-8 streams versus 1-of-4). That assumes AORC decorrelates equally
in both settings, which is an assumption rather than a measurement.

A better predictor exists in data already on disk. The with-q dumps show how much
AORC actually decorrelates from daymet/maurer/nldas. If that decorrelation is
large relative to the within-family baseline, a no-q gain should follow. If AORC
correlates with the others about as much as they correlate with each other, the
with-q gain came from something else and the no-q projection is optimistic.

Writing this down BEFORE the no-q members finish training makes the comparison
honest either way -- the prediction cannot be adjusted after seeing the result.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

D = "data/mblstm/gpu_dumps_s14"


def resid(path, subset):
    d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    d = d[d.station_id.isin(subset)]
    d["r"] = (d.ylo + d.yhi) / 2 - d.truth
    return d.set_index(["station_id", "t0"])["r"]


def main():
    ids = set(str(x).zfill(8) for x in json.load(open("data/camels_gauge_ids.json"))["531"])

    files = {}
    for f in ("daymet", "maurer", "nldas"):
        p = f"{D}/camels531_{f}_withq5_full531.csv.gz"
        if os.path.exists(p):
            files[f] = p
    p = f"{D}/camels531_aorc_withq_full531.csv.gz"
    if os.path.exists(p):
        files["aorc"] = p
    if "aorc" not in files or len(files) < 4:
        sys.exit("need all four with-q member dumps")

    res, common = {}, None
    for k, p in files.items():
        r = resid(p, ids)
        res[k] = r
        common = r.index if common is None else common.intersection(r.index)
    R = pd.DataFrame({k: v.reindex(common) for k, v in res.items()}).dropna()
    print(f"with-q residuals: {len(R)} rows, "
          f"{len(set(i[0] for i in common))} basins\n")

    existing = ["daymet", "maurer", "nldas"]
    within, across = [], []
    for i, a in enumerate(existing):
        for b in existing[i + 1:]:
            within.append(np.corrcoef(R[a], R[b])[0, 1])
        across.append(np.corrcoef(R[a], R["aorc"])[0, 1])

    print("=== how decorrelated is AORC, in the setting where it WORKED? ===")
    print(f"  among the three existing forcings : {np.mean(within):.4f}"
          f"   (n={len(within)})")
    print(f"  AORC against each of them         : {np.mean(across):.4f}"
          f"   (n={len(across)})")
    print(f"  difference                        : {np.mean(across)-np.mean(within):+.4f}")
    print()

    gap = np.mean(within) - np.mean(across)
    print("=== PRE-REGISTERED PREDICTION for the no-q member ===")
    print("  Reference points already measured in this campaign:")
    print("    a different MODEL FAMILY (dHBV vs LSTM) decorrelates by 0.039")
    print("      and was worth roughly nothing on skill")
    print(f"    AORC as a new FORCING decorrelates by {gap:.4f} here")
    print("      and was worth +0.0062 on with-q")
    print()
    if gap > 0.10:
        print(f"  AORC's decorrelation ({gap:.3f}) is LARGE relative to the family")
        print("  axis, so it is carrying genuinely new information. PREDICT the")
        print("  no-q gain lands in the upper half of 0.836-0.841.")
    elif gap > 0.04:
        print(f"  AORC's decorrelation ({gap:.3f}) is MODEST -- comparable to the")
        print("  family axis that bought nothing. PREDICT the no-q gain lands in")
        print("  the LOWER half of the range, 0.836-0.838, and may not reach 0.84.")
    else:
        print(f"  AORC's decorrelation ({gap:.3f}) is SMALL. The with-q gain likely")
        print("  came from seed count or the with-q setting rather than from the")
        print("  forcing. PREDICT little or no no-q gain, near 0.8357.")
    print()
    print("  Recorded before the no-q members finish training, so the comparison")
    print("  cannot be adjusted after the fact.")


if __name__ == "__main__":
    main()
