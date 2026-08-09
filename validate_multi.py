#!/usr/bin/env python3
"""Validate the polygon method across SEVERAL published basins, over a longer window.

The single-basin check gave precip ratio 0.932 / corr 0.952 -- an 87% reduction
of the point-sample bias, but not an exact match. Two hypotheses:

  H1 polygon mismatch: our NLDI boundary differs slightly from Andy Wood's 2024
     GAGES-II re-delineation. Then the bias should be CONSISTENTLY SIGNED across
     basins and correlate with the pressure (elevation) residual.
  H2 short-window noise: 144 hours of precip in one basin is a tiny sample and
     the ratio is dominated by a few storm hours. Then the bias should shrink and
     the correlation rise over a longer window, and scatter randomly across basins.

These imply different actions, so measure rather than assume. A longer window
(1440 h = 60 days) and 4 basins separates them.

What actually matters for a training member is CORRELATION (does the series carry
the right temporal signal) more than an absolute scale factor, since the trainer
normalizes each variable per basin. A constant multiplicative offset is largely
absorbed by normalization; a decorrelated series is not.
"""
import io, sys, tarfile
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from aorc_exact_extract import VARS, extract_year, get_polygon

BASINS = ["13240000", "13161500", "13185000", "12413000"]
YEAR, TMAX, TAR = 1990, 1440, "data/aorc_raw/water_year_1990.tar.gz"

pub = {}
with tarfile.open(TAR, "r:gz") as tf:
    idx = {m.name.split("/")[-1][:8]: m for m in tf.getmembers() if m.name.endswith(".csv")}
    for b in BASINS:
        if b in idx:
            d = pd.read_csv(io.BytesIO(tf.extractfile(idx[b]).read()))
            d["time"] = pd.to_datetime(d["time"]); pub[b] = d

print(f"{'basin':>9} {'var':22} {'ratio':>7} {'corr':>7}")
res = []
for b in BASINS:
    if b not in pub:
        print(f"{b:>9}  (not published)"); continue
    try:
        gdf = get_polygon(b)
        mine = extract_year(b, gdf, YEAR, tmax=TMAX)
    except Exception as e:
        print(f"{b:>9}  EXTRACT FAILED: {str(e)[:70]}"); continue
    j = mine.merge(pub[b], on="time", suffixes=("_mine", "_pub"))
    for v in ("APCP_surface", "TMP_2maboveground", "PRES_surface"):
        if f"{v}_pub" not in j: continue
        a, c = j[f"{v}_mine"].to_numpy(float), j[f"{v}_pub"].to_numpy(float)
        g = np.isfinite(a) & np.isfinite(c)
        r = np.corrcoef(a[g], c[g])[0, 1]
        ratio = np.nansum(a) / np.nansum(c)
        print(f"{b:>9} {v:22} {ratio:>7.3f} {r:>7.4f}")
        res.append((b, v, ratio, r))

print("\n=== summary ===")
for v in ("APCP_surface", "TMP_2maboveground", "PRES_surface"):
    rr = [x[2] for x in res if x[1] == v]
    cc = [x[3] for x in res if x[1] == v]
    if rr:
        print(f"{v:22} ratio mean={np.mean(rr):.3f} sd={np.std(rr):.3f} | "
              f"corr mean={np.mean(cc):.4f} min={np.min(cc):.4f}")
pr = [x[2] for x in res if x[1] == "APCP_surface"]
pp = [x[2] for x in res if x[1] == "PRES_surface"]
if len(pr) > 2:
    print(f"\ncorr(precip ratio, pressure ratio) across basins = "
          f"{np.corrcoef(pr, pp)[0,1]:+.3f}")
    print("  strongly negative => H1 (polygon elevation mismatch)")
    print("  near zero         => H2 (sampling noise)")
