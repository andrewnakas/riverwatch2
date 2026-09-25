#!/usr/bin/env python3
"""Ledger 45 track 2 stage 2: build basin-averaged daily series (531 CAMELS
basins) from the full 1980-2008 satellite archives, using the polygon cell
weights from ledger45_sat_weights.py (REAL areal averaging, not centroids).

Usage: ledger45_sat_extract.py {globsnow|cci_combined|cci_adjusted}
Run with ~/riverwatch2/gpu1080/.venv/bin/python (xarray).

Per file: basin value = weighted mean over VALID cells; basin vfrac = sum of
weights of valid cells (0..1). GlobSnow valid = swe >= 0 (-1 water, -2 masked);
CCI valid = finite sm.

Output: data/satellite/ledger45_{product}_daily.npz
  dates (int YYYYMMDD), value (T,531) f32 NaN when vfrac==0, vfrac (T,531) f32,
  gids (531)
plus ledger45_{product}_yearfrac.json: per-basin per-year mean valid fraction.
Artifact verification printed: date span, file count vs calendar, value range,
per-decade vfrac.
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.path.expanduser("~/riverwatch2"))
SAT = ROOT / "data/satellite"

PRODUCTS = {
    "globsnow": (SAT / "globsnow_full", "swe"),
    "cci_combined": (SAT / "cci_combined_v071", "sm"),
    "cci_adjusted": (SAT / "cci_adjusted_v071", "sm"),
}


def main():
    prod = sys.argv[1]
    dirp, var = PRODUCTS[prod]
    W = np.load(SAT / "ledger45_cell_weights.npz")
    gids = [str(g) for g in W["gids"]]
    nb = len(gids)
    if prod == "globsnow":
        b, iy, ix, w = W["gs_b"], W["gs_iy"], W["gs_ix"], W["gs_w"]
    else:
        b, iy, ix, w = W["cci_b"], W["cci_iy"], W["cci_ix"], W["cci_w"]

    files = sorted(glob.glob(str(dirp / "*.nc")))
    print(f"{prod}: {len(files)} files in {dirp}")
    import netCDF4

    dates, values, vfracs = [], [], []
    bad = []
    for k, f in enumerate(files):
        m = re.search(r"(\d{8})", os.path.basename(f))
        date = int(m.group(1))
        try:
            ds = netCDF4.Dataset(f)
            v = ds.variables[var]
            arr = v[:].astype(np.float64)
            ds.close()
        except Exception as e:
            bad.append((os.path.basename(f), str(e)))
            continue
        if arr.ndim == 3:
            arr = arr[0]
        arr = np.ma.filled(arr, np.nan)
        cell = arr[iy, ix]
        if prod == "globsnow":
            valid = np.isfinite(cell) & (cell >= 0)
        else:
            valid = np.isfinite(cell)
        wv = np.where(valid, w, 0.0)
        num = np.bincount(b, weights=np.where(valid, wv * np.nan_to_num(cell), 0.0), minlength=nb)
        den = np.bincount(b, weights=wv, minlength=nb)
        val = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        dates.append(date)
        values.append(val.astype(np.float32))
        vfracs.append(den.astype(np.float32))
        if k % 1000 == 0:
            print(f"  {k}/{len(files)} {date}", flush=True)

    dates = np.array(dates, np.int64)
    order = np.argsort(dates)
    dates = dates[order]
    value = np.stack(values)[order]
    vfrac = np.stack(vfracs)[order]
    assert len(np.unique(dates)) == len(dates), "duplicate dates"
    out = SAT / f"ledger45_{prod}_daily.npz"
    np.savez_compressed(out, dates=dates, value=value, vfrac=vfrac,
                        gids=np.array(gids))

    # per-basin per-year mean valid fraction (over the files that exist)
    years = dates // 10000
    yf = {}
    for y in np.unique(years):
        sel = years == y
        yf[int(y)] = np.round(vfrac[sel].mean(axis=0), 4).tolist()
    json.dump({"gids": gids, "year_vfrac": yf},
              open(SAT / f"ledger45_{prod}_yearfrac.json", "w"))

    # ---- artifact verification ------------------------------------------
    print(f"\nARTIFACT CHECK {prod}")
    print(f"  files parsed: {len(dates)}, failed: {len(bad)}")
    for fb in bad[:5]:
        print("   bad:", fb)
    print(f"  date span: {dates[0]} .. {dates[-1]}")
    fin = value[np.isfinite(value)]
    print(f"  value range: {fin.min():.4f} .. {fin.max():.4f} (n finite {fin.size})")
    for y0, y1 in [(1980, 1984), (1985, 1989), (1990, 1994), (1995, 1999),
                   (2000, 2004), (2005, 2008)]:
        sel = (years >= y0) & (years <= y1)
        print(f"  {y0}-{y1}: files {sel.sum():5d}  mean vfrac {vfrac[sel].mean():.3f}"
              f"  mean value {np.nanmean(value[sel]):.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
