#!/usr/bin/env python
"""Precompute + VALIDATE the NLDAS-2 basin x grid-cell coverage weight matrix.

NLDAS-2 is hourly: 2015-2025 is ~96,400 files. Reducing each one by re-running
polygon extraction would be absurd, so the coverage weights are computed ONCE
here and every hourly file is then reduced by a sparse dot product.

This needs NO Earthdata authorization — only the documented NLDAS-2 grid and the
GAGES-II polygons — so it can be built and checked while the GES DISC app
authorization is still pending.

NLDAS-2 grid (documented, and re-verified against a real granule when auth lands):
  0.125 deg regular lat/lon, 464 x 224, lower-left CELL CENTER at
  (25.0625 N, -124.9375 E)  =>  lon_i = -124.9375 + 0.125*i, i=0..463
                                lat_j =   25.0625 + 0.125*j, j=0..223

VALIDATION: the coverage-weighted cell area must reproduce each basin's known
area. That is an independent check on CRS handling, cell-edge convention and
row order — the three things that silently corrupt an areal extraction.
"""
from __future__ import annotations
import json, math, pickle, sys
from pathlib import Path

import numpy as np
import geopandas as gpd
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

ROOT = Path("/home/nakas/riverwatch2")
SHP = ROOT / "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"
OUT = ROOT / "data/modern_corpus/nldas_basin_weights.pkl"

NX, NY, RES = 464, 224, 0.125
LON0, LAT0 = -124.9375, 25.0625          # lower-left cell CENTRE
XMIN, XMAX = LON0 - RES / 2, LON0 + (NX - 1) * RES + RES / 2
YMIN, YMAX = LAT0 - RES / 2, LAT0 + (NY - 1) * RES + RES / 2


def main():
    ids = json.load(open(ROOT / "data/camels_gauge_ids.json"))
    b671, b531 = list(ids["671"]), set(ids["531"])

    g = gpd.read_file(SHP)
    g["gid"] = g["GAGE_ID"].astype(str).str.zfill(8)
    g = g[g["gid"].isin(set(b671))].copy()
    print(f"polygons: {len(g)} of {len(b671)} basins   native CRS: {g.crs.name}")

    # equal-area CRS for the area check (the shapefile is already NAD83 Albers)
    g_albers = g.to_crs("EPSG:5070")
    area_km2 = {r.gid: r.geometry.area / 1e6 for r in g_albers.itertuples()}

    g_ll = g.to_crs("EPSG:4326")
    print(f"grid: {NY} x {NX} @ {RES} deg   lon [{XMIN}, {XMAX}]  lat [{YMIN}, {YMAX}]")

    # NLDAS arrays are conventionally stored SOUTH-to-NORTH (row 0 = southmost).
    # NumPyRasterSource assumes row 0 = NORTH (top), so we compute weights in
    # top-down space and record the flag the reducer must honour.
    probe = np.zeros((NY, NX), dtype="float64")
    rs = NumPyRasterSource(probe, XMIN, YMIN, XMAX, YMAX, srs_wkt=g_ll.crs.to_wkt())

    res = exact_extract(rs, g_ll, ["cell_id", "coverage"], output="pandas")
    weights, bad = {}, []
    for gid, cid, cov in zip(g_ll["gid"], res["cell_id"], res["coverage"]):
        cid = np.asarray(cid, dtype=np.int32)
        cov = np.asarray(cov, dtype=np.float64)
        if cov.size == 0 or cov.sum() <= 0:
            bad.append(gid)
            continue
        weights[gid] = (cid, cov)

    print(f"\nweights computed for {len(weights)} basins; {len(bad)} with zero coverage")
    if bad:
        print("  zero-coverage:", bad[:10])

    # ---- VALIDATION: reproduce known basin areas from the coverage weights ----
    # cell area varies with latitude: (111.32 km * RES) * (111.32 km * RES * cos(lat))
    lat_of_row = np.array([YMAX - (r + 0.5) * RES for r in range(NY)])   # top-down
    cell_km2 = np.repeat(
        ((111.32 * RES) * (111.32 * RES * np.cos(np.radians(lat_of_row))))[:, None],
        NX, axis=1).ravel()

    errs = []
    for gid, (cid, cov) in weights.items():
        est = float((cell_km2[cid] * cov).sum())
        true = area_km2.get(gid)
        if not true or true <= 0:
            continue
        errs.append((abs(est - true) / true * 100, gid, est, true, len(cid)))
    errs.sort(reverse=True)

    pct = [e[0] for e in errs]
    pct_s = sorted(pct)
    print(f"\n=== AREA VALIDATION (coverage-weighted vs polygon area), n={len(errs)} ===")
    print(f"  median error : {pct_s[len(pct_s)//2]:.2f}%")
    print(f"  p90 / p99    : {pct_s[int(.9*len(pct_s))]:.2f}% / {pct_s[int(.99*len(pct_s))]:.2f}%")
    print(f"  worst 8:")
    for e, gid, est, true, n in errs[:8]:
        print(f"    {gid}  est={est:9.1f} km2  true={true:9.1f} km2  "
              f"err={e:6.2f}%  cells={n}")

    small = [e for e in errs if e[3] < 100]
    if small:
        sp = sorted(x[0] for x in small)
        print(f"  (basins <100 km2: n={len(small)}, median err {sp[len(sp)//2]:.2f}% "
              f"- coarse-grid discretisation, expected)")

    ncell = sorted(len(c) for c, _ in weights.values())
    print(f"\n  cells per basin: min={ncell[0]} median={ncell[len(ncell)//2]} "
          f"max={ncell[-1]}")
    onecell = sum(1 for n in ncell if n <= 1)
    print(f"  basins covered by <=1 cell: {onecell}  "
          f"(these are effectively point-sampled by NLDAS's 12 km grid)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump({"grid": {"nx": NX, "ny": NY, "res": RES,
                              "xmin": XMIN, "xmax": XMAX,
                              "ymin": YMIN, "ymax": YMAX,
                              "row0": "north",
                              "note": "NLDAS granules are stored south-to-north; "
                                      "flip the array before applying these ids"},
                     "weights": weights,
                     "n531": sum(1 for k in weights if k in b531)}, fh)
    print(f"\nwrote {OUT}  ({len(weights)} basins, "
          f"{sum(1 for k in weights if k in b531)} in the 531)")


if __name__ == "__main__":
    main()
