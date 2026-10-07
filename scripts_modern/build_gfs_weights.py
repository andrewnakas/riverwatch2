#!/usr/bin/env python
"""Precompute + VALIDATE the GFS 0.25 deg basin x cell coverage weights.

Modelled on build_nldas_weights.py, including its area check, because that check is an independent
test of the three things that silently corrupt an areal extraction: CRS handling, cell-edge
convention and ROW ORDER.

⭐ The grid is DERIVED from a real decoded GFS field, not assumed. The documented pgrb2.0p25 grid is
1440x721 with row 0 at +90 and longitudes in 0..359.75, but asserting that from documentation is
exactly how a one-row flip gets baked in. We fetch one record, read its own coordinates, and build
the CONUS subwindow from them -- then record the subwindow so the reducer applies the identical
slice.
"""
from __future__ import annotations
import json, os, pickle, tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import s3fs, xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

ROOT = Path("/home/nakas/riverwatch2")
SHP = ROOT / "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"
OUT = ROOT / "data/modern_corpus/gfs_basin_weights.pkl"
B = "noaa-gfs-bdp-pds"
BASE = "gfs.t00z.pgrb2.0p25.f024"
PROBE_DAY = "20230615"

# CONUS window with margin for every one of the 671 basins
LON_W, LON_E, LAT_S, LAT_N = -126.0, -65.0, 23.0, 54.0


def fetch_one_record(fs, var="APCP", lev="surface"):
    p = f"{B}/gfs.{PROBE_DAY}/00/atmos"
    grb = f"{p}/{BASE}"
    idx = f"{p}/{BASE}.idx"
    lines = fs.cat(idx).decode().strip().split("\n")
    for i, ln in enumerate(lines):
        f = ln.split(":")
        if len(f) > 4 and f[3] == var and lev in f[4]:
            start = int(f[1])
            nxt = lines[i + 1].split(":")[1] if i + 1 < len(lines) else None
            if nxt is None:
                return None
            # cat_file does a real ranged GET; read_block is ~255x slower here (measured)
            return fs.cat_file(grb, start=start, end=int(nxt))
    return None


def main():
    fs = s3fs.S3FileSystem(anon=True)
    blob = fetch_one_record(fs)
    assert blob and blob[:4] == b"GRIB", "could not fetch a decodable GRIB record"
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as t:
        t.write(blob); tmp = t.name
    try:
        ds = xr.open_dataset(tmp, engine="cfgrib", backend_kwargs={"indexpath": ""})
        v = list(ds.data_vars)[0]
        lat = np.asarray(ds["latitude"].values, dtype="float64")
        lon = np.asarray(ds["longitude"].values, dtype="float64")
        print(f"decoded {v}: shape {ds[v].shape}")
        print(f"  latitude  {lat[0]:.3f} .. {lat[-1]:.3f}  ({len(lat)} rows, "
              f"{'NORTH-first' if lat[0] > lat[-1] else 'SOUTH-first'})")
        print(f"  longitude {lon[0]:.3f} .. {lon[-1]:.3f}  ({len(lon)} cols)")
        res = float(abs(lat[1] - lat[0]))
        print(f"  resolution {res} deg")
    finally:
        os.unlink(tmp)

    # longitudes arrive in 0..360; convert to -180..180 for the polygon CRS
    lon180 = np.where(lon > 180.0, lon - 360.0, lon)
    ci = np.where((lon180 >= LON_W) & (lon180 <= LON_E))[0]
    ri = np.where((lat >= LAT_S) & (lat <= LAT_N))[0]
    assert len(ci) and len(ri), "CONUS window selected no cells"
    # the lon subset must be contiguous after the wrap, or a slice would scramble it
    assert np.all(np.diff(ci) == 1), "longitude subset is not contiguous"
    assert np.all(np.diff(ri) == 1), "latitude subset is not contiguous"
    sub_lat, sub_lon = lat[ri], lon180[ci]
    north_first = sub_lat[0] > sub_lat[-1]
    NY, NX = len(ri), len(ci)
    XMIN, XMAX = sub_lon.min() - res / 2, sub_lon.max() + res / 2
    YMIN, YMAX = sub_lat.min() - res / 2, sub_lat.max() + res / 2
    print(f"\nCONUS subwindow: {NY} x {NX} @ {res} deg   rows {ri[0]}..{ri[-1]}  "
          f"cols {ci[0]}..{ci[-1]}   row0={'north' if north_first else 'south'}")
    print(f"  lon [{XMIN:.3f}, {XMAX:.3f}]  lat [{YMIN:.3f}, {YMAX:.3f}]")

    ids = json.load(open(ROOT / "data/camels_gauge_ids.json"))
    b671 = list(ids["671"])
    g = gpd.read_file(SHP)
    g["gid"] = g["GAGE_ID"].astype(str).str.zfill(8)
    g = g[g["gid"].isin(set(b671))].copy()
    print(f"polygons: {len(g)} of {len(b671)}   native CRS: {g.crs.name}")
    g_albers = g.to_crs("EPSG:5070")
    area_km2 = {r.gid: r.geometry.area / 1e6 for r in g_albers.itertuples()}
    g_ll = g.to_crs("EPSG:4326")

    # NumPyRasterSource assumes row 0 = NORTH. Compute in that space and record the flag the
    # reducer must honour, exactly as the NLDAS builder does.
    probe = np.zeros((NY, NX), dtype="float64")
    rs = NumPyRasterSource(probe, XMIN, YMIN, XMAX, YMAX, srs_wkt=g_ll.crs.to_wkt())
    res_df = exact_extract(rs, g_ll, ["cell_id", "coverage"], output="pandas")

    weights, bad = {}, []
    for gid, cid, cov in zip(g_ll["gid"], res_df["cell_id"], res_df["coverage"]):
        cid = np.asarray(cid, dtype=np.int32)
        cov = np.asarray(cov, dtype=np.float64)
        if cov.size == 0 or cov.sum() <= 0:
            bad.append(gid); continue
        weights[gid] = (cid, cov)
    print(f"weights: {len(weights)} basins, {len(bad)} with no coverage" +
          (f" -> {bad[:5]}" if bad else ""))

    # VALIDATION: coverage-weighted cell area must reproduce the known basin area.
    errs = []
    for gid, (cid, cov) in weights.items():
        rows = cid // NX
        latc = sub_lat[rows]
        cell_km2 = (res * 111.32) * (res * 111.32 * np.cos(np.deg2rad(latc)))
        est = float((cov * cell_km2).sum())
        kn = area_km2.get(gid)
        if kn and kn > 0:
            errs.append(abs(est - kn) / kn)
    errs = np.array(errs)
    print(f"area check on {len(errs)} basins: median {np.median(errs)*100:.2f}%  "
          f"p90 {np.percentile(errs,90)*100:.2f}%  max {errs.max()*100:.2f}%")
    ok = np.median(errs) < 0.05
    print("✅ area check passes" if ok else "⚠️ area check FAILS - do not use these weights")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"grid": {"ny": NY, "nx": NX, "res": res,
                          "row0": "north" if north_first else "south",
                          "row_idx": [int(ri[0]), int(ri[-1])],
                          "col_idx": [int(ci[0]), int(ci[-1])],
                          "xmin": XMIN, "xmax": XMAX, "ymin": YMIN, "ymax": YMAX,
                          "lon_convention": "source 0..360, converted to -180..180",
                          "note": "apply row_idx/col_idx to the DECODED global field, then use "
                                  "cell_id = row*nx + col in this subwindow"},
                 "weights": weights, "area_median_err": float(np.median(errs)),
                 "n_basins": len(weights)},
                open(OUT, "wb"))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
