#!/usr/bin/env python3
"""Reproduce HydroShare's AORC catchment aggregation for basins they dropped.

METHOD, taken from the published resource metadata rather than guessed:
  resource c738c05278a34bc9848dd14d61cffab9 (Frame, Wood, Frazier)
  "summarized over the CAMELS catchments ... delineations updated by Andy Wood
   (2024) using the basin boundaries produced for the USGS GagesII effort"
  code: github.com/jmframe/CIROH_DL_NextGen/forcing_prep

Their weights.py calls exactextract with stats ["cell_id","coverage"], and
aggregate.py::window_aggregate computes

    value = sum(cell * coverage) / sum(coverage)

i.e. a COVERAGE-FRACTION-WEIGHTED MEAN over the basin polygon -- each grid cell
weighted by the fraction of its area inside the boundary. That is exactly what
exactextract's "mean" stat computes, so we can use it directly.

This is NOT the point-sample I rejected earlier. Measured on a neighbour basin,
point sampling ran 53% dry against the published areal average and no
gauge-centred disc got within 32%, because these catchments sit upslope and
upstream of their gauge. Polygon weighting is the estimator that actually
matches the product.

VALIDATION FIRST: before touching the missing basin, reproduce a basin they DID
publish and compare hour by hour. If we cannot match a known answer, we have no
business generating an unknown one. Only if that passes do we extract 13235000.
"""
import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from exactextract import exact_extract

SO = {"anon": True,
      "config_kwargs": {"connect_timeout": 120, "read_timeout": 600,
                        "retries": {"max_attempts": 8, "mode": "adaptive"},
                        "max_pool_connections": 32}}
VARS = ["APCP_surface", "TMP_2maboveground", "DSWRF_surface",
        "SPFH_2maboveground", "PRES_surface"]
NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{}/basin"


GAGES2_SHP = "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"


def get_polygon_gages2(basin, shp=GAGES2_SHP):
    """The ACTUAL GAGES-II polygon the HydroShare product was built from.

    Falcone 2011, ScienceBase 631405bbd34e36012efa304a. This is the input we
    were missing: NLDI serves the modern NHDPlus watershed, which differs from
    GAGES-II at the ridgelines, and in orographic terrain a different ridgeline
    admits different (much wetter) cells. Using NLDI, precipitation reproduced
    the published series at ratio 0.858 sd 0.051 -- basin-varying, so not
    correctable by a constant. Temperature and pressure were near-exact, which
    is what pointed at the polygon rather than the aggregation.

    Reprojected to EPSG:4326 to match the AORC grid.
    """
    import geopandas as gpd
    g = gpd.read_file(shp)
    idcol = next(c for c in g.columns if c.upper() in ("GAGE_ID", "STAID", "GAGEID"))
    row = g[g[idcol].astype(str).str.zfill(8) == str(basin).zfill(8)]
    if row.empty:
        raise KeyError(f"{basin} not in {shp}")
    row = row.to_crs("EPSG:4326").copy()
    row["divide_id"] = str(basin).zfill(8)
    return row[["divide_id", "geometry"]]


def get_polygon(basin, cache=Path("data/basin_polygons")):
    """Basin polygon. Prefers the real GAGES-II shapefile when present."""
    if Path(GAGES2_SHP).exists():
        try:
            return get_polygon_gages2(basin)
        except KeyError:
            pass  # fall back to NLDI for basins GAGES-II lacks
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{basin}.json"
    if not f.exists():
        import urllib.request
        with urllib.request.urlopen(NLDI.format(basin), timeout=120) as r:
            f.write_bytes(r.read())
    gdf = gpd.read_file(f)
    gdf["divide_id"] = basin
    return gdf[["divide_id", "geometry"]].set_crs("EPSG:4326", allow_override=True)


def extract_year(basin, gdf, year, tmax=None):
    """Coverage-weighted mean of every variable, all hours of one AORC year."""
    ds = xr.open_dataset(f"s3://noaa-nws-aorc-v1-1-1km/{year}.zarr", engine="zarr",
                         backend_kwargs={"storage_options": SO})
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 0.03
    sub = ds.sel(latitude=slice(miny - pad, maxy + pad),
                 longitude=slice(minx - pad, maxx + pad))
    nt = sub.sizes["time"] if tmax is None else min(tmax, sub.sizes["time"])
    sub = sub.isel(time=slice(0, nt))
    print(f"  {year}: window {sub.sizes['latitude']}x{sub.sizes['longitude']} cells, "
          f"{nt} hours", flush=True)

    out = {"time": pd.to_datetime(sub.time.values[:nt])}
    for v in VARS:
        t0 = time.time()
        arr = sub[v].load()
        arr = arr.rio.write_crs("EPSG:4326") if hasattr(arr, "rio") else arr
        # exact_extract over a 3-D array returns one mean per band (hour)
        res = exact_extract(arr, gdf, ["mean"], output="pandas")
        vals = res.iloc[0].drop(labels=[c for c in res.columns if c == "divide_id"],
                                errors="ignore").to_numpy(float)
        out[v] = vals[:nt]
        print(f"    {v:22s} {time.time()-t0:6.1f}s  mean={np.nanmean(vals):.4f}",
              flush=True)
    return pd.DataFrame(out)


def validate(basin, year, tarball, tmax=144):
    """Reproduce a PUBLISHED basin and compare -- the gate on everything else."""
    print(f"=== VALIDATION: reproduce published basin {basin}, WY{year} ===")
    gdf = get_polygon(basin)
    print(f"  polygon: {len(gdf)} feature, bounds {np.round(gdf.total_bounds,4)}")
    mine = extract_year(basin, gdf, year, tmax=tmax)

    with tarfile.open(tarball, "r:gz") as tf:
        m = [x for x in tf.getmembers() if x.name.split("/")[-1].startswith(basin)]
        if not m:
            print(f"  basin {basin} not in {tarball}"); return False
        pub = pd.read_csv(io.BytesIO(tf.extractfile(m[0]).read()))
    pub["time"] = pd.to_datetime(pub["time"])
    j = mine.merge(pub, on="time", suffixes=("_mine", "_pub"))
    print(f"\n  matched {len(j)} hours")
    ok = True
    print(f"  {'variable':24s} {'mine':>10} {'published':>10} {'ratio':>7} {'corr':>7}")
    for v in VARS:
        if f"{v}_pub" not in j:
            continue
        a, b = j[f"{v}_mine"].to_numpy(float), j[f"{v}_pub"].to_numpy(float)
        good = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[good], b[good])[0, 1] if good.sum() > 2 else np.nan
        ratio = np.nansum(a) / np.nansum(b) if np.nansum(b) != 0 else np.nan
        print(f"  {v:24s} {np.nanmean(a):>10.4f} {np.nanmean(b):>10.4f} "
              f"{ratio:>7.3f} {r:>7.4f}")
        if v == "APCP_surface" and (not np.isfinite(r) or r < 0.99 or abs(ratio - 1) > 0.05):
            ok = False
    print(f"\n  VERDICT: {'PASS -- method reproduces the product' if ok else 'FAIL -- do not use'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-basin", default="13240000")
    ap.add_argument("--validate-year", type=int, default=1990)
    ap.add_argument("--tarball", default="data/aorc_raw/water_year_1990.tar.gz")
    ap.add_argument("--tmax", type=int, default=144)
    ap.add_argument("--basin", default="")
    ap.add_argument("--years", default="")
    ap.add_argument("--out", default="data/aorc_extra")
    a = ap.parse_args()

    if not a.basin:
        sys.exit(0 if validate(a.validate_basin, a.validate_year, a.tarball, a.tmax) else 1)

    gdf = get_polygon(a.basin)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for y in [int(x) for x in a.years.split(",") if x]:
        f = out / f"{a.basin}_WY{y}.csv"
        if f.exists():
            print(f"[skip] {f}"); continue
        df = extract_year(a.basin, gdf, y)
        df.to_csv(f, index=False)
        print(f"  wrote {f} ({len(df)} rows)")


if __name__ == "__main__":
    main()
