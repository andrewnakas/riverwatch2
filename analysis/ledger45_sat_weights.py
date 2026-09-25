#!/usr/bin/env python3
"""Ledger 45 track 2 stage 2: per-basin areal-averaging cell weights for
(a) GlobSnow v3.0 EASE/LAEA 25 km grid and (b) ESA CCI SM 0.25 deg grid,
from the real GAGES-II polygons (same loader convention as
analysis/ledger45_camels_nesting.py: all aggeco shapefiles, GAGE_ID zfill(8),
dedup, EPSG:5070 for area sanity).

Run with ~/geo_venv/bin/python (geopandas; no xarray needed -- grid coords come
from data/satellite/ledger45_grids.npz written by the gpu venv).

Output: data/satellite/ledger45_cell_weights.npz
  gs_b, gs_iy, gs_ix, gs_w   : flattened (basin, cell) weight triplets, GlobSnow
  cci_b, cci_iy, cci_ix, cci_w : same for the CCI lat/lon grid
  gids                       : the 531 gauge ids (sorted), index base for *_b
Weights are basin-area fractions (sum to 1 per basin unless clipped at grid edge).

Sanity gates (script FAILS loudly if any trips):
  1. polygon area (EPSG:5070) vs area_gages2 attr: corr > 0.99
  2. pyproj LAEA forward vs the pilot's hand-rolled spherical formula: <1 km
  3. every basin gets >=1 cell in both grids; per-basin weight sum in [0.99, 1.01]
  4. the pilot centroid's nearest GlobSnow cell is among the weighted cells
     for >=95% of basins
"""
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(os.path.expanduser("~/riverwatch2"))
SAT = ROOT / "data/satellite"
EA_CRS = "EPSG:5070"
LAEA = "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +R=6371228 +units=m +no_defs"
R = 6371228.0


def load_polygons(shp_dir, ids):
    frames = []
    want = set(ids)
    for shp in sorted(Path(shp_dir).glob("*.shp")):
        g = gpd.read_file(shp)
        col = next(c for c in g.columns if c.upper() in ("GAGE_ID", "STAID", "GAGEID"))
        g["gid"] = g[col].astype(str).str.zfill(8)
        g = g[g["gid"].isin(want)]
        if len(g):
            frames.append(g[["gid", "geometry"]].to_crs(EA_CRS))
    out = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(out, geometry="geometry", crs=EA_CRS)
    dup = gdf["gid"].duplicated(keep="first")
    if dup.any():
        print(f"note: dropping {dup.sum()} duplicate polygon rows")
        gdf = gdf[~dup].reset_index(drop=True)
    gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf


def laea_north_formula(lat, lon):
    latr, lonr = np.radians(lat), np.radians(lon)
    rho = 2 * R * np.sin((np.pi / 2 - latr) / 2)
    return rho * np.sin(lonr), -rho * np.cos(lonr)


def cell_weights(geom, xc, yc, x_asc=True, y_asc=True, area_w=None):
    """Intersect one polygon with grid cells around its bbox.
    xc/yc are CELL CENTER coordinate arrays (1d, monotone). Returns (iy, ix, w)."""
    dx = np.median(np.diff(xc))
    dy = np.median(np.diff(yc))
    minx, miny, maxx, maxy = geom.bounds
    tx = np.array([(minx - xc[0]) / dx, (maxx - xc[0]) / dx])
    ty = np.array([(miny - yc[0]) / dy, (maxy - yc[0]) / dy])
    lo_x = int(np.clip(np.floor(tx.min()) - 1, 0, len(xc) - 1))
    hi_x = int(np.clip(np.ceil(tx.max()) + 1, 0, len(xc) - 1))
    lo_y = int(np.clip(np.floor(ty.min()) - 1, 0, len(yc) - 1))
    hi_y = int(np.clip(np.ceil(ty.max()) + 1, 0, len(yc) - 1))
    iys, ixs, ws = [], [], []
    for iy in range(lo_y, hi_y + 1):
        for ix in range(lo_x, hi_x + 1):
            cell = box(xc[ix] - abs(dx) / 2, yc[iy] - abs(dy) / 2,
                       xc[ix] + abs(dx) / 2, yc[iy] + abs(dy) / 2)
            a = geom.intersection(cell).area
            if a > 0:
                w = a * (area_w(yc[iy]) if area_w else 1.0)
                iys.append(iy); ixs.append(ix); ws.append(w)
    w = np.array(ws, float)
    return np.array(iys), np.array(ixs), w / w.sum() if w.sum() > 0 else w


def main():
    ids = sorted(json.load(open(ROOT / "data/camels_gauge_ids.json"))["531"])
    attrs = json.load(open(ROOT / "data/camels_attrs.json"))
    grids = np.load(SAT / "ledger45_grids.npz")
    gdf = load_polygons(ROOT / "data/gages2/boundaries-shapefiles-by-aggeco", ids)
    matched = set(gdf.gid)
    assert len(matched) == len(ids), f"polygon match {len(matched)}/{len(ids)}"
    gdf = gdf.set_index("gid").loc[ids]

    # gate 1: area sanity
    a_poly = gdf.geometry.area.values / 1e6
    a_ref = np.array([attrs[g].get("area_gages2", np.nan) for g in ids], float)
    ok = np.isfinite(a_ref)
    c = np.corrcoef(a_poly[ok], a_ref[ok])[0, 1]
    print(f"gate1 area corr poly-vs-attr: {c:.4f} (n={ok.sum()})")
    assert c > 0.99, "area sanity failed"

    # gate 2: LAEA CRS string matches the pilot formula
    g4326 = gdf.geometry.to_crs("EPSG:4326")
    cen = g4326.representative_point()
    fx, fy = laea_north_formula(cen.y.values, cen.x.values)
    plaea = gdf.geometry.to_crs(LAEA)
    claea = plaea.representative_point()
    # representative_point is not projection-invariant; reproject the 4326 points
    pts = gpd.GeoSeries(gpd.points_from_xy(cen.x, cen.y), crs="EPSG:4326").to_crs(LAEA)
    d = np.hypot(pts.x.values - fx, pts.y.values - fy)
    print(f"gate2 pyproj-vs-formula max dist: {d.max():.1f} m")
    assert d.max() < 1000, "LAEA CRS mismatch with pilot formula"

    # --- GlobSnow weights (equal-area projection: raw intersection area) ----
    xc, yc = grids["gs_x"], grids["gs_y"]
    gs_b, gs_iy, gs_ix, gs_w = [], [], [], []
    for bi, g in enumerate(ids):
        iy, ix, w = cell_weights(plaea.loc[g], xc, yc)
        assert len(w) > 0 and 0.99 < w.sum() < 1.01, f"gs weights bad for {g}"
        gs_b += [bi] * len(w); gs_iy += list(iy); gs_ix += list(ix); gs_w += list(w)

    # gate 4: centroid cell containment
    centj = json.load(open(SAT / "camels531_centroids.json"))
    hit = 0
    wdf = pd.DataFrame({"b": gs_b, "iy": gs_iy, "ix": gs_ix})
    for bi, g in enumerate(ids):
        cx, cy = laea_north_formula(centj[g]["lat"], centj[g]["lon"])
        nix = int(np.argmin(np.abs(xc - cx))); niy = int(np.argmin(np.abs(yc - cy)))
        sub = wdf[wdf.b == bi]
        if ((sub.iy == niy) & (sub.ix == nix)).any():
            hit += 1
    print(f"gate4 centroid-cell containment: {hit}/{len(ids)}")
    assert hit >= 0.95 * len(ids), "centroid containment failed"

    # --- CCI weights (lat/lon grid, cos-lat area weighting) -----------------
    lat_c, lon_c = grids["cci_lat"], grids["cci_lon"]
    cci_b, cci_iy, cci_ix, cci_w = [], [], [], []
    for bi, g in enumerate(ids):
        geom = g4326.loc[g]
        iy, ix, w = cell_weights(geom, lon_c, lat_c,
                                 area_w=lambda latc: np.cos(np.radians(latc)))
        # note: cell_weights(xc=lon, yc=lat) -> returned (iy over lat, ix over lon)
        assert len(w) > 0 and 0.99 < w.sum() < 1.01, f"cci weights bad for {g}"
        cci_b += [bi] * len(w); cci_iy += list(iy); cci_ix += list(ix); cci_w += list(w)

    ncell_gs = pd.Series(gs_b).value_counts()
    ncell_cci = pd.Series(cci_b).value_counts()
    print(f"gs cells/basin: median {ncell_gs.median():.0f} max {ncell_gs.max()}")
    print(f"cci cells/basin: median {ncell_cci.median():.0f} max {ncell_cci.max()}")

    np.savez(SAT / "ledger45_cell_weights.npz",
             gids=np.array(ids),
             gs_b=np.array(gs_b, np.int32), gs_iy=np.array(gs_iy, np.int32),
             gs_ix=np.array(gs_ix, np.int32), gs_w=np.array(gs_w, np.float64),
             cci_b=np.array(cci_b, np.int32), cci_iy=np.array(cci_iy, np.int32),
             cci_ix=np.array(cci_ix, np.int32), cci_w=np.array(cci_w, np.float64))
    print("wrote", SAT / "ledger45_cell_weights.npz")


if __name__ == "__main__":
    main()
