#!/usr/bin/env python
"""DECISIVE test: Daymet V4R1 AREAL vs the modern POINT corpus.

The first acceptance test compared V4R1 against CAMELS' CIDA-era Daymet (V2/V3),
which confounds SAMPLING with dataset VERSION. This test removes the version
confound entirely: same Daymet V4, same days, same basins — the ONLY difference
is areal-mean vs single-pixel-at-the-gauge.

Expected if the new pipeline is correct:
  * HIGH day-to-day correlation (identical underlying data), and
  * a systematic bias in the known point-sampling direction
    (point drier / warmer, growing with terrain steepness).
If instead correlation is low, the extraction itself is wrong.
"""
from __future__ import annotations
import gzip, json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
from pyproj import CRS
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

ROOT = Path("/home/nakas/riverwatch2")
SHP = ROOT / "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"
POINT = ROOT / "data/modern_corpus/camels_corpus_daymet_modern"
BASE = ("https://opendap.earthdata.nasa.gov/collections/C2532426483-ORNL_CLOUD/"
        "granules/Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{year}.nc")
X0, Y0, RES, NX, NY = -4_560_250.0, 4_984_000.0, 1000.0, 7814, 8075
LCC = ("+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 "
       "+ellps=WGS84 +units=m +no_defs")

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
NBASIN = int(sys.argv[2]) if len(sys.argv) > 2 else 8
VARS = ["prcp", "tmax", "tmin"]
COL = {"prcp": "precipitation_sum", "tmax": "temperature_2m_max",
       "tmin": "temperature_2m_min"}


def fetch(var, year, i0, i1, j0, j1, out):
    ce = (f"/x%5B{i0}:{i1}%5D;/y%5B{j0}:{j1}%5D;/time%5B0:364%5D;"
          f"/lambert_conformal_conic;"
          f"/{var}%5B0:364%5D%5B{j0}:{j1}%5D%5B{i0}:{i1}%5D")
    url = BASE.format(var=var, year=year) + ".dap.nc4?dap4.ce=" + ce
    cj = "/tmp/.edl_cookies_pva"
    r = subprocess.run(["curl", "-g", "-s", "-n", "-L", "-c", cj, "-b", cj,
                        "--max-time", "300", "-o", str(out), url],
                       capture_output=True)
    if r.returncode != 0 or out.stat().st_size < 10_000:
        raise RuntimeError(f"fetch failed rc={r.returncode} "
                           f"size={out.stat().st_size if out.exists() else 0}")
    return out


def idx(geom):
    minx, miny, maxx, maxy = geom.bounds
    return (max(0, int((minx - X0) / RES) - 2), min(NX - 1, int((maxx - X0) / RES) + 2),
            max(0, int((Y0 - maxy) / RES) - 2), min(NY - 1, int((Y0 - miny) / RES) + 2))


def load_point(gauge, year):
    p = POINT / f"{gauge}.csv.gz"
    if not p.exists():
        return {}
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        out = {}
        for line in fh:
            f = line.rstrip("\n").split(",")
            if not f[ci["date"]].startswith(str(year)):
                continue
            rec = {}
            for c in COL.values():
                try:
                    rec[c] = float(f[ci[c]])
                except (ValueError, KeyError):
                    rec[c] = None
            out[f[ci["date"]]] = rec
    return out


def main():
    ids = json.load(open(ROOT / "data/camels_gauge_ids.json"))
    attrs = json.load(open(ROOT / "data/camels_attrs.json"))
    b531 = sorted(set(ids["531"]))
    g = gpd.read_file(SHP)
    g["gid"] = g["GAGE_ID"].astype(str).str.zfill(8)

    cand = [x for x in b531 if x in attrs and attrs[x].get("slope_mean") is not None
            and (POINT / f"{x}.csv.gz").exists()]
    cand.sort(key=lambda x: float(attrs[x]["slope_mean"]))
    step = max(1, len(cand) // NBASIN)
    sample = cand[::step][:NBASIN]
    print(f"Daymet V4R1 AREAL vs modern POINT corpus, year={YEAR}, "
          f"{len(sample)} basins by slope")
    print(f"slopes: {[round(float(attrs[s]['slope_mean']),1) for s in sample]}\n")

    tmpd = Path(tempfile.mkdtemp(prefix="dm_pva_"))
    rows = []
    for n, gauge in enumerate(sample, 1):
        row = g[g["gid"] == gauge]
        if row.empty:
            continue
        geom = row.to_crs(LCC).geometry.iloc[0]
        i0, i1, j0, j1 = idx(geom)
        areal, times, weights, crs = {}, None, None, None
        try:
            for var in VARS:
                out = tmpd / f"{gauge}_{var}.nc4"
                fetch(var, YEAR, i0, i1, j0, j1, out)
                ds = xr.open_dataset(out)
                if weights is None:
                    crs = CRS.from_cf(ds["lambert_conformal_conic"].attrs)
                    gdf = row.to_crs(crs)
                    xs, ys = ds["x"].values, ds["y"].values
                    rs = NumPyRasterSource(
                        np.zeros((len(ys), len(xs))),
                        xs[0] - RES / 2, ys[-1] - RES / 2,
                        xs[-1] + RES / 2, ys[0] + RES / 2,
                        srs_wkt=crs.to_wkt())
                    w = exact_extract(rs, gdf, ["cell_id", "coverage"],
                                      output="pandas")
                    cid = np.asarray(w["cell_id"].iloc[0], dtype=int)
                    cov = np.asarray(w["coverage"].iloc[0], dtype=float)
                    weights = (cid, cov, len(ys), len(xs))
                    times = [str(np.datetime_as_string(t, unit="D"))
                             for t in ds["time"].values]
                a = ds[var].values.astype("float64")
                cid, cov, ny_, nx_ = weights
                areal[var] = (a.reshape(a.shape[0], ny_ * nx_)[:, cid] * cov
                              ).sum(1) / cov.sum()
                ds.close()
        except Exception as e:
            print(f"  [{n}] {gauge} FAIL {e}")
            continue

        pt = load_point(gauge, YEAR)
        rec = {"gauge": gauge, "slope": float(attrs[gauge]["slope_mean"]),
               "cells": len(weights[0])}
        for var in VARS:
            col = COL[var]
            A, P = [], []
            for k, t in enumerate(times):
                q = pt.get(t)
                if q and q.get(col) is not None:
                    A.append(areal[var][k]); P.append(q[col])
            if len(A) < 300:
                continue
            A, P = np.array(A), np.array(P)
            rec[col] = {"r": float(np.corrcoef(A, P)[0, 1]),
                        "areal": A.mean(), "point": P.mean(),
                        "pt_minus_areal": P.mean() - A.mean(),
                        "pct": (P.mean() - A.mean()) / A.mean() * 100
                        if A.mean() else float("nan")}
        rows.append(rec)
        print(f"  [{n}] {gauge} slope={rec['slope']:.1f} cells={rec['cells']} done")

    print("\n" + "=" * 78)
    print("POINT minus AREAL, SAME Daymet V4 version, SAME days")
    print("=" * 78)
    for r in rows:
        print(f"\n{r['gauge']}  slope={r['slope']:.1f}  cells={r['cells']}")
        for col in COL.values():
            if col not in r:
                continue
            s = r[col]
            print(f"   {col:<24} r={s['r']:.5f}  point-areal={s['pt_minus_areal']:+.4f}"
                  f"  ({s['pct']:+.2f}%)")

    print("\n--- INTERPRETATION ---")
    rs_ = {c: [r[c]["r"] for r in rows if c in r] for c in COL.values()}
    for c, v in rs_.items():
        if v:
            print(f"  {c:<24} median r={sorted(v)[len(v)//2]:.5f}  min={min(v):.5f}")
    pp = [(r["slope"], r["precipitation_sum"]["pct"]) for r in rows
          if "precipitation_sum" in r]
    if len(pp) > 3:
        xs = np.array([a for a, _ in pp]); ys = np.array([b for _, b in pp])
        print(f"\n  corr(precip point-bias %, slope) = {np.corrcoef(xs, ys)[0,1]:+.3f}")
        print("  (negative => point reads drier as terrain steepens = the "
              "orographic sampling signature)")


if __name__ == "__main__":
    main()
