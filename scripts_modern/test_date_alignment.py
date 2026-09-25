#!/usr/bin/env python
"""Is the low V4-vs-V2/V3 daily precip correlation a REAL version difference,
or a DATE MISALIGNMENT bug in the areal build?

Decisive test: correlate the rebuilt areal series against the frozen corpus at
lags -3..+3. If some non-zero lag is clearly better, we have an off-by-N date
bug and the whole rebuild is compromised. If lag 0 is best, the disagreement is
genuinely the Daymet version revision.

Runs on a leap year (2012) on purpose: that is where Daymet's 365-day calendar
and CAMELS' 366-day `_leap` files could diverge.
"""
from __future__ import annotations
import gzip, math, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
from pyproj import CRS
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

ROOT = Path("/home/nakas/riverwatch2")
SHP = ROOT / "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"
FROZEN = ROOT / "gpu1080/corpora671/camels_corpus_daymet_v2"
BASE = ("https://opendap.earthdata.nasa.gov/collections/C2532426483-ORNL_CLOUD/"
        "granules/Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{year}.nc")
LCC = ("+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 "
       "+ellps=WGS84 +units=m +no_defs")
X0, Y0, RES, NX, NY = -4_560_250.0, 4_984_000.0, 1000.0, 7814, 8075

YEAR = int(sys.argv[2]) if len(sys.argv) > 2 else 2012
GAUGES = (sys.argv[1] if len(sys.argv) > 1 else "01518862,02198100,05591550").split(",")


def fetch(var, year, i0, i1, j0, j1, out):
    ce = (f"/x%5B{i0}:{i1}%5D;/y%5B{j0}:{j1}%5D;/time%5B0:364%5D;"
          f"/lambert_conformal_conic;"
          f"/{var}%5B0:364%5D%5B{j0}:{j1}%5D%5B{i0}:{i1}%5D")
    url = BASE.format(var=var, year=year) + ".dap.nc4?dap4.ce=" + ce
    cj = "/tmp/.edl_align"
    r = subprocess.run(["curl", "-g", "-s", "-n", "-L", "-c", cj, "-b", cj,
                        "--max-time", "300", "-o", str(out), url],
                       capture_output=True)
    if r.returncode != 0 or out.stat().st_size < 10_000:
        raise RuntimeError("fetch failed")


def corr(a, b):
    if len(a) < 30:
        return float("nan")
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    g = gpd.read_file(SHP)
    g["gid"] = g["GAGE_ID"].astype(str).str.zfill(8)
    tmpd = Path(tempfile.mkdtemp(prefix="align_"))

    for gauge in GAUGES:
        row = g[g["gid"] == gauge]
        if row.empty:
            print(f"{gauge}: not in shapefile"); continue
        geom = row.to_crs(LCC).geometry.iloc[0]
        mnx, mny, mxx, mxy = geom.bounds
        i0 = max(0, int((mnx - X0) / RES) - 2); i1 = min(NX - 1, int((mxx - X0) / RES) + 2)
        j0 = max(0, int((Y0 - mxy) / RES) - 2); j1 = min(NY - 1, int((Y0 - mny) / RES) + 2)

        mine = {}
        for var in ("prcp", "tmin", "tmax", "srad", "vp", "dayl"):
            f = tmpd / f"{gauge}_{var}.nc4"
            fetch(var, YEAR, i0, i1, j0, j1, f)
            with xr.open_dataset(f) as ds:
                crs = CRS.from_cf(ds["lambert_conformal_conic"].attrs)
                gdf = row.to_crs(crs)
                xs, ys = ds["x"].values, ds["y"].values
                rs = NumPyRasterSource(np.zeros((len(ys), len(xs))),
                                       xs[0] - RES / 2, ys[-1] - RES / 2,
                                       xs[-1] + RES / 2, ys[0] + RES / 2,
                                       srs_wkt=crs.to_wkt())
                w = exact_extract(rs, gdf, ["cell_id", "coverage"], output="pandas")
                cid = np.asarray(w["cell_id"].iloc[0], int)
                cov = np.asarray(w["coverage"].iloc[0], float)
                a = ds[var].values.astype("float64")
                vals = (a.reshape(a.shape[0], len(ys) * len(xs))[:, cid] * cov).sum(1) / cov.sum()
                days = [str(np.datetime_as_string(t, unit="D")) for t in ds["time"].values]
            mine[var] = dict(zip(days, vals))

        # frozen. NOTE: srad needs dayl to convert W/m2 -> MJ/m2/day, so compare
        # the CORPUS column (already MJ) against our corpus-space value.
        # srad must be compared in CORPUS space (MJ/m2/day = srad * dayl / 1e6)
        mine["sradmj"] = {d: mine["srad"][d] * mine["dayl"][d] / 1e6
                          for d in mine["srad"] if d in mine["dayl"]}
        COLMAP = {"prcp": "precipitation_sum", "tmin": "temperature_2m_min",
                  "tmax": "temperature_2m_max", "vp": "vapor_pressure",
                  "sradmj": "shortwave_radiation_sum"}
        fr = {k: {} for k in COLMAP}
        with gzip.open(FROZEN / f"{gauge}.csv.gz", "rt") as fh:
            hdr = fh.readline().strip().split(","); ci = {c: i for i, c in enumerate(hdr)}
            for line in fh:
                f_ = line.strip().split(",")
                if not f_[0].startswith(str(YEAR)):
                    continue
                for k, col in COLMAP.items():
                    try:
                        fr[k][f_[0]] = float(f_[ci[col]])
                    except (ValueError, KeyError):
                        pass

        print(f"\n=== {gauge}  {YEAR} ===")
        dates = sorted(mine["prcp"])
        import datetime as dt
        for var in ("prcp", "tmin", "tmax", "vp", "sradmj"):
            out = []
            for lag in (-3, -2, -1, 0, 1, 2, 3):
                A, B = [], []
                for d in dates:
                    dd = (dt.date.fromisoformat(d) + dt.timedelta(days=lag)).isoformat()
                    if dd in fr[var]:
                        A.append(mine[var][d]); B.append(fr[var][dd])
                out.append((lag, corr(A, B), len(A)))
            best = max((o for o in out if not math.isnan(o[1])), key=lambda o: o[1])
            s = "  ".join(f"lag{l:+d}={c:.3f}" for l, c, _ in out)
            print(f"  {var:<5} {s}")
            print(f"        -> best lag {best[0]:+d} (r={best[1]:.3f})"
                  + ("   <-- LAG 0, alignment OK" if best[0] == 0
                     else "   <-- ⚠️ MISALIGNMENT SUSPECTED"))


if __name__ == "__main__":
    main()
