#!/usr/bin/env python
"""Add areal `swe` and `swe_delta` columns to the modern Daymet corpus.

multi5 — the only ensemble member that ever shipped — reads swe/swe_delta, and
the frozen `camels_corpus_daymet_swe_v2` carries them. Daymet V4R1 publishes
`swe`, so the modern corpus can support a snow member too.

⚠️ SWE is the variable point sampling damages WORST (measured on the repo's own
CONUS404 point/areal pair: −19% median, low in 82% of basins, worst daily
r = 0.0004), so it must come from this areal pipeline, never the single-pixel API.

Appends in place, atomically per basin, and is resumable: a basin whose header
already carries `swe` is skipped. Same Hyrax DAP4 + exactextract machinery as
`build_daymet_areal_modern.py`.
"""
from __future__ import annotations
import argparse, datetime as dt, gzip, json, math, subprocess, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
from pyproj import CRS
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

ROOT = Path("/home/nakas/riverwatch2")
SHP = ROOT / "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"
BASE = ("https://opendap.earthdata.nasa.gov/collections/C2532426483-ORNL_CLOUD/"
        "granules/Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{year}.nc")
LCC = ("+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 "
       "+ellps=WGS84 +units=m +no_defs")
X0, Y0, RES, NX, NY = -4_560_250.0, 4_984_000.0, 1000.0, 7814, 8075

_plock = threading.Lock()
NC_LOCK = threading.Lock()          # libhdf5 is not thread-safe (measured: segfault)


def log(m):
    with _plock:
        print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)


def fetch(year, i0, i1, j0, j1, out, cookies, tries=3):
    nd = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    tmax_idx = nd - 2 if nd == 366 else 364
    ce = (f"/x%5B{i0}:{i1}%5D;/y%5B{j0}:{j1}%5D;/time%5B0:{tmax_idx}%5D;"
          f"/lambert_conformal_conic;"
          f"/swe%5B0:{tmax_idx}%5D%5B{j0}:{j1}%5D%5B{i0}:{i1}%5D")
    url = BASE.format(var="swe", year=year) + ".dap.nc4?dap4.ce=" + ce
    last = ""
    for k in range(tries):
        r = subprocess.run(["curl", "-g", "-s", "-n", "-L", "-c", cookies,
                            "-b", cookies, "--max-time", "600", "-o", str(out), url],
                           capture_output=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 10_000:
            try:
                with NC_LOCK, xr.open_dataset(out) as ds:
                    if "swe" in ds and ds["swe"].shape[0] >= 365:
                        return True
                last = "opened but wrong contents"
            except Exception as e:
                last = f"unopenable: {e}"
        else:
            last = f"rc={r.returncode} size={out.stat().st_size if out.exists() else 0}"
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"swe {year}: {last}")


def idx(geom, pad=2):
    mnx, mny, mxx, mxy = geom.bounds
    return (max(0, int((mnx - X0) / RES) - pad), min(NX - 1, int((mxx - X0) / RES) + pad),
            max(0, int((Y0 - mxy) / RES) - pad), min(NY - 1, int((Y0 - mny) / RES) + pad))


def do_basin(gauge, row, corpus_dir, y0, y1, workdir):
    cp = corpus_dir / f"{gauge}.csv.gz"
    if not cp.exists():
        return ("missing", gauge, "no corpus file")
    with gzip.open(cp, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
    if "swe" in hdr:
        return ("skip", gauge, "already has swe")

    geom = row.to_crs(LCC).geometry.iloc[0]
    i0, i1, j0, j1 = idx(geom)
    cookies = str(workdir / f".c_{gauge}")
    weights = None
    series = {}
    for year in range(y0, y1 + 1):
        f = workdir / f"{gauge}_swe_{year}.nc4"
        fetch(year, i0, i1, j0, j1, f, cookies)
        with NC_LOCK, xr.open_dataset(f) as ds:
            if weights is None:
                crs = CRS.from_cf(ds["lambert_conformal_conic"].attrs)
                gdf = row.to_crs(crs)
                xs, ys = ds["x"].values, ds["y"].values
                rs = NumPyRasterSource(np.zeros((len(ys), len(xs))),
                                       xs[0] - RES / 2, ys[-1] - RES / 2,
                                       xs[-1] + RES / 2, ys[0] + RES / 2,
                                       srs_wkt=crs.to_wkt())
                w = exact_extract(rs, gdf, ["cell_id", "coverage"], output="pandas")
                cid = np.asarray(w["cell_id"].iloc[0], dtype=int)
                cov = np.asarray(w["coverage"].iloc[0], dtype=float)
                if cov.sum() <= 0:
                    raise RuntimeError("zero coverage")
                weights = (cid, cov, len(ys), len(xs))
            cid, cov, ny_, nx_ = weights
            a = ds["swe"].values.astype("float64")
            vals = (a.reshape(a.shape[0], ny_ * nx_)[:, cid] * cov).sum(1) / cov.sum()
            days = [str(np.datetime_as_string(t, unit="D")) for t in ds["time"].values]
        series.update(dict(zip(days, vals)))
        f.unlink(missing_ok=True)

    # Daymet's 365-day calendar: leap years end Dec 30. Match the corpus builder.
    for year in range(y0, y1 + 1):
        d31, d30 = f"{year}-12-31", f"{year}-12-30"
        if d31 not in series and d30 in series:
            series[d31] = series[d30]

    with gzip.open(cp, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        rows = [l.rstrip("\n").split(",") for l in fh if l.strip()]

    out_rows, prev, missing = [], None, 0
    for r in rows:
        v = series.get(r[0])
        if v is None or not math.isfinite(v):
            missing += 1
            continue
        delta = 0.0 if prev is None else v - prev
        prev = v
        out_rows.append(r + [f"{v:.4f}", f"{delta:.4f}"])
    if missing:
        raise RuntimeError(f"{missing} days without swe - refusing to write")

    tmp = cp.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt") as fh:
        fh.write(",".join(hdr + ["swe", "swe_delta"]) + "\n")
        for r in out_rows:
            fh.write(",".join(r) + "\n")
    with gzip.open(tmp, "rt") as fh:                 # verify before replacing
        assert sum(1 for _ in fh) - 1 == len(rows)
    tmp.replace(cp)
    return ("ok", gauge, f"{len(out_rows)}d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/modern_corpus/camels_corpus_daymet_modern_areal")
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--basins", default="")
    a = ap.parse_args()

    cdir = ROOT / a.corpus
    workdir = Path(tempfile.mkdtemp(prefix="swe_", dir="/tmp"))
    ids = json.load(open(ROOT / "data/camels_gauge_ids.json"))
    b531 = set(ids["531"])
    order = [g for g in ids["671"] if g in b531] + [g for g in ids["671"] if g not in b531]
    if a.basins:
        order = [x.strip() for x in a.basins.split(",")]
    if a.limit:
        order = order[:a.limit]

    g = gpd.read_file(SHP)
    g["gid"] = g["GAGE_ID"].astype(str).str.zfill(8)
    have = set(g["gid"])
    order = [x for x in order if x in have]

    log(f"adding areal swe/swe_delta to {len(order)} basins, "
        f"{a.start_year}-{a.end_year}, {a.workers} workers")
    ok = skip = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(do_basin, gid, g[g["gid"] == gid], cdir,
                          a.start_year, a.end_year, workdir): gid for gid in order}
        for i, fu in enumerate(as_completed(futs), 1):
            gid = futs[fu]
            try:
                st, _, info = fu.result()
                if st == "ok":
                    ok += 1
                elif st == "skip":
                    skip += 1
                else:
                    fail += 1; log(f"{gid} {st}: {info}")
            except Exception as e:
                fail += 1; log(f"[{i}/{len(order)}] {gid} FAIL {e}")
            if i % 25 == 0:
                el = time.time() - t0
                log(f"--- progress {i}/{len(order)} ok={ok} skip={skip} fail={fail} "
                    f"elapsed={el/3600:.2f}h eta={el/max(i,1)*(len(order)-i)/3600:.2f}h")
    log(f"FINISHED swe: ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    raise SystemExit(main())
