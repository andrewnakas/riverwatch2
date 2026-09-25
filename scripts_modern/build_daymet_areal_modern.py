#!/usr/bin/env python
"""Build the modern (2015-2025) CAMELS corpus with AREAL Daymet V4R1 basin means.

Replaces `camels_corpus_daymet_modern`, which was built from the ORNL
single-pixel API at the gauge coordinate and is therefore point-sampled, while
the training corpus is CAMELS basin_mean_forcing (areal). Measured consequence
of that mismatch on the same Daymet version and days: precipitation reads
-28% to -35% in steep basins and temperature up to +6 C too warm.

Route (verified): NASA Hyrax DAP4 constraint-expression subsets fetched with
curl using the existing ~/.netrc Earthdata login, then opened locally. The
netCDF4 `dap4://` client segfaults and DAP2 500s on year-long slices, so
curl-then-open is the only route that works.

Output schema matches the frozen corpus exactly:
  date, q_cfs, temperature_2m_mean, temperature_2m_max, temperature_2m_min,
  precipitation_sum, shortwave_radiation_sum, vapor_pressure
"""
from __future__ import annotations
import argparse, gzip, json, math, os, subprocess, sys, tempfile, threading, time
import datetime as dt
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
POINT = ROOT / "data/modern_corpus/q_store"   # MODERN-1: authoritative USGS DV,
# 1980->today. Was camels_corpus_daymet_modern, the old POINT corpus, which begins
# in 2015 -- which is why the shipped 2014 prefix has forcings but ZERO discharge on
# every basin, and why a 1980-2013 rebuild would have written 34 years of untargeted
# forcings and exited 0. See scripts_modern/build_q_store.py.
BASE = ("https://opendap.earthdata.nasa.gov/collections/C2532426483-ORNL_CLOUD/"
        "granules/Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{year}.nc")
LCC = ("+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 "
       "+ellps=WGS84 +units=m +no_defs")
X0, Y0, RES, NX, NY = -4_560_250.0, 4_984_000.0, 1000.0, 7814, 8075

VARS = ["prcp", "tmax", "tmin", "srad", "vp", "dayl"]
COLS = ["date", "q_cfs", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum",
        "vapor_pressure"]

_print_lock = threading.Lock()
# libhdf5/netCDF4 is NOT thread-safe: concurrent xr.open_dataset() segfaults the
# interpreter (measured, 10 workers). Downloads are curl subprocesses and stay
# parallel; only the decode is serialized, which costs ~0.1 s per file.
NC_LOCK = threading.Lock()


def log(msg):
    with _print_lock:
        print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def calendar_days(y0, y1):
    d, end, out = dt.date(y0, 1, 1), dt.date(y1, 12, 31), []
    while d <= end:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def fetch(var, year, i0, i1, j0, j1, out, cookies, tries=5):
    """DAP4 constraint subset via curl. Verifies the ARTIFACT, not the exit code."""
    ndays = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    tmax_idx = ndays - 2 if ndays == 366 else 364   # Daymet: 365-day calendar
    ce = (f"/x%5B{i0}:{i1}%5D;/y%5B{j0}:{j1}%5D;/time%5B0:{tmax_idx}%5D;"
          f"/lambert_conformal_conic;"
          f"/{var}%5B0:{tmax_idx}%5D%5B{j0}:{j1}%5D%5B{i0}:{i1}%5D")
    url = BASE.format(var=var, year=year) + ".dap.nc4?dap4.ce=" + ce
    # ⭐ RESUME: a var-year already on disk and decodable is done. Without this a
    # basin that fails at fetch 150 of 204 re-downloaded the 149 that had already
    # succeeded. Measured 2026-09-23 during a provider degradation: 50.6 s per
    # fetch and a 78% basin failure rate, i.e. almost all work was being thrown
    # away and redone.
    if out.exists() and out.stat().st_size > 10_000:
        try:
            with NC_LOCK:
                with xr.open_dataset(out) as ds:
                    if var in ds and ds[var].shape[0] >= 365:
                        return True
        except Exception:
            out.unlink(missing_ok=True)
    last = ""
    for k in range(tries):
        r = subprocess.run(["curl", "-g", "-s", "-n", "-L",
                            "-c", cookies, "-b", cookies,
                            "--max-time", "600", "-o", str(out), url],
                           capture_output=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 10_000:
            try:
                with NC_LOCK:
                    with xr.open_dataset(out) as ds:  # must actually decode
                        if var in ds and ds[var].shape[0] >= 365:
                            return True
                last = "opened but wrong contents"
            except Exception as e:
                last = f"unopenable: {e}"
        else:
            last = (f"rc={r.returncode} "
                    f"size={out.stat().st_size if out.exists() else 0}")
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"{var} {year}: {last}")


def basin_idx(geom, pad=2):
    minx, miny, maxx, maxy = geom.bounds
    return (max(0, int((minx - X0) / RES) - pad), min(NX - 1, int((maxx - X0) / RES) + pad),
            max(0, int((Y0 - maxy) / RES) - pad), min(NY - 1, int((Y0 - miny) / RES) + pad))


def load_q(gauge):
    """Reuse the already-fetched USGS discharge, MASKING the -999999 sentinels."""
    p = POINT / f"{gauge}.csv.gz"
    if not p.exists():
        return {}
    out = {}
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        for line in fh:
            f = line.rstrip("\n").split(",")
            try:
                v = float(f[ci["q_cfs"]])
            except (ValueError, IndexError):
                continue
            if v < 0 or not math.isfinite(v):
                continue          # -999999 = USGS NoData; mask, never zero-fill
            out[f[ci["date"]]] = v
    return out


def build_basin(gauge, row, y0, y1, outdir, workdir):
    op = outdir / f"{gauge}.csv.gz"
    want = calendar_days(y0, y1)
    if op.exists():
        try:
            with gzip.open(op, "rt") as fh:
                n = sum(1 for _ in fh) - 1
            if n == len(want):
                return ("skip", gauge, n)
        except Exception:
            pass

    geom = row.to_crs(LCC).geometry.iloc[0]
    i0, i1, j0, j1 = basin_idx(geom)
    cookies = str(workdir / f".cook_{gauge}")
    weights = crs = None
    series = {v: {} for v in VARS}

    for year in range(y0, y1 + 1):
        for var in VARS:
            f = workdir / f"{gauge}_{var}_{year}.nc4"
            fetch(var, year, i0, i1, j0, j1, f, cookies)
            with NC_LOCK, xr.open_dataset(f) as ds:
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
                    if cov.sum() <= 0:
                        raise RuntimeError("zero polygon coverage")
                    weights = (cid, cov, len(ys), len(xs))
                cid, cov, ny_, nx_ = weights
                a = ds[var].values.astype("float64")
                vals = (a.reshape(a.shape[0], ny_ * nx_)[:, cid] * cov).sum(1) / cov.sum()
                days = [str(np.datetime_as_string(t, unit="D"))
                        for t in ds["time"].values]
            for d, v in zip(days, vals):
                series[var][d] = v
            # NOT unlinked here any more: kept in the persistent cache so a basin
            # that fails later in its 204-fetch sequence resumes instead of
            # restarting. Cleaned only once the basin's CSV is written.

    # Daymet uses a 365-day calendar: leap years end Dec 30. Fill Dec 31 from
    # Dec 30 so the corpus is a real calendar and joins by date cannot slip.
    filled = 0
    for year in range(y0, y1 + 1):
        d31, d30 = f"{year}-12-31", f"{year}-12-30"
        if d31 in want and d31 not in series["prcp"] and d30 in series["prcp"]:
            for var in VARS:
                series[var][d31] = series[var][d30]
            filled += 1

    q = load_q(gauge)
    rows, nan_days = [], 0
    for d in want:
        try:
            tmax = series["tmax"][d]; tmin = series["tmin"][d]
            prcp = series["prcp"][d]; srad = series["srad"][d]
            vp = series["vp"][d];     dayl = series["dayl"][d]
        except KeyError:
            nan_days += 1
            continue
        if not all(math.isfinite(x) for x in (tmax, tmin, prcp, srad, vp, dayl)):
            nan_days += 1
            continue
        rows.append([
            d,
            q.get(d, ""),                                   # blank = masked
            round((tmax + tmin) / 2, 3), round(tmax, 3), round(tmin, 3),
            round(prcp, 4),
            round(srad * dayl / 1e6, 6),                    # W/m2 * s -> MJ/m2/day
            round(vp, 2),
        ])

    if nan_days:
        raise RuntimeError(f"{nan_days} days missing/non-finite - refusing to write")

    tmp = op.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt") as fh:
        fh.write(",".join(COLS) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    with gzip.open(tmp, "rt") as fh:                        # verify the artifact
        assert sum(1 for _ in fh) - 1 == len(want)
    tmp.replace(op)
    for _c in workdir.glob(f"{gauge}_*.nc4"):      # basin done: release its cache
        _c.unlink(missing_ok=True)
    return ("ok", gauge, f"{len(rows)}d filled={filled} qdays={len(q)}")


def assert_q_store_covers(y0: int, y1: int, store: Path, n_expected: int = 0,
                          sample: int = 12) -> None:
    """MODERN-1 GUARD. The defect this exists for: the discharge reader used to point
    at a corpus that began in 2015, so a 1980-2013 build wrote forcings with a blank
    q_cfs on every row and still exited 0. Verify the store spans the build, on real
    files, before spending a day of network."""
    if not store.exists():
        raise SystemExit(f"GUARD FAIL: no discharge store at {store} "
                         f"- run scripts_modern/build_q_store.py first")
    files = sorted(store.glob("*.csv.gz"))[:sample]
    if not files:
        raise SystemExit(f"GUARD FAIL: discharge store {store} is empty")
    need_lo, need_hi = f"{y0}-01-01", f"{y1}-12-31"
    for p in files:
        with gzip.open(p, "rt") as fh:
            fh.readline()
            first = fh.readline().split(",")[0]
            last = first
            for line in fh:
                last = line.split(",")[0]
        if first > need_lo or last < need_hi:
            raise SystemExit(f"GUARD FAIL: {p.name} covers {first}..{last}, "
                             f"build needs {need_lo}..{need_hi}")
    n_have = sum(1 for _ in store.glob("*.csv.gz"))
    if n_expected and n_have < n_expected:
        raise SystemExit(f"GUARD FAIL: discharge store has {n_have} basins, "
                         f"build needs {n_expected} - it is still filling")
    print(f"GUARD OK: discharge store covers {need_lo}..{need_hi}, "
          f"{n_have} basins ({len(files)} spans checked)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/modern_corpus/camels_corpus_daymet_modern_areal")
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--basins", default="", help="comma-separated ids (smoke test)")
    ap.add_argument("--cache-dir", default="data/modern_corpus/_daymet_cache",
                    help="persistent var-year download cache; a basin that fails "
                         "part-way resumes from here instead of refetching")
    args = ap.parse_args()

    assert_q_store_covers(args.start_year, args.end_year, POINT,
                          n_expected=0 if (args.basins or args.limit) else 671)
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    # PERSISTENT across runs, so partial basins survive a restart. Was a
    # per-run mkdtemp, which threw away every cached var-year on every relaunch.
    workdir = ROOT / args.cache_dir
    workdir.mkdir(parents=True, exist_ok=True)

    ids = json.load(open(ROOT / "data/camels_gauge_ids.json"))
    b671, b531 = list(ids["671"]), set(ids["531"])
    # 531 eval basins first, so a partial run is still a usable cohort
    order = [g for g in b671 if g in b531] + [g for g in b671 if g not in b531]
    if args.basins:
        order = [g.strip() for g in args.basins.split(",")]
    if args.limit:
        order = order[:args.limit]

    g = gpd.read_file(SHP)
    g["gid"] = g["GAGE_ID"].astype(str).str.zfill(8)
    have = set(g["gid"])
    missing = [x for x in order if x not in have]
    if missing:
        log(f"WARNING {len(missing)} basins absent from shapefile: {missing[:5]}")
    order = [x for x in order if x in have]

    log(f"building {len(order)} basins, {args.start_year}-{args.end_year}, "
        f"{args.workers} workers -> {outdir}")
    log(f"expected days per basin: {len(calendar_days(args.start_year, args.end_year))}")

    done = skipped = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_basin, gid, g[g["gid"] == gid],
                          args.start_year, args.end_year, outdir, workdir): gid
                for gid in order}
        for i, fut in enumerate(as_completed(futs), 1):
            gid = futs[fut]
            try:
                status, _, info = fut.result()
                if status == "skip":
                    skipped += 1
                else:
                    done += 1
                    log(f"[{i}/{len(order)}] {gid} OK {info}")
            except Exception as e:
                failed += 1
                log(f"[{i}/{len(order)}] {gid} FAIL {e}")
            if i % 25 == 0:
                el = time.time() - t0
                log(f"--- progress {i}/{len(order)} done={done} skip={skipped} "
                    f"fail={failed} elapsed={el/3600:.2f}h "
                    f"eta={el/max(i,1)*(len(order)-i)/3600:.2f}h")
    log(f"FINISHED done={done} skipped={skipped} failed={failed} -> {outdir}")


if __name__ == "__main__":
    raise SystemExit(main())
