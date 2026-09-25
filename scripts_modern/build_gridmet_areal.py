#!/usr/bin/env python
"""MODERN-1: build the areal gridMET corpus, 1979 -> present.

gridMET is the PRISM-anchored fourth product. Its precipitation IS PRISM
regridded to 4 km, which matters because gauge-calibrated precipitation is the
only property that has ever predicted member skill in this campaign. It is
served as annual netCDF (~160 MiB/var/year, UInt16 + scale_factor), so the whole
1979-present span is ~37 GB -- unlike raw PRISM, whose web service is capped at
about two files per minute and would need 500+ hours for the same years.

Output schema is byte-compatible with the frozen corpora:
  date, q_cfs, temperature_2m_mean, temperature_2m_max, temperature_2m_min,
  precipitation_sum, shortwave_radiation_sum, vapor_pressure

DERIVATIONS
  precipitation_sum        pr                       mm
  temperature_2m_max/min   tmmx/tmmn - 273.15       degC   (K in source)
  temperature_2m_mean      (tmax + tmin) / 2        degC   (frozen convention)
  shortwave_radiation_sum  srad * 86400 / 1e6       MJ/m2/day
                           (srad is a 24 h MEAN flux, unlike Daymet's daylight-
                            period flux, which is why this is not srad*dayl)
  vapor_pressure           (es_mean - vpd) * 1000   Pa
                           es(T) = 0.6108*exp(17.27T/(T+237.3)) kPa,
                           es_mean = (es(tmax)+es(tmin))/2  -- FAO-56, which is
                           what gridMET itself uses to form vpd, so this inverts
                           their own calculation rather than introducing a second
                           convention. Needs no surface-pressure field, which
                           gridMET does not ship.

TRAPS THIS CODE HANDLES, because each one has already cost this project time:
  * LATITUDE ORDER IS DETECTED, NEVER ASSUMED. NLDAS stores latitude ascending
    while the weights were built north-first, and getting that wrong silently
    gives every basin the wrong cells.
  * libhdf5/netCDF4 IS NOT THREAD-SAFE: decode is serialised under NC_LOCK while
    downloads stay parallel. Ten concurrent xr.open_dataset() segfaulted the
    interpreter outright.
  * The artifact is verified (row count, finiteness), not the exit code.
"""
from __future__ import annotations
import argparse, gzip, math, os, pickle, subprocess, sys, tempfile, threading, time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import geopandas as gpd
import xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

ROOT = Path(__file__).resolve().parents[1]
SHP = ROOT / "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"
BASE = "https://www.northwestknowledge.net/metdata/data/{var}_{year}.nc"
VARS = ["pr", "tmmx", "tmmn", "srad", "vpd"]
NCVAR = {"pr": "precipitation_amount",
         "tmmx": "daily_maximum_temperature",
         "tmmn": "daily_minimum_temperature",
         "srad": "daily_mean_shortwave_radiation_at_surface",
         "vpd": "daily_mean_vapor_pressure_deficit"}
COLS = ["date", "q_cfs", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum",
        "vapor_pressure"]

NC_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()
TIME_CHUNK = 30          # days per decode slice: 30*585*1386*4 B ~ 97 MiB


def log(*a):
    with PRINT_LOCK:
        print(*a, flush=True)


def fetch(var: str, year: int, dest: Path, tries: int = 6) -> None:
    """Resumable, with exponential backoff.

    ⚠️ The original gave up after 4 short tries and DELETED the partial file, and
    the caller let the exception kill the whole run: a single transient hiccup on
    `srad_1985.nc` aborted a 47-year build at year 15, and the file was provably
    fine seconds later (HEAD 200, range GET returned valid HDF5). Over a home
    connection pulling ~37 GB, transient failures are the norm, not the exception.
    The server supports ranges (206), so partials are RESUMED, not discarded."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    url = BASE.format(var=var, year=year)
    tmp = dest.with_suffix(".part")
    for k in range(tries):
        rc = subprocess.run(["curl", "-sSL", "--fail", "-C", "-",
                             "--max-time", "3600", "-o", str(tmp), url]).returncode
        if rc == 0 and tmp.exists() and tmp.stat().st_size > 1_000_000:
            tmp.replace(dest)
            return
        # keep the partial so the next attempt resumes; only drop it if unusable
        if tmp.exists() and tmp.stat().st_size < 1024:
            tmp.unlink(missing_ok=True)
        time.sleep(min(300, 10 * (2 ** k)))
    raise RuntimeError(f"download failed after {tries} tries: {url}")


def build_weights(sample_nc: Path, gauges: list[str], wpath: Path) -> dict:
    """Coverage-weighted cell ids per basin, computed once over the gridMET grid."""
    if wpath.exists():
        with open(wpath, "rb") as fh:
            return pickle.load(fh)
    with NC_LOCK, xr.open_dataset(sample_nc) as ds:
        lat = ds["lat"].values.astype(float)
        lon = ds["lon"].values.astype(float)
    # DETECTED, not assumed: exactextract wants a north-up raster.
    lat_desc = bool(lat[0] > lat[-1])
    latn = lat if lat_desc else lat[::-1]
    dy = abs(float(latn[1] - latn[0]))
    dx = abs(float(lon[1] - lon[0]))
    ny, nx = len(latn), len(lon)
    rs = NumPyRasterSource(np.zeros((ny, nx)),
                           lon[0] - dx / 2, latn[-1] - dy / 2,
                           lon[-1] + dx / 2, latn[0] + dy / 2,
                           srs_wkt='EPSG:4326')

    gdf = gpd.read_file(SHP)
    idcol = "GAGE_ID" if "GAGE_ID" in gdf.columns else gdf.columns[0]
    gdf[idcol] = gdf[idcol].astype(str).str.zfill(8)
    gdf = gdf[gdf[idcol].isin(set(gauges))].to_crs("EPSG:4326")

    out: dict = {"_lat_desc": lat_desc, "_shape": (ny, nx)}
    w = exact_extract(rs, gdf, ["cell_id", "coverage"], output="pandas",
                      include_cols=[idcol])
    for _, r in w.iterrows():
        cid = np.asarray(r["cell_id"], dtype=np.int64)
        cov = np.asarray(r["coverage"], dtype=np.float64)
        if cov.sum() <= 0:
            continue
        out[str(r[idcol]).zfill(8)] = (cid, cov / cov.sum())
    log(f"weights: {len(out) - 2} basins over a {ny}x{nx} grid "
        f"(lat_descending={lat_desc})")
    with open(wpath, "wb") as fh:
        pickle.dump(out, fh)
    return out


def reduce_year(var: str, year: int, nc: Path, gauges: list[str],
                W: dict) -> tuple[np.ndarray, list[str]]:
    """-> (n_basins, n_days) areal means, and the day strings."""
    lat_desc, (ny, nx) = W["_lat_desc"], W["_shape"]
    with NC_LOCK, xr.open_dataset(nc) as ds:
        name = NCVAR[var] if NCVAR[var] in ds else [v for v in ds.data_vars
                                                    if ds[v].ndim == 3][0]
        days = [str(np.datetime_as_string(t, unit="D")) for t in ds["day"].values]
        nt = len(days)
        out = np.full((len(gauges), nt), np.nan, dtype=np.float64)
        for a in range(0, nt, TIME_CHUNK):
            b = min(a + TIME_CHUNK, nt)
            arr = ds[name][a:b].values.astype("float64")
            if not lat_desc:                       # flip to north-up, as the weights are
                arr = arr[:, ::-1, :]
            flat = arr.reshape(b - a, ny * nx)
            for gi, g in enumerate(gauges):
                wg = W.get(g)
                if wg is None:
                    continue
                cid, cov = wg
                out[gi, a:b] = flat[:, cid] @ cov
            del arr, flat
    return out, days


def es_kpa(t_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=1980)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--out", default="data/modern_corpus/v3/camels_corpus_gridmet_m3")
    ap.add_argument("--work", default="data/modern_corpus/_gridmet_years")
    ap.add_argument("--qstore", default="data/modern_corpus/q_store")
    ap.add_argument("--ids", default="data/camels_gauge_ids.json")
    ap.add_argument("--key", default="671")
    ap.add_argument("--basins", default="", help="comma-separated (pilot/screen)")
    ap.add_argument("--workers", type=int, default=4, help="parallel DOWNLOADS")
    ap.add_argument("--keep-nc", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--allow-gaps", action="store_true",
                    help="write even if the day calendar has holes "
                         "(never for a corpus that will be trained on)")
    ap.add_argument("--shift-days", type=int, default=0,
                    help="shift the whole series by N days to land on the CAMELS "
                         "k=0 convention. Gate 1 measured gridMET at -1 on ALL six "
                         "variables (prcp r 0.835 @ lag-1 vs 0.542 @ lag 0), i.e. a "
                         "time-axis labelling difference, so +1 is the correction. "
                         "Set from the streamflow referee, never assumed.")
    a = ap.parse_args()

    os.chdir(ROOT)
    import json
    gauges = [str(i).zfill(8) for i in json.loads(Path(a.ids).read_text())[a.key]]
    if a.basins:
        want = {g.zfill(8) for g in a.basins.split(",")}
        gauges = [g for g in gauges if g in want]
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    years = list(range(a.start_year, a.end_year + 1))

    wpath = work / "gridmet_basin_weights.pkl"
    if not a.assemble_only:
        failed_years = []
        for y in years:
            ypath = work / f"gridmet_{y}.npz"
            if ypath.exists():
                log(f"year {y}: skip (have {ypath.name})")
                continue
            try:
                ncs = {}
                with ThreadPoolExecutor(max_workers=a.workers) as ex:
                    futs = {ex.submit(fetch, v, y, work / f"{v}_{y}.nc"): v
                            for v in VARS}
                    for fu in as_completed(futs):
                        fu.result()
                        ncs[futs[fu]] = work / f"{futs[fu]}_{y}.nc"
                W = build_weights(ncs["pr"], gauges, wpath)
                fields, days = {}, None
                for v in VARS:
                    arr, dd = reduce_year(v, y, ncs[v], gauges, W)
                    fields[v] = arr
                    days = dd
                    log(f"year {y} {v}: {arr.shape} nan={np.isnan(arr).mean():.4f}")
                np.savez_compressed(ypath, days=np.array(days),
                                    gauges=np.array(gauges), **fields)
                if not a.keep_nc:
                    for v in VARS:
                        ncs[v].unlink(missing_ok=True)
                log(f"year {y}: WROTE {ypath.name}")
            except Exception as e:
                # ⚠️ One bad year must not kill the other 46. A single transient
                # failure on srad_1985.nc aborted this build at year 15 of 47,
                # and the file was provably fine seconds later. Failed years are
                # listed and filled by re-running; the loop never aborts.
                log(f"year {y}: FAILED ({type(e).__name__}: {str(e)[:140]})"
                    f" - continuing with the remaining years")
                failed_years.append(y)
        if failed_years:
            log(f"YEARS FAILED, re-run to fill: {failed_years}")

    # ---- assemble per-basin csv.gz -----------------------------------------
    # Array-based, not a dict of dicts: at 47 years a per-day dict is ~11.4 M
    # entries and would contend for RAM with training on the same box. As
    # (n_basins, n_days) float32 the whole corpus is ~46 MiB per variable.
    gi_of = {g: i for i, g in enumerate(gauges)}
    day_list: list[str] = []
    blocks: list[tuple[Path, list[str], list[str]]] = []
    for y in years:
        ypath = work / f"gridmet_{y}.npz"
        if not ypath.exists():
            log(f"assemble: MISSING {ypath.name}, skipping")
            continue
        z = np.load(ypath, allow_pickle=True)
        days = [str(d) for d in z["days"]]
        if a.shift_days:
            days = [(dt.date.fromisoformat(d) + dt.timedelta(days=a.shift_days)).isoformat()
                    for d in days]
        blocks.append((ypath, days, [str(g) for g in z["gauges"]]))
        day_list.extend(days)
        z.close()
    if not day_list:
        log("assemble: nothing to assemble"); return 1
    # ⚠️ CONTIGUITY GATE. Without this the assemble happily wrote a corpus
    # spanning 1980-01-01..2026-09-22 with ZERO rows in 2017 and 2018 -- two
    # failed download years, skipped by the resilient year loop -- and reported
    # "ok=671 bad=0 FINISHED" with exit 0. The hole sat inside P-MODERN's
    # validation window. A gapped corpus must not be written at all: a partial
    # artifact that looks complete is worse than no artifact.
    _d0 = dt.date.fromisoformat(day_list[0])
    _d1 = dt.date.fromisoformat(day_list[-1])
    _want = (_d1 - _d0).days + 1
    if len(set(day_list)) != _want:
        _have = {dt.date.fromisoformat(x) for x in day_list}
        _missing = sorted({(_d0 + dt.timedelta(days=k)) for k in range(_want)} - _have)
        _years = sorted({m.year for m in _missing})
        log(f"CONTIGUITY FAIL: {len(_missing)} of {_want} days missing "
            f"({day_list[0]}..{day_list[-1]}), years affected: {_years}")
        log("REFUSING to write a gapped corpus. Re-run to fill those years, "
            "or pass --allow-gaps if a partial corpus is genuinely wanted.")
        if not getattr(a, "allow_gaps", False):
            return 1
        log("--allow-gaps given: writing anyway, DO NOT TRAIN ON THIS")
    di_of = {d: i for i, d in enumerate(day_list)}
    nD, nG = len(day_list), len(gauges)
    FIELDS = ("tmax", "tmin", "pr", "srad", "vp")
    A = {k: np.full((nG, nD), np.nan, dtype=np.float32) for k in FIELDS}
    for ypath, days, gg in blocks:
        z = np.load(ypath, allow_pickle=True)
        tmax = z["tmmx"] - 273.15
        tmin = z["tmmn"] - 273.15
        vpv = (es_kpa(tmax) + es_kpa(tmin)) / 2.0 - z["vpd"]
        src = {"tmax": tmax, "tmin": tmin, "pr": z["pr"], "srad": z["srad"], "vp": vpv}
        rows = np.array([gi_of.get(g, -1) for g in gg])
        cols = np.array([di_of[d] for d in days])
        keep = rows >= 0
        for k in FIELDS:
            A[k][np.ix_(rows[keep], cols)] = src[k][keep]
        z.close()
        del src, tmax, tmin, vpv

    nok = nbad = 0
    for g in gauges:
        gi = gi_of[g]
        finite = np.isfinite(A["pr"][gi])
        if not finite.any():
            nbad += 1
            continue
        q = {}
        qp = Path(a.qstore) / f"{g}.csv.gz"
        if qp.exists():
            with gzip.open(qp, "rt") as fh:
                fh.readline()
                for ln in fh:
                    d, v = ln.rstrip("\n").split(",")
                    if v not in ("", "nan"):
                        q[d] = v
        ok = finite & np.isfinite(A["tmax"][gi]) & np.isfinite(A["tmin"][gi]) \
            & np.isfinite(A["srad"][gi]) & np.isfinite(A["vp"][gi])
        idxs = np.flatnonzero(ok)
        if idxs.size != np.flatnonzero(finite).size:
            log(f"{g}: {np.flatnonzero(finite).size - idxs.size} partially non-finite "
                f"days - REFUSING")
            nbad += 1
            continue
        rows = []
        for j in idxs:
            d = day_list[j]
            tx = float(A["tmax"][gi, j]); tn = float(A["tmin"][gi, j])
            rows.append([d, q.get(d, ""), round((tx + tn) / 2, 3), round(tx, 3),
                         round(tn, 3), round(float(A["pr"][gi, j]), 4),
                         round(float(A["srad"][gi, j]) * 86400 / 1e6, 6),
                         round(float(A["vp"][gi, j]) * 1000, 2)])
        op = outdir / f"{g}.csv.gz"
        tmp = op.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wt") as fh:
            fh.write(",".join(COLS) + "\n")
            for r in rows:
                fh.write(",".join(str(x) for x in r) + "\n")
        with gzip.open(tmp, "rt") as fh:
            assert sum(1 for _ in fh) - 1 == len(rows)
        tmp.replace(op)
        nok += 1
    (outdir / "_BUILD_SIDECAR.json").write_text(json.dumps({
        "product": "gridmet", "shift_days": a.shift_days,
        "shift_rationale": "Gate 1 lag scan + streamflow referee; CAMELS k=0 convention",
        "years": [a.start_year, a.end_year], "n_basins": nok,
        "qstore": a.qstore}, indent=1))
    log(f"ASSEMBLE ok={nok} bad={nbad} shift_days={a.shift_days} -> {outdir}")
    log("FINISHED")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
