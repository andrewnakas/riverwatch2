#!/usr/bin/env python
"""MODERN-1: build the areal nClimGrid-Daily corpus, 1980 -> present.

nClimGrid-Daily is NOAA NCEI's official 5 km (1/24 deg) gauge-based daily grid for
CONUS, 1951->present, interpolated from GHCN-Daily by NCEI's own method -- i.e.
independent of both Daymet's interpolation and of PRISM (which is what gridMET
carries). It replaces AORC, which was withdrawn on feasibility: AORC's zarr
chunking forces ~11 TB for 1980-2025, and LEDGER 51 had already measured AORC's
member contribution at LOO -0.000008, breadth 0.495 -- a coin flip.

Source: s3://noaa-nclimgrid-daily-pds/access/grids/{year}/ncdd-{YYYYMM}-grd-{kind}.nc
~57 MiB/month, so the whole span is ~33 GB.

  prcp  mm            -> precipitation_sum
  tmax  degC          -> temperature_2m_max
  tmin  degC          -> temperature_2m_min
        (tmax+tmin)/2 -> temperature_2m_mean   (the frozen corpus convention; the
                        file also ships `tavg`, deliberately not used, so that
                        every product in the corpus forms tmean the same way)

⚠️ HONEST LIMITATION, stated here and wherever this member's LOO is quoted:
nClimGrid ships NO shortwave radiation and NO humidity. `shortwave_radiation_sum`
and `vapor_pressure` are BORROWED from the NLDAS corpus (--borrow-from). This
member is therefore independent in 4 of its 6 channels, not 6 of 6 -- though the
4 that are independent are precipitation and temperature, the channels that carry
the signal.

⚠️ nClimGrid stores LATITUDE ASCENDING. That is detected at runtime, never
assumed: getting it wrong silently hands every basin the wrong cells, which is
exactly what NLDAS's north-first weights did.
"""
from __future__ import annotations
import argparse, calendar, gzip, json, math, os, pickle, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import datetime as dt
import numpy as np
import geopandas as gpd
import xarray as xr
from exactextract import exact_extract
from exactextract.raster import NumPyRasterSource

ROOT = Path(__file__).resolve().parents[1]
SHP = ROOT / "data/gages2/boundaries-shapefiles-by-aggeco/bas_ref_all.shp"
BASE = ("https://noaa-nclimgrid-daily-pds.s3.amazonaws.com/access/grids/"
        "{year}/ncdd-{ym}-grd-{kind}.nc")
VARS = ["prcp", "tmax", "tmin"]
COLS = ["date", "q_cfs", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum",
        "vapor_pressure"]

NC_LOCK = threading.Lock()        # libhdf5 is not thread-safe; decode serialised
PRINT_LOCK = threading.Lock()


def log(*a):
    with PRINT_LOCK:
        print(*a, flush=True)


def fetch_month(year: int, month: int, dest: Path, tries: int = 3) -> Path | None:
    """Recent months exist only as `prelim`; older ones as `scaled`."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    ym = f"{year}{month:02d}"
    for kind in ("scaled", "prelim"):
        url = BASE.format(year=year, ym=ym, kind=kind)
        for k in range(tries):
            tmp = dest.with_suffix(".part")
            rc = subprocess.run(["curl", "-sL", "--fail", "--max-time", "900",
                                 "-o", str(tmp), url]).returncode
            if rc == 0 and tmp.exists() and tmp.stat().st_size > 1_000_000:
                tmp.replace(dest)
                return dest
            tmp.unlink(missing_ok=True)
            if rc == 22:            # 404: wrong kind, try the other
                break
            time.sleep(4 * (k + 1))
    return None


def build_weights(sample_nc: Path, gauges: list[str], wpath: Path) -> dict:
    if wpath.exists():
        with open(wpath, "rb") as fh:
            return pickle.load(fh)
    with NC_LOCK, xr.open_dataset(sample_nc) as ds:
        lat = ds["lat"].values.astype(float)
        lon = ds["lon"].values.astype(float)
    lat_desc = bool(lat[0] > lat[-1])          # DETECTED, never assumed
    latn = lat if lat_desc else lat[::-1]
    dy = abs(float(latn[1] - latn[0])); dx = abs(float(lon[1] - lon[0]))
    ny, nx = len(latn), len(lon)
    rs = NumPyRasterSource(np.zeros((ny, nx)),
                           lon[0] - dx / 2, latn[-1] - dy / 2,
                           lon[-1] + dx / 2, latn[0] + dy / 2,
                           srs_wkt='EPSG:4326')
    gdf = gpd.read_file(SHP)
    idc = "GAGE_ID" if "GAGE_ID" in gdf.columns else gdf.columns[0]
    gdf[idc] = gdf[idc].astype(str).str.zfill(8)
    gdf = gdf[gdf[idc].isin(set(gauges))].to_crs("EPSG:4326")
    out: dict = {"_lat_desc": lat_desc, "_shape": (ny, nx)}
    w = exact_extract(rs, gdf, ["cell_id", "coverage"], output="pandas",
                      include_cols=[idc])
    for _, r in w.iterrows():
        cid = np.asarray(r["cell_id"], dtype=np.int64)
        cov = np.asarray(r["coverage"], dtype=np.float64)
        if cov.sum() > 0:
            out[str(r[idc]).zfill(8)] = (cid, cov / cov.sum())
    log(f"weights: {len(out) - 2} basins over {ny}x{nx} (lat_descending={lat_desc})")
    with open(wpath, "wb") as fh:
        pickle.dump(out, fh)
    return out


def reduce_month(nc: Path, gauges: list[str], W: dict):
    lat_desc, (ny, nx) = W["_lat_desc"], W["_shape"]
    with NC_LOCK, xr.open_dataset(nc) as ds:
        days = [str(np.datetime_as_string(t, unit="D")) for t in ds["time"].values]
        res = {}
        for v in VARS:
            arr = ds[v].values.astype("float64")
            if not lat_desc:
                arr = arr[:, ::-1, :]
            flat = arr.reshape(len(days), ny * nx)
            o = np.full((len(gauges), len(days)), np.nan)
            for gi, g in enumerate(gauges):
                wg = W.get(g)
                if wg is None:
                    continue
                cid, cov = wg
                sub = flat[:, cid]
                m = np.isfinite(sub)
                num = np.where(m, sub, 0.0) @ cov
                den = m.astype(float) @ cov
                o[gi] = np.where(den > 0.5, num / np.maximum(den, 1e-12), np.nan)
            res[v] = o
            del arr, flat
    return res, days


def read_col(path: Path, col: str) -> dict:
    out = {}
    if not path.exists():
        return out
    with gzip.open(path, "rt") as fh:
        h = fh.readline().rstrip("\n").split(",")
        if col not in h:
            return out
        i = h.index(col)
        for ln in fh:
            f = ln.rstrip("\n").split(",")
            s = f[i] if i < len(f) else ""
            if s not in ("", "nan"):
                out[f[0]] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=1980)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--out", default="data/modern_corpus/v3/camels_corpus_nclimgrid_m3")
    ap.add_argument("--work", default="data/modern_corpus/_nclimgrid_months")
    ap.add_argument("--qstore", default="data/modern_corpus/q_store")
    ap.add_argument("--borrow-from", default="data/modern_corpus/v3/camels_corpus_nldas_m3",
                    help="corpus supplying shortwave_radiation_sum and vapor_pressure")
    ap.add_argument("--ids", default="data/camels_gauge_ids.json")
    ap.add_argument("--key", default="671")
    ap.add_argument("--basins", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-nc", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--allow-gaps", action="store_true",
                    help="write even if the day calendar has holes "
                         "(never for a corpus that will be trained on)")
    a = ap.parse_args()

    os.chdir(ROOT)
    gauges = [str(i).zfill(8) for i in json.loads(Path(a.ids).read_text())[a.key]]
    if a.basins:
        want = {g.zfill(8) for g in a.basins.split(",")}
        gauges = [g for g in gauges if g in want]
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    wpath = work / "nclimgrid_basin_weights.pkl"
    months = [(y, m) for y in range(a.start_year, a.end_year + 1)
              for m in range(1, 13)]

    if not a.assemble_only:
        todo = [(y, m) for y, m in months if not (work / f"ncdd_{y}{m:02d}.npz").exists()]
        log(f"{len(months)} months, {len(todo)} to build, {a.workers} workers")
        for y, m in todo:
            nc = work / f"ncdd_{y}{m:02d}.nc"
            got = fetch_month(y, m, nc, )
            if got is None:
                log(f"{y}-{m:02d}: NOT PUBLISHED, stopping forward fill")
                continue
            W = build_weights(nc, gauges, wpath)
            res, days = reduce_month(nc, gauges, W)
            np.savez_compressed(work / f"ncdd_{y}{m:02d}.npz",
                                days=np.array(days), gauges=np.array(gauges), **res)
            if not a.keep_nc:
                nc.unlink(missing_ok=True)
            log(f"{y}-{m:02d}: ok  nan={np.isnan(res['prcp']).mean():.4f}")

    # ---- assemble ----------------------------------------------------------
    # Array-based, not a dict of dicts. The dict form is ~2.4 M entries over the
    # screen window alone and ~11 M over the full span; measured on the gridMET
    # twin, it ran 7+ hours and produced nothing, while the array form finishes
    # all 671 basins in minutes and holds the whole corpus in ~46 MiB/variable.
    gi_of = {g: i for i, g in enumerate(gauges)}
    day_list: list[str] = []
    blocks = []
    for y, m in months:
        zp = work / f"ncdd_{y}{m:02d}.npz"
        if not zp.exists():
            continue
        z = np.load(zp, allow_pickle=True)
        days = [str(d) for d in z["days"]]
        blocks.append((zp, days, [str(g) for g in z["gauges"]]))
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
    FIELDS = ("tmax", "tmin", "prcp")
    A = {k: np.full((len(gauges), len(day_list)), np.nan, dtype=np.float32)
         for k in FIELDS}
    for zp, days, gg in blocks:
        z = np.load(zp, allow_pickle=True)
        rows = np.array([gi_of.get(g, -1) for g in gg])
        cols = np.array([di_of[d] for d in days])
        keep = rows >= 0
        for k in FIELDS:
            A[k][np.ix_(rows[keep], cols)] = z[k][keep]
        z.close()

    borrow = Path(a.borrow_from)
    nok = nbad = nskip = 0
    for g in gauges:
        gi = gi_of[g]
        ok = (np.isfinite(A["tmax"][gi]) & np.isfinite(A["tmin"][gi])
              & np.isfinite(A["prcp"][gi]))
        if not ok.any():
            nbad += 1
            continue
        q = read_col(Path(a.qstore) / f"{g}.csv.gz", "q_cfs")
        sr = read_col(borrow / f"{g}.csv.gz", "shortwave_radiation_sum")
        vp = read_col(borrow / f"{g}.csv.gz", "vapor_pressure")
        if not sr or not vp:
            nskip += 1
            continue
        rows = []
        for j in np.flatnonzero(ok):
            d = day_list[j]
            if d not in sr or d not in vp:
                continue          # no borrowed radiation/humidity for this day
            tx = float(A["tmax"][gi, j]); tn = float(A["tmin"][gi, j])
            rows.append([d, q.get(d, ""), round((tx + tn) / 2, 3), round(tx, 3),
                         round(tn, 3), round(float(A["prcp"][gi, j]), 4),
                         sr[d], vp[d]])
        if not rows:
            nbad += 1
            continue
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
    log(f"ASSEMBLE ok={nok} bad={nbad} skipped_no_borrow={nskip} -> {outdir}")
    if nskip:
        log(f"  ({nskip} basins wait on {borrow} for srad/vp - re-run --assemble-only)")
    log("FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
