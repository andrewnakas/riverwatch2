#!/usr/bin/env python
"""GPU-FREE SCREEN: is Daymet V4R1-areal a BETTER forcing than CAMELS' V2/V3-areal?

Before spending ~31 h rebuilding 1980-2014 + ~7 h retraining, decide whether V4
actually carries more hydrological signal. The referee is observed streamflow:
the forcing that better explains the same-day relative flow rise is the better
forcing. No model, no GPU, no circularity.

For each sampled basin, over the SAME years and the SAME days:
    peak_k corr( precip[t], relative dQ[t+k] )   for k in -2..+2
computed separately for V4R1-areal (rebuilt here) and CAMELS-areal (frozen).

⚠️ The lag scan is essential, not optional: V4 labels precipitation one day
earlier than CAMELS ([[daymet-v4-vs-camels-ONE-DAY-PRECIP-OFFSET]]), so comparing
at a fixed lag would score V4 as catastrophically worse for a purely clerical
reason. Comparing PEAK-over-lag is registration-invariant, which is exactly what
we want: it asks "how much signal is there", not "is it labelled the same way".

DECISION RULE, fixed before the data exists:
  V4 peak median  >  CAMELS peak median + 0.005  -> V4 is better; fund the rebuild
  within +/- 0.005                               -> a wash; DO NOT fund
  V4 peak median  <  CAMELS peak median - 0.005  -> V4 is worse; abandon
"""
from __future__ import annotations
import argparse, gzip, json, math, subprocess, sys, tempfile, threading, time
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
FROZEN = ROOT / "gpu1080/corpora671/camels_corpus_daymet_v2"
BASE = ("https://opendap.earthdata.nasa.gov/collections/C2532426483-ORNL_CLOUD/"
        "granules/Daymet_Daily_V4R1.daymet_v4_daily_na_{var}_{year}.nc")
LCC = ("+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 "
       "+ellps=WGS84 +units=m +no_defs")
X0, Y0, RES, NX, NY = -4_560_250.0, 4_984_000.0, 1000.0, 7814, 8075
LAGS = (-2, -1, 0, 1, 2)
NC_LOCK = threading.Lock()
_p = threading.Lock()


def log(m):
    with _p:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def fetch(var, year, i0, i1, j0, j1, out, cookies, tries=3):
    nd = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
    ti = nd - 2 if nd == 366 else 364
    ce = (f"/x%5B{i0}:{i1}%5D;/y%5B{j0}:{j1}%5D;/time%5B0:{ti}%5D;"
          f"/lambert_conformal_conic;/{var}%5B0:{ti}%5D%5B{j0}:{j1}%5D%5B{i0}:{i1}%5D")
    url = BASE.format(var=var, year=year) + ".dap.nc4?dap4.ce=" + ce
    for k in range(tries):
        r = subprocess.run(["curl", "-g", "-s", "-n", "-L", "-c", cookies,
                            "-b", cookies, "--max-time", "600", "-o", str(out), url],
                           capture_output=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 10_000:
            return True
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"{var} {year} fetch failed")


def idx(geom, pad=2):
    a, b, c, d = geom.bounds
    return (max(0, int((a - X0) / RES) - pad), min(NX - 1, int((c - X0) / RES) + pad),
            max(0, int((Y0 - d) / RES) - pad), min(NY - 1, int((Y0 - b) / RES) + pad))


def lagcorr(a, b, lag):
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 400 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def load_frozen(g, y0, y1):
    p = FROZEN / f"{g}.csv.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().strip().split(",")
        ci = {c: i for i, c in enumerate(hdr)}
        d, P, Q = [], [], []
        for line in fh:
            f = line.strip().split(",")
            y = int(f[0][:4])
            if y < y0 or y > y1:
                continue
            d.append(f[0])
            try:
                P.append(float(f[ci["precipitation_sum"]]))
            except (ValueError, IndexError):
                P.append(np.nan)
            try:
                q = float(f[ci["q_cfs"]])
                Q.append(q if q >= 0 else np.nan)
            except (ValueError, IndexError):
                Q.append(np.nan)
    return d, np.array(P), np.array(Q)


def do_basin(g, row, y0, y1, workdir):
    fr = load_frozen(g, y0, y1)
    if fr is None:
        return None
    dates, Pc, Q = fr
    if np.isfinite(Q).sum() < 1000:
        return None
    geom = row.to_crs(LCC).geometry.iloc[0]
    i0, i1, j0, j1 = idx(geom)
    ck = str(workdir / f".c_{g}")
    v4 = {}
    weights = None
    for year in range(y0, y1 + 1):
        f = workdir / f"{g}_{year}.nc4"
        fetch("prcp", year, i0, i1, j0, j1, f, ck)
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
                cid = np.asarray(w["cell_id"].iloc[0], int)
                cov = np.asarray(w["coverage"].iloc[0], float)
                weights = (cid, cov, len(ys), len(xs))
            cid, cov, ny_, nx_ = weights
            a = ds["prcp"].values.astype("float64")
            vals = (a.reshape(a.shape[0], ny_ * nx_)[:, cid] * cov).sum(1) / cov.sum()
            dd = [str(np.datetime_as_string(t, unit="D")) for t in ds["time"].values]
        v4.update(dict(zip(dd, vals)))
        f.unlink(missing_ok=True)

    dQ = np.full_like(Q, np.nan); dQ[1:] = Q[1:] - Q[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = dQ / np.where(Q > 0, Q, np.nan)
    Pv = np.array([v4.get(d, np.nan) for d in dates])

    out = {}
    for name, P in (("camels", Pc), ("v4", Pv)):
        cs = {l: lagcorr(P, rel, l) for l in LAGS}
        cs = {l: c for l, c in cs.items() if not math.isnan(c)}
        if not cs:
            return None
        bl = max(cs, key=lambda l: cs[l])
        out[name] = (cs[bl], bl)
    return g, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basins", type=int, default=40)
    ap.add_argument("--y0", type=int, default=2000)
    ap.add_argument("--y1", type=int, default=2009)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    ids = json.load(open(ROOT / "data/camels_gauge_ids.json"))
    attrs = json.load(open(ROOT / "data/camels_attrs.json"))
    b531 = [x for x in ids["531"] if x in attrs and attrs[x].get("slope_mean") is not None]
    b531.sort(key=lambda x: float(attrs[x]["slope_mean"]))
    step = max(1, len(b531) // a.basins)
    sample = b531[::step][:a.basins]

    g = gpd.read_file(SHP)
    g["gid"] = g["GAGE_ID"].astype(str).str.zfill(8)
    have = set(g["gid"])
    sample = [x for x in sample if x in have]
    workdir = Path(tempfile.mkdtemp(prefix="screen_", dir="/tmp"))
    log(f"screening {len(sample)} basins, {a.y0}-{a.y1}, {a.workers} workers")

    res = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(do_basin, x, g[g["gid"] == x], a.y0, a.y1, workdir): x
                for x in sample}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as e:
                log(f"  {futs[fu]} FAIL {e}"); continue
            if r:
                res.append(r)
            if i % 10 == 0:
                log(f"  {i}/{len(sample)}")

    if not res:
        sys.exit("no basins scored")
    cam = np.array([r[1]["camels"][0] for r in res])
    v4 = np.array([r[1]["v4"][0] for r in res])
    dl = [r[1]["camels"][1] for r in res]
    vl = [r[1]["v4"][1] for r in res]

    print()
    print("=" * 70)
    print(f"  SCREEN: peak corr(precip, relative dQ) — higher = more signal")
    print(f"  {len(res)} basins, {a.y0}-{a.y1}")
    print("=" * 70)
    print(f"  CAMELS V2/V3 areal : median {np.median(cam):.4f}   mean {cam.mean():.4f}")
    print(f"  Daymet V4R1  areal : median {np.median(v4):.4f}   mean {v4.mean():.4f}")
    d = np.median(v4) - np.median(cam)
    print(f"  DELTA (v4 - camels): {d:+.4f}")
    win = int((v4 > cam).sum())
    print(f"  V4 better in       : {win}/{len(res)} basins ({win/len(res)*100:.0f}%)")
    from collections import Counter
    print(f"  peak lag camels    : {dict(Counter(dl))}")
    print(f"  peak lag v4        : {dict(Counter(vl))}")
    print()
    print("  --- PRE-REGISTERED DECISION ---")
    if d > 0.005:
        print(f"  ✅ V4 carries MORE signal ({d:+.4f} > +0.005) -> FUND the rebuild.")
    elif d < -0.005:
        print(f"  ⛔ V4 carries LESS signal ({d:+.4f} < -0.005) -> ABANDON.")
    else:
        print(f"  ⛔ WASH ({d:+.4f}, within +/-0.005) -> DO NOT fund 39 h on this.")


if __name__ == "__main__":
    main()
