#!/usr/bin/env python
"""Build the NLDAS-2 areal modern corpus (2014-2025) for the 671 CAMELS basins.

Second gauge-calibrated forcing alongside areal Daymet. NLDAS-2 precipitation is
CPC gauge-only + PRISM, so it passes the screen that ERA5-Land fails
(see memory: era5land-fails-the-first-screen-question).

Route: GES DISC hourly granules via OPeNDAP DAP4 with VARIABLE SUBSETTING —
only Tair/Qair/PSurf/Rainf/SWdown, which halves the transfer (1.68 MB -> 784 KB
per granule, measured). Requires the "NASA GESDISC DATA ARCHIVE" app to be
authorized on the Earthdata account (verified working 2026-08-15).

Reduction: the basin x cell coverage weights were precomputed and validated once
(median area error 0.13% vs polygon area) in nldas_basin_weights.pkl. Each hourly
field is reduced to 671 basin means with a single bincount - no per-basin loop.

⚠️⚠️ NLDAS granules store latitude ASCENDING (south->north), while the weights
were computed in north-first raster space. The field MUST be flipped on the lat
axis before indexing, or every basin silently gets the wrong cells.

Daily aggregation (matching the corpus schema):
  precipitation_sum       = sum(Rainf)                     kg m-2/h = mm/h -> mm/day
  temperature_2m_mean/max/min = mean/max/min(Tair) - 273.15          K -> C
  shortwave_radiation_sum = mean(SWdown) * 86400 / 1e6      W/m2 -> MJ/m2/day
  vapor_pressure          = mean(e), e = q*P/(0.622+0.378q)          Pa

Checkpoints per MONTH so a failure costs at most one month of fetching.
"""
from __future__ import annotations
import argparse, calendar, datetime as dt, gzip, json, math, pickle, subprocess
import sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/home/nakas/riverwatch2")
WEIGHTS = ROOT / "data/modern_corpus/nldas_basin_weights.pkl"
CKPT = ROOT / "data/modern_corpus/_nldas_monthly"
OUT = ROOT / "data/modern_corpus/camels_corpus_nldas_modern_areal"
# ⛔ 2026-09-23: the on-prem host below returns **HTTP 410 Service Permanently
# Retired** for EVERY year (verified on 1980 and on 2014, which this pipeline had
# previously built successfully). GES DISC retired its on-premises OPeNDAP and
# moved the data to Earthdata Cloud OPeNDAP.
#   OLD: https://hydro1.gesdisc.eosdis.nasa.gov/opendap/NLDAS/NLDAS_FORA0125_H.2.0/
#        {y}/{doy:03d}/NLDAS_FORA0125_H.A{y}{m:02d}{d:02d}.{h:02d}00.020.nc
# The cloud granule id embeds the collection with a COLON, which must stay
# percent-encoded as %3A -- the un-encoded form 404s. URL taken from CMR's own
# "USE SERVICE API" RelatedUrl rather than constructed by guesswork.
BASE = ("https://opendap.earthdata.nasa.gov/collections/C2033151148-GES_DISC/granules/"
        "NLDAS_FORA0125_H.2.0%3ANLDAS_FORA0125_H.A{y}{m:02d}{d:02d}.{h:02d}00.020.nc")
# ⚠️ The `.dap.nc4` suffix is REQUIRED. Without it Hyrax returns HTTP 403
# "The requested URL directly references a data source" — a 1,087-byte HTML page,
# not data. Measured 2026-08-15: every hour failed this way until it was added.
CE = ".dap.nc4?dap4.ce=/lat;/lon;/time;/Rainf;/Tair;/SWdown;/Qair;/PSurf"
# FALLBACK, added 2026-09-23: the whole of opendap.earthdata.nasa.gov began
# returning **HTTP 503** for every collection, including bare .dmr metadata,
# while plain HTTPS data distribution kept serving 200. OPeNDAP here is only an
# optimisation (variable subsetting, 742 KB vs 1.66 MB); the full granule carries
# the same fields and the reducer reads only what it needs. So a 503 on the
# subset route is no longer fatal -- it costs 2.2x the bytes, not the build.
DIRECT = ("https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/"
          "{y}/{doy:03d}/NLDAS_FORA0125_H.A{y}{m:02d}{d:02d}.{h:02d}00.020.nc")
NY, NX = 224, 464
VARS = ["Rainf", "Tair", "SWdown", "Qair", "PSurf"]

_plock = threading.Lock()
NC_LOCK = threading.Lock()      # libhdf5 is not thread-safe (measured: segfault)


def log(m):
    with _plock:
        print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)


def load_weights():
    w = pickle.load(open(WEIGHTS, "rb"))
    basins = sorted(w["weights"].keys())
    flat_id, flat_w, bidx = [], [], []
    for i, b in enumerate(basins):
        cid, cov = w["weights"][b]
        flat_id.append(np.asarray(cid, dtype=np.int64))
        flat_w.append(np.asarray(cov, dtype=np.float64))
        bidx.append(np.full(len(cid), i, dtype=np.int64))
    flat_id = np.concatenate(flat_id)
    flat_w = np.concatenate(flat_w)
    bidx = np.concatenate(bidx)
    wsum = np.bincount(bidx, weights=flat_w, minlength=len(basins))
    return basins, flat_id, flat_w, bidx, wsum


def reduce_field(field2d, flat_id, flat_w, bidx, wsum, nb):
    """north-first flatten -> per-basin coverage-weighted mean."""
    f = np.flipud(field2d).ravel()          # NLDAS is south->north; weights are north-first
    vals = f[flat_id]
    good = np.isfinite(vals)
    if not good.all():
        num = np.bincount(bidx[good], weights=vals[good] * flat_w[good], minlength=nb)
        den = np.bincount(bidx[good], weights=flat_w[good], minlength=nb)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    num = np.bincount(bidx, weights=vals * flat_w, minlength=nb)
    return num / wsum


def preflight(workdir: Path) -> None:
    """Fetch ONE granule and assert it is HDF5 before spending days on the rest.

    Written because this build ground through **959 retries** against a dead
    endpoint: GES DISC had returned a 3,725-byte HTML "410 Service Retired" page
    with curl rc=0, which the size check classified as merely `retryable`. A
    route that is gone should fail in seconds and say so, not look like a slow
    network for hours."""
    probe = workdir / "_preflight.nc4"
    try:
        fetch_hour(1980, 1, 8, 1, probe, str(workdir / ".ck_pre"), tries=2)
    except RuntimeError as e:
        raise SystemExit(f"PREFLIGHT FAIL: {e}\n  The NLDAS route is not serving "
                         f"data. Check {BASE.split('/granules/')[0]} before retrying.")
    with open(probe, "rb") as fh:
        magic = fh.read(8)
    probe.unlink(missing_ok=True)
    if magic[:4] != b"\x89HDF":
        raise SystemExit(f"PREFLIGHT FAIL: response is not HDF5 (magic={magic!r}) "
                         f"- the endpoint is serving an error page, not data")
    log("PREFLIGHT OK: NLDAS cloud OPeNDAP returned real HDF5")


def fetch_hour(y, m, d, h, out, cookies, tries=3):
    doy = dt.date(y, m, d).timetuple().tm_yday
    urls = [BASE.format(y=y, doy=doy, m=m, d=d, h=h) + CE,      # subset (cheap)
            DIRECT.format(y=y, doy=doy, m=m, d=d, h=h)]          # full granule
    last = ""
    for k in range(tries):
        for url in urls:
            r = subprocess.run(["curl", "-g", "-s", "-n", "-L", "-c", cookies,
                                "-b", cookies, "--max-time", "300", "-o", str(out), url],
                               capture_output=True)
            if r.returncode == 0 and out.exists() and out.stat().st_size > 100_000:
                return True
            last = (f"rc={r.returncode} size="
                    f"{out.stat().st_size if out.exists() else 0} "
                    f"({'subset' if url is urls[0] else 'direct'})")
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"{y}-{m:02d}-{d:02d} {h:02d}Z: {last}")


def do_hour(args):
    y, m, d, h, workdir, W = args
    basins, flat_id, flat_w, bidx, wsum = W
    nb = len(basins)
    f = workdir / f"h_{y}{m:02d}{d:02d}{h:02d}.nc4"
    # ⚠️ ONE COOKIE JAR PER THREAD. This was `.c_{h % 8}` while the pool runs 10+
    # workers, so two concurrent curls could share a jar and race on writing it;
    # a clobbered Earthdata session cookie comes back as an HTML error page with
    # curl rc=0, i.e. it looks exactly like the retired-endpoint failure. Measured
    # 2026-09-23: hours that failed "retryable" in the pool fetched fine alone.
    ck = workdir / f".c_t{threading.get_ident()}"
    fetch_hour(y, m, d, h, f, str(ck))
    try:
        with NC_LOCK, xr.open_dataset(f) as ds:
            out = {}
            for v in VARS:
                a = ds[v].values
                a = a[0] if a.ndim == 3 else a
                out[v] = reduce_field(a.astype("float64"), flat_id, flat_w, bidx, wsum, nb)
    finally:
        f.unlink(missing_ok=True)
    return (y, m, d, h, out)


def month_ckpt(y, m):
    return CKPT / f"nldas_{y}{m:02d}.npz"


def build_month(y, m, W, workers, workdir):
    cp = month_ckpt(y, m)
    if cp.exists():
        return "skip", 0
    basins = W[0]
    ndays = calendar.monthrange(y, m)[1]
    acc = {}   # (day) -> {var: [24 arrays]}
    tasks = [(y, m, d, h, workdir, W) for d in range(1, ndays + 1) for h in range(24)]
    got = 0

    def run(batch):
        """Returns the tasks that failed, rather than aborting the month.
        A month is ~744 fetches; losing all of them to one bad hour is the
        difference between a 2-minute retry and re-fetching 16 minutes of data."""
        failed = []
        nonlocal got
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(do_hour, t): t for t in batch}
            for fu in as_completed(futs):
                t = futs[fu]
                try:
                    yy, mm, dd, hh, vals = fu.result()
                except Exception as e:
                    failed.append(t)
                    log(f"  {y}-{m:02d}-{t[2]:02d} {t[3]:02d}Z retryable: {e}")
                    continue
                acc.setdefault(dd, {}).setdefault("_n", 0)
                for v in VARS:
                    acc[dd].setdefault(v, []).append(vals[v])
                acc[dd]["_n"] += 1
                got += 1
        return failed

    failed = run(tasks)
    if failed:
        log(f"  {y}-{m:02d}: retrying {len(failed)} hours")
        time.sleep(20)
        failed = run(failed)
    if failed:
        raise RuntimeError(f"{len(failed)} hours still failing, e.g. "
                           f"{y}-{m:02d}-{failed[0][2]:02d} {failed[0][3]:02d}Z")

    nb = len(basins)
    days, P, TM, TX, TN, S, VP = [], [], [], [], [], [], []
    for d in range(1, ndays + 1):
        a = acc.get(d)
        if not a or a["_n"] != 24:
            raise RuntimeError(f"{y}-{m:02d}-{d:02d}: {0 if not a else a['_n']}/24 hours")
        rain = np.vstack(a["Rainf"])       # (24, nb) mm/h
        tair = np.vstack(a["Tair"]) - 273.15
        swd = np.vstack(a["SWdown"])
        q = np.vstack(a["Qair"])
        ps = np.vstack(a["PSurf"])
        e = q * ps / (0.622 + 0.378 * q)   # Pa
        days.append(f"{y}-{m:02d}-{d:02d}")
        P.append(np.nansum(rain, axis=0))
        TM.append(np.nanmean(tair, axis=0))
        TX.append(np.nanmax(tair, axis=0))
        TN.append(np.nanmin(tair, axis=0))
        S.append(np.nanmean(swd, axis=0) * 86400.0 / 1e6)
        VP.append(np.nanmean(e, axis=0))

    CKPT.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, basins=np.array(basins), days=np.array(days),
                        precip=np.array(P), tmean=np.array(TM), tmax=np.array(TX),
                        tmin=np.array(TN), srad=np.array(S), vp=np.array(VP))
    tmp.replace(cp)
    return "ok", got


def assemble(y0, y1):
    """Merge monthly checkpoints into per-basin CSVs (Daymet corpus schema)."""
    COLS = ["date", "q_cfs", "temperature_2m_mean", "temperature_2m_max",
            "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum",
            "vapor_pressure"]
    months = [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)]
    missing = [f"{y}{m:02d}" for y, m in months if not month_ckpt(y, m).exists()]
    if missing:
        log(f"cannot assemble - {len(missing)} months missing: {missing[:6]}")
        return 1
    # Concatenate months along the TIME axis and keep everything as arrays.
    # An earlier version looped days x basins and indexed numpy element-wise
    # (~18M boxed scalars); it had produced nothing after 10 minutes on 2 years.
    basins = None
    days_all = []
    cols = {k: [] for k in ("tmean", "tmax", "tmin", "precip", "srad", "vp")}
    for y, m in months:
        z = np.load(month_ckpt(y, m), allow_pickle=True)
        b = [str(x) for x in z["basins"]]
        if basins is None:
            basins = b
        elif b != basins:
            log(f"basin order mismatch in {y}-{m:02d}"); return 1
        days_all.extend(str(d) for d in z["days"])
        for k in cols:
            cols[k].append(np.asarray(z[k], dtype="float64"))
    for k in cols:
        cols[k] = np.concatenate(cols[k], axis=0)     # (ndays, nbasins)
    ndays = len(days_all)
    log(f"assembling {len(basins)} basins x {ndays} days")
    if any(v.shape != (ndays, len(basins)) for v in cols.values()):
        log("shape mismatch after concatenation"); return 1
    # discharge: reuse the Daymet corpus column (same USGS source, already masked)
    # MODERN-1: read the authoritative USGS DV store (1980->today), not the Daymet
    # corpus -- that corpus has no discharge before 2015 (see build_q_store.py).
    DAY = ROOT / "data/modern_corpus/q_store"
    OUT.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for i, bs in enumerate(basins):
        q = {}
        dp = DAY / f"{bs}.csv.gz"
        if dp.exists():
            with gzip.open(dp, "rt") as fh:
                hdr = fh.readline().rstrip("\n").split(",")
                qi = hdr.index("q_cfs")
                for line in fh:
                    f = line.rstrip("\n").split(",")
                    try:
                        v = float(f[qi])
                        if v >= 0 and math.isfinite(v):
                            q[f[0]] = v
                    except (ValueError, IndexError):
                        pass
        tm = cols["tmean"][:, i]; tx = cols["tmax"][:, i]; tn = cols["tmin"][:, i]
        p = cols["precip"][:, i]; s = cols["srad"][:, i]; v = cols["vp"][:, i]
        op = OUT / f"{bs}.csv.gz"
        tmp = op.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wt") as fh:
            fh.write(",".join(COLS) + "\n")
            fh.writelines(
                f"{days_all[j]},{q.get(days_all[j],'')},{tm[j]:.3f},{tx[j]:.3f},"
                f"{tn[j]:.3f},{p[j]:.4f},{s[j]:.6f},{v[j]:.2f}\n"
                for j in range(ndays))
        with gzip.open(tmp, "rt") as fh:
            assert sum(1 for _ in fh) - 1 == ndays
        tmp.replace(op)
        n_ok += 1
        if n_ok % 100 == 0:
            log(f"  assembled {n_ok}/{len(basins)}")
    log(f"assembled {n_ok} basins -> {OUT} ({ndays} days each)")
    return 0


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
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--months", default="", help="YYYYMM,YYYYMM (smoke test)")
    a = ap.parse_args()

    assert_q_store_covers(a.start_year, a.end_year,
                          ROOT / "data/modern_corpus/q_store",
                          n_expected=0 if a.months else 671)
    if a.assemble_only:
        return assemble(a.start_year, a.end_year)

    W = load_weights()
    log(f"weights: {len(W[0])} basins")
    workdir = Path(tempfile.mkdtemp(prefix="nldas_", dir="/tmp"))
    preflight(workdir)

    if a.months:
        months = [(int(x[:4]), int(x[4:6])) for x in a.months.split(",")]
    else:
        months = [(y, m) for y in range(a.start_year, a.end_year + 1)
                  for m in range(1, 13)]
    todo = [(y, m) for y, m in months if not month_ckpt(y, m).exists()]
    log(f"{len(months)} months, {len(todo)} to fetch, {a.workers} workers")

    t0 = time.time()
    for i, (y, m) in enumerate(months, 1):
        try:
            st, n = build_month(y, m, W, a.workers, workdir)
        except Exception as e:
            log(f"{y}-{m:02d} FAIL {e}")
            continue
        el = time.time() - t0
        done = sum(1 for yy, mm in months[:i] if month_ckpt(yy, mm).exists())
        log(f"[{i}/{len(months)}] {y}-{m:02d} {st} ({n} hours)  "
            f"elapsed={el/3600:.2f}h eta={el/max(i,1)*(len(months)-i)/3600:.2f}h")
    log("FETCH PHASE DONE - run with --assemble-only to write the corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
