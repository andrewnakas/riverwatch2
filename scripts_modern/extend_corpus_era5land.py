#!/usr/bin/env python
"""ERA5-Land forcing member (Maurer replacement) for the modern, yearly-updatable corpus.

ERA5-Land is a global reanalysis — methodologically INDEPENDENT from Daymet (station
interpolation) and NLDAS-2 (NARR-forced LSM), so it maximizes ensemble decorrelation
(research-verified 2026-07-29). Writes the EXACT corpus schema the trainer ingests:

    date, q_cfs, temperature_2m_mean, temperature_2m_max, temperature_2m_min,
          precipitation_sum, shortwave_radiation_sum, vapor_pressure

ARCHITECTURE: CDS is a queued service (~40s/request), so per-basin requests are infeasible
(671×11yr). Instead we pull ONE CONUS bounding box per YEAR (a handful of requests), cache the
netCDFs, then extract each basin's nearest grid cell locally. Streamflow (q_cfs) from USGS NWIS.

Per year we fetch:
  - daily-statistics (dataset derived-era5-land-daily-statistics):
      2m_temperature daily_maximum -> temperature_2m_max
      2m_temperature daily_minimum -> temperature_2m_min
      2m_temperature daily_mean    -> temperature_2m_mean
      surface_solar_radiation_downwards daily_mean (J/m2 accumulated daily) -> shortwave_radiation_sum (MJ/m2/day)
      2m_dewpoint_temperature daily_mean -> vapor_pressure (via Td)
  - hourly total_precipitation (dataset reanalysis-era5-land), summed to daily -> precipitation_sum (mm)

Units:
  ERA5-Land temps are Kelvin -> subtract 273.15 for degC.
  ssrd accumulated J/m2/day -> /1e6 = MJ/m2/day.
  total_precipitation is meters -> *1000 = mm.
  vapor pressure e (hPa) = 6.112*exp(17.67*Td_C/(Td_C+243.5)); corpus vp appears to be in the
  same magnitude as Daymet vp (Pa, ~100-2000) -> multiply hPa by 100 to get Pa.
"""
from __future__ import annotations
import argparse, gzip, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = ["date","q_cfs","temperature_2m_mean","temperature_2m_max","temperature_2m_min",
        "precipitation_sum","shortwave_radiation_sum","vapor_pressure"]
USGS = "https://waterservices.usgs.gov/nwis/dv/"
# CONUS box for all 671 basins (small margin). area = [N, W, S, E]
AREA = [49.5, -125.0, 26.5, -67.0]


def http_get(url, tries=4, timeout=60):
    import urllib.request, urllib.error
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read().decode()
        except Exception:
            if k == tries - 1: raise
            time.sleep(2 * (k + 1))


def fetch_usgs_q(site, start, end):
    url = f"{USGS}?format=json&sites={site}&parameterCd=00060&statCd=00003&startDT={start}&endDT={end}"
    try:
        j = json.loads(http_get(url))
        ts = j["value"]["timeSeries"]
        if not ts: return {}
        out = {}
        for v in ts[0]["values"][0]["value"]:
            try: out[v["dateTime"][:10]] = float(v["value"])
            except: pass
        return out
    except Exception:
        return {}


def cds_year(cache: Path, year: int):
    """Download the CONUS-box ERA5-Land data for ALL years [y0..y1] in ONE request per
    variable (6 requests total, not 6/year) — CDS queue latency dominates, so fewer
    bigger requests is far faster. Resumable: skips files already present."""
    import cdsapi
    cache.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()
    years = [str(year)]
    days = [f"{d:02d}" for d in range(1, 32)]
    months = [f"{m:02d}" for m in range(1, 13)]
    reqs = {
        "tmax": ("derived-era5-land-daily-statistics",
                 dict(variable="2m_temperature", daily_statistic="daily_maximum")),
        "tmin": ("derived-era5-land-daily-statistics",
                 dict(variable="2m_temperature", daily_statistic="daily_minimum")),
        "dew": ("derived-era5-land-daily-statistics",
                dict(variable="2m_dewpoint_temperature", daily_statistic="daily_mean")),
    }
    out = {}
    # The three whole-year daily-statistics requests (tmax/tmin/dew) were issued
    # SERIALLY and each can take hours, so they -- not the monthly loop -- were
    # the dominant cost (a 1980 tmax request sat "accepted" for 27 min with
    # nothing cached). Submit them concurrently; same rules as the monthly pool:
    # one cdsapi.Client per thread (not thread-safe) and cached files skipped.
    from concurrent.futures import ThreadPoolExecutor as _TPEy, as_completed as _acy

    def _fetch_year(key, ds, extra, fp):
        import cdsapi as _cdsapi
        _c = _cdsapi.Client(quiet=True, progress=False)
        body = dict(year=years, month=months, day=days,
                    time_zone="utc+00:00", frequency="1_hourly", area=AREA, **extra)
        _c.retrieve(ds, body, str(fp))
        return key, fp

    _yjobs = []
    for key, (ds, extra) in reqs.items():
        fp = cache / f"{year}_{key}.nc"
        out[key] = fp
        if fp.exists() and fp.stat().st_size > 1000:
            print(f"  {key}: cached", flush=True)
            continue
        _yjobs.append((key, ds, extra, fp))

    # SERIAL + BACKOFF for derived-era5-land-daily-statistics.
    # CDS caps QUEUED REQUESTS PER DATASET; this dataset has its own budget and
    # rejects with "Number queued requests for this dataset is temporarily
    # limited." Keeping at most one job in that queue, and backing off rather
    # than hammering, is the only thing that clears it -- retrying aggressively
    # is what deepened the problem before.
    import time as _time
    for _k, _ds, _extra, _fp in _yjobs:
        _body = dict(year=years, month=months, day=days,
                     time_zone="utc+00:00", frequency="1_hourly",
                     area=AREA, **_extra)
        for _attempt in range(6):
            try:
                import cdsapi as _cdsapi
                _c1 = _cdsapi.Client(quiet=True, progress=False)
                _c1.retrieve(_ds, _body, str(_fp))
                print(f"  {_k}: {year} ok", flush=True)
                break
            except Exception as _e:
                _msg = str(_e)
                _limited = ("temporarily limited" in _msg
                            or "has been rejected" in _msg)
                if _attempt == 5 or not _limited:
                    print(f"  [FAIL] {_k} {year}: {type(_e).__name__}: "
                          f"{_msg[:140]}", flush=True)
                    break
                _wait = 120 * (2 ** _attempt)          # 2,4,8,16,32 min
                print(f"  [queue-limited] {_k} {year}, backing off "
                      f"{_wait//60} min (attempt {_attempt+1}/5)", flush=True)
                _time.sleep(_wait)
    # hourly variables (tp = precip meters; srad = accumulated ssrd J/m2) are
    # too large for CDS at full-year CONUS granularity -> chunk PER MONTH.
    # out[key] becomes a list of monthly netcdf paths; extract concatenates.
    hourly = {
        "tp": "total_precipitation",
        "srad": "surface_solar_radiation_downwards",
    }
    # PARALLEL per-month retrieval. The bottleneck is CDS queue latency, not
    # transfer, and CDS serves a user's requests concurrently -- so submitting
    # several months at once scales nearly linearly. Threads are correct here:
    # cdsapi blocks on network I/O (releases the GIL) and each request writes
    # its own file, so there is no shared mutable state.
    import os as _os
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _ac
    _JOBS = int(_os.environ.get("ERA5_JOBS", "8"))

    def _fetch_one(key, varname, mm, fp):
        # Each thread needs its own client: cdsapi.Client is not thread-safe.
        import cdsapi as _cdsapi
        _c = _cdsapi.Client(quiet=True, progress=False)
        _c.retrieve("reanalysis-era5-land", dict(
            variable=varname, year=years, month=[mm], day=days,
            time=[f"{h:02d}:00" for h in range(24)], area=AREA,
            data_format="netcdf",
        ), str(fp))
        return key, mm, fp

    _jobs = []
    for key, varname in hourly.items():
        paths = []
        for mm in months:
            fp = cache / f"{year}_{mm}_{key}.nc"
            paths.append(fp)
            if fp.exists() and fp.stat().st_size > 1000:
                continue          # resume: already cached
            _jobs.append((key, varname, mm, fp))
        out[key] = paths

    if _jobs:
        print(f"  submitting {len(_jobs)} month-requests, {_JOBS} at a time "
              f"(year {year})", flush=True)
        with _TPE(max_workers=_JOBS) as _ex:
            _futs = {_ex.submit(_fetch_one, *j): j for j in _jobs}
            _done = 0
            for _f in _ac(_futs):
                _k, _v, _m, _p = _futs[_f]
                try:
                    _f.result()
                    _done += 1
                    print(f"  [{_done}/{len(_jobs)}] {_k} {year}-{_m} ok",
                          flush=True)
                except Exception as _e:
                    # one failed month must not kill the run; extract() skips
                    # missing files and the next pass retries them.
                    print(f"  [FAIL] {_k} {year}-{_m}: "
                          f"{type(_e).__name__}: {str(_e)[:120]}", flush=True)
    return out


def extract_basins(paths, latlon, year):
    """From the year's netCDFs, extract each basin's nearest-cell daily series.
    Returns {basin: {date: {col: val}}}."""
    import xarray as xr, numpy as np, pandas as pd
    def _open(p):
        # p may be a single Path (daily-stats vars) or a list of monthly Paths
        # (hourly tp/srad). Concatenate monthly files along their time coord.
        if isinstance(p, (list, tuple)):
            parts = [xr.open_dataset(q) for q in p if q.exists() and q.stat().st_size > 1000]
            if not parts:
                return None
            tc = next((c for c in ("valid_time","time") if c in parts[0].coords),
                      list(parts[0].coords)[0])
            return xr.concat(parts, dim=tc)
        return xr.open_dataset(p) if p.exists() else None
    dss = {k: d for k, p in paths.items() if (d := _open(p)) is not None}
    def latname(ds): return "latitude" if "latitude" in ds.coords else "lat"
    def lonname(ds): return "longitude" if "longitude" in ds.coords else "lon"
    def tname(ds):
        for c in ("valid_time","time","forecast_reference_time"):
            if c in ds.coords: return c
        return list(ds.coords)[0]
    # daily datasets: one value per day
    def var1(ds):
        return ds[[v for v in ds.data_vars][0]]
    tp = dss.get("tp")
    sr = dss.get("srad")
    out = {}
    for b, (blat, blon) in latlon.items():
        rec = {}
        try:
            for key in ("tmax","tmin","tmean","dew"):
                if key not in dss: continue
                da = var1(dss[key]).sel({latname(dss[key]): blat, lonname(dss[key]): blon}, method="nearest")
                tn = tname(dss[key])
                for t, val in zip(da[tn].values, da.values):
                    d = pd.Timestamp(t).strftime("%Y-%m-%d")
                    rec.setdefault(d, {})[key] = float(val)
            # precip: hourly -> daily sum (meters -> mm)
            if tp is not None:
                pv = var1(tp).sel({latname(tp): blat, lonname(tp): blon}, method="nearest")
                tn = tname(tp)
                s = pd.Series(pv.values, index=pd.to_datetime(pv[tn].values))
                daily = s.resample("1D").sum() * 1000.0
                for t, val in daily.items():
                    rec.setdefault(t.strftime("%Y-%m-%d"), {})["tp"] = float(val)
            # srad: hourly accumulated ssrd (J/m2, resets 00 UTC) -> daily total
            # = the day's max accumulated reading.
            if sr is not None:
                sv = var1(sr).sel({latname(sr): blat, lonname(sr): blon}, method="nearest")
                tn = tname(sr)
                ss = pd.Series(sv.values, index=pd.to_datetime(sv[tn].values))
                sdaily = ss.resample("1D").max()
                for t, val in sdaily.items():
                    rec.setdefault(t.strftime("%Y-%m-%d"), {})["srad"] = float(val)
        except Exception:
            continue
        out[b] = rec
    for ds in dss.values(): ds.close()
    return out


def to_corpus_rows(rec, q):
    import math
    rows = []
    for d in sorted(rec):
        r = rec[d]
        if not all(k in r for k in ("tmax","tmin","dew","srad")):
            continue
        tmax = r["tmax"] - 273.15; tmin = r["tmin"] - 273.15; tmean = (r["tmax"] + r["tmin"]) / 2 - 273.15
        td = r["dew"] - 273.15
        e_hpa = 6.112 * math.exp(17.67 * td / (td + 243.5))
        vp_pa = e_hpa * 100.0
        srad_mj = r["srad"] / 1e6  # J/m2/day -> MJ/m2/day
        prcp = r.get("tp", 0.0)
        rows.append([d, q.get(d, ""), round(tmean,3), round(tmax,3), round(tmin,3),
                     round(prcp,4), round(srad_mj,6), round(vp_pa,2)])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latlon", default="data/modern_corpus/basin_latlon.json")
    ap.add_argument("--out", default="data/modern_corpus/camels_corpus_era5land_modern")
    ap.add_argument("--cache", default="data/modern_corpus/_era5_cache")
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    latlon = json.load(open(ROOT / args.latlon))
    ids = sorted(latlon.keys())
    if args.limit: ids = ids[:args.limit]
    latlon = {b: latlon[b] for b in ids}
    outdir = ROOT / args.out; outdir.mkdir(parents=True, exist_ok=True)
    cache = ROOT / args.cache

    # per-year CDS fetch (1yr = CDS cost-cap max for CONUS daily-stats), extract, accumulate
    acc = {b: {} for b in ids}
    for yr in range(args.start_year, args.end_year + 1):
        print(f"=== year {yr}: CDS fetch (5 reqs) ===", flush=True)
        paths = cds_year(cache, yr)
        print(f"=== year {yr}: extract {len(ids)} basins ===", flush=True)
        yr_rec = extract_basins(paths, latlon, yr)
        for b in ids: acc[b].update(yr_rec.get(b, {}))

    written = 0
    for b in ids:
        q = fetch_usgs_q(b, f"{args.start_year}-01-01", f"{args.end_year}-12-31")
        rows = to_corpus_rows(acc[b], q)
        if not rows: continue
        with gzip.open(outdir / f"{b}.csv.gz", "wt") as fh:
            fh.write(",".join(COLS) + "\n")
            for r in rows: fh.write(",".join(str(x) for x in r) + "\n")
        written += 1
    print(f"DONE era5land-modern: {written}/{len(ids)} -> {outdir}")


if __name__ == "__main__":
    raise SystemExit(main())
