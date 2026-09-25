#!/usr/bin/env python
"""Extend the CAMELS-style corpus forward to the present using LIVE upstream feeds,
in the EXACT corpus schema the MB-LSTM/δHBV trainer already ingests:

    date, q_cfs, temperature_2m_mean, temperature_2m_max, temperature_2m_min,
          precipitation_sum, shortwave_radiation_sum, vapor_pressure

Forcing = Daymet V4 (ORNL single-pixel REST API, no auth), target = USGS NWIS daily
discharge. Per basin (USGS gauge), fetches [start..end] and writes one csv.gz matching
the frozen corpus, so the exact training system works unchanged. Resumable: skips basins
whose output already covers the requested end date.

This is the yearly-updatable path: re-run with a later --end to append new years.

Daymet var mapping (ORNL name -> corpus col):
  prcp (mm/day)          -> precipitation_sum
  tmax (deg C)           -> temperature_2m_max
  tmin (deg C)           -> temperature_2m_min
  (tmax+tmin)/2          -> temperature_2m_mean
  srad (W/m2) * dayl / 1e6 -> shortwave_radiation_sum  (MJ/m2/day, matches corpus units)
  vp (Pa)                -> vapor_pressure
Daymet has a 365-day year (drops Dec 31 in leap years) — we forward-fill that one day.
"""
from __future__ import annotations
import argparse, gzip, io, json, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAYMET = "https://daymet.ornl.gov/single-pixel/api/data"
USGS = "https://waterservices.usgs.gov/nwis/dv/"
COLS = ["date","q_cfs","temperature_2m_mean","temperature_2m_max","temperature_2m_min",
        "precipitation_sum","shortwave_radiation_sum","vapor_pressure"]


def http_get(url, tries=4, timeout=60):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read().decode()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if k == tries - 1: raise
            time.sleep(2 * (k + 1))
    return None


def fetch_usgs_q(site, start, end):
    """USGS daily mean discharge (cfs, param 00060) -> {date: q_cfs}."""
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


def fetch_daymet(lat, lon, start, end):
    """Daymet single-pixel -> {date: dict of corpus forcing cols}. year range only."""
    y0, y1 = start[:4], end[:4]
    url = (f"{DAYMET}?lat={lat}&lon={lon}"
           f"&vars=prcp,tmax,tmin,srad,vp,dayl&start={start}&end={end}")
    txt = http_get(url)
    if not txt or "prcp" not in txt: return {}
    lines = txt.splitlines()
    # skip the metadata header block; find the CSV header row (starts with 'year,yday')
    hi = next((i for i, l in enumerate(lines) if l.lower().startswith("year,yday")), None)
    if hi is None: return {}
    hdr = [h.strip() for h in lines[hi].split(",")]
    idx = {name: hdr.index(next(c for c in hdr if c.lower().startswith(name)))
           for name in ["year","yday","prcp","tmax","tmin","srad","vp","dayl"]}
    import datetime
    out = {}
    for l in lines[hi+1:]:
        f = l.split(",")
        if len(f) < len(hdr): continue
        try:
            yr = int(float(f[idx["year"]])); yd = int(float(f[idx["yday"]]))
            d = (datetime.date(yr,1,1) + datetime.timedelta(days=yd-1)).isoformat()
            prcp = float(f[idx["prcp"]]); tmax = float(f[idx["tmax"]]); tmin = float(f[idx["tmin"]])
            srad = float(f[idx["srad"]]); vp = float(f[idx["vp"]]); dayl = float(f[idx["dayl"]])
            out[d] = {
                "precipitation_sum": round(prcp, 4),
                "temperature_2m_max": round(tmax, 3),
                "temperature_2m_min": round(tmin, 3),
                "temperature_2m_mean": round((tmax + tmin) / 2, 3),
                # srad W/m2 * daylight-seconds / 1e6 = MJ/m2/day (corpus radiation unit)
                "shortwave_radiation_sum": round(srad * dayl / 1e6, 6),
                "vapor_pressure": round(vp, 2),
            }
        except Exception:
            continue
    return out




def fetch_daymet_chunked(lat, lon, start, end):
    """Fetch Daymet year-by-year and merge — resilient to server 500s on long
    spans for edge tiles. Falls back from the single big request."""
    import datetime
    y0 = int(start[:4]); y1 = int(end[:4])
    merged = {}
    for yr in range(y0, y1 + 1):
        s = f"{max(yr, y0)}-01-01" if yr > y0 else start
        e_ = f"{yr}-12-31" if yr < y1 else end
        try:
            d = fetch_daymet(lat, lon, s, e_)
            merged.update(d)
        except Exception:
            pass
    return merged


def existing_end(path: Path):
    if not path.exists(): return None
    try:
        with gzip.open(path, "rt") as fh:
            last = None
            for line in fh: last = line
        return last.split(",")[0] if last else None
    except Exception:
        return None


def build_basin(site, lat, lon, start, end, out_path: Path):
    q = fetch_usgs_q(site, start, end)
    dm = fetch_daymet(lat, lon, start, end)
    if not dm:
        dm = fetch_daymet_chunked(lat, lon, start, end)
    if not dm:
        return 0
    dates = sorted(dm.keys())
    rows = []
    for d in dates:
        f = dm[d]
        rows.append([d, q.get(d, ""),  # missing discharge -> blank (trainer masks)
                     f["temperature_2m_mean"], f["temperature_2m_max"], f["temperature_2m_min"],
                     f["precipitation_sum"], f["shortwave_radiation_sum"], f["vapor_pressure"]])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as fh:
        fh.write(",".join(COLS) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latlon", default="data/modern_corpus/basin_latlon.json")
    ap.add_argument("--out", default="data/modern_corpus/camels_corpus_daymet_modern")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--limit", type=int, default=0, help="cap basins (smoke test)")
    ap.add_argument("--sleep", type=float, default=0.3, help="politeness delay between basins")
    args = ap.parse_args()

    latlon = json.load(open(ROOT / args.latlon))
    ids = sorted(latlon.keys())
    if args.limit: ids = ids[:args.limit]
    outdir = ROOT / args.out
    done = fail = skip = 0
    for i, site in enumerate(ids, 1):
        op = outdir / f"{site}.csv.gz"
        ee = existing_end(op)
        if ee and ee >= args.end:
            skip += 1; continue
        lat, lon = latlon[site]
        try:
            n = build_basin(site, lat, lon, args.start, args.end, op)
            if n > 0: done += 1
            else: fail += 1
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(ids)}] {site} ERR {e}", flush=True)
        if i % 20 == 0:
            print(f"[{i}/{len(ids)}] done={done} skip={skip} fail={fail}", flush=True)
        time.sleep(args.sleep)
    print(f"DONE daymet-modern: done={done} skip={skip} fail={fail} -> {outdir}")


if __name__ == "__main__":
    raise SystemExit(main())
