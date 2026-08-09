#!/usr/bin/env python3
"""Build a provenance-tagged station-observation corpus for CAMELS-531, 1980-2010.

GHCN-Daily already IS the fusion people reach for: COOP, ASOS/WBAN and SNOTEL all
live in one archive, one format, one QC pipeline. Measured from ghcnd-inventory:
6,883 US stations carry PRCP across the full 1980-2010 window (COOP 5,845, WBAN
653, SNOTEL 385), and every one of the 531 basins has such a station within 50 km
(465 within 25 km, median 12.3 km). Fusing three separate APIs would be strictly
worse than reading the archive that already merged them.

IMPORTANT -- this corpus is NOT independent of Daymet. Daymet ingests GHCN-Daily,
and ASOS sits inside GHCN too. Independence is handled by TAGGING, not by
pretending: every station carries its source network, so an analysis can restrict
to whichever subset its claim requires. SNOTEL via AWDB REST is the one
genuinely-outside-GHCN path and is fetched separately as a control.

Two traps this code exists to avoid:
  1. GHCN stores TENTHS -- PRCP in tenths of mm, TMAX/TMIN in tenths of degC.
     Reading them raw is a silent 10x error.
  2. QFLAG must be filtered. A non-blank QFLAG means the value FAILED one of 13
     quality checks; keeping those silently poisons the corpus.

Stages (each cached, so the script is resumable):
    --stage inventory : parse the GHCN inventory, pick full-span US stations
    --stage join      : map stations to basins (in-polygon preferred, else <=25km)
    --stage fetch     : download per-station files (~196 KB each)
    --stage build     : QC + unit-convert + aggregate + write the corpus
"""
import argparse
import gzip
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(R, "data/cache/ghcn")
STATION_DIR = os.path.join(CACHE, "by_station")
OUT_DIR = os.path.join(R, "data/local_corpora/camels_corpus_station_v1")

INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
BY_STATION = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{}.csv.gz"

START_YEAR, END_YEAR = 1980, 2010
MAX_KM = 25.0
ELEMENTS = ("PRCP", "TMAX", "TMIN", "SNOW", "SNWD")

# ⚠️ UNITS DIFFER BY ELEMENT. PRCP is tenths of mm and TMAX/TMIN are tenths of
# degC, but SNOW (snowfall) and SNWD (snow depth) are WHOLE MILLIMETRES. A
# blanket /10 would silently shrink snow tenfold -- the same class of error the
# tenths conversion exists to prevent.
TENTHS = {"PRCP", "TMAX", "TMIN"}

# GHCN network, from the station-ID prefix.
NETWORKS = {"USC": "COOP", "USW": "WBAN", "USS": "SNOTEL",
            "USR": "RAWS", "USH": "USHCN"}

# QFLAG values that mean the observation FAILED a quality check. Blank = passed.
BAD_QFLAGS = set("DGIKLMNORSTWX")


def _get(url, timeout=180, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            return urllib.request.urlopen(req, timeout=timeout).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized over the second pair. Mirrors app/snotel.py::_haversine_km."""
    rk = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * rk * np.arcsin(np.sqrt(a))


def stage_inventory():
    """US stations whose PRCP (and optionally TMAX/TMIN) span the target window."""
    os.makedirs(CACHE, exist_ok=True)
    raw_path = os.path.join(CACHE, "ghcnd-inventory.txt")
    if not os.path.exists(raw_path):
        print("downloading GHCN inventory ...")
        with open(raw_path, "wb") as fh:
            fh.write(_get(INVENTORY_URL))
    raw = open(raw_path, errors="ignore").read().splitlines()
    print(f"inventory lines: {len(raw):,}")

    span = {}
    for line in raw:
        sid = line[0:11]
        if not sid.startswith("US"):
            continue
        try:
            lat, lon = float(line[12:20]), float(line[21:30])
            y1, y2 = int(line[36:40]), int(line[41:45])
        except ValueError:
            continue
        el = line[31:35].strip()
        if el not in ELEMENTS:
            continue
        if y1 <= START_YEAR and y2 >= END_YEAR:
            rec = span.setdefault(sid, {"lat": lat, "lon": lon, "elements": set()})
            rec["elements"].add(el)

    keep = {k: v for k, v in span.items() if "PRCP" in v["elements"]}
    rows = [{"station": k, "lat": v["lat"], "lon": v["lon"],
             "network": NETWORKS.get(k[:3], "other"),
             "has_tmax": "TMAX" in v["elements"],
             "has_tmin": "TMIN" in v["elements"]} for k, v in keep.items()]
    df = pd.DataFrame(rows).sort_values("station")
    out = os.path.join(CACHE, "stations_fullspan.csv")
    df.to_csv(out, index=False)
    print(f"\nUS stations with PRCP spanning {START_YEAR}-{END_YEAR}: {len(df):,}")
    print(df.network.value_counts().to_string())
    print(f"  with TMAX too: {df.has_tmax.sum():,}")
    print(f"wrote {out}")
    return df


def stage_join():
    """Map stations to basins. In-polygon if geopandas + shapefile exist, else <=25 km."""
    sp = os.path.join(CACHE, "stations_fullspan.csv")
    if not os.path.exists(sp):
        sys.exit("run --stage inventory first")
    S = pd.read_csv(sp, dtype={"station": str})

    ids = [str(x).zfill(8) for x in
           json.load(open(f"{R}/data/camels_gauge_ids.json"))["531"]]
    meta = json.load(open(f"{R}/data/camels_station_meta.json"))
    coord = {str(s["id"]).zfill(8): (float(s["lat"]), float(s["lon"]))
             for s in meta["stations"] if str(s["id"]).zfill(8) in set(ids)}
    missing = [b for b in ids if b not in coord]
    if missing:
        print(f"WARNING: {len(missing)} basins lack coordinates, skipped")

    # Try polygons; they are strictly better than a radius but need the geo stack.
    polys = None
    try:
        import geopandas as gpd  # noqa: F401
        from aorc_exact_extract import get_polygon_gages2, GAGES2_SHP
        if os.path.exists(os.path.join(R, GAGES2_SHP)):
            polys = get_polygon_gages2
            print("using POINT-IN-POLYGON (GAGES-II) with <=25 km fallback")
        else:
            print(f"shapefile absent ({GAGES2_SHP}); distance-only join")
    except Exception as e:
        print(f"geopandas/shapefile unavailable ({type(e).__name__}); distance-only join")

    la, lo = S.lat.to_numpy(), S.lon.to_numpy()
    out = []
    for b in ids:
        if b not in coord:
            continue
        blat, blon = coord[b]
        d = haversine_km(blat, blon, la, lo)
        near = np.where(d <= max(MAX_KM, 50.0))[0]
        if len(near) == 0:
            near = np.array([int(d.argmin())])
        inside = set()
        if polys is not None:
            try:
                from shapely.geometry import Point
                g = polys(b)
                geom = g.geometry.iloc[0]
                for i in near:
                    if geom.contains(Point(float(lo[i]), float(la[i]))):
                        inside.add(int(i))
            except Exception:
                pass
        for i in near:
            i = int(i)
            is_in = i in inside
            if not is_in and d[i] > MAX_KM:
                continue
            out.append({"basin": b, "station": S.station.iloc[i],
                        "network": S.network.iloc[i],
                        "km": round(float(d[i]), 2),
                        "match": "in_polygon" if is_in else "within_25km",
                        "has_tmax": bool(S.has_tmax.iloc[i])})
        if not any(r["basin"] == b for r in out):  # guarantee every basin has one
            i = int(d.argmin())
            out.append({"basin": b, "station": S.station.iloc[i],
                        "network": S.network.iloc[i], "km": round(float(d[i]), 2),
                        "match": "nearest_far", "has_tmax": bool(S.has_tmax.iloc[i])})

    M = pd.DataFrame(out)
    p = os.path.join(CACHE, "basin_station_map.csv")
    M.to_csv(p, index=False)
    print(f"\npairs: {len(M):,}   basins covered: {M.basin.nunique()} / {len(ids)}")
    print(f"distinct stations to fetch: {M.station.nunique():,}")
    print("\nmatch rule:"); print(M.match.value_counts().to_string())
    print("\nnetwork mix:"); print(M.network.value_counts().to_string())
    per = M.groupby("basin").size()
    print(f"\nstations per basin: median {per.median():.0f}  "
          f"min {per.min()}  max {per.max()}")
    print(f"wrote {p}")
    return M


def stage_fetch(limit=None):
    p = os.path.join(CACHE, "basin_station_map.csv")
    if not os.path.exists(p):
        sys.exit("run --stage join first")
    M = pd.read_csv(p, dtype={"station": str, "basin": str})
    need = sorted(M.station.unique())
    os.makedirs(STATION_DIR, exist_ok=True)
    todo = [s for s in need
            if not os.path.exists(os.path.join(STATION_DIR, f"{s}.csv.gz"))]
    if limit:
        todo = todo[:limit]
    print(f"stations needed {len(need):,}; already cached {len(need)-len([s for s in need if not os.path.exists(os.path.join(STATION_DIR, f'{s}.csv.gz'))]):,}; fetching {len(todo):,}")
    ok = fail = 0
    t0 = time.time()
    for i, s in enumerate(todo, 1):
        dest = os.path.join(STATION_DIR, f"{s}.csv.gz")
        try:
            data = _get(BY_STATION.format(s), timeout=120)
            with open(dest, "wb") as fh:
                fh.write(data)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  FAIL {s}: {type(e).__name__}")
        if i % 200 == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / max(el, 1e-9)
            print(f"  [{i}/{len(todo)}] ok={ok} fail={fail} "
                  f"{rate:.1f}/s  eta {(len(todo)-i)/max(rate,1e-9)/60:.0f} min")
    print(f"done: ok={ok} fail={fail}")


def _read_station(sid):
    """Parse one GHCN station file: QC-filter, unit-convert, pivot to daily."""
    f = os.path.join(STATION_DIR, f"{sid}.csv.gz")
    if not os.path.exists(f):
        return None, 0
    try:
        with gzip.open(f, "rt", errors="ignore") as fh:
            d = pd.read_csv(fh, header=None, usecols=[0, 1, 2, 3, 5, 6],
                            names=["station", "date", "element", "value",
                                   "qflag", "sflag"], dtype=str)
    except Exception:
        return None, 0
    d = d[d.element.isin(ELEMENTS)]
    if d.empty:
        return None, 0
    d["date"] = pd.to_datetime(d.date, format="%Y%m%d", errors="coerce")
    d = d[(d.date >= f"{START_YEAR}-01-01") & (d.date <= f"{END_YEAR}-12-31")]
    if d.empty:
        return None, 0
    n0 = len(d)
    # QC: any non-blank QFLAG means the value failed a check.
    q = d.qflag.fillna("").str.strip()
    d = d[(q == "") | (~q.str[0].isin(BAD_QFLAGS))]
    dropped = n0 - len(d)
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    d = d.dropna(subset=["value"])
    # TENTHS -> real units, but ONLY for the elements stored in tenths.
    # SNOW/SNWD are already whole mm; dividing them would lose a factor of 10.
    d["value"] = np.where(d.element.isin(TENTHS), d.value / 10.0, d.value)
    w = d.pivot_table(index="date", columns="element", values="value", aggfunc="first")
    w["sflag"] = d.groupby("date").sflag.first()
    return w, dropped


def stage_build():
    p = os.path.join(CACHE, "basin_station_map.csv")
    M = pd.read_csv(p, dtype={"station": str, "basin": str})
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = {}
    written = skipped = 0
    stats = []
    basins = sorted(M.basin.unique())
    for n, b in enumerate(basins, 1):
        sub = M[M.basin == b]
        frames, nets, dropped_tot = [], set(), 0
        for _, r in sub.iterrows():
            if r.station not in cache:
                cache[r.station] = _read_station(r.station)
            w, dr = cache[r.station]
            dropped_tot += dr
            if w is None or w.empty:
                continue
            x = w.copy()
            x["_km"] = r.km
            nets.add(r.network)
            frames.append(x)
        if not frames:
            skipped += 1
            continue
        A = pd.concat(frames)
        g = A.groupby(level=0)
        out = pd.DataFrame({
            "precipitation_sum": g["PRCP"].mean() if "PRCP" in A else np.nan,
            "temperature_2m_max": g["TMAX"].mean() if "TMAX" in A else np.nan,
            "temperature_2m_min": g["TMIN"].mean() if "TMIN" in A else np.nan,
            # Snow: SNOW is daily snowFALL, SNWD is snow DEPTH on the ground.
            # Depth is a STATE (antecedent storage) while snowfall is a flux, so
            # they carry different information -- keep both. These target the 268
            # of 531 basins with frac_snow>0.15, where melt timing drives runoff
            # and no gridded forcing in our set observes the pack directly.
            "snowfall_sum": g["SNOW"].mean() if "SNOW" in A else np.nan,
            "snow_depth": g["SNWD"].mean() if "SNWD" in A else np.nan,
            "n_stations": g.size(),
            "min_dist_km": g["_km"].min(),
        })
        out.index.name = "date"
        out = out.reset_index()
        out["src_networks"] = "+".join(sorted(nets))
        out["qc_dropped"] = dropped_tot
        out.to_csv(os.path.join(OUT_DIR, f"{b}.csv.gz"), index=False,
                   compression="gzip")
        written += 1
        ann = out.precipitation_sum.mean() * 365.25
        stats.append({"basin": b, "days": len(out), "ann_mm": ann,
                      "n_st": sub.station.nunique(), "qc_dropped": dropped_tot,
                      "nets": "+".join(sorted(nets))})
        if n % 50 == 0:
            print(f"  [{n}/{len(basins)}] written={written} skipped={skipped}")
    S = pd.DataFrame(stats)
    S.to_csv(os.path.join(CACHE, "build_stats.csv"), index=False)
    print(f"\nwrote {written} basin files to {OUT_DIR} (skipped {skipped})")
    if len(S):
        print(f"\n=== UNITS GATE (annual precip must be physical) ===")
        print(f"  median {S.ann_mm.median():.0f} mm/yr   "
              f"range {S.ann_mm.min():.0f}-{S.ann_mm.max():.0f}")
        bad = S[(S.ann_mm < 100) | (S.ann_mm > 6000)]
        if len(bad):
            print(f"  WARNING {len(bad)} basins outside 100-6000 mm/yr")
        else:
            print("  all basins physical -> tenths conversion is correct")
        print(f"\n  QC rows dropped, total: {S.qc_dropped.sum():,}")
        print(f"  stations per basin: median {S.n_st.median():.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["inventory", "join", "fetch", "build"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    {"inventory": stage_inventory, "join": stage_join,
     "fetch": lambda: stage_fetch(a.limit), "build": stage_build}[a.stage]()


if __name__ == "__main__":
    main()
