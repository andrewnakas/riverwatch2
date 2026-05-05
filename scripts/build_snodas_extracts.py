#!/usr/bin/env python3
"""v14.5c: build per-station SNODAS daily SWE + snowmelt extracts.

For each USGS gauge in `data/stations_40_enriched.json`, fetches NOHRSC
SNODAS daily tars from NSIDC's G02158 archive, samples band 1034 (SWE
snapshot, mm) and band 1044 (snowmelt-runoff at base of pack, 24-h total
in m × 1e5), converts to consistent units, and writes one JSON file per
station to `data/snodas_extracts/{station_id}.json`:

    {"2026-04-30": {"swe_in": 0.43, "melt_24h_mm": 1.2}, ...}

Incremental: each run reads the existing extract for each station, finds
the latest date present, and only fetches dates beyond that (up to the
configured tail, default 7 days, or `--days N`). For a one-time backfill,
pass `--start 2003-10-01 --end 2026-05-01` to walk the full archive.

Run from repo root:
    python scripts/build_snodas_extracts.py            # incremental (last 7 days)
    python scripts/build_snodas_extracts.py --days 30  # last 30 days
    python scripts/build_snodas_extracts.py --start 2024-10-01 --end 2025-09-30

CI invokes this with `--days 14` once per build before the forecast pass.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# v15.0: env override mirrors build_static_site.py so the same workflow
# step picks up the same station list.
STATIONS_PATH = Path(os.environ.get("RW2_STATIONS_FILE") or (ROOT / "data" / "stations_40_enriched.json"))
EXTRACTS_DIR = ROOT / "data" / "snodas_extracts"
EXTRACTS_DIR.mkdir(parents=True, exist_ok=True)

# NSIDC G02158 masked CONUS archive URL pattern (HTTPS). The masked grid is
# the post-2013 6935×3351 footprint; older files use a slightly different
# header which we don't care about because backtests only need the portion
# of history overlapping our active stations.
NSIDC_BASE = "https://noaadata.apps.nsidc.org/NOAA/G02158/masked"

# CONUS-masked grid params (post-2013-10-01). Source: NSIDC G02158 user guide.
GRID_NCOLS = 6935
GRID_NROWS = 3351
GRID_DX = 0.00833333333  # 30 arc-seconds in degrees
GRID_DY = 0.00833333333
GRID_ULX = -124.72916666666667
GRID_ULY = 52.87083333333334
GRID_NODATA = -9999

# Variable codes we extract. Scale factors confirmed from per-band .txt
# headers ("Data units: Meters / N"):
#   1034 SWE   → "Meters / 1000"   → raw int / 1000 = m   → ×1000 to get mm → mm = raw
#   1044 melt  → "Meters / 100000" → raw int / 100000 = m → ×1000 to get mm
VAR_SWE = "1034"     # column-integrated SWE snapshot, raw int = mm directly
VAR_MELT = "1044"    # snowmelt at base of pack, 24-h total, raw / 100 = mm

USER_AGENT = "riverwatch2/0.1 snodas-extractor"


def _http_open(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def _tar_url_for(d: date) -> str:
    mon_abbr = d.strftime("%b")  # "Jan", "Feb", ...
    return (
        f"{NSIDC_BASE}/{d.year:04d}/{d.month:02d}_{mon_abbr}/"
        f"SNODAS_{d.strftime('%Y%m%d')}.tar"
    )


def _latlon_to_index(lat: float, lon: float) -> Optional[tuple[int, int]]:
    """Convert (lat, lon) to (row, col) on the masked CONUS grid.
    Returns None if the point is outside the grid extent."""
    if lat is None or lon is None:
        return None
    row = int(round((GRID_ULY - lat) / GRID_DY))
    col = int(round((lon - GRID_ULX) / GRID_DX))
    if row < 0 or row >= GRID_NROWS or col < 0 or col >= GRID_NCOLS:
        return None
    return row, col


def _decode_band(buf: bytes) -> np.ndarray:
    """Big-endian int16 → 2D array shape (NROWS, NCOLS)."""
    arr = np.frombuffer(buf, dtype=">i2")
    if arr.size != GRID_NROWS * GRID_NCOLS:
        raise ValueError(
            f"unexpected band size {arr.size} (expected {GRID_NROWS*GRID_NCOLS})"
        )
    return arr.reshape(GRID_NROWS, GRID_NCOLS)


def _extract_band_from_tar(tar_bytes: bytes, var_code: str) -> Optional[np.ndarray]:
    """Pull the .dat.gz for one variable code out of a SNODAS daily tar
    and return the decoded grid. Returns None if the band file isn't in
    the tar (some early dates only ship a subset)."""
    bio = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=bio, mode="r:") as t:
        for member in t.getmembers():
            name = member.name
            if not name.endswith(".dat.gz"):
                continue
            if f"ssmv1{var_code}" not in name:
                continue
            f = t.extractfile(member)
            if f is None:
                continue
            gz = gzip.decompress(f.read())
            return _decode_band(gz)
    return None


def _sample_band(grid: np.ndarray, idxs: list[Optional[tuple[int, int]]]) -> list[Optional[float]]:
    """Look up grid values at the provided (row, col) indices. NaN-safe."""
    out: list[Optional[float]] = []
    for ix in idxs:
        if ix is None:
            out.append(None)
            continue
        row, col = ix
        v = int(grid[row, col])
        if v == GRID_NODATA:
            out.append(None)
        else:
            out.append(float(v))
    return out


def _existing_dates(station_id: str) -> set[str]:
    p = EXTRACTS_DIR / f"{station_id}.json"
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()).keys())
    except Exception:
        return set()


def _write_station(station_id: str, rec: dict) -> None:
    p = EXTRACTS_DIR / f"{station_id}.json"
    p.write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True))


def _date_range(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def _process_one_day(d: date, stations: list[dict], idxs: list[Optional[tuple[int, int]]]) -> Optional[dict]:
    """Download, decode, and sample one day's SNODAS tar for all stations.
    Returns {station_id: {swe_in, melt_24h_mm}} on success, None on failure
    (network, missing file, decode error)."""
    url = _tar_url_for(d)
    # Retry transient network errors (DNS blips, timeouts) up to 4 times with
    # exponential backoff. 404s short-circuit immediately — those days are
    # legitimately missing from the archive.
    tar_bytes = None
    last_exc: Optional[Exception] = None
    for attempt in range(4):
        try:
            with _http_open(url, timeout=120) as resp:
                tar_bytes = resp.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"  {d}: 404 (likely not yet released or never published)", flush=True)
                return None
            last_exc = e
        except Exception as e:
            last_exc = e
        # backoff: 2, 4, 8, 16s
        time.sleep(2 ** (attempt + 1))
    if tar_bytes is None:
        print(f"  {d}: fetch failed after retries: {last_exc}", flush=True)
        return None

    swe_grid = _extract_band_from_tar(tar_bytes, VAR_SWE)
    melt_grid = _extract_band_from_tar(tar_bytes, VAR_MELT)
    if swe_grid is None:
        print(f"  {d}: no SWE band in tar", flush=True)
        return None

    swe_vals = _sample_band(swe_grid, idxs)  # int16 / 1000 = mm
    melt_vals = (
        _sample_band(melt_grid, idxs) if melt_grid is not None
        else [None] * len(idxs)
    )

    out: dict = {}
    for st, swe_v, melt_v in zip(stations, swe_vals, melt_vals):
        sid = str(st.get("id"))
        if not sid:
            continue
        rec: dict = {}
        if swe_v is not None:
            # band 1034 raw int = mm directly (header: "Meters / 1000").
            # convert to inches so it slots into existing _build_features
            # `swe_in` window logic.
            rec["swe_in"] = round(float(swe_v) / 25.4, 4)
        if melt_v is not None:
            # band 1044 raw int / 100 = mm (header: "Meters / 100000" → /1e5 m
            # → ×1000 mm = /100).
            rec["melt_24h_mm"] = round(float(melt_v) / 100.0, 3)
        if rec:
            out[sid] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14,
                    help="incremental fetch tail in days (default 14)")
    ap.add_argument("--start", type=str, default=None,
                    help="explicit start date YYYY-MM-DD; overrides --days")
    ap.add_argument("--end", type=str, default=None,
                    help="explicit end date YYYY-MM-DD; defaults to yesterday")
    ap.add_argument("--limit-stations", type=int, default=0,
                    help="cap to first N stations (debugging)")
    args = ap.parse_args()

    stations_payload = json.loads(STATIONS_PATH.read_text())
    stations = stations_payload.get("stations") or stations_payload
    if args.limit_stations:
        stations = stations[: args.limit_stations]
    print(f"loaded {len(stations)} stations", flush=True)

    idxs = [
        _latlon_to_index(float(s.get("lat")), float(s.get("lon")))
        if s.get("lat") is not None and s.get("lon") is not None else None
        for s in stations
    ]
    in_grid = sum(1 for ix in idxs if ix is not None)
    print(f"  {in_grid}/{len(stations)} stations land inside CONUS grid", flush=True)

    yesterday = date.today() - timedelta(days=1)
    if args.end:
        end = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end = yesterday
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start = end - timedelta(days=args.days - 1)

    days = _date_range(start, end)
    print(f"target window: {start} → {end} ({len(days)} days)", flush=True)

    # Pre-load existing per-station records once so we don't re-read 1893
    # JSON files per day.
    records: dict[str, dict] = {}
    for st in stations:
        sid = str(st.get("id"))
        if not sid:
            continue
        p = EXTRACTS_DIR / f"{sid}.json"
        if p.exists():
            try:
                records[sid] = json.loads(p.read_text())
            except Exception:
                records[sid] = {}
        else:
            records[sid] = {}

    # Off-CONUS stations (idx is None) never get an extract, and a handful
    # of in-grid stations land on permanently-no-data pixels (coastline,
    # grid edges) so they never produce records either. We can't know which
    # is which a priori, so the skip predicate uses a coverage threshold:
    # if ≥99% of in-grid stations have the date, the day is considered
    # complete. Empirically the dead-pixel set is ~10/1849 = 0.6%.
    in_grid_ids = [
        str(s.get("id")) for s, ix in zip(stations, idxs) if ix is not None
    ]
    skip_threshold = max(1, int(len(in_grid_ids) * 0.99))

    n_added = 0
    n_skipped = 0
    n_failed = 0
    # Small inter-day pause to be a polite NSIDC client during long backfills.
    # Tar-download dominates wall time anyway (~5-10s/day) so this is in noise,
    # but it avoids triggering rate-limit / connection-refused responses on
    # multi-thousand-day walks.
    fetch_pause_s = float(os.environ.get("SNODAS_FETCH_PAUSE_S", "0.5"))
    t0 = time.time()
    for i, d in enumerate(days, 1):
        d_iso = d.isoformat()
        # If ≥99% of in-grid stations already have this date, skip the
        # network call (the missing fraction is permanently dead pixels).
        coverage = sum(1 for sid in in_grid_ids if d_iso in records.get(sid, {}))
        if coverage >= skip_threshold:
            n_skipped += 1
            continue
        day_out = _process_one_day(d, stations, idxs)
        if fetch_pause_s > 0:
            time.sleep(fetch_pause_s)
        if day_out is None:
            n_failed += 1
            continue
        for sid, rec in day_out.items():
            records.setdefault(sid, {})[d_iso] = rec
        n_added += len(day_out)
        if i % 5 == 0 or i == len(days):
            elapsed = time.time() - t0
            print(
                f"  [{i}/{len(days)}] {d_iso}: "
                f"+{len(day_out)} stations, total added={n_added}, "
                f"skipped={n_skipped}, failed={n_failed}, "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    # Persist any station whose record changed.
    n_written = 0
    for sid, rec in records.items():
        if not rec:
            continue
        _write_station(sid, rec)
        n_written += 1

    print(f"\nwrote {n_written} per-station extract files to {EXTRACTS_DIR}")
    print(
        f"days added={n_added}, skipped={n_skipped}, "
        f"failed={n_failed}, total elapsed={time.time()-t0:.0f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
