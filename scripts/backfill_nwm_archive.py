#!/usr/bin/env python3
"""v15.1-bf: backfill historical NWM medium_range_blend forecasts from
the noaa-nwm-pds operational archive on S3.

The v14.2 nwm-archive branch only accumulates one row per
(issued_date, station, target_date, horizon_day) tuple per build, so
even at the new 6-hour cron we'd need ~30 days to train the per-horizon
residual learner. NOAA's noaa-nwm-pds bucket holds ~16 months of the
*actual issued* operational forecasts (not retrospective with observed
forcings — we want forecast errors, not perfect-forcing simulations),
so we can backfill straight from there.

Per day we read one cycle (t00z) of the medium_range_blend channel_rt
files (f001..f240, 10 days at hourly resolution), slice out our 1893
stations' COMIDs, and aggregate hourly cfs into daily means. That gives
us 10 (issued_date × target_date × q_cfs_raw) rows per station per
issued day. With 30 days of backfill: 30 × 1893 × 10 = ~568k rows,
well above the 5k/horizon training threshold.

Cost:
  - One file (~14MB CONUS) opened via fsspec/h5netcdf: ~25s
  - 240 files × 1 cycle/day = 240 file-opens/day
  - With 16 parallel threads, ~25min/day
  - 30 days @ 16 threads ≈ ~12 hours wall-clock — fits a single GHA run

We DO NOT need observed cfs in this script — that join happens in
train_nwm_residual.py, which can read from data/cache/usgs_records/.
This script just produces the q_cfs_raw side of the panel.

Output: archive/<YYYY>/<MM>/<YYYY-MM-DD>.csv.gz with columns
   issued_date, station_id, target_date, horizon_day,
   q_cfs_raw, q_cfs_obs_today, bias_scale_used, schema_version
matching the existing v14.2 schema written by snapshot_nwm_archive.py.
We leave q_cfs_obs_today and bias_scale_used empty — the v14.1 bias
correction has to be re-derived at training time (or in a follow-up)
from the post-issuance analysis_assimilation overlap, which is itself
in the operational archive but isn't needed to *learn the residual* —
we can train against raw NWM and let the model pick up the bias.

Usage:
    python scripts/backfill_nwm_archive.py --start 2026-04-01 --end 2026-05-01
    python scripts/backfill_nwm_archive.py --start 2026-04-01 --end 2026-05-01 --threads 16
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
CROSSWALK = ROOT / "data" / "nwm_crosswalk.json"
SCHEMA_VERSION = "v15.1-bf"
S3_BUCKET = "noaa-nwm-pds"
CMS_TO_CFS = 35.3146667

# Per the bucket schema, t00z medium_range_blend gives forecasts in 1-hour
# steps from f001 (issued + 1h) to f240 (issued + 240h = 10 days). Each
# file is one timestep, ~14MB, holding all 2.78M reaches' streamflow.
MAX_HORIZON_HOURS = 240
HORIZON_DAYS = list(range(1, 11))  # h=1..10 covered by operational archive


def _file_url(issued: date, fhr: int) -> str:
    """t00z cycle, fhrxxx hourly forecast file URL."""
    d = issued.strftime("%Y%m%d")
    return (
        f"s3://{S3_BUCKET}/nwm.{d}/medium_range_blend/"
        f"nwm.t00z.medium_range_blend.channel_rt.f{fhr:03d}.conus.nc"
    )


def _load_crosswalk() -> tuple[list[str], np.ndarray]:
    """Returns (station_ids parallel to comids, comids as int64 array)."""
    cw = json.loads(CROSSWALK.read_text())
    pairs = []
    for sid, meta in cw.items():
        if not isinstance(meta, dict):
            continue
        comid = meta.get("comid")
        if not comid:
            continue
        try:
            pairs.append((str(sid), int(comid)))
        except (TypeError, ValueError):
            continue
    pairs.sort(key=lambda p: p[1])  # ascending COMID for binary-search alignment
    sids = [p[0] for p in pairs]
    comids = np.array([p[1] for p in pairs], dtype=np.int64)
    return sids, comids


# Globals populated by main() so threads can share without pickling.
_S3FS = None  # type: ignore
_FEATURE_INDEX_CACHE: dict[str, np.ndarray] = {}


def _get_s3():
    global _S3FS
    if _S3FS is None:
        import s3fs
        _S3FS = s3fs.S3FileSystem(anon=True)
    return _S3FS


def _read_one_file(url: str, comids_sorted: np.ndarray) -> Optional[np.ndarray]:
    """Open one channel_rt NetCDF over S3 and return q_cms aligned to
    comids_sorted (same length, NaN where missing)."""
    import xarray as xr
    fs = _get_s3()
    s3_key = url.replace("s3://", "")
    try:
        with fs.open(s3_key, "rb") as f:
            ds = xr.open_dataset(f, engine="h5netcdf")
            file_fids = ds["feature_id"].values  # int64, ascending
            # We assume the feature_id array layout doesn't drift between
            # cycles within a NWM version. Cache the slice indices keyed
            # on the array hash so we only build them once per worker.
            cache_key = f"{file_fids[0]}-{file_fids[-1]}-{len(file_fids)}"
            slice_idx = _FEATURE_INDEX_CACHE.get(cache_key)
            if slice_idx is None:
                slice_idx = np.searchsorted(file_fids, comids_sorted)
                # Validate: any slot where searchsorted lands but the value
                # doesn't actually match → that COMID isn't in this file.
                in_range = slice_idx < len(file_fids)
                slice_idx = np.where(in_range, slice_idx, 0)
                hits = file_fids[slice_idx] == comids_sorted
                slice_idx = np.where(hits, slice_idx, -1)
                _FEATURE_INDEX_CACHE[cache_key] = slice_idx
            valid = slice_idx >= 0
            q = np.full(len(comids_sorted), np.nan, dtype=np.float64)
            if valid.any():
                pulled = ds["streamflow"].isel(feature_id=slice_idx[valid]).values
                q[valid] = pulled
            return q
    except Exception:
        return None


def _backfill_one_day(
    issued: date,
    sids: list[str],
    comids_sorted: np.ndarray,
    threads: int,
) -> Optional[list[tuple]]:
    """Fetch all 240 hourly files for one t00z cycle, aggregate to daily
    means per station per horizon-day, and return rows in the v14.2
    schema."""
    daily_sums = np.zeros((len(HORIZON_DAYS), len(sids)), dtype=np.float64)
    daily_counts = np.zeros((len(HORIZON_DAYS), len(sids)), dtype=np.int32)
    fhrs = list(range(1, MAX_HORIZON_HOURS + 1))

    def fetch(fhr: int):
        url = _file_url(issued, fhr)
        return fhr, _read_one_file(url, comids_sorted)

    n_ok = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for fhr, q in pool.map(fetch, fhrs):
            if q is None:
                continue
            n_ok += 1
            # fhr 1..24 → day 1, 25..48 → day 2, etc. Day index = (fhr-1)//24
            day_idx = (fhr - 1) // 24
            if day_idx >= len(HORIZON_DAYS):
                continue
            mask = np.isfinite(q)
            daily_sums[day_idx, mask] += q[mask]
            daily_counts[day_idx, mask] += 1

    if n_ok < MAX_HORIZON_HOURS // 2:
        return None  # too gappy — skip the day

    rows: list[tuple] = []
    issued_str = issued.isoformat()
    for day_idx, h in enumerate(HORIZON_DAYS):
        target = (issued + timedelta(days=h)).isoformat()
        means = np.where(
            daily_counts[day_idx] > 0,
            daily_sums[day_idx] / np.maximum(daily_counts[day_idx], 1),
            np.nan,
        )
        for sid, q_cms in zip(sids, means):
            if not np.isfinite(q_cms):
                continue
            q_cfs = float(q_cms) * CMS_TO_CFS
            rows.append((
                issued_str,
                sid,
                target,
                h,
                f"{q_cfs:.6f}",
                "",       # q_cfs_obs_today filled by training step
                "",       # bias_scale_used filled by training step
                SCHEMA_VERSION,
            ))
    return rows


def _write_day(rows: list[tuple], issued: date) -> Path:
    out_dir = ARCHIVE / issued.strftime("%Y") / issued.strftime("%m")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{issued.isoformat()}.csv.gz"
    with gzip.open(out_path, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow([
            "issued_date", "station_id", "target_date", "horizon_day",
            "q_cfs_raw", "q_cfs_obs_today", "bias_scale_used",
            "schema_version",
        ])
        w.writerows(rows)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--threads", type=int, default=16,
                   help="Parallel S3 file reads per day (16 = ~25min/day)")
    p.add_argument("--skip-existing", action="store_true", default=True)
    args = p.parse_args()

    sids, comids = _load_crosswalk()
    print(f"loaded crosswalk: {len(sids)} stations")

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    print(f"will process {len(days)} days ({start} → {end})")

    t_total = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0
    for d in days:
        out_path = ARCHIVE / d.strftime("%Y") / d.strftime("%m") / f"{d.isoformat()}.csv.gz"
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 1024:
            n_skipped += 1
            continue
        t0 = time.time()
        rows = _backfill_one_day(d, sids, comids, args.threads)
        if rows is None:
            print(f"  {d}: not enough hourly files available — skipped")
            n_failed += 1
            continue
        path = _write_day(rows, d)
        n_done += 1
        elapsed = time.time() - t0
        size_mb = path.stat().st_size / 1e6
        print(
            f"  {d}: wrote {len(rows):>7d} rows ({size_mb:.1f}MB) "
            f"in {elapsed:.0f}s   total {n_done} done, {n_skipped} skipped, "
            f"{n_failed} failed",
            flush=True,
        )
    print(
        f"\ntotal: {n_done} new days, {n_skipped} pre-existing, {n_failed} failed "
        f"in {(time.time()-t_total)/60:.1f}min"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
