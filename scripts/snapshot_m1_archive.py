#!/usr/bin/env python3
"""Consolidate this build's MODERN-1 forecast rows into one daily archive file.

WHY THIS EXISTS
---------------
Every skill number this project can quote comes from a research window. Nothing
scores the forecasts the live site actually ISSUED against what the rivers then
did -- only the raw NWM curve was ever archived. So "the model is live" has
never been followed by "and here is how it did."

This is the first half of that: each build appends what it served, and
`scripts/score_m1_live.py` later joins it against USGS observations as they
arrive. Modelled on `scripts/snapshot_nwm_archive.py`, including stamping the
schema version at CONSOLIDATION time rather than at shard-write time, so a
future backfill can always tell which generation of the pipeline produced a
row.

Reads `archive_staging_m1/shard_*.csv.gz` (placed there by `merge_shards.py`,
deliberately outside `dist/` so it never ships to Pages) and writes
`archive_m1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz`.

Schema (9 columns):
  issued_date, station_id, target_date, horizon_day, series,
  q_cfs, q_lo_raw, q_hi_raw, schema_version

`series` is one of `mblstm_modern`, `ensemble_blend`, `persistence_lag1`, all
on IDENTICAL (station, target_date, horizon) keys -- a paired comparison needs
identical rows, and this project has already voided one screen whose two arms
differed in more than one way.

⚠️ `q_lo_raw`/`q_hi_raw` are the member's OWN quantiles, before the per-member
bias scale and before anchoring to the last observation, while `q_cfs` is the
served curve with both applied. Coverage computed from the band is therefore
the model's own calibration, not the calibration of what was displayed. The
scorer must say so.
"""
from __future__ import annotations

import csv
import gzip
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "archive_staging_m1"
ARCHIVE = ROOT / "archive_m1"

SCHEMA_VERSION = "m1-fcst-1"
HEADER = ["issued_date", "station_id", "target_date", "horizon_day", "series",
          "q_cfs", "q_lo_raw", "q_hi_raw", "schema_version"]


def main() -> int:
    if not STAGING.exists():
        print(f"no staging dir at {STAGING}; nothing to snapshot")
        return 1
    shards = sorted(STAGING.glob("shard_*.csv.gz"))
    if not shards:
        print(f"no shard files in {STAGING}; nothing to snapshot")
        return 1

    rows: list[list[str]] = []
    issued_seen: set[str] = set()
    for sp in shards:
        with gzip.open(sp, "rt", newline="") as fh:
            rd = csv.reader(fh)
            hdr = next(rd, None)
            if not hdr:
                continue
            for r in rd:
                if not r:
                    continue
                # The shard already stamps the schema; re-stamp here so the
                # consolidated file is authoritative even if a shard was
                # produced by an older build.
                if len(r) == len(HEADER):
                    r[-1] = SCHEMA_VERSION
                else:
                    r = r[:len(HEADER) - 1] + [SCHEMA_VERSION]
                rows.append(r)
                issued_seen.add(r[0])

    if not rows:
        print("staging held only empty shards; refusing to write an empty archive file")
        return 1

    # One file per issue date. A build that straddles UTC midnight can emit two.
    by_issued: dict[str, list[list[str]]] = {}
    for r in rows:
        by_issued.setdefault(r[0], []).append(r)

    for issued, rs in sorted(by_issued.items()):
        try:
            d = datetime.strptime(issued, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"skipping unparseable issued_date {issued!r} ({len(rs)} rows)")
            continue
        out_dir = ARCHIVE / f"{d.year:04d}" / f"{d.month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{issued}.csv.gz"

        # Re-running a build for the same issue date replaces that date's file
        # rather than appending, so a re-run can never double-count rows.
        tmp = out_path.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wt", newline="") as gz:
            w = csv.writer(gz)
            w.writerow(HEADER)
            w.writerows(rs)
        tmp.replace(out_path)

        series = sorted({r[4] for r in rs})
        basins = len({r[1] for r in rs})
        print(f"wrote {len(rs)} rows for {issued} -> "
              f"{out_path.relative_to(ROOT)} ({basins} basins, series {series})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
