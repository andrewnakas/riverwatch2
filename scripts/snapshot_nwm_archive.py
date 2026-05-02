#!/usr/bin/env python3
"""v14.2: consolidate per-shard raw NWM forecast snapshots into one daily file.

Reads `archive_staging/shard_*.csv.gz` (written by `merge_shards.py` after
each shard's `build_static_site.py` collects its slice of stations' raw NWM
medium_range_blend curves) and emits a single consolidated file at
`archive/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz`.

The CI job that follows commits this output to the orphan `nwm-archive`
branch so the data is durable across deploys without bloating `main`.

Schema (8 cols, all strings/numbers):
  issued_date, station_id, target_date, horizon_day,
  q_cfs_raw, q_cfs_obs_today, bias_scale_used, schema_version

The schema_version column is added here (not at shard write time) so a
backfill script can always tell which generation of the pipeline produced
a row.
"""
from __future__ import annotations

import csv
import gzip
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "archive_staging"
ARCHIVE = ROOT / "archive"
SCHEMA_VERSION = "v14.2"


def main() -> int:
    if not STAGING.exists():
        print(f"no staging dir: {STAGING} — nothing to snapshot")
        return 0
    shard_files = sorted(STAGING.glob("shard_*.csv.gz"))
    if not shard_files:
        print(f"no shard files in {STAGING} — nothing to snapshot")
        return 0

    issued = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = ARCHIVE / issued[:4] / issued[5:7]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{issued}.csv.gz"

    rows_written = 0
    with gzip.open(out_path, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow([
            "issued_date", "station_id", "target_date", "horizon_day",
            "q_cfs_raw", "q_cfs_obs_today", "bias_scale_used",
            "schema_version",
        ])
        for sf in shard_files:
            with gzip.open(sf, "rt", newline="") as gz_in:
                r = csv.reader(gz_in)
                header = next(r, None)
                if header is None:
                    continue
                for row in r:
                    if len(row) < 7:
                        continue
                    w.writerow([*row[:7], SCHEMA_VERSION])
                    rows_written += 1

    print(f"wrote {rows_written} rows → {out_path}")
    if rows_written == 0:
        # Don't keep an empty file; signal the caller.
        out_path.unlink(missing_ok=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
