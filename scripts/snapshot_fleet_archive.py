#!/usr/bin/env python3
"""Consolidate this build's fleet archive rows into one file per issue date.

PREREG §V16-0. Reads `archive_staging_fleet/{_fleet,_fleet_meta}/shard_*.csv.gz`
(placed there by `merge_shards.py`, outside `dist/`) and writes

    archive_fleet/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz        one row per gauge x lead
    archive_fleet_meta/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz   one provenance row per gauge

A later build on the same issue date REPLACES that date's files (the last
build of a UTC day is the one kept; `issued_at` in the meta file says which),
so a re-run can never double-count. Empty payloads are refused, not written.
The schema version is re-stamped here, at consolidation, as the m1 archive does.
"""
from __future__ import annotations

import csv
import gzip
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.fleet_archive import HEADER, META_HEADER, SCHEMA_VERSION  # noqa: E402

STAGING = ROOT / "archive_staging_fleet"
KINDS = (("_fleet", "archive_fleet", HEADER), ("_fleet_meta", "archive_fleet_meta", META_HEADER))


def _consolidate(sub: str, out_name: str, header: list) -> int:
    by_issued: dict = {}
    for sp in sorted((STAGING / sub).glob("shard_*.csv.gz")):
        with gzip.open(sp, "rt", newline="") as fh:
            rd = csv.reader(fh)
            hdr = next(rd, None)
            if hdr != list(header):
                print(f"skipping {sp.name}: header does not match {SCHEMA_VERSION}")
                continue
            for r in rd:
                if len(r) != len(header):
                    continue
                r[-1] = SCHEMA_VERSION
                by_issued.setdefault(r[0], []).append(r)
    written = 0
    for issued, rs in sorted(by_issued.items()):
        try:
            d = datetime.strptime(issued, "%Y-%m-%d")
        except ValueError:
            print(f"skipping unparseable issued_date {issued!r} ({len(rs)} rows)")
            continue
        out_dir = ROOT / out_name / f"{d.year:04d}" / f"{d.month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{issued}.csv.gz"
        tmp = out_path.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wt", newline="") as gz:
            w = csv.writer(gz)
            w.writerow(header)
            w.writerows(rs)
        tmp.replace(out_path)
        written += len(rs)
        print(f"wrote {len(rs)} rows for {issued} -> {out_path.relative_to(ROOT)} "
              f"({len({r[header.index('station_id')] for r in rs})} gauges)")
    return written


def main() -> int:
    if not STAGING.exists():
        print(f"no staging dir at {STAGING}; nothing to snapshot")
        return 1
    n = _consolidate(*KINDS[0])
    if not n:
        print("staging held no fleet rows; refusing to write an empty archive file")
        return 1
    _consolidate(*KINDS[1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
