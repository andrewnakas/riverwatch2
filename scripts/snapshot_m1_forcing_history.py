#!/usr/bin/env python3
"""Archive each build's FORECAST-horizon forcing, so a decoder-product gate
becomes possible later.

WHY THIS CANNOT WAIT. The served member's decoder runs on Open-Meteo point
FORECAST forcing. Areal GFS forecast forcing now exists for all 14 leads
(scripts_modern/build_gfs_areal_14lead.py) and is plausibly closer to the
training distribution -- the measured -0.048033 day-1 decay was itself produced
on areal GFS. But "plausibly" is not a measurement, and the comparison needs
BOTH products for the SAME init dates on the SAME rows.

Areal GFS can be rebuilt for any past date from S3. Open-Meteo's forecast
CANNOT: its API serves the current run, and `m1-forcing-archive` is a
single-commit force-pushed branch, so yesterday's forecast rows are gone the
moment today's land. Unless they are archived as they pass, the paired
comparison can never be run at all -- not later, not with effort.

So this captures, once per build, the forecast-horizon rows the member was
actually driven by:

  archive_m1_forcing/{YYYY}/{MM}/{init-date}.csv.gz
  station_id, init_date, target_date, lead_day, <the 6 channels>, src, schema_version

Only rows dated AFTER the init date are kept -- the observed history is already
reproducible from the corpora and would multiply the volume for nothing.
~625 basins x 16 leads is ~10k rows per build, a few hundred KB gzipped.
"""
from __future__ import annotations

import csv
import glob
import gzip
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "data" / "m1_forcing"
ARCHIVE = ROOT / "archive_m1_forcing"

SCHEMA_VERSION = "m1-forcing-hist-1"
CHANNELS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
HEADER = (["station_id", "init_date", "target_date", "lead_day"]
          + CHANNELS + ["src", "schema_version"])


def main() -> int:
    if not TABLES.exists():
        print(f"no forcing tables at {TABLES}; nothing to archive")
        return 1
    init = date.today()
    rows = []
    n_basins = 0
    for p in sorted(glob.glob(str(TABLES / "*.csv.gz"))):
        sid = Path(p).name.split(".")[0]
        if sid.startswith("_"):
            continue
        try:
            with gzip.open(p, "rt", newline="") as fh:
                recs = list(csv.DictReader(fh))
        except Exception:
            continue
        n_basins += 1
        for r in recs:
            d = r.get("date") or ""
            if not d or d <= init.isoformat():
                continue          # observed history: reproducible elsewhere
            try:
                lead = (date.fromisoformat(d) - init).days
            except ValueError:
                continue
            if lead < 1:
                continue
            rows.append([sid, init.isoformat(), d, lead]
                        + [r.get(c, "") for c in CHANNELS]
                        + [r.get("src", ""), SCHEMA_VERSION])

    if not rows:
        print("no forecast-horizon rows found; refusing to write an empty file")
        return 1

    out_dir = ARCHIVE / f"{init.year:04d}" / f"{init.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{init.isoformat()}.csv.gz"
    tmp = out.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt", newline="") as gz:
        w = csv.writer(gz)
        w.writerow(HEADER)
        w.writerows(rows)
    tmp.replace(out)

    back = 0
    with gzip.open(out, "rt", newline="") as fh:
        back = sum(1 for _ in csv.DictReader(fh))
    if back != len(rows):
        out.unlink(missing_ok=True)
        print(f"wrote {len(rows)} rows, read back {back}; refusing")
        return 1

    leads = sorted({int(r[3]) for r in rows})
    srcs = sorted({r[-2] for r in rows})
    print(f"wrote {len(rows)} rows for init {init} -> {out.relative_to(ROOT)}")
    print(f"  {n_basins} basins, leads {leads[0]}..{leads[-1]}, src {srcs}")
    print(f"  generated {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
