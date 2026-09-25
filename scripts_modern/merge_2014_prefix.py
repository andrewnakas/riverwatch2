#!/usr/bin/env python
"""Prepend a 2014 staging build onto the 2015-2025 areal corpus.

Why not just re-run the builder with --start-year 2014? Because it skips a basin
only when the existing file already has the expected row count; widening the
window changes that count from 4018 to 4383, so every basin would be rebuilt
from scratch — 12 years of fetching to gain 1.

So: build 2014 alone into a staging dir, then merge here.

Payoffs (both real):
  1. NH consumes the first 365 days as sequence warmup, so a corpus starting
     2015-01-01 makes ALL of 2015 unevaluable. Starting at 2014 recovers a full
     evaluation year.
  2. Any one-day precip shift needs 2014-12-31 as the boundary value, else
     2015-01-01 goes NaN and NH drops every sample whose lookback touches it.

Verifies the artifact: exact expected row count, no duplicate//missing dates,
contiguous calendar, and writes atomically via a temp file.
"""
from __future__ import annotations
import argparse, datetime as dt, gzip, shutil, sys
from pathlib import Path

ROOT = Path("/home/nakas/riverwatch2")
COLS = ["date", "q_cfs", "temperature_2m_mean", "temperature_2m_max",
        "temperature_2m_min", "precipitation_sum", "shortwave_radiation_sum",
        "vapor_pressure"]


def read(p):
    with gzip.open(p, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        rows = [l.rstrip("\n").split(",") for l in fh if l.strip()]
    return hdr, rows


def caldays(y0, y1):
    d, end, out = dt.date(y0, 1, 1), dt.date(y1, 12, 31), []
    while d <= end:
        out.append(d.isoformat()); d += dt.timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="data/modern_corpus/camels_corpus_daymet_modern_areal")
    ap.add_argument("--staging", default="data/modern_corpus/_areal_2014")
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    main_d, stag_d = ROOT / a.main, ROOT / a.staging
    want = caldays(a.start_year, a.end_year)
    want_set = set(want)
    pre_n = len(caldays(a.start_year, a.start_year))   # days in the prefix year

    ok = skipped = failed = 0
    for mp in sorted(main_d.glob("*.csv.gz")):
        g = mp.name.split(".")[0]
        sp = stag_d / mp.name
        try:
            hdr, mrows = read(mp)
            if len(mrows) == len(want):
                skipped += 1; continue          # already merged
            if not sp.exists():
                print(f"  {g}: no staging file"); failed += 1; continue
            shdr, srows = read(sp)
            if shdr != hdr:
                print(f"  {g}: header mismatch"); failed += 1; continue
            srows = [r for r in srows if r[0][:4] == str(a.start_year)]
            if len(srows) != pre_n:
                print(f"  {g}: staging has {len(srows)} rows, want {pre_n}")
                failed += 1; continue
            merged = srows + mrows
            dates = [r[0] for r in merged]
            if len(dates) != len(want) or dates != want:
                miss = want_set - set(dates)
                print(f"  {g}: calendar mismatch (n={len(dates)}, "
                      f"missing {sorted(miss)[:3]})")
                failed += 1; continue
            if a.dry_run:
                ok += 1; continue
            tmp = mp.with_suffix(".tmp.gz")
            with gzip.open(tmp, "wt") as fh:
                fh.write(",".join(hdr) + "\n")
                for r in merged:
                    fh.write(",".join(r) + "\n")
            with gzip.open(tmp, "rt") as fh:      # verify before replacing
                assert sum(1 for _ in fh) - 1 == len(want)
            tmp.replace(mp)
            ok += 1
        except Exception as e:
            print(f"  {g}: ERROR {e}"); failed += 1

    print(f"\nmerged={ok} already-done={skipped} failed={failed}"
          + ("   (dry run, nothing written)" if a.dry_run else ""))
    print(f"expected rows per basin: {len(want)} "
          f"({a.start_year}-01-01 .. {a.end_year}-12-31)")


if __name__ == "__main__":
    main()
