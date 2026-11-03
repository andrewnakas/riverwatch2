#!/usr/bin/env python3
"""Summarise every published forecast into one small `current.json` for the map.

The map colours 9,851 markers by how today's flow compares with that gauge's own
record for the same calendar day, and flags gauges forecast to rise. Fetching
9,851 forecast files in the browser to do that is not an option, so the merge
step writes one compact table:

    {"generated_at": ..., "fields": [...], "sites": {id: [row...]}}

Row fields (see FIELDS): last observed cfs, its date, the class of that value
against USGS daily statistics (min / p25 / p75 / max for the month-day), the
blend's 14-day peak, the peak's class against the stats for ITS day, and the
peak / current ratio.

Classes, deliberately coarse because the published stats carry only
min/p25/p50/p75/max (no p10/p90, so no "much below/above" split):

    0 unknown   1 below record low   2 below normal (<p25)
    3 normal (p25..p75)   4 above normal (>p75)   5 above record high

"Record" needs MIN_RECORD_YEARS of record for that month-day; below that the
value is demoted to below/above normal. With N years a new record for a given
date lands in each tail ~1/(N+1) of the time by chance alone -- 20% at 4 years,
which is what put ~4.5% of gauges in each record class on 2026-11-02 (sampled
record-class gauges: median ~22 years of record vs ~53 for "normal").

A class compares one day's MEAN flow with the distribution of daily means for
that month-day. It says nothing about flooding: flood stages come from NWS,
per gauge, in the station panel.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

FIELDS = ["q", "date", "cls", "peak_q", "peak_date", "peak_cls", "ratio"]
MIN_RECORD_YEARS = 10
CLASS_NAMES = ["unknown", "record low", "below normal", "normal",
               "above normal", "record high"]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def classify(q, row: dict | None) -> int:
    """Class of `q` against one month-day row of USGS daily statistics."""
    q = _num(q)
    if q is None or not row:
        return 0
    lo, p25, p75, hi = (_num(row.get(k)) for k in ("min_va", "p25_va", "p75_va", "max_va"))
    if p25 is None or p75 is None:
        return 0
    n = _num(row.get("count"))
    long_enough = n is None or n >= MIN_RECORD_YEARS
    if lo is not None and q < lo:
        return 1 if long_enough else 2
    if hi is not None and q > hi:
        return 5 if long_enough else 4
    if q < p25:
        return 2
    if q > p75:
        return 4
    return 3


def summarise(payload: dict) -> list | None:
    hist = [h for h in payload.get("history") or [] if _num(h.get("q_cfs")) is not None]
    if not hist:
        return None
    last = hist[-1]
    q, d = _num(last["q_cfs"]), str(last["date"])
    rows = {r.get("month_day"): r for r in (payload.get("daily_stats") or {}).get("rows") or []}
    cls = classify(q, rows.get(d[5:10]))

    peak_q = peak_d = None
    for b in payload.get("blend") or []:
        v = _num(b.get("q_cfs"))
        if v is not None and (peak_q is None or v > peak_q):
            peak_q, peak_d = v, str(b.get("date"))
    peak_cls = classify(peak_q, rows.get(peak_d[5:10])) if peak_d else 0
    ratio = round(peak_q / q, 3) if peak_q is not None and q and q > 0 else None
    return [round(q, 2), d, cls,
            None if peak_q is None else round(peak_q, 2), peak_d, peak_cls, ratio]


def build(forecast_dir: Path) -> dict:
    sites = {}
    for f in sorted(forecast_dir.glob("*.json")):
        try:
            row = summarise(json.loads(f.read_text()))
        except (OSError, ValueError, AttributeError, TypeError):
            continue
        if row is not None:
            sites[f.stem] = row
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": FIELDS,
        "classes": CLASS_NAMES,
        "sites": sites,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--forecasts", default="dist/forecasts")
    ap.add_argument("--out", default="dist/current.json")
    a = ap.parse_args()
    out = build(Path(a.forecasts))
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"  current.json: {len(out['sites'])} sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
