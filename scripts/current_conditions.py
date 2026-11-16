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
that month-day. It says nothing about flooding. Flooding is the separate
`flood` / `near_flood` pair: the highest NWS category (0 none, 1 action,
2 minor, 3 moderate, 4 major) that today's flow / the NEAR_DAYS peak reaches,
against NWS thresholds converted to cfs offline (`scripts/build_nws_flood.py`
-> data/nws_flood.json). Null where NWS has no thresholds for the gauge. Our
values are DAILY MEANS; the instantaneous peak runs higher, so a 0 is not
"no flooding" -- NWS's own forecast and warnings are the authority.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

FIELDS = ["q", "date", "cls", "peak_q", "peak_date", "peak_cls", "ratio",
          "near_q", "near_date", "near_cls", "near_ratio", "flood", "near_flood"]
FLOOD_NAMES = ["none", "action", "minor", "moderate", "major"]
NWS_FLOOD_FILE = Path(__file__).resolve().parents[1] / "data" / "nws_flood.json"
#: The near_* fields are the same peak restricted to NEAR_DAYS after the last
#: observation. Alerts and the map's "forecast to rise" ring read these, not the
#: 14-day peak: on 2026-11-12, 316 of 327 "rising" gauges peaked on day 6 or
#: later, where the blend ran wet (median rel. bias +0.31 at lead 7 vs +0.02 at
#: lead 1, PAPER_LEDGER C2), and the median gauge's 14-day peak was 1.31x today.
NEAR_DAYS = 5
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


def load_flood(path: Path = NWS_FLOOD_FILE) -> dict:
    """site -> [action, minor, moderate, major] cfs; {} if the file is absent/bad."""
    try:
        return {k: v[1:5] for k, v in json.loads(path.read_text()).get("sites", {}).items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def flood_cat(q, thr) -> int | None:
    """Highest category whose threshold q reaches; None if no threshold known."""
    if not thr or not any(_num(t) is not None for t in thr):
        return None
    if q is None:
        return None
    cat = 0
    for i, t in enumerate(thr, 1):
        t = _num(t)
        if t is not None and q >= t:
            cat = i
    return cat


#: A class (map colour, alert level) needs an observation this recent. On
#: 2026-11-16 the live current.json held 1,328 of 9,698 gauges whose last daily
#: value was > 30 days old (972 > 1 year, back to 1986), each still coloured as
#: if it were today's flow and its "forecast" peak classed: 97 sat at an alert
#: level (84 "record"). Older than this, every class is 0 (unknown).
CLASS_MAX_OBS_AGE_D = 14
#: A flood category needs an observation this recent (days before the build).
#: current.json carries gauges whose last daily value is years old.
FLOOD_MAX_OBS_AGE_D = 7


def plausible_thresholds(thr, stats_row: dict | None) -> list | None:
    """Drop thresholds at or below the gauge's p75 for the day.

    Most NWS thresholds are STAGES, converted to cfs through the NWPS rating.
    Checked 2026-11-16 against NWS's own observed category on the 20 gauges the
    raw conversion flagged: tidal and low-stage rating segments produced
    thresholds such as 13 cfs (action) or 0 cfs (major), flagging rivers NWS
    called "no flooding". Flood flow at or below an ordinary high flow for the
    date is not a flood threshold. Without stats nothing is dropped except <= 0.
    """
    if not thr:
        return thr
    p75 = _num((stats_row or {}).get("p75_va"))
    out = []
    for t in thr:
        t = _num(t)
        out.append(None if t is None or t <= 0 or (p75 is not None and t <= p75) else t)
    return out


def summarise(payload: dict, thr=None, today: date | None = None) -> list | None:
    hist = [h for h in payload.get("history") or [] if _num(h.get("q_cfs")) is not None]
    if not hist:
        return None
    last = hist[-1]
    q, d = _num(last["q_cfs"]), str(last["date"])
    rows = {r.get("month_day"): r for r in (payload.get("daily_stats") or {}).get("rows") or []}
    cls = classify(q, rows.get(d[5:10]))

    blend = payload.get("blend") or []
    try:
        near_end = (date.fromisoformat(d[:10]) + timedelta(days=NEAR_DAYS)).isoformat()
    except ValueError:
        near_end = ""

    def peak(within: str | None) -> list:
        pq = pd_ = None
        for b in blend:
            v, bd = _num(b.get("q_cfs")), str(b.get("date"))[:10]
            if v is None or (within is not None and bd > within):
                continue
            if pq is None or v > pq:
                pq, pd_ = v, bd
        pc = classify(pq, rows.get(pd_[5:10])) if pd_ else 0
        ratio = round(pq / q, 3) if pq is not None and q and q > 0 else None
        return [None if pq is None else round(pq, 2), pd_, pc, ratio]

    near = peak(near_end)
    full = peak(None)
    thr = plausible_thresholds(thr, rows.get(d[5:10]))
    try:
        age = ((today or date.today()) - date.fromisoformat(d[:10])).days
    except ValueError:
        age = None
    if age is None or age > FLOOD_MAX_OBS_AGE_D:
        thr = None
    if age is None or age > CLASS_MAX_OBS_AGE_D:
        cls, full[2], near[2] = 0, 0, 0
    return ([round(q, 2), d, cls] + full + near
            + [flood_cat(q, thr), flood_cat(near[0], thr)])


def build(forecast_dir: Path, flood_file: Path = NWS_FLOOD_FILE, today: date | None = None) -> dict:
    flood = load_flood(flood_file)
    sites = {}
    for f in sorted(forecast_dir.glob("*.json")):
        try:
            row = summarise(json.loads(f.read_text()), flood.get(f.stem), today)
        except (OSError, ValueError, AttributeError, TypeError):
            continue
        if row is not None:
            sites[f.stem] = row
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": FIELDS,
        "classes": CLASS_NAMES,
        "flood_classes": FLOOD_NAMES,
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
