#!/usr/bin/env python3
"""Score the forecasts this site actually issued against what the rivers did.

This is the other half of the verification archive. `snapshot_m1_archive.py`
records what was served; this joins it to USGS observations as they arrive and
publishes `benchmarks/m1_live_verification.json`, which then replaces the
borrowed research figure in the served member's public label.

⚠️ WHAT A LIVE NUMBER OUT OF THIS IS, AND IS NOT
------------------------------------------------
An NSE computed over a 30- or 90-day window is dominated by that window's flow
variance. It is NOT comparable to:
  * 0.905829 -- perfect forcing, day 1, 490 basins, a 7-year test window
  * the ~0.858 operational figure -- that is 0.905829 less a measured decay
  * 0.808659 -- GATE 1's as-served arm on val1/150 basins
Different frame, different window, and a short one. A live NSE of 0.72 in a
low-variance autumn is not a regression and 0.93 in a wet spring is not a win.

The quantity that TRANSFERS is the paired difference between two series scored
on identical rows, which is why the archive stores three series on identical
keys and why this reports paired deltas with bootstrap intervals as the
headline. That is the same rule GATE 1 and GATE 2 were registered under, and
the reason is on the record: a difference of medians has overstated an effect
in this project more than ten times, most recently by 1.9x.

Scores ALL days present, never a stride subsample -- a stride-14 frame
flattered this project's own numbers by +0.036.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARCHIVE = ROOT / "archive_m1"
OUT = ROOT / "benchmarks" / "m1_live_verification.json"
SCHEMA = "m1-live-1"

#: Below this many (basin, target_date) pairs per series we report coverage and
#: paired deltas but refuse to publish a per-basin NSE median: a median over a
#: handful of basin-days is noise wearing a decimal point.
MIN_PAIRS_FOR_NSE = 2000
#: Per-basin minimum before that basin gets its own NSE.
MIN_DAYS_PER_BASIN = 25


def read_archive(start: str, end: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(ARCHIVE / "*" / "*" / "*.csv.gz"))):
        issued = Path(p).name.split(".")[0]
        if not (start <= issued <= end):
            continue
        with gzip.open(p, "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def fetch_obs(station_ids, lo: str, hi: str) -> dict:
    """Observed daily discharge, keyed (station_id, target_date)."""
    from app import usgs
    obs = {}
    lo_d, hi_d = date.fromisoformat(lo), date.fromisoformat(hi)
    for i, sid in enumerate(sorted(station_ids), 1):
        try:
            df = usgs.fetch_daily_discharge(sid, lo_d, hi_d, max_age_hours=12)
        except Exception as exc:
            print(f"  {sid}: USGS fetch failed ({exc})")
            continue
        if df is None or not len(df):
            continue
        for d, q in zip(df["date"], df["q_cfs"]):
            try:
                v = float(q)
            except (TypeError, ValueError):
                continue
            if np.isfinite(v) and v >= 0:
                obs[(sid, str(d)[:10])] = v
        if i % 50 == 0:
            print(f"  observations: {i}/{len(station_ids)} basins")
    return obs


def nse(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = float(((truth - truth.mean()) ** 2).sum())
    if denom <= 0:
        return float("nan")
    return 1.0 - float(((pred - truth) ** 2).sum()) / denom


def paired_ci(deltas: np.ndarray, draws: int = 5000, seed: int = 0) -> list:
    if len(deltas) < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(deltas, len(deltas), replace=True)))
            for _ in range(draws)]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--horizon", type=int, default=1,
                    help="lead day to score; 1 is the only lead with a published figure")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    rows = read_archive(a.start, a.end)
    if not rows:
        print(f"no archive rows in {ARCHIVE} for {a.start}..{a.end}; nothing to score")
        return 1
    rows = [r for r in rows if int(r["horizon_day"]) == a.horizon]
    print(f"{len(rows)} archived rows at lead {a.horizon}")

    # series -> (station, target_date) -> predicted
    pred = defaultdict(dict)
    for r in rows:
        try:
            v = float(r["q_cfs"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            pred[r["series"]][(r["station_id"], r["target_date"])] = v
    for s, d in pred.items():
        print(f"  {s}: {len(d)} (basin, target_date) pairs")

    if "mblstm_modern" not in pred:
        print("the member has no archived rows yet; nothing to score")
        return 1

    # Score only rows EVERY series has, so the comparison is strictly paired.
    common = set.intersection(*(set(d) for d in pred.values()))
    print(f"common to all {len(pred)} series: {len(common)} pairs")

    tds = sorted({t for _, t in common})
    sids = {s for s, _ in common}
    obs = fetch_obs(sids, tds[0], tds[-1])
    scored = sorted(k for k in common if k in obs)
    print(f"with an observation: {len(scored)} pairs over {len({s for s,_ in scored})} basins")
    if not scored:
        print("no overlap with observations yet -- the newest target dates are "
              "still in the future, which is expected until a build's horizon "
              "has elapsed")
        return 1

    truth = np.array([obs[k] for k in scored], dtype=np.float64)
    series_names = sorted(pred)
    result = {
        "schema_version": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "lead_day": a.horizon,
        "window": {"first_target": scored[0][1], "last_target": scored[-1][1]},
        "n_pairs": len(scored),
        "n_basins": len({s for s, _ in scored}),
        "series": {},
        "paired_vs_member": {},
        "per_basin_nse": {},
    }

    # Pooled MAE per series, on identical rows.
    for s in series_names:
        p = np.array([pred[s][k] for k in scored], dtype=np.float64)
        result["series"][s] = {
            "mae": float(np.mean(np.abs(p - truth))),
            "bias": float(np.mean(p - truth)),
        }

    # THE HEADLINE: paired per-basin MAE difference vs the member, with a CI.
    by_basin = defaultdict(list)
    for i, (sid, _) in enumerate(scored):
        by_basin[sid].append(i)
    mem = np.array([pred["mblstm_modern"][k] for k in scored], dtype=np.float64)
    for s in series_names:
        if s == "mblstm_modern":
            continue
        other = np.array([pred[s][k] for k in scored], dtype=np.float64)
        d = []
        for sid, idx in by_basin.items():
            ii = np.array(idx)
            d.append(float(np.mean(np.abs(mem[ii] - truth[ii]))
                           - np.mean(np.abs(other[ii] - truth[ii]))))
        d = np.array(d, dtype=np.float64)
        result["paired_vs_member"][s] = {
            "paired_median_mae_delta": float(np.median(d)),
            "ci95": paired_ci(d),
            "breadth_member_better": float(np.mean(d < 0)),
            "n_basins": int(len(d)),
            "_note": ("member MAE minus this series' MAE, paired per basin on "
                      "identical rows. NEGATIVE means the member is better."),
        }

    # Per-basin NSE, only where there is enough of a series to mean anything.
    if len(scored) >= MIN_PAIRS_FOR_NSE:
        for s in series_names:
            vals = []
            for sid, idx in by_basin.items():
                if len(idx) < MIN_DAYS_PER_BASIN:
                    continue
                ii = np.array(idx)
                v = nse(np.array([pred[s][scored[j]] for j in ii]), truth[ii])
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                result["per_basin_nse"][s] = {
                    "median": float(np.median(vals)),
                    "n_basins": len(vals),
                    "_note": ("NOT comparable to 0.905829, ~0.858 or 0.808659: "
                              "a different frame over a short window, dominated "
                              "by this window's flow variance."),
                }
    else:
        result["per_basin_nse"]["_withheld"] = (
            f"{len(scored)} pairs is under the {MIN_PAIRS_FOR_NSE} needed; a "
            "median NSE over this little would be noise with a decimal point.")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    for s, v in result["paired_vs_member"].items():
        print(f"  member vs {s}: paired median MAE delta "
              f"{v['paired_median_mae_delta']:+.4f} cfs, CI {v['ci95']}, "
              f"member better on {v['breadth_member_better']:.1%} of basins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
