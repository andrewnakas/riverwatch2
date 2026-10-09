#!/usr/bin/env python
"""GATE 1 — build an Open-Meteo POINT corpus for the MODERN-1 cohort.

The question: what does serving `nldasm9` on Open-Meteo point forcing cost,
versus the NLDAS-2 areal forcing it was trained on? That is the gap between the
benchmark and `app/mblstm_modern.py`, and it is currently unmeasured.

CONTROLLED BY CONSTRUCTION. Each output file is the basin's existing NLDAS areal
corpus file with its SIX FORCING COLUMNS REPLACED by Open-Meteo values for the
same dates. `date` and `q_cfs` are copied verbatim, so the day set, the discharge
and the row count are identical and the ONLY thing that varies is the forcing.
That is the same design as screen B0, and the reason that screen's first run was
void: two corpora that differ in more than one way measure neither difference.

Units line up without conversion (checked against a corpus row):
  temperature_2m_{mean,max,min}  degC         <- Open-Meteo degC
  precipitation_sum              mm           <- Open-Meteo mm
  shortwave_radiation_sum        MJ/m^2/day   <- Open-Meteo MJ/m^2
  vapor_pressure                 Pa           <- derived from dew_point_2m_mean
                                                  by Tetens, as app/mblstm_modern does

A basin is SKIPPED, not partially written, if Open-Meteo returns fewer than
`--min-coverage` of its days: a corpus with holes would be scored as a forcing
difference when it is really a missing-data difference.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import modern_forcing as mf   # noqa: E402
from app.mblstm_modern import _sat_vp_pa  # noqa: E402

FORCING_COLS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure",
]


def read_source(path: Path) -> tuple[list[str], list[list[str]]]:
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split(",")
        rows = [ln.rstrip("\n").split(",") for ln in fh if ln.strip()]
    return hdr, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/modern_corpus/camels_corpus_nldas_modern_areal",
                    help="supplies date + q_cfs + the day set; only forcing is replaced")
    ap.add_argument("--out", default="data/modern_corpus/v3/camels_corpus_openmeteo_m3_cohort")
    ap.add_argument("--cohort", default="data/m1_cohort.json")
    ap.add_argument("--latlon", default="data/modern_corpus/basin_latlon.json")
    ap.add_argument("--min-coverage", type=float, default=0.98)
    # Open-Meteo weights a request by days x variables, so a 46-year 6-variable
    # pull burns the hourly budget in a few dozen basins (measured: 5 ok, 152
    # throttled). GATE 1 only scores the val window, so fetch only that plus the
    # 365-day encoder context and the request gets ~9x smaller.
    ap.add_argument("--start", default="", help="clip the window, e.g. 2013-09-01")
    ap.add_argument("--end", default="", help="clip the window, e.g. 2018-09-30")
    ap.add_argument("--retries", type=int, default=4)
    # Once the hourly quota is gone, EVERY remaining basin fails, and retrying
    # each one through its own backoff burns ~75s x the rest of the list (~110
    # min measured) for nothing. Bail out instead so the caller can sleep until
    # the next window. Exit code 2 means exactly that, and is not a fatal error.
    ap.add_argument("--max-consecutive-failures", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=1.2, help="politeness delay between API calls")
    ap.add_argument("--limit", type=int, default=0)
    # A fixed random subsample keeps the Open-Meteo request budget survivable
    # (490 basins x ~1,800 days x 7 vars exhausts the hourly quota). The seed is
    # fixed so the sample is declared, not chosen after seeing any score.
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src_dir = ROOT / a.source
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = json.loads((ROOT / a.cohort).read_text())["cohort"]
    latlon = json.loads((ROOT / a.latlon).read_text())
    if a.subsample and a.subsample < len(ids):
        import numpy as _np
        rng = _np.random.default_rng(a.seed)
        pick = sorted(rng.choice(len(ids), size=a.subsample, replace=False).tolist())
        ids = [ids[i] for i in pick]
        print(f"subsample: {len(ids)} of 490 cohort basins (seed {a.seed})", flush=True)
    if a.limit:
        ids = ids[: a.limit]

    done = skipped = failed = 0
    consec_fail = 0
    quota_exhausted = False
    for i, gid in enumerate(ids, 1):
        dst = out_dir / f"{gid}.csv.gz"
        if dst.exists():
            done += 1
            continue
        src = src_dir / f"{gid}.csv.gz"
        if not src.exists():
            print(f"[{i}/{len(ids)}] {gid} SKIP no source", flush=True)
            skipped += 1
            continue
        # basin_latlon.json stores [lat, lon] lists; tolerate a dict too.
        ll = latlon.get(gid)
        if isinstance(ll, (list, tuple)) and len(ll) >= 2:
            lat, lon = ll[0], ll[1]
        elif isinstance(ll, dict):
            lat, lon = ll.get("lat"), ll.get("lon")
        else:
            lat = lon = None
        if lat is None or lon is None:
            print(f"[{i}/{len(ids)}] {gid} SKIP no lat/lon", flush=True)
            skipped += 1
            continue

        hdr, rows = read_source(src)
        ci = {c: k for k, c in enumerate(hdr)}
        dates = [r[ci["date"]] for r in rows]
        if a.start:
            rows = [r for r in rows if r[ci["date"]] >= a.start]
        if a.end:
            rows = [r for r in rows if r[ci["date"]] <= a.end]
        dates = [r[ci["date"]] for r in rows]
        if not dates:
            print(f"[{i}/{len(ids)}] {gid} SKIP no rows in window", flush=True)
            skipped += 1
            continue
        d0 = date.fromisoformat(min(dates))
        d1 = date.fromisoformat(max(dates))

        om = None
        for attempt in range(1, a.retries + 1):
            try:
                om = mf.fetch_history(float(lat), float(lon), d0, d1)
            except Exception as exc:
                print(f"[{i}/{len(ids)}] {gid} attempt {attempt} raised {exc}", flush=True)
                om = None
            if om is not None and len(om):
                break
            # Empty almost always means throttled, not "no data" -- back off
            # rather than burning through the remaining basins at full speed.
            back = a.sleep * (2 ** attempt)
            print(f"[{i}/{len(ids)}] {gid} empty (attempt {attempt}/{a.retries}), "
                  f"backing off {back:.1f}s", flush=True)
            time.sleep(back)
        if om is None or not len(om):
            print(f"[{i}/{len(ids)}] {gid} FAIL empty after {a.retries} attempts", flush=True)
            failed += 1
            consec_fail += 1
            if consec_fail >= a.max_consecutive_failures:
                print(f"\nSTOP: {consec_fail} consecutive failures -- the hourly quota is "
                      f"exhausted. Bailing out so the caller can wait for the next window "
                      f"instead of failing through the remaining "
                      f"{len(ids) - i} basins.", flush=True)
                quota_exhausted = True
                break
            continue
        consec_fail = 0

        om = om.copy()
        om["vapor_pressure"] = _sat_vp_pa(om[mf.DEW_VAR].to_numpy(dtype="float64"))
        om["date"] = om["date"].astype(str)
        by_date = {r["date"]: r for _, r in om.iterrows()}

        have = sum(1 for d in dates if d in by_date)
        cov = have / max(len(dates), 1)
        if cov < a.min_coverage:
            print(f"[{i}/{len(ids)}] {gid} SKIP coverage {cov:.3f}", flush=True)
            skipped += 1
            time.sleep(a.sleep)
            continue

        out_hdr = ["date", "q_cfs"] + FORCING_COLS
        lines = [",".join(out_hdr)]
        written = 0
        for r in rows:
            d = r[ci["date"]]
            om_row = by_date.get(d)
            if om_row is None:
                continue
            vals = [d, r[ci["q_cfs"]]]
            ok = True
            for c in FORCING_COLS:
                v = om_row.get(c)
                if v is None or v != v:          # NaN
                    ok = False
                    break
                vals.append(f"{float(v):.6g}")
            if not ok:
                continue
            lines.append(",".join(vals))
            written += 1
        if written / max(len(dates), 1) < a.min_coverage:
            print(f"[{i}/{len(ids)}] {gid} SKIP usable {written}/{len(dates)}", flush=True)
            skipped += 1
            time.sleep(a.sleep)
            continue

        tmp = dst.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            fh.write("\n".join(lines) + "\n")
        tmp.replace(dst)                          # atomic: never a half file
        done += 1
        print(f"[{i}/{len(ids)}] {gid} ok rows={written} cov={cov:.3f}", flush=True)
        time.sleep(a.sleep)

    have = len(list(out_dir.glob("*.csv.gz")))
    print(f"\ndone={done} skipped={skipped} failed={failed} have={have}/{len(ids)} "
          f"-> {out_dir}", flush=True)
    if quota_exhausted:
        return 2            # not fatal: "come back next hour"
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
