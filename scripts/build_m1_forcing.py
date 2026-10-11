#!/usr/bin/env python3
"""Maintain the MODERN-1 forcing table the served member reads.

WHY THIS EXISTS
---------------
`app/mblstm_modern.py` needs 6 daily channels over a 365-day encoder window
plus a 14-day decoder window, for the 490 cohort basins. It cannot fetch them
at serve time: the Pages build sets `RW2_NO_FETCH=1`, and `modern_forcing`'s
cache key embeds the literal `(start, end)` pair, so `end = today-1` mints a
new cache filename every day and the cache can never warm. With 64 build
shards on a per-shard `actions/cache`, there is nowhere for a warm cache to
live either.

So this runs ONCE per build, before the matrix fans out, exactly like
`scripts/build_snodas_extracts.py`: read each basin's existing table, find how
far it already reaches, and fetch only the days beyond it. The result is
published to an orphan branch and handed to all 64 shards as one artifact.

It is incremental by design. A cold start is ~490 basins x 365 days x 7
variables; Open-Meteo weights a request by `days x variables` and the binding
cap is DAILY, so the first run does NOT finish in one window -- it bails with
exit 2 and resumes next run. Once warm, each run fetches ~1 new day per basin,
which is three orders of magnitude cheaper.

ONE INTERFACE, TWO BACKENDS. The `src` column records where each row came
from. Today every row is `om_point` (Open-Meteo at the gauge coordinate), which
GATE 1 measured costs -0.059534 paired NSE against the NLDAS-2 **areal**
forcing the member trained on. When the areal provider lands it writes rows
with `src=nldas_areal` into this same schema at this same path, and the serving
code does not change at all -- only the label does.

QUOTA DISCIPLINE, all of it learned the hard way (see
scripts_modern/build_openmeteo_point_corpus.py, which measured this):
  * FAILED requests still count against the budget, so a retry loop with no
    global bail starves its own next window. Hence --max-consecutive-failures.
  * Exit 2 means "budget gone, come back next window" and is NOT a failure.
  * The iteration order is SHUFFLED. Cohort ids are sorted USGS gauge numbers,
    which run roughly geographically, so a budget-truncated run in sorted order
    yields a geographically biased subset -- and anything later measured on it
    inherits that bias. This is the mistake that made a partial Gate 1
    unscoreable.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import modern_forcing as mf  # noqa: E402
from app.mblstm_modern import _sat_vp_pa  # noqa: E402  (units verified through it)

COLS = mf.TABLE_COLS
OUT_HEADER = ["date"] + COLS + ["src"]
SRC_POINT = "om_point"


def log(msg: str) -> None:
    print(f"[m1-forcing] {msg}", flush=True)


def load_cohort(path: Path) -> list[str]:
    return list(json.loads(path.read_text())["cohort"])


def load_latlon(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    for s in json.loads(path.read_text()).get("stations", []):
        lat, lon = s.get("lat"), s.get("lon")
        if s.get("id") and lat is not None and lon is not None:
            out[str(s["id"])] = (float(lat), float(lon))
    return out


def existing_tail(sid: str) -> date | None:
    """The newest date already stored for this basin, or None."""
    df = mf.load_table(sid)
    if df is None or df.empty:
        return None
    return max(df["date"])


def derive_vp(df: pd.DataFrame) -> pd.DataFrame:
    """Add `vapor_pressure` from dew point, exactly as the member would.

    Stored derived so the serving path does no conversion and
    `_ensure_vapor_pressure()` no-ops.
    """
    out = df.copy()
    if mf.DEW_VAR in out.columns:
        out["vapor_pressure"] = _sat_vp_pa(out[mf.DEW_VAR].to_numpy(dtype=np.float64))
    return out


def fetch_basin(lat: float, lon: float, start: date, end: date, *, horizon_days: int,
                past_days: int, retries: int, sleep_s: float) -> pd.DataFrame | None:
    """Archive rows [start, today-1] plus forecast rows, as one frame.

    Returns None on failure (so the caller can count it toward the bail), and
    an EMPTY frame is treated as a failure too -- Open-Meteo answers a 429 with
    a body that `_cached_or_fetch` swallows into an empty frame, and a silent
    empty would otherwise look like success.
    """
    today = date.today()
    for attempt in range(retries):
        try:
            hist = mf.fetch_history(lat, lon, start, today - timedelta(days=1))
            fcst = mf.fetch_forecast(lat, lon, days=horizon_days, past_days=past_days)
            if hist is None or fcst is None or hist.empty or fcst.empty:
                raise RuntimeError("empty frame (throttled, or no data for this point)")
            both = pd.concat([derive_vp(hist), derive_vp(fcst)], ignore_index=True)
            # Forecast rows win on an overlapping date: they are the newer
            # analysis, and `past_days` deliberately overlaps the archive.
            both = both.drop_duplicates(subset=["date"], keep="last")
            return both.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            if attempt == retries - 1:
                log(f"  fetch failed after {retries} attempts: {exc}")
                return None
            time.sleep(sleep_s * (2 ** attempt))
    return None


def merge_and_write(sid: str, fresh: pd.DataFrame, *, keep_days: int) -> tuple[int, int]:
    """Merge fresh rows into the basin's table and publish atomically.

    FLIP RULE: a row from a better source never loses to a worse one. Today
    there is only one source, but when areal rows arrive they must overwrite
    point-filled rows for the same date and never the other way round.
    """
    rank = {SRC_POINT: 0, "nldas_areal": 1}
    fresh = fresh.copy()
    fresh["src"] = SRC_POINT

    old = mf.load_table(sid)
    if old is not None and not old.empty:
        if "src" not in old.columns:
            old = old.assign(src=SRC_POINT)
        allrows = pd.concat([old, fresh], ignore_index=True)
    else:
        allrows = fresh

    allrows["_rank"] = allrows["src"].map(lambda s: rank.get(s, 0))
    allrows = allrows.sort_values(["date", "_rank"]).drop_duplicates(
        subset=["date"], keep="last").drop(columns=["_rank"])

    cutoff = date.today() - timedelta(days=keep_days)
    allrows = allrows[allrows["date"] >= cutoff]
    allrows = allrows.sort_values("date").reset_index(drop=True)

    for c in COLS:
        if c not in allrows.columns:
            allrows[c] = np.nan

    dst = mf.table_path(sid)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.gz")
    with gzip.open(tmp, "wt", newline="") as fh:
        allrows[OUT_HEADER].to_csv(fh, index=False)
    tmp.replace(dst)

    # Re-read what was published: a truncated write that still parses is the
    # failure mode this catches.
    back = mf.load_table(sid)
    if back is None or len(back) != len(allrows):
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"{sid}: wrote {len(allrows)} rows, read back "
                           f"{0 if back is None else len(back)}")
    return len(allrows), int(allrows[COLS].notna().all(axis=1).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/m1_cohort.json")
    ap.add_argument("--stations", default="data/stations_v15.json")
    ap.add_argument("--keep-days", type=int, default=425,
                    help="rows retained per basin: 365 encoder + slack for a stale gauge")
    ap.add_argument("--horizon-days", type=int, default=16)
    ap.add_argument("--past-days", type=int, default=10,
                    help="recent analysis days prepended to the forecast call, so a "
                         "stale gauge's earliest decoder rows are not NaN")
    ap.add_argument("--fresh-within-days", type=int, default=1,
                    help="skip a basin whose table already reaches today-N")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=0.6, help="politeness delay between basins")
    ap.add_argument("--max-consecutive-failures", type=int, default=6,
                    help="bail the window; failed requests still cost budget")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle-seed", default=os.environ.get("GITHUB_RUN_ID", "0"),
                    help="sorted order is geographic, so a truncated run must not be sorted")
    ap.add_argument("--max-seconds", type=int, default=0,
                    help="stop cleanly after N seconds (0 = no limit). MUST be set "
                         "well inside the CI job timeout: the publish steps only run "
                         "after this script returns, so a run killed by the timeout "
                         "persists NOTHING and the cold start repeats forever")
    a = ap.parse_args()

    cohort = load_cohort(Path(a.cohort))
    latlon = load_latlon(Path(a.stations))
    missing_meta = [s for s in cohort if s not in latlon]
    if missing_meta:
        log(f"WARNING: {len(missing_meta)} cohort basins have no lat/lon: {missing_meta[:5]}")
    todo = [s for s in cohort if s in latlon]

    random.Random(str(a.shuffle_seed)).shuffle(todo)
    if a.limit:
        todo = todo[:a.limit]

    today = date.today()
    start = today - timedelta(days=a.keep_days)
    fresh_bar = today - timedelta(days=a.fresh_within_days)

    log(f"{len(todo)} basins; window {start}..{today + timedelta(days=a.horizon_days)}; "
        f"shuffle seed {a.shuffle_seed}")

    n_ok = n_skip = n_fail = 0
    consec = 0
    t_start = time.time()
    for i, sid in enumerate(todo, 1):
        # TIME BUDGET. A cold start is ~490 basins x 425 days x 7 variables and
        # does not fit in one CI job; measured at roughly 4-7 s per basin, 490
        # of them overruns a 45-minute job. Stopping cleanly here lets the
        # upload and orphan-branch push run, so the next build RESUMES from
        # what this one managed. Being killed by the job timeout instead would
        # publish nothing and repeat the cold start every single build.
        if a.max_seconds and (time.time() - t_start) > a.max_seconds:
            log(f"TIME BUDGET reached after {i - 1} basins "
                f"({n_ok} written, {n_skip} already fresh, {n_fail} failed). "
                f"Publishing what is done; the next run resumes.")
            break
        tail = existing_tail(sid)
        if tail is not None and tail >= fresh_bar:
            n_skip += 1
            continue
        lat, lon = latlon[sid]
        fresh = fetch_basin(lat, lon, start, today, horizon_days=a.horizon_days,
                            past_days=a.past_days, retries=a.retries, sleep_s=a.sleep)
        if fresh is None:
            n_fail += 1
            consec += 1
            if consec >= a.max_consecutive_failures:
                log(f"BAIL after {consec} consecutive failures "
                    f"({n_ok} ok, {n_skip} already fresh, {n_fail} failed). "
                    f"Budget is probably gone; this is not fatal -- resume next window.")
                return 2
            continue
        consec = 0
        try:
            rows, complete = merge_and_write(sid, fresh, keep_days=a.keep_days)
        except Exception as exc:
            log(f"  {sid}: publish refused: {exc}")
            n_fail += 1
            continue
        n_ok += 1
        if n_ok <= 3 or n_ok % 50 == 0:
            log(f"  [{i}/{len(todo)}] {sid}: {rows} rows, {complete} with all 6 channels")
        time.sleep(a.sleep)

    log(f"done: {n_ok} written, {n_skip} already fresh, {n_fail} failed")
    published = len(list(mf.TABLE_DIR.glob("*.csv.gz"))) if mf.TABLE_DIR.exists() else 0
    log(f"table now covers {published}/{len(cohort)} cohort basins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
