#!/usr/bin/env python3
"""Build the multi-basin LSTM training corpus.

For each registered station, fetch the full daily discharge record (USGS) and
daily weather history (Open-Meteo archive) and write one merged csv.gz per
station to data/mblstm/corpus/<id>.csv.gz with columns:

    date, q_cfs, <weather.DAILY_VARS...>

Resumable: stations whose corpus file already exists are skipped (unless
--refresh). Open-Meteo rate limits are handled by sleeping between stations
and by simply skipping stations whose fetch comes back short — rerun the
script later and it picks up where it left off.

Usage:
  .venv/bin/python scripts/build_mblstm_data.py --limit 300
  .venv/bin/python scripts/build_mblstm_data.py            # all stations
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import usgs, weather  # noqa: E402

STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus"

# Minimum overlapping (q, weather) days for a station to be worth keeping.
MIN_OVERLAP_DAYS = 8 * 365


def build_one(st: dict, start: date, end: date) -> tuple[str, pd.DataFrame | None]:
    sid = st["id"]
    q = usgs.fetch_daily_discharge(sid, start, end)
    if q.empty or len(q) < MIN_OVERLAP_DAYS:
        return f"q too short ({len(q)})", None
    wx = weather.fetch_history(st["lat"], st["lon"], start, end)
    if wx.empty or len(wx) < MIN_OVERLAP_DAYS:
        return f"wx too short ({len(wx)})", None
    df = pd.merge(q, wx, on="date", how="inner")
    if len(df) < MIN_OVERLAP_DAYS:
        return f"overlap too short ({len(df)})", None
    df = df.sort_values("date").reset_index(drop=True)
    return "ok", df


def main() -> int:
    global CORPUS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--limit", type=int, default=0, help="first N stations (0 = all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.6, help="seconds between stations (Open-Meteo throttle)")
    ap.add_argument("--refresh", action="store_true", help="rebuild even if corpus file exists")
    ap.add_argument("--out-dir", default=str(CORPUS_DIR),
                    help="corpus output directory (default data/mblstm/corpus; "
                         "use data/mblstm/corpus_openmeteo for the 13-var corpus)")
    args = ap.parse_args()

    CORPUS_DIR = Path(args.out_dir)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(args.start)
    end = date.today() - timedelta(days=1)

    stations = json.loads(STATIONS_PATH.read_text())["stations"]
    if args.offset:
        stations = stations[args.offset:]
    if args.limit:
        stations = stations[: args.limit]

    done = skipped = failed = 0
    consecutive_wx_failures = 0
    t0 = time.time()
    for i, st in enumerate(stations, 1):
        out = CORPUS_DIR / f"{st['id']}.csv.gz"
        if out.exists() and not args.refresh:
            skipped += 1
            continue
        # Open-Meteo burst limits show up as empty/short weather responses.
        # Retry the same station with growing backoff before giving up on it.
        status, df = "", None
        for attempt in range(4):
            try:
                status, df = build_one(st, start, end)
            except Exception as exc:
                status, df = f"error: {exc}", None
            if df is not None or not status.startswith("wx too short"):
                break
            wait = 75 * (attempt + 1)
            print(f"[{i}/{len(stations)}] {st['id']} rate-limited? backing off {wait}s", flush=True)
            time.sleep(wait)
        if df is None:
            failed += 1
            if status.startswith("wx too short"):
                consecutive_wx_failures += 1
                if consecutive_wx_failures >= 6:
                    print(f"[{i}/{len(stations)}] sustained Open-Meteo block — "
                          f"stopping (rerun later, progress is saved)")
                    break
            print(f"[{i}/{len(stations)}] {st['id']} SKIP {status}")
        else:
            consecutive_wx_failures = 0
            df.to_csv(out, index=False, compression="gzip")
            done += 1
            print(f"[{i}/{len(stations)}] {st['id']} ok rows={len(df)} "
                  f"({df['date'].iloc[0]}..{df['date'].iloc[-1]})")
        time.sleep(args.sleep)

    print(f"\ndone={done} skipped(existing)={skipped} failed={failed} "
          f"in {time.time() - t0:.0f}s; corpus now has "
          f"{len([p for p in CORPUS_DIR.glob('*.csv.gz') if not p.name.startswith('._')])} stations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
