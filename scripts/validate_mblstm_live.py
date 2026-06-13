#!/usr/bin/env python3
"""Live-path validation of the MB-LSTM member: Daymet-train vs Open-Meteo-serve.

The model was trained on Daymet forcings; production feeds it Open-Meteo.
This script issues a real forecast from `--as-of` (default: 20 days ago) at a
sample of stations using *live USGS + Open-Meteo* inputs — the exact serving
data path — and scores it against what actually happened. If the forcing-
product gap were a problem it would show up here as MAE ratios far worse
than the Daymet-based backtest (benchmarks/mblstm_backtest_pilot_v1.json:
ratio ~0.64 h1 → ~0.35-0.45 h4-14).

Usage:
  RW2_ENABLE_MBLSTM=1 .venv/bin/python scripts/validate_mblstm_live.py --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RW2_ENABLE_MBLSTM", "1")

from app import gages2, mblstm, usgs, weather  # noqa: E402

CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus"
STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
HORIZON = 14


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--as-of", default=(date.today() - timedelta(days=20)).isoformat(),
                    help="forecast issue date (needs HORIZON days of truth after it)")
    ap.add_argument("--hist-days", type=int, default=900)
    ap.add_argument("--label", default="live_v1")
    args = ap.parse_args()

    issue = date.fromisoformat(args.as_of)
    hist_start = issue - timedelta(days=args.hist_days)
    target_end = issue + timedelta(days=HORIZON)

    registry = {s["id"]: s for s in json.loads(STATIONS_PATH.read_text())["stations"]}
    corpus_ids = [p.name.split(".")[0] for p in sorted(CORPUS_DIR.glob("*.csv.gz"))]
    step = max(1, len(corpus_ids) // args.n)
    sample = [sid for sid in corpus_ids[::step] if sid in registry][: args.n]

    per_h: dict[int, dict[str, list]] = {h: {"persist": [], "mblstm": []} for h in range(1, HORIZON + 1)}
    used = 0
    for i, sid in enumerate(sample, 1):
        st = registry[sid]
        try:
            q = usgs.fetch_daily_discharge(sid, hist_start, target_end)
            if q.empty:
                continue
            q["date"] = pd.to_datetime(q["date"]).dt.date
            q_hist = q[q["date"] <= issue].reset_index(drop=True)
            truth = q[q["date"] > issue].set_index("date")["q_cfs"]
            if len(q_hist) < 400 or len(truth) < 10:
                continue
            wx = weather.fetch_history(st["lat"], st["lon"], hist_start, target_end)
            if wx.empty:
                continue
            wx["date"] = pd.to_datetime(wx["date"]).dt.date
            wx_hist = wx[wx["date"] <= issue].reset_index(drop=True)
            wx_fcst = wx[wx["date"] > issue].reset_index(drop=True)
            if len(wx_fcst) < HORIZON - 4:
                continue
            attrs = gages2.enrich_station_attrs(dict(st))
            rows = mblstm.forecast(q_hist, wx_hist, wx_fcst, attrs, HORIZON)
            if not rows:
                print(f"[{i}] {sid} mblstm returned None", flush=True)
                continue
            persist = float(q_hist["q_cfs"].iloc[-1])
            used += 1
            for h in range(1, HORIZON + 1):
                d = issue + timedelta(days=h)
                if d not in truth.index or not np.isfinite(truth[d]):
                    continue
                per_h[h]["persist"].append(abs(truth[d] - persist))
                per_h[h]["mblstm"].append(abs(truth[d] - rows[h - 1]["q_cfs"]))
        except Exception as exc:
            print(f"[{i}] {sid} ERR {exc}", flush=True)
        time.sleep(0.3)

    print(f"\nstations used: {used}  issue={issue}  (truth through {target_end})")
    print(f"{'h':>3} {'persist':>9} {'mblstm':>9} {'ratio':>7} {'n':>4}")
    out_h = {}
    for h in range(1, HORIZON + 1):
        p, m = per_h[h]["persist"], per_h[h]["mblstm"]
        if not p:
            continue
        mp, mm = float(np.median(p)), float(np.median(m))
        out_h[h] = {"persistence": mp, "mblstm": mm, "ratio": mm / mp if mp else None, "n": len(p)}
        print(f"{h:>3} {mp:>9.1f} {mm:>9.1f} {mm / mp if mp else float('nan'):>7.3f} {len(p):>4}")

    out = ROOT / "benchmarks" / f"mblstm_live_{args.label}.json"
    out.write_text(json.dumps({"issue": issue.isoformat(), "stations": used,
                               "median_abs_err_by_h": out_h}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
