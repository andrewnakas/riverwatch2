#!/usr/bin/env python3
"""Exact-Kratzert / Li-Shen DAY-1 (rolling lead-1) NSE for the CAMELS-531 record.

Why this exists (vs scripts/eval_continuous.py)
------------------------------------------------
`eval_continuous.py` chains NON-OVERLAPPING 14-day forecast windows and keeps ALL
14 leads (t0+1 .. t0+14) in the daily series — so leads 8-14 are long-range
forecasts mixed into a "continuous" number. The record papers (Kratzert 2019/2021,
Li/Shen 2025) are **sequence-to-one**: ONE predicted discharge per calendar day at
a FIXED short lead. This script produces that number the record actually reports —
roll the 365-day encoder forward by 1 day across the whole test decade and keep
ONLY the lead-1 (next-day) prediction for each calendar day → plain NSE per basin
→ median across 531. That is the apples-to-apples comparison to the 0.83 record.

The NSE definition is already Kratzert-exact (`app.metrics.nse` = 1 - MSE/var(o),
VAR_FLOOR 1e-3) — ONLY the series construction differs from eval_continuous.py:
    - stride the issue date by 1 day (default), not by the horizon;
    - run the FULL H=14 horizon (so the δHBV HBV core warms up — a horizon=1 chain
      truncates the physics and scores NSE ~-0.3, the documented row-52 trap);
    - keep only rows[0] (the t0+1 lead-1 prediction) for the daily series.

Cost & the --stride-days knob
-----------------------------
Stride-1 over 531 basins × ~3650 days = ~1.94M single-basin `mblstm.forecast()`
calls. For LSTM members this is cheap (minutes-to-tens-of-minutes, no HBV loop);
for δHBV the 365+14-step HBV Python loop makes full-531 stride-1 heavy on CPU.
Mitigations, all supported here:
  * validate the PROTOCOL on LSTM members locally first (cheap, exact);
  * --stride-days N (e.g. 7) subsamples ISSUE DATES (still lead-1-only) for a fast
    δHBV screen at 1/N cost — documented as an issue-date subsample, not a lead mix;
  * --limit-basins / --stride-stations for a smoke screen;
  * for the full-531 δHBV HEADLINE, run this as a Kaggle GPU notebook (the HBV
    forward is batched over basins in app/hbv.py; a future batched serving path
    makes it fast — see notebooks/kaggle_eval_day1.ipynb).

Validation (before quoting any headline)
----------------------------------------
Run on the per-forcing LSTM (v2r) ensemble and confirm it lands in the ~0.808
multi-forcing LSTM-ensemble band (Kratzert 2021). If it reproduces that, the
protocol is faithful and the δHBV/grand numbers are trustworthy.

Usage
-----
  RW2_ENABLE_MBLSTM=1 .venv/bin/python scripts/eval_day1_kratzert.py \
      --ckpt a.pt:b.pt:c.pt \
      --corpus-dir /Volumes/STORAGE_SD/riverwatch2_data/camels_corpus_daymet_v2 \
      --camels-subset 531 --start 1989-10-01 --end 1999-09-30 \
      --stride-days 1 --label daymet_v2r_day1kratzert
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("RW2_ENABLE_MBLSTM", "1")

from app import gages2  # noqa: E402
from app import mblstm  # noqa: E402
from app import metrics  # noqa: E402

STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
CAMELS_PATH = ROOT / "data" / "camels_gauge_ids.json"
CAMELS_ATTRS_PATH = ROOT / "data" / "camels_attrs.json"
OUT_DIR = ROOT / "benchmarks"
CONTEXT_DAYS = 365   # must match app.mblstm.CONTEXT_DAYS
HORIZON = 14         # run the full horizon (δHBV warmup); keep only lead-1


def load_camels_ids(which: str) -> set[str]:
    if which == "none" or not CAMELS_PATH.exists():
        return set()
    data = json.loads(CAMELS_PATH.read_text())
    ids = set(data["671"]) if which == "671" else set(data.get("531", data["671"]))
    return {str(s).strip().zfill(8) for s in ids}


def simulate_station(path: Path, attrs: dict, start: str, end: str,
                     stride_days: int = 1, min_days: int = 180
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Rolling lead-1 daily (date, obs, sim) series over [start, end] for one basin.

    Issue date t0 strides by `stride_days` (1 = every calendar day, the exact
    protocol). At each t0 we run the FULL H-day forecast (so the δHBV HBV core has
    its whole 365+H warmup) but keep ONLY the lead-1 (t0+1) prediction — one value
    per issued day at a constant 1-day lead. The observed discharge is only the
    encoder history channel; no-q-input CAMELS checkpoints zero it in forecast(),
    so this stays a pure rainfall-runoff sim.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    daily = df.set_index("date").reindex(
        pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D"))

    win_start = pd.Timestamp(start)
    win_end = pd.Timestamp(end)
    # forecast for calendar day D is issued at t0 = D-1; stride the issue date.
    t0 = win_start - pd.Timedelta(days=1)
    step = pd.Timedelta(days=stride_days)
    date_list, obs_list, sim_list = [], [], []
    while t0 + pd.Timedelta(days=1) <= win_end:
        if t0 not in daily.index:
            t0 += step
            continue
        hist = daily.loc[:t0]
        if len(hist) < CONTEXT_DAYS or pd.isna(hist["q_cfs"].iloc[-1]):
            t0 += step
            continue
        q_hist = hist["q_cfs"].dropna().rename("q_cfs").reset_index()
        q_hist.columns = ["date", "q_cfs"]
        if len(q_hist) < CONTEXT_DAYS:
            t0 += step
            continue
        wx_hist = hist.reset_index().rename(columns={"index": "date"})
        fut_idx = pd.date_range(t0 + pd.Timedelta(days=1), periods=HORIZON, freq="D")
        wx_fcst = daily.reindex(fut_idx).reset_index().rename(columns={"index": "date"})
        rows = mblstm.forecast(q_hist, wx_hist, wx_fcst, attrs, HORIZON)
        if rows and len(rows) == HORIZON:
            t = fut_idx[0]                          # lead-1 target day = t0+1
            truth = daily.loc[t, "q_cfs"] if t in daily.index else np.nan
            sim = rows[0].get("q_cfs")              # keep ONLY lead-1
            if sim is not None and np.isfinite(truth) and np.isfinite(sim):
                date_list.append(t)
                obs_list.append(float(truth))
                sim_list.append(float(sim))
        t0 += step

    if len(obs_list) < min_days:
        return None
    return (np.asarray(date_list), np.asarray(obs_list, dtype=float),
            np.asarray(sim_list, dtype=float))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint path (colon-separated for a seed ensemble)")
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--start", default="1989-10-01")
    ap.add_argument("--end", default="1999-09-30")
    ap.add_argument("--camels-subset", choices=["none", "671", "531"], default="531")
    ap.add_argument("--stride-days", type=int, default=1,
                    help="stride the ISSUE date by this many days (1 = exact "
                         "protocol, every calendar day; 7 = weekly screen, still "
                         "lead-1-only, ~1/7 cost — an issue-date subsample)")
    ap.add_argument("--stride-stations", type=int, default=1,
                    help="evaluate every Nth corpus station (fast screen)")
    ap.add_argument("--limit-basins", type=int, default=0)
    ap.add_argument("--min-days", type=int, default=180)
    ap.add_argument("--label", default="day1_kratzert")
    ap.add_argument("--dump-series", default="",
                    help="write per-day (station_id,date,truth,sim) lead-1 series "
                         "to this csv.gz (for combining member series)")
    args = ap.parse_args()

    os.environ["RW2_MBLSTM_CKPT_PATH"] = args.ckpt

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    files = sorted(p for p in corpus_dir.glob("*.csv.gz")
                   if not p.name.startswith("._"))
    if not files:  # exFAT/SD glob quirk
        files = sorted(p for p in corpus_dir.iterdir()
                       if p.suffix == ".gz" and p.name.endswith(".csv.gz")
                       and not p.name.startswith("._"))
    if not files:
        print(f"no corpus files in {corpus_dir}", file=sys.stderr)
        return 1
    if args.stride_stations > 1:
        files = files[:: args.stride_stations]
    if args.limit_basins:
        files = files[: args.limit_basins]

    registry = {s["id"]: s for s in json.loads(STATIONS_PATH.read_text())["stations"]}

    # CAMELS-static / δHBV checkpoints need the 27 Addor attrs overlaid at eval or
    # static_vector() is all-NaN and NSE craters to ~0.40 (the load-bearing fix).
    camels_attrs = {}
    try:
        mblstm._try_load()
        _cfg = mblstm._cfg or {}
        _sf = set(_cfg.get("static_feats", []))
        if CAMELS_ATTRS_PATH.exists() and {"p_mean", "aridity", "elev_mean"} <= _sf:
            camels_attrs = json.loads(CAMELS_ATTRS_PATH.read_text())
            print(f"CAMELS static overlay: {len(camels_attrs)} basins "
                  f"(checkpoint uses the 27-attr CAMELS static set)", flush=True)
    except Exception as exc:
        print(f"camels attrs overlay skipped: {exc}", flush=True)

    camels_ids = load_camels_ids(args.camels_subset)

    per_station: dict[str, dict] = {}
    series_rows: list = []
    t_start = time.time()
    for i, p in enumerate(files, 1):
        sid = p.name.split(".")[0]
        attrs = gages2.enrich_station_attrs(dict(registry.get(sid, {"id": sid})))
        if camels_attrs:
            for k, v in camels_attrs.get(sid, {}).items():
                if v is not None:
                    attrs[k] = v
        try:
            res = simulate_station(p, attrs, args.start, args.end,
                                   stride_days=args.stride_days, min_days=args.min_days)
        except Exception as exc:
            print(f"[{i}/{len(files)}] {sid} ERR {exc}", flush=True)
            continue
        if res is None:
            continue
        dates, obs, sim = res
        per_station[sid] = metrics.all_point_metrics(obs, sim)
        if args.dump_series:
            for d, o, s in zip(dates, obs, sim):
                series_rows.append((sid, pd.Timestamp(d).strftime("%Y-%m-%d"),
                                    float(o), float(s)))
        if i % 25 == 0:
            nses = [m["nse"] for m in per_station.values()
                    if np.isfinite(m.get("nse", np.nan))]
            med = float(np.median(nses)) if nses else float("nan")
            print(f"[{i}/{len(files)}] {len(per_station)} basins  "
                  f"median NSE so far={med:.3f}  ({time.time()-t_start:.0f}s)", flush=True)

    if not per_station:
        print("no basins simulated — check the checkpoint + corpus", file=sys.stderr)
        return 1

    full = metrics.aggregate(per_station)
    blocks = {"full": full}
    if camels_ids:
        inter = {sid: m for sid, m in per_station.items() if sid in camels_ids}
        if inter:
            blocks[f"camels_{args.camels_subset}"] = metrics.aggregate(inter)

    def med(block, k):
        return round(block[k]["median"], 4) if k in block else None

    print(f"\nDAY-1 (rolling lead-1) '{args.label}' "
          f"({len(per_station)} basins, {args.start}..{args.end}, "
          f"stride_days={args.stride_days}):")
    print(f"  full   : NSE {med(full,'nse')}  KGE {med(full,'kge')}  "
          f"log-NSE {med(full,'log_nse')}  FHV {med(full,'fhv')}  "
          f"(scorable {full.get('nse',{}).get('scorable')})")
    if f"camels_{args.camels_subset}" in blocks:
        cb = blocks[f"camels_{args.camels_subset}"]
        print(f"  CAMELS-{args.camels_subset}: NSE {med(cb,'nse')}  KGE {med(cb,'kge')}  "
              f"(record ref: Li/Shen 2025 median NSE 0.83; Kratzert LSTM-ens ~0.808)")

    if args.dump_series and series_rows:
        sp = Path(args.dump_series)
        sp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(series_rows, columns=["station_id", "date", "truth", "sim"]
                     ).to_csv(sp, index=False, compression="gzip")
        print(f"  wrote series {sp} ({len(series_rows)} rows)")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"day1kratzert_{args.label}.json"
    out_path.write_text(json.dumps({
        "label": args.label,
        "protocol": "rolling_lead1_daily_seq_to_one",
        "stride_days": args.stride_days,
        "members": args.ckpt.split(":"),
        "corpus_dir": str(corpus_dir),
        "window": [args.start, args.end],
        "n_basins": len(per_station),
        "metrics": blocks,
        "per_station_nse": {sid: m.get("nse") for sid, m in per_station.items()},
    }, indent=2))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
