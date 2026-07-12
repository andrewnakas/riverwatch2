#!/usr/bin/env python3
"""Publication-exact CONTINUOUS-DAILY-SIMULATION eval for the CAMELS-531 record.

Why this exists
---------------
Every campaign number so far (day-1 NSE 0.816, the incoming δHBV ~0.83) is the
*windowed 14-day-forecast* proxy: a 365-day encoder → 14-day forecast, scored
per horizon over a stride of issue dates. The published record (Li/Shen 2025,
HESS 29:6829 — median NSE 0.83 on CAMELS-531) is measured on **continuous daily
simulation**: one uninterrupted daily hydrograph per basin over the whole test
decade, scored once. Those are different protocols, so "0.85 > 0.83" is only a
real claim if it is measured the same way. This script produces that number.

The continuous-sim approximation (documented so it is auditable)
---------------------------------------------------------------
We build each basin's decade-long daily series through the SAME serving path
(`app.mblstm.forecast`) the proxy uses, so LSTM and δHBV members are treated
identically and the checkpoint recipe (q-transform, no-q-input, static overlay)
is honored exactly. For every issue date t0 in the test decade we call
forecast() and keep only the **h=1 (day-ahead) prediction** — chaining those
day-1 values across all issue dates yields a continuous daily hydrograph. This
is the honest continuous-simulation analogue for a day-ahead forecast model:
  * For the LSTM, day-1 with perfect (observed) forcing is the closest thing to
    a one-step simulation the model was trained to produce.
  * For δHBV, HBV is a genuine continuous simulator; its day-1 output over a
    rolling daily issue is a continuous sim by construction.
Because we roll DAILY (stride 1) the resulting series has no gaps, and NSE is
computed on the full pooled (obs, sim) decade series per basin — exactly the
functional Li/Shen report. We reuse `app.metrics.all_point_metrics` +
`aggregate`, so the median-NSE is byte-comparable to the proxy JSONs.

Cost note: stride-1 over 531 basins × ~3650 days is heavy on CPU (the δHBV
379-step HBV loop especially). Use --limit-basins / --stride-stations to
subsample for a fast screen; the full run is the headline pass.

Usage
-----
  RW2_ENABLE_MBLSTM=1 .venv/bin/python scripts/eval_continuous.py \
      --ckpt a.pt:b.pt:c.pt \
      --corpus-dir /Volumes/STORAGE_SD/riverwatch2_data/camels_corpus_daymet_v2 \
      --camels-subset 531 --start 1989-10-01 --end 1999-09-30 \
      --label daymet_v2r_continuous
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
CONTEXT_DAYS = 365  # must match app.mblstm.CONTEXT_DAYS (forecast() needs >=365 hist)


def load_camels_ids(which: str) -> set[str]:
    """CAMELS-US gauge ids for the requested subset (mirrors backtest_mblstm)."""
    if which == "none" or not CAMELS_PATH.exists():
        return set()
    data = json.loads(CAMELS_PATH.read_text())
    ids = set(data["671"]) if which == "671" else set(data.get("531", data["671"]))
    return {str(s).strip().zfill(8) for s in ids}


def simulate_station(path: Path, attrs: dict, start: str, end: str,
                     min_days: int = 180) -> tuple[np.ndarray, np.ndarray] | None:
    """Roll a continuous daily (obs, sim) series over [start, end] for one basin.

    For each day t0 in the window with >=365 days of prior history, call
    forecast() and take the h=1 prediction as that day's simulated flow. Returns
    aligned (obs, sim) 1-D arrays over the scorable days, or None if too few.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    daily = df.set_index("date").reindex(
        pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D"))

    win_days = pd.date_range(start, end, freq="D")
    obs_list, sim_list = [], []
    for t0 in win_days:
        # t0 is the issue date; we predict flow at t0+1 (h=1). Need the observed
        # truth at t0+1 and >=365 days of history ending at t0.
        t1 = t0 + pd.Timedelta(days=1)
        if t0 not in daily.index or t1 not in daily.index:
            continue
        truth = daily.loc[t1, "q_cfs"]
        if not np.isfinite(truth):
            continue
        hist = daily.loc[:t0]
        if len(hist) < CONTEXT_DAYS + 1:
            continue
        q_hist = hist["q_cfs"].dropna().rename("q_cfs").reset_index()
        q_hist.columns = ["date", "q_cfs"]
        if len(q_hist) < CONTEXT_DAYS:
            continue
        if pd.isna(hist["q_cfs"].iloc[-1]):
            # forecast() needs a defined last observation for the history window
            continue
        # keep ALL weather columns; norm_wx selects cfg enc/dec vars itself
        wx_hist = hist.reset_index().rename(columns={"index": "date"})
        # decoder forcing = observed weather at t0+1 (perfect forcing, day-1 only)
        wx_fcst = daily.loc[[t1]].reset_index().rename(columns={"index": "date"})
        rows = mblstm.forecast(q_hist, wx_hist, wx_fcst, attrs, 1)
        if not rows:
            continue
        sim = rows[0].get("q_cfs")
        if sim is None or not np.isfinite(sim):
            continue
        obs_list.append(float(truth))
        sim_list.append(float(sim))

    if len(obs_list) < min_days:
        return None
    return np.asarray(obs_list, dtype=float), np.asarray(sim_list, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint path (colon-separated for a seed ensemble, "
                         "like backtest_mblstm)")
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--start", default="1989-10-01")
    ap.add_argument("--end", default="1999-09-30")
    ap.add_argument("--camels-subset", choices=["none", "671", "531"], default="531")
    ap.add_argument("--stations-file", default="",
                    help="restrict to ids in this file (JSON array / {ids:...} / "
                         "one-per-line)")
    ap.add_argument("--stride-stations", type=int, default=1,
                    help="evaluate every Nth corpus station (fast screen)")
    ap.add_argument("--limit-basins", type=int, default=0,
                    help="cap number of basins (smoke/screen)")
    ap.add_argument("--min-days", type=int, default=180,
                    help="min scorable days for a basin to count")
    ap.add_argument("--label", default="continuous")
    args = ap.parse_args()

    os.environ["RW2_MBLSTM_CKPT_PATH"] = args.ckpt

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    # glob can miss on flaky exFAT mounts; fall back to os.scandir via rglob
    files = sorted(p for p in corpus_dir.glob("*.csv.gz")
                   if not p.name.startswith("._"))
    if not files:  # exFAT/SD glob quirk — retry via iterdir
        files = sorted(p for p in corpus_dir.iterdir()
                       if p.suffix == ".gz" and p.name.endswith(".csv.gz")
                       and not p.name.startswith("._"))
    if not files:
        print(f"no corpus files in {corpus_dir}", file=sys.stderr)
        return 1

    if args.stations_file:
        raw = Path(args.stations_file).read_text()
        try:
            obj = json.loads(raw)
            keep = set(obj["matched_in_corpus_usgs_ids"] if isinstance(obj, dict) else obj)
        except json.JSONDecodeError:
            keep = {ln.strip() for ln in raw.splitlines() if ln.strip()}
        files = [p for p in files if p.name.split(".")[0] in keep]
    if args.stride_stations > 1:
        files = files[:: args.stride_stations]
    if args.limit_basins:
        files = files[: args.limit_basins]

    registry = {s["id"]: s for s in json.loads(STATIONS_PATH.read_text())["stations"]}

    # CAMELS-static / δHBV checkpoints need the 27 Addor attrs overlaid at eval
    # (same load-bearing fix as backtest_mblstm — without it static_vector() is
    # all-NaN and NSE craters to ~0.40). Overlay when the loaded model's static
    # set is the CAMELS one.
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
    t_start = time.time()
    for i, p in enumerate(files, 1):
        sid = p.name.split(".")[0]
        attrs = gages2.enrich_station_attrs(dict(registry.get(sid, {"id": sid})))
        if camels_attrs:
            for k, v in camels_attrs.get(sid, {}).items():
                if v is not None:
                    attrs[k] = v
        try:
            res = simulate_station(p, attrs, args.start, args.end, args.min_days)
        except Exception as exc:
            print(f"[{i}/{len(files)}] {sid} ERR {exc}", flush=True)
            continue
        if res is None:
            continue
        obs, sim = res
        per_station[sid] = metrics.all_point_metrics(obs, sim)
        if i % 25 == 0:
            nses = [m["nse"] for m in per_station.values() if np.isfinite(m.get("nse", np.nan))]
            med = float(np.median(nses)) if nses else float("nan")
            print(f"[{i}/{len(files)}] {len(per_station)} basins  "
                  f"median NSE so far={med:.3f}  ({time.time()-t_start:.0f}s)", flush=True)

    if not per_station:
        print("no basins simulated — check the checkpoint + corpus", file=sys.stderr)
        return 1

    full = metrics.aggregate(per_station)
    # CAMELS-subset block (the headline number vs the 0.83 record)
    blocks = {"full": full}
    if camels_ids:
        inter = {sid: m for sid, m in per_station.items() if sid in camels_ids}
        if inter:
            blocks[f"camels_{args.camels_subset}"] = metrics.aggregate(inter)

    def med(block, k):
        return round(block[k]["median"], 4) if k in block else None

    print(f"\ncontinuous daily sim '{args.label}' "
          f"({len(per_station)} basins, {args.start}..{args.end}):")
    print(f"  full   : NSE {med(full,'nse')}  KGE {med(full,'kge')}  "
          f"log-NSE {med(full,'log_nse')}  FHV {med(full,'fhv')}  "
          f"(scorable {full.get('nse',{}).get('scorable')})")
    if f"camels_{args.camels_subset}" in blocks:
        cb = blocks[f"camels_{args.camels_subset}"]
        print(f"  CAMELS-{args.camels_subset}: NSE {med(cb,'nse')}  KGE {med(cb,'kge')}  "
              f"(record ref: Li/Shen 2025 median NSE 0.83)")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"continuous_{args.label}.json"
    out_path.write_text(json.dumps({
        "label": args.label,
        "protocol": "continuous_daily_sim_day1_chain",
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
