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
is honored exactly. Continuous simulation for a windowed 365→H forecast model =
chain NON-OVERLAPPING H-day forecast windows across the decade: issue at t0
(365-day encoder context), take the FULL H-day forecast, advance t0 by H, repeat.
This yields a gap-free daily hydrograph while running the model exactly as
trained — the full H horizon is essential for δHBV, whose HBV core needs the
whole 365+H sequence to warm up (an earlier h=1-only chain truncated the physics
and scored NSE ~-0.3 vs 0.82 on the proxy — a bug, now fixed). The observed
streamflow is only the encoder history channel and the no-q-input CAMELS
checkpoints zero it, so this is a pure rainfall-runoff sim, not discharge
assimilation. NSE is the full pooled (obs, sim) decade series per basin via
`app.metrics.all_point_metrics` + `aggregate`, comparable to the proxy JSONs.

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
                     min_days: int = 180, horizon: int = 14
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """Continuous daily (obs, sim) series over [start, end] for one basin.

    Continuous simulation for a windowed 365→H forecast model = chain
    NON-OVERLAPPING H-day forecast windows across the decade: issue at t0 (using
    the 365 prior days as encoder context), take the full H-day forecast, advance
    t0 by H, repeat. This gives a gap-free daily hydrograph while serving the
    model exactly as trained (full H horizon — critical for δHBV, whose HBV core
    needs the whole 365+H sequence to warm up; the old h=1 chain truncated that
    and produced garbage). The observed streamflow (q_hist) is only the encoder's
    history channel — the no-q-input CAMELS checkpoints ZERO it in forecast(), so
    this stays a pure-rainfall-runoff sim, not discharge assimilation.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    daily = df.set_index("date").reindex(
        pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D"))

    win_start = pd.Timestamp(start)
    win_end = pd.Timestamp(end)
    # issue dates spaced H apart; the forecast covers t0+1 .. t0+H
    t0 = win_start - pd.Timedelta(days=1)
    obs_list, sim_list = [], []
    while t0 + pd.Timedelta(days=horizon) <= win_end:
        if t0 not in daily.index:
            t0 += pd.Timedelta(days=horizon)
            continue
        hist = daily.loc[:t0]
        if len(hist) < CONTEXT_DAYS or pd.isna(hist["q_cfs"].iloc[-1]):
            t0 += pd.Timedelta(days=horizon)
            continue
        q_hist = hist["q_cfs"].dropna().rename("q_cfs").reset_index()
        q_hist.columns = ["date", "q_cfs"]
        if len(q_hist) < CONTEXT_DAYS:
            t0 += pd.Timedelta(days=horizon)
            continue
        wx_hist = hist.reset_index().rename(columns={"index": "date"})
        fut_idx = pd.date_range(t0 + pd.Timedelta(days=1), periods=horizon, freq="D")
        wx_fcst = daily.reindex(fut_idx).reset_index().rename(columns={"index": "date"})
        rows = mblstm.forecast(q_hist, wx_hist, wx_fcst, attrs, horizon)
        if rows and len(rows) == horizon:
            for h, t in enumerate(fut_idx):
                truth = daily.loc[t, "q_cfs"] if t in daily.index else np.nan
                sim = rows[h].get("q_cfs")
                if sim is not None and np.isfinite(truth) and np.isfinite(sim):
                    obs_list.append(float(truth))
                    sim_list.append(float(sim))
        t0 += pd.Timedelta(days=horizon)

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
