#!/usr/bin/env python3
"""Benchmark forecasters across all 40 stations.

For each station:
  - pull last `eval_days + train_days` of daily discharge
  - hold out the last `eval_days`
  - evaluate persistence_lag1, runoff_ridge, chronos_bolt, ensemble blend
  - measure MAE/RMSE on the held-out window

Writes results to benchmarks/results_<timestamp>.json so we can diff across runs
as we improve the forecasting system.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import usgs, weather  # noqa: E402
from app.forecast import (  # noqa: E402
    HORIZON_DAYS,
    _get_chronos,
    _rolling_chronos_mae,
    _rolling_persistence_mae,
    chronos_forecast,
    persistence_forecast,
    runoff_ridge_forecast,
)
from sklearn.metrics import mean_absolute_error  # noqa: E402

STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
OUT_DIR = ROOT / "benchmarks"
OUT_DIR.mkdir(exist_ok=True)


def _metric_pair(yt, yh):
    yt = np.asarray(yt, dtype=float)
    yh = np.asarray(yh, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yh)
    if mask.sum() == 0:
        return {"mae": None, "rmse": None, "n": 0}
    yt = yt[mask]; yh = yh[mask]
    return {
        "mae": float(np.mean(np.abs(yt - yh))),
        "rmse": float(np.sqrt(np.mean((yt - yh) ** 2))),
        "n": int(len(yt)),
    }


def evaluate_one(station: dict, train_days: int, eval_days: int, horizon: int) -> dict:
    sid = station["id"]
    today = date.today()
    eval_start = today - timedelta(days=eval_days)
    train_end = eval_start - timedelta(days=1)
    train_start = train_end - timedelta(days=train_days)

    q_full = usgs.fetch_daily_discharge(sid, train_start, today)
    if q_full.empty or len(q_full) < 60:
        return {"station_id": sid, "error": "not enough discharge data", "rows": len(q_full)}

    q_train = q_full[q_full["date"] <= train_end].reset_index(drop=True)
    q_eval = q_full[q_full["date"] > train_end].reset_index(drop=True)
    if len(q_eval) < 5:
        return {"station_id": sid, "error": "not enough eval rows", "rows": len(q_eval)}

    try:
        wx_hist = weather.fetch_history(station["lat"], station["lon"], train_start, today - timedelta(days=1))
    except Exception:
        wx_hist = pd.DataFrame(columns=["date"] + weather.DAILY_VARS)
    try:
        wx_future = weather.fetch_history(station["lat"], station["lon"], eval_start, today)
    except Exception:
        wx_future = pd.DataFrame(columns=["date"] + weather.DAILY_VARS)

    eval_horizon = min(horizon, len(q_eval))
    yt = q_eval["q_cfs"].iloc[:eval_horizon].values

    out = {"station_id": sid, "name": station.get("name"), "state": station.get("state"),
           "rows_train": int(len(q_train)), "rows_eval": int(len(q_eval)),
           "eval_horizon": int(eval_horizon),
           "members": {}}

    # 1) persistence
    persist_pred = persistence_forecast(q_train, eval_horizon)
    out["members"]["persistence_lag1"] = _metric_pair(yt, persist_pred)

    # 2) ridge (direct multi-step on q_train, with weather knowledge of the eval window)
    try:
        ridge_pred, ridge_mae_info = runoff_ridge_forecast(q_train, wx_hist, wx_future, eval_horizon)
    except Exception as exc:
        ridge_pred = persist_pred
        ridge_mae_info = {}
        out["ridge_error"] = str(exc)
    out["members"]["runoff_ridge"] = _metric_pair(yt, ridge_pred)

    # 3) chronos zero-shot
    chronos_pred = chronos_forecast(q_train, eval_horizon)
    if chronos_pred is not None:
        out["members"]["chronos_bolt"] = _metric_pair(yt, chronos_pred)
    else:
        out["members"]["chronos_bolt"] = {"mae": None, "rmse": None, "n": 0}
        out["chronos"] = "unavailable"

    # 4) blend (inverse-MAE based on rolling validation on training data)
    member_preds = {"persistence_lag1": persist_pred, "runoff_ridge": ridge_pred}
    if chronos_pred is not None:
        member_preds["chronos_bolt"] = chronos_pred

    rolling_mae = {}
    pm = _rolling_persistence_mae(q_train, eval_horizon)
    if pm is not None:
        rolling_mae["persistence_lag1"] = pm
    if "mae_mean" in ridge_mae_info:
        rolling_mae["runoff_ridge"] = ridge_mae_info["mae_mean"]
    if chronos_pred is not None:
        cm = _rolling_chronos_mae(q_train, eval_horizon)
        if cm is not None:
            rolling_mae["chronos_bolt"] = cm

    weights = {}
    for name in member_preds:
        m = rolling_mae.get(name)
        weights[name] = (1.0 / m) if (m and math.isfinite(m) and m > 0) else 0.05
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    blend = []
    for h in range(eval_horizon):
        s = 0.0; ws = 0.0
        for name, w in weights.items():
            v = member_preds[name][h]
            if v is None or not math.isfinite(v):
                continue
            s += w * v; ws += w
        blend.append(s / ws if ws > 0 else float("nan"))
    out["members"]["ensemble_blend"] = _metric_pair(yt, blend)
    out["weights"] = weights
    out["rolling_mae_train"] = rolling_mae
    return out


def aggregate(results: list[dict]) -> dict:
    members = ["persistence_lag1", "runoff_ridge", "chronos_bolt", "ensemble_blend"]
    agg = {}
    for name in members:
        maes = [r["members"][name]["mae"] for r in results if r.get("members", {}).get(name, {}).get("mae") is not None]
        if not maes:
            agg[name] = {"mae_mean": None, "stations": 0}
        else:
            agg[name] = {"mae_mean": float(mean(maes)), "mae_median": float(np.median(maes)), "stations": len(maes)}
    return agg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-days", type=int, default=540)
    ap.add_argument("--eval-days", type=int, default=14)
    ap.add_argument("--horizon", type=int, default=HORIZON_DAYS)
    ap.add_argument("--limit", type=int, default=0, help="limit to first N stations (0 = all)")
    ap.add_argument("--label", default="baseline", help="label written into the results filename")
    args = ap.parse_args()

    payload = json.loads(STATIONS_PATH.read_text())
    stations = payload["stations"]
    if args.limit:
        stations = stations[: args.limit]

    print(f"Benchmarking {len(stations)} stations  train_days={args.train_days}  eval_days={args.eval_days}  horizon={args.horizon}")
    print(f"Chronos available: {_get_chronos() is not None}")
    print()

    results = []
    t0 = time.time()
    for i, st in enumerate(stations, 1):
        ts = time.time()
        try:
            r = evaluate_one(st, args.train_days, args.eval_days, args.horizon)
        except Exception as exc:
            r = {"station_id": st["id"], "error": str(exc)}
        elapsed = time.time() - ts
        line = f"[{i:>2}/{len(stations)}] {st['id']} ({st['state']}) {elapsed:5.1f}s "
        if "error" in r:
            line += f"ERR  {r['error']}"
        else:
            for name in ("persistence_lag1", "runoff_ridge", "chronos_bolt", "ensemble_blend"):
                m = r["members"].get(name, {}).get("mae")
                line += f"{name[:8]}={m if m is None else f'{m:6.1f}'}  "
        print(line)
        results.append(r)

    agg = aggregate([r for r in results if "error" not in r])
    summary = {
        "label": args.label,
        "ran_at": pd.Timestamp.utcnow().isoformat(),
        "args": vars(args),
        "stations": len(results),
        "succeeded": sum(1 for r in results if "error" not in r),
        "aggregate_mae": agg,
        "results": results,
    }
    out = OUT_DIR / f"results_{args.label}_{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"Aggregate MAE (mean across stations):")
    for k, v in agg.items():
        if v.get("mae_mean") is None:
            print(f"  {k:>20s}:    n/a")
        else:
            print(f"  {k:>20s}:  {v['mae_mean']:8.2f}  (n={v['stations']})")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
