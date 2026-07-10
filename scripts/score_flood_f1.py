#!/usr/bin/env python3
"""Flood-event P/R/F1 at return-period thresholds (Nearing et al. 2024 Nature),
scored offline from a backtest_mblstm.py --dump-windows csv.gz.

Protocol notes (and where we deviate, honestly):
  * Observed thresholds come from the station's FULL corpus record (annual
    maxima over water years, GEV — app.metrics.return_period_thresholds);
    stations without >= 10 good water years are skipped, not guessed.
  * Simulated thresholds are computed SEPARATELY on the model's own series,
    per the paper — a biased-but-sharp model is scored on its own flood
    frequency. Two modes:
      own-record        annual maxima of the simulated series itself. Honest,
                        but needs >= 10 water years of simulation — i.e. a
                        chained decade dump (CAMELS-style). The default.
      matched-quantile  sim threshold = the sim-series quantile at the same
                        exceedance probability the obs threshold has in the
                        scored window's obs. For short windows (the 2025
                        real-forcing year) where own-record is impossible;
                        forces event RATES to match, so it mostly measures
                        timing — flagged in the output.
  * Both the paper's ±2-day hit window and the 0-day (same-day) variant its
    critics report are always emitted.
  * chained mode stitches t0+h across leads into one continuous daily series
    (chained stride-14 dumps cover every day exactly once) — simulation-style
    flood skill. per-lead mode scores each lead's own daily series and needs
    a stride-1 dump: NaN gaps split events (see metrics._event_starts).

Usage:
  .venv/bin/python scripts/score_flood_f1.py \
      --dump data/mblstm/dumps/camels531_mse_ens2.csv.gz \
      --corpus-dir data/mblstm/camels_corpus --mode chained \
      --label camels531_mse_ens2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from app.metrics import (annual_maxima, flood_event_scores,  # noqa: E402
                         return_period_thresholds)

RETURN_PERIODS = (1.0, 2.0, 5.0, 10.0)
WINDOWS = (2, 0)  # paper's ±2-day and the critics' same-day variant
OUT_DIR = ROOT / "benchmarks"


def water_year(dates: pd.Series) -> np.ndarray:
    """USGS water year: Oct 1 of year Y-1 through Sep 30 of year Y."""
    d = pd.DatetimeIndex(dates)
    return (d.year + (d.month >= 10)).to_numpy()


def obs_thresholds(corpus_path: Path) -> dict | None:
    df = pd.read_csv(corpus_path, usecols=["date", "q_cfs"])
    am = annual_maxima(df["q_cfs"].to_numpy(), water_year(df["date"]))
    thr = return_period_thresholds(am, years=RETURN_PERIODS)
    return thr if np.isfinite(list(thr.values())).any() else None


def daily_series(sub: pd.DataFrame, point: str) -> pd.DataFrame:
    """(date, obs, sim) daily frame from dump rows, averaging duplicates
    (a stride < horizon dump covers dates from several windows)."""
    dates = pd.to_datetime(sub["t0"]) + pd.to_timedelta(sub["h"], unit="D")
    g = (pd.DataFrame({"date": dates, "obs": sub["truth"], "sim": sub[point]})
         .groupby("date", as_index=False).mean())
    full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    return g.set_index("date").reindex(full)


def sim_thresholds(series: pd.DataFrame, mode: str, obs_thr: dict) -> dict:
    sim = series["sim"].to_numpy()
    if mode == "own-record":
        am = annual_maxima(sim, water_year(series.index))
        return return_period_thresholds(am, years=RETURN_PERIODS)
    # matched-quantile: same exceedance probability as the obs threshold
    # carries in this window's observations.
    obs = series["obs"].to_numpy()
    fin_o, fin_s = obs[np.isfinite(obs)], sim[np.isfinite(sim)]
    out = {}
    for t, thr in obs_thr.items():
        p_exc = float(np.mean(fin_o > thr)) if len(fin_o) else 0.0
        out[t] = (float(np.quantile(fin_s, 1.0 - p_exc))
                  if p_exc > 0 and len(fin_s) else float("nan"))
    return out


def score_station(series: pd.DataFrame, o_thr: dict, s_thr: dict) -> dict:
    obs, sim = series["obs"].to_numpy(), series["sim"].to_numpy()
    return {str(t): {str(w): flood_event_scores(obs, sim, o_thr[t], s_thr[t],
                                                window_days=w)
                     for w in WINDOWS}
            for t in RETURN_PERIODS}


def aggregate_stations(per_station: dict) -> dict:
    """Per (T, window): station-median/mean F1/P/R over stations with >= 1
    observed event, plus a pooled micro-average from reconstructed counts."""
    out: dict = {}
    for t in RETURN_PERIODS:
        for w in WINDOWS:
            key = f"rp{t:g}_w{w}"
            f1s, ps, rs = [], [], []
            tp_s = fp_s = tp_o = fn_o = 0.0
            for scores in per_station.values():
                s = scores[str(t)][str(w)]
                if s["n_obs_events"] < 1:
                    continue
                if np.isfinite(s.get("f1", float("nan"))):
                    f1s.append(s["f1"]); ps.append(s["precision"]); rs.append(s["recall"])
                if np.isfinite(s.get("precision", float("nan"))):
                    tp_s += s["precision"] * s["n_sim_events"]
                    fp_s += (1 - s["precision"]) * s["n_sim_events"]
                if np.isfinite(s.get("recall", float("nan"))):
                    tp_o += s["recall"] * s["n_obs_events"]
                    fn_o += (1 - s["recall"]) * s["n_obs_events"]
            micro_p = tp_s / (tp_s + fp_s) if tp_s + fp_s > 0 else float("nan")
            micro_r = tp_o / (tp_o + fn_o) if tp_o + fn_o > 0 else float("nan")
            micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                        if np.isfinite(micro_p) and np.isfinite(micro_r)
                        and micro_p + micro_r > 0 else float("nan"))
            out[key] = {
                "f1_station_median": float(np.median(f1s)) if f1s else float("nan"),
                "f1_station_mean": float(np.mean(f1s)) if f1s else float("nan"),
                "precision_station_median": float(np.median(ps)) if ps else float("nan"),
                "recall_station_median": float(np.median(rs)) if rs else float("nan"),
                "scorable_stations": len(f1s),
                "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--mode", choices=("chained", "per-lead"), default="chained")
    ap.add_argument("--leads", default="1-7",
                    help="per-lead mode: inclusive lead range, e.g. 1-7")
    ap.add_argument("--point", choices=("ymed", "ymean"), default="ymed")
    ap.add_argument("--sim-thresholds", choices=("own-record", "matched-quantile"),
                    default="own-record")
    ap.add_argument("--label", default="flood_f1")
    args = ap.parse_args()

    t_start = time.time()
    dump = pd.read_csv(args.dump)
    corpus = Path(args.corpus_dir)
    lo, hi = (int(x) for x in args.leads.split("-"))

    skips = {"no_corpus": 0, "short_obs_record": 0, "short_sim_record": 0}
    results: dict = {}   # per-lead label -> per-station scores
    sids = sorted(dump["station_id"].astype(str).str.zfill(8).unique())
    dump["sid"] = dump["station_id"].astype(str).str.zfill(8)

    lead_sets = ([("chained", None)] if args.mode == "chained"
                 else [(f"lead{h}", h) for h in range(lo, hi + 1)])
    for label, lead in lead_sets:
        results[label] = {}

    for i, sid in enumerate(sids, 1):
        cpath = corpus / f"{sid}.csv.gz"
        if not cpath.exists():
            skips["no_corpus"] += 1
            continue
        o_thr = obs_thresholds(cpath)
        if o_thr is None:
            skips["short_obs_record"] += 1
            continue
        sub_all = dump[dump["sid"] == sid]
        for label, lead in lead_sets:
            sub = sub_all if lead is None else sub_all[sub_all["h"] == lead]
            if not len(sub):
                continue
            series = daily_series(sub, args.point)
            s_thr = sim_thresholds(series, args.sim_thresholds, o_thr)
            if not np.isfinite(list(s_thr.values())).any():
                skips["short_sim_record"] += 1
                continue
            results[label][sid] = score_station(series, o_thr, s_thr)
        if i % 100 == 0:
            print(f"[{i}/{len(sids)}] {time.time() - t_start:.0f}s", flush=True)

    payload = {
        "label": args.label,
        "dump": args.dump,
        "mode": args.mode,
        "point": args.point,
        "sim_thresholds": args.sim_thresholds,
        "return_periods": list(RETURN_PERIODS),
        "hit_windows_days": list(WINDOWS),
        "n_stations_scored": {k: len(v) for k, v in results.items()},
        "skips": skips,
        "aggregate": {k: aggregate_stations(v) for k, v in results.items()},
        "caveats": [
            "obs thresholds: GEV on full-corpus water-year annual maxima "
            "(>=10 good years required); sim thresholds per --sim-thresholds.",
            "matched-quantile mode equalizes event rates by construction — "
            "it measures timing, not frequency; use own-record when the "
            "simulated span allows.",
            "chained mode mixes leads into one continuous series "
            "(simulation-style skill); per-lead needs a stride-1 dump.",
        ],
    }
    out = OUT_DIR / f"flood_f1_{args.label}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({time.time() - t_start:.1f}s)")
    for label in results:
        agg = payload["aggregate"][label]
        for key in ("rp2_w2", "rp2_w0"):
            if key in agg and agg[key]["scorable_stations"]:
                a = agg[key]
                print(f"  {label} {key}: median F1 {a['f1_station_median']:.3f} "
                      f"(P {a['precision_station_median']:.3f} / "
                      f"R {a['recall_station_median']:.3f}, "
                      f"n={a['scorable_stations']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
