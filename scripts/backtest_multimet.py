#!/usr/bin/env python3
"""Score the shipped MB-LSTM on Caravan MultiMet HRES forcings — the
apples-to-apples number vs AIFL and Google's baselines on shared infrastructure.

Runs the serving entry point (app.mblstm.forecast) over MultiMet's US CAMELS
gauges using THEIR archived IFS-HRES forecasts (app.multimet) as decoder
forcing, encoder history from the CAMELS-531 corpus (Daymet + USGS obs), scored
against USGS observed flow. Because MultiMet's 671 US basins ⊇ the 531 CAMELS
basins AIFL/Google evaluate on, this is directly comparable to their published
numbers over the same 2021-2024 window.

Reference (2021-2024, CAMELS/Caravan US gauges):
    AIFL   median NSE 0.518, KGE' 0.636   (arXiv:2602.16579)
    Google median NSE 0.624, KGE' 0.678

ENCODER HISTORY: the encoder needs >=365 days of history (obs discharge +
hindcast weather) ENDING at each issue date. Use the Caravan corpus
(scripts/build_caravan_corpus.py --q-units cfs), which carries obs discharge +
ERA5-Land forcing 1981-2020. Caravan obs ends 2020-12-30, so the fully-sourced
evaluation window is 2016-01-01..2020-12-31 (1827 MultiMet issue dates with
both Caravan history AND Caravan obs targets) — same forcings/gauges/protocol
as AIFL & Google, a DIFFERENT window than their 2021-2024 (which would need
USGS obs+history extended past 2020). This window difference is a documented
methodological caveat, not a defect. The CAMELS Daymet corpus (ends 2014)
does NOT work here — zero overlap with MultiMet's 2016+ dates.

Honest caveats, surfaced in the output:
  * HRES leads cap at 10 (ours normally goes 14); scored 1-10 here.
  * MultiMet ships temperature_2m only (no tmax/tmin) and no shortwave — the
    decoder's radiation channel is NaN (see app.multimet). This handicaps us
    vs our own 5-var forcing; it is the price of using THEIR exact archive.
  * Encoder still uses the Daymet-based CAMELS corpus for history (obs +
    hindcast weather); only the forecast/decoder side is MultiMet HRES.

Usage:
  RW2_ENABLE_MBLSTM=1 .venv/bin/python scripts/backtest_multimet.py \
      --ckpt <cmalv2p ckpts> --corpus-dir data/mblstm/camels_corpus \
      --start 2021-01-01 --end 2024-09-30 --stride 14 --label multimet_camels531
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RW2_ENABLE_MBLSTM", "1")

from app import gages2, mblstm, multimet  # noqa: E402
from app.metrics import aggregate, all_point_metrics  # noqa: E402
from app.weather import DAILY_VARS  # noqa: E402

STATIONS_PATH = ROOT / "data" / "stations_v15.json"
HORIZON = 10
OUT_DIR = ROOT / "benchmarks"


def eval_station(corpus_path: Path, attrs: dict, t0_list: list[date],
                 skips: dict) -> dict | None:
    sid = corpus_path.name.split(".")[0]
    df = pd.read_csv(corpus_path)
    df["date"] = pd.to_datetime(df["date"])
    daily = df.sort_values("date").set_index("date").reindex(
        pd.date_range(df["date"].min(), df["date"].max(), freq="D"))

    per_h_truth: dict[int, list[float]] = {h: [] for h in range(1, HORIZON + 1)}
    per_h_pred: dict[int, list[float]] = {h: [] for h in range(1, HORIZON + 1)}
    for t0 in t0_list:
        ts0 = pd.Timestamp(t0)
        if ts0 not in daily.index:
            skips["no_t0"] += 1
            continue
        hist = daily.loc[:ts0]
        if len(hist) < 400 or pd.isna(hist["q_cfs"].iloc[-1]):
            skips["short_or_no_q"] += 1
            continue
        wx_fcst = multimet.forcing_window(sid, t0, HORIZON)
        if wx_fcst is None:
            skips["no_multimet"] += 1
            continue
        q_hist = hist["q_cfs"].dropna().rename("q_cfs").reset_index()
        q_hist.columns = ["date", "q_cfs"]
        # reindex (not select) so vars absent from the compat corpus become
        # NaN columns the model tolerates, matching backtest_mblstm.py — a
        # plain [] select raises KeyError on the 8 non-compat DAILY_VARS.
        wx_hist = (hist.reset_index().rename(columns={"index": "date"})
                   .reindex(columns=["date"] + DAILY_VARS))
        rows = mblstm.forecast(q_hist, wx_hist, wx_fcst, attrs, HORIZON)
        if not rows or len(rows) < HORIZON:
            skips["forecast_none"] += 1
            continue
        for h in range(1, HORIZON + 1):
            truth = daily["q_cfs"].get(ts0 + pd.Timedelta(days=h), np.nan)
            if np.isfinite(truth):
                per_h_truth[h].append(float(truth))
                per_h_pred[h].append(float(rows[h - 1]["q_cfs"]))

    obs = np.concatenate([np.asarray(per_h_truth[h]) for h in range(1, HORIZON + 1)])
    sim = np.concatenate([np.asarray(per_h_pred[h]) for h in range(1, HORIZON + 1)])
    if len(obs) < 20:
        return None
    out = all_point_metrics(obs, sim)
    out["windows"] = int(len(obs))
    out["by_h_nse"] = {}
    from app.metrics import nse
    for h in range(1, HORIZON + 1):
        if len(per_h_truth[h]) >= 20:
            out["by_h_nse"][h] = nse(per_h_truth[h], per_h_pred[h])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus-dir",
                    default="/Volumes/STORAGE_SD/riverwatch2_data/caravan_corpus_531_cfs",
                    help="Caravan cfs corpus (obs+ERA5 history 1981-2020)")
    ap.add_argument("--start", default="2016-01-01",
                    help="Caravan obs ends 2020-12-30; the fully-sourced window "
                         "is 2016-2020 (see module docstring)")
    ap.add_argument("--end", default="2020-12-31")
    ap.add_argument("--stride", type=int, default=14)
    ap.add_argument("--point", default="mean")
    ap.add_argument("--limit-stations", type=int, default=0)
    ap.add_argument("--label", default="multimet_camels531")
    args = ap.parse_args()

    os.environ["RW2_MBLSTM_CKPT_PATH"] = args.ckpt
    os.environ["RW2_MBLSTM_POINT"] = args.point

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    mm_ids = set(multimet.us_gauge_ids())
    files = sorted(p for p in corpus_dir.glob("*.csv.gz")
                   if not p.name.startswith("._") and p.name.split(".")[0] in mm_ids)
    if args.limit_stations:
        files = files[: args.limit_stations]
    print(f"cohort: {len(files)} gauges (corpus ∩ MultiMet-US)")

    all_mm = multimet.issue_dates()
    lo, hi = pd.Timestamp(args.start).date(), pd.Timestamp(args.end).date()
    t0_list = [d for d in all_mm if lo <= d <= hi][:: max(1, args.stride)]
    print(f"issue dates: {len(t0_list)} ({t0_list[0]}..{t0_list[-1]}, stride {args.stride})")

    reg_raw = json.loads(STATIONS_PATH.read_text())
    reg_list = reg_raw["stations"] if isinstance(reg_raw, dict) else reg_raw
    registry = {s["id"]: s for s in reg_list}
    skips = {"no_t0": 0, "short_or_no_q": 0, "no_multimet": 0, "forecast_none": 0}
    per_station = {}
    t_start = time.time()
    for i, p in enumerate(files, 1):
        sid = p.name.split(".")[0]
        attrs = gages2.enrich_station_attrs(dict(registry.get(sid, {"id": sid})))
        try:
            m = eval_station(p, attrs, t0_list, skips)
            if m is not None:
                per_station[sid] = m
        except Exception as exc:
            print(f"[{i}/{len(files)}] {sid} ERR {exc}", flush=True)
        if i % 25 == 0:
            print(f"[{i}/{len(files)}] {len(per_station)} scored ({time.time()-t_start:.0f}s)",
                  flush=True)

    scalar = {s: {k: v for k, v in m.items() if isinstance(v, (int, float))}
              for s, m in per_station.items()}
    agg = aggregate(scalar)
    by_h = {}
    for h in range(1, HORIZON + 1):
        vals = [m["by_h_nse"][h] for m in per_station.values()
                if h in m.get("by_h_nse", {}) and np.isfinite(m["by_h_nse"][h])]
        by_h[h] = {"median_nse": float(np.median(vals)) if vals else float("nan"),
                   "n": len(vals)}

    payload = {
        "label": args.label,
        "forcing": "Caravan MultiMet HRES (Zenodo 14161281)",
        "ckpt": args.ckpt, "point_policy": args.point,
        "window": [args.start, args.end], "stride_days": args.stride,
        "horizon": HORIZON,
        "cohort": {"n_scored": len(per_station), "n_candidate": len(files)},
        "skips": skips,
        "metrics": agg,
        "by_horizon_median_nse": by_h,
        "reference": {"AIFL": {"nse": 0.518, "kge": 0.636},
                      "Google": {"nse": 0.624, "kge": 0.678}},
        "caveats": [
            "HRES leads cap at 10; decoder radiation channel is NaN (MultiMet "
            "has no shortwave analogue) and temp_2m is used for mean=max=min — "
            "a real handicap vs our own 5-var forcing, the price of THEIR archive.",
            "encoder history is the Daymet CAMELS corpus; only the forecast "
            "side is MultiMet HRES.",
        ],
    }
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"multimet_backtest_{args.label}.json"
    out.write_text(json.dumps(payload, indent=2))
    nse_med = agg.get("nse", {}).get("median", float("nan"))
    kge_med = agg.get("kge", {}).get("median", float("nan"))
    print(f"\n{args.label}: pooled median NSE {nse_med:.3f}, KGE {kge_med:.3f} "
          f"({len(per_station)} gauges) vs AIFL 0.518 / Google 0.624")
    print(f"wrote {out} ({time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
