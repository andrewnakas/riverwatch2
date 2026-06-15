#!/usr/bin/env python3
"""Honest temporal backtest for the v16 MB-LSTM member.

Evaluates the trained checkpoint through the *serving* code path
(app.mblstm.forecast) on issue dates the model never saw in training
(default: calendar 2025 = the validation year; 2026 windows can be used once
corpus rows exist for them). For every (station, issue-date) window:

    q_hist / wx_hist : corpus rows  <= t0   (what serving would fetch)
    wx_fcst          : corpus rows in (t0, t0+H]  — observed weather as the
                       "forecast" (perfect-forcing; generous at h>3, same
                       caveat as the trainer and stated in the output)
    truth            : observed q at t0+1 .. t0+H

Reports per-horizon median-station MAE for persistence vs the MB-LSTM median
quantile, the MAE ratio, and per-station pooled NSE — same shape as
benchmarks/nwm_backtest_v4.json so the members can be compared side by side.

Usage:
  RW2_ENABLE_MBLSTM=1 .venv/bin/python scripts/backtest_mblstm.py \
      --ckpt data/mblstm/model.pt --label pilot
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
from app.weather import DAILY_VARS  # noqa: E402

CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus"
STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
CAMELS_PATH = ROOT / "data" / "camels_gauge_ids.json"
SCHEMA_VERSION = 2  # v2: full SOTA metric suite + CAMELS subset + approx-CRPS
GFS_DIR = ROOT / "data" / "mblstm" / "gfs_fcst"
HRRR_DIR = ROOT / "data" / "mblstm" / "hrrr_fcst"
OUT_DIR = ROOT / "benchmarks"
HORIZON = 14

GFS_VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum"]


def load_gfs(start: str, end: str, src_dir: Path = GFS_DIR) -> dict[pd.Timestamp, pd.DataFrame]:
    """init_date -> per-station forecast frame. A 00z init on day D has
    lead_day 1 = calendar day D, so the matching issue date is t0 = D-1
    (observations through yesterday, today's 00z run — the serving setup)."""
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for p in sorted(src_dir.glob("*.csv.gz")):
        init = pd.Timestamp(p.name.split(".")[0])
        # t0 = init-1d must fall inside the eval window.
        if not (pd.Timestamp(start) <= init - pd.Timedelta(days=1) <= pd.Timestamp(end)):
            continue
        df = pd.read_csv(p, dtype={"station_id": str})
        out[init] = df.set_index("station_id")
    return out


def gfs_wx_fcst(gfs: dict, sid: str, t0: pd.Timestamp,
                hrrr: dict | None = None) -> pd.DataFrame | None:
    """Decoder forcing for issue date t0 from the archived GFS init at t0+1.
    Returns None unless all 14 lead days are present for this station. With
    hrrr, lead days 1-2 are overlaid with the 3 km HRRR forecast where present."""
    df = gfs.get(t0 + pd.Timedelta(days=1))
    if df is None:
        return None
    try:
        rows = df.loc[[sid]] if sid in df.index else None
    except (KeyError, TypeError):
        rows = None
    if rows is None or len(rows) < HORIZON:
        return None
    rows = rows.sort_values("lead_day").iloc[:HORIZON]
    out = rows[GFS_VARS].reset_index(drop=True)
    if hrrr is not None:
        hdf = hrrr.get(t0 + pd.Timedelta(days=1))
        if hdf is not None and sid in hdf.index:
            h = hdf.loc[[sid]].sort_values("lead_day")
            for _, hr in h.iterrows():
                ld = int(hr["lead_day"])
                if 1 <= ld <= 2:
                    out.loc[ld - 1, GFS_VARS] = hr[GFS_VARS].to_numpy()
    out.insert(0, "date", [t0 + pd.Timedelta(days=int(d)) for d in rows["lead_day"]])
    return out


def load_camels_ids(which: str) -> set[str]:
    """CAMELS-US gauge ids for the requested subset (671 = full, 531 = the
    'well-behaved' subset that published median-NSE ~0.76 is usually quoted on).
    Returns an empty set if the data file is absent — the harness still runs;
    the CAMELS metric block is simply omitted."""
    if which == "none" or not CAMELS_PATH.exists():
        return set()
    data = json.loads(CAMELS_PATH.read_text())
    ids = set(data["671"]) if which == "671" else set(data.get("531", data["671"]))
    # Normalize to the corpus's zero-padded 8-digit form.
    return {str(s).strip().zfill(8) for s in ids}


def _metric_block(results: dict, sids: list[str]) -> dict:
    """Median/mean across the given stations for every per-station metric."""
    sub = {sid: results[sid] for sid in sids if sid in results}
    if not sub:
        return {}
    keys = ("nse", "kge", "log_nse", "pearson_r", "pct_bias", "fhv", "flv",
            "approx_crps", "picp90", "mpiw_norm")
    per_station = {sid: {k: r[k] for k in keys if k in r} for sid, r in sub.items()}
    agg = metrics.aggregate(per_station)
    agg["n_stations"] = len(sub)
    return agg


def eval_station(path: Path, attrs: dict, issue_dates: pd.DatetimeIndex,
                 gfs: dict | None = None, hrrr: dict | None = None) -> dict | None:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    by_date = df.set_index("date")
    daily = by_date.reindex(pd.date_range(df["date"].iloc[0], df["date"].iloc[-1], freq="D"))

    per_h: dict[int, dict[str, list]] = {h: {"persist": [], "mblstm": []} for h in range(1, HORIZON + 1)}
    pooled_y, pooled_yhat = [], []
    pooled_lo, pooled_hi = [], []  # model's own 0.1/0.9 quantiles for approx-CRPS
    n_windows = 0
    for t0 in issue_dates:
        if t0 not in daily.index:
            continue
        hist = daily.loc[:t0]
        fut = daily.loc[t0 + pd.Timedelta(days=1): t0 + pd.Timedelta(days=HORIZON)]
        if len(fut) < HORIZON or len(hist) < 400:
            continue
        q0 = hist["q_cfs"].dropna()
        if q0.empty or pd.isna(hist["q_cfs"].iloc[-1]):
            continue
        truth = fut["q_cfs"].to_numpy(dtype=float)
        if np.isfinite(truth).sum() < 7:
            continue

        q_hist = hist["q_cfs"].dropna().rename("q_cfs").reset_index()
        q_hist.columns = ["date", "q_cfs"]
        wx_hist = hist.reset_index().rename(columns={"index": "date"})[["date"] + DAILY_VARS]
        if gfs is not None:
            wx_fcst = gfs_wx_fcst(gfs, path.name.split(".")[0], t0, hrrr=hrrr)
            if wx_fcst is None:
                continue
        else:
            wx_fcst = fut.reset_index().rename(columns={"index": "date"})[["date"] + DAILY_VARS]

        rows = mblstm.forecast(q_hist, wx_hist, wx_fcst, attrs, HORIZON)
        if not rows or len(rows) < HORIZON:
            continue
        yhat = np.asarray([r["q_cfs"] for r in rows], dtype=float)
        ylo = np.asarray([r.get("q_lo", np.nan) for r in rows], dtype=float)
        yhi = np.asarray([r.get("q_hi", np.nan) for r in rows], dtype=float)
        persist = float(q_hist["q_cfs"].iloc[-1])

        n_windows += 1
        for h in range(1, HORIZON + 1):
            yt = truth[h - 1]
            if not np.isfinite(yt):
                continue
            per_h[h]["persist"].append(abs(yt - persist))
            per_h[h]["mblstm"].append(abs(yt - yhat[h - 1]))
            pooled_y.append(yt)
            pooled_yhat.append(yhat[h - 1])
            pooled_lo.append(ylo[h - 1])
            pooled_hi.append(yhi[h - 1])

    if n_windows < 5:
        return None
    pooled_y = np.asarray(pooled_y); pooled_yhat = np.asarray(pooled_yhat)
    pooled_lo = np.asarray(pooled_lo); pooled_hi = np.asarray(pooled_hi)
    # Full SOTA metric suite via the shared app.metrics module. metrics.nse
    # reproduces the historical var<1e-3 → NaN guard, so the headline NSE is
    # byte-stable with pre-metrics-module backtest JSONs.
    m = metrics.all_point_metrics(pooled_y, pooled_yhat)
    # approx-CRPS from the model's own quantiles (mean pinball over 0.1/0.5/0.9).
    crps = metrics.crps_from_quantiles(
        pooled_y, [0.1, 0.5, 0.9], np.vstack([pooled_lo, pooled_yhat, pooled_hi]))
    # 90% interval coverage (PICP) and mean width (MPIW, normalized by mean obs).
    cov = float(np.mean((pooled_y >= pooled_lo) & (pooled_y <= pooled_hi))) \
        if np.isfinite(pooled_lo).any() else float("nan")
    mpiw = float(np.mean(pooled_hi - pooled_lo) / max(np.mean(pooled_y), 1e-9)) \
        if np.isfinite(pooled_lo).any() else float("nan")
    # Tercile-stratified NSE/KGE/log-NSE to rule out big-river-only wins.
    tm = metrics.tercile_masks(pooled_y)
    by_tercile = {
        band: {
            "nse": metrics.nse(pooled_y[mask], pooled_yhat[mask]),
            "kge": metrics.kge(pooled_y[mask], pooled_yhat[mask]),
            "log_nse": metrics.log_nse(pooled_y[mask], pooled_yhat[mask]),
        }
        for band, mask in tm.items() if mask.any()
    }
    out = {
        "windows": n_windows,
        "nse": m["nse"],  # primary headline metric, unchanged definition
        "approx_crps": crps, "picp90": cov, "mpiw_norm": mpiw,
        "by_tercile": by_tercile,
        "mae_by_h": {h: {k: float(np.mean(v)) for k, v in d.items() if v} for h, d in per_h.items()},
    }
    out.update({k: v for k, v in m.items() if k != "nse"})  # kge, log_nse, r, pbias, fhv, flv
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "mblstm" / "model.pt"))
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--limit-stations", type=int, default=0)
    ap.add_argument("--stride-stations", type=int, default=1,
                    help="evaluate every Nth corpus station (fast A/B subsample)")
    ap.add_argument("--label", default="pilot")
    ap.add_argument("--gfs", action="store_true",
                    help="decoder forcing from archived GFS forecasts instead of "
                         "observed future weather (issue dates snap to GFS inits)")
    ap.add_argument("--hrrr", action="store_true",
                    help="with --gfs: overlay 3km HRRR on decoder lead days 1-2")
    ap.add_argument("--camels-subset", choices=["none", "671", "531"], default="none",
                    help="also report metrics on the CAMELS-US basin subset for "
                         "apples-to-apples comparison to published median-NSE")
    args = ap.parse_args()

    os.environ["RW2_MBLSTM_CKPT_PATH"] = args.ckpt
    gfs = hrrr = None
    if args.hrrr and not args.gfs:
        print("--hrrr requires --gfs")
        return 1
    if args.gfs:
        gfs = load_gfs(args.start, args.end)
        if not gfs:
            print(f"no GFS init files in window — run scripts/fetch_gfs_forcings.py first")
            return 1
        issue_dates = pd.DatetimeIndex(sorted(i - pd.Timedelta(days=1) for i in gfs))
        print(f"GFS mode: {len(issue_dates)} issue dates "
              f"{issue_dates[0].date()}..{issue_dates[-1].date()}")
        if args.hrrr:
            hrrr = load_gfs(args.start, args.end, src_dir=HRRR_DIR)
            print(f"HRRR overlay: {len(hrrr)} matching inits")
    else:
        issue_dates = pd.date_range(args.start, args.end, freq=f"{args.stride}D")

    registry = {s["id"]: s for s in json.loads(STATIONS_PATH.read_text())["stations"]}
    files = sorted(CORPUS_DIR.glob("*.csv.gz"))
    if args.stride_stations > 1:
        files = files[:: args.stride_stations]  # deterministic subsample for fast A/B
    if args.limit_stations:
        files = files[: args.limit_stations]

    results: dict[str, dict] = {}
    t0 = time.time()
    for i, p in enumerate(files, 1):
        sid = p.name.split(".")[0]
        attrs = gages2.enrich_station_attrs(dict(registry.get(sid, {"id": sid})))
        try:
            r = eval_station(p, attrs, issue_dates, gfs=gfs, hrrr=hrrr)
        except Exception as exc:
            print(f"[{i}/{len(files)}] {sid} ERR {exc}", flush=True)
            continue
        if r is None:
            continue
        results[sid] = r
        if i % 50 == 0:
            print(f"[{i}/{len(files)}] {len(results)} stations evaluated "
                  f"({time.time() - t0:.0f}s)", flush=True)

    if not results:
        print("no stations evaluated — is the checkpoint present and corpus built?")
        return 1

    # Median-across-stations MAE per horizon + ratio, matching the NWM
    # backtest's presentation.
    print(f"\nstations={len(results)}  windows/station~="
          f"{np.median([r['windows'] for r in results.values()]):.0f}  "
          f"window: {args.start}..{args.end} stride {args.stride}d")
    print(f"{'h':>3} {'persist':>9} {'mblstm':>9} {'ratio':>7}")
    summary_h = {}
    for h in range(1, HORIZON + 1):
        pm = [r["mae_by_h"][h]["persist"] for r in results.values() if h in r["mae_by_h"] and "persist" in r["mae_by_h"][h]]
        mm = [r["mae_by_h"][h]["mblstm"] for r in results.values() if h in r["mae_by_h"] and "mblstm" in r["mae_by_h"][h]]
        med_p, med_m = float(np.median(pm)), float(np.median(mm))
        summary_h[h] = {"persistence": med_p, "mblstm": med_m,
                        "ratio": med_m / med_p if med_p > 0 else None}
        print(f"{h:>3} {med_p:>9.1f} {med_m:>9.1f} {med_m / med_p:>7.3f}")
    nses = np.asarray([r["nse"] for r in results.values()], dtype=float)
    scorable = nses[np.isfinite(nses)]  # drop the not-NSE-scorable flat-flow gauges
    print(f"\npooled-horizon NSE (cfs): median={np.nanmedian(scorable):.3f}  "
          f"mean={np.nanmean(scorable):.3f}  frac>0.5={np.mean(scorable > 0.5):.2f}  "
          f"(scorable {len(scorable)}/{len(nses)})")

    # Full SOTA metric suite, full corpus and (optionally) CAMELS subset.
    full_block = _metric_block(results, list(results.keys()))
    metric_blocks = {"full": full_block}
    print("\nSOTA metrics (median across stations):")
    for k in ("nse", "kge", "log_nse", "pearson_r", "pct_bias", "fhv", "flv",
              "approx_crps", "picp90", "mpiw_norm"):
        if k in full_block:
            print(f"  {k:>11}: {full_block[k]['median']:+.3f}  (n={full_block[k]['scorable']})")

    camels_ids = load_camels_ids(args.camels_subset)
    if args.camels_subset != "none":
        if not camels_ids:
            print(f"\nCAMELS-{args.camels_subset}: gauge-id file {CAMELS_PATH.name} "
                  f"absent — skipping subset block (run still valid for full corpus)")
        else:
            inter = sorted(set(results) & camels_ids)
            print(f"\nCAMELS-{args.camels_subset} subset: {len(inter)} of "
                  f"{len(results)} evaluated stations are CAMELS basins")
            blk = _metric_block(results, inter)
            metric_blocks[f"camels_{args.camels_subset}"] = blk
            if blk:
                print(f"  CAMELS median NSE={blk.get('nse',{}).get('median',float('nan')):.3f}  "
                      f"KGE={blk.get('kge',{}).get('median',float('nan')):.3f}  "
                      f"(published ensembled-LSTM ref ~0.76 median NSE)")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"mblstm_backtest_{args.label}.json"
    out.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "label": args.label, "ckpt": args.ckpt,
        "window": [args.start, args.end], "stride_days": args.stride,
        "caveat": ("decoder forcing = archived HRRR d1-2 + GFS d3-14 hybrid via "
                   "dynamical.org (real forecast error)" if args.hrrr else
                   "decoder forcing = archived GFS forecasts via dynamical.org "
                   "(real forecast error)" if args.gfs else
                   "decoder forcing = observed archive weather (perfect forcing); generous at h>3"),
        "crps_caveat": "approx_crps = mean pinball over {0.1,0.5,0.9}, NOT integrated CRPS "
                       "(valid for internal A/B only; resolved when a CMAL head ships).",
        "stations": len(results),
        "median_mae_by_h": summary_h,
        "nse_median": float(np.nanmedian(scorable)), "nse_mean": float(np.nanmean(scorable)),
        "nse_scorable_stations": int(len(scorable)),
        "metrics": metric_blocks,
        "by_tercile_median": _tercile_summary(results),
        "per_station": {
            sid: {k: r.get(k) for k in
                  ("nse", "kge", "log_nse", "pct_bias", "fhv", "flv",
                   "approx_crps", "picp90", "windows")}
            for sid, r in results.items()
        },
    }, indent=2))
    print(f"wrote {out}")
    return 0


def _tercile_summary(results: dict) -> dict:
    """Median across stations of the per-station tercile NSE/KGE/log-NSE."""
    bands = ("low", "mid", "high")
    out: dict = {}
    for band in bands:
        for metric in ("nse", "kge", "log_nse"):
            vals = [r["by_tercile"][band][metric]
                    for r in results.values()
                    if "by_tercile" in r and band in r["by_tercile"]
                    and np.isfinite(r["by_tercile"][band].get(metric, np.nan))]
            out.setdefault(band, {})[metric] = float(np.median(vals)) if vals else float("nan")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
