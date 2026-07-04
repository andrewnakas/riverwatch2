#!/usr/bin/env python3
"""Offline anchor-decay x point-policy sweep over a backtest window dump.

Consumes the csv.gz written by backtest_mblstm.py --dump-windows (unanchored
per-window lo/med/hi/mean quantiles + truth + persistence) and evaluates every
(decay_h, point_policy) combination WITHOUT re-running model inference — the
anchoring correction and point-policy composition are pure post-processing.

One stride-3 inference pass (~30 min) + this script (~1 min) replaces a dozen
hour-long harness runs.

Usage:
  .venv/bin/python scripts/sweep_anchor_point.py \
      --dump data/mblstm/dumps/ens4ft_hybrid_str3.csv.gz --label str3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import metrics  # noqa: E402

DECAYS = [0.0, 1.0, 2.0, 3.0, 4.0, 7.0]
REPORT_H = (1, 2, 5, 14)  # horizons surfaced in the MAE-ratio columns


def policies(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {
        "median": df["ymed"].to_numpy(float),
        "mean3": ((df["ylo"] + df["ymed"] + df["yhi"]) / 3.0).to_numpy(float),
    }
    for w in (0.1, 0.2, 0.3):
        out[f"blend{w}"] = ((1 - w) * df["ymed"] + w * df["yhi"]).to_numpy(float)
    if df["ymean"].notna().any():
        out["mean"] = df["ymean"].to_numpy(float)
    return out


def eval_combo(df: pd.DataFrame, pt: np.ndarray, decay: float) -> dict:
    """Median-across-stations metrics for one (policy, decay) combo."""
    d = df.copy()
    d["pt"] = pt
    # Anchor offset per window: (persist - point[h=1]) decayed across leads.
    h1 = d.loc[d["h"] == 1, ["station_id", "t0", "pt"]].rename(columns={"pt": "pt_h1"})
    d = d.merge(h1, on=["station_id", "t0"], how="left")
    if decay > 0:
        w = np.clip(1.0 - (d["h"].to_numpy(float) - 1.0) / decay, 0.0, 1.0)
    else:
        w = 0.0
    adj = (d["persist"].to_numpy(float) - d["pt_h1"].to_numpy(float)) * w
    for col in ("pt", "ylo", "ymed", "yhi"):
        d["a_" + col] = np.clip(d[col].to_numpy(float) + adj, 0.0, None)

    per_station: dict[str, dict] = {}
    ratios: dict[int, list] = {h: [] for h in REPORT_H}
    for sid, g in d.groupby("station_id", sort=False):
        y = g["truth"].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 30:
            continue
        y, yhat = y[ok], g["a_pt"].to_numpy(float)[ok]
        lo, med, hi = (g["a_" + c].to_numpy(float)[ok] for c in ("ylo", "ymed", "yhi"))
        m = metrics.all_point_metrics(y, yhat)
        m["approx_crps"] = metrics.crps_from_quantiles(
            y, [0.1, 0.5, 0.9], np.vstack([lo, med, hi]))
        m["picp90"] = float(np.mean((y >= lo) & (y <= hi)))
        per_station[sid] = m
        hh = g["h"].to_numpy(int)[ok]
        pers = g["persist"].to_numpy(float)[ok]
        for h in REPORT_H:
            hm = hh == h
            if hm.sum() >= 10:
                mae_p = float(np.mean(np.abs(y[hm] - pers[hm])))
                mae_m = float(np.mean(np.abs(y[hm] - yhat[hm])))
                if mae_p > 0:
                    ratios[h].append(mae_m / mae_p)

    keys = ("nse", "kge", "log_nse", "fhv", "pct_bias", "approx_crps", "picp90")
    out = {}
    for k in keys:
        vals = np.asarray([v[k] for v in per_station.values()], float)
        vals = vals[np.isfinite(vals)]
        out[k] = float(np.median(vals)) if len(vals) else float("nan")
    for h in REPORT_H:
        out[f"ratio_h{h}"] = float(np.median(ratios[h])) if ratios[h] else float("nan")
    out["n_stations"] = len(per_station)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--label", default="sweep")
    ap.add_argument("--decays", default=",".join(str(d) for d in DECAYS))
    args = ap.parse_args()

    df = pd.read_csv(args.dump, dtype={"station_id": str})
    print(f"{len(df)} rows, {df['station_id'].nunique()} stations, "
          f"{df.groupby('station_id')['t0'].nunique().median():.0f} windows/station")

    decays = [float(x) for x in args.decays.split(",")]
    rows = []
    pol = policies(df)
    for pname, pt in pol.items():
        for decay in decays:
            r = eval_combo(df, pt, decay)
            r.update({"policy": pname, "decay_h": decay})
            rows.append(r)
            print(f"{pname:>9} decay={decay:>3.0f} | NSE {r['nse']:+.3f} "
                  f"KGE {r['kge']:+.3f} logNSE {r['log_nse']:+.3f} "
                  f"FHV {r['fhv']:+.1f} CRPS {r['approx_crps']:.1f} "
                  f"PICP {r['picp90']:.2f} | ratio h1 {r['ratio_h1']:.3f} "
                  f"h2 {r['ratio_h2']:.3f} h5 {r['ratio_h5']:.3f} "
                  f"h14 {r['ratio_h14']:.3f}", flush=True)

    best = max(rows, key=lambda r: r["nse"])
    print(f"\nbest by NSE: policy={best['policy']} decay={best['decay_h']} "
          f"NSE {best['nse']:+.3f} FHV {best['fhv']:+.1f}")
    out = ROOT / "benchmarks" / f"sweep_anchor_point_{args.label}.json"
    out.write_text(json.dumps({"dump": args.dump, "combos": rows}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
