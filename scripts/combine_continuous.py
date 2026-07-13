#!/usr/bin/env python
"""Combine per-member CONTINUOUS daily series (from eval_continuous --dump-series)
into a continuous grand-ensemble median NSE — the publication-exact number
directly comparable to the CAMELS-531 record (Li/Shen 2025 ~0.83).

Each input is a (station_id, date, truth, sim) csv.gz of ONE prediction per
calendar day (non-overlapping-window continuous sim). We inner-join members on
(station_id, date), average their sim (equal weights, or --weights), then compute
one NSE per basin over the pooled daily series and take the median across basins —
exactly the record's protocol (no forecast horizon, no window-stacking).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app import metrics  # noqa: E402

KEY = ["station_id", "date"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", nargs="+", required=True,
                    help="member continuous-series csv.gz (from --dump-series)")
    ap.add_argument("--weights", default="", help="comma-sep per-member weights")
    ap.add_argument("--min-days", type=int, default=180)
    ap.add_argument("--label", default="continuous_grand")
    args = ap.parse_args()

    dfs = []
    for i, p in enumerate(args.series):
        d = pd.read_csv(p, dtype={"station_id": str})
        dfs.append(d.rename(columns={"sim": f"sim{i}"}))
    base = dfs[0]
    for i, d in enumerate(dfs[1:], 1):
        base = base.merge(d.drop(columns="truth"), on=KEY, how="inner")
    sim_cols = [c for c in base.columns if c.startswith("sim")]
    w = (np.array([float(x) for x in args.weights.split(",")]) if args.weights
         else np.ones(len(sim_cols)))
    w = w / w.sum()
    base["sim"] = (base[sim_cols].to_numpy() * w[None, :]).sum(axis=1)

    per_station = {}
    for sid, g in base.groupby("station_id", sort=False):
        o = g["truth"].to_numpy(float); s = g["sim"].to_numpy(float)
        m = np.isfinite(o) & np.isfinite(s)
        if m.sum() < args.min_days or np.var(o[m]) < 1e-9:
            continue
        per_station[sid] = metrics.all_point_metrics(o[m], s[m])
    agg = metrics.aggregate(per_station)
    nse_med = agg["nse"]["median"]
    nses = sorted(m["nse"] for m in per_station.values() if np.isfinite(m["nse"]))
    n = len(nses)
    p = lambda q: nses[int(q * (n - 1))] if n else float("nan")
    print(f"CONTINUOUS grand-ensemble '{args.label}' "
          f"({len(sim_cols)} members, {n} basins):")
    print(f"  median NSE {nse_med:.4f}  KGE {agg.get('kge',{}).get('median'):.4f}  "
          f"log-NSE {agg.get('log_nse',{}).get('median'):.4f}  "
          f"FHV {agg.get('fhv',{}).get('median'):.2f}")
    print(f"  distribution: p25 {p(.25):.3f}  p50 {p(.50):.3f}  p75 {p(.75):.3f}  "
          f"frac>0.8 {np.mean([x>0.8 for x in nses]):.2f}  "
          f"frac<0 {np.mean([x<0 for x in nses]):.2f}")
    print(f"  (record ref: Li/Shen 2025 median NSE 0.83)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
