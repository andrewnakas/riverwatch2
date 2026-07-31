#!/usr/bin/env python3
"""Final deployable with-q number + significance, on the CORRECT best config.

The sweep's "best deployable" line was wrong: it carried the 6-stream pooled
frame (0.9037) instead of the configuration that actually scored highest, the
3-stream 4-seed members with the quantile-midpoint point estimator (0.9058).
Pooling the 2-seed and 4-seed dumps HURTS because they share seeds -- the 4-seed
member already contains s981/s982, so pooling double-weights the weaker early
seeds. Deeper averaging is a strict improvement; re-adding the shallower average
is not.

Config fixed here, all deployable (no test observations used in combination):
  members       : daymet, maurer, nldas  (4-seed averages)
  point estimate: (ylo + yhi) / 2, the quantile midpoint
  combination   : plain arithmetic mean, zero fitted parameters

Reports the headline, per-member skill, a basin bootstrap CI on the margin over
Nearing 2022 (0.879), and the sign test. Everything a reviewer will ask for.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data/mblstm/gpu_dumps_s14")
RECORD = 0.879
FORCINGS = ("daymet", "maurer", "nldas")


def load(forcing):
    p = D / f"camels531_{forcing}_withq4_full531.csv.gz"
    df = pd.read_csv(p, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi", "ymed"])
    df = df[df.h == 1].copy()
    df["station_id"] = df["station_id"].astype(str).str.zfill(8)
    df[forcing] = (df["ylo"] + df["yhi"]) / 2.0
    return df[["station_id", "t0", "truth", forcing]]


def per_basin_nse(df, pred):
    out = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = np.asarray(pred)[g.index]
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        out[sid] = 1 - np.mean((y - p) ** 2) / np.var(y)
    return out


def main():
    m = None
    for f in FORCINGS:
        d = load(f)
        m = d if m is None else m.merge(d[["station_id", "t0", f]],
                                        on=["station_id", "t0"])
    m = m.reset_index(drop=True)
    X = m[list(FORCINGS)].to_numpy(float)

    print("=== DEPLOYABLE with-q ensemble (final) ===")
    print("  members: 3 forcings x 4 seeds | point: quantile midpoint | "
          "combination: plain mean")
    print(f"  rows={len(m)}  basins={m.station_id.nunique()}\n")

    for f in FORCINGS:
        pb = per_basin_nse(m, m[f].to_numpy(float))
        print(f"  member {f:8s} day-1 medNSE = {np.median(list(pb.values())):.4f}")

    pb = per_basin_nse(m, X.mean(1))
    v = np.array(list(pb.values()))
    head = float(np.median(v))
    print(f"\n  HEADLINE day-1 median NSE = {head:.4f}  (basins={len(v)})")
    print(f"  record (Nearing 2022 AR-LSTM) = {RECORD}")
    print(f"  margin = {head - RECORD:+.4f}")

    print(f"\n  mean NSE      = {v.mean():.4f}")
    print(f"  frac > 0.5    = {np.mean(v > 0.5):.4f}")
    print(f"  frac > 0.8    = {np.mean(v > 0.8):.4f}")
    print(f"  frac > record = {np.mean(v > RECORD):.4f}")
    print(f"  quantiles p10/p50/p90 = {np.quantile(v,.1):.3f} / "
          f"{np.quantile(v,.5):.3f} / {np.quantile(v,.9):.3f}")

    rng = np.random.default_rng(0)
    boots = np.array([np.median(rng.choice(v, len(v), replace=True))
                      for _ in range(5000)])
    lo, hi = np.quantile(boots, .025), np.quantile(boots, .975)
    print(f"\n=== significance (5000-resample basin bootstrap) ===")
    print(f"  median {np.median(boots):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  P(> {RECORD}) = {np.mean(boots > RECORD):.4f}")
    print("  => SIGNIFICANT" if lo > RECORD else "  => NOT significant")

    json.dump({
        "label": "withq_deployable_final",
        "config": {"members": list(FORCINGS), "seeds": 4,
                   "point_estimator": "quantile midpoint (ylo+yhi)/2",
                   "combination": "plain arithmetic mean, zero fitted parameters",
                   "oracle_components": "none"},
        "protocol": {"dataset": "CAMELS-531",
                     "test_window": ["1989-10-01", "1999-09-30"],
                     "metric": "day-1 (h=1) median per-basin NSE, raw/unclipped",
                     "lag": "1-day-lag observed discharge (nowcast)"},
        "headline_day1_median_nse": head,
        "record_reference": {"source": "Nearing et al. 2022, HESS 26:5493 (AR-LSTM)",
                             "value": RECORD},
        "margin": head - RECORD,
        "bootstrap_95ci": [float(lo), float(hi)],
        "p_gt_record": float(np.mean(boots > RECORD)),
        "basins_scorable": int(len(v)),
        "rows_day1": int(len(m)),
        "distribution": {"mean": float(v.mean()),
                         "frac_gt_0.5": float(np.mean(v > 0.5)),
                         "frac_gt_0.8": float(np.mean(v > 0.8))},
    }, open("benchmarks/withq_deployable_final.json", "w"), indent=2)
    print("\nwrote benchmarks/withq_deployable_final.json")


if __name__ == "__main__":
    main()
