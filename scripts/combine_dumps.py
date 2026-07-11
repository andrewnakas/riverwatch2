#!/usr/bin/env python
"""Grand-ensemble combiner for heterogeneous MB-LSTM / δHBV member dumps.

The backtest's --ckpt a:b:c ensembler requires all members to share ONE cfg
(same enc_vars / normalization), so it cannot combine models trained on
different corpora (per-forcing single-forcing LSTMs, or a δHBV member with a
different input recipe). This combiner does that combination OFFLINE from the
per-window dumps (--dump-windows), which are already in physical cfs space.

Each dump is the --dump-windows schema:
    station_id, t0, h, truth, ylo, ymed, yhi, ymean, persist

Combination (default): inner-join all dumps on (station_id, t0, h) and average
the point column across members — Vincentization in physical space, matching the
z-space quantile-averaging the seed ensembler uses internally (app/mblstm.py).
Point column is `ymed` (the served point; for CMAL dumps ymean carries the
distribution mean — use --point ymean for those). Optional --weights fits a
single LSTM-vs-δHBV mixing weight, but ONLY on a held-out slice (--weight-val-
end) so the test decade is never used to tune the combination.

Scoring reuses app.metrics exactly as backtest_mblstm.py does: per-station
pooled all_point_metrics → aggregate (median/mean/scorable), plus a day-1
(h==1) slice and tercile guard. Emits a JSON in the mblstm_backtest_*.json
shape so it drops into the comparison table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from app import metrics  # noqa: E402

KEY = ["station_id", "t0", "h"]


def load_dump(path: Path, point: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"station_id": str})
    miss = [c for c in KEY + ["truth", point] if c not in df.columns]
    if miss:
        raise SystemExit(f"{path}: missing columns {miss}")
    return df[KEY + ["truth", point]].rename(columns={point: "sim"})


def combine(dumps: list[pd.DataFrame], weights: list[float] | None) -> pd.DataFrame:
    """Inner-join members on KEY; weighted-average their sim columns. truth is
    taken from the first member (identical across members by construction)."""
    base = dumps[0].rename(columns={"sim": "sim_0"})
    for i, d in enumerate(dumps[1:], 1):
        base = base.merge(d.drop(columns="truth").rename(columns={"sim": f"sim_{i}"}),
                          on=KEY, how="inner")
    sim_cols = [c for c in base.columns if c.startswith("sim_")]
    w = np.asarray(weights if weights else [1.0] * len(sim_cols), dtype=float)
    w = w / w.sum()
    base["sim"] = (base[sim_cols].to_numpy() * w[None, :]).sum(axis=1)
    return base[KEY + ["truth", "sim"]]


def score(df: pd.DataFrame, day1_only: bool = False) -> dict:
    """Per-station pooled metrics → aggregate, matching backtest_mblstm.py."""
    d = df[df["h"] == 1] if day1_only else df
    per_station: dict[str, dict] = {}
    for sid, g in d.groupby("station_id", sort=False):
        obs = g["truth"].to_numpy(dtype=float)
        sim = g["sim"].to_numpy(dtype=float)
        m = np.isfinite(obs) & np.isfinite(sim)
        if m.sum() < 5:
            continue
        per_station[sid] = metrics.all_point_metrics(obs[m], sim[m])
    return metrics.aggregate(per_station)


def fit_weights(dumps, val_end: str):
    """Optimize per-member weights to maximize median NSE on the VAL slice ONLY
    (t0 <= val_end), so the test decade is never used to tune the combination.
    scipy differential-evolution (the paper's 'GA') on simplex weights."""
    from scipy.optimize import differential_evolution
    # merge members once; split by t0 date
    merged = dumps[0].rename(columns={"sim": "sim_0"})
    for i, d in enumerate(dumps[1:], 1):
        merged = merged.merge(d.drop(columns="truth").rename(columns={"sim": f"sim_{i}"}),
                              on=KEY, how="inner")
    sim_cols = [c for c in merged.columns if c.startswith("sim_")]
    val = merged[merged["t0"] <= val_end]
    if len(val) < 100:
        print(f"WARN: only {len(val)} val rows (t0<={val_end}) — weights may be noisy",
              file=sys.stderr)
    S = val[sim_cols].to_numpy(); y = val["truth"].to_numpy()
    fin = np.isfinite(y) & np.isfinite(S).all(axis=1)
    S, y = S[fin], y[fin]
    codes, _ = pd.factorize(val["station_id"].to_numpy()[fin])   # 0..G-1
    G = codes.max() + 1
    cnt = np.bincount(codes, minlength=G)
    # per-station obs variance (SST/n) is candidate-independent — precompute once
    ybar = np.bincount(codes, weights=y, minlength=G) / np.maximum(cnt, 1)
    sst = np.bincount(codes, weights=(y - ybar[codes]) ** 2, minlength=G)
    ok = (cnt >= 5) & (sst > 1e-9)

    def neg_med_nse(w):
        w = np.abs(w); w = w / (w.sum() + 1e-9)
        sim = S @ w
        sse = np.bincount(codes, weights=(y - sim) ** 2, minlength=G)
        nse = 1.0 - sse[ok] / sst[ok]
        return -(np.median(nse) if nse.size else -1.0)

    n = len(sim_cols)
    res = differential_evolution(neg_med_nse, [(0.0, 1.0)] * n, seed=0,
                                 maxiter=40, tol=1e-4, polish=True)
    w = np.abs(res.x); w = w / w.sum()
    print(f"fitted weights (val medNSE={-res.fun:.4f}): "
          + ", ".join(f"{c}={wi:.3f}" for c, wi in zip(sim_cols, w)))
    return list(w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="member dump csv.gz paths (grand-ensemble members)")
    ap.add_argument("--point", default="ymed",
                    help="point column to combine (ymed default; ymean for CMAL)")
    ap.add_argument("--weights", default="",
                    help="comma-separated per-member weights (default equal)")
    ap.add_argument("--fit-weights", action="store_true",
                    help="optimize member weights on the val slice (the beat-0.83 "
                         "lever); overrides --weights. Fit is t0<=--val-end only.")
    ap.add_argument("--val-end", default="1998-09-30",
                    help="fit-weights uses windows with t0 on/before this date")
    ap.add_argument("--label", default="grand_ensemble")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    paths = [Path(p) for p in args.dumps]
    for p in paths:
        if not p.exists():
            print(f"FATAL: missing dump {p}", file=sys.stderr)
            return 1
    weights = ([float(x) for x in args.weights.split(",")] if args.weights else None)
    if weights and len(weights) != len(paths):
        print(f"FATAL: {len(weights)} weights for {len(paths)} dumps", file=sys.stderr)
        return 1

    dumps = [load_dump(p, args.point) for p in paths]
    if args.fit_weights:
        weights = fit_weights(dumps, args.val_end)   # val-only fit (no test leakage)
    combined = combine(dumps, weights)
    n_windows = combined[["station_id", "t0"]].drop_duplicates().shape[0]
    full = score(combined, day1_only=False)
    day1 = score(combined, day1_only=True)

    def med(block, k):
        return round(block[k]["median"], 4) if k in block else None
    print(f"grand ensemble '{args.label}' ({len(paths)} members, {n_windows} windows):")
    print(f"  pooled : NSE {med(full,'nse')}  KGE {med(full,'kge')}  "
          f"log-NSE {med(full,'log_nse')}  FHV {med(full,'fhv')}  "
          f"(scorable {full.get('nse',{}).get('scorable')})")
    print(f"  day-1  : NSE {med(day1,'nse')}  KGE {med(day1,'kge')}")

    out_path = Path(args.out) if args.out else (
        REPO / "benchmarks" / f"combine_{args.label}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "label": args.label,
        "members": [str(p) for p in paths],
        "point": args.point,
        "weights": (weights or "equal"),
        "n_windows": int(n_windows),
        "metrics": {"full": full, "day1": day1},
    }, indent=2))
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
