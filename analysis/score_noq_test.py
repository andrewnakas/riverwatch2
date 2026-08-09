#!/usr/bin/env python3
"""THE PRE-REGISTERED NO-Q TEST QUERY — one-shot, user-approved 2026-08-09.

Scores the FROZEN 9-stream configuration on the Li/Song TEST window
(1995-10-01 -> 2010-09-30; dumps actually end 2008-12-21 because MAURER ENDS
2008, so the scored window is ~13.2 years, not 15).

FROZEN CONFIG (nothing here may be tuned after seeing the output):
  7 base streams : lstm_{daymet,nldas,maurer}, lstm_multi, dhbv_{daymet,nldas,maurer}
  + lstm_multi5  : 5-seed average (s111..s555)
  + lstm_multi6  : 3-seed average (s111..s333)
  weighting      : inverse-MSE (theta=4.0, lam=0.25) FIT ON TRAIN DUMPS ONLY

Weights are fit on the matching _TRAIN_ dumps. Fitting on the scored frame
would be leakage; phase_c.py's first patch attempted that and its own guard
correctly refused. That guard is reproduced here.

--mode equal   : CONTROL. Equal-weight 7-stream base only, must reproduce
                 the committed 0.8298 Li/Song figure. Run this FIRST.
--mode frozen  : THE QUERY. 9 streams, inverse-MSE weights.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L          # noqa: E402
from phase_c import load_seed_avg              # noqa: E402

THETA, LAM = 4.0, 0.25          # validated on the 3-way fit/select/score split
MULTI5_SEEDS = ("s111", "s222", "s333", "s444", "s555")
MULTI6_SEEDS = ("s111", "s222", "s333")


def per_basin_nse(df, col):
    """NaN-masked per-basin NSE (TRAIN dumps carry ~3.2% NaN truth)."""
    out = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = g[col].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(p)
        y, p = y[m], p[m]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        out[sid] = 1.0 - np.mean((y - p) ** 2) / np.var(y)
    return pd.Series(out)


def add_stream(merged, name, paths):
    """Inner-join a seed-averaged extra stream, asserting the join stays fat."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{name}: missing dumps {missing}")
    before_rows, before_basins = len(merged), merged.station_id.nunique()
    d = load_seed_avg([Path(p) for p in paths])[["station_id", "date", "pred"]]
    d = d.rename(columns={"pred": name})
    merged = merged.merge(d, on=["station_id", "date"], how="inner")
    print(f"  +{name:<14} {len(paths)} seeds   rows {before_rows:,} -> {len(merged):,}"
          f"   basins {before_basins} -> {merged.station_id.nunique()}")
    if len(merged) < 0.95 * before_rows:
        raise SystemExit(f"{name}: join lost >5% of rows — schema/date mismatch")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("equal", "frozen"), required=True)
    args = ap.parse_args()

    print("=" * 78)
    print(f"NO-Q SCORER — mode={args.mode}")
    print("=" * 78)

    # ---- TEST-side frame -------------------------------------------------
    merged, cols = build_merged(train=False)
    print(f"\nbase (7 streams) TEST: rows {len(merged):,}  "
          f"basins {merged.station_id.nunique()}")
    tt = pd.to_datetime(merged.date)
    print(f"scored window: {tt.min().date()} -> {tt.max().date()}")

    if args.mode == "equal":
        merged["ens"] = merged[cols].mean(axis=1)
        s = per_basin_nse(merged, "ens")
        print(f"\n{'='*78}\nCONTROL: 7-stream equal-weight TEST median NSE = "
              f"{s.median():.4f}  (basins={len(s)})")
        print("expected 0.8298 (Li/Song reproduction; paper 0.8294)")
        print("=" * 78)
        return

    # ---- FROZEN: add multi5 + multi6 on BOTH sides -----------------------
    print("\nadding frozen extra streams (TEST):")
    merged = add_stream(merged, "lstm_multi5",
                        [L / f"camels531ls_multi5_nhlstm_{s}.csv.gz" for s in MULTI5_SEEDS])
    merged = add_stream(merged, "lstm_multi6",
                        [L / f"camels531ls_multi6_nhlstm_{s}.csv.gz" for s in MULTI6_SEEDS])
    cols = cols + ["lstm_multi5", "lstm_multi6"]

    tr, tcols = build_merged(train=True)
    print("\nadding frozen extra streams (TRAIN, for weight fitting):")
    tr = add_stream(tr, "lstm_multi5",
                    [L / f"camels531ls_multi5_nhlstm_TRAIN_{s}.csv.gz" for s in MULTI5_SEEDS])
    tr = add_stream(tr, "lstm_multi6",
                    [L / f"camels531ls_multi6_nhlstm_TRAIN_{s}.csv.gz" for s in MULTI6_SEEDS])
    tcols = tcols + ["lstm_multi5", "lstm_multi6"]

    if tcols != cols:
        raise SystemExit(f"stream mismatch TRAIN {tcols} vs TEST {cols}")

    # ---- inverse-MSE weights, fit on TRAIN ONLY --------------------------
    truth_tr = tr["truth"].to_numpy(float)
    mse = np.array([np.nanmean((tr[c].to_numpy(float) - truth_tr) ** 2) for c in cols])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    eq = np.ones(len(cols)) / len(cols)
    w = LAM * eq + (1 - LAM) * raw
    w = w / w.sum()
    if w.std() < 1e-6:
        raise SystemExit("invmse: weights collapsed to uniform — check scale")

    print(f"\n--- inverse-MSE weights (fit on {len(tr):,} TRAIN rows, theta={THETA}, "
          f"lam={LAM}) ---")
    for c, x in zip(cols, w):
        print(f"    {c:<16} {x:.4f}")

    merged["ens"] = (merged[cols].to_numpy(float) * w).sum(axis=1)
    s = per_basin_nse(merged, "ens")

    # reference: equal-weight on the same 9-stream frame
    merged["_eq"] = merged[cols].mean(axis=1)
    s_eq = per_basin_nse(merged, "_eq")

    print("\n" + "=" * 78)
    print("*** NO-Q HELD-OUT TEST RESULT (pre-registered, one-shot) ***")
    print("=" * 78)
    print(f"  streams            : {len(cols)}  {cols}")
    print(f"  basins             : {len(s)}")
    print(f"  rows               : {len(merged):,}")
    print(f"  window             : {tt.min().date()} -> {tt.max().date()}")
    print(f"  9-stream equal-wt  : {s_eq.median():.4f}")
    print(f"  9-stream inv-MSE   : {s.median():.4f}   <-- HEADLINE")
    print(f"  train-side figure  : 0.8347 (for reference; NOT comparable directly)")
    print("=" * 78)

    # per-stream on the common sample, for the paper's table
    print("\n--- each stream alone on the scored sample ---")
    for c in cols:
        print(f"  {c:<16} {per_basin_nse(merged, c).median():.4f}")

    out = Path("noq_test_result.json")
    out.write_text(json.dumps({
        "mode": "frozen_9stream_invmse",
        "median_nse": float(s.median()),
        "median_nse_equal_weight": float(s_eq.median()),
        "basins": int(len(s)),
        "rows": int(len(merged)),
        "window": [str(tt.min().date()), str(tt.max().date())],
        "streams": cols,
        "weights": {c: float(x) for c, x in zip(cols, w)},
        "theta": THETA, "lam": LAM,
        "per_basin_nse": {k: float(v) for k, v in s.items()},
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
