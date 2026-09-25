#!/usr/bin/env python3
"""LEDGER 42 — WHAT DOES THE MEDIAN-NSE GATE DISCARD? Zero GPU, train-side only.

THE QUESTION
------------
Every gate in this campaign is **median NSE**. The aim problem measured that
members help below-median basins 3-7x more than the ~78 that set a median, so
only **5-12%** of their summed gain reaches the reported statistic — and Track C
then showed the high-leverage cohort is **unaimable**, so that discard is
structural, not a targeting mistake.

But "the median discards it" is a property of **the median**, not of the models.
A MEAN is not a rank statistic; every basin's improvement counts in proportion.
So the same member can be worth ~0 to the gate and materially more to a
different aggregate — and nobody here has ever looked.

This measures, for the members that SHIPPED and for one that did not:

    delta under  median NSE   (the gate)      vs
                 MEAN   NSE   (not a rank statistic)
                 median KGE / mean KGE        (the field's other standard)
                 FHV                          (high-flow bias -- the flood-relevant one)

⛔ THIS IS NOT A ROUTE TO 0.84 AND MUST NEVER BE PRESENTED AS ONE.
The Li/Song benchmark IS median NSE; changing the reported statistic to make a
number look better is exactly the closed "reframing is not a gain" failure
([[allleads-weights-are-not-a-gain]], [[phase-robust-weighting-CLOSED]]).
This is here to answer "what is our work actually worth, and to whom" -- not to
move the record.

Frame: TRAIN val slice 1990-10-01..1995-09-30, frozen production weighting,
leave-one-out on the member under test. No test-window read.

Run on the 1080:  gpu1080/.venv/bin/python analysis/metric_blindspot.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))

from gate_eval import build_merged, L          # noqa: E402
from phase_c import load_seed_avg              # noqa: E402

THETA, LAM = 4.0, 0.25
CV_FIT_END, CV_VAL_START, CV_VAL_END = "1990-09-30", "1990-10-01", "1995-09-30"
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")
# High-flow bias segment. ⚠️ The standard FHV uses the top 2% of the FDC, and
# that returned nan for EVERY basin here: the h==1 val slice is stride-14, so a
# basin has only ~130 rows and the top 2% is ~3 points — below the guard. The
# campaign's own peak analyses use the TOP DECILE ("96.5% of error on top-10%
# flow days"), which gives ~13 points and is consistent with the rest of the
# corpus. Reported as top-decile high-flow bias, NOT as standard FHV(2%).
HIGH_Q = 0.90
MIN_HIGH = 8


def nse(y, p):
    return 1.0 - np.mean((y - p) ** 2) / np.var(y)


def kge(y, p):
    if y.std() < 1e-9 or p.std() < 1e-9 or abs(y.mean()) < 1e-12:
        return np.nan
    r = np.corrcoef(y, p)[0, 1]
    a = p.std() / y.std()
    b = p.mean() / y.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def fhv(y, p):
    """% bias on the TOP DECILE of flows (see HIGH_Q). Closer to 0 is better."""
    t = np.quantile(y, HIGH_Q)
    m = y >= t
    if m.sum() < MIN_HIGH or y[m].sum() <= 0:
        return np.nan
    return 100.0 * (p[m].sum() - y[m].sum()) / y[m].sum()


def per_basin(df, col):
    rows = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = g[col].to_numpy(float)
        k = np.isfinite(y) & np.isfinite(p)
        y, p = y[k], p[k]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        rows[sid] = {"nse": nse(y, p), "kge": kge(y, p), "fhv": fhv(y, p)}
    return pd.DataFrame(rows).T


def add(m, name, paths):
    d = load_seed_avg([Path(p) for p in paths])[["station_id", "date", "pred"]]
    return m.merge(d.rename(columns={"pred": name}), on=["station_id", "date"],
                   how="inner")


def build_once():
    """Build the 9-stream frame ONCE and return (fit, val, cols).

    ⚠️ The first version built a full frame per arm — three 9-stream frames of
    ~2.9M rows each — and was OOM-killed on a 15 GB box that was also training.
    All three arms are the SAME frame under different weight vectors, so one
    build serves all of them: 3x less memory and 3x faster. float32 halves it
    again (NSE/KGE/FHV need nothing like float64 precision here).
    """
    tr, cols = build_merged(train=True)
    cols = list(cols)
    for name, seeds in (("lstm_multi5", MULTI5), ("lstm_multi6", MULTI6)):
        key = name.split("_")[1]
        tr = add(tr, name, [L / f"camels531ls_{key}_nhlstm_TRAIN_{s}.csv.gz"
                            for s in seeds])
        cols.append(name)
    d = pd.to_datetime(tr["date"])
    keep = ["station_id", "truth"] + cols
    fit = tr.loc[d <= CV_FIT_END, keep].copy()
    val = tr.loc[(d >= CV_VAL_START) & (d <= CV_VAL_END), keep].copy()
    del tr
    for f in (fit, val):
        for c in ["truth"] + cols:
            f[c] = f[c].astype("float32")
    print(f"one build: fit {len(fit):,} rows, val {len(val):,} rows, "
          f"{len(cols)} streams", flush=True)
    return fit, val, cols


def arm(fit, val, cols, drop_member=None):
    """Re-weight the SAME frame; refit inverse-MSE on the retained columns."""
    use = [c for c in cols if c != drop_member]
    truth = fit["truth"].to_numpy(float)
    mse = np.array([np.nanmean((fit[c].to_numpy(float) - truth) ** 2) for c in use])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM * (np.ones(len(use)) / len(use)) + (1 - LAM) * raw
    w = w / w.sum()
    ens = (val[use].to_numpy(float) * w).sum(axis=1)
    return per_basin(val.assign(_ens=ens), "_ens")


def main():
    print("=" * 78)
    print("WHAT THE MEDIAN-NSE GATE DISCARDS  (train frame, leave-one-out)")
    print("=" * 78)
    fit, val, cols = build_once()
    full = arm(fit, val, cols)
    print(f"full 9-stream ensemble: {len(full)} basins   "
          f"medNSE {full['nse'].median():.6f}  meanNSE {full['nse'].mean():.6f}",
          flush=True)

    out = {"full": {k: float(v) for k, v in
                    {"median_nse": full["nse"].median(),
                     "mean_nse": full["nse"].mean(),
                     "median_kge": full["kge"].median(),
                     "mean_kge": full["kge"].mean(),
                     "median_abs_fhv": full["fhv"].abs().median()}.items()},
           "members": {}}

    for member in ("lstm_multi5", "lstm_multi6"):
        wo = arm(fit, val, cols, drop_member=member)
        idx = sorted(set(full.index) & set(wo.index))
        f, o = full.loc[idx], wo.loc[idx]
        d_nse = f["nse"] - o["nse"]
        d_kge = f["kge"] - o["kge"]
        # FHV: improvement means |bias| gets SMALLER
        d_fhv = o["fhv"].abs() - f["fhv"].abs()

        rec = {
            "median_nse_delta": float(f["nse"].median() - o["nse"].median()),
            "mean_nse_delta": float(d_nse.mean()),
            "paired_median_nse_delta": float(d_nse.median()),
            "breadth_nse": float((d_nse > 0).mean()),
            "median_kge_delta": float(f["kge"].median() - o["kge"].median()),
            "mean_kge_delta": float(np.nanmean(d_kge)),
            "median_abs_fhv_delta": float(o["fhv"].abs().median()
                                          - f["fhv"].abs().median()),
            "mean_abs_fhv_improvement": float(np.nanmean(d_fhv)),
            "basins": len(idx),
        }
        out["members"][member] = rec
        ratio = (rec["mean_nse_delta"] / rec["median_nse_delta"]
                 if abs(rec["median_nse_delta"]) > 1e-9 else float("inf"))
        print(f"\n--- {member} (leave-one-out, {len(idx)} basins) ---")
        print(f"  NSE  median delta (THE GATE) : {rec['median_nse_delta']:+.6f}")
        print(f"  NSE  MEAN   delta            : {rec['mean_nse_delta']:+.6f}"
              f"   ({ratio:.1f}x the gate's view)")
        print(f"  NSE  paired-median / breadth : {rec['paired_median_nse_delta']:+.6f}"
              f"  /  {100*rec['breadth_nse']:.1f}%")
        print(f"  KGE  median / mean delta     : {rec['median_kge_delta']:+.6f}"
              f"  /  {rec['mean_kge_delta']:+.6f}")
        print(f"  |FHV| median / mean improvement (+ = better peaks): "
              f"{rec['median_abs_fhv_delta']:+.3f}%  /  "
              f"{rec['mean_abs_fhv_improvement']:+.3f}%")

    print("\n" + "=" * 78)
    print("READING — the ratio mean/median is how much of a member's measured")
    print("value the REPORTED statistic throws away. It is a property of the")
    print("aggregate, not of the model. ⛔ It is NOT a route to 0.84.")
    print("=" * 78)
    Path("benchmarks/ledger42_metric_blindspot.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("wrote benchmarks/ledger42_metric_blindspot.json")


if __name__ == "__main__":
    sys.exit(main())
