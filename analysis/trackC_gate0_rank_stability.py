#!/usr/bin/env python3
"""LEDGER 42 TRACK C — GATE 0: is near-median membership LEARNABLE? Zero GPU.

THE QUESTION THAT DECIDES TRACK C BEFORE IT COSTS ANYTHING
-----------------------------------------------------------
The aim problem says a treatment aimed at the ~78 basins within +/-0.01 of the
median has ~7x the metric leverage of an untargeted one. But
[[median-leverage-the-targeting-error]] names the blocker in advance:

  "selecting them requires knowing the ranking, which is a TEST-SIDE quantity.
   Fitting a near-median specialist on TRAIN ranks and hoping the ranking
   transfers is a real experiment, NOT a free win -- rank is noisy and basins
   move between windows."

This is the same shape that killed per-basin weighting: oracle +0.0070, but
best-member choice was only 21.1% stable vs 15.1% chance, unlearnable from the
27 statics, and -0.0086 to act on. ⇒ **Measure the stability FIRST.**

⛔ PRE-REGISTERED BARS (written before the number exists):
   Let O = |train-near-median ∩ test-near-median|, and E = the chance overlap
   |train| * |test| / N under independence.
     LIFT = O / E
     PASS         LIFT >= 3.0  AND  O >= 15   -> membership carries real signal;
                                                 Track C may proceed to design.
     FAIL         LIFT <  2.0                 -> unlearnable; TRACK C CLOSED at
                                                 zero GPU, recorded as a result.
     INCONCLUSIVE otherwise                   -> report, do not proceed.
   A second, stronger read is also reported: whether the STATICS predict test
   near-median membership better than the base rate (the per-basin-weighting
   analogue that failed at 22.1% vs a 25.5% majority baseline).

READS (both registered NOT-A-QUERY, ledger42_branch_table):
  class 2  the ALREADY-SPENT per-basin test NSEs in noq_test_result.json
  train    the production 9-stream ensemble on the val slice (unlimited)
⛔ No new test-window read. No predictions are loaded from the test frame.

Run on the 1080:  gpu1080/.venv/bin/python analysis/trackC_gate0_rank_stability.py
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
CV_VAL_START, CV_VAL_END = "1990-10-01", "1995-09-30"
WINDOW = 0.01
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")
LIFT_PASS, LIFT_FAIL, MIN_OVERLAP = 3.0, 2.0, 15


def per_basin_nse(df, col):
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


def train_side_nse():
    """Production 9-stream ensemble, TRAIN frame, val slice."""
    tr, cols = build_merged(train=True)
    for name, seeds in (("lstm_multi5", MULTI5), ("lstm_multi6", MULTI6)):
        key = name.split("_")[1]
        d = load_seed_avg([L / f"camels531ls_{key}_nhlstm_TRAIN_{s}.csv.gz"
                           for s in seeds])[["station_id", "date", "pred"]]
        tr = tr.merge(d.rename(columns={"pred": name}),
                      on=["station_id", "date"], how="inner")
        cols = list(cols) + [name]
    d = pd.to_datetime(tr["date"])
    fit = tr[d <= "1990-09-30"]
    val = tr[(d >= CV_VAL_START) & (d <= CV_VAL_END)].copy()
    truth = fit["truth"].to_numpy(float)
    mse = np.array([np.nanmean((fit[c].to_numpy(float) - truth) ** 2) for c in cols])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM * (np.ones(len(cols)) / len(cols)) + (1 - LAM) * raw
    w = w / w.sum()
    val["ens"] = (val[cols].to_numpy(float) * w).sum(axis=1)
    return per_basin_nse(val, "ens")


def near(s, window=WINDOW):
    med = float(s.median())
    return set(s[(s - med).abs() <= window].index)


def main():
    # ---- test side: the ALREADY-SPENT query (class 2) ----------------------
    src = next(p for p in (Path("noq_test_result.json"),
                           Path("analysis/noq_test_result.json")) if p.exists())
    res = json.load(open(src))
    if abs(res["median_nse"] - 0.8362893021622821) > 1e-9:
        raise SystemExit(f"{src}: not the recorded frozen query")
    te = pd.Series({str(k).zfill(8): float(v)
                    for k, v in res["per_basin_nse"].items()})

    tr = train_side_nse()
    tr.index = [str(i).zfill(8) for i in tr.index]
    common = sorted(set(tr.index) & set(te.index))
    tr, te = tr[common], te[common]
    N = len(common)
    print("=" * 76)
    print("TRACK C GATE 0 — is near-median membership LEARNABLE?")
    print("=" * 76)
    print(f"basins common to both frames: {N}")
    print(f"train median {tr.median():.6f}   test median {te.median():.6f}")

    A, B = near(tr), near(te)
    O = len(A & B)
    E = len(A) * len(B) / N
    lift = O / E if E > 0 else float("nan")
    print(f"\ntrain near-median cohort: {len(A)} basins")
    print(f"test  near-median cohort: {len(B)} basins")
    print(f"OVERLAP observed {O}   expected-by-chance {E:.1f}   LIFT {lift:.2f}x")

    # hypergeometric tail: P(overlap >= O) by simulation (no scipy dependency)
    rng = np.random.default_rng(0)
    idx = np.arange(N)
    sims = np.array([len(set(rng.choice(idx, len(A), replace=False))
                         & set(rng.choice(idx, len(B), replace=False)))
                     for _ in range(4000)])
    p = float((sims >= O).mean())
    print(f"  null overlap distribution: mean {sims.mean():.1f}, "
          f"95th pct {np.percentile(sims, 95):.0f}   p(>={O}) = {p:.4f}")

    # a WIDER window is the fairer test of the mechanism: does a treatment aimed
    # at the train cohort land anywhere near the test cohort?
    print("\nsensitivity to the window (train cohort vs test cohort):")
    rows = []
    for wdw in (0.005, 0.01, 0.02, 0.05):
        a, b = near(tr, wdw), near(te, wdw)
        o = len(a & b)
        e = len(a) * len(b) / N
        rows.append({"window": wdw, "n_train": len(a), "n_test": len(b),
                     "overlap": o, "expected": e,
                     "lift": (o / e if e > 0 else float("nan"))})
        print(f"  ±{wdw:<6} train {len(a):3d}  test {len(b):3d}  "
              f"overlap {o:3d}  chance {e:5.1f}  lift {rows[-1]['lift']:.2f}x")

    # rank correlation: the weaker but assumption-free view
    rho = float(pd.Series(tr).rank().corr(pd.Series(te).rank()))
    print(f"\nSpearman rank corr train vs test: {rho:.4f}")

    if lift >= LIFT_PASS and O >= MIN_OVERLAP:
        verdict = "PASS — membership carries real signal; Track C may be designed"
    elif lift < LIFT_FAIL:
        verdict = ("FAIL — near-median membership is not transferable; "
                   "TRACK C CLOSED at zero GPU")
    else:
        verdict = "INCONCLUSIVE — report, do not proceed"
    print(f"\n{'=' * 76}\nGATE 0 VERDICT: {verdict}\n{'=' * 76}")
    if lift < LIFT_FAIL:
        print("⭐ This closes the TARGETING axis the aim problem pointed at: the "
              "leverage is real (7x) but the cohort cannot be IDENTIFIED in "
              "advance — the same shape as per-basin weighting (oracle +0.0070, "
              "unlearnable, −0.0086 to act on).")

    Path("benchmarks/ledger42_trackC_gate0.json").write_text(json.dumps({
        "n_basins": N, "train_median": float(tr.median()),
        "test_median": float(te.median()),
        "window": WINDOW, "n_train_cohort": len(A), "n_test_cohort": len(B),
        "overlap": O, "expected_by_chance": E, "lift": lift,
        "p_value_sim": p, "spearman_rank_corr": rho,
        "window_sensitivity": rows,
        "bars": {"lift_pass": LIFT_PASS, "lift_fail": LIFT_FAIL,
                 "min_overlap": MIN_OVERLAP},
        "verdict": verdict,
    }, indent=2, default=float))
    print("wrote benchmarks/ledger42_trackC_gate0.json")


if __name__ == "__main__":
    sys.exit(main())
