#!/usr/bin/env python3
"""LEDGER 42 B1 — F1 vs F2 (vs F3) on the TRAIN frame. Zero GPU.

Resolves, BEFORE the single test query:
  (1) the registered F2 statistic on the train frame;
  (2) the stride-14 vs continuous-daily question;
  (3) why phase_full9.py's h==1 read (0.946695) MISSES the production anchor
      (0.950458) by -0.0038 while rescore9.py reproduces it exactly.

⭐ THE FORKING PATH THIS CLOSES (registered in ledger42_branch_table.md, S0.1):

  F2 = the FROZEN F1 weight vector, scored on a bigger SAMPLE.
       A sample change, never a configuration change.

  phase_full9.py instead REFIT the inverse-MSE vector on all-leads rows. That
  is a different ensemble, and it reads -0.0038 at h==1 -- the size of the
  entire gap to 0.84. Both numbers are defensible; only one is F2, and which
  one is F2 had to be decided before the number was seen. It was.

Reported here:
  F1        h==1, frozen weights          -> must reproduce 0.950458
  F2        all leads, FROZEN weights     -> THE registered statistic
  F2-refit  all leads, weights refit      -> SECONDARY, clearly labelled
  per-phase h=1..14, frozen weights       -> the phase-spread table

F3: S0.2 measured stride=14, dup=0, gap1=1.0 on every full-14 dump, so the
stride-14 windows TILE the calendar with no lead overlap => each target date
appears exactly once => F2 IS the continuous-daily frame. F3 is not a separate
score; this script verifies the tiling property it rests on.

Run on the 1080:  gpu1080/.venv/bin/python analysis/all_leads_train.py
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

import gate_eval as G                                    # noqa: E402
from phase_lever import load_allh, VAL_START, VAL_END, THETA, LAM   # noqa: E402

D = Path("gpu1080/dumps")
HD = Path("dhbv_dumps")
PREREG = 0.950458          # production anchor, h==1, frozen weights
ANCHOR_TOL = 0.0005        # registered reproduction tolerance
GAP = 0.0037

MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")


def per_basin_nse(df, col):
    """NaN-masked per-basin NSE. Identical rule to score_noq_test.py."""
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


def build_full9_allh():
    """The frozen 9 streams, ALL leads, train side. Mirrors phase_full9."""
    streams, manifest = {}, {}
    for f in ("daymet", "nldas", "maurer"):
        s = G._lstm_seeds(f, True)
        streams[f"lstm_{f}"] = load_allh(s)
        manifest[f"lstm_{f}"] = [p.name for p in s]
    m = G._multi_seeds(True)
    streams["lstm_multi"] = load_allh(m)
    manifest["lstm_multi"] = [p.name for p in m]
    for tag, seeds in (("multi5", MULTI5), ("multi6", MULTI6)):
        s = [D / f"camels531ls_{tag}_nhlstm_TRAIN_{x}.csv.gz" for x in seeds]
        missing = [str(p) for p in s if not p.exists()]
        if missing:
            raise SystemExit(f"lstm_{tag}: missing {missing}")
        streams[f"lstm_{tag}"] = load_allh(s)
        manifest[f"lstm_{tag}"] = [p.name for p in s]
    # dHBV t0 convention differs by one day; shift_days=1 (phase_full9 precedent)
    for f in ("daymet", "nldas", "maurer"):
        s = sorted(HD.glob(f"camels531ls_{f}_dhbv_ALLH_TRAIN_s*.csv.gz"))
        if not s:
            raise SystemExit(f"dhbv_{f}: no ALLH TRAIN dumps")
        streams[f"dhbv_{f}"] = load_allh(s, shift_days=1)
        manifest[f"dhbv_{f}"] = [p.name for p in s]

    print("--- frozen 9-stream manifest (ALL LEADS, TRAIN) ---")
    for k, v in manifest.items():
        print(f"  {k:14s} n={len(v)}")

    merged, names = None, []
    for name, df in streams.items():
        d = df.rename(columns={"pred": name})[["station_id", "date", "h", "truth", name]]
        if merged is None:
            merged = d
        else:
            merged = merged.merge(d.drop(columns=["truth"]),
                                  on=["station_id", "date", "h"], how="inner")
        names.append(name)
    return merged, names, manifest


def fit_invmse(fit_df, cols):
    """PRODUCTION weight rule (noq_harness.fit_global_invmse): mse**(-THETA),
    shrunk toward equal by LAM. Fit rows are the caller's choice."""
    truth = fit_df["truth"].to_numpy(float)
    mse = np.array([np.nanmean((fit_df[c].to_numpy(float) - truth) ** 2) for c in cols])
    if not np.all(np.isfinite(mse)):
        raise SystemExit(f"non-finite MSE: {dict(zip(cols, mse))}")
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM * (np.ones(len(cols)) / len(cols)) + (1 - LAM) * raw
    return w / w.sum(), mse


def med(df, cols, w):
    ens = (df[cols].to_numpy(float) * w).sum(axis=1)
    s = per_basin_nse(df.assign(_e=ens), "_e")
    return float(s.median()), int(len(s)), s


def main():
    print("=" * 78)
    print("LEDGER 42 B1 — F1 vs F2 on the TRAIN frame (frozen 9 streams)")
    print("=" * 78)
    merged, names, manifest = build_full9_allh()
    merged = merged.sort_values(["station_id", "date", "h"])
    print(f"\nmerged: {len(merged):,} rows, {merged.station_id.nunique()} basins, "
          f"leads {sorted(merged.h.unique())}")

    # ---- F3 tiling verification (the stride-14 question) -------------------
    dup = int(merged.duplicated(["station_id", "date"]).sum())
    gaps = merged.drop_duplicates(["station_id", "date"]) \
                 .groupby("station_id")["date"].diff().dt.days.dropna()
    frac1 = float((gaps == 1).mean())
    print(f"\nF3 TILING CHECK: duplicate (station,date) rows = {dup}   "
          f"day-gap==1 fraction = {frac1:.4f}")
    tiling = (dup == 0 and frac1 > 0.99)
    print("  => stride-14 windows TILE the calendar; F2 IS the continuous-daily "
          "frame (F3 == F2)." if tiling else
          "  => NOT a clean tiling; F3 must be reported separately.")

    fit_all = merged[merged.date < VAL_START]
    val_all = merged[(merged.date >= VAL_START) & (merged.date <= VAL_END)].copy()
    fit_h1, val_h1 = fit_all[fit_all.h == 1], val_all[val_all.h == 1]
    print(f"\nfit(all leads) {len(fit_all):,}   val(all leads) {len(val_all):,}")
    print(f"fit(h==1)      {len(fit_h1):,}   val(h==1)      {len(val_h1):,}")

    # ---- the FROZEN vector: fit on TRAIN h==1 rows, exactly as production ---
    w_frozen, mse_frozen = fit_invmse(fit_h1, names)
    print("\nFROZEN production weight vector (fit on train h==1 rows):")
    for n, x in sorted(zip(names, w_frozen), key=lambda t: -t[1]):
        print(f"   {n:14s} {x:.4f}")

    # ---- F1: the anchor ----------------------------------------------------
    f1, n_f1, s_f1 = med(val_h1, names, w_frozen)
    diff = f1 - PREREG
    print("\n" + "-" * 78)
    print(f"F1  h==1, frozen weights = {f1:.6f}   (PREREG {PREREG:.6f}, "
          f"diff {diff:+.6f}, n={n_f1})")
    ok = abs(diff) <= ANCHOR_TOL
    print("  ✅ ANCHOR REPRODUCED — the frame is production." if ok else
          "  *** ANCHOR MISSED — STOP. Nothing below is trustworthy. ***")
    print("-" * 78)
    if not ok:
        Path("benchmarks/ledger42_b1_all_leads_train.json").write_text(json.dumps(
            {"status": "ANCHOR_FAIL", "F1": f1, "prereg": PREREG,
             "diff": diff, "tol": ANCHOR_TOL}, indent=2))
        raise SystemExit(1)

    # ---- F2: THE registered statistic — frozen weights, all leads ----------
    f2, n_f2, s_f2 = med(val_all, names, w_frozen)
    # ---- F2-refit: SECONDARY, clearly labelled -----------------------------
    w_refit, _ = fit_invmse(fit_all, names)
    f2r, n_f2r, _ = med(val_all, names, w_refit)
    f1r, _, _ = med(val_h1, names, w_refit)

    # ---- per-phase table, frozen weights -----------------------------------
    print("\nPER-PHASE (frozen weights):")
    per_phase = []
    for h in range(1, 15):
        m, n, _ = med(val_all[val_all.h == h], names, w_frozen)
        per_phase.append(m)
        star = "   <-- F1, THE REPORTED FRAME" if h == 1 else ""
        print(f"   h={h:2d}  medNSE={m:.6f}  basins={n}{star}")
    pp = np.array(per_phase)
    spread = float(np.nanmax(pp) - np.nanmin(pp))
    rank = int((pp > pp[0]).sum()) + 1

    print("\n" + "=" * 78)
    print("RESULTS (train frame, val slice 1990-10-01..1995-09-30)")
    print("=" * 78)
    print(f"  F1  h==1,      frozen weights : {f1:.6f}   (anchor {PREREG:.6f})")
    print(f"  F2  all leads, frozen weights : {f2:.6f}   [THE REGISTERED STATISTIC]")
    print(f"      F2 - F1                   : {f2 - f1:+.6f}   "
          f"({(f2 - f1) / GAP:+.2f}x the gap to 0.84)")
    print(f"  --- secondary, clearly labelled, NOT F2 ---")
    print(f"  all leads, weights REFIT      : {f2r:.6f}")
    print(f"  h==1,      weights REFIT      : {f1r:.6f}   "
          f"(anchor diff {f1r - PREREG:+.6f})")
    print(f"  ⇒ refitting on all-leads rows moves the h==1 read by "
          f"{f1r - f1:+.6f} — this is why phase_full9.py missed the anchor.")
    print(f"\n  phase spread                  : {spread:.6f}  "
          f"({spread / GAP:.2f}x the gap)")
    print(f"  h=1 rank of 14                : {rank}")
    print(f"  F3 == F2 (stride-14 tiles)    : {tiling}")

    Path("benchmarks/ledger42_b1_all_leads_train.json").write_text(json.dumps({
        "status": "OK",
        "frame": {"val_start": str(VAL_START.date()), "val_end": str(VAL_END.date()),
                  "rows_all": int(len(val_all)), "rows_h1": int(len(val_h1)),
                  "basins": int(merged.station_id.nunique())},
        "anchor": {"F1": f1, "prereg": PREREG, "diff": diff, "tol": ANCHOR_TOL,
                   "reproduced": ok},
        "F1_h1_frozen": f1,
        "F2_allleads_frozen": f2,
        "F2_minus_F1": f2 - f1,
        "secondary_allleads_refit": f2r,
        "secondary_h1_refit": f1r,
        "refit_moves_h1_by": f1r - f1,
        "per_phase_frozen": [float(x) for x in per_phase],
        "phase_spread": spread,
        "h1_rank_of_14": rank,
        "F3_tiling": {"dup_station_date": dup, "frac_gap_1day": frac1,
                      "F2_equals_F3": tiling},
        "weights_frozen": {n: float(x) for n, x in zip(names, w_frozen)},
        "weights_refit_allleads": {n: float(x) for n, x in zip(names, w_refit)},
        "manifest": manifest,
    }, indent=2))
    print("\nwrote benchmarks/ledger42_b1_all_leads_train.json")


if __name__ == "__main__":
    sys.exit(main())
