#!/usr/bin/env python3
"""LEDGER 44 / S2 — THE GATE TRANSFER FUNCTION.

⛔⛔ NO-SHIP CLAUSE — pre-registered in benchmarks/ledger44_branch_table.md
    BEFORE any number here was computed:

      This script reads the HELD-OUT window for candidates that have ALREADY
      BEEN REJECTED. No candidate may be shipped, reopened, or re-seeded on the
      basis of this read. Its sole purpose is to estimate how a TRAIN-side gate
      delta maps to a HELD-OUT delta. If a rejected candidate reads positive
      held-out, that is SELECTION ON TEST; it is recorded as such and NOT acted on.

WHY
---
Every member verdict in ledgers 40-43 was read on the TRAIN-side surface, which
is INSIDE the training window (train 01/10/1980-30/09/1995; the harness scores
1990-10-01..1995-09-30). Nobody has ever measured what a train-side gate delta
is worth on the surface the record is claimed on. This measures it, and the
deliverable is the RESIDUAL SCATTER of the regression -- the train-side gate's
resolution expressed in held-out NSE.

ONE SCORER, TWO SURFACES
------------------------
Both the train-side and held-out deltas are computed by this same code path, so
the pairing cannot be an artifact of two differently-configured historical
anchors (an anchor must name its artifact AND its configuration; the multi8
1-seed and 2-seed anchors differ by 30%).

VALIDATION AGAINST A KNOWN ANSWER, RUN FIRST
--------------------------------------------
    --validate  reproduces gate_multi8_swap_2seed.json (multi8, 2 seeds, swap
    vs lstm_multi6, TRAIN surface, h==1): paired -0.00019128746007801567,
    diff-of-medians -0.0008253363264107971, breadth 0.4670433145009416.
If that does not reproduce, nothing this script says about anything counts.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
import gate_eval as G                                        # noqa: E402
from phase_lever import load_allh, VAL_START, VAL_END        # noqa: E402
from rescore9 import glob_seeds                              # noqa: E402

THETA, LAM = 4.0, 0.25
HD = Path("dhbv_dumps")
TEST_END = pd.Timestamp("2008-12-07")     # MAURER ENDS 2008 — the record's window

KNOWN = {"paired": -0.00019128746007801567,
         "diff_of_medians": -0.0008253363264107971,
         "breadth": 0.4670433145009416}


def build_streams(train):
    """The frozen 9, ALL LEADS, on either surface. h==1 is a filter on this."""
    t = "TRAIN_" if train else ""
    s = {}
    for f in ("daymet", "nldas", "maurer"):
        s[f"lstm_{f}"] = load_allh(G._lstm_seeds(f, train))
    s["lstm_multi"] = load_allh(G._multi_seeds(train))
    for k in ("multi5", "multi6"):
        s[f"lstm_{k}"] = load_allh(glob_seeds(
            f"camels531ls_{k}_nhlstm_{t}s*.csv.gz", G.L))
    for f in ("daymet", "nldas", "maurer"):
        s[f"dhbv_{f}"] = load_allh(glob_seeds(
            f"camels531ls_{f}_dhbv_ALLH_{t}s*.csv.gz", HD), shift_days=1)
    for k, v in s.items():
        if v is None or not len(v):
            raise SystemExit(f"{k}: no dumps (train={train})")
    return s


def merge(streams):
    merged = None
    for name, df in streams.items():
        d = df[["station_id", "date", "h", "pred"]].rename(columns={"pred": name})
        merged = (df[["station_id", "date", "h", "truth"]].merge(d, on=["station_id", "date", "h"])
                  if merged is None else merged.merge(d, on=["station_id", "date", "h"]))
    ids = set(str(x).zfill(8) for x in json.load(open(G.SUBSET))["531"])
    return merged[merged.station_id.isin(ids)].reset_index(drop=True)


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


def invmse_weights(fit, cols):
    truth = fit["truth"].to_numpy(float)
    mse = np.array([np.nanmean((fit[c].to_numpy(float) - truth) ** 2) for c in cols])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM * np.ones(len(cols)) / len(cols) + (1 - LAM) * raw
    return w / w.sum()


def score(tr, te, cols, frame, surface):
    """PRODUCTION rule. Weights ALWAYS fit on TRAIN rows (the fourth frame).

    train surface : fit date <  1990-10-01, score 1990-10-01..1995-09-30
    test  surface : fit the whole TRAIN frame, score the TEST frame
    """
    f = (lambda d: d[d.h == 1]) if frame == "h1" else (lambda d: d)
    trf = f(tr)
    if surface == "train":
        fit = trf[trf.date < VAL_START]
        sc = trf[(trf.date >= VAL_START) & (trf.date <= VAL_END)]
    else:
        fit = trf
        sc = f(te)
        sc = sc[sc.date <= TEST_END]
    w = invmse_weights(fit, cols)
    sc = sc.copy()
    sc["_ens"] = (sc[cols].to_numpy(float) * w).sum(axis=1)
    return per_basin_nse(sc, "_ens"), w


def swap_delta(tr, te, cand, replaces, frame, surface):
    base_cols = [c for c in tr.columns if c not in ("station_id", "date", "h", "truth")
                 and not c.startswith("cand_")]
    cand_cols = [c for c in base_cols if c != replaces] + [cand]
    b, _ = score(tr, te, base_cols, frame, surface)
    c, wc = score(tr, te, cand_cols, frame, surface)
    idx = b.index.intersection(c.index)
    d = (c[idx] - b[idx])
    return {"baseline_median": float(b[idx].median()),
            "candidate_median": float(c[idx].median()),
            "paired": float(d.median()),
            "diff_of_medians": float(c[idx].median() - b[idx].median()),
            "breadth": float((d > 0).mean()),
            "n_basins": int(len(idx))}


# candidates that already have BOTH surfaces' dumps (found by S0.1: 6 of 7).
# All are entered by the SAME construction -- swap vs lstm_multi6 -- because the
# question is how a train-side delta maps to a held-out delta under an identical
# construction, NOT a re-adjudication of each candidate's own pre-registered arm.
# multi14 joined on 2026-08-27 once ledger 44 produced its TEST dump (the 7th
# candidate; the other six already had one). It is the best train-side
# all-leads delta ever measured (+0.000213), so it anchors the top of the range.
CANDIDATES = ["multi8", "multi9", "multi10", "multi11", "multi5b", "multidrop",
              "multi14"]


def add_candidate(merged, tag, train):
    t = "TRAIN_" if train else ""
    paths = glob_seeds(f"camels531ls_{tag}_nhlstm_{t}s*.csv.gz", G.L)
    if not paths:
        raise SystemExit(f"{tag}: no dumps (train={train})")
    d = load_allh(paths)[["station_id", "date", "h", "pred"]].rename(
        columns={"pred": f"cand_{tag}"})
    before = len(merged)
    out = merged.merge(d, on=["station_id", "date", "h"], how="inner")
    if len(out) < 0.95 * before:
        raise SystemExit(f"{tag}: join lost {100*(1-len(out)/before):.1f}% of rows")
    return out, [Path(p).name for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="reproduce gate_multi8_swap_2seed.json and stop")
    ap.add_argument("--out", default="benchmarks/ledger44_gate_transfer.json")
    args = ap.parse_args()

    print("building TRAIN streams (all leads)...", file=sys.stderr)
    TR = merge(build_streams(True))
    print(f"  TRAIN merged: {len(TR):,} rows, {TR.station_id.nunique()} basins",
          file=sys.stderr)

    if args.validate:
        print("\n" + "=" * 78)
        print("VALIDATION — reproduce gate_multi8_swap_2seed.json (TRAIN, h==1, swap)")
        print("=" * 78)
        tr2, files = add_candidate(TR, "multi8", True)
        print(f"  candidate dumps: {files}")
        got = swap_delta(tr2, None, "cand_multi8", "lstm_multi6", "h1", "train")
        allok = True
        for k, want in KNOWN.items():
            d = got[k] - want
            ok = abs(d) < 1e-9
            allok &= ok
            print(f"  {k:<16} got {got[k]:+.12f}  want {want:+.12f}  "
                  f"diff {d:+.2e}  {'OK' if ok else 'MISMATCH'}")
        print(f"  => {'PASS' if allok else 'FAIL'}")
        return 0 if allok else 1

    print("building TEST streams (all leads)...", file=sys.stderr)
    TE = merge(build_streams(False))
    print(f"  TEST merged: {len(TE):,} rows, {TE.station_id.nunique()} basins",
          file=sys.stderr)

    rows = []
    print("\n" + "=" * 100)
    print("S2 — TRAIN-SIDE (in-sample) vs HELD-OUT swap delta, identical construction")
    print("     ⛔ NO-SHIP CLAUSE APPLIES: these candidates are already rejected.")
    print("=" * 100)
    hdr = (f"{'candidate':<12} {'seeds':>5} | {'TRAIN h1':>11} {'TEST h1':>11} | "
           f"{'TRAIN all':>11} {'TEST all':>11}")
    print(hdr); print("-" * len(hdr))
    for tag in CANDIDATES:
        try:
            tr2, ftr = add_candidate(TR, tag, True)
            te2, fte = add_candidate(TE, tag, False)
        except SystemExit as e:
            print(f"{tag:<12} SKIPPED: {e}")
            continue
        if len(ftr) != len(fte):
            print(f"{tag:<12} SKIPPED: {len(ftr)} TRAIN seeds vs {len(fte)} TEST seeds")
            continue
        r = {"candidate": tag, "n_seeds": len(ftr),
             "train_dumps": ftr, "test_dumps": fte}
        for frame in ("h1", "all"):
            r[f"train_{frame}"] = swap_delta(tr2, None, f"cand_{tag}", "lstm_multi6",
                                             frame, "train")
            r[f"test_{frame}"] = swap_delta(tr2, te2, f"cand_{tag}", "lstm_multi6",
                                            frame, "test")
        rows.append(r)
        print(f"{tag:<12} {len(ftr):>5} | {r['train_h1']['paired']:>+11.6f} "
              f"{r['test_h1']['paired']:>+11.6f} | {r['train_all']['paired']:>+11.6f} "
              f"{r['test_all']['paired']:>+11.6f}")

    # ------------------------------------------------ the transfer function
    out = {"no_ship_clause": "Candidates here are already rejected. No candidate "
                             "may be shipped, reopened or re-seeded on the basis "
                             "of this read. Purpose: transfer function only.",
           "candidates": rows}
    print("\n" + "=" * 78)
    print("THE TRANSFER FUNCTION — held-out delta regressed on train-side delta")
    print("=" * 78)
    for frame in ("h1", "all"):
        x = np.array([r[f"train_{frame}"]["paired"] for r in rows])
        y = np.array([r[f"test_{frame}"]["paired"] for r in rows])
        if len(x) < 3:
            print(f"  [{frame}] n={len(x)} — too few points to regress")
            continue
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        scatter = float(resid.std(ddof=min(2, len(x) - 1)))
        r_p = float(np.corrcoef(x, y)[0, 1])
        sr = pd.Series(x).corr(pd.Series(y), method="spearman")
        out[f"transfer_{frame}"] = {
            "n": int(len(x)), "intercept": float(coef[0]), "slope": float(coef[1]),
            "pearson_r": r_p, "spearman_r": float(sr), "residual_scatter": scatter,
            "train_delta_range": [float(x.min()), float(x.max())],
            "test_delta_range": [float(y.min()), float(y.max())]}
        print(f"\n  [{frame}]  n={len(x)}")
        print(f"    held-out = {coef[0]:+.6f} + {coef[1]:+.4f} * train-side")
        print(f"    pearson r  {r_p:+.3f}    spearman r {sr:+.3f}")
        print(f"    RESIDUAL SCATTER {scatter:.6f}  <- the train-side gate's "
              f"resolution in held-out NSE")
        print(f"    train-side deltas span [{x.min():+.6f}, {x.max():+.6f}]")
        print(f"    held-out  deltas span [{y.min():+.6f}, {y.max():+.6f}]")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
