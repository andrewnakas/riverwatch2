#!/usr/bin/env python3
"""LEDGER 44 / S1 — THE HELD-OUT ENSEMBLING LAW.

WHY THIS EXISTS
---------------
Two closures in this campaign were measured on the TRAIN-side surface:
    "seed depth SATURATED"        (marginal seed -0.00010)
    "ensemble size peaks at 4"    (greedy curve, adds 5-9 sum negative)
That surface is INSIDE THE TRAINING WINDOW.  check_split.py pins Li/Song
training to 01/10/1980 -> 30/09/1995 and the harness scores 1990-10-01 ->
1995-09-30, so the networks were trained on every row of it.  Measured
in-sample inflation (daymet s111, h==1): 0.9024 vs 0.7634 held-out = 0.139 NSE.

Ensembling is a VARIANCE-REDUCTION effect: it pays off out-of-sample.  An
in-sample surface is therefore the one surface on which its value is
systematically understated -- which is exactly where both closures were read.
This script recomputes both curves on the HELD-OUT window and fits

    NSE(n) = a - b/n

so the asymptote `a` (what infinite compute buys) and the headroom `a - current`
become numbers instead of judgements.

PROTOCOL (frames are the trap that has cost this campaign most)
--------------------------------------------------------------
  * weights are ALWAYS fit on TRAIN rows, on both surfaces -- production rule,
    theta=4.0, lam=0.25, one GLOBAL inverse-MSE vector (score_noq_test.py).
  * TRAIN surface : fit on rows <= CV_FIT_END, score rows >= CV_VAL_START
                    (reproduces noq_harness.py --mode baseline -> 0.950458)
  * TEST  surface : fit on the whole TRAIN frame, score the TEST frame
                    (reproduces score_noq_test.py --mode frozen -> 0.8362893...)
  * everything is h==1 (load_seed_avg filters it); all-leads is a separate frame
    and is handled by ledger44_allleads.py, not here.

VALIDATION GATE: both anchors must reproduce before any new number is printed.
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())

from gate_eval import (L, H, SUBSET, CV_FIT_END, CV_VAL_START,          # noqa: E402
                       _lstm_seeds, _multi_seeds, _dhbv_seeds)
from phase_c import load_seed_avg                                       # noqa: E402

THETA, LAM = 4.0, 0.25
MULTI5_SEEDS = ("s111", "s222", "s333", "s444", "s555")
MULTI6_SEEDS = ("s111", "s222", "s333")

ANCHOR_TRAIN = 0.950458
ANCHOR_TEST = 0.8362893021622821


# --------------------------------------------------------------- stream tables
def seed_paths(train):
    """The 9 shipped streams -> their seed dumps, honouring every exclusion rule
    in gate_eval (superseded nldas s111, _MULTI_WEAK_SEEDS, _OLD/_bak markers).
    dhbv streams carry shift_days=1, recorded alongside the paths."""
    tag = "TRAIN_" if train else ""
    out = {}
    for f in ("daymet", "nldas", "maurer"):
        out[f"lstm_{f}"] = (_lstm_seeds(f, train), 0)
    out["lstm_multi"] = (_multi_seeds(train), 0)
    out["lstm_multi5"] = ([L / f"camels531ls_multi5_nhlstm_{tag}{s}.csv.gz"
                           for s in MULTI5_SEEDS], 0)
    out["lstm_multi6"] = ([L / f"camels531ls_multi6_nhlstm_{tag}{s}.csv.gz"
                           for s in MULTI6_SEEDS], 0)
    for f in ("daymet", "nldas", "maurer"):
        out[f"dhbv_{f}"] = (_dhbv_seeds(f, train), 1)
    for k, (ps, _) in out.items():
        miss = [p for p in ps if not Path(p).exists()]
        if miss:
            raise SystemExit(f"{k}: missing dumps {miss}")
        if not ps:
            raise SystemExit(f"{k}: no seed dumps found (train={train})")
    return out


def load_surface(train):
    """Load EVERY seed dump once, h==1, as a (station,date)-indexed Series.

    Loading per-seed rather than per-stream is the whole point: build_merged
    seed-averages internally and cannot express 'this stream with k of its
    seeds', which is the quantity both curves are made of.
    """
    tbl = seed_paths(train)
    ids = set(str(x).zfill(8) for x in json.load(open(SUBSET)).get("531", []))
    per_seed, truth = {}, None
    for stream, (paths, shift) in tbl.items():
        cols = []
        for p in paths:
            d = load_seed_avg([p], shift_days=shift)
            d = d[d.station_id.isin(ids)]
            s = d.set_index(["station_id", "date"])["pred"]
            s.name = f"{stream}::{Path(p).name}"
            cols.append(s)
            if truth is None:
                truth = d.set_index(["station_id", "date"])["truth"]
        per_seed[stream] = cols
        print(f"  {stream:14s} {len(cols)} seeds  "
              f"{[Path(p).name.split('_')[-1].replace('.csv.gz','') for p in paths]}",
              file=sys.stderr)
    # ONE common index across every seed of every stream, so changing the seed
    # count never changes the row set (which would confound the whole curve).
    idx = None
    for cols in per_seed.values():
        for s in cols:
            idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.intersection(truth.index)
    frame = {"truth": truth.reindex(idx).to_numpy(float)}
    for stream, cols in per_seed.items():
        frame[stream] = np.stack([s.reindex(idx).to_numpy(float) for s in cols])
    dates = np.array([d for _, d in idx])
    sids = np.array([s for s, _ in idx])
    # builtin min/max: these are fixed-width unicode arrays and np.min has no
    # loop for them on this numpy build.
    print(f"  -> common frame: {len(idx):,} rows, {len(set(sids))} basins, "
          f"{min(dates.tolist())} .. {max(dates.tolist())}", file=sys.stderr)
    return frame, sids, dates


# ------------------------------------------------------------- production rule
def per_basin_nse(sids, truth, pred):
    out = []
    for b in np.unique(sids):
        m = sids == b
        y, p = truth[m], pred[m]
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        out.append(1.0 - np.mean((y - p) ** 2) / np.var(y))
    return np.array(out)


def fit_global_invmse(mat, truth, theta=THETA, lam=LAM):
    """score_noq_test.py's rule verbatim: nanmean MSE per stream, inverse-MSE
    to the power theta, shrunk lam toward equal weight."""
    mse = np.array([np.nanmean((row - truth) ** 2) for row in mat])
    if not np.all(np.isfinite(mse)):
        raise SystemExit(f"non-finite MSE in weight fit: {mse}")
    raw = mse ** (-theta)
    raw = raw / raw.sum()
    eq = np.ones(len(mat)) / len(mat)
    w = lam * eq + (1 - lam) * raw
    return w / w.sum()


def stack(frame, streams, pick):
    """Average each stream over its selected seeds -> one row per stream."""
    return np.stack([frame[s][list(pick[s])].mean(axis=0) for s in streams])


def score(TR, TE, streams, pick, surface):
    """Fit on TRAIN rows, score on the surface's scoring rows. Returns medNSE."""
    trf, trs, trd = TR
    tef, tes, ted = TE
    if surface == "train":
        fit_m = trd <= CV_FIT_END
        val_m = trd >= CV_VAL_START
        wm = stack(trf, streams, pick)[:, fit_m]
        w = fit_global_invmse(wm, trf["truth"][fit_m])
        sm = stack(trf, streams, pick)[:, val_m]
        ens = (sm * w[:, None]).sum(axis=0)
        return float(np.median(per_basin_nse(trs[val_m], trf["truth"][val_m], ens))), w
    wm = stack(trf, streams, pick)
    w = fit_global_invmse(wm, trf["truth"])
    sm = stack(tef, streams, pick)
    ens = (sm * w[:, None]).sum(axis=0)
    return float(np.median(per_basin_nse(tes, tef["truth"], ens))), w


def fit_law(ns, ys):
    """NSE(n) = a - b/n, least squares. Returns (a, b)."""
    n = np.asarray(ns, float)
    A = np.vstack([np.ones_like(n), -1.0 / n]).T
    a, b = np.linalg.lstsq(A, np.asarray(ys, float), rcond=None)[0]
    return float(a), float(b)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=6,
                    help="random seed-subset draws per depth (averages out which "
                         "seeds get picked; one arbitrary ordering is what made "
                         "multi_s333 look like a -0.022 marginal)")
    ap.add_argument("--out", default="benchmarks/ledger44_heldout_ensembling_law.json")
    args = ap.parse_args()
    rng = np.random.default_rng(20260827)

    print("=" * 78)
    print("LEDGER 44 / S1 — THE HELD-OUT ENSEMBLING LAW")
    print("=" * 78)
    print("\nloading TRAIN surface (in-sample: inside 1980-10-01..1995-09-30)",
          file=sys.stderr)
    TR = load_surface(train=True)
    print("\nloading TEST surface (held-out)", file=sys.stderr)
    TE = load_surface(train=False)
    streams = [s for s in TR[0] if s != "truth"]
    full = {s: tuple(range(len(TR[0][s]))) for s in streams}
    full_te = {s: tuple(range(len(TE[0][s]))) for s in streams}
    for s in streams:
        if len(full[s]) != len(full_te[s]):
            raise SystemExit(f"{s}: TRAIN has {len(full[s])} seeds, TEST has "
                             f"{len(full_te[s])} — train/test streams would "
                             f"average different members")
    n_full = sum(len(v) for v in full.values())
    print(f"\nstreams: {len(streams)}   nets at shipped depth: {n_full}")

    # ---------------------------------------------------- VALIDATION GATE
    print("\n" + "-" * 78)
    print("VALIDATION GATE — both anchors must reproduce before anything new")
    print("-" * 78)
    tr_now, _ = score(TR, TE, streams, full, "train")
    te_now, w_te = score(TR, TE, streams, full, "test")
    d_tr, d_te = tr_now - ANCHOR_TRAIN, te_now - ANCHOR_TEST
    print(f"  TRAIN-side  got {tr_now:.6f}   anchor {ANCHOR_TRAIN:.6f}   diff {d_tr:+.6f}")
    print(f"  TEST record got {te_now:.16f}")
    print(f"              anchor {ANCHOR_TEST:.16f}   diff {d_te:+.9f}")
    ok = abs(d_tr) < 5e-6 and abs(d_te) < 5e-6
    print(f"  => {'PASS' if ok else 'MISMATCH'}")
    if not ok:
        print("\n  Anchors did not reproduce. Per the checklist: report nothing else.")
        json.dump({"status": "ANCHOR_MISMATCH", "train_got": tr_now,
                   "test_got": te_now}, open(args.out, "w"), indent=2)
        return 1

    res = {"status": "OK", "n_nets_shipped": n_full,
           "anchor_train": tr_now, "anchor_test": te_now,
           "streams": {s: len(full[s]) for s in streams}}

    # ------------------------------------------------- S1a seed-depth curves
    print("\n" + "=" * 78)
    print("S1a — SEED-DEPTH CURVE, both surfaces")
    print("     (uniform cap k; each stream uses min(k, its seeds); "
          f"{args.draws} random subset draws per k)")
    print("=" * 78)
    kmax = max(len(v) for v in full.values())
    curves = {"train": [], "test": []}
    print(f"\n{'k':>2} {'nets':>5} | {'TRAIN (in-sample)':>22} | {'TEST (held-out)':>22}")
    print("-" * 60)
    for k in range(1, kmax + 1):
        nets = sum(min(k, len(full[s])) for s in streams)
        picks = []
        for _ in range(1 if k == kmax else args.draws):
            picks.append({s: tuple(rng.choice(len(full[s]),
                                              size=min(k, len(full[s])),
                                              replace=False))
                          for s in streams})
        row = {}
        for surf in ("train", "test"):
            vals = [score(TR, TE, streams, p, surf)[0] for p in picks]
            row[surf] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
            curves[surf].append((nets, float(np.mean(vals))))
        print(f"{k:>2} {nets:>5} | {row['train'][0]:>10.6f} +-{row['train'][1]:<8.6f} | "
              f"{row['test'][0]:>10.6f} +-{row['test'][1]:<8.6f}")
    res["seed_depth"] = {s: curves[s] for s in curves}

    # marginals
    print(f"\n{'step':>10} | {'TRAIN marginal':>16} | {'TEST marginal':>16} | ratio")
    print("-" * 62)
    for i in range(1, len(curves["train"])):
        mt = curves["train"][i][1] - curves["train"][i - 1][1]
        me = curves["test"][i][1] - curves["test"][i - 1][1]
        r = me / mt if abs(mt) > 1e-9 else float("nan")
        print(f"{curves['train'][i-1][0]:>4}->{curves['train'][i][0]:<5} | "
              f"{mt:>+16.6f} | {me:>+16.6f} | {r:>6.2f}x")

    # ------------------------------------------------------ S1c fit the law
    print("\n" + "=" * 78)
    print("S1c — FIT NSE(n) = a - b/n  AND THE HEADROOM PURE COMPUTE CAN BUY")
    print("=" * 78)
    for surf, cur in curves.items():
        ns = [n for n, _ in cur]
        ys = [y for _, y in cur]
        a, b = fit_law(ns, ys)
        current = a - b / n_full
        anchor = tr_now if surf == "train" else te_now
        res[f"law_{surf}"] = {"a": a, "b": b, "n_shipped": n_full,
                              "predicted_at_shipped": current,
                              "measured_at_shipped": anchor,
                              "headroom_to_asymptote": a - anchor}
        print(f"\n  [{surf.upper()}]  NSE(n) = {a:.6f} - {b:.6f}/n")
        print(f"     asymptote a                     = {a:.6f}")
        print(f"     measured at shipped n={n_full:<3d}       = {anchor:.6f}")
        print(f"     HEADROOM  a - measured          = {a - anchor:+.6f}")
        for tgt in (n_full, 2 * n_full, 4 * n_full, 10 * n_full):
            print(f"       n={tgt:<4d} -> {a - b / tgt:.6f}   "
                  f"gain vs shipped {a - b / tgt - current:+.6f}")

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
