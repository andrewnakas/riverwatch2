#!/usr/bin/env python3
"""LEDGER 42 STAGE Q — THE SINGLE TEST QUERY. Written before any result existed.

⛔⛔ ONE EXECUTION, USER-APPROVED IMMEDIATELY BEFORE RUNNING. The Li/Song test
window is a budgeted resource; every read of it is recorded, including a read
that is later discarded ("I looked but didn't use it" does not survive contact
with a writeup). Branch readings are pre-written in
benchmarks/ledger42_branch_table.md and are NOT to be composed after seeing a
number.

WHAT IT EMITS — one JSON, all quantities in one execution:
  q0  equal-weight 7-stream, F1   -> MUST reproduce 0.8298   else VOID
  q1  frozen 9-stream,      F1    -> MUST reproduce 0.8363   else VOID
  q2  frozen 9-stream,      F2    (all leads, THE SAME FROZEN WEIGHTS)
  q3  iff a Track-A candidate SHIPPED: the recombined config at F1 and F2

THE FRAMES (S0.1, frozen before any number):
  F1  h==1, the record's frame, reported in EVERY branch.
  F2  every lead, **the same frozen weight vector** — a SAMPLE change, never a
      configuration change. Refitting weights on all-leads rows produces a
      different ensemble that misses the production anchor by -0.0038 (B1);
      that number is not F2 and is not reported here.
  F3  == F2: stride-14 windows tile the calendar with no lead overlap, so each
      target date appears exactly once (S0.2 measured dup=0, gap1=1.0000).

VOID RULE: if q0 or q1 misses its anchor by more than ANCHOR_TOL, the run is a
BUG, not a verdict. The JSON is written with status VOID and quarantined, and
it still counts as a spent query.

⛔ Weights are fit on TRAIN dumps only. score_noq_test.py's first patch tried to
slice the train period out of a TEST-only frame and its own guard refused;
that guard is reproduced here.

Usage:
  python analysis/score_noq_test42.py --mode equal    # control, run FIRST
  python analysis/score_noq_test42.py --mode query    # THE QUERY
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
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))

from gate_eval import build_merged, L                       # noqa: E402
from phase_c import load_seed_avg                           # noqa: E402
from phase_lever import load_allh                           # noqa: E402
import gate_eval as G                                       # noqa: E402

HD = Path("dhbv_dumps")
THETA, LAM = 4.0, 0.25
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")

ANCHOR_EQUAL = 0.8298
ANCHOR_FROZEN = 0.8362893021622821
ANCHOR_TOL = 0.0005
TARGET = 0.84


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


def boot_ci(s, n=2000, seed=0):
    """Basin bootstrap. REPORTED AS CONTEXT, NEVER AS A PASS BAR — a basin
    bootstrap cannot see seed variance and passed on pure seed noise once."""
    rng = np.random.default_rng(seed)
    v = s.to_numpy(float)
    m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def add_h1(merged, name, paths, label):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{label}: missing {missing}")
    before = len(merged)
    d = load_seed_avg([Path(p) for p in paths])[["station_id", "date", "pred"]]
    merged = merged.merge(d.rename(columns={"pred": name}),
                          on=["station_id", "date"], how="inner")
    if len(merged) < 0.95 * before:
        raise SystemExit(f"{label}: join lost >5% of rows")
    return merged


def build_f1(train, multi_override=None):
    """F1 frame (h==1). multi_override swaps the lstm_multi stream for q3."""
    drop = ("lstm_multi",) if multi_override else ()
    m, cols = build_merged(train=train, drop=drop)
    cols = list(cols)
    if multi_override:
        m = add_h1(m, "lstm_multi", multi_override, "lstm_multi(override)")
        cols += ["lstm_multi"]
    sfx = "_TRAIN" if train else ""
    m = add_h1(m, "lstm_multi5",
               [L / f"camels531ls_multi5_nhlstm{sfx}_{s}.csv.gz" for s in MULTI5],
               "lstm_multi5")
    m = add_h1(m, "lstm_multi6",
               [L / f"camels531ls_multi6_nhlstm{sfx}_{s}.csv.gz" for s in MULTI6],
               "lstm_multi6")
    return m, cols + ["lstm_multi5", "lstm_multi6"]


def build_f2(train, multi_override=None):
    """F2 frame: every lead. dHBV needs the _ALLH_ dumps (B3 on the test side)."""
    streams, names = {}, []
    for f in ("daymet", "nldas", "maurer"):
        streams[f"lstm_{f}"] = load_allh(G._lstm_seeds(f, train))
    streams["lstm_multi"] = load_allh(
        [Path(p) for p in multi_override] if multi_override
        else list(G._multi_seeds(train)))
    sfx = "_TRAIN" if train else ""
    for tag, seeds in (("multi5", MULTI5), ("multi6", MULTI6)):
        streams[f"lstm_{tag}"] = load_allh(
            [L / f"camels531ls_{tag}_nhlstm{sfx}_{s}.csv.gz" for s in seeds])
    for f in ("daymet", "nldas", "maurer"):
        pat = (f"camels531ls_{f}_dhbv_ALLH_TRAIN_s*.csv.gz" if train
               else f"camels531ls_{f}_dhbv_ALLH_s*.csv.gz")
        s = sorted(p for p in HD.glob(pat)
                   if ("TRAIN" in p.name) == bool(train))
        if not s:
            raise SystemExit(f"dhbv_{f}: no ALLH dumps for "
                             f"{'TRAIN' if train else 'TEST'} — F2 UNDEFINED. "
                             "Per-row renormalisation over available streams is "
                             "PROHIBITED (that is a new combination rule).")
        streams[f"dhbv_{f}"] = load_allh(s, shift_days=1)

    merged = None
    for name, df in streams.items():
        d = df.rename(columns={"pred": name})[
            ["station_id", "date", "h", "truth", name]]
        merged = d if merged is None else merged.merge(
            d.drop(columns=["truth"]), on=["station_id", "date", "h"], how="inner")
        names.append(name)
    ids = set(str(x).zfill(8) for x in
              json.load(open("data/camels_gauge_ids.json")).get("531", []))
    merged = merged[merged.station_id.isin(ids)].reset_index(drop=True)
    return merged, names


def fit_frozen_weights(cols, multi_override_train=None):
    """The production vector: inverse-MSE on TRAIN h==1 rows. Never test rows."""
    tr, tcols = build_f1(train=True, multi_override=multi_override_train)
    # Match by NAME, not by position: build_merged and the F2 builder order the
    # nine streams differently, and an order-coupled equality check dies on a
    # frame that is in fact identical. The SET must match exactly — a missing or
    # extra stream is still fatal.
    if set(tcols) != set(cols):
        raise SystemExit(f"stream MISMATCH: TRAIN {sorted(tcols)} vs "
                         f"requested {sorted(cols)}")
    cols = list(cols)
    t = pd.to_datetime(tr["date"])
    if len(tr) == 0 or t.max() >= pd.Timestamp("1995-10-01"):
        raise SystemExit("weight-fit frame is empty or reaches the test window "
                         "— refusing to fit (leakage guard)")
    truth = tr["truth"].to_numpy(float)
    mse = np.array([np.nanmean((tr[c].to_numpy(float) - truth) ** 2) for c in cols])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM * (np.ones(len(cols)) / len(cols)) + (1 - LAM) * raw
    return w / w.sum()


def score(df, cols, w):
    ens = (df[cols].to_numpy(float) * w).sum(axis=1)
    return per_basin_nse(df.assign(_e=ens), "_e")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("equal", "query"), required=True)
    ap.add_argument("--q3-multi", nargs="*", default=[],
                    help="TEST dumps of a SHIPPED Track-A stream (q3 only)")
    ap.add_argument("--q3-multi-train", nargs="*", default=[],
                    help="matching TRAIN dumps, for the weight refit")
    ap.add_argument("--out", default="benchmarks/ledger42_stageQ.json")
    a = ap.parse_args()

    out = {"mode": a.mode, "target": TARGET, "status": "OK", "frames": {}}
    print("=" * 78)
    print(f"LEDGER 42 STAGE Q — mode={a.mode}")
    print("=" * 78)

    m1, cols = build_f1(train=False)
    tt = pd.to_datetime(m1.date)
    print(f"F1: {len(m1):,} rows, {m1.station_id.nunique()} basins, "
          f"{tt.min().date()} -> {tt.max().date()}")

    # ---- q0 CONTROL -------------------------------------------------------
    base = [c for c in cols if c not in ("lstm_multi5", "lstm_multi6")]
    s0 = per_basin_nse(m1.assign(_e=m1[base].mean(axis=1)), "_e")
    q0 = float(s0.median())
    ok0 = abs(q0 - ANCHOR_EQUAL) <= ANCHOR_TOL
    print(f"\nq0 equal-weight 7-stream F1 = {q0:.6f}  "
          f"(anchor {ANCHOR_EQUAL}, diff {q0 - ANCHOR_EQUAL:+.6f})  "
          f"{'OK' if ok0 else 'MISMATCH'}")
    out["q0_equal_F1"] = {"value": q0, "anchor": ANCHOR_EQUAL, "ok": ok0}
    if a.mode == "equal":
        out["status"] = "OK" if ok0 else "VOID (q0 anchor missed)"
        Path(a.out.replace(".json", "_control.json")).write_text(
            json.dumps(out, indent=2))
        print(f"\nCONTROL ONLY — rerun with --mode query to spend the query.")
        return 0 if ok0 else 1

    # ---- q1 THE ANCHOR ----------------------------------------------------
    w = fit_frozen_weights(cols)
    print("\nfrozen weights (fit on TRAIN h==1 rows):")
    for n, x in sorted(zip(cols, w), key=lambda t: -t[1]):
        print(f"   {n:14s} {x:.4f}")
    s1 = score(m1, cols, w)
    q1 = float(s1.median())
    ok1 = abs(q1 - ANCHOR_FROZEN) <= ANCHOR_TOL
    print(f"\nq1 frozen 9-stream F1 = {q1:.6f}  "
          f"(anchor {ANCHOR_FROZEN:.6f}, diff {q1 - ANCHOR_FROZEN:+.6f})  "
          f"{'OK' if ok1 else 'MISMATCH'}")
    out["q1_frozen_F1"] = {"value": q1, "anchor": ANCHOR_FROZEN, "ok": ok1,
                           "ci95": boot_ci(s1), "basins": int(len(s1)),
                           "weights": {n: float(x) for n, x in zip(cols, w)}}

    if not (ok0 and ok1):
        out["status"] = "VOID — an anchor was missed; this is a BUG, not a verdict"
        print(f"\n*** {out['status']} ***\n*** The query is still RECORDED AS "
              "SPENT. Quarantine this JSON. ***")
        Path(a.out.replace(".json", "_VOID.json")).write_text(
            json.dumps(out, indent=2, default=float))
        return 2

    # ---- q2 F2, SAME FROZEN WEIGHTS --------------------------------------
    try:
        m2, names2 = build_f2(train=False)
        if names2 != cols:
            w2 = np.array([dict(zip(cols, w))[n] for n in names2])
        else:
            w2 = w
        s2 = score(m2, names2, w2)
        q2 = float(s2.median())
        print(f"\nq2 frozen 9-stream F2 (all leads) = {q2:.6f}   "
              f"rows {len(m2):,}   (q2 - q1 = {q2 - q1:+.6f})")
        out["q2_frozen_F2"] = {"value": q2, "ci95": boot_ci(s2),
                               "rows": int(len(m2)), "delta_vs_q1": q2 - q1,
                               "note": "F3 == F2 (stride-14 tiles, S0.2)"}
    except SystemExit as e:
        print(f"\nq2 UNDEFINED: {e}")
        out["q2_frozen_F2"] = {"value": None, "undefined_reason": str(e)}
        q2 = None

    # ---- q3 the shipped candidate ----------------------------------------
    q3 = None
    if a.q3_multi:
        m3, c3 = build_f1(train=False, multi_override=[Path(p) for p in a.q3_multi])
        w3 = fit_frozen_weights(c3, multi_override_train=[Path(p) for p in a.q3_multi_train])
        s3 = score(m3, c3, w3)
        q3 = float(s3.median())
        print(f"\nq3 shipped-candidate F1 = {q3:.6f}   "
              f"(vs q1 {q1:.6f}, Δ {q3 - q1:+.6f})")
        out["q3_candidate_F1"] = {"value": q3, "ci95": boot_ci(s3),
                                  "delta_vs_q1": q3 - q1}

    # ---- pre-written branch reading --------------------------------------
    if q3 is not None and q3 >= TARGET:
        branch = ("SKILL CLAIM — the only branch that moves the record. "
                  f"New record {q3:.4f} (F1).")
    elif q3 is not None and q3 > q1:
        branch = (f"Real improvement, target NOT crossed. Record -> {q3:.4f}; "
                  "no 0.84 claim.")
    elif q3 is not None:
        branch = ("The shipped member does NOT transfer held-out. Record "
                  "stands; the train-side contribution pass is recorded as a "
                  "FAILED TRANSFER, stated plainly.")
    elif q2 is None:
        branch = "F2 undefined — q1 only; the frame question stays OPEN."
    elif q2 >= TARGET and q1 < TARGET:
        branch = ("MEASUREMENT FINDING, never a skill gain: scored on every day "
                  "rather than one weekday in fourteen, the SAME frozen model "
                  f"reads {q2:.4f}. Headline remains q1 = {q1:.4f}.")
    elif q2 < q1:
        branch = ("Phase lever CLOSED NEGATIVE held-out: all-leads scores LOWER "
                  f"({q2:.4f} < {q1:.4f}). h==1 was the luckier phase.")
    else:
        branch = (f"Phase lever real but insufficient ({q2:.4f} < {TARGET}); "
                  "closed, spread reported.")
    print(f"\n{'=' * 78}\nPRE-WRITTEN BRANCH READING:\n  {branch}\n{'=' * 78}")
    print(f"REPORTED IN EVERY BRANCH — q1 (F1, h==1) = {q1:.6f}, "
          f"CI95 {out['q1_frozen_F1']['ci95']}")
    print("CIs are context, never pass bars (a basin bootstrap cannot see seed "
          "variance).")
    out["branch_reading"] = branch
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
