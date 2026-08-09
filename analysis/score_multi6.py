#!/usr/bin/env python3
"""Score multi6 against the PRE-REGISTERED gate (multi6-prereg-gate memory).

Written BEFORE the result exists, so the verdict cannot be chosen after seeing
the numbers. Runs every gate in the registered order:

  1. MEMBER   -- day-1 median NSE vs multi3 s111 (0.8065), same basin set.
                 Prediction on record: +0.002 .. +0.010.
  2. ENSEMBLE -- added as an extra stream to the 7-stream TRAIN ensemble.
                 Ship threshold: >= +0.001.
  3. CONTROL  -- the same test with a redundant multi3 seed instead. If the
                 control gains as much, the effect is "an 8th stream", not soil
                 moisture. (This is what made multi5's +0.0015 believable:
                 its controls came out NEGATIVE.)
  4. ARID     -- per-basin NSE for aridity>1 reported separately. multi6 exists
                 FOR that cohort; a median gain with a flat arid cohort means it
                 works for a different reason than claimed.
  5. ROBUST   -- per-basin sign test, bootstrap CI, and the gain must hold in
                 BOTH temporal halves of the train window.

The pre-registered FALSIFIER: multi6 ~= 0.000 AND a flat arid cohort would mean
antecedent wetness is already recoverable by the LSTM from its input sequence,
and the "missing physical variable" framing is wrong. That is a publishable
negative -- report it, do not bury it.

TEST QUERY IS NOT SPENT HERE. Everything below is train-side.
"""
import glob
import argparse as _argparse

_ap = _argparse.ArgumentParser(add_help=False)
_ap.add_argument("--seed", default="s111",
                 help="which multi6 seed to score (s111, s222, ...)")
_args, _ = _ap.parse_known_args()
SEED = _args.seed
print(f"\n*** SCORING multi6 SEED: {SEED} ***")

import json
import os
import sys

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
from gate_eval import build_merged, L
from phase_c import load_seed_avg

RNG = np.random.default_rng(0)
CV_SPLIT = "1990-09-30"
D = "gpu1080/dumps"


def per_basin_nse(df, col):
    """NaN-masked per-basin NSE. TRAIN dumps carry ~3.2% NaN truth; without
    masking np.var returns NaN and every basin silently scores NaN."""
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


def load_test(path):
    d = pd.read_csv(path, usecols=["station_id", "t0", "h", "truth", "ymed"])
    d = d[d.h == 1].copy()
    d["station_id"] = d.station_id.astype(str).str.zfill(8)
    return d.rename(columns={"ymed": "pred"})[["station_id", "truth", "pred"]]


print("=" * 78)
print("GATE 1 — MEMBER  (pre-registered prediction: +0.002 .. +0.010)")
print("=" * 78)
m6p = f"{D}/camels531ls_multi6_nhlstm_{SEED}.csv.gz"
if not os.path.exists(m6p):
    sys.exit(f"missing {m6p} — multi6 not dumped yet")
m6 = load_test(m6p)
assert m6.station_id.nunique() == 531, f"multi6 covers {m6.station_id.nunique()} basins"
s6 = per_basin_nse(m6, "pred")

WEAK = ("s333", "s444")
cands = sorted(f for f in glob.glob(f"{D}/camels531ls_multi_nhlstm_s*.csv.gz")
               if "TRAIN" not in f and not any(f.endswith(f"_{w}.csv.gz") for w in WEAK))
print(f"multi6: {len(m6):,} rows, 531 basins, median NSE {s6.median():.4f}\n")

best = None
for f in cands:
    s3 = per_basin_nse(load_test(f), "pred")
    common = sorted(set(s6.index) & set(s3.index))
    assert len(common) > 500, f"thin join vs {f}"
    a, b = s6.reindex(common), s3.reindex(common)
    tag = os.path.basename(f).replace("camels531ls_multi_nhlstm_", "").replace(".csv.gz", "")
    print(f"  vs multi3 {tag:<7} {s3.median():.4f}  same-{len(common)}: "
          f"multi6 {a.median():.4f}  delta {a.median()-b.median():+.4f}")
    if best is None or s3.median() > best[1]:
        best = (tag, s3.median(), b.median(), a.median(), len(common), common)

tag, _, m3c, m6c, n, common = best
dm = m6c - m3c
print(f"\n  VERDICT vs best multi3 seed ({tag}), {n} basins: {dm:+.4f}")
print("  " + ("PREDICTION HELD" if dm >= 0.002 else
              "BELOW PREDICTION — see falsifier" if dm > -0.002 else "NEGATIVE"))

print("\n" + "=" * 78)
print("GATE 4 — ARID COHORT  (this member exists for these basins)")
print("=" * 78)
try:
    a = pd.read_csv("gpu1080/nh_data_multi/attributes/attributes.csv")
    idc = [c for c in a.columns if c.lower() in ("gauge_id", "basin_id", "station_id")][0]
    a[idc] = a[idc].astype(str).str.zfill(8)
    a = a.set_index(idc)
    s3best = per_basin_nse(load_test(f"{D}/camels531ls_multi_nhlstm_{tag}.csv.gz"), "pred")
    for lab, sel in (("arid (aridity>1)", a.index[a.aridity > 1.0]),
                     ("humid (<=1)", a.index[a.aridity <= 1.0]),
                     ("worst decile (multi3 NSE)", s3best.nsmallest(53).index)):
        ids = sorted(set(common) & set(sel))
        if len(ids) > 10:
            x, y = s6.reindex(ids).median(), s3best.reindex(ids).median()
            print(f"  {lab:<26} n={len(ids):>3}  multi6 {x:.4f}  multi3 {y:.4f}  {x-y:+.4f}")
except Exception as e:
    print(f"  (skipped: {e})")

print("\n" + "=" * 78)
print("GATE 2+3 — ENSEMBLE (ship if >= +0.001) and CONTROL")
print("  baseline = 7 streams + multi5 (multi6 must beat what multi5 already gives)")
print("=" * 78)
merged, cols = build_merged(train=True)
# ⚠️ multi5 MUST be in the baseline. optimizer_pass1 measured that multi5's gain
# is concentrated in the 53 worst basins (+0.0078 vs +0.0015 median) -- the SAME
# arid/runoff-generation cohort multi6 targets. Scoring multi6 against a
# multi5-free baseline would credit it for a gain multi5 already delivers, i.e.
# it would look ADDITIVE when it is SUBSTITUTING.
_m5 = [p for p in (L / f"camels531ls_multi5_nhlstm_TRAIN_{s}.csv.gz"
                   for s in ("s111", "s222", "s333", "s444")) if p.exists()]
if _m5:
    _d = load_seed_avg(_m5)[["station_id", "date", "pred"]].rename(
        columns={"pred": "lstm_multi5"})
    merged = merged.merge(_d, on=["station_id", "date"])
    cols = cols + ["lstm_multi5"]
    print(f"\n  baseline INCLUDES multi5 ({len(_m5)} seeds) -- see note above")
else:
    print("\n  ⚠️ multi5 dumps absent; baseline is multi5-FREE and multi6 will "
          "look better than it is")

extras = {"lstm_multi6": L / f"camels531ls_multi6_nhlstm_TRAIN_{SEED}.csv.gz",
          "ctrl_seed":   L / "camels531ls_multi_nhlstm_TRAIN_s444.csv.gz"}
mm = merged.copy()
have = []
for name, p in extras.items():
    if not p.exists():
        print(f"  (skip {name}: {p.name} missing)")
        continue
    d = load_seed_avg([p])[["station_id", "date", "pred"]].rename(columns={"pred": name})
    mm = mm.merge(d, on=["station_id", "date"])
    have.append(name)
assert mm.station_id.nunique() >= 500, "thin ensemble join"
print(f"  rows {len(mm):,}  basins {mm.station_id.nunique()}")

mm["_base"] = mm[cols].mean(axis=1)
base = per_basin_nse(mm, "_base").median()
print(f"\n  {'configuration':<38}{'train medNSE':>14}{'delta':>10}")
print("  " + "-" * 62)
print(f"  {'base (7 streams)':<38}{base:>14.4f}{'--':>10}")
res = {}
for name in have:
    mm["_e"] = mm[cols + [name]].mean(axis=1)
    s = per_basin_nse(mm, "_e").median()
    res[name] = s - base
    lab = {"lstm_multi6": "+multi6 (CANDIDATE)",
           "ctrl_seed": "+multi3 s444 (CONTROL: no new info)"}[name]
    print(f"  {lab:<38}{s:>14.4f}{s-base:>+10.4f}")

if "lstm_multi6" in res:
    g, c = res["lstm_multi6"], res.get("ctrl_seed", -99)
    print(f"\n  ensemble gain {g:+.4f}   control {c:+.4f}")
    ship = g >= 0.001 and g > c + 0.0005
    print("  => " + ("PASSES both gates — worth a test eval (ASK USER FIRST)."
                     if ship else
                     "FAILS: gain below +0.001 or not clearly above the control."))

    print("\n" + "=" * 78)
    print("GATE 5 — ROBUSTNESS")
    print("=" * 78)
    mm["_w"] = mm[cols + ["lstm_multi6"]].mean(axis=1)
    b = per_basin_nse(mm, "_base"); w = per_basin_nse(mm, "_w")
    ids = sorted(set(b.index) & set(w.index))
    d = (w.reindex(ids) - b.reindex(ids)).dropna()
    print(f"  sign test: {(d>0).sum()}/{len(d)} = {(d>0).mean():.1%} basins improved")
    bs = [np.median(w.reindex(ids).to_numpy()[s]) - np.median(b.reindex(ids).to_numpy()[s])
          for s in (RNG.choice(len(ids), len(ids), True) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}]  "
          f"({'excludes zero' if lo > 0 else 'STRADDLES ZERO'})")
    mm["_dt"] = pd.to_datetime(mm.date)
    for lab, sub in (("1980-90", mm[mm._dt <= CV_SPLIT]), ("1990-95", mm[mm._dt > CV_SPLIT])):
        bb = per_basin_nse(sub, "_base"); ww = per_basin_nse(sub, "_w")
        ii = sorted(set(bb.index) & set(ww.index))
        print(f"  {lab}: {bb.reindex(ii).median():.4f} -> {ww.reindex(ii).median():.4f}"
              f"  delta {ww.reindex(ii).median()-bb.reindex(ii).median():+.5f}")
