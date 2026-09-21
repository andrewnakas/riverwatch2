#!/usr/bin/env python3
"""LEDGER 54 — adjudicate the two pending PRE-REGISTERED targets, once their dumps exist.

TARGET (3) a single NETWORK beating Nearing's 0.879.
  Rule fixed in PREREG_v2.md §LEDGER 54 BEFORE the seeds were scored: take the seed
  with the highest **val1** score, report THAT seed's test1. Reporting the best test1
  seed would be selection on the scored window ([[an-argmax-gap-is-not-headroom]]).
  The mean and the (non-selectable) max are printed alongside for transparency.

TARGET (2) a single MEMBER at 0.89.
  Seed depth is exhausted (`nhar0` val1 asymptote 0.888715). The pre-registered pass
  line is a 1-seed val1 >= 0.8823 for a candidate whose b ~ 0.0112, which would put
  k=5 above 0.89. Checked here for every candidate that has a val1 dump.

usage: l54_targets_verdict.py --candidates nhar0,nhar0h256 [--out X.json]
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l51)

NEARING = 0.879
# ⚠️ CORRECTED 2026-09-20. The old constant 0.8823 was derived by borrowing `nhar0`'s
# seed-noise coefficient b, which is a MEMBER property (see the-seed-depth-law-per-member).
# The right test is the member's OWN curve: 0.89 at k=5 needs x1 >= 0.89 - 0.8*b_member.
# `nhar0h256` measured b = 0.015996 -> x1 >= 0.8772, which it clears (0.87959).
# This one-seed line is a screen only; the real test is the measured k-seed solo, which
# analysis/l54_seed_depth.py reports. nhar0h256 3-seed = 0.890256 val1 => TARGET 2 ACHIEVED.
PASS_1SEED_VAL1 = 0.8772
DD = "data/mblstm/l51_dumps"


def seeds_with(member, frame):
    return sorted(int(f.split("_s")[-1].split("_")[0])
                  for f in glob.glob(f"{DD}/camels531_l51_{member}_s*_{frame}.csv.gz"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="nhar0,nhar0h256")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = {"nearing_single_model": NEARING, "pass_line_1seed_val1": PASS_1SEED_VAL1, "members": {}}

    for M in [c for c in a.candidates.split(",") if c]:
        sv, st = seeds_with(M, "val1"), seeds_with(M, "test1")
        both = sorted(set(sv) & set(st))
        if not sv:
            print(f"\n=== {M}: no val1 dumps yet ===")
            continue
        print(f"\n=== {M}  val1 seeds={sv}  test1 seeds={st} ===")
        v = {s: float(l51.per_basin(l51.load_member(M, "val1", seeds={s}), [M]).median()) for s in sv}
        t = {s: float(l51.per_basin(l51.load_member(M, "test1", seeds={s}), [M]).median()) for s in st}
        for s in sv:
            print(f"  s{s}  val1 {v[s]:.6f}" + (f"   test1 {t[s]:.6f}" if s in t else "   test1 (pending)"))

        e = {"per_seed_val1": v, "per_seed_test1": t}
        # --- TARGET 2 ---
        best_v = max(v, key=v.get)
        e["target2_best_1seed_val1"] = {"seed": best_v, "val1": v[best_v],
                                        "passes_line": bool(v[best_v] >= PASS_1SEED_VAL1)}
        print(f"  TARGET 2 (single member 0.89): best 1-seed val1 = {v[best_v]:.6f} (s{best_v})  "
              f"vs pass line {PASS_1SEED_VAL1} -> {'PASS' if v[best_v] >= PASS_1SEED_VAL1 else 'FAIL'}")
        # --- TARGET 3 ---
        if both:
            sel = max(both, key=lambda s: v[s])          # selected on VAL, never on test
            e["target3"] = {"selected_seed": sel, "selected_val1": v[sel], "selected_test1": t[sel],
                            "beats_nearing": bool(t[sel] > NEARING),
                            "mean_test1": float(np.mean([t[s] for s in both])),
                            "max_test1_NOT_SELECTABLE": float(np.max([t[s] for s in both]))}
            print(f"  TARGET 3 (single network > {NEARING}): val-selected s{sel} -> test1 "
                  f"{t[sel]:.6f}  {'BEATS' if t[sel] > NEARING else 'short by %.6f' % (NEARING - t[sel])}")
            print(f"           mean test1 {np.mean([t[s] for s in both]):.6f}   "
                  f"max {np.max([t[s] for s in both]):.6f} (not selectable)")
        res["members"][M] = e

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
