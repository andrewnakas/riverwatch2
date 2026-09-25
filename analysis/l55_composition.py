#!/usr/bin/env python3
"""LEDGER 55 GATE 0b + 0.4 — composition subsets and the marginal-member ladder.

Two questions, both zero GPU, both on ONE join so every composition is scored on
IDENTICAL rows (the join trap: a 6-member inner join has MORE rows than an
8-member one, so scoring them separately compares different row sets).

GATE 0b — THE SYMMETRIC REMOVAL CHECK.
  The protocol says composition removal is decided on val1. That rule was applied
  to nldash1/maurerh1 (both positive on val1, both kept) and never to the AR pair,
  whose val1 LOO is -0.000055 (nhar) and -0.000009 (nhar0).
  Recorded in advance: on test1 those two contribute +0.000201 and +0.000393, so
  acting on val1 may LOWER the headline. The rule decides, not the headline.

0.4 — THE LADDER. Fit the marginal-member decay from the ACTUAL LOO values rather
  than from three ladder points with zero degrees of freedom, and report the
  implied asymptote of "more members of the same class" with a bootstrap CI.

usage: l55_composition.py --frame val1 [--seeds 501,502,503,504,505] [--out X.json]
"""
import argparse
import importlib.util
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "l51", os.path.join(os.path.dirname(os.path.abspath(__file__)), "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l51)

MB = ["fused3h1", "daymeth1", "nldash1", "maurerh1", "aorch1"]
AR = ["nhar", "nhar0", "nhar0h256"]
ALL8 = MB + AR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=["val1", "test1"])
    ap.add_argument("--seeds", default=None, help="restrict every member to these seeds")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = {int(s) for s in a.seeds.split(",")} if a.seeds else None

    print(f"loading 8 members on one join, frame={a.frame}")
    m = l51.join(ALL8, a.frame, seeds=seeds)
    print(f"joined rows={len(m):,} basins={m.station_id.nunique()}")

    def score(cols):
        return l51.per_basin(m, list(cols))

    base = score(ALL8)
    res = {"frame": a.frame, "rows": int(len(m)), "basins": int(m.station_id.nunique()),
           "seeds": sorted(seeds) if seeds else "all",
           "ensemble_8": float(base.median()), "compositions": {}, "loo": {}}

    # ---- every composition that drops a subset of the two zero-LOO AR members,
    #      plus the same-sized controls that drop MB members instead.
    cand = {
        "ALL8": ALL8,
        "drop_nhar": [c for c in ALL8 if c != "nhar"],
        "drop_nhar0": [c for c in ALL8 if c != "nhar0"],
        "drop_nhar_and_nhar0": [c for c in ALL8 if c not in ("nhar", "nhar0")],
        "drop_nldash1": [c for c in ALL8 if c != "nldash1"],
        "drop_maurerh1": [c for c in ALL8 if c != "maurerh1"],
        "drop_two_weakest_MB": [c for c in ALL8 if c not in ("nldash1", "aorch1")],
    }
    print(f"\n{'composition':<24} {'n':>2} {'median':>10} {'vs ALL8':>11} "
          f"{'CI95':>26} {'breadth':>8}")
    for name, cols in cand.items():
        s = score(cols)
        d, lo, hi, br, nb = l51.paired_ci(s, base)
        res["compositions"][name] = {"members": cols, "n": len(cols),
                                     "median": float(s.median()), "paired_vs_all8": d,
                                     "ci": [lo, hi], "breadth": br, "n_basins": nb}
        print(f"{name:<24} {len(cols):>2} {s.median():>10.6f} {d:>+11.6f} "
              f"[{lo:>+.6f},{hi:>+.6f}] {br:>8.3f}")

    # ---- LOO for every member on this fixed join
    print(f"\n{'member':<12} {'solo':>10} {'LOO':>11} {'CI95':>26} {'breadth':>8}")
    for c in ALL8:
        solo = score([c])
        without = score([x for x in ALL8 if x != c])
        d, lo, hi, br, nb = l51.paired_ci(base, without)
        res["loo"][c] = {"solo": float(solo.median()), "delta": d, "ci": [lo, hi],
                         "breadth": br}
        print(f"{c:<12} {solo.median():>10.6f} {d:>+11.6f} "
              f"[{lo:>+.6f},{hi:>+.6f}] {br:>8.3f}")

    # ---- 0.4 the ladder: what does "one more member of the same class" buy?
    # Build the ladder empirically: add members one at a time in the order they
    # were actually added, scoring every prefix on the SAME rows.
    order = MB + ["nhar", "nhar0", "nhar0h256"]
    prefix, inc = [], []
    print(f"\n{'ladder step':<28} {'median':>10} {'increment':>11}")
    prev = None
    for c in order:
        prefix.append(c)
        if len(prefix) < 5:
            continue
        v = float(score(prefix).median())
        if prev is not None:
            inc.append(v - prev)
            print(f"+{c:<27} {v:>10.6f} {v - prev:>+11.6f}")
        else:
            print(f"{'base5':<28} {v:>10.6f} {'-':>11}")
        prev = v
    res["ladder_increments"] = inc
    if len(inc) >= 2:
        ratios = [inc[i + 1] / inc[i] for i in range(len(inc) - 1) if inc[i] != 0]
        r = float(np.mean(ratios)) if ratios else float("nan")
        remaining = inc[-1] * r / (1 - r) if 0 < r < 1 else float("inf")
        res["ladder"] = {"ratios": ratios, "r": r, "sum_remaining": float(remaining),
                         "asymptote": float(prev + remaining),
                         "r_required_for_0.90": float((0.90 - prev) / (inc[-1] + (0.90 - prev)))}
        print(f"\nratios {['%.3f' % x for x in ratios]}  mean r = {r:.3f}")
        print(f"sum of ALL remaining same-class members = {remaining:+.6f}")
        print(f"=> asymptote {prev + remaining:.6f}   (0.90 needs r >= "
              f"{res['ladder']['r_required_for_0.90']:.3f})")
        print("WARNING: this fit has as many parameters as data points. It is a "
              "direction, not a measurement.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
