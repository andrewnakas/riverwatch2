#!/usr/bin/env python3
"""LEDGER 53 — adjudicate the pre-registered compositions with the PAIRED statistic.

PREREG_v2.md §LEDGER 53 fixes three candidates and says the choice is made on
**val1** by paired Δ (never difference-of-medians — that has disagreed with the
paired test six times across ledgers 51-52 and lost every time):

  base5 = fused3h1, daymeth1, nldash1, maurerh1, aorch1      (the 0.888355 record)
  A     = base5 + nhar, equal weight
  B     = nhar REPLACES aorch1 (whose LOO was -0.000008)
  C     = base5 + nhar at 2x weight

All four are scored on the SAME ROWS (the 6-member inner join), because a
comparison is only a comparison if both sides are evaluated on the same rows
([[a-null-must-be-computed-on-the-same-rows]]).

usage: l53_composition.py --frame val1|test1 [--nhar-seeds 501,502,503] --out X.json
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l51)          # chdir's to ~/riverwatch2

FIVE = ["fused3h1", "daymeth1", "nldash1", "maurerh1", "aorch1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=["val1", "test1"])
    ap.add_argument("--nhar-seeds", default="", help="restrict nhar seed depth, e.g. 501,502,503")
    ap.add_argument("--new", default="nhar", help="the new member")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    N = a.new
    seeds = set(int(x) for x in a.nhar_seeds.split(",") if x) or None
    print(f"LEDGER 53 composition — frame={a.frame} new={N} "
          f"{'seeds=' + str(sorted(seeds)) if seeds else '(all seeds)'}")
    # per-member load so the new member can have its own seed restriction
    parts = [l51.load_member(x, a.frame) for x in FIVE]
    parts.append(l51.load_member(N, a.frame, seeds=seeds))
    m = parts[0]
    for d in parts[1:]:
        m = m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    print(f"joined rows {len(m):,}  basins {m.station_id.nunique()}  "
          f"(all compositions scored on THESE rows)\n")

    W2 = {**{k: 1.0 for k in FIVE}, N: 2.0}
    comps = {
        "base5": (FIVE, None),
        "A": (FIVE + [N], None),
        "B": ([x for x in FIVE if x != "aorch1"] + [N], None),
        "C": (FIVE + [N], W2),
    }
    per = {k: l51.per_basin(m, cols, w) for k, (cols, w) in comps.items()}
    res = {"frame": a.frame, "new_member": N, "nhar_seeds": sorted(seeds) if seeds else "all",
           "rows": int(len(m)), "basins": int(m.station_id.nunique()),
           "median": {k: float(v.median()) for k, v in per.items()},
           "solo": {x: float(l51.per_basin(m, [x]).median()) for x in FIVE + [N]}}
    print("MEDIAN NSE")
    for k, v in res["median"].items():
        print(f"  {k:<6} {v:.6f}")
    print("\nSOLO")
    for k, v in res["solo"].items():
        print(f"  {k:<10} {v:.6f}")

    print("\nPAIRED COMPARISONS (median of per-basin differences, basin bootstrap 95% CI, breadth)")
    pairs = [("A", "base5"), ("B", "base5"), ("C", "base5"), ("C", "A"), ("B", "A"), ("A", "B")]
    res["paired"] = {}
    for x, y in pairs:
        d, lo, hi, br, n = l51.paired_ci(per[x], per[y])
        sig = "CI excludes 0" if (lo > 0 or hi < 0) else "CI straddles 0"
        res["paired"][f"{x}_vs_{y}"] = {"paired_delta": d, "ci": [lo, hi], "breadth": br,
                                        "median_delta": float(per[x].median() - per[y].median()),
                                        "significant": bool(lo > 0 or hi < 0), "n_basins": n}
        print(f"  {x:>5} vs {y:<6} median {per[x].median()-per[y].median():+.6f}   "
              f"paired {d:+.6f}  CI[{lo:+.6f},{hi:+.6f}]  breadth {br:.3f}  {sig}")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
