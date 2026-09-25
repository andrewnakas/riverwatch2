#!/usr/bin/env python3
"""LEDGER 44 / S1b-B — THE TAKE-ALL CONFIGURATION ON BOTH FRAMES, WITH CIs.

AMENDMENT 1 requires >= +0.003 on BOTH h==1 and all-leads, signs agreeing.
ledger44_breadth_curve.py measured h==1 only (+0.002546 same-frame, D-PARTIAL).
This adds the all-leads frame and a bootstrap CI on each, which is what turns a
point estimate into a claim or refuses to.

⛔ The stream list is FIXED by AMENDMENT 1's train-side solo screen and is NOT
re-derived here. `multi14` was TRAIN-only when the pool was scanned, so it is
NOT in the pre-registered take-all; it is reported separately and labelled as a
declared deviation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "riverwatch2"))
import gate_eval as G                                              # noqa: E402
from phase_lever import load_allh                                  # noqa: E402
from rescore9 import glob_seeds                                    # noqa: E402
from ledger44_gate_transfer import (build_streams, merge, score,   # noqa: E402
                                    per_basin_nse, TEST_END)

# fixed by AMENDMENT 1's train-side solo screen, in its train-side order
KEPT = ["multih512", "multi9", "multi8", "multi5b", "multi730", "multi671",
        "multi_eps05", "multi10", "multidrop", "multih128", "aorc"]
RECORD = 0.8362893021622821


def add(merged, tag, train):
    t = "TRAIN_" if train else ""
    ps = glob_seeds(f"camels531ls_{tag}_nhlstm_{t}s*.csv.gz", G.L)
    if not ps:
        raise SystemExit(f"{tag}: no dumps (train={train})")
    d = load_allh(ps)[["station_id", "date", "h", "pred"]].rename(
        columns={"pred": f"pool_{tag}"})
    return merged.merge(d, on=["station_id", "date", "h"], how="inner")


def boot(delta, n=4000, seed=20260827):
    rng = np.random.default_rng(seed)
    d = np.asarray(delta, float)
    bs = np.array([np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(n)])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-multi14", action="store_true",
                    help="declared deviation: multi14 was TRAIN-only at pre-registration")
    ap.add_argument("--drop-aorc", action="store_true",
                    help="AMENDMENT 2: drop aorc for FRAME COVERAGE (not score) — "
                         "it is the only kept stream that truncates the test window")
    ap.add_argument("--out", default="benchmarks/ledger44_breadth_allleads.json")
    args = ap.parse_args()

    pool = [t for t in KEPT if not (args.drop_aorc and t == "aorc")]
    pool = pool + (["multi14"] if args.with_multi14 else [])
    print(f"pool ({len(pool)}): {pool}", file=sys.stderr)

    print("building TRAIN (all leads)...", file=sys.stderr)
    TR = merge(build_streams(True))
    print("building TEST (all leads)...", file=sys.stderr)
    TE = merge(build_streams(False))
    for tag in pool:
        TR = add(TR, tag, True)
        TE = add(TE, tag, False)
    print(f"  TRAIN {len(TR):,} rows / {TR.station_id.nunique()} basins", file=sys.stderr)
    print(f"  TEST  {len(TE):,} rows / {TE.station_id.nunique()} basins", file=sys.stderr)

    ship = [c for c in TR.columns if c not in ("station_id", "date", "h", "truth")
            and not c.startswith("pool_")]
    allc = ship + [f"pool_{t}" for t in pool]

    out = {"pool": pool, "n_shipped": len(ship), "n_all": len(allc),
           "drop_aorc": args.drop_aorc,
           "with_multi14": args.with_multi14,
           "test_rows": int(len(TE)), "test_basins": int(TE.station_id.nunique())}
    print("\n" + "=" * 78)
    print(f"TAKE-ALL ({len(allc)} streams) vs SHIPPED ({len(ship)}), SAME FRAME")
    print("=" * 78)
    for frame in ("h1", "all"):
        b, _ = score(TR, TE, ship, frame, "test")
        c, _ = score(TR, TE, allc, frame, "test")
        idx = b.index.intersection(c.index)
        d = (c[idx] - b[idx])
        lo, hi = boot(d.to_numpy())
        rec = {"shipped": float(b[idx].median()), "take_all": float(c[idx].median()),
               "paired_delta": float(d.median()),
               "diff_of_medians": float(c[idx].median() - b[idx].median()),
               "breadth": float((d > 0).mean()), "n_basins": int(len(idx)),
               "boot95": [lo, hi]}
        out[frame] = rec
        print(f"\n  [{frame}]  n={len(idx)} basins")
        print(f"    shipped {rec['shipped']:.6f}   take-all {rec['take_all']:.6f}")
        print(f"    PAIRED DELTA {rec['paired_delta']:+.6f}   "
              f"boot95 [{lo:+.6f}, {hi:+.6f}]")
        print(f"    diff-of-medians {rec['diff_of_medians']:+.6f}   "
              f"breadth {rec['breadth']:.4f}")
        excl0 = lo > 0
        print(f"    CI excludes zero: {excl0}     clears +0.003: {lo >= 0.003}")

    both = min(out["h1"]["paired_delta"], out["all"]["paired_delta"])
    signs = np.sign(out["h1"]["paired_delta"]) == np.sign(out["all"]["paired_delta"])
    if both >= 0.003 and signs:
        v = "D-PASS — a defensible claim on both frames"
    elif both > 0 and signs:
        v = "D-PARTIAL — positive on both frames, sub-bar; measured, NOT claimed as skill"
    else:
        v = "D-FAIL"
    out["verdict"] = v
    print(f"\n  => {v}")
    print(f"  (worst frame {both:+.6f} vs bar +0.003; signs agree: {bool(signs)})")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
