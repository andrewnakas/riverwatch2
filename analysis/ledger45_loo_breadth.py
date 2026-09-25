#!/usr/bin/env python3
"""LEDGER 45 / T3d — LEAVE-ONE-OUT BREADTH AUDIT OF THE SHIPPED 9.

T3b showed a median can move without most basins improving. T3c cleared the
ensemble as a whole. This asks the same question of EACH shipped member: is its
contribution BROAD, or is it moving the rank statistic narrowly?
LOO: full 9 vs the same rule refit on the other 8, paired, with breadth.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "riverwatch2"))
from ledger45_weightest2 import (build_base, cols_of, frames, w_prod, sc_nse,
                                 boot_ci, RECORD)

BTR, BTE = build_base()
ship, _ = cols_of(BTR)
out = {"record": RECORD, "streams": ship, "note":
       "LOO contribution of each shipped stream: full-9 minus refit-on-8, paired."}
for fr in ("h1", "all"):
    fit, sc = frames(BTR, BTE, fr)
    full = sc_nse(sc, ship, w_prod(fit, ship))
    rows = {}
    print(f"\n[{fr}] full-9 median {float(full.median()):.6f}")
    print(f"  {'dropped stream':<16} {'paired':>10} {'breadth':>8}  {'boot95':>26}")
    for s in ship:
        sub = [c for c in ship if c != s]
        cand = sc_nse(sc, sub, w_prod(fit, sub))
        i = full.index.intersection(cand.index)
        d = (full[i] - cand[i]).to_numpy()      # >0 => the stream HELPS
        lo, hi = boot_ci(d)
        rows[s] = {"paired_contribution": float(np.median(d)),
                   "breadth": float((d > 0).mean()), "boot95": [lo, hi],
                   "median_without": float(cand[i].median())}
        flag = "" if (d > 0).mean() > 0.5 else "  <-- NARROW"
        print(f"  {s:<16} {np.median(d):>+10.6f} {(d>0).mean():>8.3f}  "
              f"[{lo:+.6f},{hi:+.6f}]{flag}")
    out[fr] = rows
json.dump(out, open("benchmarks/ledger45_loo_breadth.json", "w"), indent=2, default=float)
print("\nwrote benchmarks/ledger45_loo_breadth.json md5 "
      + hashlib.md5(open("benchmarks/ledger45_loo_breadth.json","rb").read()).hexdigest())
