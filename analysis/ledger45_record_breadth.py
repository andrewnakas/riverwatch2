#!/usr/bin/env python3
"""LEDGER 45 / T3c — SELF-AUDIT OF THE RECORD, prompted by T3b.

T3b showed a median can be lifted by rank manipulation while most basins get
worse. That makes one question mandatory: IS OUR OWN RECORD BROAD?
Production invmse weights vs EQUAL weights, same 9 streams, same frame, PAIRED.
If breadth < 0.5 our own headline gain is the thing T3b warns about.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "riverwatch2"))
from ledger45_weightest2 import (build_base, cols_of, frames, w_prod, sc_nse,
                                 boot_ci, RECORD)

BTR, BTE = build_base()
ship, _ = cols_of(BTR)
out = {"record": RECORD, "streams": ship}
for fr in ("h1", "all"):
    fit, sc = frames(BTR, BTE, fr)
    wp = w_prod(fit, ship)
    we = np.ones(len(ship)) / len(ship)
    P = sc_nse(sc, ship, wp)
    E = sc_nse(sc, ship, we)
    i = P.index.intersection(E.index)
    d = (P[i] - E[i]).to_numpy()
    lo, hi = boot_ci(d)
    r = {"weighted_median": float(P[i].median()), "equal_median": float(E[i].median()),
         "diff_of_medians": float(P[i].median() - E[i].median()),
         "paired": float(np.median(d)), "breadth": float((d > 0).mean()),
         "boot95": [lo, hi], "n": int(len(i))}
    out[fr] = r
    print(f"[{fr}] weighted {r['weighted_median']:.6f}  equal {r['equal_median']:.6f}")
    print(f"      diff-of-medians {r['diff_of_medians']:+.6f}   PAIRED {r['paired']:+.6f}"
          f"  [{lo:+.6f},{hi:+.6f}]   BREADTH {r['breadth']:.4f}")
    print(f"      => {'BROAD (majority of basins improve)' if r['breadth']>0.5 else '⚠️ NARROW — rank-manipulation shape'}")
json.dump(out, open("benchmarks/ledger45_record_breadth.json", "w"), indent=2, default=float)
print("\nwrote benchmarks/ledger45_record_breadth.json md5 "
      + hashlib.md5(open("benchmarks/ledger45_record_breadth.json","rb").read()).hexdigest())
