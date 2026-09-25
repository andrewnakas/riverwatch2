#!/usr/bin/env python3
"""LEDGER 42 STAGE Q — SECOND READ. User-authorised 2026-08-23 after the first
was VOID. Both reads are recorded; neither is discarded.

WHAT WENT WRONG THE FIRST TIME
------------------------------
q0 was checked against **0.8298**, which is the 7-stream equal-weight mean on
the **7-STREAM frame**. My code computed the 7-stream mean on the **9-STREAM
frame** (build_f1 inner-joins multi5/multi6, a different row set) and got
0.830512. The number was fine; **the anchor belonged to a different object.**
The run then VOIDed and **aborted before computing q2 — the only thing it was
spent for.**

TWO DESIGN FIXES
----------------
1. ⭐ **A validity check gates INTERPRETATION, not EXECUTION.** Everything is
   computed unconditionally; controls set a `valid` flag. A broken control must
   never again destroy the measurement.
2. ⭐ **Every anchor names the COMPUTATION that produced it**, and the code
   performs that same computation on that same frame. An anchor is a claim
   about an object, not a number to match.

THE THREE EQUAL-WEIGHT OBJECTS (the confusion that caused the VOID)
  0.829800  7-stream equal-weight on the 7-STREAM frame  (build_merged only)
  0.833765  9-stream equal-weight on the 9-STREAM frame  (noq_test_result.json)
  0.830512  7-stream mean on the 9-stream frame          (what q0 wrongly did)

QUANTITIES (all computed, always)
  c1  7-stream equal on the 7-stream frame   anchor 0.829800   (tol 5e-4)
  c2  9-stream equal on the 9-stream frame   anchor 0.833765   (tol 5e-4)
  q1  frozen 9-stream, F1 (h==1)             anchor 0.836289   (tol 5e-4)
  q2  frozen 9-stream, F2 (all leads, SAME frozen weight vector)  <- THE TARGET

⛔ q2 uses the F1 weight vector remapped BY NAME. F2 is a SAMPLE change, never a
configuration change (the B1 finding: refitting on all-leads rows is a different
ensemble that misses the anchor by -0.0038).

Branch readings are pre-written in benchmarks/ledger42_branch_table.md and are
unchanged by this re-run.
"""
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
from score_noq_test42 import (build_f1, build_f2, fit_frozen_weights,  # noqa: E402
                              per_basin_nse, boot_ci, score, TARGET)

A_C1, A_C2, A_Q1, TOL = 0.8298, 0.8337646534246352, 0.8362893021622821, 0.0005


def main():
    out = {"read": 2, "authorised": "user 2026-08-23", "target": TARGET,
           "quantities": {}, "controls": {}}
    print("=" * 78)
    print("LEDGER 42 STAGE Q — SECOND READ (recorded)")
    print("=" * 78)

    # ---- c1: 7-stream equal on the 7-STREAM frame (the object 0.8298 names)
    m7, c7 = build_merged(train=False)
    s = per_basin_nse(m7.assign(_e=m7[c7].mean(axis=1)), "_e")
    c1 = float(s.median())
    ok1 = abs(c1 - A_C1) <= TOL
    print(f"\nc1  7-stream equal / 7-stream frame = {c1:.6f}   "
          f"anchor {A_C1}  diff {c1 - A_C1:+.6f}  {'OK' if ok1 else 'MISMATCH'}"
          f"   (rows {len(m7):,})")
    out["controls"]["c1_7stream_on_7frame"] = {"value": c1, "anchor": A_C1,
                                               "ok": ok1, "rows": int(len(m7))}

    # ---- the 9-stream frame ------------------------------------------------
    m1, cols = build_f1(train=False)
    print(f"\nF1 frame: {len(m1):,} rows, {m1.station_id.nunique()} basins")

    # ---- c2: 9-stream equal on the 9-STREAM frame -------------------------
    s = per_basin_nse(m1.assign(_e=m1[cols].mean(axis=1)), "_e")
    c2 = float(s.median())
    ok2 = abs(c2 - A_C2) <= TOL
    print(f"c2  9-stream equal / 9-stream frame = {c2:.6f}   "
          f"anchor {A_C2:.6f}  diff {c2 - A_C2:+.6f}  {'OK' if ok2 else 'MISMATCH'}")
    out["controls"]["c2_9stream_on_9frame"] = {"value": c2, "anchor": A_C2, "ok": ok2}

    # ---- q1: the record ----------------------------------------------------
    w = fit_frozen_weights(cols)
    s1 = score(m1, cols, w)
    q1 = float(s1.median())
    okq1 = abs(q1 - A_Q1) <= TOL
    print(f"\nq1  frozen 9-stream F1            = {q1:.6f}   "
          f"anchor {A_Q1:.6f}  diff {q1 - A_Q1:+.6f}  {'OK' if okq1 else 'MISMATCH'}")
    ci1 = boot_ci(s1)
    out["quantities"]["q1_frozen_F1"] = {"value": q1, "anchor": A_Q1, "ok": okq1,
                                         "ci95": ci1, "basins": int(len(s1))}

    # ---- q2: THE TARGET — computed unconditionally -------------------------
    q2 = None
    try:
        m2, names2 = build_f2(train=False)
        wmap = dict(zip(cols, w))
        w2 = np.array([wmap[n] for n in names2])
        s2 = score(m2, names2, w2)
        q2 = float(s2.median())
        print(f"q2  frozen 9-stream F2 (ALL LEADS) = {q2:.6f}   "
              f"rows {len(m2):,}   q2-q1 = {q2 - q1:+.6f}")
        out["quantities"]["q2_frozen_F2"] = {
            "value": q2, "ci95": boot_ci(s2), "rows": int(len(m2)),
            "delta_vs_q1": q2 - q1, "note": "F3 == F2 (stride-14 tiles, S0.2)"}
    except SystemExit as e:
        print(f"q2  UNDEFINED: {e}")
        out["quantities"]["q2_frozen_F2"] = {"value": None, "reason": str(e)}

    # ---- validity gates INTERPRETATION, not execution ----------------------
    valid = ok1 and ok2 and okq1
    out["valid"] = bool(valid)
    print(f"\n{'=' * 78}")
    print(f"VALIDITY: {'ALL CONTROLS PASS — numbers are interpretable' if valid else 'A CONTROL FAILED — see flags'}")
    print("=" * 78)

    if q2 is not None:
        if q2 >= TARGET and q1 < TARGET:
            branch = ("MEASUREMENT FINDING, never a skill gain: scored on every "
                      "day rather than one weekday in fourteen, the SAME frozen "
                      f"model reads {q2:.4f}. Headline remains q1 = {q1:.4f}.")
        elif q2 < q1:
            branch = ("Phase lever CLOSED NEGATIVE held-out: all-leads scores "
                      f"LOWER ({q2:.4f} < {q1:.4f}). h==1 was the luckier phase.")
        else:
            branch = (f"Phase lever real but insufficient ({q2:.4f} < {TARGET}); "
                      "closed, spread reported.")
    else:
        branch = "F2 undefined — q1 only; the frame question stays OPEN."
    print(f"\nPRE-WRITTEN BRANCH READING:\n  {branch}\n")
    print(f"REPORTED IN EVERY BRANCH — q1 (F1, h==1) = {q1:.6f}, CI95 {ci1}")
    print("CIs are context, never pass bars.")
    out["branch_reading"] = branch

    Path("benchmarks/ledger42_stageQ_read2.json").write_text(
        json.dumps(out, indent=2, default=float))
    print("\nwrote benchmarks/ledger42_stageQ_read2.json")


if __name__ == "__main__":
    sys.exit(main())
