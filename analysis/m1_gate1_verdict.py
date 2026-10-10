#!/usr/bin/env python
"""GATE 1 verdict — what Open-Meteo POINT forcing costs `nldasm9`.

Registered in PREREG_v2.md §GATE 1 before any number was produced: statistic,
sample (150 cohort basins, seed 0), band (-0.08 .. -0.01), STOP (-0.12) and the
decision table are all fixed there. This script only reads them out.

Both arms are the SAME checkpoints and the SAME rows; only the forcing corpus
differs, which is why the two dumps can be joined per (station_id, t0) and
differenced per basin. Reuses `analysis/l51_withq_score.py` so the scoring is
literally the record's scoring, including its provenance guard.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).parent

spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l51)

# Fixed in the prereg. Read, never recomputed here.
BAND = (-0.08, -0.01)
STOP = -0.12
DECISIONS = [
    (-0.03, None, "SERVE FROM OPEN-METEO — no near-real-time areal pipeline needed; "
                  "ship the member with its measured discount stated"),
    (-0.12, -0.03, "SERVE, LABELLED — ship on point forcing but label the member with its "
                   "measured as-served skill, and schedule the areal provider"),
    (None, -0.12, "DO NOT SHIP ON POINT FORCING — build the areal provider first"),
]


def decide(delta: float) -> str:
    if delta >= -0.03:
        return DECISIONS[0][2]
    if delta > -0.12:
        return DECISIONS[1][2]
    return DECISIONS[2][2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="val1")
    ap.add_argument("--member", default="nldasm9")
    ap.add_argument("--areal-prefix", default="g1nl_")
    ap.add_argument("--point-prefix", default="g1om_")
    ap.add_argument("--seeds", default="501,502,503,504,505")
    ap.add_argument("--protocol", default="modern-v3-9yr")
    ap.add_argument("--out", default="benchmarks/m1_gate1_verdict.json")
    a = ap.parse_args()
    os.chdir(ROOT)

    # INTs, not strings: l51.load_member filters with
    #   int(f.split("_s")[-1].split("_")[0]) in seeds
    # so a list of strings matches NOTHING and it exits "no dumps for member".
    # Every other caller passes ints (l51_withq_score.py:197, l53_arfamily_map.py:71).
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    areal = l51.load_member(a.member, a.frame, seeds=seeds, protocol=a.protocol,
                            prefix=a.areal_prefix)
    point = l51.load_member(a.member, a.frame, seeds=seeds, protocol=a.protocol,
                            prefix=a.point_prefix)
    areal = areal.rename(columns={a.member: "areal"})
    point = point.rename(columns={a.member: "point"})
    m = areal.merge(point, on=["station_id", "t0"], how="inner")
    if m.empty:
        print("ABORT: the two dumps share no (station_id, t0) rows")
        return 1
    # Both arms carry their own `truth`, so the merge suffixes them to
    # truth_x/truth_y and l51.per_basin -- which reads df.truth -- raises.
    #
    # They are the same observations, but NOT bit-identical: the dumps store
    # discharge as float32 and the two arms reach it by different arithmetic
    # paths, so 43% of rows differ by a few float32 ULPs (1569.999878 vs
    # 1570.0, 73.000031 vs 73.0). Measured bound over 217,153 rows: max
    # |diff| = 0.015624 cfs, against a median flow of 61.3 and a max of
    # 67,700 -- i.e. 2 ULPs at the top of the range, hydrologically nothing.
    # An exact-equality assert is therefore TIGHTER THAN THE STORED PRECISION
    # and takes the wrong branch; the tolerance is tied to float32 spacing at
    # the data's own magnitude instead.
    #
    # Then score BOTH arms against ONE truth. The choice cannot matter at
    # 0.0156 cfs, but it makes the contrast strictly one-variable (forcing)
    # and the delta exactly reproducible.
    import numpy as _np
    _ta = m.truth_x.to_numpy(float)
    _tp = m.truth_y.to_numpy(float)
    _d = _np.abs(_ta - _tp)
    _dmax = float(_np.nanmax(_d))
    _tol = 4.0 * float(_np.spacing(_np.float32(_np.nanmax(_np.abs(_ta)))))
    if _dmax > _tol:
        print(f"ABORT: the two arms disagree on truth beyond float32 precision "
              f"(max |diff| = {_dmax:.6g} > tol {_tol:.6g}) -- that is a target "
              f"revision or a bad join, not storage noise")
        return 1
    print(f"  truth agreement: max |diff| = {_dmax:.6g} cfs (tol {_tol:.6g}, "
          f"float32 storage noise); scoring both arms on the areal arm's truth")
    m = m.drop(columns=["truth_y"]).rename(columns={"truth_x": "truth"})

    nse_a = l51.per_basin(m, ["areal"])
    nse_p = l51.per_basin(m, ["point"])
    common = sorted(set(nse_a.index) & set(nse_p.index))
    # l51.per_basin returns a pd.Series indexed by station_id (not a frame),
    # so there is no column axis to index -- .loc[common, "areal"] raises
    # "Too many indexers".
    sa = nse_a.loc[common]
    sp = nse_p.loc[common]
    delta, lo, hi, breadth, n = l51.paired_ci(sp, sa)      # point MINUS areal

    med_a, med_p = float(sa.median()), float(sp.median())
    in_band = BAND[0] <= delta <= BAND[1]
    verdict = decide(delta)
    falsified = delta > 0

    out = {
        "_note": ("GATE 1: the cost of serving nldasm9 on Open-Meteo point forcing instead "
                  "of the NLDAS-2 areal forcing it was trained on. Same checkpoints, same "
                  "rows, forcing is the only difference. Registered in PREREG_v2.md §GATE 1."),
        "frame": a.frame, "member": a.member, "seeds": seeds, "protocol": a.protocol,
        "rows_joined": int(len(m)), "basins": int(n),
        "median_nse_areal_trained_on": med_a,
        "median_nse_point_as_served": med_p,
        "paired_delta_point_minus_areal": delta,
        "ci95": [lo, hi], "breadth": breadth,
        "registered_band": list(BAND), "registered_stop": STOP,
        "in_registered_band": bool(in_band),
        "verdict": verdict,
        "my_framing_falsified": bool(falsified),
        "falsified_note": ("Delta is POSITIVE: point forcing beat the areal corpus the model "
                           "trained on. Either the corpus swap is not doing what I think, or "
                           "screen B0's coupling result does not transfer to NSE. Label this "
                           "unexplained; do not bank it as good news."
                           if falsified else ""),
    }
    Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"GATE 1 — {a.member}, {a.frame}, {n} basins, {len(m)} rows joined")
    print(f"  median NSE, NLDAS-2 areal (trained on) : {med_a:.6f}")
    print(f"  median NSE, Open-Meteo point (served)  : {med_p:.6f}")
    print(f"  PAIRED delta (point - areal)           : {delta:+.6f}  "
          f"[{lo:+.6f}, {hi:+.6f}]  breadth {breadth:.3f}")
    print(f"  registered band {BAND[0]:+.3f}..{BAND[1]:+.3f}  "
          f"=> {'IN BAND' if in_band else 'OUTSIDE BAND'}")
    print(f"  VERDICT: {verdict}")
    if falsified:
        print(f"  ⚠️ {out['falsified_note']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
