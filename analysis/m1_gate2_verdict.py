#!/usr/bin/env python
"""GATE 2 verdict — what the 5-day NLDAS-2 LATENCY GAP costs `nldasm9`.

Registered in PREREG_v2.md #GATE 2 before any number was produced: statistic,
sample, band (-0.020..-0.002), STOP (-0.030), the decision table and a
falsifier are all fixed there. This script only reads them out.

Control is the EXISTING GATE 1 areal arm (`g1nl_`) -- no new baseline is
produced, so the control cannot drift. Treatment (`g2sp4_`) is the same
checkpoints and rows with the newest 4 encoder forcing days replaced by
Open-Meteo point values, per window, through the same scaler.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).parent
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)

BAND = (-0.020, -0.002)
STOP = -0.030
GATE1_FULL = -0.059534          # substituting ALL 365 days, for context only


def decide(d: float) -> str:
    if d > -0.010:
        return ("BUILD THE AREAL PROVIDER -- point-filling the newest 4 days costs less than a "
                "sixth of GATE 1's full substitution, so areal serving recovers most of -0.059534")
    if d > -0.030:
        return ("BUILD IT, BUT RE-GATE THE FILL -- the newest days need a better source than "
                "Open-Meteo point (e.g. the provider's own short-range GFS analysis)")
    return ("DO NOT BUILD IT ON THIS DESIGN -- the newest days dominate, so areal-with-point-fill "
            "recovers little of GATE 1's loss; serve on point with the label or find a "
            "low-latency areal source")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="val1")
    ap.add_argument("--member", default="nldasm9")
    ap.add_argument("--control-prefix", default="g1nl_")
    ap.add_argument("--treat-prefix", default="g2sp4_")
    ap.add_argument("--splice-days", type=int, default=4)
    ap.add_argument("--seeds", default="501,502,503,504,505")
    ap.add_argument("--protocol", default="modern-v3-9yr")
    ap.add_argument("--out", default="benchmarks/m1_gate2_verdict.json")
    a = ap.parse_args()
    os.chdir(ROOT)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    ctl = l51.load_member(a.member, a.frame, seeds=seeds, protocol=a.protocol,
                          prefix=a.control_prefix).rename(columns={a.member: "areal"})
    trt = l51.load_member(a.member, a.frame, seeds=seeds, protocol=a.protocol,
                          prefix=a.treat_prefix).rename(columns={a.member: "spliced"})
    m = ctl.merge(trt, on=["station_id", "t0"], how="inner")
    if m.empty:
        print("ABORT: the two dumps share no (station_id, t0) rows"); return 1

    import numpy as np
    _ta, _tp = m.truth_x.to_numpy(float), m.truth_y.to_numpy(float)
    _dmax = float(np.nanmax(np.abs(_ta - _tp)))
    _tol = 4.0 * float(np.spacing(np.float32(np.nanmax(np.abs(_ta)))))
    if _dmax > _tol:
        print(f"ABORT: arms disagree on truth beyond float32 precision "
              f"({_dmax:.6g} > {_tol:.6g})"); return 1
    m = m.drop(columns=["truth_y"]).rename(columns={"truth_x": "truth"})

    # Pre-committed no-op guard: a splice that did nothing reads 0.000000.
    moved = int((m.areal.to_numpy(float) != m.spliced.to_numpy(float)).sum())
    if moved == 0:
        print("ABORT: the two arms are IDENTICAL -- the splice is a silent no-op"); return 1

    nse_c = l51.per_basin(m, ["areal"])
    nse_t = l51.per_basin(m, ["spliced"])
    common = sorted(set(nse_c.index) & set(nse_t.index))
    sc, stt = nse_c.loc[common], nse_t.loc[common]
    delta, lo, hi, breadth, n = l51.paired_ci(stt, sc)        # spliced MINUS areal
    med_c, med_t = float(sc.median()), float(stt.median())
    in_band = BAND[0] <= delta <= BAND[1]
    recovered = 1.0 - (delta / GATE1_FULL) if GATE1_FULL else float("nan")
    # ⚠️ Report BOTH statistics. The paired median is the registered decision
    # statistic and is the right question for "does the gap hurt a basin". But
    # the difference of MEDIANS is what a published cohort headline moves by,
    # and here the two disagree by 9x: the typical basin barely moves while a
    # left tail (24 of 150 basins worse than -0.05) drags the cohort median
    # level down. Quoting only the paired number would overstate the recovery.
    dom = med_t - med_c
    GATE1_DOM = 0.808659 - 0.893695        # GATE 1's own difference of medians
    recovered_dom = 1.0 - (dom / GATE1_DOM) if GATE1_DOM else float("nan")
    import numpy as _np
    _d = (stt - sc).to_numpy(float)
    tail = {"n_worse_than_-0.05": int((_d < -0.05).sum()),
            "n_worse_than_-0.20": int((_d < -0.20).sum()),
            "p05": float(_np.percentile(_d, 5)), "p25": float(_np.percentile(_d, 25)),
            "p75": float(_np.percentile(_d, 75)), "p95": float(_np.percentile(_d, 95)),
            "min": float(_d.min()), "mean": float(_d.mean())}

    out = {
        "_note": ("GATE 2: the cost of an AREAL serving provider that cannot supply the newest "
                  f"{a.splice_days} days of the 365-day encoder context (NLDAS-2 publication "
                  "latency is 5 days), so those days are point-filled. Same checkpoints, same "
                  "rows; only the newest encoder forcing days differ. PREREG_v2.md #GATE 2."),
        "frame": a.frame, "member": a.member, "seeds": seeds, "protocol": a.protocol,
        "splice_recent_days": a.splice_days,
        "rows_joined": int(len(m)), "rows_moved": moved, "basins": int(n),
        "median_nse_areal_control": med_c,
        "median_nse_spliced_treatment": med_t,
        "paired_delta_spliced_minus_areal": delta,
        "ci95": [lo, hi], "breadth": breadth,
        "registered_band": list(BAND), "registered_stop": STOP,
        "in_registered_band": bool(in_band),
        "gate1_full_substitution": GATE1_FULL,
        "fraction_of_gate1_loss_recovered_by_areal": recovered,
        "difference_of_medians": dom,
        "gate1_difference_of_medians": GATE1_DOM,
        "fraction_recovered_on_difference_of_medians": recovered_dom,
        "_statistics_disagree": ("the paired median and the difference of medians disagree by ~9x; "
                                "the typical basin barely moves while a left tail drags the cohort "
                                "median level. Decision uses the REGISTERED paired statistic; any "
                                "published cohort headline must use the level."),
        "per_basin_delta_tail": tail,
        "verdict": decide(delta),
        "my_framing_falsified": bool(delta >= 0),
        "falsified_note": ("delta is NON-NEGATIVE: point-filling the newest days did not hurt. "
                           "Either the splice is not doing what I think or recency is not what "
                           "drives GATE 1. UNEXPLAINED -- debug, do not bank."
                           if delta >= 0 else ""),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nGATE 2 — {a.member}, {a.frame}, {n} basins, {len(m)} rows joined")
    print(f"  median NSE, areal control (all 365 d areal) : {med_c:.6f}")
    print(f"  median NSE, newest {a.splice_days} d point-filled      : {med_t:.6f}")
    print(f"  PAIRED delta (spliced - areal)              : {delta:+.6f}  [{lo:+.6f}, {hi:+.6f}]  breadth {breadth:.3f}")
    print(f"  registered band {BAND[0]:.3f}..{BAND[1]:.3f} => {'IN BAND' if in_band else 'OUT OF BAND'}")
    print(f"  difference of medians                       : {dom:+.6f}   <- {abs(dom/delta):.1f}x the paired delta")
    print(f"  GATE 1 paired {GATE1_FULL:+.6f} => areal recovers {100*recovered:.1f}% on the PAIRED statistic")
    print(f"  GATE 1 d-o-m  {GATE1_DOM:+.6f} => areal recovers {100*recovered_dom:.1f}% on the LEVEL")
    print(f"  left tail: {tail['n_worse_than_-0.05']} basins worse than -0.05, min {tail['min']:+.6f}")
    print(f"  VERDICT: {out['verdict']}")
    if out["my_framing_falsified"]:
        print(f"  ⛔ {out['falsified_note']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
