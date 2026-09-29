#!/usr/bin/env python3
"""MODERN-1 GATE A — the registered pipeline control (PREREG_v2.md MODERN-1).

The question Gate A exists to answer: has the NEW extraction pipeline and the NEW
forcing products broken something, independently of the window change?

It isolates that by training the modern fused corpus on **P-NEARING** -- the frozen
record's own window -- and scoring it against the frozen `fused3h1` member on the
same test1 frame. Any difference is then PIPELINE + PRODUCT, not decade:

    fused3h1  = CAMELS Daymet + Maurer + NLDAS, frozen corpora, solo test1 0.872162
    fused3mN  = areal Daymet V4R1 + gridMET + nClimGrid, new pipeline, same window

Registered band: -0.015 .. +0.005 solo.  ⛔ STOP below -0.03 ⇒ halt and re-run the
Phase-1 screens on the BUILT corpus before any further GPU. Maurer is gone and
Daymet V4R1 measurably carries less signal (-0.0244 on the precip/streamflow
screen), so a modest loss is the EXPECTATION, not a failure.

⚠️ Two medians are reported for the reference and they are not interchangeable:
the frozen 0.872162 is over all 531 basins, while the paired comparison runs on
the ~490-basin cohort join. Quoting the 531 number against a 490-basin candidate
would be a difference of medians across different samples -- the exact trap that
has disagreed with the paired statistic nine times. The VERDICT uses the paired
per-basin delta on the join; the 531 number is context only.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)

BAND = (-0.015, 0.005)
STOP = -0.03


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="test1",
                    help="test1: P-NEARING's test decade. This is the LEGACY window, "
                         "already read many times; it is NOT the modern headline frame, "
                         "so reading it here selects nothing about P-MODERN.")
    ap.add_argument("--cand", default="fused3mN", help="new-pipeline member (m1_ prefix)")
    ap.add_argument("--ref", default="fused3h1", help="frozen member (camels531_l51_ prefix)")
    ap.add_argument("--seeds", default="501,502,503")
    ap.add_argument("--cohort", default="data/m1_cohort.json")
    ap.add_argument("--out", default="benchmarks/m1_gate_a_verdict.json")
    a = ap.parse_args()
    seeds = set(int(s) for s in a.seeds.split(","))

    # Same protocol guard for BOTH sides -- that is the whole point of Gate A --
    # but different dump prefixes, so the new arm is found at all.
    C = l51.load_member(a.cand, a.frame, seeds=seeds, protocol="nearing2022", prefix="m1_")
    R = l51.load_member(a.ref, a.frame, seeds=seeds, protocol="nearing2022")

    cohort = set(str(x).zfill(8) for x in json.load(open(a.cohort))["cohort"])
    ref_full = l51.per_basin(R.rename(columns={a.ref: "ref"}), ["ref"])
    median_ref_531 = float(np.median(ref_full.to_numpy()))

    C = C.rename(columns={a.cand: "cand"}); R = R.rename(columns={a.ref: "ref"})
    m = C.merge(R.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    m = m[m.station_id.isin(cohort)]
    if not len(m):
        raise SystemExit("no overlapping rows - is the candidate dumped on this frame?")
    print(f"\njoined {len(m):,} rows  {m.station_id.nunique()} basins  seeds={sorted(seeds)}")

    nse_c = l51.per_basin(m, ["cand"]); nse_r = l51.per_basin(m, ["ref"])
    common = sorted(set(nse_c.index) & set(nse_r.index))
    sc, sr = nse_c.loc[common], nse_r.loc[common]
    delta, lo, hi, breadth, n = l51.paired_ci(sc, sr)
    med_c, med_r = float(np.median(sc.to_numpy())), float(np.median(sr.to_numpy()))

    print("=" * 78)
    print(f"{'new pipeline, P-NEARING  (' + a.cand + ')':52s} {med_c:.6f}")
    print(f"{'frozen, cohort join      (' + a.ref + ')':52s} {med_r:.6f}")
    print(f"{'frozen, all 531 (context, NOT the comparison)':52s} {median_ref_531:.6f}")
    print(f"{'PAIRED delta (new - frozen)':52s} {delta:+.6f}")
    print(f"{'   95% CI':52s} [{lo:+.6f}, {hi:+.6f}]")
    print(f"{'   breadth (basins favouring the new pipeline)':52s} {breadth:.3f}")
    print(f"{'   basins':52s} {len(common)}")
    print("=" * 78)

    if delta < STOP:
        verdict = (f"⛔ STOP ({delta:+.6f} < {STOP}). The registered falsifier fired. "
                   f"Halt GPU, re-run the Phase-1 lag/streamflow screens on the BUILT "
                   f"fused corpus, and diagnose registration or extraction first.")
        ok = False
    elif BAND[0] <= delta <= BAND[1]:
        verdict = (f"✅ IN BAND {BAND}. Pipeline and products behave as registered; "
                   f"the modern corpus is sound. Proceed to the P-MODERN family.")
        ok = True
    elif delta > BAND[1]:
        verdict = (f"✅ ABOVE BAND ({delta:+.6f} > {BAND[1]}). The new products are BETTER "
                   f"than the frozen three on the frozen window - the pipeline is sound and "
                   f"the registered expectation was too pessimistic. Proceed.")
        ok = True
    else:
        verdict = (f"⚠️ BELOW BAND but above the STOP line ({STOP} < {delta:+.6f} < {BAND[0]}). "
                   f"Build-but-flag: proceed, and carry this deficit as a STATED cost of "
                   f"modernity in the write-up rather than discovering it at scoring time.")
        ok = True
    print(f"\nVERDICT: {verdict}\n")

    json.dump({"_note": "MODERN-1 Gate A. Band, STOP line and frame fixed in PREREG_v2.md "
                        "before the run. Verdict uses the PAIRED per-basin delta on the "
                        "cohort join; the 531-basin frozen median is context only.",
               "frame": a.frame, "seeds": sorted(seeds), "candidate": a.cand, "reference": a.ref,
               "rows": int(len(m)), "basins": len(common),
               "median_new_pipeline": med_c, "median_frozen_on_join": med_r,
               "median_frozen_all531_context": median_ref_531,
               "paired_delta": float(delta), "ci95": [float(lo), float(hi)],
               "breadth": float(breadth), "registered_band": list(BAND), "stop_line": STOP,
               "passed": ok, "verdict": verdict}, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    return 0 if ok else 9


if __name__ == "__main__":
    raise SystemExit(main())
