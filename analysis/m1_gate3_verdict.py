#!/usr/bin/env python
"""GATE 3 verdict — may `nldasm9` serve basins it was never trained on?

Registered in PREREG_v2.md §GATE 3 (band 0.82..0.89, STOP 0.75, decision table)
and its ADDENDUM 2026-10-16, which fixes the scoreable subset in
`benchmarks/m1_gate3_coverage.json`. This script reads those out; it does not
choose any of them.

⚠️⚠️ THE ONE THING THAT MAKES THIS SCRIPT NECESSARY RATHER THAN A ONE-LINER.
`l51_withq_score.load_member` filters every dump through a module-level
`IDS` = the **531 benchmark subset**. Only **12 of the 130** registered
treatment basins are in it, because 140 of the 181 out-of-cohort CAMELS basins
were never 531 candidates at all. Scoring naively would therefore have returned
a median over 12 basins, labelled 130, with no error and nothing in the log --
and 12 basins is far inside the noise this project has measured for a median.
So `IDS` is widened here, and the widening is ASSERTED against the basin count
that actually comes back: a patch that silently fails to apply is the recorded
failure mode (`assert-the-patch-applied`), and the whole point of this gate is
a basin count.

The deciding statistic is an UNPAIRED difference of levels between two disjoint
basin sets. That is a deviation from this project's standing rule against
differences of medians -- recorded, in advance, in the registration: an
out-of-cohort basin has no in-cohort counterpart, so pairing is not merely
unused here, it does not exist.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).parent
spec = importlib.util.spec_from_file_location(
    "l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l51)

COVERAGE = ROOT / "benchmarks" / "m1_gate3_coverage.json"
COHORT = ROOT / "data" / "m1_cohort.json"

# Fixed in the prereg. Read, never recomputed here.
BAND = (0.82, 0.89)
STOP = 0.75


def decide(median: float, paired_vs_persistence: float | None,
           ci_excludes_zero: bool, breadth: float | None) -> str:
    if median < 0.75:
        return ("STOP — DO NOT SERVE OUTSIDE THE COHORT.")
    if median < 0.78:
        return ("SERVE ONLY basins passing the pre-declared per-basin screen "
                "(own-basin val1 NSE >= 0.70), and label every such basin as "
                "an IN-SAMPLE selection on val.")
    if median < 0.86:
        return ("SERVE, LABELLED OUT-OF-COHORT, quoting the out-of-cohort "
                "median and never the cohort one. The UI must mark these "
                "basins as a distinct tier.")
    ok = (paired_vs_persistence is not None and paired_vs_persistence >= 0.10
          and ci_excludes_zero and breadth is not None and breadth >= 0.80)
    if ok:
        return ("SERVE the scoreable out-of-cohort basins, labelled "
                "out-of-cohort with their own measured median: coverage goes "
                "490 -> 620. The 2 partial-coverage basins are a SEPARATE "
                "decision on their own reported median, and the 3 unscoreable "
                "ones may not be served on the strength of this gate -- so "
                "this row does not reach the registration's 625.")
    return ("SERVE, LABELLED OUT-OF-COHORT (the median clears 0.86 but the "
            "persistence conditions do not all hold, so the top row of the "
            "decision table does not apply).")


def median_ci(s: pd.Series, n: int = 5000, seed: int = 0):
    """Bootstrap the median LEVEL over basins. Not a paired delta: see the
    module docstring."""
    v = s.to_numpy(float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    draws = np.array([np.median(v[rng.integers(0, len(v), len(v))])
                      for _ in range(n)])
    return (float(np.median(v)), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), len(v))


def persistence_from_qstore(rows: pd.DataFrame, qdir: Path) -> pd.Series:
    """Persistence at h=1 is the discharge observed AT t0 -- the forecast
    "tomorrow equals today". Built from q_store rather than from the dump,
    which carries only the target.
    """
    out = {}
    for sid, g in rows.groupby("station_id"):
        hits = sorted(qdir.glob(f"{sid}*"))
        if not hits:
            continue
        d = pd.read_csv(hits[0])
        dc = next(c for c in d.columns if c.lower() in ("date", "datetime", "time"))
        qc = next(c for c in d.columns if "q" in c.lower() and c != dc)
        ser = pd.Series(d[qc].to_numpy(float),
                        index=pd.to_datetime(d[dc]).dt.strftime("%Y-%m-%d").to_numpy())
        ser = ser[~ser.index.duplicated(keep="last")]
        out[sid] = ser.reindex(g.t0.to_numpy()).to_numpy(float)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="val1")
    ap.add_argument("--member", default="nldasm9")
    ap.add_argument("--treatment-prefix", default="g3oc_")
    ap.add_argument("--control-prefix", default="m1_")
    ap.add_argument("--seeds", default="501,502,503,504,505")
    ap.add_argument("--protocol", default="modern-v3-9yr")
    ap.add_argument("--qstore", default="data/modern_corpus/q_store")
    ap.add_argument("--out", default="benchmarks/m1_gate3_verdict.json")
    a = ap.parse_args()
    os.chdir(ROOT)

    cov = json.loads(COVERAGE.read_text())
    treat_ids = set(cov["groups"]["scoreable_full"]["ids"])
    part_ids = set(cov["groups"]["partial"]["ids"])
    unscoreable = sorted(cov["groups"]["unscoreable"]["ids"])
    cohort_ids = {str(x) for x in json.loads(COHORT.read_text())["cohort"]}
    assert not (treat_ids & cohort_ids), "treatment overlaps the training cohort"

    # ints, not strings: load_member filters with int(...) in seeds, so a list
    # of strings matches nothing and it exits "no dumps for member".
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    # ---- the IDS widening, and the assertion that it took effect -----------
    before = len(l51.IDS)
    l51.IDS = set(treat_ids) | set(part_ids) | set(cohort_ids)
    print(f"  widened l51.IDS from {before} (the 531 subset) to {len(l51.IDS)} "
          f"= {len(treat_ids)} treatment + {len(part_ids)} partial + "
          f"{len(cohort_ids)} control")

    treat = l51.load_member(a.member, a.frame, seeds=seeds,
                            protocol=a.protocol, prefix=a.treatment_prefix)
    ctrl = l51.load_member(a.member, a.frame, seeds=seeds,
                           protocol=a.protocol, prefix=a.control_prefix)

    t_full = treat[treat.station_id.isin(treat_ids)].copy()
    t_part = treat[treat.station_id.isin(part_ids)].copy()
    c = ctrl[ctrl.station_id.isin(cohort_ids)].copy()

    got_t = t_full.station_id.nunique()
    if got_t != len(treat_ids):
        print(f"ABORT: the treatment arm came back with {got_t} basins, not the "
              f"{len(treat_ids)} registered. If this says 12, the IDS widening "
              f"did not take effect and the number below would be a 12-basin "
              f"median wearing a 130-basin label.")
        return 1
    got_c = c.station_id.nunique()
    print(f"  arms: treatment {got_t} basins / {len(t_full):,} rows; "
          f"control {got_c} basins / {len(c):,} rows")
    if got_c < 400:
        print(f"ABORT: control arm has only {got_c} basins; expected ~490")
        return 1

    nse_t = l51.per_basin(t_full, [a.member])
    nse_c = l51.per_basin(c, [a.member])
    med_t, lo_t, hi_t, n_t = median_ci(nse_t)
    med_c, lo_c, hi_c, n_c = median_ci(nse_c)

    # (b) the paired statistic, on the out-of-cohort rows only.
    pers = persistence_from_qstore(t_full, Path(a.qstore))
    frames = []
    for sid, g in t_full.groupby("station_id"):
        if sid not in pers:
            continue
        gg = g.copy()
        gg["persistence"] = pers[sid]
        frames.append(gg)
    paired = None
    if frames:
        pf = pd.concat(frames, ignore_index=True)
        pf = pf[np.isfinite(pf.persistence.to_numpy(float))]
        nse_m = l51.per_basin(pf, [a.member])
        nse_p = l51.per_basin(pf.rename(columns={"persistence": "p"}), ["p"])
        d, plo, phi, breadth, npair = l51.paired_ci(nse_m, nse_p)
        paired = {"delta_member_minus_persistence": d, "ci95": [plo, phi],
                  "breadth": breadth, "basins": npair,
                  "median_nse_persistence": float(nse_p.median())}

    nse_part = l51.per_basin(t_part, [a.member]) if len(t_part) else pd.Series(dtype=float)
    pd_ok = paired is not None
    verdict = decide(med_t,
                     paired["delta_member_minus_persistence"] if pd_ok else None,
                     bool(pd_ok and paired["ci95"][0] > 0),
                     paired["breadth"] if pd_ok else None)

    out = {
        "_note": (
            "GATE 3: whether nldasm9 may serve CAMELS basins outside its "
            "490-basin training cohort. Same checkpoints, same areal forcing, "
            "same val1 window; only training-set membership differs. The "
            "deciding statistic is an UNPAIRED difference of LEVELS between "
            "two disjoint basin sets -- pairing does not exist here, and the "
            "deviation from this project's standing rule was recorded in "
            "advance in PREREG_v2.md. Registered there with its band, STOP "
            "and decision table before any number existed; the scoreable "
            "subset is fixed in benchmarks/m1_gate3_coverage.json."),
        "frame": a.frame, "member": a.member, "seeds": seeds,
        "protocol": a.protocol,
        "out_of_cohort": {
            "basins": n_t, "rows": int(len(t_full)),
            "median_nse": med_t, "ci95": [lo_t, hi_t]},
        "in_cohort_control": {
            "basins": n_c, "rows": int(len(c)),
            "median_nse": med_c, "ci95": [lo_c, hi_c]},
        "level_gap_out_minus_in": med_t - med_c,
        "vs_persistence_out_of_cohort": paired,
        "partial_coverage_basins_reported_separately": {
            "ids": sorted(part_ids),
            "n_scored": int(len(nse_part)),
            "median_nse": (float(nse_part.median()) if len(nse_part) else None),
            "_why": ("25-90% of the val window observed. Never pooled into the "
                     "deciding median: a basin contributing ~1,100 days is not "
                     "comparable to one contributing 1,461, and a median whose "
                     "rows differ between arms has misled this project before.")},
        "unscoreable_basins_may_not_be_served": {
            "ids": unscoreable,
            "_why": ("gauged recently but not during val1, so this gate can "
                     "say nothing about them in either direction")},
        "registered_band": list(BAND),
        "in_band": BAND[0] <= med_t <= BAND[1],
        "stop_threshold": STOP,
        "stopped": med_t < STOP,
        "falsifier_fired": med_t >= med_c,
        "falsifier_note": (
            "The registration requires that an out-of-cohort median AT OR ABOVE "
            "the control be reported as UNEXPLAINED and checked against basin "
            "area and regulation before being banked -- not treated as evidence "
            "that domain transfer is free."
            if med_t >= med_c else None),
        "verdict": verdict,
    }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n  out-of-cohort median NSE {med_t:.6f} [{lo_t:.6f}, {hi_t:.6f}] "
          f"over {n_t} basins")
    print(f"  in-cohort control         {med_c:.6f} [{lo_c:.6f}, {hi_c:.6f}] "
          f"over {n_c} basins")
    print(f"  level gap (out - in)      {med_t - med_c:+.6f}")
    if paired:
        print(f"  vs persistence            {paired['delta_member_minus_persistence']:+.6f} "
              f"[{paired['ci95'][0]:+.6f}, {paired['ci95'][1]:+.6f}] "
              f"breadth {paired['breadth']:.3f} over {paired['basins']} basins")
    print(f"  band {BAND} -> in band: {out['in_band']}   STOP {STOP} -> "
          f"stopped: {out['stopped']}")
    if out["falsifier_fired"]:
        print(f"  ⚠️ FALSIFIER: {out['falsifier_note']}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
