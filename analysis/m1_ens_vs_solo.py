#!/usr/bin/env python3
"""Paired comparison of a COMBINED ensemble against its best single member.

Why this needs its own script: analysis/m1_composition.py combines with the equal-weight
MEAN, so `--against <solo>` answers the question for the mean even when the selected
combiner is the TRIMMED mean. Measured 2026-10-05: with-q mean 0.911910 LOSES to
fused4md49 solo 0.910369 (paired -0.000601, CI excluding zero) while trimmed 0.914227 is
well above it. Reporting the trimmed headline while having only tested the mean would be
quoting one estimator and validating another.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--solo", required=True)
    ap.add_argument("--frame", default="val1")
    ap.add_argument("--protocol", default="modern-v3-9yr")
    ap.add_argument("--seeds", default="501,502,503,504,505")
    ap.add_argument("--basin-list", default="data/m1_cohort.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    members = [x for x in a.members.split(",") if x]
    seeds = set(int(x) for x in a.seeds.split(",") if x)
    ids = set(str(x).zfill(8) for x in json.load(open(a.basin_list))["cohort"])

    base = None
    for m in members:
        d = l51.load_member(m, a.frame, seeds=seeds, protocol=a.protocol)
        base = d if base is None else base.merge(d.drop(columns=["truth"]),
                                                 on=["station_id", "t0"], how="inner")
    base = base[base.station_id.isin(ids)]
    P = base[members].to_numpy(float)
    n = P.shape[1]
    res = {"frame": a.frame, "protocol": a.protocol, "seeds": sorted(seeds),
           "members": members, "solo": a.solo, "basins": int(base.station_id.nunique()),
           "rows": int(len(base)), "comparisons": {}}

    solo_s = l51.per_basin(base, [a.solo])
    print(f"\njoined {len(base):,} rows  {base.station_id.nunique()} basins  seeds={sorted(seeds)}")
    print(f"{'best solo (' + a.solo + ')':44s} {float(solo_s.median()):.6f}")

    combos = {"mean": P.mean(1)}
    if n >= 4:
        combos["trimmed"] = (P.sum(1) - P.max(1) - P.min(1)) / (n - 2)
    for name, pred in combos.items():
        ens = l51.per_basin(base.assign(_e=pred), ["_e"])
        common = sorted(set(ens.index) & set(solo_s.index))
        d, lo, hi, br, k = l51.paired_ci(ens.loc[common], solo_s.loc[common])
        verdict = ("ENSEMBLE WINS" if lo > 0 else
                   "ENSEMBLE LOSES" if hi < 0 else "NULL (CI includes zero)")
        print(f"{'ensemble ' + name:44s} {float(ens.median()):.6f}")
        print(f"{'   paired vs solo':44s} {d:+.6f} CI[{lo:+.6f},{hi:+.6f}] breadth {br:.3f}  => {verdict}")
        res["comparisons"][name] = {"ensemble_median": float(ens.median()),
                                    "paired_delta": d, "ci95": [lo, hi],
                                    "breadth": br, "verdict": verdict}
    res["solo_median"] = float(solo_s.median())
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
