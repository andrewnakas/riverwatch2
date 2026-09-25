#!/usr/bin/env python3
"""LEDGER 56 — seed-matched verdict for the all-product-decoder fused member.

Three questions on ONE join (identical rows for every statistic):
  1. SOLO: fused3h1d3 vs fused3h1, both restricted to the SAME seeds (3-vs-3 at
     the screen, 5-vs-5 at the ship bar). Paired per-basin delta, basin-bootstrap
     CI, breadth. Pre-registered falsifier: delta < +0.002 or CI including zero.
  2. STRUCTURE: median per-basin error correlation of each against the four
     single-forcing MB members and the three AR members (does fixing the decoder
     pull the member toward the AR block, which also sees every product?).
  3. ENSEMBLE: REPLACE fused3h1 by fused3h1d3 in the frozen 8 (the other seven at
     their shipped 5 seeds; the swapped slot seed-matched on both sides). Paired
     delta of the 8-with-d3 vs the 8-with-fused3h1. Also the 9-member ADD for
     reference. The replacement rule was fixed in PREREG_v2.md before the run.
usage: l56_d3_verdict.py --frame val1 --cand fused3h1d3 --seeds 501,502,503 [--out X.json]
"""
import argparse, importlib.util, json, os, sys
import numpy as np

spec = importlib.util.spec_from_file_location(
    "l51", os.path.join(os.path.dirname(os.path.abspath(__file__)), "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)
MB4 = ["daymeth1", "nldash1", "maurerh1", "aorch1"]
AR3 = ["nhar", "nhar0", "nhar0h256"]
CTL = "fused3h1"


def err_corr(m, a, b):
    out = []
    ea = m[a].to_numpy(float) - m.truth.to_numpy(float)
    eb = m[b].to_numpy(float) - m.truth.to_numpy(float)
    for s, idx in m.groupby("station_id").indices.items():
        x, y = ea[idx], eb[idx]
        k = np.isfinite(x) & np.isfinite(y)
        if k.sum() < 20 or np.std(x[k]) < 1e-9 or np.std(y[k]) < 1e-9:
            continue
        out.append(np.corrcoef(x[k], y[k])[0, 1])
    return float(np.median(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=["val1", "test1"])
    ap.add_argument("--cand", default="fused3h1d3")
    ap.add_argument("--seeds", default="501,502,503")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    seeds = {int(x) for x in a.seeds.split(",") if x}
    C = a.cand
    # one join: seven shipped members at full depth, the two fused variants seed-matched
    parts = [l51.load_member(x, a.frame) for x in MB4 + AR3]
    parts += [l51.load_member(CTL, a.frame, seeds=seeds), l51.load_member(C, a.frame, seeds=seeds)]
    m = parts[0]
    for d in parts[1:]:
        m = m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    print(f"\njoined {len(m):,} rows  {m.station_id.nunique()} basins  seeds={sorted(seeds)}\n")

    res = {"frame": a.frame, "cand": C, "control": CTL, "seeds": sorted(seeds),
           "rows": int(len(m)), "basins": int(m.station_id.nunique())}
    # 1. solo, seed-matched
    sc, sk = l51.per_basin(m, [C]), l51.per_basin(m, [CTL])
    d, lo, hi, br, nb = l51.paired_ci(sc, sk)
    res["solo"] = {C: float(sc.median()), CTL: float(sk.median()),
                   "paired_delta": d, "ci95": [lo, hi], "breadth": br, "n": nb}
    print(f"SOLO  {C}={sc.median():.6f}  {CTL}={sk.median():.6f}  paired {d:+.6f} "
          f"CI [{lo:+.6f},{hi:+.6f}] breadth {br:.3f} n={nb}")
    falsified = (d < 0.002) or (lo <= 0.0)
    res["solo"]["prereg_falsifier_triggered"] = bool(falsified)
    print(f"      pre-registered falsifier (delta < +0.002 or CI incl. 0): "
          f"{'TRIGGERED' if falsified else 'not triggered'}")
    # 2. structure
    res["err_corr"] = {}
    for x in (C, CTL):
        vs_mb = float(np.mean([err_corr(m, x, y) for y in MB4]))
        vs_ar = float(np.mean([err_corr(m, x, y) for y in AR3]))
        res["err_corr"][x] = {"vs_MB4": vs_mb, "vs_AR3": vs_ar}
        print(f"CORR  {x:11s} vs MB4 {vs_mb:.4f}   vs AR3 {vs_ar:.4f}")
    res["err_corr"]["cand_vs_control"] = err_corr(m, C, CTL)
    print(f"CORR  {C} vs {CTL}: {res['err_corr']['cand_vs_control']:.4f}")
    # 3. ensemble: replacement (seed-matched slot) and add
    e_ctl = l51.per_basin(m, MB4 + AR3 + [CTL])
    e_rep = l51.per_basin(m, MB4 + AR3 + [C])
    e_add = l51.per_basin(m, MB4 + AR3 + [CTL, C])
    for name, e in (("replace", e_rep), ("add9", e_add)):
        d, lo, hi, br, nb = l51.paired_ci(e, e_ctl)
        res[name] = {"median": float(e.median()), "paired_delta": d, "ci95": [lo, hi], "breadth": br}
        print(f"ENS   {name:8s} {e.median():.6f}  vs 8-with-{CTL} {e_ctl.median():.6f}  paired {d:+.6f} "
              f"CI [{lo:+.6f},{hi:+.6f}] breadth {br:.3f}")
    res["ens_control_median"] = float(e_ctl.median())
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
