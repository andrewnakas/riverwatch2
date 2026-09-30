#!/usr/bin/env python3
"""MODERN-1 — the decade probe. ONE checkpoint, TWO evaluation decades.

Registered in PREREG_v2.md §MODERN-1 before any number existed. It exists because a
literature check found **no published work evaluating the CAMELS-531 lineage on a
post-2014 test window** (the CAMELS forcings stop at 2014-09-30), so P-MODERN has no
external reference and its headline is an uninterpretable level without an internal one.

Gate A isolated the PRODUCT component at fixed window (−0.017036). This isolates the
EVALUATION-DECADE component at fixed weights: the same `fused3md3q` checkpoint (trained
2005-10-01…2014-09-30) scored on 1989-10-01…1999-09-30 and on 2018-10-01…2025-09-30.

⚠️ What this is NOT: a clean causal decade effect. 1989-1999 sits ~6 years BEFORE the
training window and 2018-2025 ~4 years after — comparable in magnitude, opposite in sign.
It answers "is the modern decade harder FOR THIS MODEL", and is reported in those words.
⚠️ Both windows lie fully outside 2005-2014, so neither is in-sample. The retracted 0.9253
came from exactly that error; `check_sidecar` still enforces the training window.
⚠️ The cohort was frozen on MODERN discharge coverage (≥90% over 2018-2025), so some of its
basins may be sparsely gauged in 1989-1999. The paired set is the intersection of basins
scoreable in BOTH windows, and the attrition is reported rather than absorbed.
"""
from __future__ import annotations
import argparse, importlib.util, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)

BAND = (-0.030, -0.005)   # registered: the modern decade scores LOWER by 0.005..0.030


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fused3md3q")
    ap.add_argument("--modern-frame", default="test1", help="the member's own test window")
    ap.add_argument("--legacy-frame", default="xdecade1", help="the 1989-1999 dump")
    ap.add_argument("--protocol", default="modern-v3-9yr")
    ap.add_argument("--seeds", default="501,502,503")
    ap.add_argument("--basin-list", default="data/m1_cohort.json")
    ap.add_argument("--out", default="benchmarks/m1_decade_probe.json")
    a = ap.parse_args()
    seeds = set(int(s) for s in a.seeds.split(","))

    # Same protocol guard on BOTH frames: the training window is identical by
    # construction (one checkpoint), and that is the thing the guard protects.
    M = l51.load_member(a.member, a.modern_frame, seeds=seeds, protocol=a.protocol)
    L = l51.load_member(a.member, a.legacy_frame, seeds=seeds, protocol=a.protocol)
    ids = set(str(x).zfill(8) for x in json.load(open(a.basin_list))["cohort"])
    M = M[M.station_id.isin(ids)]; L = L[L.station_id.isin(ids)]

    nse_m = l51.per_basin(M.rename(columns={a.member: "y"}), ["y"])
    nse_l = l51.per_basin(L.rename(columns={a.member: "y"}), ["y"])
    common = sorted(set(nse_m.index) & set(nse_l.index))
    if not common:
        raise SystemExit("no basins scoreable in both windows")
    sm, sl = nse_m.loc[common], nse_l.loc[common]
    delta, lo, hi, breadth, n = l51.paired_ci(sm, sl)   # modern − legacy

    print(f"\n{'basins scoreable in the MODERN window':52s} {len(nse_m)}")
    print(f"{'basins scoreable in the LEGACY window':52s} {len(nse_l)}")
    print(f"{'PAIRED on both (the comparison set)':52s} {len(common)}")
    att = len(ids) - len(common)
    print(f"{'attrition from the 490-basin cohort':52s} {att} "
          f"({att / max(len(ids), 1):.1%}) — the cohort was frozen on MODERN coverage")
    print("=" * 78)
    print(f"{'modern decade 2018-2025  (' + a.modern_frame + ')':52s} {np.median(sm.to_numpy()):.6f}")
    print(f"{'legacy decade 1989-1999  (' + a.legacy_frame + ')':52s} {np.median(sl.to_numpy()):.6f}")
    print(f"{'PAIRED delta (modern − legacy)':52s} {delta:+.6f}")
    print(f"{'   95% CI':52s} [{lo:+.6f}, {hi:+.6f}]")
    print(f"{'   breadth (basins where modern is better)':52s} {breadth:.3f}")
    print("=" * 78)

    if lo <= 0 <= hi:
        verdict = ("NULL — the two decades are indistinguishable for this model. The modern "
                   "shortfall is then NOT about the decade, and Gate A's product cost plus the "
                   "cohort change must carry the whole explanation.")
    elif delta > 0:
        verdict = (f"⛔ FALSIFIER FIRED: the MODERN decade scores HIGHER ({delta:+.6f}). The "
                   f"'modern data is harder' framing is wrong. Attribute the shortfall to "
                   f"products and cohort — a cleaner result, not a worse one.")
    elif BAND[0] <= delta <= BAND[1]:
        verdict = (f"IN BAND {BAND} — the modern decade is harder for this model by {delta:+.6f}. "
                   f"Report alongside Gate A's product cost as the second component of the gap.")
    elif delta < BAND[0]:
        verdict = (f"BELOW BAND ({delta:+.6f} < {BAND[0]}) — the decade penalty is LARGER than "
                   f"registered. It dominates the product cost; say so and do not attribute the "
                   f"shortfall to the pipeline.")
    else:
        verdict = (f"ABOVE BAND ({delta:+.6f} > {BAND[1]}) — real but smaller than registered.")
    print(f"VERDICT: {verdict}\n")

    json.dump({"_note": "MODERN-1 decade probe. ONE checkpoint, TWO evaluation decades; "
                        "training window identical by construction. Measures whether the modern "
                        "decade is harder FOR THIS MODEL, not a clean causal decade effect: "
                        "1989-1999 is ~6y before training, 2018-2025 ~4y after.",
               "member": a.member, "seeds": sorted(seeds), "protocol": a.protocol,
               "modern_frame": a.modern_frame, "legacy_frame": a.legacy_frame,
               "basins_modern": int(len(nse_m)), "basins_legacy": int(len(nse_l)),
               "basins_paired": len(common), "cohort_attrition": att,
               "median_modern": float(np.median(sm.to_numpy())),
               "median_legacy": float(np.median(sl.to_numpy())),
               "paired_delta_modern_minus_legacy": float(delta),
               "ci95": [float(lo), float(hi)], "breadth_modern_better": float(breadth),
               "registered_band": list(BAND), "verdict": verdict},
              open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
