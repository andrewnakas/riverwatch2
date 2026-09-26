#!/usr/bin/env python
"""MODERN-1 GATE B: does a 34-year training window beat a 9-year one?

The record trains on 9 years because Nearing 2022 did, purely for comparability.
P-MODERN offers 34, and nothing in this campaign has ever tested whether the extra
25 buy skill -- the legacy split had no room to try (its test decade sits between
train and val). It is also a budget decision: 34 years is ~6.0M training windows
against ~1.6M, i.e. ~3.5x the GPU cost per seed across ~50 runs.

The two arms have DIFFERENT protocols by construction (train_start 1980-10-01 vs
2005-10-01), so they cannot be loaded through one protocol guard -- each side is
loaded under its own, then joined on identical rows.

Pre-registered (PREREG_v2.md MODERN-1): +0.005 .. +0.025 for 34 years, adjudicated
on VAL, 3 seeds each. Counter-hypothesis on record: LEDGER 55 measured that the
cosine LR schedule, not the window count, decides the MB training budget -- so a
NULL is a live outcome, and on a null the 9-year window is kept and that is the
answer. The losing branch is not re-litigated.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(l51)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="val1", help="val1 decides; test1 only at the end")
    ap.add_argument("--long", default="gridmetm", help="the 34-year arm")
    ap.add_argument("--short", default="gridmetm9", help="the 9-year arm")
    ap.add_argument("--long-protocol", default="modern-v3")
    ap.add_argument("--short-protocol", default="modern-v3-9yr")
    ap.add_argument("--seeds", default="501,502,503")
    ap.add_argument("--out", default="benchmarks/m1_gateb_verdict.json")
    a = ap.parse_args()
    seeds = set(int(s) for s in a.seeds.split(","))

    L = l51.load_member(a.long, a.frame, seeds=seeds, protocol=a.long_protocol)
    S = l51.load_member(a.short, a.frame, seeds=seeds, protocol=a.short_protocol)
    L = L.rename(columns={a.long: "long"})
    S = S.rename(columns={a.short: "short"})

    m = L.merge(S.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
    print(f"\njoined {len(m):,} rows  {m.station_id.nunique()} basins  seeds={sorted(seeds)}")
    if not len(m):
        sys.exit("no overlapping rows - are both arms dumped on the same frame?")

    nse_l = l51.per_basin(m, ["long"])
    nse_s = l51.per_basin(m, ["short"])
    common = sorted(set(nse_l.index) & set(nse_s.index))
    sl, ss = nse_l.loc[common], nse_s.loc[common]
    delta, lo, hi, breadth, n = l51.paired_ci(sl, ss)

    print("=" * 76)
    print(f"{'34-year window (' + a.long + ')':46s} {np.median(sl.to_numpy()):.6f}")
    print(f"{'9-year window  (' + a.short + ')':46s} {np.median(ss.to_numpy()):.6f}")
    print(f"{'PAIRED delta (34yr - 9yr)':46s} {delta:+.6f}")
    print(f"{'   95% CI':46s} [{lo:+.6f}, {hi:+.6f}]")
    print(f"{'   breadth (basins favouring 34yr)':46s} {breadth:.3f}")
    print(f"{'   basins':46s} {len(common)}")
    print("=" * 76)

    band = (0.005, 0.025)
    if lo <= 0 <= hi:
        verdict = ("NULL - CI includes zero. Keep the 9-year window; it is ~3.5x "
                   "cheaper per seed and buys nothing. Prediction falsified.")
    elif delta < 0:
        verdict = ("NEGATIVE - more data is WORSE. Keep 9 years and record the "
                   "direction as another failed causal claim.")
    elif band[0] <= delta <= band[1]:
        verdict = f"IN BAND {band} - adopt the 34-year window for every member."
    elif delta > band[1]:
        verdict = (f"ABOVE BAND ({delta:+.6f} > {band[1]}) - adopt 34 years, and "
                   f"note the prediction was too conservative.")
    else:
        verdict = (f"POSITIVE BUT BELOW BAND ({delta:+.6f} < {band[0]}). CI excludes "
                   f"zero, so it is real but small; weigh against 3.5x the GPU cost.")
    print(f"\nVERDICT: {verdict}\n")

    json.dump({"_note": "MODERN-1 Gate B. Band and adjudication frame fixed in "
                        "PREREG_v2.md before the runs.",
               "frame": a.frame, "seeds": sorted(seeds),
               "long_arm": a.long, "short_arm": a.short,
               "rows": int(len(m)), "basins": len(common),
               "median_34yr": float(np.median(sl.to_numpy())),
               "median_9yr": float(np.median(ss.to_numpy())),
               "paired_delta": float(delta), "ci95": [float(lo), float(hi)],
               "breadth": float(breadth), "registered_band": list(band),
               "verdict": verdict}, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
