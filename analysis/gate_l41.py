#!/usr/bin/env python3
"""Ledger 41, gate G3: does the pretrained initialisation beat random init?

Scores finetuned members against the matched random-init controls on the
TRAIN-SIDE frame only. The held-out window (1995-2010) is a budgeted resource
and is not touched here.

⚠️ THE BARS ARE SET AGAINST A MEASURED NULL, not chosen by taste. An earlier
version of this gate compared ONE treatment seed to ONE control seed with a
bootstrap CI, and passing it required nothing: running it on two seeds of the
same production recipe (daymet s222 vs s111, no treatment at all) returned
median dNSE +0.0110 with a CI excluding zero, 69.9% breadth, and near-median
+0.0097 -- a clean sweep of all three bars on pure seed noise. Across the
control seeds the null spans median dNSE -0.0127..+0.0127, breadth 29.8..70.2%,
and near-median -0.0221..+0.0097.

The bootstrap CI cannot see this: it resamples BASINS, so it measures basin
sampling variance, while two seeds are two different converged models whose
per-basin differences are correlated. A tight CI around a seed difference is
expected and says nothing about reproducibility.

So the primary bar is COMPLETE SEPARATION across seeds -- every treatment seed
above every control seed. For 3 vs 3 that is p = 0.05 by rank-sum and is the
strongest non-parametric claim the sample size supports. Paired median, breadth
and near-median deltas are computed on the seed-AVERAGED arms and reported as
shape, never as pass/fail on their own.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CV_VAL_START = "1990-10-01"


def load(path, point_col="ymed"):
    df = pd.read_csv(path, compression="gzip",
                     usecols=lambda c: c in ("station_id", "t0", "h", "truth", point_col))
    df = df[df["h"] == 1].copy()
    df["station_id"] = df["station_id"].astype(str).str.zfill(8)
    df["date"] = pd.to_datetime(df["t0"])
    return df[["station_id", "date", "truth", point_col]].rename(columns={point_col: "pred"})


def seed_average(paths, from_date):
    """Mean prediction across seeds on the common (basin, date) grid.

    This is how the production ensemble combines seeds, so the seed-averaged arm
    is the object the rest of the pipeline would actually consume.
    """
    frames = []
    for p in paths:
        d = load(p)
        d = d[d["date"] >= from_date]
        frames.append(d)
    cat = pd.concat(frames, ignore_index=True)
    return cat.groupby(["station_id", "date"], as_index=False).agg(
        pred=("pred", "mean"), truth=("truth", "first"))


def per_basin_nse(df):
    out = {}
    for sid, g in df.groupby("station_id"):
        yt = g["truth"].to_numpy(float)
        yh = g["pred"].to_numpy(float)
        ok = np.isfinite(yt) & np.isfinite(yh)
        yt, yh = yt[ok], yh[ok]
        if len(yt) < 20 or np.var(yt) < 1e-9:
            continue
        out[sid] = 1.0 - np.mean((yt - yh) ** 2) / np.var(yt)
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatment", nargs="+", required=True,
                    help="finetuned TRAIN dumps, one per seed")
    ap.add_argument("--control", nargs="+", required=True,
                    help="random-init TRAIN dumps, one per seed")
    ap.add_argument("--from-date", default=CV_VAL_START)
    ap.add_argument("--near-median-window", type=float, default=0.01)
    ap.add_argument("--min-seeds", type=int, default=3,
                    help="seeds required PER ARM before separation carries a "
                         "claim. 3 v 3 is p=0.05 by rank-sum; fewer degenerates "
                         "to an ordering and the verdict becomes INCONCLUSIVE.")
    ap.add_argument("--out", default="benchmarks/ledger41_gate_g3.json")
    args = ap.parse_args()

    # --- per-seed medians: the primary evidence, never hidden inside a mean
    per_seed = {"treatment": [], "control": []}
    for arm, paths in (("treatment", args.treatment), ("control", args.control)):
        for p in paths:
            d = load(p)
            d = d[d["date"] >= args.from_date]
            n = per_basin_nse(d)
            per_seed[arm].append((Path(p).name, float(n.median()), len(n)))
    print(f"train-side val slice from {args.from_date}\n")
    for arm in ("control", "treatment"):
        print(f"{arm}:")
        for name, med, nb in per_seed[arm]:
            print(f"  {med:.6f}  ({nb} basins)  {name}")

    t_meds = [m for _, m, _ in per_seed["treatment"]]
    c_meds = [m for _, m, _ in per_seed["control"]]
    separated = min(t_meds) > max(c_meds)

    # Complete separation only carries a claim when each arm has enough seeds.
    # At 3 v 3 it corresponds to p = 0.05 by rank-sum; at 1 v 1 it degenerates to
    # "is A > B", which any ordering satisfies. Measured: running this gate with
    # daymet s333 as treatment and s111 as control -- two seeds of the SAME
    # recipe -- returns "separation PASS", because with one seed per arm there is
    # nothing for separation to mean.
    min_seeds = min(len(t_meds), len(c_meds))
    underpowered = min_seeds < args.min_seeds
    print(f"\ntreatment range [{min(t_meds):.6f}, {max(t_meds):.6f}]  "
          f"control range [{min(c_meds):.6f}, {max(c_meds):.6f}]")

    # --- seed-averaged arms: the object the ensemble would consume
    ta = seed_average(args.treatment, args.from_date)
    ca = seed_average(args.control, args.from_date)
    nt, nc = per_basin_nse(ta), per_basin_nse(ca)
    common = sorted(set(nt.index) & set(nc.index))
    if len(common) < 400:
        sys.exit(f"only {len(common)} basins in common -- refusing to score. "
                 "A shrunken join is the classic silent failure here.")
    nt, nc = nt[common], nc[common]
    d = nt - nc
    med_t, med_c, med_d = float(nt.median()), float(nc.median()), float(d.median())
    breadth = float((d > 0).mean())
    near = nc[(nc - med_c).abs() <= args.near_median_window].index
    near_d = float(d[near].mean()) if len(near) else float("nan")

    print(f"\nseed-averaged arms, {len(common)} basins:")
    print(f"  control   median NSE {med_c:.6f}")
    print(f"  treatment median NSE {med_t:.6f}")
    print(f"  paired median delta  {med_d:+.6f}   breadth {100*breadth:.1f}%   "
          f"near-median {near_d:+.6f} (n={len(near)})")
    print("  (descriptive -- the pass/fail bar is seed separation, because the "
          "measured null for these three statistics is\n   median dNSE "
          "-0.0127..+0.0127, breadth 29.8..70.2%, near-median -0.0221..+0.0097)")

    print(f"\n{'PRIMARY BAR: complete seed separation':44s} "
          f"{'PASS' if separated else 'FAIL'}")
    if underpowered:
        print(f"  ⚠️ UNDERPOWERED: {len(t_meds)} treatment seed(s) vs "
              f"{len(c_meds)} control seed(s); {args.min_seeds} per arm are "
              f"required for separation to carry a claim. At 3 v 3 separation "
              f"is p=0.05 by rank-sum; below that it is just an ordering.")
    if not separated:
        # Distinguish genuine overlap from separation with the sign reversed.
        # The canned "the arms overlap" string was FALSE on the ledger-41 result
        # -- every treatment seed sat BELOW every control seed, which is a
        # strictly stronger finding than "within the seed spread", and the
        # message understated it. A reader quoting the gate output alone would
        # have recorded a weaker result than the data supports.
        if max(t_meds) < min(c_meds):
            print("  ⚠️ REVERSE SEPARATION: every treatment seed is BELOW every "
                  "control seed")
            print(f"     treatment max {max(t_meds):.6f} < control min "
                  f"{min(c_meds):.6f}")
            print("     This is NOT 'within the seed spread' -- it is complete "
                  "separation with the sign reversed, i.e. the treatment is "
                  "consistently WORSE, not indistinguishable.")
        else:
            print("  the arms OVERLAP, so this is within the seed spread: "
                  f"treatment [{min(t_meds):.6f}, {max(t_meds):.6f}] vs "
                  f"control [{min(c_meds):.6f}, {max(c_meds):.6f}]")
    verdict = separated and not underpowered
    label = "INCONCLUSIVE (underpowered)" if (separated and underpowered) else ("PASS" if verdict else "FAIL")
    print(f"\nG3 VERDICT: {label}")
    if not separated:
        print("FAIL -> STOP. Do not proceed to Stage 5.")
    if med_d > 0.02:
        print("\n⚠️⚠️ TRIPWIRE: a gain above +0.02 was pre-registered as a "
              "LEAKAGE signal, not a success. Re-audit the pool (including "
              "nested basins, which the 1 km rule does not screen).")

    Path(args.out).write_text(json.dumps({
        "from_date": args.from_date,
        "per_seed": {k: [{"dump": n, "median_nse": m, "n_basins": b}
                         for n, m, b in v] for k, v in per_seed.items()},
        "seed_separation": bool(separated),
        "reverse_separation": bool(max(t_meds) < min(c_meds)),
        "treatment_range": [min(t_meds), max(t_meds)],
        "control_range": [min(c_meds), max(c_meds)],
        "seed_averaged": {"n_basins": len(common), "median_nse_control": med_c,
                          "median_nse_treatment": med_t,
                          "paired_median_delta": med_d, "breadth": breadth,
                          "near_median_delta": near_d,
                          "n_near_median": int(len(near))},
        "null_calibration": {"median_delta": [-0.0127, 0.0127],
                             "breadth": [0.298, 0.702],
                             "near_median": [-0.0221, 0.0097]},
        "n_seeds": {"treatment": len(t_meds), "control": len(c_meds)},
        "min_seeds_required": args.min_seeds, "underpowered": bool(underpowered),
        "verdict": label}, indent=1))
    print(f"-> {args.out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
