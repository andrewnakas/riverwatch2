#!/usr/bin/env python3
"""LEDGER 42 S0.3 — the MEASURED NULL for the lstm_multi recipe. Zero GPU.

`gate_l41.py` carries a null measured on the **daymet** recipe (median dNSE
-0.0127..+0.0127, breadth 29.8..70.2%, near-median -0.0221..+0.0097). Ledger
42's treatment host is **lstm_multi**, a different recipe with a different seed
spread, and reading a treatment against another recipe's null is exactly the
kind of frame-crossing this campaign keeps paying for.

This measures the lstm_multi null directly: every ordered pair of SAME-RECIPE
seeds is run through the gate's own statistics, with NO treatment present. Any
treatment effect that falls inside this envelope is indistinguishable from
"we trained it again with a different seed."

⚠️ The envelope is DESCRIPTIVE. It does not become a pass bar — the primary bar
stays complete seed separation. Its job is to stop a delta inside the noise
band from being reported as an effect.

Run on the 1080:  gpu1080/.venv/bin/python analysis/null_multi_seed_calibration.py
"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))

from gate_l41 import load, per_basin_nse, CV_VAL_START   # noqa: E402
import gate_eval as G                                     # noqa: E402

NEAR_WINDOW = 0.01
OUT = "benchmarks/ledger42_s03_null_multi.json"


def main():
    # the production seed set for lstm_multi, with the gate's own exclusions
    # (s333 collapsed at val 0.6007, s444 weak at 0.7646 -- _multi_seeds applies
    # this; using the raw glob would measure a null inflated by known-bad seeds)
    paths = list(G._multi_seeds(True))
    print(f"lstm_multi TRAIN seeds after the production gate: {len(paths)}")
    for p in paths:
        print(f"  {p.name}")
    if len(paths) < 3:
        raise SystemExit("need >=3 same-recipe seeds to measure a null")

    nse = {}
    for p in paths:
        d = load(p)
        d = d[d["date"] >= CV_VAL_START]
        nse[p.name] = per_basin_nse(d)
        print(f"  {p.name:<44} median {nse[p.name].median():.6f} "
              f"({len(nse[p.name])} basins)")

    rows = []
    for a, b in itertools.permutations(paths, 2):   # ordered: sign matters
        na, nb = nse[a.name], nse[b.name]
        common = sorted(set(na.index) & set(nb.index))
        if len(common) < 400:
            continue
        x, y = na[common], nb[common]
        d = x - y
        med_y = float(y.median())
        near = y[(y - med_y).abs() <= NEAR_WINDOW].index
        rows.append({
            "treatment_seed": a.name, "control_seed": b.name,
            "basins": len(common),
            "median_delta": float(d.median()),
            "breadth": float((d > 0).mean()),
            "near_median_delta": float(d[near].mean()) if len(near) else np.nan,
            "near_median_n": int(len(near)),
        })

    md = np.array([r["median_delta"] for r in rows])
    br = np.array([r["breadth"] for r in rows])
    nm = np.array([r["near_median_delta"] for r in rows], float)
    per_seed_med = sorted(float(v.median()) for v in nse.values())

    env = {
        "recipe": "lstm_multi",
        "seeds": [p.name for p in paths],
        "n_pairs": len(rows),
        "per_seed_medians": per_seed_med,
        "per_seed_median_spread": per_seed_med[-1] - per_seed_med[0],
        "median_delta_range": [float(md.min()), float(md.max())],
        "median_delta_abs_max": float(np.abs(md).max()),
        "breadth_range": [float(br.min()), float(br.max())],
        "near_median_delta_range": [float(np.nanmin(nm)), float(np.nanmax(nm))],
        "near_median_abs_max": float(np.nanmax(np.abs(nm))),
        "pairs": rows,
    }

    print("\n" + "=" * 74)
    print("MEASURED NULL — same recipe, different seed, NO treatment")
    print("=" * 74)
    print(f"  per-seed medians      : "
          f"{', '.join(f'{m:.6f}' for m in per_seed_med)}")
    print(f"  per-seed median SPREAD: {env['per_seed_median_spread']:.6f}")
    print(f"  median delta          : [{md.min():+.6f}, {md.max():+.6f}]")
    print(f"  breadth               : [{br.min():.1%}, {br.max():.1%}]")
    print(f"  near-median delta     : [{np.nanmin(nm):+.6f}, {np.nanmax(nm):+.6f}]")
    print("\n⚠️ READ THIS BEFORE QUOTING ANY LEDGER-42 DELTA:")
    print(f"   a treatment effect smaller than ±{env['median_delta_abs_max']:.6f} "
          f"(median) or ±{env['near_median_abs_max']:.6f} (near-median)")
    print("   is INSIDE what re-running the same recipe with a new seed produces.")
    print(f"   The ship bar near-median >= +0.0008 is "
          f"{'INSIDE' if env['near_median_abs_max'] >= 0.0008 else 'outside'} "
          f"this null envelope.")

    Path(OUT).write_text(json.dumps(env, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
