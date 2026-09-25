#!/usr/bin/env python3
"""LEDGER 42 S0.3b — is the SHIP BAR discriminable? Measure its own null.

S0.3 measured the lstm_multi STREAM-level seed null and found the ship bar
(near-median Δ >= +0.0008) sits INSIDE it: near-median seed noise spans
-0.0061..+0.0047, roughly 6x the bar. That would make the bar meaningless --
except the bar is not applied to the stream. It is applied to the RECOMBINED
9-STREAM ENSEMBLE, where one stream carries weight ~0.131 and seeds are
averaged, both of which damp the noise.

So the stream-level null does NOT settle it, and the ensemble-level null has
never been measured. Asserting "the weight damps it by 0.131/sqrt(3)" would be
a causal claim of exactly the kind this campaign has had overturned 20+ times.
This MEASURES it, by doing to the ensemble precisely what the gate will do:

  for each pair of same-recipe lstm_multi seeds (NO treatment anywhere):
      swap seed A's dumps into the frozen 9-stream in place of lstm_multi
      refit the ONE GLOBAL inverse-MSE vector on train rows
      score the val slice; same for seed B; take the delta
  -> the distribution of ensemble median / near-median deltas under a pure
     seed change is the null the ship bar must clear.

⚠️ This is a VALIDITY check on a pre-registered bar, run BEFORE the treatment
exists, so it cannot be tuned to any result. If the bar is inside its own null,
that is recorded as a defect in the bar -- the bar is NOT moved.

Run on the 1080:  gpu1080/.venv/bin/python analysis/null_ensemble_ship_bar.py
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

from gate_eval import build_merged, L                 # noqa: E402
from phase_c import load_seed_avg                     # noqa: E402
import gate_eval as G                                 # noqa: E402

THETA, LAM = 4.0, 0.25
CV_FIT_END, CV_VAL_START, CV_VAL_END = "1990-09-30", "1990-10-01", "1995-09-30"
ANCHOR = 0.950458
NEAR_WINDOW = 0.01
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")
SHIP_BAR_NEAR_MEDIAN = 0.0008
SHIP_BAR_MEDIAN = 0.954458


def per_basin_nse(df, col):
    out = {}
    for sid, g in df.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = g[col].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(p)
        y, p = y[m], p[m]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        out[sid] = 1.0 - np.mean((y - p) ** 2) / np.var(y)
    return pd.Series(out)


def add(merged, name, paths):
    d = load_seed_avg([Path(p) for p in paths])[["station_id", "date", "pred"]]
    return merged.merge(d.rename(columns={"pred": name}),
                        on=["station_id", "date"], how="inner")


def build(multi_paths):
    """The frozen 9 streams on the TRAIN frame, with lstm_multi supplied by the
    caller (this is exactly the substitution the gate performs)."""
    tr, cols = build_merged(train=True, drop=("lstm_multi",))
    tr = add(tr, "lstm_multi", multi_paths)
    tr = add(tr, "lstm_multi5",
             [L / f"camels531ls_multi5_nhlstm_TRAIN_{s}.csv.gz" for s in MULTI5])
    tr = add(tr, "lstm_multi6",
             [L / f"camels531ls_multi6_nhlstm_TRAIN_{s}.csv.gz" for s in MULTI6])
    cols = list(cols) + ["lstm_multi", "lstm_multi5", "lstm_multi6"]
    return tr, cols


def score(tr, cols):
    d = pd.to_datetime(tr["date"])
    fit = tr[d <= CV_FIT_END]
    val = tr[(d >= CV_VAL_START) & (d <= CV_VAL_END)].copy()
    truth = fit["truth"].to_numpy(float)
    mse = np.array([np.nanmean((fit[c].to_numpy(float) - truth) ** 2) for c in cols])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM * (np.ones(len(cols)) / len(cols)) + (1 - LAM) * raw
    w = w / w.sum()
    val["ens"] = (val[cols].to_numpy(float) * w).sum(axis=1)
    return per_basin_nse(val, "ens"), dict(zip(cols, w))


def main():
    paths = list(G._multi_seeds(True))
    print(f"same-recipe lstm_multi seeds: {len(paths)}")
    for p in paths:
        print(f"  {p.name}")
    if len(paths) < 3:
        raise SystemExit("need >=3 seeds")

    # anchor: the production ensemble uses ALL seeds averaged
    s_all, w_all = score(*build(paths))
    print(f"\nANCHOR all-seed ensemble median = {s_all.median():.6f}  "
          f"(prereg {ANCHOR:.6f}, diff {s_all.median() - ANCHOR:+.6f})")
    if abs(s_all.median() - ANCHOR) > 0.0005:
        raise SystemExit("anchor missed — frame is wrong, refusing to report")
    print(f"  lstm_multi weight = {w_all['lstm_multi']:.4f}")

    # single-seed ensembles: the null
    per = {}
    for p in paths:
        s, _ = score(*build([p]))
        per[p.name] = s
        print(f"  ensemble with ONLY {p.name:<40} median {s.median():.6f}")

    rows = []
    for a, b in itertools.permutations(paths, 2):
        sa, sb = per[a.name], per[b.name]
        common = sorted(set(sa.index) & set(sb.index))
        x, y = sa[common], sb[common]
        d = x - y
        med_y = float(y.median())
        near = y[(y - med_y).abs() <= NEAR_WINDOW].index
        rows.append({"a": a.name, "b": b.name,
                     "median_delta": float(d.median()),
                     "near_median_delta": float(d[near].mean()) if len(near) else np.nan,
                     "near_n": int(len(near))})

    md = np.array([r["median_delta"] for r in rows])
    nm = np.array([r["near_median_delta"] for r in rows], float)
    ens_meds = sorted(float(v.median()) for v in per.values())

    print("\n" + "=" * 76)
    print("ENSEMBLE-LEVEL NULL — one stream swapped for another SEED of itself")
    print("=" * 76)
    print(f"  ensemble medians       : "
          f"{', '.join(f'{m:.6f}' for m in ens_meds)}")
    print(f"  ensemble median spread : {ens_meds[-1] - ens_meds[0]:.6f}")
    print(f"  median delta null      : [{md.min():+.6f}, {md.max():+.6f}]")
    print(f"  near-median delta null : [{np.nanmin(nm):+.6f}, {np.nanmax(nm):+.6f}]")

    nm_abs = float(np.nanmax(np.abs(nm)))
    md_abs = float(np.abs(md).max())
    inside = nm_abs >= SHIP_BAR_NEAR_MEDIAN
    print(f"\n  SHIP BAR near-median >= +{SHIP_BAR_NEAR_MEDIAN}")
    print(f"    null |near-median| max = {nm_abs:.6f}  => the bar is "
          f"{'INSIDE its own null (NOT discriminable at 1 seed)' if inside else 'OUTSIDE the null (discriminable)'}")
    print(f"  SHIP BAR median >= {SHIP_BAR_MEDIAN} "
          f"(anchor {ANCHOR} + 0.004)")
    print(f"    null |median delta| max = {md_abs:.6f}  => a +0.004 move is "
          f"{md_abs / 0.004:.2f}x the single-seed null")
    print("\n  ⚠️ These are 1-seed-vs-1-seed nulls. The gate averages 3 seeds "
          "per arm,\n     which shrinks seed noise by roughly sqrt(3) but does "
          "NOT remove it.\n     Bars are NOT moved on this evidence; it is "
          "recorded as the context every\n     ledger-42 delta must be read "
          "against.")

    Path("benchmarks/ledger42_s03b_null_ensemble.json").write_text(json.dumps({
        "anchor": float(s_all.median()), "prereg_anchor": ANCHOR,
        "lstm_multi_weight": float(w_all["lstm_multi"]),
        "ensemble_medians_single_seed": ens_meds,
        "ensemble_median_spread": ens_meds[-1] - ens_meds[0],
        "median_delta_null": [float(md.min()), float(md.max())],
        "near_median_delta_null": [float(np.nanmin(nm)), float(np.nanmax(nm))],
        "ship_bar_near_median": SHIP_BAR_NEAR_MEDIAN,
        "ship_bar_inside_own_null_at_1seed": bool(inside),
        "pairs": rows,
    }, indent=2, default=float))
    print("\nwrote benchmarks/ledger42_s03b_null_ensemble.json")


if __name__ == "__main__":
    sys.exit(main())
