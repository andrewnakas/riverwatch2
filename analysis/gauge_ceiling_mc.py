#!/usr/bin/env python3
"""Estimate the OBSERVATIONAL CEILING on NSE from discharge-gauge uncertainty.

WHY THIS IS OURS TO DERIVE
--------------------------
A literature search with primary-source verification found NO paper that
converts discharge uncertainty into a maximum-achievable NSE. The closest work
(Aerts et al. 2024, HESS 28:5011) does per-catchment significance testing on
KGE-NP, not a ceiling. So this number cannot be cited -- it must be derived, and
deriving it is itself publishable.

METHOD
------
USGS daily discharge is not measured; it is stage read through a RATING CURVE.
So the "truth" column in our dumps carries error. We perturb the observed series
within a literature-calibrated, FLOW-DEPENDENT error model and recompute NSE of
the *unperturbed* truth against the perturbed truth. That is the NSE a PERFECT
model would score against a noisy gauge -- i.e. the ceiling.

ERROR MODEL (all verified primary sources)
  Coxon et al. 2015 (WRR 51:5531, 500 UK gauges): 95% limits +-25% at low flow
    narrowing to +-13% at the highest normalised flows.
  Westerberg & McMillan 2015 (HESS 19:3951): high-flow Q0.1 half-width 19.6-22.8%.
  Aerts et al. 2024 (HESS 28:5011, CAMELS-GB): median 20% low / 15% avg / 12% high.
  Kiang et al. 2018 (WRR 54:7149): 41-200% where the rating is EXTRAPOLATED.

We interpolate sigma linearly in log-flow between the low- and high-flow anchors
and apply MULTIPLICATIVE lognormal noise (discharge error is relative, and
lognormal keeps it positive).

THE PART THE LITERATURE INSISTS ON
----------------------------------
Coxon could not even ASSESS high-flow uncertainty at 44% of gauges because the
rating was extrapolated beyond the highest direct gauging. Westerberg &
McMillan: that happens where "peak flows occur seldom and last only a few
hours" -- which is exactly our failing cohort (flashy, ephemeral, arid,
p99/median 54.9). So we run THREE scenarios, not one:

  optimistic  -- everyone well gauged           (13% high / 25% low)
  literature  -- Coxon central                  (+ flashy basins degraded)
  extrapolated-- flashy basins in Kiang's 41-200% regime

Reporting only the optimistic scenario would understate the floor in precisely
the cohort that dominates our error budget.
"""
import argparse

import numpy as np
import pandas as pd

SCENARIOS = {
    # name: (sigma_low, sigma_high, sigma_high_for_flashy_basins)
    "optimistic":   (0.25, 0.13, 0.13),
    "literature":   (0.30, 0.18, 0.35),
    "extrapolated": (0.40, 0.20, 0.70),
}


def nse(truth, pred):
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(truth) & np.isfinite(pred)
    if ok.sum() < 10:
        return np.nan
    truth, pred = truth[ok], pred[ok]
    den = ((truth - truth.mean()) ** 2).sum()
    if den <= 0:
        return np.nan
    return 1.0 - ((pred - truth) ** 2).sum() / den


def sigma_for(q, s_low, s_high):
    """Relative sigma, interpolated in log-flow: high flow -> s_high."""
    q = np.asarray(q, float)
    pos = q[q > 0]
    if pos.size < 10:
        return np.full(q.shape, (s_low + s_high) / 2)
    lo, hi = np.log(np.percentile(pos, 5)), np.log(np.percentile(pos, 95))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full(q.shape, (s_low + s_high) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        lq = np.log(np.where(q > 0, q, np.nan))
    f = np.clip((lq - lo) / (hi - lo), 0.0, 1.0)      # 0 at low flow, 1 at high
    f = np.where(np.isfinite(f), f, 0.5)
    return s_low + f * (s_high - s_low)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True,
                    help="a TEST dump; only its truth column is used")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = pd.read_csv(a.dump, usecols=["station_id", "t0", "h", "truth"])
    d = d[d.h == 1].dropna(subset=["truth"])
    print(f"loaded {len(d):,} day-1 rows, {d.station_id.nunique()} basins\n")

    # flashiness per basin -- the literature's extrapolation risk proxy
    flash = d.groupby("station_id")["truth"].apply(
        lambda s: (np.percentile(s[s > 0], 99) / np.median(s[s > 0]))
        if (s > 0).sum() > 20 else np.nan)
    thr = flash.median()
    flashy = set(flash[flash > thr].index)
    print(f"flashiness (p99/median): median {thr:.1f}; "
          f"{len(flashy)} basins above it treated as extrapolation-prone\n")

    rng = np.random.default_rng(a.seed)
    groups = list(d.groupby("station_id"))

    print(f"{'scenario':<14}{'ceiling NSE':>13}{'flashy':>10}{'well-gauged':>13}")
    print("-" * 50)
    for name, (s_lo, s_hi, s_hi_flashy) in SCENARIOS.items():
        per_basin, per_flashy, per_well = [], [], []
        for sid, g in groups:
            q = g["truth"].to_numpy(float)
            hi = s_hi_flashy if sid in flashy else s_hi
            sig = sigma_for(q, s_lo, hi)
            # lognormal multiplicative noise, median-unbiased
            vals = []
            for _ in range(a.reps):
                z = rng.normal(size=q.shape)
                obs = q * np.exp(sig * z - 0.5 * sig ** 2)
                vals.append(nse(q, obs))
            v = float(np.nanmean(vals))
            per_basin.append(v)
            (per_flashy if sid in flashy else per_well).append(v)
        med = np.nanmedian(per_basin)
        print(f"{name:<14}{med:>13.4f}{np.nanmedian(per_flashy):>10.4f}"
              f"{np.nanmedian(per_well):>13.4f}")

    print("\nInterpretation: this is the median NSE a PERFECT model would score")
    print("against a gauge carrying this much error. Our targets are no-q 0.845")
    print("and with-q 0.95 -- compare directly.")
    print("\nNOTE: our own derivation, NOT a literature value. No published paper")
    print("states a max-achievable NSE (verified). Error model is calibrated to")
    print("Coxon 2015 / Kiang 2018 / Westerberg 2015, whose multi-gauge base")
    print("rates are UK, not USGS.")


if __name__ == "__main__":
    main()
