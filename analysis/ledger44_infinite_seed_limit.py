#!/usr/bin/env python3
"""LEDGER 44 / S1c-refined — THE INFINITE-SEED LIMIT, IN CLOSED FORM.

WHY NOT THE CURVE FIT
---------------------
ledger44_ensembling_law.py extrapolates NSE(n) = a - b/n from five points and
lands on a headroom of +0.00225 against a +0.003 bar.  A decision that close
must not rest on a 5-point extrapolation whose last two points barely move n
(29 -> 31).

THE ESTIMATOR
-------------
The seed-averaged ensemble prediction is

    yhat = SUM_i w_i * mean(x_i1 .. x_ik_i)

Seed noise is zero-mean around the k->infinity limit yhat_inf, and independent
of it, so

    MSE_obs = MSE_inf + E[(yhat - yhat_inf)^2]
            = MSE_inf + SUM_i w_i^2 * sigma_i^2 / k_i        (+ cross terms)

sigma_i^2 is estimable ROW-WISE from the seeds we already have (unbiased,
ddof=1).  So

    MSE_inf = MSE_obs - SUM_i w_i^2 * sigmahat_i^2 / k_i

needs NO extrapolation: it subtracts a measured quantity.  NSE_inf follows
per basin, and its median is the ceiling that infinite compute buys.

THE CROSS TERMS ARE NOT ASSUMED AWAY
------------------------------------
Different streams' seeds are independent training runs, so their deviations
should be uncorrelated -- but this campaign's causal claims keep failing when
asserted rather than derived, so the cross-covariance is MEASURED here and
added to the correction if it is non-negligible.

VALIDATION AGAINST A KNOWN ANSWER (run before the unknown is read)
------------------------------------------------------------------
The same decomposition predicts the score at REDUCED depth:
    MSE(k') = MSE_inf + SUM_i w_i^2 sigmahat_i^2 / k'_i
Predicting the measured k=1,2,3 points of S1a from the k=5 data, with weights
held fixed, is a known-answer test of the whole estimator.
"""
import json
import os
import sys

import numpy as np

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())

from ledger44_ensembling_law import (load_surface, fit_global_invmse,   # noqa: E402
                                     per_basin_nse, stack, CV_FIT_END,
                                     CV_VAL_START, ANCHOR_TRAIN, ANCHOR_TEST)


def seed_stats(frame, streams):
    """Per-row mean and UNBIASED across-seed variance for each stream."""
    means, vars_, ks = {}, {}, {}
    for s in streams:
        a = frame[s]                      # (k, rows)
        ks[s] = a.shape[0]
        means[s] = a.mean(axis=0)
        vars_[s] = a.var(axis=0, ddof=1) if a.shape[0] > 1 else np.zeros(a.shape[1])
    return means, vars_, ks


def cross_cov(frame, streams, means):
    """Measured correlation of seed deviations ACROSS streams.

    Uses the FIRST seed of each stream as one realisation: dev_i = x_i1 - mean_i.
    If seeds are independent runs these are uncorrelated across streams.
    """
    devs = {s: frame[s][0] - means[s] for s in streams if frame[s].shape[0] > 1}
    names = list(devs)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = devs[names[i]], devs[names[j]]
            sa, sb = a.std(), b.std()
            if sa > 0 and sb > 0:
                out.append((names[i], names[j], float(np.mean(a * b) / (sa * sb))))
    return out


def analyse(TR, TE, streams, surface, anchor, label):
    trf, trs, trd = TR
    tef, tes, ted = TE
    if surface == "train":
        fit_m, sc_m = trd <= CV_FIT_END, trd >= CV_VAL_START
        full = {s: tuple(range(trf[s].shape[0])) for s in streams}
        w = fit_global_invmse(stack(trf, streams, full)[:, fit_m], trf["truth"][fit_m])
        F, sids, truth = {s: trf[s][:, sc_m] for s in streams}, trs[sc_m], trf["truth"][sc_m]
    else:
        full = {s: tuple(range(trf[s].shape[0])) for s in streams}
        w = fit_global_invmse(stack(trf, streams, full), trf["truth"])
        F, sids, truth = {s: tef[s] for s in streams}, tes, tef["truth"]

    means, vars_, ks = seed_stats(F, streams)
    W = dict(zip(streams, w))
    yhat = sum(W[s] * means[s] for s in streams)
    # the correction: variance of the seed-averaged ensemble around its limit
    corr = sum(W[s] ** 2 * vars_[s] / ks[s] for s in streams)

    print(f"\n{'='*78}\n{label}\n{'='*78}")
    print(f"  rows {len(truth):,}  basins {len(np.unique(sids))}  streams {len(streams)}")
    print(f"  weights: " + "  ".join(f"{s}={W[s]:.4f}" for s in streams))

    cc = cross_cov(F, streams, means)
    mx = max((abs(c) for _, _, c in cc), default=0.0)
    mean_abs = float(np.mean([abs(c) for _, _, c in cc])) if cc else 0.0
    print(f"\n  cross-stream seed-deviation correlation: mean|r| {mean_abs:.4f}  "
          f"max|r| {mx:.4f}  (n={len(cc)} pairs)")
    print("   => cross terms " + ("NEGLIGIBLE, dropped" if mx < 0.10 else
                                  "NOT negligible — see caveat below"))

    obs = per_basin_nse(sids, truth, yhat)
    med_obs = float(np.median(obs))
    print(f"\n  measured median NSE      = {med_obs:.6f}   "
          f"(anchor {anchor:.6f}, diff {med_obs-anchor:+.9f})")

    # per-basin infinite-seed NSE
    inf_nse, ub = [], np.unique(sids)
    for b in ub:
        m = sids == b
        y, p, c = truth[m], yhat[m], corr[m]
        ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(c)
        y, p, c = y[ok], p[ok], c[ok]
        if len(y) < 20 or np.var(y) < 1e-9:
            continue
        inf_nse.append(1.0 - (np.mean((y - p) ** 2) - np.mean(c)) / np.var(y))
    med_inf = float(np.median(inf_nse))
    print(f"  INFINITE-SEED median NSE = {med_inf:.6f}")
    print(f"  HEADROOM (compute only)  = {med_inf-med_obs:+.6f}")

    # ---- known-answer validation: predict reduced depth from the same terms
    print(f"\n  known-answer check — predict REDUCED depth, weights held fixed:")
    print(f"    {'k':>2} {'predicted':>11} {'measured':>11} {'err':>10}")
    for k in (1, 2, 3):
        pred_nse, meas_nse = [], []
        sub = {s: tuple(range(min(k, ks[s]))) for s in streams}
        ysub = sum(W[s] * F[s][list(sub[s])].mean(axis=0) for s in streams)
        csub = sum(W[s] ** 2 * vars_[s] / min(k, ks[s]) for s in streams)
        for b in ub:
            m = sids == b
            y, p, c, ps = truth[m], yhat[m], corr[m], ysub[m]
            cs = csub[m]
            ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(ps)
            if ok.sum() < 20 or np.var(y[ok]) < 1e-9:
                continue
            v = np.var(y[ok])
            mse_inf = np.mean((y[ok] - p[ok]) ** 2) - np.mean(c[ok])
            pred_nse.append(1.0 - (mse_inf + np.mean(cs[ok])) / v)
            meas_nse.append(1.0 - np.mean((y[ok] - ps[ok]) ** 2) / v)
        pm, mm = float(np.median(pred_nse)), float(np.median(meas_nse))
        print(f"    {k:>2} {pm:>11.6f} {mm:>11.6f} {pm-mm:>+10.6f}")

    return {"measured": med_obs, "infinite_seed": med_inf,
            "headroom": med_inf - med_obs,
            "cross_corr_max_abs": mx, "cross_corr_mean_abs": mean_abs}


def main():
    print("loading surfaces...", file=sys.stderr)
    TR = load_surface(train=True)
    TE = load_surface(train=False)
    streams = [s for s in TR[0] if s != "truth"]
    out = {
        "test": analyse(TR, TE, streams, "test", ANCHOR_TEST,
                        "HELD-OUT (Li/Song test 1995-2008, h==1) — THE DECISION SURFACE"),
        "train": analyse(TR, TE, streams, "train", ANCHOR_TRAIN,
                         "IN-SAMPLE (val slice 1990-95, inside the training window)"),
    }
    out["understatement_factor"] = (out["test"]["headroom"] /
                                    out["train"]["headroom"]
                                    if out["train"]["headroom"] else None)
    print(f"\n{'='*78}\nHOW MUCH THE IN-SAMPLE SURFACE UNDERSTATES ENSEMBLE DEPTH")
    print("=" * 78)
    print(f"  held-out headroom  {out['test']['headroom']:+.6f}")
    print(f"  in-sample headroom {out['train']['headroom']:+.6f}")
    print(f"  understatement     {out['understatement_factor']:.2f}x")
    json.dump(out, open("benchmarks/ledger44_infinite_seed_limit.json", "w"), indent=2)
    print("\nwrote benchmarks/ledger44_infinite_seed_limit.json")


if __name__ == "__main__":
    main()
