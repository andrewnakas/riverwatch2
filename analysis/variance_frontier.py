#!/usr/bin/env python3
"""BEST-FORECAST TRACK — the NSE/KGE frontier. Zero GPU, post-processing only.

THE MECHANISM (this is arithmetic, not a hypothesis)
-----------------------------------------------------
Gupta et al. 2009:   NSE = 2*alpha*r - alpha^2 - beta_n^2
                     KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
with alpha = sd_sim/sd_obs, beta = mean_sim/mean_obs, beta_n = (mu_s-mu_o)/sd_o.

dNSE/dalpha = 2r - 2*alpha  =>  **the NSE optimum is at alpha = r, NOT alpha = 1.**
Minimising squared error therefore **necessarily under-disperses** the
prediction. That is precisely this campaign's central measured symptom — the
model is "ON TIME and TOO SMALL", 450/531 basins under-predict peaks
([[noq-timing-CLOSED-magnitude-not-phase]], [[peak-error-is-the-only-lever]]) —
and it has been chased as a DATA problem for months. It is a property of the
LOSS.

The exchange rate is closed-form:
    NSE(alpha=r) - NSE(alpha=1) = (1-r)^2            <- the cost, 2nd order
    KGE gains the whole (alpha-1)^2 term             <- the gain, 1st order
At r ~ 0.975 that is ~0.0006 NSE for ~0.010 KGE: **~16:1 in KGE's favour.**

⭐ WHY THIS IS NOT THE CLOSED PEAK-SCALING ARM.
[[peak-error-is-the-only-lever]] killed a per-basin HIGH-FLOW MULTIPLIER
(-0.0089, 246/531 improved = chance). That fit **531 free parameters on ~13
peak days each** — fitting scatter. This fits **one variance ratio per basin
from the FULL fit-period series** (~1,300 points), or a **single global scalar**,
and the correction is **derived** rather than tuned. Different degrees of
freedom, different object.

WHAT IS SWEPT
    pred'(k) = mu_fit + k * (pred - mu_fit),  k = 1 + lam*(1/alpha_fit - 1)
    lam = 0 -> untouched;  lam = 1 -> alpha restored to 1.
Plus a pure BIAS correction (beta -> 1), which is expected to improve **both**
NSE and KGE and is therefore free.

⛔ EVERYTHING IS FIT ON THE FIT PERIOD (<= 1990-09-30) AND SCORED ON THE VAL
SLICE. No test-window read. This is post-processing of the frozen ensemble and
must be declared as such -- it changes no model weights.

Run on the 1080:  gpu1080/.venv/bin/python -u analysis/variance_frontier.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))

from gate_eval import build_merged, L          # noqa: E402
from phase_c import load_seed_avg              # noqa: E402

THETA, LAM_W = 4.0, 0.25
FIT_END, VAL_START, VAL_END = "1990-09-30", "1990-10-01", "1995-09-30"
# NESTED SELECTION: lambda is chosen INSIDE the fit period (params on A,
# lambda scored on B) so the val slice is never used to pick it. Sweeping
# lambda and reporting the best val row is selection on the evaluation set.
SEL_A_END, SEL_B_START = "1987-09-30", "1987-10-01"
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")
HIGH_Q, MIN_HIGH = 0.90, 8
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def metrics(y, p):
    if len(y) < 20 or np.var(y) < 1e-9:
        return None
    nse = 1.0 - np.mean((y - p) ** 2) / np.var(y)
    r = np.corrcoef(y, p)[0, 1] if p.std() > 1e-12 else np.nan
    a = p.std() / y.std()
    b = p.mean() / y.mean() if abs(y.mean()) > 1e-12 else np.nan
    kge = (1.0 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
           if np.isfinite(r) and np.isfinite(b) else np.nan)
    t = np.quantile(y, HIGH_Q)
    m = y >= t
    fhv = (100.0 * (p[m].sum() - y[m].sum()) / y[m].sum()
           if m.sum() >= MIN_HIGH and y[m].sum() > 0 else np.nan)
    return {"nse": nse, "kge": kge, "r": r, "alpha": a, "beta": b, "fhv": fhv}


def add(m, name, paths):
    d = load_seed_avg([Path(p) for p in paths])[["station_id", "date", "pred"]]
    return m.merge(d.rename(columns={"pred": name}), on=["station_id", "date"],
                   how="inner")


def main():
    tr, cols = build_merged(train=True)
    cols = list(cols)
    for name, seeds in (("lstm_multi5", MULTI5), ("lstm_multi6", MULTI6)):
        key = name.split("_")[1]
        tr = add(tr, name, [L / f"camels531ls_{key}_nhlstm_TRAIN_{s}.csv.gz"
                            for s in seeds])
        cols.append(name)
    d = pd.to_datetime(tr["date"])
    fitm = (d <= FIT_END).to_numpy()
    valm = ((d >= VAL_START) & (d <= VAL_END)).to_numpy()

    truth = tr.loc[fitm, "truth"].to_numpy(float)
    mse = np.array([np.nanmean((tr.loc[fitm, c].to_numpy(float) - truth) ** 2)
                    for c in cols])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM_W * (np.ones(len(cols)) / len(cols)) + (1 - LAM_W) * raw
    w = w / w.sum()
    tr["ens"] = (tr[cols].to_numpy(float) * w).sum(axis=1)

    fit = tr.loc[fitm, ["station_id", "truth", "ens"]]
    val = tr.loc[valm, ["station_id", "truth", "ens"]]
    print(f"fit {len(fit):,} rows   val {len(val):,} rows   "
          f"{val.station_id.nunique()} basins", flush=True)

    # ---- per-basin fit-period statistics (the ONLY thing that is fitted) ----
    par = {}
    for sid, g in fit.groupby("station_id"):
        y = g["truth"].to_numpy(float)
        p = g["ens"].to_numpy(float)
        k = np.isfinite(y) & np.isfinite(p)
        y, p = y[k], p[k]
        if len(y) < 50 or y.std() < 1e-9 or p.std() < 1e-9:
            continue
        par[sid] = {"mu_p": p.mean(), "mu_y": y.mean(),
                    "alpha": p.std() / y.std(),
                    "r": np.corrcoef(y, p)[0, 1]}
    print(f"fit-period params for {len(par)} basins", flush=True)
    A = np.array([v["alpha"] for v in par.values()])
    R = np.array([v["r"] for v in par.values()])
    print(f"  median alpha {np.median(A):.4f}   median r {np.median(R):.4f}   "
          f"alpha-r {np.median(A - R):+.4f}", flush=True)
    print("  (alpha < r  => under-dispersed even beyond the NSE optimum; "
          "alpha ~ r => sitting exactly at it)", flush=True)

    rows = []

    def score(transform, label):
        recs = {}
        for sid, g in val.groupby("station_id"):
            if sid not in par:
                continue
            y = g["truth"].to_numpy(float)
            p = g["ens"].to_numpy(float)
            k = np.isfinite(y) & np.isfinite(p)
            y, p = y[k], p[k]
            if len(y) < 20:
                continue
            m = metrics(y, transform(p, par[sid]))
            if m:
                recs[sid] = m
        df = pd.DataFrame(recs).T
        rec = {"arm": label, "basins": int(len(df)),
               "median_nse": float(df["nse"].median()),
               "median_kge": float(df["kge"].median()),
               "median_abs_fhv": float(df["fhv"].abs().median()),
               "median_alpha": float(df["alpha"].median()),
               "median_beta": float(df["beta"].median()),
               "mean_nse": float(df["nse"].mean())}
        rows.append(rec)
        return rec

    base = score(lambda p, q: p, "baseline")
    print(f"\n{'arm':<26}{'medNSE':>10}{'dNSE':>10}{'medKGE':>10}{'dKGE':>9}"
          f"{'|FHV|':>9}{'alpha':>8}", flush=True)
    print("-" * 74)
    print(f"{'baseline':<26}{base['median_nse']:>10.6f}{0.0:>10.6f}"
          f"{base['median_kge']:>10.6f}{0.0:>9.6f}"
          f"{base['median_abs_fhv']:>9.3f}{base['median_alpha']:>8.4f}",
          flush=True)

    # bias correction: expected to help BOTH (free)
    rec = score(lambda p, q: p + (q["mu_y"] - q["mu_p"]), "bias-correct (beta->1)")
    print(f"{rec['arm']:<26}{rec['median_nse']:>10.6f}"
          f"{rec['median_nse'] - base['median_nse']:>+10.6f}"
          f"{rec['median_kge']:>10.6f}"
          f"{rec['median_kge'] - base['median_kge']:>+9.6f}"
          f"{rec['median_abs_fhv']:>9.3f}{rec['median_alpha']:>8.4f}", flush=True)

    # variance inflation sweep
    for lam in LAMBDAS:
        if lam == 0.0:
            continue
        def tf(p, q, lam=lam):
            k = 1.0 + lam * (1.0 / max(q["alpha"], 1e-6) - 1.0)
            return q["mu_p"] + k * (p - q["mu_p"])
        rec = score(tf, f"inflate lam={lam:.2f}")
        print(f"{rec['arm']:<26}{rec['median_nse']:>10.6f}"
              f"{rec['median_nse'] - base['median_nse']:>+10.6f}"
              f"{rec['median_kge']:>10.6f}"
              f"{rec['median_kge'] - base['median_kge']:>+9.6f}"
              f"{rec['median_abs_fhv']:>9.3f}{rec['median_alpha']:>8.4f}",
              flush=True)

    # both together
    for lam in (0.5, 1.0):
        def tf(p, q, lam=lam):
            k = 1.0 + lam * (1.0 / max(q["alpha"], 1e-6) - 1.0)
            return q["mu_y"] + k * (p - q["mu_p"])
        rec = score(tf, f"bias+inflate lam={lam:.2f}")
        print(f"{rec['arm']:<26}{rec['median_nse']:>10.6f}"
              f"{rec['median_nse'] - base['median_nse']:>+10.6f}"
              f"{rec['median_kge']:>10.6f}"
              f"{rec['median_kge'] - base['median_kge']:>+9.6f}"
              f"{rec['median_abs_fhv']:>9.3f}{rec['median_alpha']:>8.4f}",
              flush=True)

    # ---- NESTED SELECTION of lambda, entirely inside the fit period -------
    selA = tr.loc[fitm & (d <= SEL_A_END).to_numpy(), ["station_id", "truth", "ens"]]
    selB = tr.loc[fitm & (d >= SEL_B_START).to_numpy(), ["station_id", "truth", "ens"]]
    parA = {}
    for sid, g in selA.groupby("station_id"):
        y = g["truth"].to_numpy(float); q = g["ens"].to_numpy(float)
        k = np.isfinite(y) & np.isfinite(q); y, q = y[k], q[k]
        if len(y) < 50 or y.std() < 1e-9 or q.std() < 1e-9:
            continue
        parA[sid] = {"mu_p": q.mean(), "mu_y": y.mean(),
                     "alpha": q.std() / y.std(), "r": np.corrcoef(y, q)[0, 1]}

    def score_on(frame, params, transform):
        recs = {}
        for sid, g in frame.groupby("station_id"):
            if sid not in params:
                continue
            y = g["truth"].to_numpy(float); q = g["ens"].to_numpy(float)
            k = np.isfinite(y) & np.isfinite(q); y, q = y[k], q[k]
            if len(y) < 20:
                continue
            m = metrics(y, transform(q, params[sid]))
            if m:
                recs[sid] = m
        return pd.DataFrame(recs).T

    print("\n--- NESTED SELECTION (params on <=1987-09, lambda scored on "
          "1987-10..1990-09) ---", flush=True)
    best_lam, best_kge = None, -np.inf
    b0 = score_on(selB, parA, lambda q, s: q)
    for lam in LAMBDAS:
        def tf(q, s, lam=lam):
            kk = 1.0 + lam * (1.0 / max(s["alpha"], 1e-6) - 1.0)
            return s["mu_p"] + kk * (q - s["mu_p"])
        df = score_on(selB, parA, tf)
        dn = float(df["nse"].median() - b0["nse"].median())
        dk = float(df["kge"].median() - b0["kge"].median())
        star = ""
        # pick the largest KGE gain that does NOT cost NSE beyond the null
        if dn >= -0.000165 and dk > best_kge:
            best_kge, best_lam, star = dk, lam, "  <- selected"
        print(f"   lam={lam:.2f}  dNSE {dn:+.6f}  dKGE {dk:+.6f}{star}", flush=True)
    print(f"   SELECTED lambda = {best_lam} (chosen without touching the val slice)",
          flush=True)

    if best_lam is not None:
        def tf_final(q, s, lam=best_lam):
            kk = 1.0 + lam * (1.0 / max(s["alpha"], 1e-6) - 1.0)
            return s["mu_p"] + kk * (q - s["mu_p"])
        fin = score_on(val, par, tf_final)
        print(f"\n   HONEST OUT-OF-SAMPLE (val slice, lambda pre-selected):", flush=True)
        print(f"     medNSE {fin['nse'].median():.6f}  "
              f"dNSE {fin['nse'].median() - base['median_nse']:+.6f}", flush=True)
        print(f"     medKGE {fin['kge'].median():.6f}  "
              f"dKGE {fin['kge'].median() - base['median_kge']:+.6f}", flush=True)
        print(f"     |FHV|  {fin['fhv'].abs().median():.3f}  "
              f"(baseline {base['median_abs_fhv']:.3f})", flush=True)
        rows.append({"arm": f"NESTED-SELECTED lam={best_lam}",
                     "median_nse": float(fin["nse"].median()),
                     "median_kge": float(fin["kge"].median()),
                     "median_abs_fhv": float(fin["fhv"].abs().median()),
                     "selected_lambda": best_lam})

    print("\nREADING: any row with dNSE >= -0.000165 (the measured ensemble "
          "seed null)\nis 'free' in NSE terms. The NESTED-SELECTED row is the "
          "only one whose\nlambda was not chosen on the slice it is scored on.",
          flush=True)
    Path("benchmarks/ledger42_variance_frontier.json").write_text(
        json.dumps({"fit_alpha_median": float(np.median(A)),
                    "fit_r_median": float(np.median(R)),
                    "arms": rows}, indent=2, default=float))
    print("wrote benchmarks/ledger42_variance_frontier.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
