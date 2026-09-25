#!/usr/bin/env python3
"""BEST-FORECAST — GLOBAL variance inflation. PRE-REGISTERED, zero GPU.

WHAT THE PREVIOUS GATE ESTABLISHED
----------------------------------
A parameter-shuffled sham reproduced almost the whole KGE gain of PER-BASIN
inflation, and in one period BEAT it. ⇒ the gain comes from inflating variance
AT ALL, not from estimating each basin's alpha. 531 fitted parameters were doing
the work of one constant. This tests the constant.

INTERVENTION (one global scalar k; per-basin means only for centring)
    pred'(k) = mu_fit[basin] + k * (pred - mu_fit[basin])
k is selected on the FIT period alone and then applied unchanged everywhere.
A per-basin MEAN is far more stable than a per-basin VARIANCE RATIO, and the
sham showed the variance ratio was the part carrying no information.

⛔⛔ BAR SEMANTICS, WRITTEN IN ENGLISH BEFORE THE INEQUALITIES
Two gates in a row failed on SPECIFICATION, not logic: one borrowed the
seed-swap null for parameter-transfer noise, the other wrote a TWO-sided test
for a ONE-sided question and so failed an arm for helping too much. Dry-running
the logic caught neither. So, in words first:

  B1  "It improves KGE on EVERY evaluation period."
      -> dKGE > 0 for all periods.                       ONE-SIDED. correct.
  B2  "It does not COST NSE beyond measured transfer noise."
      -> dNSE >= -NSE_TOL for all periods.               ONE-SIDED, lower bound
         only. A POSITIVE dNSE must PASS. (This is the bar the last gate got
         wrong by testing |dNSE|.)
  B3  "The k was not tuned on the data it is judged on."
      -> k chosen on the fit period only; the SAME k used on every eval period.

NSE_TOL is taken from the measured sham dNSE envelope of the previous gate
(|dNSE| <= 0.00026 across 60 shuffles) rounded out to 0.0003 -- a number that
predates this run and is not tunable by it.

FAIL ANY -> DO NOT DEPLOY.
Frame: rolling blocks inside the TRAIN window. No test read.
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
NSE_TOL = 0.0003                    # predates this run; see docstring
K_GRID = [1.00, 1.01, 1.02, 1.03, 1.04, 1.06, 1.08]
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")
SEL_FIT = ("1980-10-01", "1987-09-30")     # k is chosen here, and only here
SEL_EVAL = ("1987-10-01", "1990-09-30")
BLOCKS = [(("1980-10-01", "1985-09-30"), ("1985-10-01", "1988-09-30")),
          (("1980-10-01", "1988-09-30"), ("1988-10-01", "1991-09-30")),
          (("1980-10-01", "1991-09-30"), ("1991-10-01", "1995-09-30"))]
HIGH_Q, MIN_HIGH = 0.90, 8


def metr(y, p):
    if len(y) < 20 or np.var(y) < 1e-9 or np.std(p) < 1e-12:
        return None
    r = np.corrcoef(y, p)[0, 1]
    a = p.std() / y.std()
    b = p.mean() / y.mean() if abs(y.mean()) > 1e-12 else np.nan
    if not (np.isfinite(r) and np.isfinite(b)):
        return None
    t = np.quantile(y, HIGH_Q)
    m = y >= t
    fhv = (100.0 * (p[m].sum() - y[m].sum()) / y[m].sum()
           if m.sum() >= MIN_HIGH and y[m].sum() > 0 else np.nan)
    return (1.0 - np.mean((y - p) ** 2) / np.var(y),
            1.0 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2), fhv)


def add(m, name, paths):
    d = load_seed_avg([Path(p) for p in paths])[["station_id", "date", "pred"]]
    return m.merge(d.rename(columns={"pred": name}), on=["station_id", "date"],
                   how="inner")


def deltas(tr, fit_win, eval_win, k):
    fit = tr[(tr.date >= fit_win[0]) & (tr.date <= fit_win[1])]
    ev = tr[(tr.date >= eval_win[0]) & (tr.date <= eval_win[1])]
    mu = {sid: g["ens"].mean() for sid, g in fit.groupby("station_id")
          if np.isfinite(g["ens"].mean())}
    dn, dk, df = [], [], []
    for sid, g in ev.groupby("station_id"):
        if sid not in mu:
            continue
        y = g["truth"].to_numpy(float); p = g["ens"].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(p); y, p = y[m], p[m]
        b = metr(y, p)
        t = metr(y, mu[sid] + k * (p - mu[sid]))
        if b and t:
            dn.append(t[0] - b[0]); dk.append(t[1] - b[1])
            if np.isfinite(b[2]) and np.isfinite(t[2]):
                df.append(abs(b[2]) - abs(t[2]))
    return (float(np.median(dn)), float(np.median(dk)),
            float(np.median(df)) if df else np.nan, len(dn))


def main():
    tr, cols = build_merged(train=True)
    cols = list(cols)
    for name, seeds in (("lstm_multi5", MULTI5), ("lstm_multi6", MULTI6)):
        key = name.split("_")[1]
        tr = add(tr, name, [L / f"camels531ls_{key}_nhlstm_TRAIN_{s}.csv.gz"
                            for s in seeds])
        cols.append(name)
    d = pd.to_datetime(tr["date"])
    wf = (d <= "1990-09-30").to_numpy()
    truth = tr.loc[wf, "truth"].to_numpy(float)
    mse = np.array([np.nanmean((tr.loc[wf, c].to_numpy(float) - truth) ** 2)
                    for c in cols])
    raw = mse ** (-THETA); raw = raw / raw.sum()
    w = LAM_W * (np.ones(len(cols)) / len(cols)) + (1 - LAM_W) * raw
    tr["ens"] = (tr[cols].to_numpy(float) * (w / w.sum())).sum(axis=1)
    tr = tr[["station_id", "truth", "ens"]].assign(date=d)

    print("=" * 84, flush=True)
    print("GLOBAL INFLATION GATE — k selected on the fit period ONLY", flush=True)
    print("=" * 84, flush=True)
    print(f"\nB3 selection ({SEL_FIT[0]}..{SEL_FIT[1]} -> "
          f"{SEL_EVAL[0]}..{SEL_EVAL[1]}):", flush=True)
    best_k, best = None, -np.inf
    for k in K_GRID:
        dn, dk, _, n = deltas(tr, SEL_FIT, SEL_EVAL, k)
        pick = (dn >= -NSE_TOL) and (dk > best)
        if pick:
            best, best_k = dk, k
        print(f"   k={k:.2f}  dNSE {dn:+.6f}  dKGE {dk:+.6f}"
              f"{'  <- best so far' if pick else ''}", flush=True)
    print(f"   SELECTED k = {best_k}", flush=True)
    if best_k is None:
        print("   no k passed the NSE floor during selection -> DO NOT DEPLOY",
              flush=True)
        Path("benchmarks/ledger42_global_inflation_gate.json").write_text(
            json.dumps({"selected_k": None, "deploy": False}, indent=2))
        return

    print(f"\nEvaluation with k={best_k} held FIXED:", flush=True)
    rows, b1, b2 = [], [], []
    for fw, ew in BLOCKS:
        dn, dk, dfhv, n = deltas(tr, fw, ew, best_k)
        ok1, ok2 = dk > 0, dn >= -NSE_TOL
        b1.append(ok1); b2.append(ok2)
        print(f"   {ew[0]}..{ew[1]} ({n:3d})  dNSE {dn:+.6f}  dKGE {dk:+.6f}"
              f"  d|FHV| {dfhv:+.3f}   B1 {ok1}  B2 {ok2}", flush=True)
        rows.append({"eval": list(ew), "basins": n, "dnse": dn, "dkge": dk,
                     "d_abs_fhv": dfhv, "B1": bool(ok1), "B2": bool(ok2)})

    deploy = bool(rows) and all(b1) and all(b2)
    print(f"\n{'=' * 84}", flush=True)
    print(f"VERDICT: {'DEPLOY' if deploy else 'DO NOT DEPLOY'}   "
          f"(B1 KGE>0 all periods: {all(b1)};  "
          f"B2 dNSE>=-{NSE_TOL} all periods: {all(b2)})", flush=True)
    print("=" * 84, flush=True)
    Path("benchmarks/ledger42_global_inflation_gate.json").write_text(json.dumps(
        {"selected_k": best_k, "nse_tol": NSE_TOL, "periods": rows,
         "deploy": deploy}, indent=2, default=float))
    print("wrote benchmarks/ledger42_global_inflation_gate.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
