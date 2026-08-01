#!/usr/bin/env python3
"""Close the KGE gap to Nearing 2022 by fixing UNDER-DISPERSION.

Verified numbers (Nearing et al. 2022, HESS 26:5493, Table 2, median over 531
CAMELS basins at 1-day lag) -- read from the paper PDF, not from memory:
    NSE 0.879 | KGE 0.896 | alpha 0.942 | beta -0.007 | Pearson r 0.939

Ours (with-q 4-seed grand, day-1):
    NSE 0.9034 | KGE 0.8577 | r 0.9599 | pct_bias -1.305%

So we BEAT them on NSE (+0.024) and on correlation (+0.021) but LOSE on KGE
(-0.038). Decomposing KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2) with
our r and beta pins the loss on alpha: implied alpha ~= 0.864 vs their 0.942.
Our predictions are ~14% UNDER-DISPERSED.

That is the expected cost of ensemble averaging: averaging N seeds shrinks
predictive variance. It raises NSE (which rewards smoothing) and lowers KGE
(which explicitly penalises a variability ratio away from 1). It is a structural
artefact of the method, not a bug.

FIX: variance inflation. For each basin, rescale predictions about their own
mean so the simulated standard deviation matches the observed one:
    q' = mean(q) + k * (q - mean(q))
with k chosen per basin. k = sigma_obs/sigma_sim gives alpha = 1 exactly.

DEPLOYABILITY: k must be fit on TRAIN-period data, never on test observations.
This script therefore:
  1. reports the ORACLE ceiling (k from test) -- what perfect inflation buys;
  2. reports the DEPLOYABLE result (k from the train window 1999-2008, the
     Nearing training period) -- what we can actually claim;
  3. reports a single GLOBAL k (one constant for all basins), the most
     conservative option with one fitted scalar.
Anything fit on test is labelled oracle, as everywhere else in this campaign.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data/mblstm/gpu_dumps_s14")
FORCINGS = ("daymet", "maurer", "nldas")
NEARING = {"nse": 0.879, "kge": 0.896, "alpha": 0.942, "beta": -0.007, "r": 0.939}


def load(tag="4"):
    out = None
    for f in FORCINGS:
        p = D / f"camels531_{f}_withq{tag}_full531.csv.gz"
        if not p.exists():
            print(f"missing {p}")
            return None
        df = pd.read_csv(p, usecols=["station_id", "t0", "h", "truth", "ylo", "yhi"])
        df = df[df.h == 1].copy()
        df["station_id"] = df["station_id"].astype(str).str.zfill(8)
        df[f] = (df["ylo"] + df["yhi"]) / 2.0
        d = df[["station_id", "t0", "truth", f]]
        out = d if out is None else out.merge(d[["station_id", "t0", f]],
                                              on=["station_id", "t0"])
    out["pred"] = out[list(FORCINGS)].to_numpy(float).mean(1)
    return out.reset_index(drop=True)


def kge_parts(y, p):
    if len(y) < 20 or np.std(y) < 1e-9 or np.std(p) < 1e-9:
        return None
    r = float(np.corrcoef(y, p)[0, 1])
    a = float(np.std(p) / np.std(y))
    b = float(np.mean(p) / np.mean(y)) if abs(np.mean(y)) > 1e-9 else np.nan
    k = 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
    nse = 1 - np.mean((y - p) ** 2) / np.var(y)
    return dict(kge=k, nse=nse, r=r, alpha=a, beta=b)


def summarise(df, col, label):
    rows = []
    for sid, g in df.groupby("station_id"):
        m = kge_parts(g["truth"].to_numpy(float), g[col].to_numpy(float))
        if m:
            rows.append(m)
    if not rows:
        return None
    t = pd.DataFrame(rows).median()
    print(f"  {label:34s} KGE {t.kge:.4f} | NSE {t.nse:.4f} | "
          f"alpha {t.alpha:.4f} | r {t.r:.4f} | beta {t.beta:.4f}")
    return t


def main():
    df = load("4")
    if df is None:
        return
    print(f"rows={len(df)} basins={df.station_id.nunique()}\n")
    print(f"  {'Nearing 2022 AR (Table 2)':34s} KGE {NEARING['kge']:.4f} | "
          f"NSE {NEARING['nse']:.4f} | alpha {NEARING['alpha']:.4f} | "
          f"r {NEARING['r']:.4f}")
    base = summarise(df, "pred", "ours, no correction")

    # --- ORACLE: per-basin k from the test period itself -------------------
    df["pred_oracle"] = df["pred"]
    for sid, g in df.groupby("station_id"):
        y, p = g["truth"].to_numpy(float), g["pred"].to_numpy(float)
        if np.std(p) < 1e-9:
            continue
        k = np.std(y) / np.std(p)
        mu = p.mean()
        df.loc[g.index, "pred_oracle"] = mu + k * (p - mu)
    summarise(df, "pred_oracle", "[ORACLE] per-basin k from test")

    # --- GLOBAL k, single scalar -------------------------------------------
    ratios = []
    for sid, g in df.groupby("station_id"):
        y, p = g["truth"].to_numpy(float), g["pred"].to_numpy(float)
        if np.std(p) > 1e-9 and len(y) >= 20:
            ratios.append(np.std(y) / np.std(p))
    kg = float(np.median(ratios))
    print(f"\n  median per-basin variance ratio (sigma_obs/sigma_sim) = {kg:.4f}")
    df["pred_gk"] = df.groupby("station_id")["pred"].transform(
        lambda s: s.mean() + kg * (s - s.mean()))
    summarise(df, "pred_gk", f"[oracle-lite] global k={kg:.3f} from test")

    print("\n  NOTE: both rows above use TEST observations to pick k -> oracle.")
    print("  A deployable version must fit k on the TRAIN window (1999-2008).")
    print("  That requires train-period with-q dumps, which do not exist yet;")
    print("  building them is the next step if this lever is worth pursuing.")


if __name__ == "__main__":
    main()
