#!/usr/bin/env python3
"""BEST-FORECAST TRACK — score the DISTRIBUTIONAL heads on the metrics they
were never judged on. Zero GPU, train-side only.

WHY
---
CMAL and GMM heads were built, trained, and KILLED on **median NSE**
([[distributional-arm-closed-two-heads]]). But median NSE is precisely the
metric that **REWARDS under-dispersion**: dNSE/dalpha = 2r - 2*alpha, so its
optimum sits at alpha = r < 1 ([[on-time-and-too-small-is-the-LOSS-not-the-data]]).
A head whose whole purpose is to represent spread was therefore judged by a
criterion that penalises spread. **They were never scored on KGE, on coverage,
or on any probabilistic loss** — and the dumps have carried
`ylo/ymed/yhi/ymean` the entire time.

`ylo`/`yhi` are the **10th/90th percentiles** (`nh_to_dump.py`), so the nominal
coverage of [ylo, yhi] is **80%**. Point models collapse ylo==ymed==yhi.

WHAT IS SCORED (per basin, val slice, then aggregated)
  ymed  : NSE, KGE, r, alpha, beta      <- the production readout
  ymean : NSE, KGE, alpha               <- the other readout
  PICP  : P(ylo <= obs <= yhi)          <- nominal 0.80; < 0.80 = over-confident
  width : mean(yhi - ylo) / sd(obs)     <- sharpness (smaller = sharper)
  pinball: mean quantile loss at 0.1/0.5/0.9  <- a 3-point CRPS proxy, lower better

⛔ Fair A/B: CMAL/GMM were trained as head+loss swaps on the SAME corpus/seed as
`lstm_multi`, so `lstm_multi` is the control. Train-side val slice only; no test
read; nothing here changes the record.

Run:  gpu1080/.venv/bin/python -u analysis/score_distributional.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))
sys.path.insert(0, os.getcwd())

D = Path("gpu1080/dumps")
VAL_START, VAL_END = "1990-10-01", "1995-09-30"
QLO, QHI = 0.10, 0.90
NOMINAL = QHI - QLO

ARMS = {
    "lstm_multi (point, CONTROL)": ["camels531ls_multi_nhlstm_TRAIN_s111.csv.gz"],
    "cmal  s111": ["camels531ls_cmal_nhlstm_TRAIN_s111.csv.gz"],
    "gmm   s111": ["camels531ls_gmm_nhlstm_TRAIN_s111.csv.gz"],
    "gmm   s222": ["camels531ls_gmm_nhlstm_TRAIN_s222.csv.gz"],
}


def load(name):
    df = pd.read_csv(D / name,
                     usecols=["station_id", "t0", "h", "truth",
                              "ylo", "ymed", "yhi", "ymean"])
    df = df[df["h"] == 1]
    df["station_id"] = df["station_id"].astype(str).str.zfill(8)
    df["date"] = pd.to_datetime(df["t0"])
    return df[(df.date >= VAL_START) & (df.date <= VAL_END)]


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1.0) * d))


def fit_stats(y, p):
    if len(y) < 20 or np.var(y) < 1e-9 or np.std(p) < 1e-12:
        return None
    r = np.corrcoef(y, p)[0, 1]
    a = p.std() / y.std()
    b = p.mean() / y.mean() if abs(y.mean()) > 1e-12 else np.nan
    return {
        "nse": 1.0 - np.mean((y - p) ** 2) / np.var(y),
        "kge": (1.0 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
                if np.isfinite(r) and np.isfinite(b) else np.nan),
        "alpha": a,
    }


def main():
    out = {}
    print("=" * 96, flush=True)
    print("DISTRIBUTIONAL HEADS ON THE METRICS THEY WERE NEVER JUDGED ON "
          "(val slice, h==1)", flush=True)
    print("=" * 96, flush=True)
    hdr = (f"{'arm':<28}{'NSE(med)':>10}{'KGE(med)':>10}{'a(med)':>8}"
           f"{'NSE(mean)':>11}{'KGE(mean)':>10}{'a(mean)':>9}"
           f"{'PICP':>7}{'width':>7}{'pinball':>9}")
    print(hdr, flush=True)
    print("-" * 96, flush=True)

    for label, files in ARMS.items():
        f = files[0]
        if not (D / f).exists():
            print(f"{label:<28} MISSING {f}", flush=True)
            continue
        df = load(f)
        recs = {}
        for sid, g in df.groupby("station_id"):
            y = g["truth"].to_numpy(float)
            med = g["ymed"].to_numpy(float)
            mean = g["ymean"].to_numpy(float)
            lo = g["ylo"].to_numpy(float)
            hi = g["yhi"].to_numpy(float)
            k = np.isfinite(y) & np.isfinite(med) & np.isfinite(lo) & np.isfinite(hi)
            y, med, mean, lo, hi = y[k], med[k], mean[k], lo[k], hi[k]
            sm = fit_stats(y, med)
            if sm is None:
                continue
            sa = fit_stats(y, mean) or {"nse": np.nan, "kge": np.nan, "alpha": np.nan}
            recs[sid] = {
                "nse_med": sm["nse"], "kge_med": sm["kge"], "a_med": sm["alpha"],
                "nse_mean": sa["nse"], "kge_mean": sa["kge"], "a_mean": sa["alpha"],
                "picp": float(np.mean((y >= lo) & (y <= hi))),
                "width": float(np.mean(hi - lo) / max(y.std(), 1e-9)),
                "pinball": float(np.mean([pinball(y, lo, QLO),
                                          pinball(y, med, 0.5),
                                          pinball(y, hi, QHI)]) / max(y.std(), 1e-9)),
            }
        t = pd.DataFrame(recs).T
        rec = {k: float(t[k].median()) for k in t.columns}
        rec["basins"] = int(len(t))
        out[label] = rec
        print(f"{label:<28}{rec['nse_med']:>10.4f}{rec['kge_med']:>10.4f}"
              f"{rec['a_med']:>8.4f}{rec['nse_mean']:>11.4f}"
              f"{rec['kge_mean']:>10.4f}{rec['a_mean']:>9.4f}"
              f"{rec['picp']:>7.3f}{rec['width']:>7.3f}{rec['pinball']:>9.4f}",
              flush=True)

    print("-" * 96, flush=True)
    print(f"PICP nominal = {NOMINAL:.2f}. PICP < nominal => OVER-CONFIDENT "
          "(intervals too narrow).", flush=True)
    print("width and pinball are normalised by the basin's observed sd "
          "(pinball: lower is better).", flush=True)
    print("⚠️ A point model has ylo==ymed==yhi ⇒ PICP ~ 0 and width 0 BY "
          "CONSTRUCTION — that is the", flush=True)
    print("   whole point: it offers no uncertainty at all, which median NSE "
          "cannot see.", flush=True)

    Path("benchmarks/ledger42_distributional_rescore.json").write_text(
        json.dumps({"nominal_picp": NOMINAL, "arms": out}, indent=2, default=float))
    print("\nwrote benchmarks/ledger42_distributional_rescore.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
