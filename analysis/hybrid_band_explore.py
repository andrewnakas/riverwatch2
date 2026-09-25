#!/usr/bin/env python3
"""BEST-FORECAST — HYBRID: ensemble median + CMAL uncertainty band. EXPLORATORY.

⚠️⚠️ THIS RUN HAS NO BARS AND MAKES NO DEPLOY CLAIM.
Two consecutive gates in this ledger failed on bar SPECIFICATION (one borrowed
the wrong null, one wrote a two-sided test for a one-sided question). So this is
deliberately separated into "look at it" now and "gate it" later: if the numbers
warrant it, a deploy gate gets its OWN pre-registration with semantics written
in English first.

THE IDEA (measured motivation, not a hunch)
-------------------------------------------
[[distributional-heads-lose-deterministic-win-probabilistic]]:
  * the point ensemble owns the CENTRE   (NSE 0.934 / KGE 0.901 vs CMAL 0.876 / 0.743)
  * CMAL owns the SPREAD                 (pinball 0.0406 vs 0.0654, -38%)
Neither is better at both. So transplant: keep the ensemble's median, and take
CMAL's *RELATIVE* interval as the band.

    rel_lo = (ymed_cmal - ylo_cmal) / ymed_cmal
    rel_hi = (yhi_cmal - ymed_cmal) / ymed_cmal
    band   = [ens * (1 - c*rel_lo),  ens * (1 + c*rel_hi)]

c is a single global width multiplier, fit on the FIT period so that coverage
hits nominal, then applied unchanged -- the same one-constant discipline that
worked for variance inflation ([[global-variance-inflation-k104-DEPLOY]]).
CMAL is over-confident on its own (PICP 0.74 vs 0.80 nominal), so c > 1 is
expected.

⭐ NSE/KGE of the hybrid are the ENSEMBLE's BY CONSTRUCTION -- the central
estimate is untouched. The only open question is whether the transplanted band
is CALIBRATED. That is what this measures.

ARMS
  A  ensemble alone            (degenerate band: PICP ~ 0, pinball penalised)
  B  cmal alone
  C  hybrid, c = 1             (raw transplant)
  D  hybrid, c fit on FIT period to hit nominal coverage

Frame: train-side val slice; fit period for c is <= 1990-09-30. No test read.
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
QLO, QHI = 0.10, 0.90
NOMINAL = QHI - QLO
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")
CMAL = "camels531ls_cmal_nhlstm_TRAIN_s111.csv.gz"
C_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1.0) * d))


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
    wf = (d <= FIT_END).to_numpy()
    truth = tr.loc[wf, "truth"].to_numpy(float)
    mse = np.array([np.nanmean((tr.loc[wf, c].to_numpy(float) - truth) ** 2)
                    for c in cols])
    raw = mse ** (-THETA); raw = raw / raw.sum()
    w = LAM_W * (np.ones(len(cols)) / len(cols)) + (1 - LAM_W) * raw
    tr["ens"] = (tr[cols].to_numpy(float) * (w / w.sum())).sum(axis=1)
    ens = tr[["station_id", "truth", "ens"]].assign(date=d)

    cm = pd.read_csv(L / CMAL, usecols=["station_id", "t0", "h", "ylo", "ymed", "yhi"])
    cm = cm[cm["h"] == 1]
    cm["station_id"] = cm["station_id"].astype(str).str.zfill(8)
    cm["date"] = pd.to_datetime(cm["t0"])
    cm = cm[["station_id", "date", "ylo", "ymed", "yhi"]]

    m = ens.merge(cm, on=["station_id", "date"], how="inner")
    med = m["ymed"].to_numpy(float)
    ok = np.isfinite(med) & (np.abs(med) > 1e-9)
    m = m[ok].copy()
    m["rel_lo"] = ((m["ymed"] - m["ylo"]) / m["ymed"]).clip(lower=0)
    m["rel_hi"] = ((m["yhi"] - m["ymed"]) / m["ymed"]).clip(lower=0)
    print(f"joined ens x cmal: {len(m):,} rows, {m.station_id.nunique()} basins",
          flush=True)

    fit = m[m.date <= FIT_END]
    val = m[(m.date >= VAL_START) & (m.date <= VAL_END)]

    def score(df, lo, hi, centre, label):
        recs = {}
        for sid, g in df.groupby("station_id"):
            y = g["truth"].to_numpy(float)
            c = centre(g); l = lo(g); h = hi(g)
            k = np.isfinite(y) & np.isfinite(c) & np.isfinite(l) & np.isfinite(h)
            y, c, l, h = y[k], c[k], l[k], h[k]
            if len(y) < 20 or y.std() < 1e-9:
                continue
            recs[sid] = {
                "picp": float(np.mean((y >= l) & (y <= h))),
                "width": float(np.mean(h - l) / y.std()),
                "pinball": float(np.mean([pinball(y, l, QLO), pinball(y, c, 0.5),
                                          pinball(y, h, QHI)]) / y.std()),
            }
        t = pd.DataFrame(recs).T
        return {k: float(t[k].median()) for k in t.columns} | {"n": int(len(t))}

    E = lambda g: g["ens"].to_numpy(float)
    rows = {}
    rows["A ensemble alone"] = score(val, E, E, E, "A")
    rows["B cmal alone"] = score(
        val, lambda g: g["ylo"].to_numpy(float), lambda g: g["yhi"].to_numpy(float),
        lambda g: g["ymed"].to_numpy(float), "B")

    def hyb(c):
        return (lambda g: g["ens"].to_numpy(float) * (1 - c * g["rel_lo"].to_numpy(float)),
                lambda g: g["ens"].to_numpy(float) * (1 + c * g["rel_hi"].to_numpy(float)))

    lo1, hi1 = hyb(1.0)
    rows["C hybrid c=1.0"] = score(val, lo1, hi1, E, "C")

    # fit c on the FIT period: pick the c whose fit-period PICP is closest to nominal
    print("\nfitting c on the FIT period (target PICP = "
          f"{NOMINAL:.2f}):", flush=True)
    best_c, best_gap = None, 1e9
    for c in C_GRID:
        lo, hi = hyb(c)
        s = score(fit, lo, hi, E, f"fit c={c}")
        gap = abs(s["picp"] - NOMINAL)
        if gap < best_gap:
            best_gap, best_c = gap, c
        print(f"   c={c:.1f}  fit-period PICP {s['picp']:.3f}", flush=True)
    print(f"   SELECTED c = {best_c}", flush=True)
    loB, hiB = hyb(best_c)
    rows[f"D hybrid c={best_c} (fit)"] = score(val, loB, hiB, E, "D")

    print(f"\n{'arm':<26}{'PICP':>8}{'width':>8}{'pinball':>10}{'n':>6}", flush=True)
    print("-" * 58, flush=True)
    for k, v in rows.items():
        print(f"{k:<26}{v['picp']:>8.3f}{v['width']:>8.3f}"
              f"{v['pinball']:>10.4f}{v['n']:>6}", flush=True)
    print(f"\nnominal PICP = {NOMINAL:.2f}. NSE/KGE of arms A, C, D are IDENTICAL "
          "(same centre).", flush=True)
    print("⚠️ EXPLORATORY — no bars, no deploy claim.", flush=True)

    Path("benchmarks/ledger42_hybrid_band_explore.json").write_text(
        json.dumps({"nominal": NOMINAL, "selected_c": best_c, "arms": rows},
                   indent=2, default=float))
    print("wrote benchmarks/ledger42_hybrid_band_explore.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
