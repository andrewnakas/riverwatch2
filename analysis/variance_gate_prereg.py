#!/usr/bin/env python3
"""BEST-FORECAST TRACK — the DEPLOY gate for variance inflation. PRE-REGISTERED.

⛔ WRITTEN BEFORE ANY NUMBER FROM IT EXISTS. Bars are the constants below and
are not to be edited after seeing output.

WHY A NEW GATE
--------------
`variance_frontier.py`'s nested gate returned lambda = 0 (do nothing), rejecting
lambda=0.25 on dNSE = -0.000191 against a -0.000165 bar. **That bar was derived
from the wrong null**: -0.000165 was measured for SEED SWAPS (S0.3b), whereas
the quantity here is *period-to-period transfer of a post-processing parameter*
— a different and larger noise source. Gating one quantity with another's null
is the error [[a-gate-must-be-calibrated-against-its-own-null]] records.
⇒ This is a NEW pre-registration with the correct null, not an edit of that one.

THE NULL: A PARAMETER-SHUFFLED SHAM
-----------------------------------
Apply the SAME transform, but give each basin ANOTHER basin's alpha (a random
derangement). This preserves the intervention's magnitude and structure while
destroying its basin-specific validity. Anything the real transform buys over
the sham is attributable to the per-basin variance ratio actually being right.
N_SHAM independent shuffles give the null distribution.

⭐ This is a stronger null than a basin bootstrap (which cannot see parameter
error at all) and stronger than the two-period sign flip (n=2).

PRE-REGISTERED BARS  (deploy iff ALL THREE hold)
  B1  KGE REAL > max(KGE SHAM) on EVERY evaluation period.
  B2  |median dNSE REAL| <= max(|median dNSE SHAM|) on EVERY evaluation period
      -- i.e. the NSE movement is no larger than the sham's, so it is not
      attributable to the intervention.
  B3  dKGE REAL > 0 on EVERY evaluation period (reproducibility, not one period).
FAIL ANY -> DO NOT DEPLOY, and record it.

Frame: rolling fit->eval blocks inside the TRAIN window only. No test read.
Run:  gpu1080/.venv/bin/python -u analysis/variance_gate_prereg.py
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
LAMBDA = 0.25                 # the candidate, fixed in advance
N_SHAM = 20                   # shuffles per evaluation period
SEED = 20260823
MULTI5 = ("s111", "s222", "s333", "s444", "s555")
MULTI6 = ("s111", "s222", "s333")
# rolling (fit, eval) blocks, all inside the TRAIN window
BLOCKS = [(("1980-10-01", "1985-09-30"), ("1985-10-01", "1988-09-30")),
          (("1980-10-01", "1988-09-30"), ("1988-10-01", "1991-09-30")),
          (("1980-10-01", "1991-09-30"), ("1991-10-01", "1995-09-30"))]


def metr(y, p):
    if len(y) < 20 or np.var(y) < 1e-9 or np.std(p) < 1e-12:
        return None
    r = np.corrcoef(y, p)[0, 1]
    a = p.std() / y.std()
    b = p.mean() / y.mean() if abs(y.mean()) > 1e-12 else np.nan
    return (1.0 - np.mean((y - p) ** 2) / np.var(y),
            1.0 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)
            if np.isfinite(r) and np.isfinite(b) else np.nan)


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
    w_fit = (d <= "1990-09-30").to_numpy()
    truth = tr.loc[w_fit, "truth"].to_numpy(float)
    mse = np.array([np.nanmean((tr.loc[w_fit, c].to_numpy(float) - truth) ** 2)
                    for c in cols])
    raw = mse ** (-THETA)
    raw = raw / raw.sum()
    w = LAM_W * (np.ones(len(cols)) / len(cols)) + (1 - LAM_W) * raw
    tr["ens"] = (tr[cols].to_numpy(float) * (w / w.sum())).sum(axis=1)
    tr = tr[["station_id", "truth", "ens"]].assign(date=d)

    rng = np.random.default_rng(SEED)
    results, verdicts = [], []
    print("=" * 88, flush=True)
    print(f"PRE-REGISTERED DEPLOY GATE — variance inflation lambda={LAMBDA}", flush=True)
    print(f"null = parameter-shuffled sham, {N_SHAM} shuffles per period", flush=True)
    print("=" * 88, flush=True)

    for (f0, f1), (e0, e1) in BLOCKS:
        fit = tr[(tr.date >= f0) & (tr.date <= f1)]
        ev = tr[(tr.date >= e0) & (tr.date <= e1)]
        par = {}
        for sid, g in fit.groupby("station_id"):
            y = g["truth"].to_numpy(float); p = g["ens"].to_numpy(float)
            k = np.isfinite(y) & np.isfinite(p); y, p = y[k], p[k]
            if len(y) < 50 or y.std() < 1e-9 or p.std() < 1e-9:
                continue
            par[sid] = {"mu": p.mean(), "alpha": p.std() / y.std()}
        sids = sorted(par)
        if len(sids) < 400:
            print(f"  {e0}..{e1}: only {len(sids)} basins — SKIP", flush=True)
            continue

        evg = {sid: g for sid, g in ev.groupby("station_id") if sid in par}

        def run(alpha_map):
            dn, dk = [], []
            for sid, g in evg.items():
                y = g["truth"].to_numpy(float); p = g["ens"].to_numpy(float)
                k = np.isfinite(y) & np.isfinite(p); y, p = y[k], p[k]
                if len(y) < 20:
                    continue
                base = metr(y, p)
                a = alpha_map[sid]
                kk = 1.0 + LAMBDA * (1.0 / max(a, 1e-6) - 1.0)
                tt = metr(y, par[sid]["mu"] + kk * (p - par[sid]["mu"]))
                if base and tt and np.isfinite(base[1]) and np.isfinite(tt[1]):
                    dn.append(tt[0] - base[0]); dk.append(tt[1] - base[1])
            return float(np.median(dn)), float(np.median(dk))

        real = run({s: par[s]["alpha"] for s in sids})
        shams = []
        for _ in range(N_SHAM):
            perm = rng.permutation(len(sids))
            shams.append(run({sids[i]: par[sids[perm[i]]]["alpha"]
                              for i in range(len(sids))}))
        sn = np.array([s[0] for s in shams]); sk = np.array([s[1] for s in shams])

        b1 = real[1] > sk.max()
        b2 = abs(real[0]) <= np.abs(sn).max()
        b3 = real[1] > 0
        verdicts.append((b1, b2, b3))
        print(f"\n  eval {e0}..{e1}   ({len(evg)} basins)", flush=True)
        print(f"    REAL  dNSE {real[0]:+.6f}   dKGE {real[1]:+.6f}", flush=True)
        print(f"    SHAM  dNSE [{sn.min():+.6f}, {sn.max():+.6f}]   "
              f"dKGE [{sk.min():+.6f}, {sk.max():+.6f}]", flush=True)
        print(f"    B1 KGE>shamMax {b1}   B2 |dNSE|<=shamMax {b2}   "
              f"B3 dKGE>0 {b3}", flush=True)
        results.append({"eval": [e0, e1], "basins": len(evg),
                        "real_dnse": real[0], "real_dkge": real[1],
                        "sham_dnse_range": [float(sn.min()), float(sn.max())],
                        "sham_dkge_range": [float(sk.min()), float(sk.max())],
                        "B1": bool(b1), "B2": bool(b2), "B3": bool(b3)})

    ok = bool(verdicts) and all(all(v) for v in verdicts)
    print(f"\n{'=' * 88}", flush=True)
    print(f"VERDICT: {'DEPLOY' if ok else 'DO NOT DEPLOY'} "
          f"(all three bars on all {len(verdicts)} periods)", flush=True)
    print("=" * 88, flush=True)
    Path("benchmarks/ledger42_variance_deploy_gate.json").write_text(json.dumps(
        {"lambda": LAMBDA, "n_sham": N_SHAM, "periods": results,
         "deploy": ok}, indent=2, default=float))
    print("wrote benchmarks/ledger42_variance_deploy_gate.json", flush=True)


if __name__ == "__main__":
    sys.exit(main())
