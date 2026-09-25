#!/usr/bin/env python
"""LEDGER 40 VERDICT — arm C (Caravan-pretrained finetune) vs arm A (random init).

Reports exactly the bars fixed in the pre-registration, in the registered order,
and nothing else:

  BAR 1  competence : C median NSE >= A median NSE            (the headline)
  BAR 2  effect size: paired median delta (C-A), bootstrap 95% CI excluding 0
  BAR 3  breadth    : C better in >= 60% of basins

⚠️ PRE-COMMITTED INTERPRETATION RULE: if C beats A by MORE THAN +0.02, treat that
as a leakage signal FIRST, not a success. Caravan's hysets source names US
stations by USGS gauge id, and 407 of the 531 eval basins already reappeared
outside the `camels_` prefix once (memory: ledger40-caravan-leakage-audit-hysets-trap).

⚠️ Both arms are scored on the SAME window with the SAME 5 dynamic inputs and
9 static attributes. Do NOT compare either to the 27-static 0.7542 baseline —
the attribute set alone is worth -0.0317.
"""
from __future__ import annotations
import argparse, glob, sys
import numpy as np
import pandas as pd

A_DEFAULT = "runs/l40a_control_daymet_9stat_s222_1708_170326"
C_DEFAULT = "runs/l40c_finetune_from_caravan_s222_1808_072105"


def load(run, epoch=None):
    pat = f"{run}/test/model_epoch*/test_metrics.csv"
    fs = sorted(glob.glob(pat))
    if not fs:
        sys.exit(f"no test_metrics.csv under {run} — evaluate it first")
    f = fs[-1] if epoch is None else [x for x in fs if f"epoch{epoch:03d}" in x][-1]
    d = pd.read_csv(f)
    col = [c for c in d.columns if c.upper() == "NSE"][0]
    return f, dict(zip(d["basin"].astype(str), d[col].values))


def boot(x, n=10000, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(x), size=(n, len(x)))
    med = np.median(x[idx], axis=1)
    return np.percentile(med, 2.5), np.percentile(med, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=A_DEFAULT)
    ap.add_argument("--c", default=C_DEFAULT)
    a = ap.parse_args()

    fa, A = load(a.a)
    fc, C = load(a.c)
    common = sorted(set(A) & set(C))
    va = np.array([A[b] for b in common])
    vc = np.array([C[b] for b in common])
    m = np.isfinite(va) & np.isfinite(vc)
    va, vc = va[m], vc[m]
    d = vc - va

    print("=" * 68)
    print("  LEDGER 40 VERDICT — initialization axis")
    print("=" * 68)
    print("  arm A (random init)      : " + fa.split("/")[-4])
    print("  arm B/C/D (pretrained)   : " + fc.split("/")[-4])
    print(f"  paired basins            : {len(va)}")
    print()
    print(f"  BAR 1  competence")
    print(f"     A median NSE : {np.median(va):.4f}")
    print(f"     C median NSE : {np.median(vc):.4f}")
    ok1 = np.median(vc) >= np.median(va)
    print(f"     => {'PASS' if ok1 else 'FAIL'} (C >= A)")
    print()
    lo, hi = boot(d)
    print(f"  BAR 2  effect size")
    print(f"     paired median delta (C-A) : {np.median(d):+.5f}")
    print(f"     bootstrap 95% CI          : [{lo:+.5f}, {hi:+.5f}]")
    ok2 = (lo > 0) or (hi < 0)
    print(f"     => {'excludes 0' if ok2 else 'INCLUDES 0 (indistinguishable)'}")
    print()
    br = (vc > va).mean() * 100
    print(f"  BAR 3  breadth")
    print(f"     C better in : {(vc>va).sum()}/{len(va)} ({br:.1f}%)   [gate >= 60%]")
    print(f"     => {'PASS' if br >= 60 else 'FAIL'}")
    print()
    print("  distribution")
    for nm, v in (("A", va), ("C", vc)):
        print(f"     {nm}: mean {np.mean(v):.4f}  p25 {np.percentile(v,25):.4f}  "
              f"p75 {np.percentile(v,75):.4f}  >0.5 {(v>0.5).mean()*100:.1f}%")
    print()
    print("  --- VERDICT ---")
    delta = np.median(vc) - np.median(va)
    if delta > 0.02:
        print(f"  ⚠️⚠️ C beats A by {delta:+.4f} — ABOVE the +0.02 pre-committed threshold.")
        print("  SUSPECT RESIDUAL LEAKAGE FIRST. Re-audit the pretrain basin list against")
        print("  the 531 eval gauges (ids AND coordinates) before claiming any success.")
    elif ok1 and ok2 and br >= 60:
        print(f"  ✅ ALL THREE BARS PASS ({delta:+.4f}). Pretraining beats random init.")
        print("  ⚠️ Still a SECONDARY question whether it ships as an ensemble member —")
        print("     ensembles saturate at 4 and a 10th stream fights that headwind.")
    else:
        print(f"  ⛔ FAIL ({delta:+.4f}). Caravan pretraining does not beat random init")
        print("     under a matched protocol.")
        print("  ⚠️ Note the two registered confounds before generalising: the finetune")
        print("     inherits Caravan's NORMALISATION (nh-finetune-inherits-the-base-scaler),")
        print("     and the LR schedule was held identical to A, which can wash out the prior.")
        print("     Registered follow-ups: low-LR finetune; domain-aligned pretraining.")


if __name__ == "__main__":
    main()
