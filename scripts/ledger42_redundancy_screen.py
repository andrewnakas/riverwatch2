#!/usr/bin/env python3
"""LEDGER 42 A1-3 — THE REDUNDANCY SCREEN. The one A1 bar that survived.

A1-1/A1-2 died in calibration: the magnitude statistic separates a catastrophic
product hugely but cannot order near-equals, which is exactly what screening a
candidate against daymet needs (AMENDMENT 2). A1-3 survives because it needs
only a THRESHOLD, never discriminative power.

THE BAR (registered, unchanged): **event-day R² < 0.9 against every existing
precip input** (daymet, nldas, maurer). The station corpus shares GHCN
provenance with daymet, so a construction that merely re-derives its own base
is a null dressed as a candidate — this is the guard that catches that.

⚠️ It is TWO-SIDED in spirit even though only one side is a registered bar:
  R² ~ 1.00  => the candidate IS daymet; nothing to test, close it.
  R² very low => the candidate has drifted off the gauge-anchored base that is
                 the ONLY property ever found to predict member skill. Not a
                 registered fail, but it is REPORTED and must be looked at.

Events = the same definition as the magnitude referee: top-1% observed-flow
days that are the local maximum within +/-3 days. Train window only.

Usage:
  python scripts/ledger42_redundancy_screen.py --candidate camels_corpus_eventqm_v1
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(os.path.expanduser("~/riverwatch2"))

CORP = Path("gpu1080/corpora671")
SUBSET = Path("data/camels_gauge_ids.json")
TRAIN_START, TRAIN_END = "1981-01-01", "1995-09-30"
EXISTING = ["camels_corpus_daymet_v2", "camels_corpus_nldas_v2",
            "camels_corpus_maurer_v2"]
R2_BAR = 0.9
MIN_BASINS = 400   # this screen is a threshold, not a ranking; see note below


def event_mask(q):
    """top-1% flow days that are the local max within +/-3 days."""
    pos = q[np.isfinite(q) & (q > 0)]
    if pos.size < 200:
        return None
    thr = np.percentile(pos, 99)
    m = np.zeros(len(q), bool)
    for t in range(3, len(q) - 3):
        if np.isfinite(q[t]) and q[t] >= thr:
            w = q[t - 3:t + 4]
            if np.isfinite(w).all() and q[t] >= np.nanmax(w):
                m[t - 3:t + 1] = True      # the depth window of that event
    return m


ALT_ROOTS = [Path("gpu1080/corpora671"), Path("data/local_corpora")]


def series(corpus, b, cols=("date", "precipitation_sum")):
    p = next((r / corpus / f"{b}.csv.gz" for r in ALT_ROOTS
              if (r / corpus / f"{b}.csv.gz").exists()), None)
    if p is None:
        return None
    d = pd.read_csv(p, usecols=list(cols))
    d["date"] = pd.to_datetime(d["date"])
    return d[(d.date >= TRAIN_START) & (d.date <= TRAIN_END)].sort_values("date")


def align(ref_dates, df, col="precipitation_sum"):
    """Reindex a series onto the reference date axis.

    ⚠️ The first version compared arrays by POSITION and skipped any corpus
    whose row count differed (the station corpus has 11,140 rows vs daymet's
    12,785). That silently reports NO DATA for exactly the comparison that
    matters most. Align on dates, never on length."""
    if df is None:
        return None
    s = df.set_index("date")[col]
    return s.reindex(ref_dates).to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/ledger42_a13_redundancy.json")
    a = ap.parse_args()

    raw = json.loads(SUBSET.read_text())
    basins = [str(b).zfill(8) for b in (raw["531"] if isinstance(raw, dict) else raw)]
    if a.limit:
        basins = basins[:a.limit]

    per = {e: [] for e in EXISTING}
    n_ok = 0
    for b in basins:
        qd = series("camels_corpus_daymet_v2", b, cols=("date", "q_cfs"))
        if qd is None:
            continue
        ref_dates = pd.DatetimeIndex(qd["date"])          # the common axis
        m = event_mask(qd["q_cfs"].to_numpy(float))
        if m is None or m.sum() < 20:
            continue
        cv = align(ref_dates, series(a.candidate, b))
        if cv is None:
            continue
        n_ok += 1
        for e in EXISTING:
            ev = align(ref_dates, series(e, b))
            if ev is None:
                continue
            k = m & np.isfinite(cv) & np.isfinite(ev)
            if k.sum() < 20 or np.var(ev[k]) < 1e-12 or np.var(cv[k]) < 1e-12:
                continue
            r = np.corrcoef(cv[k], ev[k])[0, 1]
            per[e].append(float(r * r))

    print(f"candidate: {a.candidate}   basins with events: {n_ok}")
    out = {"candidate": a.candidate, "basins": n_ok, "R2_bar": R2_BAR,
           "min_basins": MIN_BASINS, "vs": {}}
    verdicts = []
    for e in EXISTING:
        v = np.array(per[e], float)
        if v.size == 0:
            print(f"  vs {e:<30} NO DATA")
            out["vs"][e] = {"n": 0}
            verdicts.append(None)
            continue
        med = float(np.median(v))
        frac_hi = float((v >= R2_BAR).mean())
        ok = med < R2_BAR
        verdicts.append(ok)
        print(f"  vs {e:<30} n={v.size:3d}  median event-day R²={med:.4f}  "
              f"basins≥{R2_BAR}: {frac_hi:.1%}   {'PASS' if ok else 'FAIL'}")
        out["vs"][e] = {"n": int(v.size), "median_event_R2": med,
                        "frac_basins_at_or_above_bar": frac_hi, "pass": ok}

    if n_ok < MIN_BASINS and not a.limit:
        verdict = "INCONCLUSIVE (too few basins)"
    elif a.limit:
        verdict = "REFUSED (--limit smoke run; no verdict permitted)"
    elif any(v is None for v in verdicts):
        verdict = "INCONCLUSIVE (a comparison had no data)"
    elif all(verdicts):
        verdict = "PASS A1-3"
    else:
        verdict = "FAIL A1-3 — candidate is redundant with an existing input"
    print(f"\nVERDICT: {verdict}")
    out["verdict"] = verdict
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
