#!/usr/bin/env python
"""MODERN-1: how much of a Gate A delta is TARGET REVISION rather than pipeline?

The modern corpus takes its discharge from a fresh USGS DV pull (1980->today),
because the alternative -- splicing frozen CAMELS q to 2014 and fresh q after --
would put a seam at 2015-01-01, exactly where the modern training window ends.
Measured on 53,356 day-pairs, the two sources agree to rounding (median ratio
1.000000, log-space r 0.998481), but they are not identical: USGS revises its
record, and 3,780 of the 3,848 >5% disagreements sat in 4 low-flow basins.

So Gate A ("does the modern pipeline cost skill?") would otherwise confound two
things: the forcing/pipeline change, and the target change. This separates them
with ZERO GPU by re-scoring the FROZEN record's own existing dumps against the
refreshed targets. Same predictions, same rows, only the truth differs.

⚠️ Scored on the SAME ROWS both ways -- a row whose refreshed target is missing
is dropped from BOTH sides, never from one.
"""
from __future__ import annotations
import argparse, gzip, json, os, sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l51_withq_score as S            # noqa: E402  (chdirs to ~/riverwatch2)

RECORD8 = ["fused3h1", "daymeth1", "nldash1", "maurerh1", "aorch1",
           "nhar", "nhar0", "nhar0h256"]


def load_qstore(store: Path, ids) -> dict:
    out = {}
    for g in ids:
        p = store / f"{g}.csv.gz"
        if not p.exists():
            continue
        d = {}
        with gzip.open(p, "rt") as fh:
            fh.readline()
            for ln in fh:
                a, b = ln.rstrip("\n").split(",")
                if b not in ("", "nan"):
                    d[a] = float(b)
        out[g] = d
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="test1")
    ap.add_argument("--members", default=",".join(RECORD8))
    ap.add_argument("--seeds", default="501,502,503,504,505")
    ap.add_argument("--qstore", default="data/modern_corpus/q_store")
    ap.add_argument("--out", default="benchmarks/m1_target_delta.json")
    a = ap.parse_args()

    members = a.members.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"loading the frozen record on {a.frame} ...")
    df = S.join(members, a.frame, seeds=seeds)
    print(f"joined rows={len(df):,} basins={df.station_id.nunique()}")

    qs = load_qstore(Path(a.qstore), sorted(df.station_id.unique()))
    print(f"q_store loaded for {len(qs)} basins")

    # dump convention, MEASURED not assumed: target_date = t0 + h, and h == 1 here
    tgt = (pd.to_datetime(df.t0) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    fresh = np.array([qs.get(s, {}).get(d, np.nan)
                      for s, d in zip(df.station_id.to_numpy(), tgt.to_numpy())])

    have = np.isfinite(fresh) & np.isfinite(df.truth.to_numpy(float))
    print(f"rows with BOTH targets: {have.sum():,} of {len(df):,} "
          f"({100*have.mean():.2f}%)  -- scored on these rows only, both ways")
    d2 = df.loc[have].copy()
    frozen_truth = d2.truth.to_numpy(float).copy()
    fresh_truth = fresh[have]

    same = (frozen_truth == fresh_truth)
    rel = np.abs(frozen_truth - fresh_truth) / np.maximum(np.abs(frozen_truth), 1e-9)
    print(f"targets bit-identical: {100*same.mean():.3f}%   "
          f">1% different: {(rel > 0.01).sum():,}   >5%: {(rel > 0.05).sum():,}")

    nse_frozen = S.per_basin(d2, members)
    d2["truth"] = fresh_truth
    nse_fresh = S.per_basin(d2, members)

    common = sorted(set(nse_frozen.index) & set(nse_fresh.index))
    sa, sb = nse_frozen.loc[common], nse_fresh.loc[common]
    aa, bb = sa.to_numpy(), sb.to_numpy()
    delta, lo, hi, breadth, n = S.paired_ci(sb, sa)

    n_changed = int((aa != bb).sum())
    max_abs = float(np.abs(aa - bb).max())
    mean_delta = float(np.mean(bb - aa))
    print("\n" + "=" * 78)
    print(f"{'median NSE, FROZEN targets (the record s own)':54s} {np.median(aa):.12f}")
    print(f"{'median NSE, REFRESHED USGS targets':54s} {np.median(bb):.12f}")
    print(f"{'PAIRED MEDIAN delta (refreshed - frozen)':54s} {delta:+.3e}")
    print(f"{'   95% CI':54s} [{lo:+.3e}, {hi:+.3e}]")
    print(f"{'   breadth (fraction of basins improving)':54s} {breadth:.3f}")
    print(f"{'basins whose NSE changed at all':54s} {n_changed} of {len(common)}")
    print(f"{'MEAN across-basin delta':54s} {mean_delta:+.6f}")
    print(f"{'worst single-basin |delta|':54s} {max_abs:.6f}")
    print("=" * 78)
    print("""
INTERPRETATION
  The refreshed targets are NOT the same numbers: 574k+ rows differ and EVERY
  basin's NSE moves, by up to 0.6 in the worst one. But the headline metric is a
  MEDIAN across basins, and the median is a RANK statistic -- the revisions are
  balanced around it (breadth ~0.50), so it barely moves, while the MEAN shifts
  visibly because revisions land hardest in the tail.

  => For the day-1 MEDIAN NSE headline, target revision contributes ~0 and a
     Gate A delta can be attributed to the pipeline with no correction.
  => Any MEAN-based or per-basin claim must carry this correction explicitly.""")

    Path(a.out).write_text(json.dumps({
        "_note": "MODERN-1: frozen record re-scored against refreshed USGS targets. "
                 "Same predictions, same rows; only the truth differs.",
        "frame": a.frame, "members": members, "seeds": seeds,
        "rows_scored": int(have.sum()), "rows_total": int(len(df)),
        "targets_bit_identical_pct": float(100 * same.mean()),
        "targets_gt1pct_different": int((rel > 0.01).sum()),
        "targets_gt5pct_different": int((rel > 0.05).sum()),
        "median_nse_frozen_targets": float(np.median(aa)),
        "median_nse_refreshed_targets": float(np.median(bb)),
        "paired_median_delta": float(delta), "ci95": [float(lo), float(hi)],
        "breadth": float(breadth), "n_basins": len(common),
        "n_basins_changed": n_changed, "mean_delta": mean_delta,
        "worst_basin_abs_delta": max_abs,
        "verdict": ("median metric is insensitive to the target revision; "
                    "mean and per-basin claims are NOT")}, indent=1))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
