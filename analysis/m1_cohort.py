#!/usr/bin/env python
"""MODERN-1: freeze the modern evaluation cohort.

The 531 Kratzert basins are CAMELS reference basins certified for WY1990-2009.
Nothing guarantees their gauges still report in the 2018-2025 test window, and
measurement says many do not: over the 671, 28 basins have ZERO daily discharge
observations in 2019-2025 and 38 are under 50%.

So the modern cohort is computed from data, once, by the rule fixed in
PREREG_v2.md section MODERN-1 BEFORE any number was seen:

    a basin enters the cohort iff
      (1) it is in data/camels_gauge_ids.json["531"], AND
      (2) it has >= MIN_FRAC daily discharge observations over the P-MODERN
          test window, AND
      (3) it has >= MIN_ROWS scoreable rows with non-degenerate flow variance.

N is whatever it is. It is reported, asserted by every launcher, and never
adjusted after a score is seen.
"""
from __future__ import annotations
import argparse, gzip, json, os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# --- the rule, fixed in PREREG_v2.md MODERN-1 before any number was seen -----
TEST_START = "2018-10-01"
TEST_END   = "2025-09-30"
MIN_FRAC   = 0.90
MIN_ROWS   = 20
# ----------------------------------------------------------------------------


def q_coverage(path: Path, lo: str, hi: str) -> tuple[int, int, float]:
    """(n_obs, n_days, variance) for q_cfs on [lo, hi]. Sentinels already masked
    at build time (q < 0 -> empty), but we re-mask here: a guard you only apply
    upstream is a guard you cannot audit."""
    n_days = n_obs = 0
    vals: list[float] = []
    with gzip.open(path, "rt") as fh:
        head = fh.readline().rstrip("\n").split(",")
        try:
            iq = head.index("q_cfs")
        except ValueError:
            return 0, 0, 0.0
        for line in fh:
            f = line.rstrip("\n").split(",")
            d = f[0]
            if d < lo:
                continue
            if d > hi:
                break
            n_days += 1
            s = f[iq] if iq < len(f) else ""
            if s in ("", "nan", "NaN"):
                continue
            try:
                v = float(s)
            except ValueError:
                continue
            if v < 0 or not np.isfinite(v):
                continue
            n_obs += 1
            vals.append(v)
    var = float(np.var(vals)) if len(vals) > 1 else 0.0
    return n_obs, n_days, var


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/modern_corpus/camels_corpus_daymet_modern_areal",
                    help="any corpus carrying the modern q_cfs series")
    ap.add_argument("--ids", default="data/camels_gauge_ids.json")
    ap.add_argument("--test-start", default=TEST_START)
    ap.add_argument("--test-end", default=TEST_END)
    ap.add_argument("--also-require", default="",
                    help="comma-separated extra corpus dirs the basin must also exist in")
    ap.add_argument("--out", default="data/m1_cohort.json")
    a = ap.parse_args()

    os.chdir(ROOT)
    ids = json.loads(Path(a.ids).read_text())["531"]
    ids = [str(i).zfill(8) for i in ids]
    corpus = Path(a.corpus)

    keep, dropped = [], {}
    for gid in ids:
        p = corpus / f"{gid}.csv.gz"
        if not p.exists():
            dropped[gid] = "absent_from_corpus"
            continue
        n_obs, n_days, var = q_coverage(p, a.test_start, a.test_end)
        frac = n_obs / n_days if n_days else 0.0
        if n_days == 0:
            dropped[gid] = "window_absent"
        elif frac < MIN_FRAC:
            dropped[gid] = f"coverage_{frac:.4f}"
        elif n_obs < MIN_ROWS:
            dropped[gid] = f"rows_{n_obs}"
        elif var <= 1e-9:
            dropped[gid] = "zero_variance"
        else:
            keep.append(gid)

    extra = [d for d in a.also_require.split(",") if d]
    intersect = {}
    for d in extra:
        present = {p.name.split(".")[0] for p in Path(d).glob("*.csv.gz")}
        intersect[d] = sorted(set(keep) & present)

    out = {
        "_note": ("MODERN-1 evaluation cohort. Rule fixed in PREREG_v2.md MODERN-1 "
                  "BEFORE any number was seen; never adjusted after a score."),
        "rule": {"source_list": "camels_gauge_ids.json[531]",
                 "test_window": [a.test_start, a.test_end],
                 "min_obs_fraction": MIN_FRAC, "min_rows": MIN_ROWS,
                 "min_variance": 1e-9},
        "corpus_used": str(corpus),
        "n_candidates": len(ids),
        "n_cohort": len(keep),
        "cohort": keep,
        "dropped": dropped,
        "intersections": {k: {"n": len(v)} for k, v in intersect.items()},
    }
    Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"candidates      {len(ids)}")
    print(f"COHORT          {len(keep)}")
    print(f"dropped         {len(dropped)}")
    reasons: dict[str, int] = {}
    for r in dropped.values():
        k = r.split("_")[0] if not r.startswith("coverage") else "coverage<0.90"
        reasons[k] = reasons.get(k, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v}")
    for k, v in intersect.items():
        print(f"intersect {k}: {len(v)}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
