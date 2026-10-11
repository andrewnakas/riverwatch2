#!/usr/bin/env python3
"""GATE 3 candidate set: which basins outside the cohort could be served.

Produced BEFORE any out-of-cohort score exists, so the treatment set cannot be
chosen after seeing a result. Mirrors the structure of `data/m1_cohort.json`,
including recording every drop with its reason.

WHY THE COHORT RULE IS NOT THE RIGHT RULE HERE. `data/m1_cohort.json` selects
on >=90% discharge coverage over the 2018-2025 TEST window -- an evaluation
rule. Serving needs something else: enough recent observed discharge to fill a
365-day encoder window ending at the last observation. A basin dropped for a
gappy test window may be perfectly servable now, and one that passed may have
gone dead since. So this measures the serving requirement directly.

The ceiling is the 27 Addor statics: `data/camels_attrs.json` holds exactly 671
basins from a fixed Zenodo table, so 671 is a hard limit and non-CAMELS gauges
would need the attributes DERIVED, not looked up. (GAGES-II carries 18
different quantities, and CAMELS-27 plus 8 screened GAGES-II attributes already
failed 3 of 6 bars -- PREREG_v2.md, ledger 43.)
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MIN_WINDOW_FRACTION = 0.90   # of the 365-day encoder window
MAX_GAUGE_STALENESS = 14     # days since the last observation


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qstore", default="data/modern_corpus/q_store")
    ap.add_argument("--asof", default="", help="YYYY-MM-DD; default = q_store's own end")
    ap.add_argument("--out", default="data/m1_domain_candidates.json")
    a = ap.parse_args()

    ids = json.loads((ROOT / "data" / "camels_gauge_ids.json").read_text())
    attrs = set(json.loads((ROOT / "data" / "camels_attrs.json").read_text()))
    cohort = set(json.loads((ROOT / "data" / "m1_cohort.json").read_text())["cohort"])
    served = {str(s["id"]) for s in
              json.loads((ROOT / "data" / "stations_v15.json").read_text())["stations"]}

    all671 = set(ids["671"])
    qdir = Path(a.qstore)
    if not qdir.is_absolute():
        qdir = ROOT / qdir

    steps = {
        "camels_with_27_statics": len(all671 & attrs),
        "outside_the_490_cohort": len((all671 & attrs) - cohort),
        "also_in_the_served_list": len(((all671 & attrs) - cohort) & served),
    }
    cands = sorted(((all671 & attrs) - cohort) & served)

    asof = date.fromisoformat(a.asof) if a.asof else None
    servable, dropped = [], []
    for sid in cands:
        p = qdir / f"{sid}.csv.gz"
        if not p.exists():
            dropped.append({"id": sid, "reason": "no_q_store_file"})
            continue
        dates = []
        with gzip.open(p, "rt") as fh:
            fh.readline()
            for ln in fh:
                d, _, v = ln.partition(",")
                if v.strip():
                    dates.append(d)
        if not dates:
            dropped.append({"id": sid, "reason": "no_observations"})
            continue
        last = max(dates)
        ref = asof or date.fromisoformat(last)
        win_start = (ref - timedelta(days=364)).isoformat()
        n = sum(1 for d in dates if d >= win_start)
        frac = n / 365.0
        stale = (ref - date.fromisoformat(last)).days
        if stale > MAX_GAUGE_STALENESS:
            dropped.append({"id": sid, "reason": f"gauge_stale_{stale}d", "last": last})
        elif frac < MIN_WINDOW_FRACTION:
            dropped.append({"id": sid, "reason": f"window_coverage_{frac:.3f}", "last": last})
        else:
            servable.append({"id": sid, "window_coverage": round(frac, 4), "last": last})

    # The as-of date is whatever the store actually reaches, recorded so the
    # set can be reproduced rather than re-measured against a moving target.
    store_end = max((c["last"] for c in servable), default=None)
    out = {
        "_note": ("GATE 3 candidate set, fixed before any out-of-cohort score. "
                  "See PREREG_v2.md GATE 3."),
        "rule": {
            "source_list": "camels_gauge_ids.json[671] (the 27-statics ceiling)",
            "excludes": "the 490-basin training cohort",
            "requires": "presence in stations_v15.json",
            "min_window_fraction": MIN_WINDOW_FRACTION,
            "encoder_window_days": 365,
            "max_gauge_staleness_days": MAX_GAUGE_STALENESS,
            "asof": a.asof or store_end,
        },
        "funnel": steps,
        "n_servable": len(servable),
        "n_dropped": len(dropped),
        "servable": [c["id"] for c in servable],
        "servable_detail": servable,
        "dropped": dropped,
    }
    dst = Path(a.out)
    if not dst.is_absolute():
        dst = ROOT / dst
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dst.relative_to(ROOT)}")
    for k, v in steps.items():
        print(f"  {k:28s} {v}")
    print(f"  servable                     {len(servable)}")
    print(f"  dropped                      {len(dropped)}")
    from collections import Counter
    # Group by KIND, not by the exact string: "gauge_stale_55d" and
    # "gauge_stale_4963d" are one reason, not two.
    kinds = Counter("gauge_stale" if d["reason"].startswith("gauge_stale")
                    else "window_coverage" if d["reason"].startswith("window_coverage")
                    else d["reason"] for d in dropped)
    print("  drop reasons:", dict(kinds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
