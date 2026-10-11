#!/usr/bin/env python3
"""Seed the published MODERN-1 forcing table from the NLDAS-2 areal corpus.

The daily append (`nldas_areal_append.py`) only maintains the recent tail. The
served member needs a 365-day encoder window, so the table has to be SEEDED
once from the corpus the member actually trained on -- after which the append
keeps it current and `build_m1_forcing.py` point-fills only the newest days
NLDAS-2 has not published yet.

Reads  data/modern_corpus/camels_corpus_nldas_modern_areal/{basin}.csv.gz
       (date, q_cfs, + the 6 forcing channels; 1980-01-01 onward)
Writes data/m1_forcing/{basin}.csv.gz
       (date, + the 6 channels, + src=nldas_areal)

⚠️ q_cfs is DROPPED on purpose. The app gets discharge live from USGS, and not
carrying it here keeps `assert_q_store_covers` and its hardcoded 671 out of the
serving path entirely.

Refuses a basin rather than publishing a hole: `norm_wx` turns a missing day
into 0 -- the TRAINING MEAN -- so a gap in this table is served silently as
average weather. Hole-freeness has to be a property of the artifact.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
HEADER = ["date"] + COLS + ["src"]
SRC = "nldas_areal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/modern_corpus/camels_corpus_nldas_modern_areal")
    ap.add_argument("--out", default="data/m1_forcing")
    ap.add_argument("--basins", default="data/m1_cohort.json",
                    help="JSON with 'cohort'/'servable', or a plain list, or 'ALL'")
    ap.add_argument("--extra-basins", default="",
                    help="second JSON whose ids are added (e.g. m1_domain_candidates.json)")
    ap.add_argument("--days", type=int, default=425,
                    help="trailing days to publish: 365 encoder + slack for a stale gauge")
    ap.add_argument("--min-coverage", type=float, default=1.0,
                    help="required fraction of the window present with all 6 channels")
    a = ap.parse_args()

    corpus = Path(a.corpus) if Path(a.corpus).is_absolute() else ROOT / a.corpus
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    if not corpus.exists():
        print(f"ERROR: no corpus at {corpus}", file=sys.stderr)
        return 1

    def ids_from(spec: str):
        p = Path(spec) if Path(spec).is_absolute() else ROOT / spec
        raw = json.loads(p.read_text())
        if isinstance(raw, dict):
            return [str(x) for x in (raw.get("cohort") or raw.get("servable") or [])]
        return [str(x) for x in raw]

    if a.basins == "ALL":
        ids = sorted(p.name.split(".")[0] for p in corpus.glob("*.csv.gz"))
    else:
        ids = ids_from(a.basins)
    if a.extra_basins:
        extra = ids_from(a.extra_basins)
        before = len(ids)
        ids = sorted(set(ids) | set(extra))
        print(f"basins: {before} + {len(extra)} extra -> {len(ids)} unique")
    print(f"seeding {len(ids)} basins, trailing {a.days} days")

    out.mkdir(parents=True, exist_ok=True)
    n_ok = n_short = n_missing = n_holey = 0
    spans = set()
    for sid in ids:
        src = corpus / f"{sid}.csv.gz"
        if not src.exists():
            n_missing += 1
            continue
        with gzip.open(src, "rt", newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or not set(COLS) <= set(rd.fieldnames):
                print(f"  {sid}: corpus lacks {sorted(set(COLS) - set(rd.fieldnames or []))}")
                n_missing += 1
                continue
            rows = [r for r in rd]
        if not rows:
            n_missing += 1
            continue

        last = max(r["date"] for r in rows)
        first_wanted = (date.fromisoformat(last) - timedelta(days=a.days - 1)).isoformat()
        win = [r for r in rows if r["date"] >= first_wanted]

        # Complete rows only -- an empty cell here becomes the training mean.
        good = [r for r in win if all(r.get(c) not in (None, "") for c in COLS)]
        if len(win) < a.days * a.min_coverage:
            n_short += 1
            continue
        if len(good) < len(win) * a.min_coverage:
            print(f"  {sid}: {len(win) - len(good)} incomplete rows in the window")
            n_holey += 1
            continue
        # And no gap in the daily calendar.
        ds = sorted(r["date"] for r in good)
        d0, d1 = date.fromisoformat(ds[0]), date.fromisoformat(ds[-1])
        if (d1 - d0).days + 1 != len(ds):
            print(f"  {sid}: calendar gap in {ds[0]}..{ds[-1]} ({len(ds)} rows)")
            n_holey += 1
            continue

        dst = out / f"{sid}.csv.gz"
        tmp = dst.with_suffix(".tmp.gz")
        with gzip.open(tmp, "wt", newline="") as gz:
            w = csv.writer(gz)
            w.writerow(HEADER)
            for r in sorted(good, key=lambda x: x["date"]):
                w.writerow([r["date"]] + [r[c] for c in COLS] + [SRC])
        tmp.replace(dst)
        n_ok += 1
        spans.add((ds[0], ds[-1]))

    print(f"seeded {n_ok} | short {n_short} | missing {n_missing} | holey {n_holey}")
    if spans:
        print(f"distinct spans: {len(spans)}; e.g. {sorted(spans)[0]}")
        if len(spans) > 1:
            print("  ⚠️ basins do not share one span -- the append will level them up")
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
