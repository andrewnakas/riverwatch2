#!/usr/bin/env python
"""MODERN-1: build a SYNTHETIC proxy corpus to smoke-test the Phase 3 plumbing.

⛔⛔ THIS IS NOT SCIENTIFIC DATA AND MUST NEVER ENTER A RESULT. ⛔⛔
It exists so the modern launcher, var-sets, guards, dump path and sidecars can be
exercised end-to-end BEFORE the real corpus finishes -- so that when the real
corpus lands, training starts instead of failing on a typo.

It takes the frozen 24-var fused4 corpus and does two purely cosmetic things:
  1. renames the product suffixes  maurer -> gridmet, aorc -> nclimgrid,
     so the columns match MODERN_PRODUCTS and `--enc-vars camelsm4v2*` resolves;
  2. shifts every date forward by 17 years, so the 1980-2008 span lands on
     1997-2025 and therefore overlaps P-MODERN's train/val/test windows.

The numbers are real CAMELS numbers under fake labels on fake dates. Any score
produced from it is meaningless; the point is exclusively that the machinery runs.
Output goes to a directory with SMOKE in its name so it cannot be mistaken for
`data/modern_corpus/v3/...`.
"""
from __future__ import annotations
import argparse, gzip, json, os
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RENAME = {"maurer": "gridmet", "aorc": "nclimgrid"}
# ⚠️ MUST be a multiple of 4 that crosses no non-leap century, or 29 Feb has no
# image and collapsing it onto the 28th creates DUPLICATE DATES -- which the
# trainer rejects with "cannot reindex on an axis with duplicate labels".
# +16 maps 1980-2008 -> 1996-2024 and preserves every leap year (2000 is one).
SHIFT_YEARS = 16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="gpu1080/corpora671/camels_corpus_fused4_v2_531")
    ap.add_argument("--out", default="data/modern_corpus/SMOKE_fused4m_cohort")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    os.chdir(ROOT)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.csv.gz"))[:a.n]
    if not files:
        raise SystemExit(f"no source files in {src}")
    n = 0
    for f in files:
        with gzip.open(f, "rt") as fh:
            hdr = fh.readline().rstrip("\n").split(",")
            rows = [ln.rstrip("\n").split(",") for ln in fh]
        new_hdr = []
        for c in hdr:
            for old, new in RENAME.items():
                if c.endswith("_" + old):
                    c = c[: -len(old)] + new
                    break
            new_hdr.append(c)
        op = out / f.name
        with gzip.open(op, "wt") as fh:
            fh.write(",".join(new_hdr) + "\n")
            for r in rows:
                d = dt.date.fromisoformat(r[0])
                # no fallback: with a leap-preserving shift every date has an
                # image, and a silent fallback is what created duplicates before
                r[0] = d.replace(year=d.year + SHIFT_YEARS).isoformat()
                fh.write(",".join(r) + "\n")
        n += 1
    (out / "_SMOKE_README.txt").write_text(
        "SYNTHETIC. Product suffixes renamed and dates shifted +17y purely to "
        "exercise the MODERN-1 launcher. NOT scientific data. Never score it.\n")
    print(f"SMOKE corpus: {n} basins -> {out}")
    print(f"  products now: {sorted({c.rsplit('_', 1)[1] for c in new_hdr if '_' in c})}")
    print(f"  span: {rows[0][0]} (shifted) .. see files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
