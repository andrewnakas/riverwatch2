#!/usr/bin/env python
"""Validate the CAMELS-531 MB-LSTM corpus built by scripts/build_camels_corpus.py.

Checks, per basin:
  1. file exists (531 expected) and has the exact corpus column set
  2. >=95% non-NaN coverage of q_cfs AND every forcing column inside BOTH
     the train window (1999-10-01..2008-09-30) and the test window
     (1989-10-01..1999-09-30)
  3. plausibility: q_cfs >= 0 for (almost) all valid days, precipitation_sum >= 0,
     |temperature| < 60 C, shortwave_radiation_sum in (0, 45] MJ/m2/day

Exit code 0 iff all basins pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = REPO / "data" / "mblstm" / "camels_corpus"

EXPECTED_COLS = [
    "date",
    "q_cfs",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "shortwave_radiation_sum",
]
FORCING_COLS = EXPECTED_COLS[2:]
TEMP_COLS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"]

TRAIN = ("1999-10-01", "2008-09-30")
TEST = ("1989-10-01", "1999-09-30")
MIN_COVERAGE = 0.95
# tolerate a handful of negative q records (bad gauge days), not systematic ones
MAX_NEG_Q_FRAC = 0.001


def window_coverage(df: pd.DataFrame, col: str, start: str, end: str) -> float:
    """Fraction of calendar days in [start, end] with a non-NaN value for col."""
    ndays = (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
    sub = df[(df["date"] >= start) & (df["date"] <= end)]
    return float(sub[col].notna().sum()) / ndays


def validate_basin(path: Path) -> list[str]:
    problems: list[str] = []
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLS:
        problems.append(f"columns={list(df.columns)}")
        return problems
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].duplicated().any():
        problems.append("duplicate dates")

    for wname, (ws, we) in (("train", TRAIN), ("test", TEST)):
        for col in ["q_cfs"] + FORCING_COLS:
            cov = window_coverage(df, col, ws, we)
            if cov < MIN_COVERAGE:
                problems.append(f"{wname} {col} coverage {cov:.3f}")

    q = df["q_cfs"].dropna()
    if len(q):
        neg_frac = float((q < 0).sum()) / len(q)
        if neg_frac > MAX_NEG_Q_FRAC:
            problems.append(f"q<0 frac {neg_frac:.4f} ({int((q < 0).sum())} rows)")
    if (df["precipitation_sum"].dropna() < 0).any():
        problems.append("negative precipitation")
    for col in TEMP_COLS:
        bad = int((df[col].dropna().abs() >= 60).sum())
        if bad:
            problems.append(f"{col} |t|>=60C on {bad} rows")
    rad = df["shortwave_radiation_sum"].dropna()
    if ((rad <= 0) | (rad > 45)).any():
        problems.append("shortwave_radiation_sum outside (0, 45]")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--ids-json", type=Path, default=REPO / "data" / "camels_gauge_ids.json")
    ap.add_argument("--ids-key", default="531")
    args = ap.parse_args()

    ids = json.load(open(args.ids_json))[args.ids_key]
    files = {p.name[:8]: p for p in args.corpus_dir.glob("*.csv.gz")
             if not p.name.startswith("._")}

    missing = [g for g in ids if g not in files]
    extra = sorted(set(files) - set(ids))
    print(f"expected {len(ids)} basins; found {len(files)} csv.gz files "
          f"({len(missing)} missing, {len(extra)} extra)")
    if missing:
        print("missing:", missing[:20])

    n_pass, failures = 0, {}
    for i, gid in enumerate(ids, 1):
        if gid not in files:
            failures[gid] = ["file missing"]
            continue
        probs = validate_basin(files[gid])
        if probs:
            failures[gid] = probs
        else:
            n_pass += 1
        if i % 100 == 0:
            print(f"  ...{i}/{len(ids)} checked", flush=True)

    print(f"\nPASS: {n_pass}/{len(ids)} basins")
    if failures:
        print(f"FAIL: {len(failures)} basins")
        for gid, probs in sorted(failures.items()):
            print(f"  {gid}: {'; '.join(probs)}")
    return 0 if not failures and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
