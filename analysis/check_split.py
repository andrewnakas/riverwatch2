#!/usr/bin/env python3
"""Refuse to dump a run that is not on the Li/Song split.

WHY THIS EXISTS
---------------
Twice now a run has reached the dump step on the WRONG evaluation split:
  - 2026-08-03: all four AORC no-q members (~20 GPU-h)
  - 2026-08-08: rw2_aorcx_lstm_mm_s222, whose test period (1989-1999) sits
                ENTIRELY INSIDE our training window (1980-1995)

Both times the run looked perfect by every other signal -- correct epoch count,
clean loss curve, 531 basins, intact corpus. Only train/test_start_date reveals
it, and nothing checked them.

Li/Song (every no-q stream we score):
    train 01/10/1980 - 30/09/1995
    test  01/10/1995 - 30/09/2010

Usage:
    check_split.py <run_dir>            # exits 1 if the split is wrong
    check_split.py <run_dir> --quiet    # no output, just the exit code

Wire into any dump step:
    check_split.py "$R" || { echo "WRONG SPLIT, refusing to dump"; exit 1; }
"""
import sys
from pathlib import Path

EXPECT = {
    "train_start_date": "01/10/1980",
    "train_end_date": "30/09/1995",
    "test_start_date": "01/10/1995",
    "test_end_date": "30/09/2010",
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv
    if not args:
        print(__doc__)
        return 2

    cfg = Path(args[0])
    if cfg.is_dir():
        cfg = cfg / "config.yml"
    if not cfg.exists():
        if not quiet:
            print(f"FAIL: no config at {cfg}")
        return 1

    got = {}
    for line in cfg.read_text().splitlines():
        line = line.strip()
        for k in EXPECT:
            if line.startswith(k + ":"):
                got[k] = line.split(":", 1)[1].strip().strip('"').strip("'")

    bad = [k for k, v in EXPECT.items() if got.get(k) != v]
    if not quiet:
        for k, v in EXPECT.items():
            mark = "ok " if got.get(k) == v else "BAD"
            print(f"  {mark} {k:<18} got {got.get(k, '<missing>'):<12} want {v}")
    if bad:
        if not quiet:
            print(f"\nWRONG SPLIT ({len(bad)} field(s) differ) -- refusing.")
            print("This is the Kratzert-split failure that cost ~20 GPU-h in Aug.")
        return 1
    if not quiet:
        print("\nLi/Song split confirmed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
