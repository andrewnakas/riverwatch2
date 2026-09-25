#!/usr/bin/env python3
"""Audit every stream's seeds for one that is dragging its seed average down.

Streams enter the production ensemble as a MEAN over seeds, so a single badly
converged seed silently costs skill on every downstream number. This campaign
has already caught two by hand -- nldas s111 (0.642 against siblings at 0.73)
and LSTMmulti s333/s444 (val NSE 0.60/0.76 against a 0.911-0.917 cluster) -- but
those were found because someone happened to look. This checks all of them the
same way, on the train-side frame, so the screen is complete rather than
anecdotal.

Zero GPU: it reads dumps that already exist.
"""
import glob
import re
import sys
from collections import defaultdict

sys.path.insert(0, "analysis")
from gate_l41 import load, per_basin_nse

SKIP = ("STALE", "QUARANTINE", "_OLD", "_bak", ".bak", "WRONG", "_test_")
GAP_BAR = 0.03

files = [f for f in sorted(glob.glob("gpu1080/dumps/*_TRAIN_*.csv.gz"))
         if not any(m in f for m in SKIP)]
groups = defaultdict(list)
for f in files:
    m = re.search(r"(.+)_TRAIN_(s\d+)\.csv\.gz$", f)
    if m:
        groups[m.group(1)].append((m.group(2), f))

print(f"{len(files)} TRAIN dumps across {len(groups)} streams\n")
print(f"{'stream':40s} {'n':>2s}  per-seed median NSE (val slice from 1990-10-01)")
flags = []
for stream, seeds in sorted(groups.items()):
    if len(seeds) < 2:
        continue
    meds = []
    for s, f in sorted(seeds):
        d = load(f)
        d = d[d["date"] >= "1990-10-01"]
        n = per_basin_nse(d)
        meds.append((s, float(n.median()), len(n)))
    best = max(m for _, m, _ in meds)
    name = stream.replace("gpu1080/dumps/", "")
    print(f"{name:40s} {len(meds):2d}  " + "  ".join(f"{s}={m:.4f}" for s, m, _ in meds))
    for s, m, _ in meds:
        if best - m > GAP_BAR:
            flags.append((name, s, m, best, best - m))

print()
if flags:
    print(f"SEEDS MORE THAN {GAP_BAR} BELOW THEIR STREAM'S BEST:")
    for name, s, m, best, gap in sorted(flags, key=lambda x: -x[4]):
        print(f"  {name:38s} {s}  {m:.4f} vs best {best:.4f}   gap -{gap:.4f}")
else:
    print(f"no seed more than {GAP_BAR} below its stream's best")
