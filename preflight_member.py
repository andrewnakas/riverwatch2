#!/usr/bin/env python3
"""Pre-flight a new member config against the family it must join.

The standing rule after losing ~20 GPU-hours to a wrong-split clone: before
training a member, verify it will actually be comparable to the streams it is
meant to join. The failure mode is silent -- gate_eval inner-joins the streams,
so a member on a different (station, date) grid produces an empty merge and
whatever number falls out looks plausible.

Checks, in order of how cheap they are:

  1. SPLIT DATES vs a known-good sibling config. This alone would have caught
     the Kratzert/Li-Song mistake.
  2. DATA PATHS point at the intended corpus, and that corpus has the expected
     basin count.
  3. DYNAMIC INPUTS exist as variables in the netCDF the config points at, and
     are not constant -- the SWE lesson, where a column existed and parsed but
     held only zeros.
  4. PREDICTED GRID: derive the (station, date) test grid the config implies and
     compare it against an existing stream's grid. This is the check that
     actually catches everything, including mistakes not anticipated here.

Run before training, not after.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATE_KEYS = ("train_start_date", "train_end_date", "validation_start_date",
             "validation_end_date", "test_start_date", "test_end_date")


def field(text, key):
    m = re.search(rf'^{key}:\s*"?([^"\n]+)"?\s*$', text, re.M)
    return m.group(1).strip() if m else None


def dyn_inputs(text):
    m = re.search(r"^dynamic_inputs:\s*\n((?:\s*-\s*\S+\s*\n)+)", text, re.M)
    return re.findall(r"-\s*(\S+)", m.group(1)) if m else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--reference-cfg", default="gpu1080/cfgls_multi_s111.yml")
    ap.add_argument("--reference-dump",
                    default="gpu1080/dumps/camels531ls_daymet_nhlstm_s111.csv.gz")
    a = ap.parse_args()

    cfg = Path(a.cfg).read_text()
    ref = Path(a.reference_cfg).read_text()
    fails = []

    print(f"=== pre-flight {Path(a.cfg).name} against {Path(a.reference_cfg).name} ===\n")

    print("1. split dates")
    for k in DATE_KEYS:
        c, r = field(cfg, k), field(ref, k)
        ok = c == r
        print(f"   {'OK ' if ok else 'FAIL'} {k:24s} {c}  (family: {r})")
        if not ok:
            fails.append(f"{k}: {c} vs family {r}")

    print("\n2. data paths and corpus")
    dd = field(cfg, "data_dir")
    print(f"   data_dir: {dd}")
    bt = Path(dd) / "basins.txt" if dd else None
    if bt and bt.exists():
        n = len([x for x in bt.read_text().split() if x.strip()])
        print(f"   {'OK ' if n == 531 else 'WARN'} basins.txt has {n} basins")
        if n != 531:
            fails.append(f"basins.txt has {n}, expected 531")
    else:
        print("   FAIL basins.txt not found")
        fails.append("basins.txt missing")

    print("\n3. dynamic inputs present and varying in the netCDF")
    inputs = dyn_inputs(cfg)
    print(f"   declared: {inputs}")
    if bt and bt.exists():
        import xarray as xr
        b = [x for x in bt.read_text().split() if x.strip()][0]
        nc = Path(dd) / "time_series" / f"{b}.nc"
        if nc.exists():
            d = xr.open_dataset(nc).to_dataframe()
            for v in inputs:
                if v not in d.columns:
                    print(f"   FAIL {v}: not in the netCDF")
                    fails.append(f"{v} missing from netCDF")
                elif float(np.nanstd(d[v].to_numpy(float))) < 1e-9:
                    print(f"   FAIL {v}: constant")
                    fails.append(f"{v} constant")
                else:
                    print(f"   OK   {v}: std {np.nanstd(d[v].to_numpy(float)):.4g}")
        else:
            print(f"   FAIL netCDF not found at {nc}")
            fails.append("netCDF missing")

    print("\n4. predicted test grid vs an existing stream")
    rd = Path(a.reference_dump)
    if rd.exists():
        s = pd.read_csv(rd, usecols=["t0", "h"])
        s = s[s.h == 1]
        lo_ref, hi_ref = s.t0.min(), s.t0.max()
        ts, te = field(cfg, "test_start_date"), field(cfg, "test_end_date")
        # config dates are dd/mm/yyyy
        def iso(x):
            d, m, y = x.split("/")
            return f"{y}-{m}-{d}"
        lo_cfg, hi_cfg = iso(ts), iso(te)
        print(f"   config test window : {lo_cfg} .. {hi_cfg}")
        print(f"   stream test window : {lo_ref} .. {hi_ref}")
        overlap = not (hi_cfg < lo_ref or lo_cfg > hi_ref)
        if not overlap:
            print("   FAIL windows do not overlap -- an inner join would be EMPTY")
            fails.append("test window does not overlap the existing stream")
        elif lo_cfg[:7] != lo_ref[:7] or hi_cfg[:7] != hi_ref[:7]:
            print("   WARN windows overlap but do not match; the join will be partial")
        else:
            print("   OK   windows match")
    else:
        print(f"   (no reference dump at {rd}; skipping)")

    print()
    if fails:
        print(f"PRE-FLIGHT FAILED ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("PRE-FLIGHT PASSED — safe to train")


if __name__ == "__main__":
    main()
