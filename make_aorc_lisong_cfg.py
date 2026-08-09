#!/usr/bin/env python3
"""Regenerate the AORC no-q configs from the CORRECT template.

The previous configs were cloned from cfg_daymet_s111.yml, which is a with-q era
file on the KRATZERT split (train 1999-2008, test 1989-99). Every no-q stream in
gate_eval uses the LI/SONG split (train 1980-95, test 1995-2010). The two produce
non-overlapping (station, date) grids, so the members could never join the
ensemble -- an inner join against an existing stream returned zero rows.

The earlier clone script asserted that only the experiment name and data paths
differed from its reference, and that assertion PASSED, because the dates matched
the reference exactly. The reference was the wrong file. An assertion of internal
consistency is blind to that.

So this generates from nh_config_lisong.yml.tmpl -- the template the existing
streams are actually built from -- and then validates the OUTPUT against the
family it must join, which is the check that would have caught the original
error:

  * the split dates must match an existing no-q config exactly
  * the data paths must point at the intended corpus
  * the dynamic inputs must be the intended set

Emits configs for the plain AORC member and, behind a flag, the intensity
variant, so the intensity comparison can be redone on the right split if it is
ever worth revisiting.
"""
import argparse
import re
import sys
from pathlib import Path

TMPL = Path("gpu1080/nh_config_lisong.yml.tmpl")
REF = Path("gpu1080/cfgls_multi_s111.yml")      # a known-good no-q config
BASE = "/home/nakas/riverwatch2/gpu1080"

DATE_KEYS = ("train_start_date", "train_end_date", "validation_start_date",
             "validation_end_date", "test_start_date", "test_end_date")
INTENSITY = ["p_max_1h", "p_max_3h", "p_hours", "p_cv", "p_centroid"]


def dates_of(text):
    return {k: re.search(rf'^{k}:\s*"?([^"\n]+)"?\s*$', text, re.M).group(1).strip()
            for k in DATE_KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcing", default="aorc", help="aorc or aorcx")
    ap.add_argument("--seeds", default="111,222,333")
    args = ap.parse_args()

    tmpl = TMPL.read_text()
    ref_dates = dates_of(REF.read_text())
    print(f"reference no-q split ({REF.name}):")
    for k, v in ref_dates.items():
        print(f"  {k:24s} {v}")

    made = []
    for seed in [s.strip() for s in args.seeds.split(",") if s.strip()]:
        t = tmpl.replace("__BASE__", BASE)
        t = t.replace("FORCING", args.forcing).replace("SEED", seed)

        if args.forcing == "aorcx":
            m = re.search(r"^dynamic_inputs:\s*\n((?:\s*-\s*\S+\s*\n)+)", t, re.M)
            if not m:
                sys.exit("no dynamic_inputs block in the template")
            indent = re.match(r"(\s*)-", m.group(1)).group(1)
            t = (t[:m.end(1)] + "".join(f"{indent}- {v}\n" for v in INTENSITY)
                 + t[m.end(1):])

        out = Path(f"gpu1080/cfgls_{args.forcing}_s{seed}.yml")
        out.write_text(t)
        made.append(out)

    print(f"\nwrote: {[p.name for p in made]}")

    # validate against the FAMILY, not against whatever file was nearest
    bad = 0
    for p in made:
        txt = p.read_text()
        d = dates_of(txt)
        if d != ref_dates:
            print(f"  FAIL {p.name}: split differs from {REF.name}")
            for k in DATE_KEYS:
                if d[k] != ref_dates[k]:
                    print(f"      {k}: {d[k]!r} vs expected {ref_dates[k]!r}")
            bad += 1
            continue
        if f"/nh_data/{args.forcing}" not in txt:
            print(f"  FAIL {p.name}: data paths do not point at nh_data/{args.forcing}")
            bad += 1
            continue
        if args.forcing == "aorcx" and not all(v in txt for v in INTENSITY):
            print(f"  FAIL {p.name}: missing intensity inputs")
            bad += 1
            continue
        print(f"  OK   {p.name}: split matches the no-q family, paths correct")
    if bad:
        sys.exit(f"{bad} config(s) failed validation")
    print("\nall configs validated against the existing no-q split")


if __name__ == "__main__":
    main()
