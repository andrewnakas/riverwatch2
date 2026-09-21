#!/usr/bin/env python3
"""LEDGER 54 — freeze the shipped record artifact, at the PRE-REGISTERED ship bar.

The ship bar (PREREG_v2.md, unchanged since LEDGER 51):
  * 5 matched seeds for the new member (3 is a SCREEN, not a ship);
  * composition decided on **val1**, paired per-basin delta, basin-bootstrap 95% CI
    excluding zero, breadth >= 0.5;
  * duplicate-member control reported (medians AND paired — the medians form has
    misled 7x, so both are recorded);
  * test1 read ONCE, after val1 has decided, and it selects nothing.

⚠️ This deliberately uses the ship-bar seed count, NOT whichever seed count scores
highest: quoting the best depth is an argmax over seed depth
([[seed-depth-helps-the-member-and-can-hurt-the-ensemble]]).

usage: l54_freeze_record.py --members ... --new nhar0h256 --seeds 501,..,505 --out X.json
"""
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l51)          # chdir's to ~/riverwatch2

DD = "data/mblstm/l51_dumps"
NEARING = 0.879


def provenance(members, frame, seeds_of):
    """Every dump that backs the number, with its sidecar's guarded window + ckpt md5."""
    out = []
    for m in members:
        for f in sorted(glob.glob(f"{DD}/camels531_l51_{m}_s*_{frame}.csv.gz")):
            s = int(f.split("_s")[-1].split("_")[0])
            if m in seeds_of and s not in seeds_of[m]:
                continue
            side = json.load(open(f.replace(".csv.gz", ".json")))
            assert side["train_start"] == "1999-10-01" and side["train_end"] == "2008-09-30", side
            out.append({"member": m, "seed": s, "frame": frame, "rows": side["rows"],
                        "ckpt_md5": side.get("ckpt_md5"), "engine": side.get("engine", "mblstm"),
                        "protocol": side["protocol"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True, help="full composition, comma separated")
    ap.add_argument("--new", required=True, help="the member under the ship bar")
    ap.add_argument("--seeds", required=True, help="the new member's ship-bar seeds")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    members = [x for x in a.members.split(",") if x]
    seeds = sorted(int(x) for x in a.seeds.split(",") if x)
    seeds_of = {a.new: set(seeds)}
    assert a.new in members, f"{a.new} not in the composition"

    have = sorted(int(f.split("_s")[-1].split("_")[0])
                  for f in glob.glob(f"{DD}/camels531_l51_{a.new}_s*_test1.csv.gz"))
    missing = [s for s in seeds if s not in have]
    if missing:
        sys.exit(f"SHIP BAR NOT MET: {a.new} is missing test1 dumps for seeds {missing} "
                 f"(have {have}). Refusing to freeze a record below the pre-registered depth.")

    res = {"frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "composition": members, "new_member": a.new, "ship_bar_seeds": seeds,
           "protocol": "nearing2022: train 1999-10-01..2008-09-30, val 1980-10-01..1989-09-30, "
                       "test 1989-10-01..1999-09-30, 531 basins, day-1, ALL daily observations, median NSE",
           "combination": "equal weight, zero fitted parameters", "readout": "(ylo+yhi)/2",
           "reference_nearing_2022_single_model": NEARING}
    try:
        res["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        res["git_commit"] = None

    base = [m for m in members if m != a.new]
    for frame in ("val1", "test1"):                    # val1 FIRST; test1 is read after
        m = None
        for name in members:
            d = l51.load_member(name, frame, seeds=seeds_of.get(name))
            m = d if m is None else m.merge(d.drop(columns=["truth"]), on=["station_id", "t0"], how="inner")
        ens = l51.per_basin(m, members)
        sub = l51.per_basin(m, base)
        d, lo, hi, br, n = l51.paired_ci(ens, sub)
        blk = {"rows": int(len(m)), "basins": int(m.station_id.nunique()),
               "ensemble_median_nse": float(ens.median()),
               "without_new_member": float(sub.median()),
               "new_member_solo": float(l51.per_basin(m, [a.new]).median()),
               "solo": {x: float(l51.per_basin(m, [x]).median()) for x in members},
               "new_member_loo_paired": {"delta": d, "ci": [lo, hi], "breadth": br,
                                         "significant": bool(lo > 0)},
               "loo_all": {}, "dup_control_medians": {}}
        for x in members:
            rest = [y for y in members if y != x]
            s2 = l51.per_basin(m, rest)
            dd, l2, h2, b2, _ = l51.paired_ci(ens, s2)
            blk["loo_all"][x] = {"delta": dd, "ci": [l2, h2], "breadth": b2}
            mm = m.copy(); mm["_dup"] = mm[x]
            blk["dup_control_medians"][x] = float(l51.per_basin(mm, members + ["_dup"]).median() - ens.median())
        res[frame] = blk
        print(f"[{frame}] ensemble {ens.median():.6f}  without {a.new} {sub.median():.6f}  "
              f"{a.new} solo {blk['new_member_solo']:.6f}  paired {d:+.6f} "
              f"CI[{lo:+.6f},{hi:+.6f}] breadth {br:.3f}")

    v = res["val1"]["new_member_loo_paired"]
    res["ship_bar"] = {"seeds_ok": True, "val1_paired_positive": v["delta"] > 0,
                       "val1_ci_excludes_zero": v["significant"], "val1_breadth_ge_0.5": v["breadth"] >= 0.5,
                       "PASS": bool(v["delta"] > 0 and v["significant"] and v["breadth"] >= 0.5)}
    res["record_test1"] = res["test1"]["ensemble_median_nse"]
    res["margin_vs_nearing"] = res["record_test1"] - NEARING
    res["provenance"] = {f: provenance(members, f, seeds_of) for f in ("val1", "test1")}

    json.dump(res, open(a.out, "w"), indent=1)
    h = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    print(f"\nSHIP BAR: {'PASS' if res['ship_bar']['PASS'] else 'FAIL'}   "
          f"RECORD (test1) = {res['record_test1']:.6f}   vs Nearing {NEARING} = "
          f"{res['margin_vs_nearing']:+.6f}")
    print(f"wrote {a.out}  md5 {h}  ({len(res['provenance']['test1'])} test1 dumps backing it)")
    return 0 if res["ship_bar"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
