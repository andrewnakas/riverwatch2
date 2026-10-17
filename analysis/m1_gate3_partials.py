#!/usr/bin/env python
"""GATE 3 — the two partial-coverage basins, reported SEPARATELY.

PREREG_v2.md GATE 3 ADDENDUM 2026-10-16 fixed, before any score, that the
basins with 25-90% of the val window observed are scored and reported on their
own and are NEVER pooled into the deciding median: a basin contributing ~1,100
days is not comparable to one contributing 1,461, and a median whose rows
differ between arms has misled this project before.

The main run reported n_scored = 0 for them because the treatment corpus was
built from the 130-basin scoreable list, so the dump never covered them. This
closes that as a deviation rather than leaving an unexplained zero.

⚠️ With n = 2 there is no median worth bootstrapping and no CI worth printing.
Both basins are reported individually, and that is the honest limit of what two
basins can say.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = Path(HERE).parent
spec = importlib.util.spec_from_file_location(
    "l51", os.path.join(HERE, "l51_withq_score.py"))
l51 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l51)

COVERAGE = ROOT / "benchmarks" / "m1_gate3_coverage.json"


def main() -> int:
    os.chdir(ROOT)
    cov = json.loads(COVERAGE.read_text())
    part = cov["groups"]["partial"]
    ids = set(part["ids"])
    days = part.get("observed_days") or {}

    before = len(l51.IDS)
    l51.IDS = set(ids)
    print(f"  widened l51.IDS from {before} (the 531 subset) to {len(l51.IDS)} "
          f"= the {len(ids)} partial-coverage basins")

    d = l51.load_member("nldasm9", "val1", seeds=[501, 502, 503, 504, 505],
                        protocol="modern-v3-9yr", prefix="g3pt_")
    got = set(d.station_id.unique())
    if got != ids:
        print(f"ABORT: dumps cover {sorted(got)}, registered set is {sorted(ids)}")
        return 1

    nse = l51.per_basin(d, ["nldasm9"])
    verdict = json.loads((ROOT / "benchmarks" / "m1_gate3_verdict.json").read_text())
    full_median = verdict["out_of_cohort"]["median_nse"]

    rows = []
    print()
    print("%-12s %7s %12s %10s" % ("station_id", "rows", "val1 NSE", "obs days"))
    for sid in sorted(ids):
        n_rows = int((d.station_id == sid).sum())
        v = float(nse[sid]) if sid in nse.index else float("nan")
        rows.append({"station_id": sid, "rows": n_rows, "val1_nse": v,
                     "observed_days_in_val_window": days.get(sid)})
        print("%-12s %7d %12.6f %10s" % (sid, n_rows, v, days.get(sid, "?")))

    out = {
        "_note": (
            "GATE 3's partial-coverage basins, scored separately and NEVER "
            "pooled into the deciding out-of-cohort median. Registered in "
            "PREREG_v2.md GATE 3 ADDENDUM 2026-10-16. With n=2 there is no "
            "median worth bootstrapping; both basins are reported "
            "individually and that is the limit of what they can say."),
        "frame": "val1", "member": "nldasm9",
        "seeds": [501, 502, 503, 504, 505], "protocol": "modern-v3-9yr",
        "basins": rows,
        "deciding_median_they_are_excluded_from": full_median,
        "may_be_served": None,
        "_may_be_served_why": (
            "Not decided by this artifact. These basins clear the SERVING "
            "criterion (recent discharge) but carry only 25-90% of the val "
            "window, so their own numbers rest on fewer days than every other "
            "basin in the gate. Serving them is a separate decision and must "
            "quote these per-basin figures, not the 130-basin median."),
    }
    p = ROOT / "benchmarks" / "m1_gate3_partials.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    print()
    print(f"  for reference, the deciding 130-basin median was {full_median:.6f}")
    print(f"  these two are NOT part of it")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
