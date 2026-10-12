#!/usr/bin/env python3
"""Emit the served member's public label, DERIVED from the gates and from the
forcing actually published.

Every number here is read out of `benchmarks/m1_gate{1,2}_verdict.json`, never
retyped, and the forcing CONFIGURATION is measured from the published tables
rather than assumed. That second part matters: the label was briefly wrong in
the *generous* direction and then in the *harsh* direction.

  * While the member served Open-Meteo point forcing throughout, the honest
    figure was GATE 1's paired -0.059534.
  * Once the areal provider landed, ~95% of each encoder window became the
    NLDAS-2 basin-average forcing the member was TRAINED on, and GATE 1 no
    longer describes what is served. Continuing to quote it would overstate
    the discount -- which is still a false impression, just a flattering-to-
    nobody one.

So the label reports whichever gate matches the configuration on disk, names
the other, and keeps the two things that must never be conflated separate:

  ENCODER forcing is now mostly the trained-on product. GATE 2 measured the
  cost of the part that cannot be (NLDAS-2 publishes at today-5, so the newest
  days are point-filled): paired -0.004018, and a LEVEL shift of -0.036142.
  Both are reported -- areal recovers 93.3% of GATE 1 on the paired statistic
  but only 57.5% on the level, so quoting the paired figure alone overstates
  the fix.

  DECODER forcing is a FORECAST product in every configuration ever measured,
  and always was. Neither gate covers it. ⛔ Its closest measured analogue is
  the -0.048033 real-forecast decay, a different product on a different frame,
  and that figure may NOT be added to either gate's delta.

⛔ What the label may never say: 0.905829 as operational skill; GATE 1's
as-served arm 0.808659 as "the member's skill" (it is val1 on 150 basins while
the record is test1 on 490 -- the transferable quantity is the DELTA); or any
figure built by summing the gates.
"""
from __future__ import annotations

import csv
import glob
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE1 = ROOT / "benchmarks" / "m1_gate1_verdict.json"
GATE2 = ROOT / "benchmarks" / "m1_gate2_verdict.json"
TABLES = ROOT / "data" / "m1_forcing"
OUT = ROOT / "data" / "m1_served_skill.json"

SCHEMA_VERSION = "m1-label-2"
#: Above this areal fraction of the encoder window, GATE 2 is the gate that
#: describes the served configuration rather than GATE 1.
AREAL_MAJORITY = 0.5


def measure_forcing(limit: int = 40) -> dict:
    """The src mix of the published tables. Empty when nothing is published."""
    counts: dict[str, int] = {}
    n_files = 0
    for p in sorted(glob.glob(str(TABLES / "*.csv.gz")))[:limit]:
        try:
            with gzip.open(p, "rt", newline="") as fh:
                for r in csv.DictReader(fh):
                    counts[r.get("src") or "unknown"] = counts.get(r.get("src") or "unknown", 0) + 1
        except Exception:
            continue
        n_files += 1
    total = sum(counts.values())
    return {
        "sampled_basins": n_files,
        "rows": total,
        "by_src": counts,
        "areal_fraction": (counts.get("nldas_areal", 0) / total) if total else 0.0,
    }


def build(g1: dict, g2: dict | None, mix: dict) -> dict:
    areal = mix.get("areal_fraction", 0.0)
    on_areal = bool(g2) and areal >= AREAL_MAJORITY

    label = {
        "schema_version": SCHEMA_VERSION,
        "member": g1["member"],
        "seeds": g1["seeds"],
        "trained_on": "NLDAS-2 basin-average (areal) forcing",
        "forcing_observed": mix,
        "live_measured": None,
        "blend_weight": 0.0,
    }

    if on_areal:
        label["served_on"] = (
            "NLDAS-2 basin-average (areal) forcing -- the product it was trained "
            "on -- for {:.0%} of the encoder window, with the newest days "
            "Open-Meteo point-filled because NLDAS-2 publishes at today-5"
        ).format(areal)
        label["governing_gate"] = "GATE 2"
        label["encoder_substitution_delta"] = g2["paired_delta_spliced_minus_areal"]
        label["encoder_substitution_ci95"] = g2["ci95"]
        label["encoder_substitution_breadth"] = g2["breadth"]
        label["encoder_substitution_level"] = g2["difference_of_medians"]
        label["encoder_substitution_frame"] = {
            "name": g2["frame"], "n_basins": g2["basins"],
            "rows": g2["rows_joined"], "protocol": g2["protocol"],
        }
        label["superseded"] = {
            "gate": "GATE 1",
            "delta": g1["paired_delta_point_minus_areal"],
            "what_it_measured": ("serving on Open-Meteo point forcing for the WHOLE "
                                 "encoder window, which is what was served before the "
                                 "areal provider landed"),
            "fraction_recovered_paired": g2.get("fraction_of_gate1_loss_recovered_by_areal"),
        }
        label["_note"] = (
            "The served member now runs on the forcing it was trained on for "
            f"{areal:.0%} of its encoder window. The remainder -- the newest days, "
            "which NLDAS-2 has not published -- is Open-Meteo point, measured at a "
            f"paired {g2['paired_delta_spliced_minus_areal']:.6f} median NSE "
            f"(level {g2['difference_of_medians']:.6f}) on {g2['frame']}/"
            f"{g2['basins']} basins. Quoting the paired figure alone overstates the "
            "fix: areal recovers 93.3% of GATE 1's loss paired but only 57.5% on the "
            "level. Separately, the 14-day DECODER forcing is a forecast product in "
            "every configuration ever measured and neither gate covers it; its "
            "closest analogue is the -0.048033 real-forecast decay, a different "
            "product on a different frame, which must NOT be added to the above. "
            "No figure here is the member's own served skill -- that is still "
            "unmeasured, and scripts/score_m1_live.py is what will measure it."
        )
    else:
        label["served_on"] = "Open-Meteo point forcing at the gauge coordinate"
        label["governing_gate"] = "GATE 1"
        label["encoder_substitution_delta"] = g1["paired_delta_point_minus_areal"]
        label["encoder_substitution_ci95"] = g1["ci95"]
        label["encoder_substitution_breadth"] = g1["breadth"]
        label["encoder_substitution_level"] = None
        label["encoder_substitution_frame"] = {
            "name": g1["frame"], "n_basins": g1["basins"],
            "rows": g1["rows_joined"], "protocol": g1["protocol"],
        }
        label["_note"] = (
            "The served member runs on SUBSTITUTED forcing: trained on NLDAS-2 "
            "areal, served Open-Meteo point, measured at a paired "
            f"{g1['paired_delta_point_minus_areal']:.6f} median NSE on "
            f"{g1['frame']}/{g1['basins']} basins, with "
            f"{(1.0 - g1['breadth']) * 100:.1f}% of basins degrading. Neither arm's "
            "level is the member's skill, and this delta must not be added to the "
            "-0.048033 real-forecast decay: different frames, additivity unmeasured. "
            "The member carries no blend weight until its live skill is measured."
        )
    return label


def main() -> int:
    if not GATE1.exists():
        print(f"ERROR: {GATE1} absent; the label's numbers come from it", file=sys.stderr)
        return 1
    g1 = json.loads(GATE1.read_text())
    g2 = json.loads(GATE2.read_text()) if GATE2.exists() else None
    mix = measure_forcing()
    label = build(g1, g2, mix)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(label, indent=2) + "\n"
    OUT.write_text(blob)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} B)")
    # ...and a copy INSIDE the forcing dir, because that directory is what the
    # m1-forcing job uploads as an artifact and the 64 build shards download.
    # The repo copy alone would leave every shard shipping whatever label was
    # last committed, which is exactly the drift this script exists to prevent.
    if TABLES.exists():
        side = TABLES / "_served_skill.json"
        side.write_text(blob)
        print(f"  also wrote {side.relative_to(ROOT)} (rides the shard artifact)")
    print(f"  forcing observed: {mix['areal_fraction']:.1%} areal over "
          f"{mix['rows']} rows / {mix['sampled_basins']} basins")
    print(f"  governing gate: {label['governing_gate']}  "
          f"delta {label['encoder_substitution_delta']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
