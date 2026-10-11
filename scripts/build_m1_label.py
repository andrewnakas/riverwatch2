#!/usr/bin/env python3
"""Emit the served member's public label, DERIVED from the gate artifact.

Every number here is read out of `benchmarks/m1_gate1_verdict.json`, never
retyped. A label that quotes a hand-copied figure drifts the moment the gate is
re-read, and this project has already retracted two records for numbers that
were not what they were believed to be.

WHAT THE LABEL MAY AND MAY NOT SAY
----------------------------------
The member trained on NLDAS-2 **areal** (basin-mean) forcing and is served
Open-Meteo **point** forcing. GATE 1 measured that substitution at a paired
-0.059534 median NSE. The honest framing, fixed by the gate's own
pre-registration, is that **the transferable quantity is the DELTA**, not
either arm's level:

  * 0.905829 is perfect-forcing day-1 median NSE on 490 basins (test1). It is
    NOT operational skill and must never label served output.
  * 0.808659 is the as-served median on GATE 1's own sample -- val1, 150
    basins. It is NOT "the served member's skill" either, and quoting it as
    such swaps a 150-basin validation figure for a 490-basin test one.
  * -0.048033 is the measured real-forecast decay, a DIFFERENT frame against a
    DIFFERENT baseline. ⛔ It may not be added to -0.059534; their additivity
    is unmeasured, and a combined figure built by adding them is exactly the
    kind of composed claim this project has been burned by.

So the label states the delta, its CI, its breadth, and the frame it was
measured on -- and says plainly that the served skill has not yet been
measured. `scripts/score_m1_live.py` is what will eventually fill that in,
from forecasts this site actually issued.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE1 = ROOT / "benchmarks" / "m1_gate1_verdict.json"
OUT = ROOT / "data" / "m1_served_skill.json"

SCHEMA_VERSION = "m1-label-1"


def build(gate1: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "member": gate1["member"],
        "seeds": gate1["seeds"],
        "trained_on": "NLDAS-2 basin-mean (areal) forcing",
        "served_on": "Open-Meteo point forcing at the gauge coordinate",
        # The transferable quantity, per GATE 1's own pre-registration.
        "substitution_delta_nse": gate1["paired_delta_point_minus_areal"],
        "substitution_ci95": gate1["ci95"],
        "substitution_breadth": gate1["breadth"],
        "substitution_frame": {
            "name": gate1["frame"],
            "n_basins": gate1["basins"],
            "rows": gate1["rows_joined"],
            "protocol": gate1["protocol"],
        },
        # Filled by scripts/score_m1_live.py once the site has issued enough
        # forecasts to score. Null is the honest value until then, and the UI
        # must say "not yet measured" rather than borrow a research number.
        "live_measured": None,
        "blend_weight": 0.0,
        "_note": (
            "The served member runs on SUBSTITUTED forcing: it trained on "
            "NLDAS-2 areal and is served Open-Meteo point, measured at a paired "
            f"{gate1['paired_delta_point_minus_areal']:.6f} median NSE on "
            f"{gate1['frame']}/{gate1['basins']} basins, with "
            f"{(1.0 - gate1['breadth']) * 100:.1f}% of basins degrading. "
            "Neither arm's level is the served member's skill, and this delta "
            "must not be added to the -0.048033 real-forecast decay: different "
            "frames, additivity unmeasured. The member carries no blend weight "
            "until its live skill is measured on issued forecasts."
        ),
    }


def main() -> int:
    if not GATE1.exists():
        print(f"ERROR: {GATE1} absent; the label's numbers come from it", file=sys.stderr)
        return 1
    label = build(json.loads(GATE1.read_text()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(label, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size} B)")
    print(f"  delta {label['substitution_delta_nse']:.6f} "
          f"CI {label['substitution_ci95']}  breadth {label['substitution_breadth']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
