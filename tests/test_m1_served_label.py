"""The shipped label must match the gate artifact it claims to report.

A label is public copy about a model's measured skill. If it drifts from the
artifact -- because someone retyped a number, or the gate was re-read and the
label was not regenerated -- the site states something no measurement supports.
This project has already retracted two records over numbers that were not what
they were believed to be, so the label is DERIVED and this test pins it.

Mirrors `test_shipped_skill_file_matches_what_the_ui_reads` for the transfer
sidecar: same contract, different artifact.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LABEL = ROOT / "data" / "m1_served_skill.json"
GATE1 = ROOT / "benchmarks" / "m1_gate1_verdict.json"


def _label():
    if not LABEL.exists():
        pytest.skip("label not generated in this checkout")
    return json.loads(LABEL.read_text())


def test_label_matches_the_gate_artifact():
    if not GATE1.exists():
        pytest.skip("gate 1 verdict absent")
    lab, g1 = _label(), json.loads(GATE1.read_text())
    assert lab["substitution_delta_nse"] == g1["paired_delta_point_minus_areal"]
    assert lab["substitution_ci95"] == g1["ci95"]
    assert lab["substitution_breadth"] == g1["breadth"]
    assert lab["substitution_frame"]["n_basins"] == g1["basins"]
    assert lab["substitution_frame"]["name"] == g1["frame"]
    assert lab["member"] == g1["member"]


def test_label_is_regenerated_not_edited():
    """`scripts/build_m1_label.py` rebuilds it byte-for-byte from the artifact.
    If this fails, someone hand-edited the label."""
    if not GATE1.exists():
        pytest.skip("gate 1 verdict absent")
    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.build_m1_label import build
    assert build(json.loads(GATE1.read_text())) == _label()


def test_served_skill_is_null_until_measured():
    """The honest value until the site has issued enough forecasts to score.
    A borrowed research number here would be the whole problem."""
    lab = _label()
    assert lab["live_measured"] is None or isinstance(lab["live_measured"], dict)
    assert lab["blend_weight"] == 0.0, "the member must carry no blend weight"


def test_label_never_carries_a_forbidden_figure():
    blob = json.dumps(_label())
    for forbidden in ("0.905829", "0.9058", "0.1076"):
        assert forbidden not in blob, \
            f"{forbidden} is not this member's served skill and must not appear"


def test_label_stays_small_enough_to_ship_per_page():
    assert LABEL.stat().st_size < 4096
