"""The shipped label must describe the forcing that is actually served.

A label is public copy about a model's measured skill, and it can be wrong in
two directions. It was briefly wrong in both:

  * Before the areal provider, GATE 1's paired -0.059534 was the honest figure.
  * After it, ~98% of each encoder window is the NLDAS-2 basin-average forcing
    the member was TRAINED on, and GATE 1 no longer describes what is served.
    Continuing to quote it OVERSTATES the discount -- still a false impression.

So the label is derived from the gate artifacts AND from the forcing on disk,
and these tests pin that it switches with the configuration rather than being
fixed at whatever was true when someone last edited it.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_m1_label.py"
LABEL = ROOT / "data" / "m1_served_skill.json"
GATE1 = ROOT / "benchmarks" / "m1_gate1_verdict.json"
GATE2 = ROOT / "benchmarks" / "m1_gate2_verdict.json"


def _load(name):
    p = ROOT / "benchmarks" / name
    if not p.exists():
        pytest.skip(f"{name} absent")
    return json.loads(p.read_text())


def _build(areal_rows: int, point_rows: int, tmp_path):
    """Run the real builder against a synthetic forcing table of a given mix."""
    import gzip, csv
    sand = tmp_path
    (sand / "scripts").mkdir(parents=True, exist_ok=True)
    (sand / "scripts" / "build_m1_label.py").write_text(BUILDER.read_text())
    (sand / "benchmarks").mkdir(parents=True, exist_ok=True)
    for n in ("m1_gate1_verdict.json", "m1_gate2_verdict.json"):
        src = ROOT / "benchmarks" / n
        if src.exists():
            (sand / "benchmarks" / n).write_text(src.read_text())
    td = sand / "data" / "m1_forcing"
    td.mkdir(parents=True, exist_ok=True)
    cols = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
    if areal_rows or point_rows:
        with gzip.open(td / "00000001.csv.gz", "wt", newline="") as gz:
            w = csv.writer(gz)
            w.writerow(["date"] + cols + ["src"])
            i = 0
            for src, n in (("nldas_areal", areal_rows), ("om_point", point_rows)):
                for _ in range(n):
                    i += 1
                    w.writerow([f"2026-01-{i % 28 + 1:02d}"] + [1.0] * 6 + [src])
    else:
        td.rmdir()
    r = subprocess.run([sys.executable, "scripts/build_m1_label.py"], cwd=sand,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((sand / "data" / "m1_served_skill.json").read_text())


def test_no_forcing_published_reports_gate1(tmp_path):
    """With nothing published the member runs on the live point fetch, which is
    what GATE 1 measured."""
    lab = _build(0, 0, tmp_path)
    g1 = _load("m1_gate1_verdict.json")
    assert lab["governing_gate"] == "GATE 1"
    assert lab["encoder_substitution_delta"] == g1["paired_delta_point_minus_areal"]


def test_mostly_areal_forcing_reports_gate2(tmp_path):
    """Once the areal provider dominates the window, GATE 2 is the gate that
    measured the served configuration."""
    if not GATE2.exists():
        pytest.skip("gate 2 verdict absent")
    lab = _build(400, 20, tmp_path)
    g2 = _load("m1_gate2_verdict.json")
    assert lab["governing_gate"] == "GATE 2"
    assert lab["encoder_substitution_delta"] == g2["paired_delta_spliced_minus_areal"]
    assert lab["encoder_substitution_level"] == g2["difference_of_medians"]
    assert lab["forcing_observed"]["areal_fraction"] > 0.9
    # The superseded gate must still be NAMED, not quietly dropped.
    assert lab["superseded"]["gate"] == "GATE 1"
    assert lab["superseded"]["delta"] == _load("m1_gate1_verdict.json")[
        "paired_delta_point_minus_areal"]


def test_gate2_label_carries_the_level_not_just_the_paired_delta(tmp_path):
    """Areal recovers 93.3% of GATE 1's loss paired but only 57.5% on the
    level, so the paired figure alone overstates the fix."""
    if not GATE2.exists():
        pytest.skip("gate 2 verdict absent")
    lab = _build(400, 20, tmp_path)
    assert lab["encoder_substitution_level"] is not None
    assert "level" in lab["_note"]


def test_the_shipped_label_matches_its_own_builder():
    """If this fails, the shipped label was hand-edited or left stale."""
    if not LABEL.exists():
        pytest.skip("label not generated in this checkout")
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_lbl", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shipped = json.loads(LABEL.read_text())
    rebuilt = mod.build(json.loads(GATE1.read_text()),
                        json.loads(GATE2.read_text()) if GATE2.exists() else None,
                        mod.measure_forcing())
    assert rebuilt == shipped, "shipped label differs from what the builder produces"


def test_served_skill_is_null_until_measured():
    if not LABEL.exists():
        pytest.skip("label absent")
    lab = json.loads(LABEL.read_text())
    assert lab["live_measured"] is None or isinstance(lab["live_measured"], dict)
    assert lab["blend_weight"] == 0.0, "the member must carry no blend weight"


def test_label_never_carries_a_forbidden_figure():
    if not LABEL.exists():
        pytest.skip("label absent")
    blob = json.dumps(json.loads(LABEL.read_text()))
    for forbidden in ("0.905829", "0.9058", "0.1076"):
        assert forbidden not in blob, \
            f"{forbidden} is not this member's served skill and must not appear"


def test_label_stays_small_enough_to_ship_per_page():
    if not LABEL.exists():
        pytest.skip("label absent")
    assert LABEL.stat().st_size < 4096


# ---------------------------------------------------------------------------
# Two copies of the label can exist and disagree. Resolution must be in ONE
# place: it was briefly duplicated with different rules, and the live site then
# showed a /benchmarks page quoting GATE 1 beside a forecast panel quoting
# GATE 2 -- for the same member, in the same build.
# ---------------------------------------------------------------------------

def test_the_artifact_copy_wins_over_the_committed_one(tmp_path, monkeypatch):
    import importlib
    from app import modern_forcing as mf
    importlib.reload(mf)
    monkeypatch.setattr(mf, "TABLE_DIR", tmp_path / "m1_forcing")
    (tmp_path / "m1_forcing").mkdir(parents=True)
    (tmp_path / mf.LABEL_NAME).write_text("{}")
    assert mf.label_path().name == mf.LABEL_NAME, "committed copy when no artifact copy"
    (tmp_path / "m1_forcing" / "_served_skill.json").write_text("{}")
    assert mf.label_path().parent.name == "m1_forcing", "artifact copy must win"


def test_every_consumer_resolves_through_the_one_function():
    """A consumer that builds the path itself will drift the moment the two
    copies differ, which is exactly what happened."""
    for rel in ("scripts/build_benchmarks_page.py", "scripts/build_static_site.py"):
        src = (ROOT / rel).read_text()
        assert "label_path" in src, f"{rel} does not use the resolver"
        # and must not hand-build the artifact path alongside it
        assert '"_served_skill.json"' not in src, \
            f"{rel} hand-builds the artifact path instead of resolving it"
