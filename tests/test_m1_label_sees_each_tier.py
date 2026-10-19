"""A pooled median over a heterogeneous served domain cannot see a subgroup fail.

Measured live 2026-10-19, two days after GATE 3 widened the served domain to
620 basins. The areal append's `--basins` named only `data/m1_cohort.json`, so
nothing ever appended to the 130 tier-2 basins: they sat frozen at the date
they were seeded, with an areal gap of **10** days and climbing, while tier 1
sat at 7.

The published label reported a single `substituted_encoder_days_median` of 7
and looked unremarkable -- because 130 of 620 cannot move a median. GATE 3
measured those basins on FULL areal forcing, so an ever-widening point fill is
a configuration no gate has priced, and the one metric that should have caught
it was arithmetically incapable of doing so.

A basin is served from its own tier, never from the average of both. So the
label reports every tier and keys its verdict on the WORST one.
"""
import gzip
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADER = ("date,temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
          "precipitation_sum,shortwave_radiation_sum,vapor_pressure,src")


def _load():
    spec = importlib.util.spec_from_file_location(
        "_m1tier", ROOT / "scripts" / "build_m1_label.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m1tier"] = mod
    spec.loader.exec_module(mod)
    return mod


def _table(d: Path, sid: str, gap: int):
    today = date.today()
    t0 = today - timedelta(days=1)
    newest_areal = t0 - timedelta(days=gap)
    lines = [HEADER]
    cur = today - timedelta(days=400)
    while cur <= t0:
        src = "nldas_areal" if cur <= newest_areal else "om_point"
        lines.append(f"{cur},10,15,5,1.0,12.0,900,{src}")
        cur += timedelta(days=1)
    for i in range(14):
        lines.append(f"{today + timedelta(days=i)},10,15,5,1.0,12.0,900,om_point")
    with gzip.open(d / f"{sid}.csv.gz", "wt", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


def _mix(tmp_path, monkeypatch, *, tier1_gap, tier2_gap, n1=12, n2=4):
    """Tier 2 deliberately the MINORITY, as it is in reality (130 of 620), so a
    pooled median genuinely cannot reach it."""
    lab = _load()
    td = tmp_path / "m1_forcing"
    td.mkdir(parents=True, exist_ok=True)
    t1 = sorted(json.loads((ROOT / "data" / "m1_cohort.json").read_text())["cohort"])[:n1]
    t2 = sorted(json.loads((ROOT / "data" / "m1_cohort_tier2.json").read_text())["cohort"])[:n2]
    assert t1 and t2 and not (set(t1) & set(t2))
    for sid in t1:
        _table(td, sid, tier1_gap)
    for sid in t2:
        _table(td, sid, tier2_gap)
    (td / "_built.json").write_text(json.dumps(
        {"schema_version": 1,
         "built": {s: date.today().isoformat() for s in t1 + t2}}))
    monkeypatch.setattr(lab, "TABLES", td, raising=True)
    return lab, lab.measure_forcing()


def test_a_minority_tier_failing_is_reported_not_averaged_away(tmp_path, monkeypatch):
    """THE REGRESSION: tier 1 healthy at 4, tier 2 broken at 10."""
    lab, mix = _mix(tmp_path, monkeypatch, tier1_gap=4, tier2_gap=10)
    by = mix["substituted_encoder_days_by_tier"]
    assert by["tier1_trained"] == 4
    assert by["tier2_gate3"] == 10
    assert mix["substituted_encoder_days_worst_tier"] == 10
    # The pooled median is blind here -- that is the whole point.
    assert mix["substituted_encoder_days_median"] == 4, \
        "precondition: the pooled median cannot see the minority tier"
    # And the verdict must follow the worst tier, not the pooled median.
    assert mix["within_what_gate2_measured"] is False, \
        "a pooled pass over a failing subgroup is a false reassurance"


def test_the_warning_names_the_tiers_when_they_disagree(tmp_path, monkeypatch):
    lab, mix = _mix(tmp_path, monkeypatch, tier1_gap=4, tier2_gap=10)
    g1 = json.loads(lab.GATE1.read_text())
    g2 = json.loads(lab.GATE2.read_text())
    w = lab.build(g1, g2, mix).get("extrapolation_warning") or ""
    assert "by tier" in w
    assert "tier2_gate3 10" in w and "tier1_trained 4" in w
    assert "pooled median would understate it" in w
    assert w.lstrip().startswith("10 days"), \
        "the headline number must be the worst tier, not the pooled median"


def test_both_tiers_healthy_stays_silent(tmp_path, monkeypatch):
    lab, mix = _mix(tmp_path, monkeypatch, tier1_gap=4, tier2_gap=4)
    assert mix["substituted_encoder_days_worst_tier"] == 4
    assert mix["within_what_gate2_measured"] is True
    g1 = json.loads(lab.GATE1.read_text())
    g2 = json.loads(lab.GATE2.read_text())
    assert (lab.build(g1, g2, mix).get("extrapolation_warning") or "") == ""


def test_no_tier_spread_is_not_reported_as_a_spread(tmp_path, monkeypatch):
    """When both tiers are equally behind, the message must not imply one tier
    is worse -- a caveat that invents a distinction is as bad as one that hides
    it."""
    lab, mix = _mix(tmp_path, monkeypatch, tier1_gap=7, tier2_gap=7)
    g1 = json.loads(lab.GATE1.read_text())
    g2 = json.loads(lab.GATE2.read_text())
    w = lab.build(g1, g2, mix).get("extrapolation_warning") or ""
    assert "7 days" in w
    assert "by tier" not in w


def test_the_appender_is_told_about_every_served_tier():
    """The repair, not just the detection: the gap existed because the append
    named one basin list."""
    wf = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "--basins data/m1_cohort.json,data/m1_cohort_tier2.json" in wf
    src = (ROOT / "scripts_modern" / "nldas_areal_append.py").read_text()
    assert 'for part in str(a.basins).split(",")' in src, \
        "the appender must accept more than one basin list"


def test_the_archive_covers_every_served_tier():
    """A basin served but not archived can never have its live skill measured,
    which defeats the reason GATE 3 authorised it."""
    src = (ROOT / "scripts" / "build_static_site.py").read_text()
    i = src.index("def _load_m1_cohort")
    body = src[i:i + 1400]
    assert "m1_cohort_tier2.json" in body
