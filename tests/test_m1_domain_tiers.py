"""GATE 3 widened the served domain from 490 basins to 620, in two tiers.

The two tiers have DIFFERENT measured skill -- val1 median NSE 0.876335 for the
490 the member was trained on, 0.854214 [0.838676, 0.895274] for the 130 GATE 3
authorised -- and GATE 3's decision table requires the out-of-cohort basins be
marked as a distinct tier and quoted their own number, "never the cohort one".

So the one thing that must not happen is the two becoming indistinguishable.
That is why `in_cohort` was NOT widened: it still means "trained on", which is
what decides which figure applies, and widening it would have silently
re-pointed every caller that picks a number. "May the member run here" is a
separate question with a separate function.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ca():
    import sys
    sys.path.insert(0, str(ROOT))
    from app import camels_attrs
    return camels_attrs


def test_the_two_tiers_are_disjoint_and_sum_to_620():
    ca = _ca()
    t1, t2 = ca.cohort(), ca.tier2()
    assert len(t1) == 490, f"tier 1 is the training cohort: {len(t1)}"
    assert len(t2) == 130, f"tier 2 is GATE 3's scoreable set: {len(t2)}"
    assert not (t1 & t2), "a basin cannot be both trained on and out of cohort"
    assert len(t1 | t2) == 620


def test_in_cohort_still_means_trained_on():
    """THE REGRESSION GUARD. If this ever returns True for a tier-2 basin, every
    caller that uses it to choose a measured number starts quoting 0.876335 for
    a basin measured at 0.854214."""
    ca = _ca()
    for sid in sorted(ca.tier2())[:5]:
        assert not ca.in_cohort(sid), \
            f"{sid} is tier 2 and must not read as trained-on"
        assert ca.in_served_domain(sid), f"{sid} is tier 2 and must be servable"
        assert ca.domain_tier(sid) == 2


def test_the_unscoreable_basins_are_not_served():
    """GATE 3 can say nothing about them in either direction -- they are gauged
    recently but barely at all during the val window -- so its verdict does not
    license them."""
    ca = _ca()
    cov = json.loads((ROOT / "benchmarks" / "m1_gate3_coverage.json").read_text())
    for sid in cov["groups"]["unscoreable"]["ids"]:
        assert ca.domain_tier(sid) == 0, f"{sid} is unscoreable and must be out of domain"
        assert not ca.in_served_domain(sid)


def test_the_partial_coverage_basins_are_not_served_either():
    """Measured separately and deliberately excluded: 0.912948 and 0.564878 on
    1,113 and 1,092 observed days. One is well above the tier-2 median and one
    well below any sensible bar, and n=2 generalises to nothing."""
    ca = _ca()
    cov = json.loads((ROOT / "benchmarks" / "m1_gate3_coverage.json").read_text())
    for sid in cov["groups"]["partial"]["ids"]:
        assert ca.domain_tier(sid) == 0, \
            f"{sid} has partial val coverage; serving it is a separate decision"


def test_tier2_is_exactly_what_gate3_scored():
    """The served set and the measured set cannot be allowed to drift. The
    artifact is derived from the coverage file rather than hand-listed, and this
    pins that they still agree."""
    t2 = json.loads((ROOT / "data" / "m1_cohort_tier2.json").read_text())
    cov = json.loads((ROOT / "benchmarks" / "m1_gate3_coverage.json").read_text())
    verdict = json.loads((ROOT / "benchmarks" / "m1_gate3_verdict.json").read_text())
    assert t2["cohort"] == cov["groups"]["scoreable_full"]["ids"]
    assert len(t2["cohort"]) == verdict["out_of_cohort"]["basins"] == 130
    assert t2["measured"]["median_nse"] == verdict["out_of_cohort"]["median_nse"]


def test_a_missing_tier2_file_serves_fewer_basins_not_more():
    """The fallback on any failure must be to serve LESS, never to serve a
    basin whose skill was never measured."""
    ca = _ca()
    src = (ROOT / "app" / "camels_attrs.py").read_text()
    i = src.index("def tier2(")
    body = src[i:i + 900]
    assert "except Exception:" in body and "frozenset()" in body, \
        "tier2() must fall back to empty, as cohort() does"


def test_the_hook_marks_the_tier_on_the_served_forecast():
    """GATE 3: "The UI must mark these basins as a distinct tier." A tier that
    lives only in a cohort file nobody ships cannot be marked."""
    src = (ROOT / "app" / "forecast.py").read_text()
    assert "member_labels: Dict[str, str] = field(default_factory=dict)" in src
    assert 'member_labels=dict(member_labels),' in src
    assert '"out_of_cohort_gate3"' in src
    assert "0.854214" in src, "the tier-2 basin's own median must travel with it"
    # rindex, not index: the first occurrence is the dataclass's own comment
    # describing the field. The note that reaches a reader is at the assignment.
    i = src.rindex('member_labels["mblstm_modern"] = "out_of_cohort_gate3"')
    note = src[i:i + 900]
    assert "never the latter" in note, \
        "the note must say which number NOT to quote, per the decision table"


def test_the_forcing_builder_refreshes_everything_it_serves():
    """Its own warning: "widening the cohort REQUIRES widening this script's
    basin list in the same change." A served basin whose forcing is never
    refreshed serves a frozen table."""
    src = (ROOT / "scripts" / "build_m1_forcing.py").read_text()
    assert '"--tier2"' in src
    assert 'default="data/m1_cohort_tier2.json"' in src
    i = src.index("cohort = load_cohort(Path(a.cohort))")
    body = src[i:i + 1400]
    assert "tier2" in body and "load_cohort(t2)" in body, \
        "tier 2 must be unioned into the refresh list"
