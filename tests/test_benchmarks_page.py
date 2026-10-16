"""Unit tests for scripts/build_benchmarks_page.py.

This page makes public claims about model skill, so the tests are mostly about
claims rather than markup: that the operational number leads, that no headline
appears without its forcing regime, that the retraction trail is rendered in
full rather than summarised away, and that the test window is not mislabelled.
"""
import importlib
import json
import re
from pathlib import Path

import pytest

bp = importlib.import_module("scripts.build_benchmarks_page")

ROOT = Path(__file__).resolve().parents[1]

MIN_ARTIFACTS = {
    "withq": {
        "headline": 0.9058291860163704, "basins": 490, "rows": 1243864,
        "composition": ["fused4md49fix3", "fused4m9fix3", "nldasm9"],
        "protocol": "modern-v3-9yr", "protocol_window": ["2005-10-01", "2014-09-30"],
        "readout": "(ylo+yhi)/2", "frozen_utc": "2026-10-08T03:10:45.821321+00:00",
        "git_commit": "dfa93050c5ecf9c96c3bd2b975769e84141bc936",
        "ship_bar": {"a": [501, 502, 503, 504, 505], "b": [501, 502, 503, 504, 505]},
    },
    "noq": {"headline": 0.834557542823926, "composition": ["lstm_fused4md4fix3"]},
    "decay": {
        "paired_delta": -0.048033307111507106,
        "ci95": [-0.05980848008244488, -0.039718437289156805],
        "rows": 56224, "basins_scored": 489,
        "frame_caveat": "sparse: archive starts 2021-05, 14-day stride",
    },
    "withq_leads": {"by_lead": {str(i): {"ensemble": 0.9 - i * 0.004} for i in range(1, 15)},
                    "decay": -0.055669, "crosscheck": {"difference": 6.459e-10}},
    "noq_leads": {"by_lead": {str(i): {"ensemble": 0.83 + i * 0.001} for i in range(1, 15)},
                  "decay": 0.011036, "crosscheck": {"difference": 3.57e-09}},
    "test_reads": [],
}


# ------------------------------------------------------- the test-window pin

def test_test_window_pin_matches_the_launcher():
    """The frozen artifacts record the TRAINING window as `protocol_window` and
    carry no test window, so the page pins it. If the launcher ever changes, this
    fails rather than letting a training window be published as a test window."""
    sh = (ROOT / "gpu1080" / "queue_m1h_withq.sh").read_text()
    # Both assignments sit on one line: `TS=2018-10-01;  TE=2025-09-30`.
    ts = re.search(r"\bTS=(\d{4}-\d{2}-\d{2})", sh)
    te = re.search(r"\bTE=(\d{4}-\d{2}-\d{2})", sh)
    assert ts and te, "could not read TS=/TE= from queue_m1h_withq.sh"
    assert bp.TEST_WINDOWS["modern-v3-9yr"] == (ts.group(1), te.group(1))


def test_training_window_is_not_published_as_the_test_window():
    html = bp.build_html(MIN_ARTIFACTS)
    i = html.find("held-out test window")
    assert i > 0
    window_sentence = html[i:i + 260]
    assert "2018-10-01" in window_sentence
    # The training window may appear, but only labelled as such.
    assert "trained on" in window_sentence


def test_unknown_protocol_yields_no_window_claim():
    a = dict(MIN_ARTIFACTS, withq=dict(MIN_ARTIFACTS["withq"], protocol="something-else"))
    assert bp.test_window("something-else") is None
    assert "2018-10-01" not in bp.build_html(a)


# ------------------------------------------------------------ the claims

def test_operational_number_leads_the_page():
    """The perfect-forcing headline must not be the first number a reader sees."""
    html = bp.build_html(MIN_ARTIFACTS)
    body = html[html.find("<body"):]
    i_op = body.find("0.858")
    i_perfect = body.find("0.905829")
    assert i_op > 0, "the operational ~0.858 is missing"
    assert i_op < i_perfect, "perfect-forcing skill appears before the operational number"


def test_operational_number_is_derived_not_hardcoded():
    a = dict(MIN_ARTIFACTS)
    a["decay"] = dict(a["decay"], paired_delta=-0.10)
    html = bp.build_html(a)
    assert "0.806" in html          # 0.905829 - 0.10
    assert "0.858" not in html


def test_every_forbidden_claim_is_stated():
    html = bp.build_html(MIN_ARTIFACTS)
    for title, body in bp._FORBIDDEN_CLAIMS:
        assert title in html, f"missing disclaimer: {title}"
        assert body[:40] in html


def test_perfect_forcing_is_labelled_wherever_it_appears():
    html = bp.build_html(MIN_ARTIFACTS)
    assert "perfect-forcing" in html or "perfect forcing" in html
    # The claim that it is NOT operational must be present verbatim.
    assert "not operational skill" in html


def test_nearing_reference_is_not_presented_as_a_win():
    html = bp.build_html(MIN_ARTIFACTS)
    assert "0.879" in html                       # named, as context
    for phrase in ("beats the record", "beats Nearing", "new record", "state of the art"):
        assert phrase.lower() not in html.lower()


def test_no_trimmed_mean_is_quoted():
    html = bp.build_html(MIN_ARTIFACTS)
    assert "0.904644" not in html                # the degenerate median-of-3
    assert "trimmed mean is undefined" in html


def test_opposite_lead_decay_is_reported_as_unexplained():
    html = bp.build_html(MIN_ARTIFACTS)
    assert "unexplained" in html.lower()
    assert "opposite directions" in html


# --------------------------------------------------------- the retraction trail

def test_parse_test_reads_structures_rows():
    txt = (
        "2026-10-07T04:38:01+00:00 arm=withq frame=test1 members=a,b headline=0.910438 is_record=True\n"
        "2026-10-08T03:08:44+00:00 RETRACTION arm=withq target=x.json headline=0.910438 "
        "reason=UNENTITLED_FORCING_AT_LEAD_1\n"
        "2026-10-08T03:08:44+00:00 WITHDRAWN artifact=y.json reason=KNOWN_POSITIVE_CONTROL\n"
    )
    rows = bp.parse_test_reads(txt)
    assert [r["kind"] for r in rows] == ["read", "retraction", "withdrawn"]
    assert rows[0]["headline"] == "0.910438"
    assert rows[1]["reason"] == "UNENTITLED_FORCING_AT_LEAD_1"


def test_parse_test_reads_keeps_unparseable_lines():
    rows = bp.parse_test_reads("2026-01-01 some free text with no pairs\n")
    assert len(rows) == 1 and "raw" in rows[0]


def test_parse_test_reads_ignores_blank_lines():
    assert bp.parse_test_reads("\n\n  \n") == []


def test_retractions_are_rendered_not_summarised():
    a = dict(MIN_ARTIFACTS, test_reads=bp.parse_test_reads(
        (ROOT / "benchmarks" / "m1_test_reads.log").read_text()))
    html = bp.build_html(a)
    assert html.count("RETRACTED") >= 2, "both retractions must appear"
    assert "WITHDRAWN" in html
    assert "0.910438" in html, "the retracted headline must be shown, not hidden"


# ------------------------------------------------------------------- hygiene

def test_real_artifacts_build_without_holes():
    a = bp.load_artifacts()
    if "withq" not in a:
        pytest.skip("frozen artifacts not present in this checkout")
    html = bp.build_html(a)
    assert "None" not in html and "nan" not in html.lower().replace("nan=", "")
    assert len(html) > 6000


def test_missing_optional_artifact_omits_its_section():
    a = {k: v for k, v in MIN_ARTIFACTS.items() if k != "noq_leads"}
    html = bp.build_html(a)
    assert "<h1>" in html          # still builds


def test_seed_count_reports_disagreement_rather_than_the_first_member():
    assert bp._seed_count({"ship_bar": {"a": [1, 2, 3], "b": [1, 2, 3]}}) == "3"
    out = bp._seed_count({"ship_bar": {"a": [1, 2, 3], "b": [1, 2]}})
    assert "varying" in out and "2" in out and "3" in out


def test_html_escapes_artifact_strings():
    a = dict(MIN_ARTIFACTS,
             withq=dict(MIN_ARTIFACTS["withq"], composition=["<script>bad()</script>"]))
    assert "<script>bad()" not in bp.build_html(a)


def test_emit_requires_the_load_bearing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "load_artifacts", lambda *_, **__: {"noq": {}})
    with pytest.raises(RuntimeError, match="load-bearing"):
        bp.emit(tmp_path)


# ----------------------------------------------- the ungauged-transfer section

TRANSFER = {
    "overall": {"n_pairs": 322, "median_nse": 0.7044, "mean_nse": -2.1601,
                "frac_above_0": 0.752, "frac_above_0.5": 0.606},
    "window": ["2021-01-01", "2025-12-31"],
    "bins": [
        {"label": "1.25x", "n_pairs": 73, "median_nse": 0.8201, "frac_above_0.5": 0.753,
         "max_abs_log_ratio": 0.2231},
        {"label": "2x", "n_pairs": 156, "median_nse": 0.8215, "frac_above_0.5": 0.744,
         "max_abs_log_ratio": 0.6931},
    ],
    "by_direction": {"upstream": {"n_pairs": 141, "median_nse": 0.6880},
                     "downstream": {"n_pairs": 181, "median_nse": 0.7459}},
}


def test_transfer_section_states_what_it_is_not():
    html = bp.build_html(dict(MIN_ARTIFACTS, transfer=TRANSFER))
    assert "Clicking a point with no gauge" in html
    assert "no model run at that point" in html
    assert "mm/day" in html
    assert "neighbour-transfer" in html


def test_transfer_section_shows_the_decay_not_just_the_best_bin():
    html = bp.build_html(dict(MIN_ARTIFACTS, transfer=TRANSFER))
    for b in TRANSFER["bins"]:
        assert f"{b['median_nse']:.4f}" in html, f"bin {b['label']} missing"


def test_transfer_section_reports_the_bad_tail():
    """The mean is far below the median; hiding that would flatter the method."""
    html = bp.build_html(dict(MIN_ARTIFACTS, transfer=TRANSFER))
    assert "-2.160" in html or "−2.160" in html


def test_transfer_section_records_the_refuted_heuristic():
    html = bp.build_html(dict(MIN_ARTIFACTS, transfer=TRANSFER))
    assert "refuted a heuristic" in html
    assert "unpaired" in html


def test_transfer_section_absent_when_artifact_missing():
    html = bp.build_html(MIN_ARTIFACTS)
    assert "Clicking a point with no gauge" not in html
    assert "<h1>" in html


def test_shipped_skill_file_matches_what_the_ui_reads():
    """app/static/river_click.js:skillLabel() reads bins[].{max_abs_log_ratio,
    median_nse,n_pairs,label}. A shape change here would silently blank the
    label in the panel."""
    p = ROOT / "data" / "river_transfer_skill.json"
    if not p.exists():
        pytest.skip("transfer skill not measured in this checkout")
    d = json.loads(p.read_text())
    assert d.get("bins"), "no bins"
    for b in d["bins"]:
        for k in ("max_abs_log_ratio", "median_nse", "n_pairs", "label"):
            assert k in b, f"bin missing {k}"
        assert -1.0 <= b["median_nse"] <= 1.0
    # The browser downloads this on every click; the per-pair rows belong in the
    # provenance artifact, not here.
    assert "pairs" not in d, "per-pair rows must stay in benchmarks/river_transfer_study.json"
    assert p.stat().st_size < 20000


# ---------------------------------------------------------------------------
# The served member runs on forcing it was not trained on, and the page has to
# say so. Before this section existed the strings "substitut", "Open-Meteo"
# and "NLDAS-2" appeared ZERO times -- which was fine only while the member
# was dark, and became a false impression the moment it was switched on.
# ---------------------------------------------------------------------------

GATE1_FIXTURE = {
    "member": "nldasm9",
    "seeds": [501, 502, 503, 504, 505],
    "frame": "val1",
    "basins": 150,
    "rows_joined": 217153,
    "protocol": "modern-v3-9yr",
    "median_nse_areal_trained_on": 0.8936953283991611,
    "median_nse_point_as_served": 0.8086593898958421,
    "paired_delta_point_minus_areal": -0.0595344589878411,
    "ci95": [-0.0839106068765077, -0.03835835590277631],
    "breadth": 0.08666666666666667,
}


def _html_with_gate1():
    a = dict(MIN_ARTIFACTS)
    a["gate1"] = GATE1_FIXTURE
    return bp.build_html(a)


def test_gate1_section_names_both_forcings_and_the_cost():
    html = _html_with_gate1()
    assert "NLDAS-2" in html
    assert "Open-Meteo" in html
    assert "0.059534" in html, "the measured substitution cost must be stated"
    assert "150" in html and "val1" in html, "the frame must travel with the number"


def test_gate1_section_absent_when_artifact_missing():
    """Mirrors the transfer/sampling contract: a research artifact that is not
    present omits its section rather than failing the build, because this page
    ships from inside the site build and must never block a deploy."""
    html = bp.build_html(dict(MIN_ARTIFACTS))
    assert "What the served model runs on" not in html


def test_the_as_served_level_is_never_presented_as_the_members_skill():
    """0.808659 is val1 on 150 basins; the record is the test window on 490.
    Quoting it bare would swap a validation figure for a test one, so wherever
    it appears the sentence has to carry its frame."""
    html = _html_with_gate1()
    i = html.find("0.808")
    assert i != -1, "the as-served arm is reported"
    window = html[max(0, i - 400): i + 400]
    assert "150" in window and "490" in window, \
        "the as-served level must be qualified by its frame in the same breath"


def test_the_two_discounts_are_never_summed():
    """-0.059534 and -0.048033 were measured on different frames against
    different baselines. Their additivity is unmeasured, so a page that shows
    their sum -- or a headline derived from it -- is making an untested claim."""
    html = _html_with_gate1()
    for forbidden in ("0.1076", "0.10757", "0.798", "0.7975"):
        assert forbidden not in html, f"page shows the composed figure {forbidden}"
    assert "additivity" in html, "the page must say why they cannot be added"


def test_gate1_numbers_are_derived_not_hardcoded():
    """Perturb the artifact; the page must move with it."""
    a = dict(MIN_ARTIFACTS)
    a["gate1"] = dict(GATE1_FIXTURE, paired_delta_point_minus_areal=-0.0123456)
    html = bp.build_html(a)
    assert "0.012346" in html or "0.012345" in html


def test_no_forbidden_claim_title_contains_an_escapable_character():
    """`test_every_forbidden_claim_is_stated` compares the RAW title against the
    rendered HTML, so a title containing `'`, `&`, `<` or `>` is escaped on the
    page and the assertion fails for a reason that looks like a missing
    disclaimer. That cost a debugging cycle; this makes the real cause explicit
    at the point the claim is written."""
    for title, body in bp._FORBIDDEN_CLAIMS:
        for ch in ("'", "&", "<", ">", '"'):
            assert ch not in title, (
                f"claim title contains {ch!r}, which _esc() will escape: {title!r}")


#: Mirrors benchmarks/m1_gate2_verdict.json. A literal rather than a read of
#: the file, matching GATE1_FIXTURE above: these tests check that the PAGE
#: renders a configuration, and reading the live artifact would make them pass
#: or fail on whatever is currently committed.
GATE2_FIXTURE = {
    "frame": "val1", "member": "nldasm9", "basins": 150,
    "seeds": [501, 502, 503, 504, 505], "protocol": "modern-v3-9yr",
    "splice_recent_days": 4,
    "median_nse_areal_control": 0.8936953283991611,
    "median_nse_spliced_treatment": 0.8575537523189201,
    "paired_delta_spliced_minus_areal": -0.004018177634188769,
    "ci95": [-0.007396350167453103, -0.0019625893495204827],
    "breadth": 0.2866666666666667,
    "difference_of_medians": -0.036141576080241045,
    "fraction_of_gate1_loss_recovered_by_areal": 0.9325061706892067,
}


def _html_with_label(lbl):
    """The served-configuration block needs gate1, gate2 and the label: the
    page follows the LIVE gate, so a fixture that omits one describes a
    different configuration than the one under test."""
    a = dict(MIN_ARTIFACTS)
    a["gate1"] = GATE1_FIXTURE
    a["gate2"] = GATE2_FIXTURE
    a["served_label"] = lbl
    return bp.build_html(a)


def test_the_page_states_the_horizon_the_member_actually_publishes():
    """A missing forcing row becomes the TRAINING MEAN in `norm_wx`, not a
    refusal, so a short forcing table would otherwise put a plausible number on
    the page as a 14-day forecast. The member truncates instead, and the page
    has to say so -- and has to say it from the LABEL, so the sentence follows
    the configuration rather than describing one that was true once.

    Measured live 2026-10-16: 11 of 14 leads, while every other number on the
    page read healthy.
    """
    lbl = {
        "governing_gate": "GATE 2",
        "forcing_observed": {
            "areal_fraction": 0.958,
            "decoder_leads_median": 11,
            "decoder_leads_expected": 14,
            "substituted_encoder_days_median": 7,
            "gate2_measured_substituted_days": 4,
        },
        "extrapolation_warning": "the table carries only 11 of the 14 decoder leads",
    }
    html = _html_with_label(lbl)
    assert "only the lead days it has" in html
    assert "training mean" in html.lower()
    assert ">11<" in html and ">14<" in html, "the live lead counts must appear"
    # The encoder gap, stated against what the gate measured.
    assert ">7<" in html and ">4<" in html
    assert "extrapolation here" in html
    assert "Live self-report" in html


def test_the_healthy_configuration_does_not_claim_an_extrapolation():
    """A caveat that fires when nothing is wrong teaches the reader to skip it."""
    lbl = {
        "governing_gate": "GATE 2",
        "forcing_observed": {
            "areal_fraction": 0.962,
            "decoder_leads_median": 14,
            "decoder_leads_expected": 14,
            "substituted_encoder_days_median": 4,
            "gate2_measured_substituted_days": 4,
        },
        "extrapolation_warning": None,
    }
    html = _html_with_label(lbl)
    assert "which is the configuration it measured" in html
    assert "extrapolation here" not in html
    assert "Live self-report" not in html
    assert "so the remaining leads are not shown" not in html
