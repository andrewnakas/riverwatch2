"""A frozen forcing table must report itself, not look perfect.

On 2026-10-16 every number on the live page was healthy -- 96.3% areal, GATE 2
governing, 95% of the cohort served -- while the published table had not been
refreshed for five days and carried 10 of the member's 16 decoder leads. The
areal FRACTION cannot see this (one row in ~440), the src mix cannot see it,
and neither can a metric that only reads the forcing that is present. The same
blindness as a file gate that iterates glob(): absence is invisible to a survey
of what is there.

So the label measures three things the mix cannot express -- how many encoder
days are substituted, how many decoder leads exist, and how old the refresh is
-- and each has to be able to fire the public warning on its own.
"""
import glob
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


def _load_label():
    p = ROOT / "scripts" / "build_m1_label.py"
    spec = importlib.util.spec_from_file_location("_m1lab", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m1lab"] = mod
    spec.loader.exec_module(mod)
    return mod


def _table(d: Path, sid: str, *, newest_areal_days_before_t0: int, dec_leads: int):
    """Healthy shape by default: areal to today-5, point to t0, 16 leads."""
    today = date.today()
    t0 = today - timedelta(days=1)
    newest_areal = t0 - timedelta(days=newest_areal_days_before_t0)
    lines = [HEADER]
    cur = today - timedelta(days=400)
    while cur <= t0:
        src = "nldas_areal" if cur <= newest_areal else "om_point"
        lines.append(f"{cur},10,15,5,1.0,12.0,900,{src}")
        cur += timedelta(days=1)
    for i in range(dec_leads):
        lines.append(f"{today + timedelta(days=i)},10,15,5,1.0,12.0,900,om_point")
    with gzip.open(d / f"{sid}.csv.gz", "wt", newline="") as fh:
        fh.write("\n".join(lines) + "\n")


def _measure(tmp_path, monkeypatch, *, gap=4, dec_leads=14, built_days_ago=0):
    lab = _load_label()
    monkeypatch.setattr(lab, "TABLES", tmp_path, raising=True)
    _table(tmp_path, "01022500", newest_areal_days_before_t0=gap, dec_leads=dec_leads)
    if built_days_ago is not None:
        (tmp_path / "_built.json").write_text(json.dumps({
            "schema_version": 1,
            "built": {"01022500": (date.today()
                                   - timedelta(days=built_days_ago)).isoformat()},
        }))
    return lab, lab.measure_forcing()


def test_the_healthy_configuration_raises_nothing(tmp_path, monkeypatch):
    """A warning that fires when nothing is wrong trains the reader to ignore
    it, so the steady state GATE 2 measured must be silent: NLDAS publishes at
    today-5, the encoder ends at today-1, so FOUR substituted days is correct."""
    lab, mix = _measure(tmp_path, monkeypatch, gap=4, dec_leads=14, built_days_ago=0)
    assert mix["substituted_encoder_days_median"] == 4
    assert mix["within_what_gate2_measured"] is True
    assert mix["decoder_leads_median"] == 14
    assert mix["decoder_horizon_short"] is False
    assert mix["forcing_age_days_median"] == 0
    assert mix["forcing_stale"] is False


def test_a_short_decoder_horizon_is_measured(tmp_path, monkeypatch):
    """THE LIVE DEFECT: 10 of 16 leads, invisible to every other metric."""
    lab, mix = _measure(tmp_path, monkeypatch, gap=4, dec_leads=10, built_days_ago=0)
    assert mix["decoder_leads_median"] == 10
    assert mix["decoder_horizon_short"] is True
    # ...while the mix it would otherwise be judged by stays healthy.
    assert mix["areal_fraction"] > 0.9
    assert mix["within_what_gate2_measured"] is True


def test_a_stale_refresh_is_measured(tmp_path, monkeypatch):
    lab, mix = _measure(tmp_path, monkeypatch, gap=4, dec_leads=14, built_days_ago=5)
    assert mix["forcing_age_days_median"] == 5
    assert mix["forcing_stale"] is True


def test_a_missing_sidecar_reads_as_stale(tmp_path, monkeypatch):
    lab, mix = _measure(tmp_path, monkeypatch, gap=4, dec_leads=14, built_days_ago=None)
    assert mix["forcing_age_days_median"] is None
    assert mix["forcing_stale"] is True, "unknown age must fail safe"


def _warn(lab, mix):
    """Built from the REAL committed gate verdicts, not a hand-written stub.

    A stub drifts from the artifacts the label actually reads, and then this
    test passes while the live label raises a KeyError -- which is how the
    published label would silently vanish.
    """
    g1 = json.loads(lab.GATE1.read_text())
    g2 = json.loads(lab.GATE2.read_text()) if lab.GATE2.exists() else None
    assert g2, "GATE 2's verdict must be committed for this path to be live"
    return lab.build(g1, g2, mix).get("extrapolation_warning")


def test_each_defect_can_fire_the_public_warning_on_its_own(tmp_path, monkeypatch):
    lab, healthy = _measure(tmp_path, monkeypatch, gap=4, dec_leads=14, built_days_ago=0)
    assert _warn(lab, healthy) is None

    assert "substituted" in (_warn(lab, dict(healthy, **{
        "substituted_encoder_days_median": 7,
        "within_what_gate2_measured": False})) or "")

    short = _warn(lab, dict(healthy, decoder_horizon_short=True,
                            decoder_leads_median=10)) or ""
    assert "TRAINING MEAN" in short and "10" in short

    stale = _warn(lab, dict(healthy, forcing_stale=True,
                            forcing_age_days_median=5)) or ""
    assert "STALE FORECAST" in stale and "5 days ago" in stale


def test_all_three_defects_are_reported_together(tmp_path, monkeypatch):
    """The live state on 2026-10-16 had all three at once. A warning channel
    that reports only the first would have hidden the other two."""
    lab, healthy = _measure(tmp_path, monkeypatch, gap=4, dec_leads=14, built_days_ago=0)
    w = _warn(lab, dict(healthy, **{
        "substituted_encoder_days_median": 7,
        "within_what_gate2_measured": False,
        "decoder_horizon_short": True, "decoder_leads_median": 10,
        "forcing_stale": True, "forcing_age_days_median": 5})) or ""
    assert "substituted" in w and "TRAINING MEAN" in w and "STALE FORECAST" in w
