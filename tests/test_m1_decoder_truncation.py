"""The member must DROP the leads it has no forcing for, not serve them.

`app/mblstm.py`'s `norm_wx` ends in `nan_to_num(arr, nan=0.0)` on z-scores, so a
decoder row the forcing table does not cover becomes the TRAINING MEAN -- a
plausible-looking number, drawn on the page as a 14-day forecast, with no error
and no note anywhere. The strict gate only ever checked leads 1-3, so anything
past lead 3 was unprotected.

Measured live on 2026-10-16: the published table carried 11 of the member's 16
leads, so leads 12-16 on the deployed site were model output driven by
zero-filled inputs. That is a false impression rather than a degraded forecast,
which is why the fix drops those leads instead of flagging them.

A forward LSTM decoder cannot see a later lead's input while producing an
earlier one, so leads 1..n_real are bit-for-bit what they would have been --
truncating costs nothing that was real.

The source-text assertions run torch-free, which is the CI configuration (the
test job installs no torch and `pytest.importorskip` skips on nothing but
ModuleNotFoundError).
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "app" / "mblstm_modern.py").read_text()


def test_the_emit_loop_ranges_over_the_covered_leads_not_the_horizon():
    """THE REGRESSION, pinned as source text because the alternative needs a
    checkpoint, torch, and a 365-day window that CI does not have."""
    assert "for i in range(n_emit):" in SRC, \
        "the row loop must stop at the last covered lead"
    assert "for i in range(horizon):" not in SRC, \
        "emitting the full horizon serves the training mean past the forcing"


def test_coverage_is_counted_contiguously_from_lead_one():
    """A forward decoder consumes leads in order, so the quantity that matters
    is the unbroken run from lead 1 -- not the count of covered leads, which a
    hole in the middle would overstate."""
    assert "n_real = 0" in SRC
    m = re.search(r"for ok in cov_all:\s*\n\s*if not ok:\s*\n\s*break\s*\n\s*n_real \+= 1", SRC)
    assert m, "the run must stop at the first uncovered lead"


def test_the_degradation_has_its_own_channel():
    """Reusing last_skip_reason would make a served-but-short member look like a
    refusal, and the one fact a reader needs -- how many leads are real -- would
    be lost either way."""
    assert "last_degrade_reason" in SRC
    assert "last_degrade_reason: str | None = None" in SRC, "must be declared at module level"
    # Reset at entry: it is a module global and a stale reason otherwise leaks
    # onto the NEXT station, which already happened once with last_skip_reason.
    body = SRC[SRC.index("global last_skip_reason, last_degrade_reason"):]
    assert "last_degrade_reason = None" in body[:600], "must be reset at function entry"


def test_the_hook_reports_the_degradation_separately_from_a_refusal():
    hook = (ROOT / "app" / "forecast.py").read_text()
    assert 'notes.append("mblstm_modern degraded: "' in hook
    assert "mblstm_modern unavailable:" in hook, "the refusal note must still exist"


def test_truncation_is_exercised_on_a_real_window():
    """The behavioural half, skipped wherever torch is absent (i.e. in CI)."""
    torch = pytest.importorskip("torch")
    import numpy as np
    import pandas as pd
    from app import mblstm_modern as mm

    if not mm._try_load():
        pytest.skip(f"checkpoints unloadable: {mm._load_reason}")
    import json
    cohort = json.loads((ROOT / "data" / "m1_cohort.json").read_text())["cohort"]
    sid = str(cohort[0]).zfill(8)

    today = pd.Timestamp.today().normalize()
    last_obs = today - pd.Timedelta(days=1)
    idx = pd.date_range(last_obs - pd.Timedelta(days=500), last_obs, freq="D")
    # NOT constant: q_norm_stats refuses a zero-variance history, which is
    # correct and which made the first version of this test skip silently.
    rng = np.random.default_rng(0)
    q = pd.DataFrame({"date": idx,
                      "q_cfs": 100.0 * np.exp(rng.normal(0, 0.3, len(idx)))})
    cols = {"temperature_2m_mean": 10.0, "temperature_2m_max": 15.0,
            "temperature_2m_min": 5.0, "precipitation_sum": 1.0,
            "shortwave_radiation_sum": 12.0, "vapor_pressure": 900.0}
    hist = pd.DataFrame({"date": idx, **{k: np.full(len(idx), v) for k, v in cols.items()}})

    # The site serves HORIZON_DAYS = 14 leads (pages.yml passes --horizon 14),
    # not the 16 the forcing builder fetches -- those two are slack for a stale
    # gauge. Asking for 16 here would test a configuration nothing serves.
    H = 14
    full_idx = pd.date_range(today, periods=H, freq="D")
    full = pd.DataFrame({"date": full_idx,
                         **{k: np.full(H, v) for k, v in cols.items()}})
    rows_full = mm.forecast(q, hist, full, {"id": sid}, H)
    if not rows_full:
        pytest.skip(f"member refused on the synthetic window: {mm.last_skip_reason}")
    assert len(rows_full) == H
    assert mm.last_degrade_reason is None

    # Eleven leads present -- the live state on 2026-10-16 -- must yield eleven.
    short = full.iloc[:11].copy()
    rows_short = mm.forecast(q, hist, short, {"id": sid}, H)
    assert rows_short is not None and len(rows_short) == 11, \
        f"a table ending at lead 11 must serve 11 leads, not {H}"
    assert mm.last_degrade_reason and "11" in mm.last_degrade_reason

    # No covered lead at all must be a REFUSAL WITH A REASON. Which reason is
    # not pinned: an empty decoder frame is caught by the channel check before
    # the lead count, and with the strict lead-1..3 gate in place the zero-lead
    # branch is in practice unreachable -- it is kept as insurance, not as the
    # path exercised here. What matters is that no refusal is silent, because a
    # bare None is read by the hook as "the member is not enabled".
    assert mm.forecast(q, hist, full.iloc[:0].copy(), {"id": sid}, H) is None
    assert mm.last_skip_reason, "a refusal with no reason is a silent failure"
    # And the leads that WERE covered are untouched by the truncation: a
    # forward decoder cannot see a later input.
    for a, b in zip(rows_full[:11], rows_short):
        assert a["date"] == b["date"]
        assert a["q_cfs"] == pytest.approx(b["q_cfs"], rel=1e-9, abs=1e-9)


def test_no_refusal_path_in_the_member_is_silent():
    """Every `return None` in forecast() must either set a reason or be the one
    documented exception (out of cohort, which is 9,361 of 9,851 gauges by
    design and would be pure noise).

    Five paths were silent until 2026-10-16: the bare `except Exception`, a
    degenerate discharge history, non-finite output, an unreachable zero-lead
    guard, and the out-of-cohort case. A silent refusal is indistinguishable
    from the member being switched off, which is exactly how a member crashing
    on every station would go unnoticed.
    """
    body = SRC[SRC.index("def forecast("):]
    lines = body.splitlines()
    silent = []
    for i, ln in enumerate(lines):
        if ln.strip() != "return None":
            continue
        # Look back over the preceding few lines for an assignment or a comment
        # that marks the documented exception.
        back = "\n".join(lines[max(0, i - 12):i])
        if "last_skip_reason =" in back or "Out of domain" in back:
            continue
        if "_is_enabled()" in back:   # the member is genuinely off; not a skip
            continue
        silent.append(i + 1)
    assert not silent, (
        f"{len(silent)} refusal(s) in forecast() set no reason, at body lines "
        f"{silent}; each is indistinguishable from the member being disabled")
