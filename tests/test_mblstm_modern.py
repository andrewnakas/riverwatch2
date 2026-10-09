"""Unit tests for the MODERN-1 served member (app/mblstm_modern.py).

Offline and fast: a tiny checkpoint built with the modern recipe's cfg shape, no
network, no real weights. These run inside the pages.yml `test` job, which hard-
gates the deploy, so they must not touch USGS or Open-Meteo.

The behaviours under test are the ones that would otherwise fail SILENTLY:
  * serving a station outside the 490-basin cohort (extrapolation),
  * serving with a forcing channel absent, which norm_wx would turn into "the
    training mean, forever",
  * serving a recipe the assembly does not implement (dec_q / dec_lead / no_q),
  * averaging seeds whose input recipes differ.
"""
import importlib
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import mblstm, mblstm_modern

M1_VARS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
           "precipitation_sum", "shortwave_radiation_sum", "vapor_pressure"]
ADDOR_27 = [
    "p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "elev_mean", "slope_mean", "area_gages2",
    "soil_depth_pelletier", "soil_depth_statsgo", "soil_porosity",
    "soil_conductivity", "max_water_content", "sand_frac", "silt_frac",
    "clay_frac", "frac_forest", "lai_max", "gvf_max", "gvf_diff",
    "root_depth_50", "carbonate_rocks_frac", "geol_permeability",
]
COHORT_ID = "01022500"       # a real id; the cohort set is monkeypatched anyway


def _cfg(**over):
    cfg = {
        "enc_vars": list(M1_VARS), "dec_vars": list(M1_VARS),
        "static_feats": list(ADDOR_27), "quantiles": [0.1, 0.5, 0.9],
        "hidden": 8, "horizon": 14, "context": 365,
        "head": "quantile", "q_transform": "linear",
        "no_q_input": False, "dec_q": False, "enc_lead": None, "dec_lead": None,
        "wx_mean": {c: 0.5 for c in M1_VARS}, "wx_std": {c: 1.0 for c in M1_VARS},
        "static_median": [0.0] * 27, "static_mean": [0.0] * 27, "static_std": [1.0] * 27,
    }
    cfg.update(over)
    return cfg


def _ckpt(tmp_path, name="m1.pt", **over):
    import torch
    cfg = _cfg(**over)
    model = mblstm.build_model(cfg)
    p = tmp_path / name
    torch.save({"state_dict": model.state_dict(), "cfg": cfg}, p)
    return p


def _fresh(monkeypatch, *, enabled=True, ckpts=None, cohort=(COHORT_ID,), point=None):
    mod = importlib.reload(mblstm_modern)
    if enabled:
        monkeypatch.setenv("RW2_ENABLE_MBLSTM_MODERN", "1")
    else:
        monkeypatch.delenv("RW2_ENABLE_MBLSTM_MODERN", raising=False)
    if ckpts is not None:
        monkeypatch.setenv("RW2_MBLSTM_MODERN_CKPT_PATH",
                           ":".join(str(c) for c in ckpts))
    if point is not None:
        monkeypatch.setenv("RW2_MBLSTM_MODERN_POINT", point)
    else:
        monkeypatch.delenv("RW2_MBLSTM_MODERN_POINT", raising=False)
    # Decouple from data/m1_cohort.json and data/camels_attrs.json.
    monkeypatch.setattr(mod.camels_attrs, "cohort", lambda: frozenset(cohort))
    monkeypatch.setattr(mod.camels_attrs, "in_cohort", lambda s: str(s) in set(cohort))
    monkeypatch.setattr(mod.camels_attrs, "enrich_station_attrs",
                        lambda sid, base=None: dict(base or {}))
    return mod


def _inputs(days=420, horizon=14, drop=(), extra=None):
    d0 = date(2024, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(days)]
    rng = np.random.default_rng(0)
    q = 200 + 150 * np.sin(2 * np.pi * np.arange(days) / 365.25)
    q_hist = pd.DataFrame({"date": dates, "q_cfs": np.clip(q, 1, None)})
    cols = [c for c in M1_VARS if c not in drop]
    wx_hist = pd.DataFrame({"date": dates})
    for c in cols:
        wx_hist[c] = rng.random(days) + 0.5
    fdates = [dates[-1] + timedelta(days=i + 1) for i in range(horizon)]
    wx_fcst = pd.DataFrame({"date": fdates})
    for c in cols:
        wx_fcst[c] = rng.random(horizon) + 0.5
    if extra:
        for c, v in extra.items():
            wx_hist[c] = v
            wx_fcst[c] = v
    attrs = {"id": COHORT_ID, "lat": 44.6, "lon": -67.9, "drain_area_sqmi": 227.0}
    return q_hist, wx_hist, wx_fcst, attrs


# --------------------------------------------------------------------- gates

def test_gate_off_returns_none(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, enabled=False, ckpts=[_ckpt(tmp_path)])
    assert mod.forecast(*_inputs(), 14) is None


def test_missing_ckpt_returns_none(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[tmp_path / "nope.pt"])
    assert mod.forecast(*_inputs(), 14) is None


def test_out_of_cohort_refused_without_noise(monkeypatch, tmp_path):
    """9,361 of 9,851 served gauges are out of domain by design, so the skip
    must be silent -- a note on each would be pure noise."""
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)], cohort=("99999999",))
    assert mod.forecast(*_inputs(), 14) is None
    assert mod.last_skip_reason is None


def test_missing_station_id_refused(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)])
    q, wx, wf, _ = _inputs()
    assert mod.forecast(q, wx, wf, {}, 14) is None


def test_short_history_refused_with_reason(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)])
    q, wx, wf, attrs = _inputs()
    assert mod.forecast(q.tail(100), wx, wf, attrs, 14) is None
    assert "365" in (mod.last_skip_reason or "")


# ------------------------------------------------------------------- contract

def test_forecast_contract(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)])
    rows = mod.forecast(*_inputs(), 14)
    assert rows is not None and len(rows) == 14
    for r in rows:
        assert set(r) == {"date", "q_cfs", "q_lo", "q_hi", "q_med"}
        assert all(np.isfinite(r[k]) for k in ("q_cfs", "q_lo", "q_hi", "q_med"))
        assert r["q_lo"] <= r["q_med"] <= r["q_hi"] + 1e-9
        assert r["q_cfs"] >= 0.0


def test_seeds_are_averaged_not_concatenated(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path, "a.pt"), _ckpt(tmp_path, "b.pt")])
    rows = mod.forecast(*_inputs(), 14)
    assert rows is not None and len(rows) == 14
    assert len(mod._models) == 2


def test_seed_cfg_mismatch_refused(monkeypatch, tmp_path):
    """Averaging models that saw different inputs is not an ensemble."""
    a = _ckpt(tmp_path, "a.pt")
    b = _ckpt(tmp_path, "b.pt", hidden=16)
    mod = _fresh(monkeypatch, ckpts=[a, b])
    assert mod.forecast(*_inputs(), 14) is None


# -------------------------------------------------------------------- readout

def test_default_readout_is_mid_of_lo_and_hi(monkeypatch, tmp_path):
    """The record's point prediction is (ylo+yhi)/2, recorded as "readout" in
    benchmarks/m1_FINAL_withq_corrected.json -- NOT the median."""
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)])
    rows = mod.forecast(*_inputs(), 14)
    for r in rows:
        # rel=1e-6, not tighter: the network emits float32, so a float64
        # recomputation from the stored row differs around the 8th significant
        # digit. A tolerance below the STORED precision tests the dtype, not
        # the readout.
        assert r["q_cfs"] == pytest.approx(0.5 * (r["q_lo"] + r["q_hi"]), rel=1e-6)


def test_median_readout_override(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)], point="median")
    rows = mod.forecast(*_inputs(), 14)
    for r in rows:
        assert r["q_cfs"] == pytest.approx(r["q_med"], rel=1e-6)


# ------------------------------------------------- unservable recipes refused

@pytest.mark.parametrize("over,token", [
    ({"dec_q": True}, "dec_q"),
    ({"dec_lead": 1}, "dec_lead"),
    ({"enc_lead": 1}, "enc_lead"),
    ({"no_q_input": True}, "no_q_input"),
    ({"head": "cmal", "cmal_k": 3}, "head"),
    ({"context": 180}, "context"),
])
def test_unservable_cfg_refused(monkeypatch, tmp_path, over, token):
    """Each of these changes the input tensors WITHOUT changing their shape, so
    a width check cannot catch them and a wrong assumption would read as a
    measurement."""
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path, **over)])
    assert mod.forecast(*_inputs(), 14) is None
    assert mod._cfg_is_servable(_cfg(**over)) is not None
    assert token in mod._cfg_is_servable(_cfg(**over))


def test_servable_cfg_accepted():
    assert mblstm_modern._cfg_is_servable(_cfg()) is None


# ------------------------------------------------------- forcing completeness

def test_absent_channel_refused_not_imputed(monkeypatch, tmp_path):
    """norm_wx() would reindex a missing column to NaN then nan_to_num it to 0,
    which in z-space IS the training mean. Refusing is the honest behaviour."""
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)])
    assert mod.forecast(*_inputs(drop=("vapor_pressure",)), 14) is None
    assert "vapor_pressure" in (mod.last_skip_reason or "")


def test_all_nan_channel_counts_as_missing(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)])
    q, wx, wf, attrs = _inputs()
    wf["vapor_pressure"] = np.nan
    assert mod.forecast(q, wx, wf, attrs, 14) is None
    assert "vapor_pressure" in (mod.last_skip_reason or "")


def test_dewpoint_derivation_serves(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)])
    rows = mod.forecast(*_inputs(drop=("vapor_pressure",),
                                 extra={"dew_point_2m_mean": 8.0}), 14)
    assert rows is not None and len(rows) == 14


def test_humidity_derivation_serves(monkeypatch, tmp_path):
    mod = _fresh(monkeypatch, ckpts=[_ckpt(tmp_path)])
    rows = mod.forecast(*_inputs(drop=("vapor_pressure",),
                                 extra={"relative_humidity_2m_mean": 70.0}), 14)
    assert rows is not None and len(rows) == 14


# ------------------------------------------------------------ vapour pressure

@pytest.mark.parametrize("t_c,expected", [(0.0, 611.2), (10.0, 1227.0), (20.0, 2337.0), (30.0, 4243.0)])
def test_saturation_vapour_pressure_pa(t_c, expected):
    """Tetens over water, in Pa -- the unit the CAMELS/NLDAS corpora store."""
    assert mblstm_modern._sat_vp_pa(t_c) == pytest.approx(expected, rel=0.005)


def test_vapour_pressure_from_relative_humidity():
    df = pd.DataFrame({"temperature_2m_mean": [20.0], "relative_humidity_2m_mean": [50.0]})
    out = mblstm_modern._ensure_vapor_pressure(df)
    assert out["vapor_pressure"].iloc[0] == pytest.approx(0.5 * 2337.0, rel=0.005)


def test_vapour_pressure_never_overwrites_native():
    df = pd.DataFrame({"vapor_pressure": [999.0], "dew_point_2m_mean": [10.0]})
    out = mblstm_modern._ensure_vapor_pressure(df)
    assert out["vapor_pressure"].iloc[0] == 999.0


def test_missing_channels_detects_absent_and_dead():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, np.nan]})
    assert mblstm_modern._missing_channels(df, ["a"]) == []
    assert mblstm_modern._missing_channels(df, ["b"]) == ["b"]
    assert mblstm_modern._missing_channels(df, ["c"]) == ["c"]
    assert mblstm_modern._missing_channels(pd.DataFrame(), ["a"]) == ["a"]


# ------------------------------------------------------------ roster wiring

def test_member_is_in_the_canonical_roster():
    """Without this the member can never be reported as dropped, so a silent
    disappearance would look identical to it never being attempted."""
    import re
    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "app" / "forecast.py").read_text()
    m = re.search(r"_MEMBER_ROSTER = \((.*?)\)", src, re.S)
    assert m and "mblstm_modern" in m.group(1)


# ------------------------------------------- regression: the pass-2 code path

def _modern_hook_source(*, code_only: bool = False) -> str:
    """The mblstm_modern block inside app/forecast.py:forecast_station.

    `code_only` drops comment lines, so a check for a bare identifier is not
    satisfied (or tripped) by prose that merely mentions it.
    """
    src = (Path(__file__).resolve().parents[1] / "app" / "forecast.py").read_text()
    i = src.index("from . import mblstm_modern as _mblstm_modern")
    j = src.index("# v13: NOAA National Water Model", i)
    block = src[i:j]
    if code_only:
        block = "\n".join(l for l in block.splitlines() if not l.lstrip().startswith("#"))
    return block


def test_hook_does_not_use_locals_absent_on_the_pass2_path():
    """`forecast_station` has two paths: it either fetches (`start`/`today`
    assigned) or reuses a precomputed `StationInputs` bundle, where **`start` is
    never assigned** and only `today` is. The first version of this hook read
    `start` and raised UnboundLocalError on every cohort station in the real
    build — caught only because the member reports why it skipped.

    `history_days` is a function parameter, so it is safe on both paths.
    """
    hook = _modern_hook_source(code_only=True)
    assert "history_days" in hook, "window must derive from the parameter"
    assert re.search(r"\bm1_start\b", hook), "expected a locally-derived start"
    # The bare name `start` must not be read here.
    assert not re.search(r"[^_\w]start\b(?!\w)", hook.replace("m1_start", "")), \
        "hook references `start`, which is unassigned on the pass-2 path"


def test_hook_reports_why_it_skipped():
    """A handled failure must not be a silent one: the hook has to surface both
    an exception and a refusal, or a dead member looks identical to a disabled
    one."""
    hook = _modern_hook_source()
    assert "last_skip_reason" in hook
    assert "notes.append" in hook
    assert "mblstm_modern failed" in hook


def test_hook_gates_on_cohort_before_fetching():
    """Fetching forcing for all 9,851 served gauges to then discard 9,361 of
    them would be ~19x wasted API traffic per build."""
    hook = _modern_hook_source()
    i_gate = hook.index("in_cohort")
    i_fetch = hook.index("fetch_history")
    assert i_gate < i_fetch, "cohort gate must precede the forcing fetch"
