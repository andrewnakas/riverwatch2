"""Unit tests for the AUDIT Phase 1 deploy failure-rate gate
(scripts.build_static_site._failure_gate)."""
import importlib

import pytest

bss = importlib.import_module("scripts.build_static_site")


@pytest.mark.parametrize("n_fail,n_total,max_rate,should_fail", [
    (0, 100, 0.25, False),     # clean shard
    (25, 100, 0.25, False),    # exactly at threshold -> pass (strict >)
    (26, 100, 0.25, True),     # over threshold -> fail
    (1, 1, 0.25, True),        # single-station total failure
    (0, 0, 0.25, True),        # empty shard always fails
    (50, 100, 1.0, False),     # threshold disabled (rate never exceeds 1.0)
])
def test_failure_gate(n_fail, n_total, max_rate, should_fail):
    assert bss._failure_gate(n_fail, n_total, max_rate) is should_fail


def _no_data(sid):
    return {"station_id": sid, "error": f"no USGS daily discharge for {sid}"}


def _real(sid, msg="boom"):
    return {"station_id": sid, "error": msg}


def test_classify_separates_no_data_from_real():
    failures = [_no_data("a"), _real("b"), _no_data("c"), _real("d", "ValueError: x")]
    no_data, real = bss._classify_failures(failures)
    assert {f["station_id"] for f in no_data} == {"a", "c"}
    assert {f["station_id"] for f in real} == {"b", "d"}


def test_classify_empty():
    assert bss._classify_failures([]) == ([], [])


def test_shard10_scenario_passes_gate():
    # The real shard-10 regression: 19 failures, ALL "no USGS daily discharge",
    # out of 154 stations -> 0 real failures over 135 forecastable -> must NOT
    # fail the gate (these gauges simply have no data to forecast).
    failures = [_no_data(f"sid{i}") for i in range(19)]
    no_data, real = bss._classify_failures(failures)
    n_total = 154
    n_with_data = n_total - len(no_data)
    assert len(real) == 0
    assert n_with_data == 135
    # n_with_data > 0 and 0/135 == 0% <= 25% -> gate does not fire.
    assert bss._failure_gate(len(real), n_with_data, 0.25) is False


def test_real_mass_failure_still_trips_gate():
    # A genuine regression: half the forecastable gauges crash -> must fail.
    failures = [_no_data(f"n{i}") for i in range(10)] + [_real(f"r{i}") for i in range(60)]
    no_data, real = bss._classify_failures(failures)
    n_with_data = 120 - len(no_data)  # 110 forecastable, 60 crashed
    assert bss._failure_gate(len(real), n_with_data, 0.25) is True
