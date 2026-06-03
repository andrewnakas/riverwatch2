"""Unit tests for the AUDIT Phase 1 deploy failure-rate gate
(scripts.build_static_site._failure_gate)."""
import importlib

import pytest

bss = importlib.import_module("scripts.build_static_site")


@pytest.mark.parametrize("n_fail,n_total,max_rate,should_fail", [
    (0, 100, 0.05, False),     # clean shard
    (5, 100, 0.05, False),     # exactly at threshold -> pass (strict >)
    (6, 100, 0.05, True),      # over threshold -> fail
    (1, 1, 0.05, True),        # single-station total failure
    (0, 0, 0.05, True),        # empty shard always fails
    (50, 100, 1.0, False),     # threshold disabled (rate never exceeds 1.0)
])
def test_failure_gate(n_fail, n_total, max_rate, should_fail):
    assert bss._failure_gate(n_fail, n_total, max_rate) is should_fail
