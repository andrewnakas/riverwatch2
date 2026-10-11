"""The gate verdict scripts must be runnable from a checkout of this branch.

They were not. `analysis/m1_gate1_verdict.py` and `m1_gate2_verdict.py` both
load the record's scoring module by absolute path out of their own directory:

    spec_from_file_location("l51", os.path.join(HERE, "l51_withq_score.py"))

...and `l51_withq_score.py` was tracked only on the research branch, so on the
deploy branch -- the one that carries both verdict scripts AND both verdict
ARTIFACTS -- neither gate could import its scorer. GATE 1 and GATE 2 were
therefore reproducible on one machine and nowhere else, which for a result the
public site now cites is the wrong number of machines.

The scorer additionally did `os.chdir(os.path.expanduser("~/riverwatch2"))` at
import time, which is correct on exactly one host. It now derives the root from
its own location; on that host the two expressions resolve to the same
directory, so the change is a no-op where every number was produced.

These tests do not score anything -- the dumps are large and live elsewhere.
They pin the thing that was actually broken: can the gate reach its scorer.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "analysis" / "l51_withq_score.py"
GATES = ["m1_gate1_verdict.py", "m1_gate2_verdict.py"]


def test_the_scorer_is_present_on_this_branch():
    assert SCORER.exists(), (
        "analysis/l51_withq_score.py is the record's scoring module and both "
        "gate verdict scripts import it by path; without it they cannot run "
        "from a clone")


def test_the_scorer_imports_without_assuming_a_hostname():
    cwd0 = os.getcwd()
    try:
        spec = importlib.util.spec_from_file_location("l51_probe", SCORER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd0)
    for fn in ("check_sidecar", "load_member", "per_basin", "paired_ci"):
        assert callable(getattr(mod, fn, None)), f"{fn} missing from the scorer"


def test_the_scorer_chdirs_to_this_checkout_not_a_fixed_path():
    src = SCORER.read_text()
    assert 'expanduser("~/riverwatch2")' not in src, \
        "a fixed home path makes the gates reproducible on one host only"
    assert "__file__" in src or "RW2_ROOT" in src, \
        "the root must come from the file's location or an explicit override"


@pytest.mark.parametrize("gate", GATES)
def test_each_gate_can_resolve_its_scorer(gate):
    p = ROOT / "analysis" / gate
    if not p.exists():
        pytest.skip(f"{gate} not on this branch")
    src = p.read_text()
    if "l51_withq_score.py" not in src:
        pytest.skip(f"{gate} does not load the scorer by path")
    assert (p.parent / "l51_withq_score.py").exists(), \
        f"{gate} loads the scorer from its own directory, which must contain it"
