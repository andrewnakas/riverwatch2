"""Every script the workflow invokes must exist on this branch.

This is the cheapest test in the suite and it would have saved three days.

On 2026-10-13 the areal-append step was added to `pages.yml` calling
`scripts_modern/nldas_areal_append.py`, which lived only on the research
branch. Every scheduled build since then ran:

    python: can't open file '.../scripts_modern/nldas_areal_append.py':
    [Errno 2] No such file or directory

and the step's `|| echo "::warning::areal append failed this run"` turned it
into one line of a log nobody reads, inside a `continue-on-error: true` job.
The pipeline stayed green. The served forcing's areal window aged exactly one
day per day, the public label faithfully reported the gap growing from 4 to 7
substituted encoder days, and the CAUSE was invisible -- I spent that time
checking Earthdata credentials, which were never the problem.

A missing file is a DEPLOY error. It is not a transient fetch failure, and the
two must not produce the same output.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))

#: `python -` and `python - <<EOF` read a heredoc, not a file on disk.
STDIN = {"-"}


def _referenced(text: str) -> set[str]:
    out = set()
    # `python foo/bar.py`, `python3 foo/bar.py`, `$PY foo/bar.py`,
    # `python -m` excluded (module, not path).
    for m in re.finditer(r'(?:python3?|\$PY)\s+(?!-m\b)([^\s<|;&)"\']+\.py)', text):
        tok = m.group(1)
        if tok in STDIN or tok.startswith(("$", "{", "/tmp")):
            continue
        out.add(tok)
    return out


@pytest.mark.parametrize("wf", WORKFLOWS, ids=lambda p: p.name)
def test_every_python_script_the_workflow_runs_is_present(wf):
    missing = sorted(s for s in _referenced(wf.read_text())
                     if not (ROOT / s).exists())
    assert not missing, (
        f"{wf.name} invokes {len(missing)} script(s) absent from this branch: "
        f"{missing}. A workflow that calls a file which is not here fails at "
        f"runtime, often into a swallowed warning, and the symptom shows up "
        f"somewhere else entirely.")


def test_the_test_can_actually_fail():
    """A gate that cannot fail is not a gate. This project has shipped one
    that printed PASS over 48 of 115 artifacts, and a shell loop that reported
    "0 passed, no failures" over 24 files it never ran."""
    found = _referenced("        python scripts_modern/definitely_absent.py --x 1\n")
    assert found == {"scripts_modern/definitely_absent.py"}
    assert not (ROOT / "scripts_modern/definitely_absent.py").exists()


def test_heredoc_and_variable_invocations_are_not_mistaken_for_paths():
    assert _referenced("python - <<'PY'\nprint(1)\nPY\n") == set()
    assert _referenced("$PY $SOME_VAR.py\n") == set()


def test_the_areal_append_scripts_are_present_by_name():
    """Named explicitly, because these two are the pair that was missing and
    the append is the step that moves the served configuration out of GATE 1's
    -0.059534 regime into GATE 2's -0.004018."""
    for f in ("scripts_modern/nldas_areal_append.py",
              "scripts_modern/build_nldas2_areal.py"):
        assert (ROOT / f).exists(), f"{f} is required by the m1-forcing job"


def test_a_missing_script_is_an_error_not_a_warning():
    """The workflow must distinguish a deploy error from a fetch failure."""
    wf = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    i = wf.index("Append NLDAS-2 areal forcing")
    step = wf[i:i + 4000]
    assert "::error::" in step, "a missing script must raise ::error::, not ::warning::"
    assert "is missing from this branch" in step
