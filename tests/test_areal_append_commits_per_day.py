"""A killed append must keep the days it already fetched.

The appender used to fetch every wanted day into a dict and write all 490 basin
tables once at the end. In CI that step shares a job with the Open-Meteo
refresh, so the realistic end is a job timeout -- and a timeout before the write
discarded every NASA granule the run had pulled.

The compounding part is what makes it serious rather than wasteful: the run
would be cut off at the same point every time, so the "days already areal
everywhere" check at the top would find the same days missing, re-fetch them,
and be cut off again. The areal window would age one day per day forever while
the log showed a step that started and never finished, and the only visible
symptom would be the public label's gap growing -- which is exactly the symptom
that took three days to trace the last time, when the cause was a script
missing from the branch.

So: commit after EACH day, and stop on a budget rather than be killed.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "scripts_modern" / "nldas_areal_append.py").read_text()


def test_the_write_happens_inside_the_day_loop():
    """Source-text, because the alternative needs Earthdata credentials, 24
    NASA granules per day and ~90 seconds per granule -- not a unit test."""
    i_loop = SRC.index("for d in todo:")
    tail = SRC[i_loop:]
    i_commit = tail.index("commit({d.isoformat(): chans})")
    # The commit must come before the loop ends, i.e. before the post-loop
    # summary print.
    i_after = tail.index('if not done_days:')
    assert i_commit < i_after, "the commit must be inside the per-day loop"


def test_no_accumulate_then_write_at_the_end():
    """THE REGRESSION. A dict that collects every day and a single write after
    the loop is the shape that discarded the work."""
    assert "produced = {}" not in SRC, \
        "an accumulate-all-days dict is the pattern that lost the fetches"
    # The per-day commit passes exactly one day.
    assert "commit({d.isoformat(): chans})" in SRC


def test_the_appender_has_its_own_time_budget():
    assert '"--max-seconds"' in SRC
    assert "TIME BUDGET reached after" in SRC
    i = SRC.index("for d in todo:")
    body = SRC[i:i + 900]
    assert "a.max_seconds" in body, \
        "the budget must be checked inside the day loop, not once before it"


def test_the_flip_rule_survives_the_restructure():
    """Areal must replace point and never the reverse -- the mechanism the
    whole two-backend design rests on."""
    assert 'rank = {"om_point": 0, SRC: 1}' in SRC
    assert "FLIP RULE: areal replaces point, never the reverse." in SRC


def test_a_day_that_writes_nothing_is_not_rewritten():
    """`touched` exists so a basin whose rows all lost the flip comparison is
    not rewritten -- with a per-day commit the write loop now runs once per day
    instead of once per run, so a needless rewrite is multiplied."""
    assert "if not touched:" in SRC
    assert "continue" in SRC[SRC.index("if not touched:"):][:40]


def test_a_failed_day_is_reported_and_not_fatal():
    """A day NLDAS has not published yet is normal: the point builder keeps it
    filled and the next run retries. Silence would be the bug."""
    assert "SKIPPED:" in SRC
    i = SRC.index("except Exception as exc:")
    assert "print(" in SRC[i:i + 400]
