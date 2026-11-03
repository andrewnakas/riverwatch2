"""ntfy alerts post only on upward transitions, never on a first run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import alerts  # noqa: E402

F = ["q", "date", "cls", "peak_q", "peak_date", "peak_cls", "ratio"]


def cur(**sites):
    return {"fields": F, "sites": {k: v for k, v in sites.items()}}


RISING = [100, "2026-11-01", 3, 300, "2026-11-05", 4, 3.0]
RECORD = [100, "2026-11-01", 3, 900, "2026-11-05", 5, 9.0]
FLAT = [100, "2026-11-01", 3, 110, "2026-11-05", 3, 1.1]
HIGH_BUT_FLAT = [500, "2026-11-01", 4, 520, "2026-11-05", 4, 1.04]


def test_levels():
    r = alerts.rows(cur(a=RISING, b=RECORD, c=FLAT, d=HIGH_BUT_FLAT))
    assert [alerts.level(r[k]) for k in "abcd"] == [1, 2, 0, 0]


def test_first_run_posts_nothing_but_records_state():
    ev, st = alerts.transitions(cur(a=RISING), None)
    assert ev == [] and st["levels"]["a"]["level"] == 1


def test_only_upward_transitions_post():
    prev = {"levels": {"a": {"level": 1, "since": "x"}, "b": {"level": 1, "since": "x"},
                       "c": {"level": 2, "since": "x"}}}
    ev, st = alerts.transitions(cur(a=RISING, b=RECORD, c=RISING, d=RISING, e=FLAT), prev)
    assert [e["id"] for e in ev] == ["b", "d"]          # record first, then new riser
    assert st["levels"]["a"]["since"] == "x"            # unchanged level keeps its start
    assert "e" not in st["levels"]                      # falling/never clears state


def test_message_says_it_is_not_a_warning():
    e = {"id": "1", "level": 1, **dict(zip(F, RISING))}
    title, body = alerts.message(e, "Test River")
    assert "rise" in title and "not a warning" in body
