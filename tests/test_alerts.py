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
    t = "2026-11-04T00:00:00+00:00"
    prev = {"levels": {"a": {"level": 1, "alerted_at": t}, "b": {"level": 1, "alerted_at": t},
                       "c": {"level": 2, "alerted_at": t}}}
    ev, st = alerts.transitions(cur(a=RISING, b=RECORD, c=RISING, d=RISING, e=FLAT), prev,
                                now="2026-11-04T06:00:00+00:00")
    assert [e["id"] for e in ev] == ["b", "d"]          # record first, then new riser
    assert st["levels"]["a"]["alerted_at"] == t         # unchanged level: no repost
    assert st["levels"]["c"]["level"] == 2              # dip below record is held
    assert "e" not in st["levels"]


def test_wobble_across_threshold_posts_once():
    """2026-11-04: 117 'new' transitions of 641 in one build without hysteresis."""
    h = lambda n: f"2026-11-04T{n:02d}:00:00+00:00"
    _, st = alerts.transitions(cur(a=RISING), {"levels": {}}, now=h(0))
    ev1, st = alerts.transitions(cur(a=FLAT), st, now=h(6))       # dips below
    ev2, st = alerts.transitions(cur(a=RISING), st, now=h(12))    # back up
    assert ev1 == [] and ev2 == []
    assert st["levels"]["a"]["level"] == 1


def test_level_clears_after_24h_below_then_reposts():
    h = lambda d, n: f"2026-11-0{d}T{n:02d}:00:00+00:00"
    _, st = alerts.transitions(cur(a=RISING), {"levels": {}}, now=h(4, 0))
    _, st = alerts.transitions(cur(a=FLAT), st, now=h(4, 6))
    _, st = alerts.transitions(cur(a=FLAT), st, now=h(5, 7))       # 25 h below -> cleared
    assert "a" not in st["levels"]
    ev, _ = alerts.transitions(cur(a=RISING), st, now=h(5, 12))
    assert [e["id"] for e in ev] == ["a"]


def test_held_level_reposts_after_72h():
    t0 = "2026-11-04T00:00:00+00:00"
    ev, _ = alerts.transitions(cur(a=RISING), {"levels": {"a": {"level": 1, "alerted_at": t0}}},
                               now="2026-11-07T01:00:00+00:00")
    assert [e["id"] for e in ev] == ["a"]


def test_message_says_it_is_not_a_warning():
    e = {"id": "1", "level": 1, **dict(zip(F, RISING))}
    title, body = alerts.message(e, "Test River")
    assert "rise" in title and "not a warning" in body
