#!/usr/bin/env python3
"""Push alerts: post to a gauge's public ntfy.sh topic when it STARTS rising.

Anyone can subscribe to a gauge at https://ntfy.sh/rw2-gauge-<usgs id> (the ntfy
phone app, or the web). There is no subscriber list and no backend: the build
compares this run's `current.json` with the previous run's published
`alerts_state.json` and posts only TRANSITIONS into an alert level.

Levels (from current.json, see scripts/current_conditions.py):
    1 rising  5-day forecast peak above normal for its date AND >= 1.5x today's flow
    2 record  5-day forecast peak above the gauge's record for its date

A gauge posts when its level goes UP (or re-enters a level it left more than
CLEAR_AFTER_H ago, or holds it past REPEAT_AFTER_H). A level is held through
dips shorter than CLEAR_AFTER_H, so a forecast wobbling across a threshold
between builds posts once, not every build.

⚠️ Guards, each for a failure that would otherwise spam strangers' phones:
  * no previous state (first run, or the fetch failed) -> record, post nothing;
    otherwise every gauge already rising would read as a new transition;
  * at most MAX_POSTS per run, highest level then largest rise first -- the
    anonymous ntfy.sh tier allows a few hundred messages a day per sender;
  * posting needs RW2_ALERTS=1, so a local or test run cannot post.

This is not a flood warning service, and every message says so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

TOPIC_PREFIX = "rw2-gauge-"
NTFY = "https://ntfy.sh/"
SITE = "https://andrewnakas.github.io/riverwatch2/"
MAX_POSTS = 50
LEVEL_NAMES = {1: "rising", 2: "record"}


def level(row: dict) -> int:
    pc, ratio = row.get("peak_cls") or 0, row.get("ratio")
    if pc >= 5:
        return 2
    if pc >= 4 and ratio is not None and ratio >= 1.5:
        return 1
    return 0


NEAR = ("q", "date", "cls", "ratio")


def _near(r: dict) -> dict:
    """Judge the peak within current_conditions.NEAR_DAYS, not the 14-day peak.

    On 2026-11-12, 1,191 gauges sat at an alert level and 316 of the 327
    "rising" ones peaked on day 6+, where the blend ran wet; 129-155 new
    transitions per build hit the 50-post cap. A current.json without near_*
    (older build) falls back to the 14-day peak."""
    if "near_cls" not in r:
        return r
    return dict(r, **{f"peak_{k}" if k != "ratio" else k: r.get(f"near_{k}") for k in NEAR})


def rows(current: dict) -> dict:
    f = current.get("fields") or []
    return {sid: _near(dict(zip(f, r))) for sid, r in (current.get("sites") or {}).items()}


#: Hysteresis. The first live diff (2026-11-04 02:24Z) found 117 "new"
#: transitions among 641 gauges in ONE 6-hour build: forecasts wobble across
#: the 1.5x / p75 thresholds between builds, and without memory each wobble
#: re-alerts. So a level is HELD until the gauge has been below it for
#: CLEAR_AFTER_H, and the same level cannot re-post within REPEAT_AFTER_H.
CLEAR_AFTER_H = 24
REPEAT_AFTER_H = 72


def _hours(a: str, b: str) -> float:
    return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() / 3600


def transitions(current: dict, prev: dict | None, now: str | None = None) -> tuple[list[dict], dict]:
    """(events to post, new state). `prev` None means no usable history.

    State per gauge: level (held), alerted_at (last post of that level or
    higher), below_since (first build it read below the held level, if any).
    """
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = (prev or {}).get("levels") or {}
    cur = {sid: (level(r), r) for sid, r in rows(current).items()}
    new_levels, events = {}, []
    for sid in set(cur) | set(old):
        lv, r = cur.get(sid, (0, None))
        o = old.get(sid) or {}
        held = int(o.get("level", 0))
        alerted = o.get("alerted_at") or o.get("since")
        if lv >= held and lv > 0:
            fire = prev is not None and (
                lv > held or alerted is None or _hours(alerted, now) >= REPEAT_AFTER_H)
            new_levels[sid] = {"level": lv, "alerted_at": now if fire or alerted is None else alerted}
            if fire and r is not None:
                events.append({"id": sid, "level": lv, **r})
        elif held > 0:
            below = o.get("below_since") or now
            if _hours(below, now) < CLEAR_AFTER_H:
                new_levels[sid] = {"level": held, "alerted_at": alerted, "below_since": below}
    events.sort(key=lambda e: (-e["level"], -(e.get("ratio") or 0)))
    return events, {"generated_at": now, "levels": new_levels}


def unmark(state: dict, prev: dict | None, ids: list[str]) -> dict:
    """Restore the previous entry of each id whose event was NOT posted (over
    MAX_POSTS, or the POST failed). Marking them alerted silenced them for
    REPEAT_AFTER_H without anyone being told -- 79-105 per build in Nov 2026."""
    old = (prev or {}).get("levels") or {}
    for sid in ids:
        if sid in old:
            state["levels"][sid] = old[sid]
        else:
            state["levels"].pop(sid, None)
    return state


def message(e: dict, name: str) -> tuple[str, str]:
    q, pk, pd_ = e.get("q"), e.get("peak_q"), str(e.get("peak_date") or "")
    head = (f"{name}: forecast above its record for {pd_[5:]}" if e["level"] == 2
            else f"{name}: forecast to rise")
    body = (f"Now {q:,.0f} cfs. Forecast peak {pk:,.0f} cfs on {pd_} "
            f"({(e.get('ratio') or 0):.1f}x today). Daily-mean model forecast, "
            "not a warning: check NWS for flood warnings.")
    return head, body


def post(e: dict, name: str, timeout: float = 15.0) -> bool:
    title, body = message(e, name)
    req = urllib.request.Request(
        NTFY + TOPIC_PREFIX + e["id"], data=body.encode(), method="POST",
        headers={"Title": title.encode("utf-8").decode("latin-1", "replace"),
                 "Tags": "ocean" if e["level"] == 1 else "warning",
                 "Priority": "4" if e["level"] == 2 else "3",
                 "Click": f"{SITE}#site={e['id']}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def fetch_prev(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url + f"?t={int(time.time())}", timeout=30) as r:
            d = json.loads(r.read())
        return d if isinstance(d.get("levels"), dict) else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--current", default="dist/current.json")
    ap.add_argument("--stations", default="dist/stations.json")
    ap.add_argument("--prev-url", default=SITE + "alerts_state.json")
    ap.add_argument("--out", default="dist/alerts_state.json")
    a = ap.parse_args()
    current = json.loads(Path(a.current).read_text())
    names = {s["id"]: s.get("name") or s["id"]
             for s in json.loads(Path(a.stations).read_text()).get("stations", [])}
    prev = fetch_prev(a.prev_url)
    events, state = transitions(current, prev)
    Path(a.out).write_text(json.dumps(state, separators=(",", ":")))
    print(f"  alerts: {len(state['levels'])} gauges at a level; "
          f"{'no previous state, posting nothing' if prev is None else f'{len(events)} new transitions'}")
    if os.environ.get("RW2_ALERTS") != "1" or not events:
        return 0
    sent, unsent = 0, [e["id"] for e in events[MAX_POSTS:]]
    for e in events[:MAX_POSTS]:
        ok = post(e, names.get(e["id"], e["id"]))
        sent += ok
        if not ok:
            unsent.append(e["id"])
        time.sleep(1.0)
    if unsent:   # not posted -> not "alerted": leave them to fire next build
        Path(a.out).write_text(json.dumps(unmark(state, prev, unsent), separators=(",", ":")))
    print(f"  alerts: posted {sent}/{min(len(events), MAX_POSTS)}"
          + (f" (capped; {len(events) - MAX_POSTS} not sent)" if len(events) > MAX_POSTS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
