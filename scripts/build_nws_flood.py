"""NWS flood thresholds, in cfs, for every served USGS gauge -> data/nws_flood.json.

The site already shows NWS flood stages in a gauge's panel, fetched by the
browser from NOAA's National Water Prediction Service (NWPS). The MAP cannot do
that for 9,851 gauges, so map colours and alerts knew only percentile classes.
This script does the lookup once, offline, so `current_conditions.py` can mark
a gauge whose forecast reaches action / minor / moderate / major flow.

Per gauge: GET /gauges/{usgs_id}. Thresholds NWS publishes as flow are used
as-is; thresholds published only as stage are converted through the gauge's
NWPS rating (/gauges/{lid}/ratings), the same conversion the panel does
(`rivers.js` NWS.stageToFlow). A stage outside the rating gives no flow.

Output (compact):
  {"generated_at": ..., "fields": ["lid", "action", "minor", "moderate", "major"],
   "sites": {"09380000": ["CLFA3", null, null, null, null], ...}}
Gauges with no NWS point are omitted. A gauge whose lookup FAILS keeps its
previous entry (the file is refreshed in place), so one bad run cannot blank
the map. Run weekly; thresholds change rarely.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://api.water.noaa.gov/nwps/v1/gauges/"
CATS = ("action", "minor", "moderate", "major")
FIELDS = ["lid", *CATS]
_MISSING = object()   # lookup failed (network) -- keep the old entry


def _get(url: str, timeout: float) -> Optional[dict]:
    """JSON, or None for a definite 'no such gauge' (4xx). Raises on network trouble."""
    req = Request(url, headers={"User-Agent": "riverwatch2-build"})
    try:
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code is not None and 400 <= code < 500 and code != 429:
            return None
        raise


def interp(xs: list, ys: list, x: Optional[float]) -> Optional[float]:
    """Linear interpolation inside [xs[0], xs[-1]]; None outside (never extrapolate)."""
    if x is None or len(xs) < 2 or x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return y0 if x1 == x0 else y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return None


def thresholds(gauge: dict, rating: Optional[dict]) -> list:
    """[action, minor, moderate, major] in cfs (None where unknown)."""
    cats = ((gauge.get("flood") or {}).get("categories")) or {}
    rs = []
    if rating:
        rs = sorted((r["stage"], r["flow"]) for r in rating.get("data") or []
                    if isinstance(r.get("stage"), (int, float)) and isinstance(r.get("flow"), (int, float)))
    out = []
    for k in CATS:
        c = cats.get(k) or {}
        flow, stage = c.get("flow"), c.get("stage")
        if isinstance(flow, (int, float)) and flow > 0:
            out.append(round(float(flow), 1))
        elif isinstance(stage, (int, float)) and stage > -900 and rs:
            v = interp([s for s, _ in rs], [f for _, f in rs], float(stage))
            out.append(None if v is None else round(v, 1))
        else:
            out.append(None)
    # A threshold must not sit below a lower category's (bad rating segment).
    last = None
    for i, v in enumerate(out):
        if v is not None and last is not None and v < last:
            out[i] = None
        elif v is not None:
            last = v
    return out


def lookup(site: str, timeout: float):
    """[lid, a, mi, mo, ma], None (no NWS point), or _MISSING (failed)."""
    try:
        g = _get(BASE + site, timeout)
        if not g or not g.get("lid") or str(g.get("usgsId") or "") != site:
            return None
        cats = ((g.get("flood") or {}).get("categories")) or {}
        need_rating = any(
            not (isinstance((cats.get(k) or {}).get("flow"), (int, float)) and cats[k]["flow"] > 0)
            and isinstance((cats.get(k) or {}).get("stage"), (int, float)) and cats[k]["stage"] > -900
            for k in CATS)
        rating = _get(BASE + g["lid"] + "/ratings", timeout) if need_rating else None
        return [g["lid"], *thresholds(g, rating)]
    except Exception:
        return _MISSING


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stations", default=str(ROOT / "data" / "stations_v15.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "nws_flood.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--limit", type=int, default=0, help="first N gauges only (testing)")
    a = ap.parse_args()

    st = json.loads(Path(a.stations).read_text())
    sites = [str(s.get("site_no") or s.get("id")) for s in (st.get("stations", st))]
    if a.limit:
        sites = sites[: a.limit]
    out_p = Path(a.out)
    prev = {}
    if out_p.exists():
        try:
            prev = json.loads(out_p.read_text()).get("sites", {})
        except ValueError:
            prev = {}

    res, failed, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(lookup, s, a.timeout): s for s in sites}
        for i, fu in enumerate(as_completed(futs), 1):
            s, v = futs[fu], fu.result()
            if v is _MISSING:
                failed += 1
                if s in prev:
                    res[s] = prev[s]
            elif v is not None:
                res[s] = v
            if i % 1000 == 0:
                print(f"  {i}/{len(sites)} looked up, {len(res)} NWS points, {failed} failed "
                      f"({time.time() - t0:.0f}s)", flush=True)
    if failed > len(sites) // 2:
        print(f"⛔ {failed}/{len(sites)} lookups failed; leaving {out_p} unchanged")
        return 1
    with_any = sum(1 for v in res.values() if any(x is not None for x in v[1:]))
    out_p.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": FIELDS, "sites": dict(sorted(res.items())),
    }, separators=(",", ":")))
    print(f"-> {out_p}: {len(res)} NWS points, {with_any} with ≥1 flow threshold, "
          f"{failed} lookups failed ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
