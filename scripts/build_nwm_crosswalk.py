#!/usr/bin/env python3
"""Build USGS gauge → NWM reachId (COMID) crosswalk via USGS NLDI.

For each station in stations_40_enriched.json, look up the NHDPlus COMID
via the USGS Network Linked Data Index (NLDI). The NWM uses COMID as its
primary identifier (`feature_id` / `reachId`), so this is the same key used
by NOAA's NWPS API.

Output: data/nwm_crosswalk.json with shape:
  {
    "<usgs_id>": {"comid": "<comid>", "lat": ..., "lon": ...},
    ...
  }

Stations without NLDI matches (rare — typically very small or non-NHD-flagged
sites) get null entries. Skipped entries are retried on subsequent runs.
Resume-safe: rewrites the whole file each run from cached + new lookups.

Run-time: ~1s per gauge × 1893 gauges = ~30 min cold; <5s warm.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ROOT / "data" / "stations_40_enriched.json"
OUT = ROOT / "data" / "nwm_crosswalk.json"

NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{usgs_id}"
TIMEOUT = 15


def _fetch(usgs_id: str) -> dict | None:
    url = NLDI.format(usgs_id=quote(usgs_id))
    try:
        req = Request(url, headers={"User-Agent": "riverwatch2/0.1"})
        with urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc)[:80]}
    feats = data.get("features") or []
    if not feats:
        return None
    props = (feats[0].get("properties") or {})
    coords = (feats[0].get("geometry") or {}).get("coordinates") or [None, None]
    comid = props.get("comid")
    if not comid:
        return None
    return {
        "comid": str(comid),
        "lon": coords[0],
        "lat": coords[1],
        "name": props.get("name"),
    }


def main() -> int:
    stations = json.loads(STATIONS.read_text())["stations"]
    existing: dict = {}
    if OUT.exists():
        try:
            existing = json.loads(OUT.read_text())
        except Exception:
            existing = {}

    todo = [s for s in stations if s["site_no"] not in existing or "comid" not in (existing.get(s["site_no"]) or {})]
    print(f"crosswalk: {len(stations)} stations, {len(existing)} cached, {len(todo)} to fetch")

    if not todo:
        print("all stations already cached")
        return 0

    crosswalk = dict(existing)
    started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch, s["site_no"]): s for s in todo}
        for i, f in enumerate(as_completed(futures), 1):
            s = futures[f]
            res = f.result()
            crosswalk[s["site_no"]] = res
            if i % 50 == 0 or i == len(todo):
                elapsed = time.time() - started
                rate = i / max(elapsed, 0.01)
                eta = (len(todo) - i) / max(rate, 0.01)
                hits = sum(1 for v in crosswalk.values() if v and "comid" in v)
                print(f"  {i}/{len(todo)} ({rate:.1f}/s, eta {eta:.0f}s) | total hits: {hits}")
                # Write incrementally so a crash doesn't lose progress.
                OUT.write_text(json.dumps(crosswalk, indent=2, sort_keys=True))

    OUT.write_text(json.dumps(crosswalk, indent=2, sort_keys=True))
    final_hits = sum(1 for v in crosswalk.values() if v and "comid" in v)
    print(f"done: {final_hits}/{len(stations)} have COMID ({100*final_hits/len(stations):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
