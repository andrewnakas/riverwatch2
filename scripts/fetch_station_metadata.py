#!/usr/bin/env python3
"""Fetch USGS site metadata (lat/lon, drainage area, elevation) for the 40-station subset.

Writes to data/stations_40_enriched.json so the Flask app and map don't need to hit
USGS on every page load.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "stations_40.json"
OUT = ROOT / "data" / "stations_40_enriched.json"


def fetch_usgs_site(site_no: str) -> dict:
    params = {
        "format": "rdb",
        "sites": site_no,
        "siteOutput": "expanded",
        "siteStatus": "all",
    }
    url = "https://waterservices.usgs.gov/nwis/site/?" + urlencode(params)
    req = Request(url, headers={"User-Agent": "riverwatch2/0.1"})
    with urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    header, values = None, None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cols = line.split("\t")
        if header is None:
            header = cols
            continue
        if values is None and cols and cols[0] == "5s":
            continue
        values = cols
        break
    if not (header and values):
        return {}
    row = dict(zip(header, values))
    def f(key: str):
        v = row.get(key)
        if v in (None, ""):
            return None
        try:
            return float(v)
        except ValueError:
            return None
    return {
        "site_no": row.get("site_no"),
        "name": row.get("station_nm"),
        "lat": f("dec_lat_va"),
        "lon": f("dec_long_va"),
        "drain_area_sqmi": f("drain_area_va"),
        "alt_ft": f("alt_va"),
        "huc_cd": row.get("huc_cd"),
        "state_cd": row.get("state_cd"),
    }


def main() -> int:
    src = json.loads(SRC.read_text())
    enriched = []
    for i, station in enumerate(src["stations"], 1):
        sid = station["id"]
        try:
            meta = fetch_usgs_site(sid)
        except Exception as exc:
            print(f"  ! {sid}: {exc}", file=sys.stderr)
            meta = {}
        merged = {**station, **{k: v for k, v in meta.items() if v is not None}}
        if "lat" not in merged or "lon" not in merged:
            print(f"  ! {sid}: missing coordinates after fetch", file=sys.stderr)
        enriched.append(merged)
        print(f"[{i:>2}/{len(src['stations'])}] {sid}  lat={merged.get('lat')}  lon={merged.get('lon')}")
        time.sleep(0.25)

    out = {**src, "stations": enriched}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT} ({len(enriched)} stations)")
    missing = [s["id"] for s in enriched if s.get("lat") is None]
    if missing:
        print(f"WARNING: {len(missing)} stations missing coordinates: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
