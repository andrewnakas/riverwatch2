"""SNOTEL site inventory and CAMELS-531 basin matching.

Step 1a of the multi8 member: establish which basins can carry SNOTEL channels
at all, and how far back each site's record goes. SNOTEL is NRCS-operated, so
it sits outside the COOP/GHCN gauge base that daymet/nldas/maurer/aorc share.

Writes analysis/out/snotel_basin_match.json:
  {basin_id: [{triplet, name, distance_km, elev_ft, begin_date}, ...]}
sorted nearest-first, sites within MATCH_KM whose record starts by START_BY.

Run: python analysis/snotel_inventory.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.snotel import _load_all_sntl, _haversine_km  # noqa: E402

MATCH_KM = 25.0      # the radius the coverage probe used (68 basins)
START_BY = "1990-01-01"  # site must be reporting before the train window ends
OUT = ROOT / "analysis" / "out"


def basin_coords() -> dict[str, tuple[float, float]]:
    meta = json.loads((ROOT / "data" / "camels_station_meta.json").read_text())
    ids = set(json.loads((ROOT / "data" / "camels_gauge_ids.json").read_text())["531"])
    return {
        s["id"]: (s["lat"], s["lon"])
        for s in meta["stations"]
        if s["id"] in ids and s.get("lat") is not None and s.get("lon") is not None
    }


def site_begin(site: dict) -> str | None:
    """Earliest date this site reports, from the catalogue metadata."""
    for key in ("beginDate", "startDate", "periodOfRecordBeginDate"):
        v = site.get(key)
        if v:
            return str(v)[:10]
    return None


def main() -> int:
    coords = basin_coords()
    print(f"CAMELS-531 basins with coordinates: {len(coords)}")

    sites = _load_all_sntl()
    print(f"SNTL sites in catalogue: {len(sites)}")
    if not sites:
        print("FAIL: empty catalogue — AWDB unreachable?")
        return 1

    early = []
    for s in sites:
        b = site_begin(s)
        if b and b <= START_BY:
            early.append((s, b))
    print(f"sites with record starting <= {START_BY}: {len(early)}")

    match: dict[str, list[dict]] = {}
    for bid, (lat, lon) in coords.items():
        near = []
        for s, b in early:
            d = _haversine_km(lat, lon, s["latitude"], s["longitude"])
            if d <= MATCH_KM:
                near.append({
                    "triplet": s["stationTriplet"],
                    "name": s.get("name"),
                    "distance_km": round(d, 2),
                    "elev_ft": s.get("elevation"),
                    "begin_date": b,
                })
        if near:
            near.sort(key=lambda r: r["distance_km"])
            match[bid] = near

    n_sites = len({r["triplet"] for v in match.values() for r in v})
    print(f"\nbasins with >=1 early SNOTEL site within {MATCH_KM} km: {len(match)}")
    print(f"distinct sites used: {n_sites}")
    for lim in (5.0, 10.0, 15.0, 25.0):
        n = sum(1 for v in match.values() if v[0]["distance_km"] <= lim)
        print(f"  nearest site <= {lim:5.1f} km: {n:3d} basins")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "snotel_basin_match.json").write_text(json.dumps(match, indent=1))
    print(f"\nwrote {OUT / 'snotel_basin_match.json'}")

    if len(match) < 30:
        print("\n⚠️  Too few basins matched to carry a broad member "
              "(breadth gate is 85% of 531).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
