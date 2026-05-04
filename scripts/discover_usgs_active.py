#!/usr/bin/env python3
"""v15.0: discover every active USGS daily-flow gauge in the U.S.

The current v14.x build runs against a curated 1893-gauge list
(`data/stations_40_enriched.json` — name kept for git-history reasons).
v15 expands to the full active fleet. NWIS exposes a per-state site
service that, with the right filters, returns every operating
streamflow station: `siteType=ST` (stream), `parameterCd=00060`
(discharge), `siteStatus=active`, and `outputDataTypeCd=dv` (only
sites with daily-value records — drops continuous-only / event sites).

We walk all 50 states + DC + PR/VI + AK + HI and merge results.
USGS publishes ~10-12k active dv-flow gauges depending on year.

Output: `data/stations_v15.json` with the same per-station schema as
`stations_40_enriched.json` so the downstream build pipeline keeps
working — just bigger.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stations_v15.json"

USER_AGENT = "riverwatch2/0.1 v15-station-discovery"

# Lower 48 + DC + AK + HI + PR + VI. NWIS uses the FIPS state postal codes.
STATE_CODES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL",
    "IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE",
    "NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD",
    "TN","TX","UT","VT","VA","WA","WV","WI","WY","PR","VI",
]


def _fetch_state(state: str, *, retries: int = 3, timeout: int = 90) -> list[dict]:
    """Pull every active dv-flow gauge in one state."""
    params = {
        "format": "rdb",
        "stateCd": state,
        "siteType": "ST",            # stream sites only (drops springs, lakes, wells)
        "siteStatus": "active",
        "parameterCd": "00060",      # discharge cfs
        "hasDataTypeCd": "dv",       # ensures dv data actually exists
        "siteOutput": "expanded",    # gives drain_area_va, alt_va, huc_cd, etc.
    }
    url = "https://waterservices.usgs.gov/nwis/site/?" + urlencode(params)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            break
        except Exception as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
            else:
                raise
    return _parse_rdb(text)


def _parse_rdb(text: str) -> list[dict]:
    """Parse NWIS RDB tab-delimited output into list of dicts.
    Skips comments, the header row, and the type-spec row ("5s", "10s", ...).
    """
    rows: list[dict] = []
    header: list[str] | None = None
    type_row_seen = False
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if header is None:
            header = cols
            continue
        if not type_row_seen:
            # the row immediately after the header is column-width hints
            # ("5s", "15s", "10n", ...). Skip it once.
            type_row_seen = True
            if cols and (cols[0].endswith("s") or cols[0].endswith("n")):
                continue
        rows.append(dict(zip(header, cols)))
    return rows


def _to_station(row: dict) -> dict | None:
    """Map an NWIS RDB row to our station schema. Returns None on bad rows."""
    site_no = (row.get("site_no") or "").strip()
    if not site_no:
        return None

    def _f(k: str):
        v = row.get(k)
        if v in (None, "", " "):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    lat = _f("dec_lat_va")
    lon = _f("dec_long_va")
    if lat is None or lon is None:
        return None

    return {
        "id": site_no,
        "site_no": site_no,
        "name": (row.get("station_nm") or "").strip(),
        "state": (row.get("state_cd") or "").strip(),
        "state_cd": (row.get("state_cd") or "").strip(),
        "huc_cd": (row.get("huc_cd") or "").strip() or None,
        "lat": lat,
        "lon": lon,
        "drain_area_sqmi": _f("drain_area_va"),
        "alt_ft": _f("alt_va"),
        "site_type_cd": (row.get("site_tp_cd") or "").strip() or None,
    }


def main() -> int:
    seen: dict[str, dict] = {}
    fail: list[tuple[str, str]] = []
    t0 = time.time()
    for i, st in enumerate(STATE_CODES, 1):
        try:
            rows = _fetch_state(st)
        except Exception as exc:
            print(f"  ! {st}: fetch failed: {exc}", flush=True)
            fail.append((st, str(exc)))
            continue
        added = 0
        for row in rows:
            s = _to_station(row)
            if not s:
                continue
            # Some sites cross state lines and show up twice — keep the first
            # version we saw.
            if s["id"] in seen:
                continue
            seen[s["id"]] = s
            added += 1
        print(
            f"  [{i:2d}/{len(STATE_CODES)}] {st}: {added:5d} active dv-flow "
            f"(running total {len(seen):5d}, elapsed {time.time()-t0:.0f}s)",
            flush=True,
        )

    stations = sorted(seen.values(), key=lambda s: s["id"])
    payload = {
        "version": "v15.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_stations": len(stations),
        "n_states_attempted": len(STATE_CODES),
        "n_states_failed": len(fail),
        "states_failed": fail,
        "stations": stations,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(
        f"\nwrote {len(stations)} active dv-flow gauges to {OUT.relative_to(ROOT)}"
    )
    if fail:
        print(f"  {len(fail)} states failed: {[f[0] for f in fail]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
