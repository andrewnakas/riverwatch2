#!/usr/bin/env python3
"""Build `data/iv_nowcast/nowcast.json` (PREREG §V16 H-LAG): per-gauge
local-day mean discharge for the last complete days, from USGS instantaneous
values, one bulk request per state. See `app/iv_nowcast.py`.

A state that fails is skipped (its gauges simply get no nowcast); the script
exits 0 as long as it writes a table, so it can never block a build.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.iv_nowcast import TABLE, daily_means_from_rdb  # noqa: E402

IV_URL = "https://waterservices.usgs.gov/nwis/iv/"


def main() -> int:
    st = json.loads((ROOT / "data" / "stations_v15.json").read_text())
    st = st["stations"] if isinstance(st, dict) else st
    want = {s["id"] for s in st}
    states = sorted({(s.get("state") or "").lower() for s in st} - {""})
    sites: dict = {}
    failed = []
    for code in states:
        url = f"{IV_URL}?format=rdb&stateCd={code}&parameterCd=00060&period=P3D&siteStatus=active"
        for attempt in range(3):
            try:
                with urlopen(Request(url, headers={"User-Agent": "riverwatch2/0.1"}), timeout=240) as r:
                    got = daily_means_from_rdb(r.read().decode("utf-8", "replace"))
                sites.update({k: v for k, v in got.items() if k in want})
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  {code} attempt {attempt + 1}: {exc}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        else:
            failed.append(code)
        time.sleep(0.5)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(),
                                 "failed_states": failed, "sites": sites}, separators=(",", ":")))
    print(f"iv nowcast: {len(sites)}/{len(want)} gauges, {len(failed)} states failed {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
