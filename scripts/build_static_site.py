#!/usr/bin/env python3
"""Build a fully-static copy of the map + per-station forecasts into ./dist.

This is the GitHub Pages target. The CI workflow runs this script and uploads
./dist as a Pages artifact.

For each of the 40 stations, we run the live forecaster (USGS + Open-Meteo +
ridge + Chronos-Bolt + ensemble blend) and write the result to
  dist/forecasts/<station_id>.json

Plus:
  dist/index.html                copy of the Flask template
  dist/static/...                copy of the Flask static assets
  dist/stations.json             station list with metadata
  dist/index_summary.json        aggregate MAE / freshness payload

The frontend (rewritten static_app.js) calls these JSON files directly instead
of hitting Flask.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.forecast import forecast_station  # noqa: E402

STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"
SRC_TEMPLATE = ROOT / "app" / "templates" / "index.html"
SRC_STATIC = ROOT / "app" / "static"
DIST = ROOT / "dist"
FORECAST_DIR = DIST / "forecasts"
HISTORY_DIR = DIST / "history"
USGS_RECORDS_DIR = ROOT / "data" / "cache" / "usgs_records"


def _clean_dist() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    FORECAST_DIR.mkdir(parents=True)
    HISTORY_DIR.mkdir(parents=True)
    (DIST / "static").mkdir(parents=True)


def _emit_history(site_no: str) -> bool:
    """Copy the cached USGS daily-discharge record into dist/history/<id>.json.

    The frontend lazy-fetches this only when the user opens the year-compare
    widget on the climatology chart, so it's fine for it to be the larger of
    the per-station files (a 100yr record is ~50KB compressed).
    """
    src = USGS_RECORDS_DIR / f"{site_no}.json"
    if not src.exists():
        return False
    try:
        rec = json.loads(src.read_text())
    except Exception:
        return False
    rows = rec.get("rows") or {}
    if not rows:
        return False
    payload = {
        "site_no": site_no,
        "first_known": rec.get("first_known") or (min(rows.keys()) if rows else None),
        "last_known": rec.get("last_known") or (max(rows.keys()) if rows else None),
        "rows": rows,  # {date: q_cfs}
    }
    (HISTORY_DIR / f"{site_no}.json").write_text(json.dumps(payload, separators=(",", ":")))
    return True


def _copy_assets() -> None:
    # static/ assets — but swap app.js for static_app.js
    for item in SRC_STATIC.iterdir():
        if item.name in {"app.js", "static_app.js"}:
            continue
        dst = DIST / "static" / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    # static_app.js → static/app.js
    static_app = SRC_STATIC / "static_app.js"
    shutil.copy2(static_app, DIST / "static" / "app.js")
    # index.html — rewrite paths so it works at any subpath
    html = SRC_TEMPLATE.read_text()
    html = html.replace('href="/static/', 'href="static/').replace(
        'src="/static/', 'src="static/'
    )
    (DIST / "index.html").write_text(html)


def _to_jsonable(payload):
    """Strip non-JSON-friendly bits (pandas Timestamps slip through asdict).

    Also rewrite NaN/Inf -> None so the emitted JSON is valid; the browser's
    JSON.parse rejects bare `Infinity` even though Python json.dumps writes it.
    """
    import math

    def _scrub(v):
        if isinstance(v, float):
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(v, dict):
            return {k: _scrub(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_scrub(x) for x in v]
        return v

    return _scrub(json.loads(json.dumps(payload, default=str)))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--no-forecasts", action="store_true",
                    help="copy assets but skip forecast generation (smoke test)")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="this shard's index, 0..total_shards-1")
    ap.add_argument("--total-shards", type=int, default=1,
                    help="how many shards the build is split across")
    args = ap.parse_args()

    sharded = args.total_shards > 1
    is_first_shard = args.shard_id == 0

    # In sharded mode the first shard owns the assets; subsequent shards
    # only emit forecast JSON files (and a per-shard summary).
    if not sharded or is_first_shard:
        _clean_dist()
        _copy_assets()
    else:
        DIST.mkdir(parents=True, exist_ok=True)
        FORECAST_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    payload = json.loads(STATIONS_PATH.read_text())
    all_stations = payload["stations"]
    if args.limit:
        all_stations = all_stations[: args.limit]

    if not sharded or is_first_shard:
        # The full station list lives once at dist/stations.json (asset shard owns it).
        (DIST / "stations.json").write_text(json.dumps({"stations": all_stations}, indent=2))

    # Slice this shard's portion of the station list. stride = total_shards
    # gives every shard a similar runtime distribution (USGS hot/cold mix).
    if sharded:
        stations = all_stations[args.shard_id :: args.total_shards]
    else:
        stations = all_stations

    if args.no_forecasts:
        print("--no-forecasts: skipping live forecast runs")
        return 0

    successes = 0
    failures: list[dict] = []
    member_rolling: dict[str, list[float]] = {}
    t0 = time.time()
    for i, st in enumerate(stations, 1):
        sid = st["id"]
        ts = time.time()
        try:
            f = forecast_station(sid, st["lat"], st["lon"], horizon=args.horizon)
            data = asdict(f)
            data["station"] = st
            data["has_history_file"] = _emit_history(sid)
            (FORECAST_DIR / f"{sid}.json").write_text(json.dumps(_to_jsonable(data), indent=2))
            successes += 1
            for name, pts in f.members.items():
                rm = f.rolling_mae.get(name)
                if rm is not None and np.isfinite(rm):
                    member_rolling.setdefault(name, []).append(float(rm))
            shard_tag = f"shard {args.shard_id}/{args.total_shards}: " if sharded else ""
            print(f"{shard_tag}[{i:>3}/{len(stations)}] {sid} OK  chosen={f.chosen}  {time.time()-ts:5.1f}s")
        except Exception as exc:
            print(f"[{i:>3}/{len(stations)}] {sid} FAIL  {exc}")
            failures.append({"station_id": sid, "error": str(exc)})

    summary = {
        "shard_id": args.shard_id,
        "total_shards": args.total_shards,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stations_in_shard": len(stations),
        "stations_succeeded": successes,
        "stations_failed": [f["station_id"] for f in failures],
        "horizon_days": args.horizon,
        "rolling_mae_mean_by_member": {
            k: (mean(v) if v else None) for k, v in member_rolling.items()
        },
        "build_seconds": round(time.time() - t0, 1),
        "failures": failures,
    }
    if sharded:
        (DIST / f"index_summary_shard_{args.shard_id}.json").write_text(json.dumps(summary, indent=2))
    else:
        # In single-shard / dev mode, also write top-level summary as before so
        # the frontend's "Built ..." note still works locally.
        summary["stations_total"] = len(stations)
        (DIST / "index_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nShard {args.shard_id}/{args.total_shards} built in {summary['build_seconds']}s — {successes}/{len(stations)} stations")
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
