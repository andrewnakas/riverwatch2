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


def _clean_dist() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    FORECAST_DIR.mkdir(parents=True)
    (DIST / "static").mkdir(parents=True)


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
    """Strip non-JSON-friendly bits (pandas Timestamps slip through asdict)."""
    return json.loads(json.dumps(payload, default=str))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--no-forecasts", action="store_true",
                    help="copy assets but skip forecast generation (smoke test)")
    args = ap.parse_args()

    _clean_dist()
    _copy_assets()

    payload = json.loads(STATIONS_PATH.read_text())
    stations = payload["stations"]
    if args.limit:
        stations = stations[: args.limit]

    # Stations payload (frontend reads this for marker placement)
    (DIST / "stations.json").write_text(json.dumps({"stations": stations}, indent=2))

    if args.no_forecasts:
        print("--no-forecasts: skipping live forecast runs")
        return 0

    successes = 0
    failures: list[dict] = []
    blend_maes: list[float] = []
    member_rolling = {"persistence_lag1": [], "runoff_ridge": [], "chronos_bolt": []}
    t0 = time.time()
    for i, st in enumerate(stations, 1):
        sid = st["id"]
        ts = time.time()
        try:
            f = forecast_station(sid, st["lat"], st["lon"], horizon=args.horizon)
            data = asdict(f)
            data["station"] = st
            (FORECAST_DIR / f"{sid}.json").write_text(json.dumps(_to_jsonable(data), indent=2))
            successes += 1
            for name, pts in f.members.items():
                rm = f.rolling_mae.get(name)
                if rm is not None and np.isfinite(rm):
                    member_rolling.setdefault(name, []).append(float(rm))
            print(f"[{i:>2}/{len(stations)}] {sid} OK  chosen={f.chosen}  {time.time()-ts:5.1f}s")
        except Exception as exc:
            print(f"[{i:>2}/{len(stations)}] {sid} FAIL  {exc}")
            failures.append({"station_id": sid, "error": str(exc)})

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stations_total": len(stations),
        "stations_succeeded": successes,
        "stations_failed": [f["station_id"] for f in failures],
        "horizon_days": args.horizon,
        "rolling_mae_mean_by_member": {
            k: (mean(v) if v else None) for k, v in member_rolling.items()
        },
        "build_seconds": round(time.time() - t0, 1),
        "failures": failures,
    }
    (DIST / "index_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBuilt dist/ in {summary['build_seconds']}s — {successes}/{len(stations)} stations")
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
