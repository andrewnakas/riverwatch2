"""Flask server: map UI + on-demand forecast API for the 43-station subset."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from flask import Flask, jsonify, render_template, request

from .forecast import forecast_station

ROOT = Path(__file__).resolve().parents[1]
STATIONS_PATH = ROOT / "data" / "stations_40_enriched.json"

app = Flask(__name__, template_folder="templates", static_folder="static")

_stations: Dict[str, dict] = {}
_forecast_cache: Dict[str, tuple[float, dict]] = {}
_forecast_locks: Dict[str, threading.Lock] = {}
_global_lock = threading.Lock()
CACHE_TTL_SECONDS = 30 * 60


def _load_stations() -> None:
    global _stations
    payload = json.loads(STATIONS_PATH.read_text())
    _stations = {s["id"]: s for s in payload["stations"]}


_load_stations()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/stations")
def api_stations():
    return jsonify({"stations": list(_stations.values())})


@app.get("/api/forecast/<station_id>")
def api_forecast(station_id: str):
    if station_id not in _stations:
        return jsonify({"error": f"unknown station {station_id}"}), 404

    bypass = request.args.get("refresh") == "1"
    cached = _forecast_cache.get(station_id)
    if cached and not bypass and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return jsonify({"cached": True, **cached[1]})

    with _global_lock:
        lock = _forecast_locks.setdefault(station_id, threading.Lock())

    with lock:
        cached = _forecast_cache.get(station_id)
        if cached and not bypass and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
            return jsonify({"cached": True, **cached[1]})

        meta = _stations[station_id]
        try:
            f = forecast_station(station_id, meta["lat"], meta["lon"])
        except Exception as exc:
            return jsonify({"error": str(exc), "station_id": station_id}), 500
        payload = asdict(f)
        payload["station"] = meta
        _forecast_cache[station_id] = (time.time(), payload)
        return jsonify({"cached": False, **payload})


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "stations": len(_stations)})


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
