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
import os
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

from app.forecast import forecast_station, prepare_station_inputs, recompute_blend_with_stacker  # noqa: E402
from app.pooled_lgbm import PooledTrainer  # noqa: E402
from app.stacker import StackerTrainer  # noqa: E402
from app.gages2 import enrich_station_attrs, coverage_summary as gages2_coverage  # noqa: E402

# v15.0: bootstrap landed (run #62 succeeded with cold caches filled),
# so the scheduled-run default now points at the full 9851-gauge list.
# RW2_STATIONS_FILE still overrides for local dev / benchmarks (fall back
# to data/stations_40_enriched.json if you need the legacy 1893 set).
STATIONS_PATH = Path(os.environ.get("RW2_STATIONS_FILE") or (ROOT / "data" / "stations_v15.json"))
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

    # v14.5a: enrich station dicts in place with curated GAGES-II static
    # attributes (~68% coverage). Stations without a GAGES-II row are
    # untouched and the new feature cells stay NaN at LightGBM training time.
    g2_summary = gages2_coverage()
    print(f"v14.5a GAGES-II: {g2_summary['stations_with_gages2']} rows in table, "
          f"{len(g2_summary['keys'])} keys per station")
    enriched_in_shard = 0
    for st in stations:
        before = len(st)
        enrich_station_attrs(st)
        if len(st) > before:
            enriched_in_shard += 1
    print(f"v14.5a GAGES-II: enriched {enriched_in_shard}/{len(stations)} stations in this shard")

    if args.no_forecasts:
        print("--no-forecasts: skipping live forecast runs")
        return 0

    successes = 0
    failures: list[dict] = []
    member_rolling: dict[str, list[float]] = {}
    blend_rolling: list[float] = []
    blend_h7: list[float] = []
    blend_h14: list[float] = []
    # v14.2: collect raw NWM medium_range_blend curves so the snapshot job can
    # concat all 16 shards into one daily archive parquet on the nwm-archive
    # branch. Tuple per row: (issued_date, station_id, target_date, horizon_day,
    # q_cfs_raw, q_cfs_obs_today, bias_scale_used).
    nwm_raw_rows: list[tuple] = []
    t0 = time.time()

    # v13.2: 2-pass build. Pass 1 collects fetched + feature-engineered inputs
    # for every station in this shard, accumulates pooled-LGBM training rows,
    # and fits 14 per-horizon models. Pass 2 runs forecast_station with those
    # cached inputs + the trained pooled predictor to produce the new
    # `lgbm_pooled` ensemble member alongside the existing 6.
    print(f"v13.2 pass 1: collecting inputs across {len(stations)} stations…")
    pooled = PooledTrainer(horizon=args.horizon)
    inputs_by_sid: dict[str, object] = {}
    pass1_failures: list[str] = []
    for i, st in enumerate(stations, 1):
        sid = st["id"]
        ts = time.time()
        try:
            si = prepare_station_inputs(
                sid, st["lat"], st["lon"], horizon=args.horizon, station_attrs=st
            )
            inputs_by_sid[sid] = si
            pooled.add_station(
                sid, si.feats, st, si.cols, si.qs, si.has_clim,
            )
            if i % 25 == 0 or i == len(stations):
                print(f"  pass1 [{i}/{len(stations)}] sid={sid} rows={pooled.n_rows()} {time.time()-ts:.1f}s")
        except Exception as exc:
            pass1_failures.append(sid)
            print(f"  pass1 [{i}/{len(stations)}] sid={sid} FAIL {exc}")

    pooled_ok = False
    if pooled.n_rows() >= 1000:
        t_fit = time.time()
        print(f"v13.2 pass 1 fit: {pooled.n_rows()} rows across {len(inputs_by_sid)} stations…")
        pooled_ok = pooled.fit()
        print(f"  pooled fit: ok={pooled_ok} {time.time()-t_fit:.1f}s")
    else:
        print(f"v13.2 pass 1 skipped: only {pooled.n_rows()} pooled rows (need >=1000)")

    # v14.4: pass 2 = produce per-station forecasts (v13.1 rule + NWM-shrinkage
    # blend), holding them in memory; we feed their `_v14_4_panel`s into the
    # pooled stacker. Pass 3 fits the stacker on the panel union and rewrites
    # each station's blend + rolling_mae_blend in place. JSON is only written
    # after pass 3 so we never publish the rule-blend version.
    print(f"v13.2 pass 2: per-station forecasts…")
    forecasts_by_sid: dict[str, object] = {}
    station_by_sid: dict[str, dict] = {}
    for i, st in enumerate(stations, 1):
        sid = st["id"]
        ts = time.time()
        try:
            si = inputs_by_sid.get(sid)
            f = forecast_station(
                sid, st["lat"], st["lon"], horizon=args.horizon, station_attrs=st,
                inputs=si,
                pooled_predictor=pooled if pooled_ok else None,
            )
            forecasts_by_sid[sid] = f
            station_by_sid[sid] = st
            shard_tag = f"shard {args.shard_id}/{args.total_shards}: " if sharded else ""
            print(f"{shard_tag}[{i:>3}/{len(stations)}] {sid} OK  chosen={f.chosen}  {time.time()-ts:5.1f}s")
        except Exception as exc:
            print(f"[{i:>3}/{len(stations)}] {sid} FAIL  {exc}")
            failures.append({"station_id": sid, "error": str(exc)})

    # v14.4 pass 2.5: aggregate holdout panels into a pooled blend stacker
    # and fit per-horizon LightGBMs. Skipped if disabled or insufficient data.
    import os as _os
    stacker_ok = False
    stacker_obj: object = None
    if _os.environ.get("RW2_STACKER_OFF") == "1":
        print("v14.4 stacker: disabled via RW2_STACKER_OFF=1")
    else:
        st_train = StackerTrainer(horizon=args.horizon)
        for sid, f in forecasts_by_sid.items():
            panel = getattr(f, "_v14_4_panel", None)
            if panel is None or panel.get("attrs") is None:
                continue
            try:
                st_train.add_station(
                    sid,
                    panel["attrs"],
                    panel["qs"],
                    panel["member_preds"],
                    panel["offset_to_issued_doy"],
                )
            except Exception as exc:
                print(f"  stacker.add_station FAIL {sid}: {exc}")
        n_rows = st_train.n_rows()
        # 6 offsets × 14 h × 118 stations ≈ 10K rows max per shard, so a
        # 200-row floor on any single horizon is comfortable.
        if n_rows >= 200:
            t_fit = time.time()
            print(f"v14.4 stacker fit: {n_rows} rows across {len(forecasts_by_sid)} stations…")
            stacker_ok = st_train.fit()
            stacker_obj = st_train if stacker_ok else None
            print(f"  stacker fit: ok={stacker_ok} {time.time()-t_fit:.1f}s")
        else:
            print(f"v14.4 stacker skipped: only {n_rows} pooled rows (need >=200)")

    # v14.4 pass 3: rewrite each station's blend and write JSON.
    print(f"v14.4 pass 3: finalize + write per-station JSON…")
    for i, st in enumerate(stations, 1):
        sid = st["id"]
        f = forecasts_by_sid.get(sid)
        if f is None:
            continue  # pass-2 failure; already recorded
        try:
            if stacker_ok and stacker_obj is not None:
                f = recompute_blend_with_stacker(f, stacker_obj)
            # Strip the in-memory panel before serializing — large dict, not for Pages.
            try:
                f._v14_4_panel = None
            except Exception:
                pass
            data = asdict(f)
            # asdict still includes _v14_4_panel as None; drop the key entirely.
            data.pop("_v14_4_panel", None)
            data["station"] = st
            data["has_history_file"] = _emit_history(sid)
            (FORECAST_DIR / f"{sid}.json").write_text(json.dumps(_to_jsonable(data), indent=2))
            successes += 1
            # v14.2: capture the raw NWM curve for the snapshot job.
            if f.nwm_raw_forecast:
                issued = f.issued_at[:10]  # YYYY-MM-DD
                q_obs_today = (f.history[-1]["q_cfs"] if f.history else None)
                bs_used = f.nwm_bias_scale_used
                for h_idx, pt in enumerate(f.nwm_raw_forecast, start=1):
                    nwm_raw_rows.append((
                        issued, sid, pt.get("date"), h_idx,
                        pt.get("q_cfs"), q_obs_today, bs_used,
                    ))
            for name, pts in f.members.items():
                rm = f.rolling_mae.get(name)
                if rm is not None and np.isfinite(rm):
                    member_rolling.setdefault(name, []).append(float(rm))
            blend_mae = f.rolling_mae.get("ensemble_blend")
            if blend_mae is not None and np.isfinite(blend_mae):
                blend_rolling.append(float(blend_mae))
            b7 = f.rolling_mae_h7.get("ensemble_blend") if f.rolling_mae_h7 else None
            if b7 is not None and np.isfinite(b7):
                blend_h7.append(float(b7))
            b14 = f.rolling_mae_h14.get("ensemble_blend") if f.rolling_mae_h14 else None
            if b14 is not None and np.isfinite(b14):
                blend_h14.append(float(b14))
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
        "rolling_mae_blend_mean": (mean(blend_rolling) if blend_rolling else None),
        "rolling_mae_blend_h7_mean": (mean(blend_h7) if blend_h7 else None),
        "rolling_mae_blend_h14_mean": (mean(blend_h14) if blend_h14 else None),
        "stations_with_blend_mae": len(blend_rolling),
        "build_seconds": round(time.time() - t0, 1),
        "failures": failures,
        # v14.4: stacker observability
        "stacker_used": bool(stacker_ok),
        "stacker_horizons_fit": (
            sorted(int(h) for h in (stacker_obj._models.keys()))  # type: ignore[attr-defined]
            if stacker_ok and stacker_obj is not None else []
        ),
    }
    # v14.2: emit this shard's NWM raw rows into dist/_nwm_raw/shard_N.csv.gz
    # so the snapshot job can pick them up alongside the deploy artifact and
    # ship one consolidated archive parquet per day to the nwm-archive branch.
    if nwm_raw_rows:
        import csv, gzip
        nwm_dir = DIST / "_nwm_raw"
        nwm_dir.mkdir(parents=True, exist_ok=True)
        out_path = nwm_dir / f"shard_{args.shard_id}.csv.gz"
        with gzip.open(out_path, "wt", newline="") as gz:
            w = csv.writer(gz)
            w.writerow(["issued_date", "station_id", "target_date",
                        "horizon_day", "q_cfs_raw", "q_cfs_obs_today",
                        "bias_scale_used"])
            w.writerows(nwm_raw_rows)
        print(f"v14.2: wrote {len(nwm_raw_rows)} NWM raw rows → {out_path}")
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
