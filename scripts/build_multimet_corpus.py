#!/usr/bin/env python3
"""Fetch a 2014-2024 encoder-history corpus for the MultiMet HRES evaluation.

Caravan's CAMELS discharge stops at 2014 (verified 2026-07-07), so it cannot
supply encoder history or scoring targets for MultiMet's 2016-2024 forecast
window. This builds that corpus directly from source:

  q_cfs        USGS daily values (parameter 00060, stat 00003) via the same
               fetch the NWM backtest uses (backtest_nwm_residual._fetch_dv_chunk)
  weather      Open-Meteo daily archive (app.weather.fetch_history) in the
               DAILY_VARS schema, sliced to the compat 5 the shipped model uses

Output: one {gid}.csv.gz per gauge in the corpus schema
    date, q_cfs, temperature_2m_mean, temperature_2m_max, temperature_2m_min,
    precipitation_sum, shortwave_radiation_sum
covering --start..--end (default 2014-01-01..2024-12-31: the 2015-2020 stretch
gives >=365d encoder history for every MultiMet issue date 2016-2024, and the
same USGS obs serve as scoring truth).

Lat/lon come from stations_v15.json, falling back to the Caravan 'other'
attributes for the ~16 CAMELS gauges absent from the registry. Resumable:
skips a gauge whose output already spans the requested window.

Usage:
  .venv/bin/python scripts/build_multimet_corpus.py \
      --start 2014-01-01 --end 2024-12-31 \
      --out-dir /Volumes/STORAGE_SD/riverwatch2_data/multimet_corpus_531
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import weather  # noqa: E402

# Reuse the NWM backtest's USGS DV fetch verbatim (proven, cached).
_spec = importlib.util.spec_from_file_location(
    "backtest_nwm_residual", ROOT / "scripts" / "backtest_nwm_residual.py")
nwm_bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nwm_bt)

COMPAT_VARS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "shortwave_radiation_sum",
]
CORPUS_COLS = ["date", "q_cfs", *COMPAT_VARS]
DEFAULT_OUT = Path("/Volumes/STORAGE_SD/riverwatch2_data/multimet_corpus_531")
CARAVAN_ATTRS = Path("/Volumes/STORAGE_SD/riverwatch2_data/external/caravan_base/"
                     "Caravan/attributes/camels/attributes_other_camels.csv")


def load_latlon() -> dict[str, tuple[float, float]]:
    """gid -> (lat, lon), registry first then Caravan attributes fallback."""
    out: dict[str, tuple[float, float]] = {}
    raw = json.loads((ROOT / "data" / "stations_v15.json").read_text())
    for s in (raw["stations"] if isinstance(raw, dict) else raw):
        if s.get("lat") is not None and s.get("lon") is not None:
            out[s["id"]] = (float(s["lat"]), float(s["lon"]))
    if CARAVAN_ATTRS.exists():
        a = pd.read_csv(CARAVAN_ATTRS)
        latc = "gauge_lat" if "gauge_lat" in a.columns else None
        lonc = "gauge_lon" if "gauge_lon" in a.columns else None
        if latc and lonc:
            for _, r in a.iterrows():
                gid = str(r["gauge_id"]).split("_", 1)[1]
                if gid not in out and pd.notna(r[latc]) and pd.notna(r[lonc]):
                    out[gid] = (float(r[latc]), float(r[lonc]))
    return out


def _prefetched(gid: str, start: date, end: date) -> pd.DataFrame | None:
    """Reuse a corpus_openmeteo file if it already spans the window — the
    trickle has ~37 CAMELS gauges at full 1990-2026 coverage, free of the
    Open-Meteo quota."""
    p = ROOT / "data" / "mblstm" / "corpus_openmeteo" / f"{gid}.csv.gz"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].min().date() > start or df["date"].max().date() < end:
        return None
    cols = ["date"] + [c for c in COMPAT_VARS if c in df.columns]
    return df[cols]


def build_gauge(gid: str, latlon: tuple[float, float], obs: dict[str, float],
                start: date, end: date, out_dir: Path,
                prefetched_only: bool = False) -> dict | None:
    lat, lon = latlon
    wx = _prefetched(gid, start, end)
    if wx is None:
        if prefetched_only:
            return None  # skip API-dependent gauges in this pass
        wx = weather.fetch_history(lat, lon, start, end)  # persistent cache; quota-limited
    wx = wx[["date"] + [c for c in COMPAT_VARS if c in wx.columns]].copy()
    wx["date"] = pd.to_datetime(wx["date"])
    full = pd.date_range(start, end, freq="D")
    wx = wx.set_index("date").reindex(full).reset_index().rename(columns={"index": "date"})
    for c in COMPAT_VARS:
        if c not in wx.columns:
            wx[c] = np.nan
    q = pd.Series({pd.Timestamp(d): v for d, v in obs.items()})
    wx["q_cfs"] = wx["date"].map(q).astype(float)
    df = wx[CORPUS_COLS].copy()
    stats = {"id": gid, "rows": len(df), "q_valid": int(df["q_cfs"].notna().sum()),
             "wx_valid": int(df["temperature_2m_mean"].notna().sum())}
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(out_dir / f"{gid}.csv.gz", index=False, compression="gzip")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-json", default=str(ROOT / "data" / "camels_gauge_ids.json"))
    ap.add_argument("--ids-key", default="531")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refetch", action="store_true",
                    help="rebuild gauges even if a valid output already exists")
    ap.add_argument("--quota-stop", type=int, default=8,
                    help="stop after this many consecutive weatherless fetches "
                         "(Open-Meteo 429 throttle); re-run resumes")
    ap.add_argument("--prefetched-only", action="store_true",
                    help="build ONLY gauges already covered by corpus_openmeteo "
                         "(no Open-Meteo API calls) — an immediate quota-free "
                         "cohort; run without this later to fill the rest")
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start).date(), pd.Timestamp(args.end).date()
    ids = json.loads(Path(args.ids_json).read_text())[args.ids_key]
    if args.limit:
        ids = ids[: args.limit]
    latlon = load_latlon()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # USGS obs: chunked fetch (cached under data/cache/backtest_obs via nwm_bt).
    print(f"fetching USGS DV {start}..{end} for {len(ids)} gauges ...", flush=True)
    obs_all: dict[str, dict[str, float]] = {}
    chunks = [ids[i:i + 100] for i in range(0, len(ids), 100)]
    for i, chunk in enumerate(chunks, 1):
        got = nwm_bt._fetch_dv_chunk(chunk, start, end)
        obs_all.update(got)
        print(f"  DV chunk {i}/{len(chunks)}: +{len(got)} gauges", flush=True)
        time.sleep(0.5)

    no_latlon, built, thin, skipped = [], 0, [], 0
    empty_streak = 0
    t0 = time.time()
    for i, gid in enumerate(ids, 1):
        out_p = args.out_dir / f"{gid}.csv.gz"
        if out_p.exists() and not args.refetch:
            # resumable: already built with valid weather? skip.
            prev = pd.read_csv(out_p, usecols=["temperature_2m_mean"])
            if prev["temperature_2m_mean"].notna().sum() >= 365:
                skipped += 1
                continue
        if gid not in latlon:
            no_latlon.append(gid)
            continue
        try:
            st = build_gauge(gid, latlon[gid], obs_all.get(gid, {}), start, end,
                             args.out_dir, prefetched_only=args.prefetched_only)
            if st is None:  # prefetched-only pass, no cached weather → skip
                continue
            built += 1
            if st["wx_valid"] < 365:
                empty_streak += 1
                thin.append(gid)
            else:
                empty_streak = 0
            if st["q_valid"] < 365 and gid not in thin:
                thin.append(gid)
        except Exception as exc:
            print(f"[{i}/{len(ids)}] {gid} ERR {exc}", flush=True)
        # Open-Meteo quota: after N straight weatherless builds we're throttled;
        # stop so a re-run (resumable) picks up in the next quota window rather
        # than filling the corpus with empty-weather files.
        if not args.prefetched_only and empty_streak >= args.quota_stop:
            print(f"\nSTOPPING: {empty_streak} straight weatherless fetches — "
                  f"Open-Meteo quota hit. Re-run to resume (built {built}, "
                  f"skipped {skipped} already-done).", flush=True)
            break
        if i % 25 == 0:
            print(f"[{i}/{len(ids)}] built {built}, skipped {skipped} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\nbuilt {built}/{len(ids)} -> {args.out_dir}")
    if no_latlon:
        print(f"WARNING: {len(no_latlon)} gauges had no lat/lon: {no_latlon[:10]}")
    if thin:
        print(f"NOTE: {len(thin)} gauges have <365 valid obs or weather days "
              f"(sparse — will score fewer windows): {thin[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
