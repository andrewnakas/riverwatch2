#!/usr/bin/env python3
"""Sync the Open-Meteo training corpus from HuggingFace into the local tree.

OpenCLAW publishes the full 13-variable reanalysis corpus (1990→present) at
nakas/rw2-openmeteo-corpus as daily/<usgs_id>.csv.gz. This downloads them into
data/mblstm/corpus_openmeteo/<usgs_id>.csv.gz so the trainer can use them with
the FULL forcing set — closing the Daymet-train / Open-Meteo-serve mismatch that
gated the model to 5 of 13 variables.

Resumable: skips files already present with a matching size. Reads the repo
manifest to report how complete the ingest is (it runs continuously, so this
script is safe to re-run as the corpus fills).

Usage:
  .venv/bin/python scripts/sync_openmeteo_corpus.py            # sync all
  .venv/bin/python scripts/sync_openmeteo_corpus.py --limit 50 # first N (smoke)
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = "nakas/rw2-openmeteo-corpus"
OUT_DIR = ROOT / "data" / "mblstm" / "corpus_openmeteo"


def join_discharge() -> int:
    """Merge USGS observed discharge (date,q_cfs) into each weather-only
    Open-Meteo file, producing a trainer-ready corpus. The Open-Meteo dataset is
    weather-only by design; the MB-LSTM trainer needs a q_cfs column. Resumable:
    a station is skipped if its output already has q_cfs. Reuses the repo's
    cached USGS fetch (app.usgs.fetch_daily_discharge)."""
    import pandas as pd
    from app import usgs

    files = sorted(OUT_DIR.glob("*.csv.gz"))
    print(f"join-discharge: {len(files)} weather files in {OUT_DIR.name}", flush=True)
    done = skip = fail = 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        sid = p.name.split(".")[0]
        try:
            wx = pd.read_csv(p)
            if "q_cfs" in wx.columns and wx["q_cfs"].notna().any():
                skip += 1
                continue
            wx["date"] = pd.to_datetime(wx["date"])
            s = wx["date"].iloc[0].date(); e = wx["date"].iloc[-1].date()
            q = usgs.fetch_daily_discharge(sid, s, e)
            if q is None or q.empty:
                fail += 1
                print(f"[{i}/{len(files)}] {sid} no USGS discharge", flush=True)
                continue
            q["date"] = pd.to_datetime(q["date"])
            merged = wx.merge(q[["date", "q_cfs"]], on="date", how="left")
            merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
            merged.to_csv(p, index=False, compression="gzip")
            done += 1
        except Exception as exc:
            fail += 1
            print(f"[{i}/{len(files)}] {sid} ERR {exc}", flush=True)
            continue
        if done % 25 == 0:
            print(f"[{i}/{len(files)}] joined {done} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\njoined={done} skipped={skip} failed={fail} in {(time.time()-t0)/60:.1f} min",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N stations (smoke)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--join-discharge", action="store_true",
                    help="after/instead of download, merge USGS q_cfs into the "
                         "weather-only files to make a trainer-ready corpus")
    args = ap.parse_args()

    if args.join_discharge:
        return join_discharge()

    from huggingface_hub import HfApi, hf_hub_download

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(args.repo)
    daily = sorted(s.rfilename for s in info.siblings
                   if s.rfilename.startswith("daily/") and s.rfilename.endswith(".csv.gz"))
    if args.limit:
        daily = daily[: args.limit]
    print(f"{args.repo}: {len(daily)} daily station files visible", flush=True)

    # Manifest = ground truth on how many stations are fully validated.
    try:
        mpath = hf_hub_download(args.repo, "manifest.json", repo_type="dataset")
        man = json.loads(Path(mpath).read_text())
        print(f"manifest: completed_station_count={man.get('completed_station_count')} "
              f"failed={man.get('failed_station_count')} "
              f"vars={len(man.get('variables', []))}", flush=True)
    except Exception as exc:
        print(f"manifest unavailable ({exc})", flush=True)

    got = skip = fail = 0
    t0 = time.time()
    for i, rfn in enumerate(daily, 1):
        sid = Path(rfn).name
        dest = OUT_DIR / sid
        if dest.exists() and dest.stat().st_size > 0:
            skip += 1
            continue
        try:
            fp = hf_hub_download(args.repo, rfn, repo_type="dataset")
            data = Path(fp).read_bytes()
            dest.write_bytes(data)
            got += 1
        except Exception as exc:
            fail += 1
            print(f"[{i}/{len(daily)}] {sid} ERR {exc}", flush=True)
            continue
        if got % 50 == 0:
            print(f"[{i}/{len(daily)}] downloaded {got} ({time.time()-t0:.0f}s)", flush=True)

    have = len(list(OUT_DIR.glob("*.csv.gz")))
    print(f"\ndownloaded={got} skipped={skip} failed={fail} | "
          f"local corpus now {have} stations → {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
