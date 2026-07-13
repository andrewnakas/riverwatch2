#!/usr/bin/env python3
"""GRDC → USGS gauge crosswalk for the Google/AIFL benchmark cohorts.

Google's flood-model archive (Zenodo 10.5281/zenodo.10397664) and AIFL's
evaluation both key US gauges as GRDC_<grdc_no>. The GRDC catalogue snapshot
shipped inside Google's own metadata.tgz (grdc_stations_20220320.csv) carries
no national-ID column, so the mapping is recovered by matching GRDC station
coordinates + drainage area against the USGS registry (data/stations_v15.json):

  match = nearest USGS station within MAX_DIST_DEG great-circle-ish degrees
          whose drainage area agrees within AREA_TOL (when both sides have
          area; GRDC area is km^2, USGS drain_area_sqmi * 2.58999).

Ambiguous or unmatched gauges are listed, not guessed. Output:
data/grdc_usgs_crosswalk.json with per-gauge match info + which matches are
in the mblstm corpus (our scoreable cohort).

Usage:
  .venv/bin/python scripts/build_grdc_usgs_crosswalk.py \
      --gauge-list "/Volumes/STORAGE_SD/riverwatch2_data/external/google_flood_zenodo/gauge_groups_for_paper/dual_lstm/country_splits/united_states.txt" \
      --grdc-catalogue "/Volumes/STORAGE_SD/riverwatch2_data/external/google_flood_zenodo/metadata/grdc_stations_20220320.csv"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ROOT / "data" / "stations_v15.json"
CORPUS_DIR = ROOT / "data" / "mblstm" / "corpus"
OUT = ROOT / "data" / "grdc_usgs_crosswalk.json"

MAX_DIST_DEG = 0.05   # ~5 km — gauge coordinates differ by rounding, not km
AREA_TOL = 0.25       # relative drainage-area disagreement allowed
SQMI_TO_KM2 = 2.58999


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge-list", required=True)
    ap.add_argument("--grdc-catalogue", required=True)
    args = ap.parse_args()

    gauge_ids = [ln.strip() for ln in Path(args.gauge_list).read_text().splitlines()
                 if ln.strip()]
    grdc_nos = {int(g.split("_")[1]): g for g in gauge_ids}
    cat = pd.read_csv(args.grdc_catalogue)
    cat = cat[cat["grdc_no"].isin(grdc_nos)]
    print(f"gauge list: {len(gauge_ids)}; found in catalogue: {len(cat)}")

    reg = json.loads(STATIONS.read_text())
    reg = reg["stations"] if isinstance(reg, dict) else reg
    ulat = np.array([s["lat"] for s in reg])
    ulon = np.array([s["lon"] for s in reg])
    uarea = np.array([(s.get("drain_area_sqmi") or np.nan) for s in reg],
                     dtype=float) * SQMI_TO_KM2
    uid = [s["id"] for s in reg]

    corpus_sids = {p.name.split(".")[0] for p in CORPUS_DIR.glob("*.csv.gz")
                   if not p.name.startswith("._")}

    matches, unmatched = {}, []
    for _, r in cat.iterrows():
        # cheap planar distance is fine at 0.05 deg scale; lon compressed by cos(lat)
        d = np.hypot(ulat - r["lat"], (ulon - r["long"]) * np.cos(np.radians(r["lat"])))
        order = np.argsort(d)[:5]
        best = None
        for j in order:
            if d[j] > MAX_DIST_DEG:
                break
            if np.isfinite(uarea[j]) and np.isfinite(r["area"]) and r["area"] > 0:
                if abs(uarea[j] - r["area"]) / r["area"] > AREA_TOL:
                    continue  # co-located but different river/size — keep looking
            best = j
            break
        gid = grdc_nos[int(r["grdc_no"])]
        if best is None:
            unmatched.append(gid)
            continue
        matches[gid] = {
            "usgs_id": uid[best],
            "dist_deg": round(float(d[best]), 5),
            "grdc_area_km2": None if pd.isna(r["area"]) else float(r["area"]),
            "usgs_area_km2": None if not np.isfinite(uarea[best]) else round(float(uarea[best]), 1),
            "in_corpus": uid[best] in corpus_sids,
        }

    in_corpus = sorted(m["usgs_id"] for m in matches.values() if m["in_corpus"])
    payload = {
        "source_gauge_list": args.gauge_list,
        "grdc_catalogue": Path(args.grdc_catalogue).name,
        "rules": {"max_dist_deg": MAX_DIST_DEG, "area_tol": AREA_TOL},
        "n_gauge_list": len(gauge_ids),
        "n_in_catalogue": int(len(cat)),
        "n_matched": len(matches),
        "n_unmatched": len(unmatched),
        "n_matched_in_corpus": len(in_corpus),
        "matches": matches,
        "unmatched": unmatched,
        "matched_in_corpus_usgs_ids": in_corpus,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"matched {len(matches)}/{len(gauge_ids)} "
          f"({len(unmatched)} unmatched); {len(in_corpus)} matches in our corpus")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
