#!/usr/bin/env python3
"""Ledger 45 / Track 4: MEASURED per-gauge streamflow-observation sigma table.

Builds, for the 531 benchmark CAMELS basins, a per-gauging pairs table from USGS
field measurements (manual discharge gaugings) and derives per-station relative
discharge uncertainty (sigma) by flow band, two ways:

  1. RATING-BASED (convention, NOT measured): USGS quality rating of each
     gauging mapped to conventional relative standard errors:
         Excellent -> 0.02, Good -> 0.05, Fair -> 0.08, Poor -> 0.12
     ("Excellent ~2%, Good ~5%, Fair ~8%, Poor >8%" is the USGS convention;
      0.12 for Poor is our stated choice for ">8%". "Unspecified" is excluded.)
     Per band we report the RMS of the mapped sigmas.

  2. EMPIRICAL (measured): leave-one-out scatter of measured discharge about a
     locally-fitted (in time) stage-discharge rating curve per site.
     For each gauging with a paired mean gage height, we fit a polynomial of
     log(Q) on gage height to the OTHER gaugings within +/- RATING_WINDOW_YEARS
     years (ice-affected gaugings excluded from fits), predict the held-out
     gauging, and take the relative residual  Q/Qhat - 1.  Per band the sigma is
     the robust std (1.4826*MAD) of those relative residuals.
     CAVEAT: this scatter includes rating-curve shift within the time window and
     measurement hysteresis, so it is an UPPER BOUND on pure measurement error;
     but unlike route 1 it is genuinely measured, not a lookup.

Flow bands: per-site terciles of in-window measured discharge (low/mid/high).
Window: 1980-01-01 .. 2008-12-31 (the benchmark window; Maurer ends 2008).

Data source (the legacy /nwis/measurements RDB endpoint is decommissioned):
  https://api.waterdata.usgs.gov/ogcapi/v0/collections/field-measurements/items
    parameter_code 00060 / reading_type "Discharge"      -> measured Q (ft^3/s)
       with per-gauging quality in "measurement_rated"
    parameter_code 00065 / reading_type "MeanGageHeight" -> stage (ft),
       joined on field_visit_id.

Outputs:
  data/usgs_field_meas/USGS-<site>.json    raw per-site cache (merged pages)
  data/usgs_field_meas/pairs_table.csv.gz  one row per discharge gauging
  analysis/ledger45_gauge_sigma.json       per-station sigmas + summary

Usage:  python3 analysis/ledger45_build_gauge_sigma.py [--fetch-only|--derive-only]
Idempotent: cached sites are not re-fetched.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "data", "usgs_field_meas")
PAIRS_PATH = os.path.join(CACHE_DIR, "pairs_table.csv.gz")
OUT_JSON = os.path.join(ROOT, "analysis", "ledger45_gauge_sigma.json")
BASIN_SRC = os.path.join(ROOT, "analysis", "noq_test_result.json")
SUMMARY_CSV = os.path.join(ROOT, "analysis", "field_measurements.csv")

API = ("https://api.waterdata.usgs.gov/ogcapi/v0/collections/"
       "field-measurements/items")
PAGE_LIMIT = 10000
MAX_WORKERS = 1          # API budget is 1000 req/hour; single file
RETRIES = 6              # for non-throttle errors
THROTTLE_MAX_WAIT = 7200  # per page, seconds of cumulative 429 backoff
POLITE_DELAY = 4.2   # 1000 req/hour budget => >=3.6s; 1.0 was 3.6x OVER and got us hard-throttled
WINDOW = (date(1980, 1, 1), date(2008, 12, 31))
RATING_WINDOW_YEARS = 5.0
MIN_NEIGHBORS = 15          # min gaugings in local time window to fit a curve
MIN_BAND_N = 10             # min residuals/ratings per band to report a sigma
RATING_MAP = {"Excellent": 0.02, "Good": 0.05, "Fair": 0.08, "Poor": 0.12}

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------- fetch phase

def load_basins():
    with open(BASIN_SRC) as f:
        d = json.load(f)
    basins = sorted(d["per_basin_nse"].keys())
    assert len(basins) == 531, f"expected 531 basins, got {len(basins)}"
    return basins


def validate_page(obj):
    """USGS returns HTTP 200 with error bodies; validate structure, not status."""
    if not isinstance(obj, dict) or obj.get("type") != "FeatureCollection":
        raise ValueError(f"not a FeatureCollection: {str(obj)[:200]}")
    feats = obj.get("features")
    if not isinstance(feats, list):
        raise ValueError("features missing")
    for f in feats[:5]:
        p = f.get("properties", {})
        if "parameter_code" not in p or "value" not in p:
            raise ValueError(f"malformed feature: {str(f)[:200]}")
    return feats


def fetch_site(site):
    """Fetch all field-measurement features for one site, paginating."""
    path = os.path.join(CACHE_DIR, f"USGS-{site}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                cached = json.load(f)
            if cached.get("complete"):
                return site, len(cached["features"]), True
        except Exception:
            pass  # corrupt cache -> refetch
    sess = requests.Session()
    sess.headers["User-Agent"] = "riverwatch2-ledger45 (research; contact via USGS WDFN feedback)"
    features, offset = [], 0
    while True:
        params = {"monitoring_location_id": f"USGS-{site}", "f": "json",
                  "limit": PAGE_LIMIT, "offset": offset}
        last_err = None
        attempt = 0
        throttled = 0.0
        feats = None
        while True:
            try:
                r = sess.get(API, params=params, timeout=180)
                if r.status_code in (429, 502, 503, 504):
                    # throttled: back off hard, honour Retry-After, don't
                    # count against the ordinary retry budget
                    ra = r.headers.get("Retry-After")
                    try:
                        wait = max(float(ra), 15.0) if ra else 0.0
                    except ValueError:
                        wait = 0.0
                    wait = wait or min(30.0 * (1 + throttled / 60.0), 120.0)
                    wait = min(wait + 5.0, 1800.0)  # cap a huge Retry-After
                    throttled += wait
                    if throttled > THROTTLE_MAX_WAIT:
                        raise RuntimeError(
                            f"{site}: throttled >{THROTTLE_MAX_WAIT}s")
                    time.sleep(wait)
                    continue
                if r.status_code != 200:
                    raise ValueError(f"HTTP {r.status_code}: {r.text[:200]}")
                page = r.json()
                feats = validate_page(page)
                break
            except RuntimeError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                attempt += 1
                if attempt >= RETRIES:
                    raise RuntimeError(
                        f"{site}: failed after {RETRIES} tries: {last_err}")
                time.sleep(5.0 * attempt)
        features.extend(feats)
        if len(feats) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        if offset > 200000:
            raise RuntimeError(f"{site}: runaway pagination")
    out = {"site": site, "fetched_utc": datetime.now(timezone.utc).isoformat(),
           "source": API, "complete": True, "n_features": len(features),
           "features": features}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    time.sleep(POLITE_DELAY)  # politeness
    return site, len(features), False


def run_fetch(basins):
    os.makedirs(CACHE_DIR, exist_ok=True)
    done = 0
    counts = {}
    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(fetch_site, s): s for s in basins}
        for fut in as_completed(futs):
            try:
                site, n, cached = fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((futs[fut], str(e)[:200]))
                print(f"  FAILED {futs[fut]}: {str(e)[:200]}", flush=True)
                done += 1
                continue
            counts[site] = n
            done += 1
            if done % 25 == 0 or done == len(basins):
                print(f"  fetched {done}/{len(basins)} (last {site}: "
                      f"{n} features{' cached' if cached else ''})", flush=True)
    empty = [s for s, n in counts.items() if n == 0]
    print(f"fetch complete: {len(counts)} sites ok, "
          f"{sum(counts.values())} features, "
          f"{len(empty)} sites with ZERO features: {empty}")
    if failed:
        print(f"FETCH FAILURES ({len(failed)}): {failed}")
    return counts


# ------------------------------------------------------------ pairs table

def parse_features(site):
    with open(os.path.join(CACHE_DIR, f"USGS-{site}.json")) as f:
        d = json.load(f)
    q_rows, gh_by_visit = [], {}
    bad_unit_q = bad_unit_gh = 0
    for feat in d["features"]:
        p = feat["properties"]
        pc, rt = p.get("parameter_code"), p.get("reading_type")
        val = p.get("value")
        if val in (None, ""):
            continue
        try:
            v = float(val)
        except ValueError:
            continue
        if pc == "00060" and rt == "Discharge":
            if p.get("unit_of_measure") not in ("ft^3/s", "cfs"):
                bad_unit_q += 1
                continue
            q_rows.append(p | {"_q": v})
        elif pc == "00065" and rt == "MeanGageHeight":
            if p.get("unit_of_measure") != "ft":
                bad_unit_gh += 1
                continue
            gh_by_visit.setdefault(p.get("field_visit_id"), []).append(v)
    rows = []
    for p in q_rows:
        t = p.get("time")
        try:
            dt = datetime.fromisoformat(t)
        except (TypeError, ValueError):
            continue
        ghs = gh_by_visit.get(p.get("field_visit_id"), [])
        gh = float(np.mean(ghs)) if ghs else math.nan
        cc = p.get("control_condition") or ""
        rows.append({
            "site_no": site,
            "time": dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "date": dt.strftime("%Y-%m-%d"),
            "q_cfs": p["_q"],
            "gage_height_ft": gh,
            "rated": p.get("measurement_rated") or "Unspecified",
            "control_condition": cc,
            "ice_affected": int("ice" in cc.lower()),
            "procedure": p.get("observing_procedure") or "",
            "field_visit_id": p.get("field_visit_id") or "",
            "in_window": int(WINDOW[0] <= dt.date() <= WINDOW[1]),
        })
    return rows, bad_unit_q, bad_unit_gh


def build_pairs(basins):
    import csv
    all_rows = []
    buq = bug = 0
    missing = []
    for i, s in enumerate(basins):
        if not os.path.exists(os.path.join(CACHE_DIR, f"USGS-{s}.json")):
            missing.append(s)
            continue
        rows, bq, bg = parse_features(s)
        buq += bq
        bug += bg
        all_rows.extend(rows)
        if (i + 1) % 100 == 0:
            print(f"  parsed {i+1}/531 sites, {len(all_rows)} gaugings so far",
                  flush=True)
    all_rows.sort(key=lambda r: (r["site_no"], r["time"]))
    cols = ["site_no", "time", "date", "q_cfs", "gage_height_ft", "rated",
            "control_condition", "ice_affected", "procedure",
            "field_visit_id", "in_window"]
    with gzip.open(PAIRS_PATH, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"pairs table: {len(all_rows)} gaugings "
          f"({sum(r['in_window'] for r in all_rows)} in 1980-2008 window) "
          f"-> {PAIRS_PATH}")
    if buq or bug:
        print(f"  NOTE: skipped {buq} discharge / {bug} gage-height rows with "
              f"unexpected units")
    if missing:
        print(f"  WARNING: {len(missing)} sites have NO cache file (fetch "
              f"failed or never ran): {missing}")
    return all_rows


# ------------------------------------------------------------ sigma derivation

def robust_std(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def loo_residuals(times, gh, logq):
    """Leave-one-out relative residuals about a local-in-time rating curve."""
    n = len(times)
    resid = np.full(n, np.nan)
    win = RATING_WINDOW_YEARS * 365.25 * 86400.0
    for i in range(n):
        m = (np.abs(times - times[i]) <= win)
        m[i] = False
        if m.sum() < MIN_NEIGHBORS:
            continue
        g, lq = gh[m], logq[m]
        # no extrapolation beyond the neighbors' stage range
        if not (g.min() - 1e-9 <= gh[i] <= g.max() + 1e-9):
            continue
        deg = 3 if m.sum() >= 40 else 2
        g0, gs = g.mean(), max(g.std(), 1e-6)
        try:
            coef = np.polyfit((g - g0) / gs, lq, deg)
        except Exception:  # noqa: BLE001
            continue
        pred = np.polyval(coef, (gh[i] - g0) / gs)
        resid[i] = math.exp(logq[i] - pred) - 1.0  # relative residual Q/Qhat-1
    return resid


def derive_sigma(all_rows, basins, fetch_counts):
    by_site = {}
    for r in all_rows:
        by_site.setdefault(r["site_no"], []).append(r)

    stations = {}
    band_names = ["low", "mid", "high"]
    coll_rating = {b: [] for b in band_names}
    coll_emp = {b: [] for b in band_names}
    n_insufficient_rating = n_insufficient_emp = 0
    total_resid = 0
    pooled_resid = {b: [] for b in band_names}   # every residual, all stations
    n_ice_window = 0

    for site in basins:
        rows = by_site.get(site, [])
        wrows = [r for r in rows if r["in_window"] and r["q_cfs"] > 0]
        rec = {
            "n_meas_total": len(rows),
            "n_meas_window": len(wrows),
            "date_first": rows[0]["date"] if rows else None,
            "date_last": rows[-1]["date"] if rows else None,
            "window_date_first": wrows[0]["date"] if wrows else None,
            "window_date_last": wrows[-1]["date"] if wrows else None,
            "rating_counts": {},
            "q_terciles_cfs": None,
            "sigma_rating": {b: None for b in band_names},
            "n_rating": {b: 0 for b in band_names},
            "sigma_empirical": {b: None for b in band_names},
            "n_empirical": {b: 0 for b in band_names},
            "flags": [],
        }
        for r in wrows:
            rec["rating_counts"][r["rated"]] = \
                rec["rating_counts"].get(r["rated"], 0) + 1
        rec["n_ice_affected_window"] = sum(r["ice_affected"] for r in wrows)
        rec["n_gh_paired_window"] = sum(
            0 if math.isnan(r["gage_height_ft"]) else 1 for r in wrows)
        n_ice_window += rec["n_ice_affected_window"]
        if len(wrows) < MIN_BAND_N:
            rec["flags"].append("insufficient_measurements")
            n_insufficient_rating += 1
            n_insufficient_emp += 1
            stations[site] = rec
            continue

        q = np.array([r["q_cfs"] for r in wrows])
        t1, t2 = np.quantile(q, [1 / 3, 2 / 3])
        rec["q_terciles_cfs"] = [float(t1), float(t2)]
        band = np.where(q <= t1, 0, np.where(q <= t2, 1, 2))

        # --- rating-based (convention)
        sig_map = np.array([RATING_MAP.get(r["rated"], np.nan) for r in wrows])
        any_rating = False
        for bi, b in enumerate(band_names):
            s = sig_map[(band == bi) & ~np.isnan(sig_map)]
            rec["n_rating"][b] = int(len(s))
            if len(s) >= MIN_BAND_N:
                rec["sigma_rating"][b] = float(np.sqrt(np.mean(s ** 2)))
                coll_rating[b].append(rec["sigma_rating"][b])
                any_rating = True
        if not any_rating:
            rec["flags"].append("no_rating_band")
            n_insufficient_rating += 1

        # --- empirical rating-curve scatter (measured)
        fit_rows = [r for r in wrows
                    if not math.isnan(r["gage_height_ft"])
                    and not r["ice_affected"]]
        rec["n_pairs_gh_window"] = len(fit_rows)
        any_emp = False
        if len(fit_rows) >= MIN_NEIGHBORS + 1:
            tt = np.array([datetime.fromisoformat(r["time"]).timestamp()
                           for r in fit_rows])
            gh = np.array([r["gage_height_ft"] for r in fit_rows])
            lq = np.log(np.array([r["q_cfs"] for r in fit_rows]))
            resid = loo_residuals(tt, gh, lq)
            qf = np.exp(lq)
            bandf = np.where(qf <= t1, 0, np.where(qf <= t2, 1, 2))
            ok = ~np.isnan(resid)
            total_resid += int(ok.sum())
            for bi, b in enumerate(band_names):
                rr = resid[ok & (bandf == bi)]
                rec["n_empirical"][b] = int(len(rr))
                pooled_resid[b].extend(rr.tolist())
                if len(rr) >= MIN_BAND_N:
                    rec["sigma_empirical"][b] = robust_std(rr)
                    coll_emp[b].append(rec["sigma_empirical"][b])
                    any_emp = True
        if not any_emp:
            rec["flags"].append("no_empirical_band")
            n_insufficient_emp += 1
        stations[site] = rec

    def med(v):
        return float(np.median(v)) if v else None

    def qtl(v, p):
        return float(np.quantile(v, p)) if v else None

    summary = {
        "n_stations": len(basins),
        "n_stations_with_any_gaugings": sum(
            1 for s in basins if stations[s]["n_meas_total"] > 0),
        "n_stations_with_window_gaugings": sum(
            1 for s in basins if stations[s]["n_meas_window"] > 0),
        "n_gaugings_total_all_time": sum(
            stations[s]["n_meas_total"] for s in basins),
        "n_gaugings_in_window": sum(
            stations[s]["n_meas_window"] for s in basins),
        "n_loo_residuals": total_resid,
        "n_stations_insufficient_rating": n_insufficient_rating,
        "n_stations_insufficient_empirical": n_insufficient_emp,
        "n_stations_rating_ok_per_band": {
            b: len(coll_rating[b]) for b in band_names},
        "n_stations_empirical_ok_per_band": {
            b: len(coll_emp[b]) for b in band_names},
        "median_sigma_rating": {b: med(coll_rating[b]) for b in band_names},
        "median_sigma_empirical": {b: med(coll_emp[b]) for b in band_names},
        "iqr_sigma_empirical": {
            b: [qtl(coll_emp[b], .25), qtl(coll_emp[b], .75)]
            for b in band_names},
        "iqr_sigma_rating": {
            b: [qtl(coll_rating[b], .25), qtl(coll_rating[b], .75)]
            for b in band_names},
        "pooled_sigma_empirical": {
            b: (robust_std(pooled_resid[b]) if len(pooled_resid[b]) >= 100
                else None) for b in band_names},
        "n_pooled_residuals": {b: len(pooled_resid[b]) for b in band_names},
        "n_ice_affected_in_window": n_ice_window,
        "assumed_sigma_ledger42_central": {"low": 0.30, "mid": 0.18,
                                           "high": 0.35},
        "assumed_sigma_ledger42_optimistic": {"low": 0.25, "mid": 0.13,
                                              "high": 0.13},
    }

    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "builder": "analysis/ledger45_build_gauge_sigma.py",
        "source_api": API,
        "window": [WINDOW[0].isoformat(), WINDOW[1].isoformat()],
        "flow_bands": "per-site terciles of in-window measured discharge",
        "rating_sigma_map_CONVENTION_NOT_MEASURED": RATING_MAP,
        "rating_map_note": ("USGS convention: Excellent ~2%, Good ~5%, Fair "
                            "~8%, Poor >8%; we use 0.12 for Poor. "
                            "'Unspecified' excluded. Per-band value is the RMS "
                            "of mapped sigmas."),
        "empirical_sigma_note": (
            "MEASURED: robust std (1.4826*MAD) of leave-one-out relative "
            "residuals Q/Qhat-1 about a local-in-time rating curve "
            f"(poly deg 2-3 of logQ on stage, fitted to non-ice gaugings "
            f"within +/-{RATING_WINDOW_YEARS} yr, >= {MIN_NEIGHBORS} "
            "neighbors, no stage extrapolation). Includes rating shift and "
            "hysteresis within the window, so it is an UPPER BOUND on pure "
            "measurement error."),
        "min_band_n": MIN_BAND_N,
        "fetch_feature_counts_nonzero_sites": sum(
            1 for v in (fetch_counts or {}).values() if v > 0)
        if fetch_counts else None,
    }
    out = {"meta": meta, "summary": summary, "stations": stations}
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    os.replace(tmp, OUT_JSON)
    md5 = hashlib.md5(open(OUT_JSON, "rb").read()).hexdigest()
    print(f"wrote {OUT_JSON}  md5={md5}")
    print(json.dumps(summary, indent=1))
    return out


def cross_check(stations):
    """Compare against the pre-existing per-station summary (counts only)."""
    if not os.path.exists(SUMMARY_CSV):
        return
    import csv
    diffs = []
    with open(SUMMARY_CSV) as f:
        for row in csv.DictReader(f):
            s = row["station_id"]
            if s in stations:
                old = int(row["n_meas"])
                new = stations[s]["n_meas_window"]
                if old and abs(new - old) / max(old, 1) > 0.10:
                    diffs.append((s, old, new))
    print(f"cross-check vs analysis/field_measurements.csv: "
          f"{len(diffs)} stations differ by >10% in in-window count")
    for s, o, n in diffs[:15]:
        print(f"  {s}: old n_meas={o} new in-window={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--derive-only", action="store_true",
                    help="skip fetch; use existing cache")
    args = ap.parse_args()
    basins = load_basins()
    print(f"{len(basins)} basins from {BASIN_SRC}")
    counts = None
    if not args.derive_only:
        counts = run_fetch(basins)
        if args.fetch_only:
            return
    rows = build_pairs(basins)
    out = derive_sigma(rows, basins, counts)
    cross_check(out["stations"])


if __name__ == "__main__":
    sys.exit(main())
