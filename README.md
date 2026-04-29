# RiverWatch2

Live, on-demand river-discharge forecasts for a 43-station benchmark subset of USGS gauges (40 picks + Yellowstone-Livingston + 2× Lochsa).

A Flask app serves a Leaflet map of all 40 sensors. Clicking any marker triggers
a fresh forecast that runs the following models against live USGS NWIS daily
discharge and Open-Meteo weather:

- `persistence_lag1` — naive baseline (yhat = last observed)
- `runoff_ridge` — Ridge regression on lagged log-discharge + day-of-year +
  rolling precip / temperature / snowfall windows. Recursive multi-step.
- `chronos_bolt` — [Amazon Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting)
  zero-shot foundation model (T5-based, ~50 MB, CPU inference). Optional but recommended.

Each member is rolling-validated on the training window and combined into an
inverse-MAE-weighted ensemble blend.

## Live demo

GitHub Pages: **https://andrewnakas.github.io/riverwatch2**

The Pages site is rebuilt every 2 hours by `.github/workflows/pages.yml` and on
every push to `main`. It runs the same forecast pipeline as the Flask app, dumps
each station's forecast to a static JSON file, and uploads `dist/` as the Pages
artifact. The frontend reads those JSONs directly — no backend.

## Quickstart

```bash
cd riverwatch2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (one-time) refresh USGS site metadata for the 43-station subset
python scripts/fetch_station_metadata.py

# serve the map UI
python -m app.server --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

First forecast for a station takes 5-30 s (cold USGS + Open-Meteo fetch + Chronos
init). Subsequent calls are cached for 30 minutes. Use the **Force refresh** button
to bypass cache.

## Benchmarking

```bash
python scripts/benchmark_40.py --label v2-h14 --eval-days 14 --horizon 14 --train-days 1095
```

Writes `benchmarks/results_<label>_<ts>.json` with per-station and aggregate MAE
for every member and the ensemble blend. Re-run with a new `--label` after each
modeling change to keep a clean diff trail.

## Project structure

```
app/
  server.py        Flask app: /, /api/stations, /api/forecast/<id>
  forecast.py      The three forecasters + ensemble blend
  usgs.py          USGS NWIS daily + instantaneous discharge with caching
  weather.py       Open-Meteo historical + forecast with caching
  templates/       index.html
  static/          app.js + styles.css
data/
  stations_40.json            Hand-picked 43-station benchmark subset
  stations_40_enriched.json   With lat/lon + drainage + elevation from USGS
  cache/                      On-disk JSON cache for USGS + Open-Meteo
benchmarks/
  results_*.json              Per-run benchmark snapshots
scripts/
  fetch_station_metadata.py   One-shot USGS site lookup for the subset
  benchmark_40.py             Full-subset MAE evaluation
  build_static_site.py        Builds dist/ for GitHub Pages deploy
```

## What the 40-station subset is

Picked from the upstream `north-america-river-watch` "mixed-corrected-cache"
benchmark, sorted by ensemble MAE ascending and capped per-state for geographic
spread (max 8 AK, 8 MT, 4 WY, 3 elsewhere). Skewed toward Mountain West +
Yellowstone + Alaska panhandle hydrology, with sentinel CONUS sites for
contrast.

## Current baseline (`benchmarks/baseline_v2_h14.json`)

14-day held-out eval window, **14-day** forecast horizon, 40/43 stations
succeeded (3 AK stations skipped: USGS hadn't reported recent enough daily
values for the held-out window):

| forecaster        | mean MAE (cfs) | median MAE (cfs) |
|-------------------|----------------|------------------|
| persistence_lag1  | 88.58          | 11.73            |
| runoff_ridge      | 114.50         | 12.37            |
| chronos_bolt      | 98.22          | 10.29            |
| **ensemble_blend**| **95.73**      | **7.93**         |

Median MAE is the more useful number — the mean is skewed by a handful of
high-discharge snowmelt stations (Lochsa, Gallatin, Big Sky) where every
forecaster has cfs error in the hundreds. On the median, the blend (7.93)
already beats every individual member.

What's new vs. v1:
- 14-day horizon (was 7)
- 3 new stations: Yellowstone-Livingston (06192500), Lochsa nr Lowell
  (13337000), Lochsa at L.S. (13336500)
- Ridge switched from recursive to **direct multi-step** (one model per
  horizon day, no compounding error)
- Training lookback bumped from 540 → 1095 days
- Chronos forecasts blended 50/50 with a per-station seasonal climatology
  ratio so it can anchor on DOY without snow forcing
- Rolling MAE for all members now computed on the full horizon, so blend
  weights compare like-for-like

(Previous 7-day baseline was `baseline_v1.json`: ensemble mean MAE 17.88
across 40 stations.)

## Roadmap toward better MAE

- [x] Baseline ensemble: persistence + ridge + Chronos-Bolt zero-shot
- [x] Direct multi-step ridge (no recursion, no compounding error)
- [x] Per-station seasonal scaling for Chronos via DOY climatology
- [ ] Try `chronos-bolt-base` (~200 MB) instead of `-small` for the foundation arm
- [ ] Add elevation-aware Open-Meteo precip + degree-day melt features
- [ ] Add SNOTEL SWE for stations that have a station within 50 km
- [ ] Per-station ensemble weights persisted across runs (warm start blend)
- [ ] Try TimesFM-2 / Apex once they have a stable PyPI release
