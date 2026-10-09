# RiverWatch2

Live 14-day discharge forecasts for **9,851 active USGS stream gauges** across the
US (`data/stations_v15.json`), with deep coverage of paddler-favoured runs — Pacific
NW, Cascades, Sierra, Mountain West, AK panhandle, Appalachia, Texas Hill Country,
NE / Driftless.

A Leaflet map shows every gauge. **Clicking a marker** gives that gauge's ensemble
forecast; **clicking anywhere near a river** snaps to the NHD flowline via USGS NLDI,
delineates the upstream basin, and estimates the runoff there from the nearest
comparable gauge by drainage-area ratio (reported in mm/day as well as cfs, and
labelled with the donor and the measured transfer skill).

The ensemble combines the following members against live USGS NWIS daily discharge
and Open-Meteo weather:

- `persistence_lag1` — naive baseline (yhat = last observed)
- `runoff_ridge` — Ridge regression on lagged log-discharge + day-of-year +
  rolling precip / temperature / snowfall windows. Recursive multi-step.
- `chronos_bolt` — [Amazon Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting)
  zero-shot foundation model (T5-based, ~50 MB, CPU inference). Optional but recommended.

Each member is rolling-validated on the training window and combined into an
inverse-MAE-weighted ensemble blend.

## 🏆 Streamflow benchmark — the research track

Beyond the live app, this repo hosts a rainfall–runoff / streamflow-forecasting
research effort on the standard **CAMELS-US** benchmark. The current results are
from the **MODERN-1** campaign (closed 2026-10-08), trained on 2005-10-01…2014-09-30
and read once on a held-out test window of **2018-10-01…2025-09-30** across a
**490-basin** cohort (1,243,864 rows, 5 seeds per member, equal-weight mean).

**The number to use for anything operational is ≈0.86**, not the benchmark figure.

| setting | day-1 median NSE | forcing |
|---|---|---|
| with observed discharge assimilated | **0.905829** | perfect (the weather that actually occurred) |
| without observed discharge (rainfall–runoff only) | **0.834558** | perfect |
| **with discharge, under real 1-day-ahead GFS forecasts** | **≈0.858** | **real forecast** |

The gap between the second-to-last and last rows is the part almost nobody
publishes: a measured paired penalty of **−0.048033** (CI [−0.059808, −0.039718])
for using a real weather forecast instead of the observed weather. It is larger
than every modelling improvement made during the campaign.

Full 14-lead tables for both settings are on the evidence page, and they decay in
**opposite directions** — with discharge assimilated skill falls 0.0557 from lead 1
to lead 14, without it skill *rises* 0.0110. That rise is measured and remains
**unexplained**.

### Clicking a point with no gauge

The map also answers "what's the runoff *here*" for a point with no gauge on it. There is
no model run at that point: USGS NLDI snaps the click to a mapped NHD channel, delineates
the upstream basin, and the nearest comparable gauge's hydrograph is rescaled by the
**drainage-area ratio**. The headline is runoff depth in **mm/day**, which is the quantity
that transfers between catchments of different size.

That transfer was measured by doing exactly it between **322 real gauge pairs** on the same
mainstem (donor's *observed* flow rescaled to the target, scored against the target's own
record, 2021-2025):

| area ratio within | pairs | median NSE | frac > 0.5 |
|---|---|---|---|
| 1.25× | 73 | **0.8201** | 0.753 |
| 2× | 156 | **0.8215** | 0.744 |
| 4× | 251 | 0.7708 | 0.661 |
| 10× | 322 | 0.7044 | 0.606 |

The panel always names the donor gauge, the area ratio and the applicable number; beyond
10× area mismatch it reports nothing rather than extrapolating. Two honest caveats: the
**mean is −2.16** against a median of 0.70, so a minority of pairs transfer badly; and this
is *neighbour-transfer* skill, not ungauged-basin model skill — it needs a gauge on the same
river. Artifacts: `data/river_transfer_skill.json` (what the UI reads),
`benchmarks/river_transfer_study.json` (all 322 pairs),
`analysis/measure_area_ratio_transfer.py`.

### ⚠️ What these numbers do not claim

- **0.905829 is not operational skill.** It is perfect-forcing. See the row above.
- **Neither figure is a first, virgin held-out read.** Both are *second* test reads.
  `benchmarks/m1_test_reads.log` is append-only and records the earlier ones,
  including **two retracted records** (0.910438 and 0.843090, withdrawn after an
  audit found their members were reading precipitation for the *next* day at the
  step being scored) and one **withdrawn verification** of my own that had
  reported "no leak" on a case built to contain one.
- **Not comparable like-for-like to Nearing et al. 2022 (0.879) or to this repo's
  own legacy 0.836289.** Different decade, cohort and forcing-product set. Context only.
- **Not "four independent forcing products"** — independent for precipitation and
  temperature only. gridMET carries NLDAS-2's radiation and humidity (srad slope
  0.99941, r 0.9996) and nClimGrid's radiation/vapour pressure are byte-identical
  to gridMET's.
- **No trimmed-mean figure applies here.** A trimmed mean is undefined for a
  three-member roster and silently becomes the median (~+0.0016 higher). The record
  is the equal-weight mean.

**Evidence page:** [`benchmarks.html`](https://andrewnakas.github.io/riverwatch2/benchmarks.html) ·
**experiment log:** [`benchmarks/EXPERIMENTS.md`](benchmarks/EXPERIMENTS.md) ·
artifacts `benchmarks/m1_FINAL_{withq,noq}_corrected.json`,
`m1_{withq,noq}_allleads_test14_FINAL.json`, `m1_decay_day1_nldasm9.json`, plus a
nine-vector leakage audit under `analysis/m1_leak_*.py`.

Key infra: `scripts/train_mblstm.py` (the member architecture),
`analysis/l51_withq_score.py` (scoring, with a provenance guard that refuses a dump
whose checkpoint does not match the named protocol), `app/mblstm.py` (serving),
`app/hbv.py` / `app/dhbv.py` (the differentiable-HBV hybrid).

### Method discipline

Every difference quoted is a **paired per-basin** delta with a bootstrap interval —
a difference of medians has overstated an effect in this campaign more than ten
times. Selection happens on a validation window only; the test window is read once
and every read is logged, including retracted ones. Scores use every available day
(a 14-day-stride evaluation frame flattered results by +0.036). Nothing is admitted
whose confidence interval includes zero.

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

**NWM head-to-head:** [benchmarks/NWM_HEADTOHEAD.md](benchmarks/NWM_HEADTOHEAD.md)
— the first public per-lead NSE/KGE/MAE table for operationally-issued NWM v3
medium-range forecasts at USGS gauges, measured against this system on real
2026 issuances (229 stations). RiverWatch2 beats bias-corrected NWM at all
14 leads; NWM's median NSE goes negative beyond day 4.

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
