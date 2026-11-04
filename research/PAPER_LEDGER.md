# Paper evidence ledger

One row per claim the paper may make. Every claim names its number, the artifact
that produces it, and its caveat. A claim with no artifact does not go in the
paper. Status: **final** (read once, logged) · **live** (accumulating; read date
fixed) · **pending** (registered, not yet read).

## A. Benchmark results (CAMELS-US, MODERN-1 campaign)

| # | Claim | Number | Artifact | Caveat | Status |
|---|---|---|---|---|---|
| A1 | Day-1 median NSE, discharge assimilated, perfect forcing, 490 basins, test 2018-10..2025-09 | 0.905829 | `benchmarks/m1_FINAL_withq_corrected.json` | second test read; two earlier records retracted (next-day precip leak) | final |
| A2 | Same, no discharge (rainfall–runoff only) | 0.834558 | `benchmarks/m1_FINAL_noq_corrected.json` | second test read | final |
| A3 | Paired penalty for real 1-day GFS forecasts instead of observed weather | −0.048033 [−0.059808, −0.039718] | README; `benchmarks/m1_decay_day1_nldasm9.json` | larger than every modelling gain in the campaign | final |
| A4 | With-q skill falls 0.0557 lead 1→14; no-q skill *rises* 0.0110 | — | `benchmarks/m1_{withq,noq}_allleads_test14_FINAL.json` | the rise is unexplained | final |
| A5 | Out-of-cohort transfer (GATE 3): 130 untrained basins, val1 median NSE | 0.854214 [0.838676, 0.895274] | `benchmarks/m1_gate3_verdict.json` | median hides a tail — see C1 | final |
| A6 | Neighbour transfer by drainage-area ratio, 322 gauge pairs, median NSE within 2× area | 0.8215 | `benchmarks/river_transfer_study.json` | mean −2.16; needs a gauge on the same river | final |
| A7 | NWM v3 medium-range head-to-head, 229 stations, real 2026 issuances | beats bias-corrected NWM at all 14 leads | `benchmarks/NWM_HEADTOHEAD.md` | backtest, not the live archive | final |

## B. Method innovations (the paper's "how")

| # | Practice | Evidence it mattered | Where |
|---|---|---|---|
| B1 | Pre-registered gates with decision tables fixed before numbers; append-only test-read log incl. retractions | two records retracted; one self-verification withdrawn | `PREREG_v2.md`, `benchmarks/m1_test_reads.log` |
| B2 | Paired per-basin deltas with bootstrap CIs, never differences of medians | difference of medians overstated effects >10×, once by 1.9× | `scripts/score_m1_live.py` docstring |
| B3 | Score every day, never a stride subsample | stride-14 frame flattered results by +0.036 | README |
| B4 | Live verification archive: what was *served*, joined to what the river did | found C1 and C2 below | `origin/m1-archive`, `scripts/snapshot_m1_archive.py` |
| B5 | Refuse rather than serve the training mean (forcing gaps, short decoders, degenerate histories) | 10-of-16-lead table silently served leads 11–16 at training mean (2026-10-16) | `app/mblstm_modern.py` |

## C. Failures found in live operation, and fixes

| # | Finding | Number | Fix / artifact | Status |
|---|---|---|---|---|
| C1 | Static-attribute extrapolation: tier-2 basins up to area z=+28.8; the 8 with any \|z\|>10 ran at 23× persistence MAE (lead 1) and carried ~all tier-2 error | 5,414 cfs issued vs 1,630 observed (13340000) | static envelope guard, `outside_static_envelope`; 34/130 refused; commit d8fcede | final (margin chosen post hoc — a guard, not a measurement) |
| C2 | Stacker train/serve skew: trained on raw member output, served bias-scaled + anchored output | blend median rel. bias +0.31 at lead 7 vs persistence +0.09; beat persistence on ~30% of basins | `_stacker_members_panel`, commit cde055e; `analysis/score_live_leads.py` | **pending — PREREG §LIVE-1, read window fixed in PREREG §LIVE-1 addendum: issued 11-04..11-17, leads 5–10 readable ≥ 2026-11-28** |
| C3 | "Record high/low" on a single calendar day is common by chance (≈1/(N+1) per tail) | 4.2% / 4.8% of gauges on day one; record-class gauges median ~22 yr vs ~53 | ≥10-year rule, `scripts/current_conditions.py` | final |
| C4 | Scheduled builds run cache-only (`RW2_NO_FETCH=1`), so gauges never cached at bootstrap never got USGS daily statistics. A first fix (retries) was a misdiagnosis: share unchanged 13.1% → 13.0% | 1,265 / 9,697 gauges with no statistics (2026-11-03) | cold-cache fetch budget + circuit breaker + negative cache, `app/usgs_stats.py` | final: unknown share 13.0% → **4.6%** (446 / 9,697) on the first build with the fix (2026-11-04 02:24Z) |
| C5 | Threshold alerts on a 6-hourly forecast churn: forecasts cross the trigger back and forth between builds | 117 "new" transitions among 641 alerting gauges in one build | 24 h hold + 72 h repeat suppression, `scripts/alerts.py` (d74b220) | live (count transitions per build) |

## D. Pending reads (fixed dates; do not read early)

| # | Question | Earliest read | Registration |
|---|---|---|---|
| D1 | GATE 4: areal GFS vs Open-Meteo point decoder | after the 2026-11-10 build (≥30 archived init dates) | `PREREG_v2.md` |
| D2 | §LIVE-1: does the skew fix remove the wet bias / beat persistence at leads 5–10 | issued 2026-11-04..11-17; leads 5–10 readable ≥ 2026-11-28 | `PREREG_v2.md` §LIVE-1 |

## E. Product (the system the science runs in)

9,851 gauges; 14-day forecasts rebuilt every 6 h on GitHub Actions (64 shards); flow-class map from
USGS daily statistics; NWS flood stages, rating, gage height and official forecast per gauge
(client-side, NWPS); search, deep links, favorites, PWA; push alerts on public ntfy.sh topics, posted only on upward transitions diffed against the previous deploy (`scripts/alerts.py`). Not a warning service.
