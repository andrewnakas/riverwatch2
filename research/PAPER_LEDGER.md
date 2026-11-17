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
| B6 | Fleet archive with input provenance: every served gauge-lead, every member, and the age of the weather it ran on; champion/challenger promotion rule fixed before any candidate exists | found by code reading 2026-11-13: cache-only builds reuse the newest weather file however old; stacker trains `q_obs_today` on the lead-1 target (both registered as hypotheses, not yet measured) | `app/fleet_archive.py`, `PREREG_v2.md` §V16 |

## C. Failures found in live operation, and fixes

| # | Finding | Number | Fix / artifact | Status |
|---|---|---|---|---|
| C1 | Static-attribute extrapolation: tier-2 basins up to area z=+28.8; the 8 with any \|z\|>10 ran at 23× persistence MAE (lead 1) and carried ~all tier-2 error | 5,414 cfs issued vs 1,630 observed (13340000) | static envelope guard, `outside_static_envelope`; 34/130 refused; commit d8fcede | final (margin chosen post hoc — a guard, not a measurement) |
| C2 | Stacker train/serve skew: trained on raw member output, served bias-scaled + anchored output | blend median rel. bias +0.31 at lead 7 vs persistence +0.09; beat persistence on ~30% of basins | `_stacker_members_panel`, commit cde055e; `analysis/score_live_leads.py` | **pending — PREREG §LIVE-1, read window fixed in PREREG §LIVE-1 addendum: issued 11-04..11-17, leads 5–10 readable ≥ 2026-11-28** |
| C3 | "Record high/low" on a single calendar day is common by chance (≈1/(N+1) per tail) | 4.2% / 4.8% of gauges on day one; record-class gauges median ~22 yr vs ~53 | ≥10-year rule, `scripts/current_conditions.py`. Audit 2026-11-16 (live current.json, obs ≤ 14 d old, 8,212 gauges): 246 (3.0%) above record high, 352 (4.3%) below record low. Record-high gauges: 230/246 observed 11-15, concentrated in TX 68, CO 38, NM 37, UT 29, AZ 22 (a regional rain event on dry-season streams, not a stats/date bug); median q / record = 1.67; median 28 yr of record vs 40.5 for a random 150 'normal' gauges; 71 have 10–19 yr. 7 had a record of 0 cfs (dry wash), so any flow read as a record → demoted to above normal when max_va = 0. Raising the 10-yr floor was not done: it would be chosen after seeing the numbers | final |
| C4 | Scheduled builds run cache-only (`RW2_NO_FETCH=1`), so gauges never cached at bootstrap never got USGS daily statistics. A first fix (retries) was a misdiagnosis: share unchanged 13.1% → 13.0% | 1,265 / 9,697 gauges with no statistics (2026-11-03) | cold-cache fetch budget + circuit breaker + negative cache, `app/usgs_stats.py` | final: unknown share 13.0% → **4.6%** (446 / 9,697) on the first build with the fix (2026-11-04 02:24Z) |
| C5 | Threshold alerts on a 6-hourly forecast churn: forecasts cross the trigger back and forth between builds | 117 "new" transitions among 641 alerting gauges in one build | 24 h hold + 72 h repeat suppression, `scripts/alerts.py` (d74b220) | live (count transitions per build) |
| C6 | Hysteresis did not fix C5: the trigger read the 14-day peak, and days 6–14 of the blend run wet (C2). Also, events over the 50-post cap were marked alerted though never sent | 2026-11-11/12: 1,105–1,292 gauges at a level, 129–155 new transitions per build (79–105 unsent); 316 of 327 "rising" gauges peaked on day ≥6; median 14-day peak / today = 1.31 | trigger on the 5-day peak (`near_*` in current.json, NEAR_DAYS=5) for alerts and the map ring; unsent events not marked. 600-gauge sample: 6.3% → 2.3% of gauges at a level. First live build (2026-11-13, run 43327483376): 438 / 9,698 (4.5%) at a level on the 5-day rule (391 record, 47 rising); 42 new transitions, 42/42 posted | live (count transitions per build) |
| C7 | The map and alerts classed gauges on observations years old: current.json takes the last daily value whatever its date | 2026-11-16 live current.json: 1,328 / 9,698 gauges with last value > 30 d old (972 > 1 yr, back to 1986); 97 at an alert level (84 "record", 13 "rising") of 491 | classes → unknown when the last value is > 14 d old (`CLASS_MAX_OBS_AGE_D`, `scripts/current_conditions.py`); explains part of the open "391 record" question. The panel now says "not reporting" past 60 d. Dropping these gauges from the BUILD is deferred to after the §LIVE-1 read (2026-11-28): they sit in the per-shard stacker's training pool, so removing them would change served values inside the frozen window | live (count alert-level gauges next build) |
| C8 | NWS flood categories on the map: most NWS thresholds are stages, and converting them through the NWPS rating gives nonsense on tidal and low-stage segments | NWPS lookup 2026-11-16: 6,471 of 9,851 gauges have an NWS point, 4,259 a flow threshold. Raw conversion flagged 20 gauges; vs NWS's own observed category, 10 were on observations > 7 d old and 3 had thresholds of 0–13 cfs or tidal. With guards (obs ≤ 7 d, threshold > the day's p75): 9 flagged, 4 match NWS's category, 2 one step off, 2 NWS "no flooding", 1 NWS obs not current | `scripts/build_nws_flood.py` → `data/nws_flood.json` (weekly `nws-flood.yml`); `flood`/`near_flood` in current.json; map rings. Daily means, not instantaneous: a ring is not a warning | live |

## D. Pending reads (fixed dates; do not read early)

| # | Question | Earliest read | Registration |
|---|---|---|---|
| D1 | GATE 4: areal GFS vs Open-Meteo point decoder | **read 2026-11-16: STOP.** lead-1 tier-1 median ΔNSE +0.041926 [+0.023712, +0.064924], 464 basins, 31 inits — beyond the registered \|Δ\| ≤ 0.030 bound, so no switch. Re-check: both arms correctly aligned; point archive stale on ~3.6 of 30 init dates (10-14, 10-15, 10-16 part, 11-04). All-lead table ≥ 2026-11-27, reporting only; a decision needs a fresh registration (GATE 4b) | `PREREG_v2.md` GATE 4 lead-1 read; `data/gate4/score_31_lead1.log`; `analysis/gate4_forcing_check.py` |
| D2 | §LIVE-1: does the skew fix remove the wet bias / beat persistence at leads 5–10 | issued 2026-11-04..11-17; leads 5–10 readable ≥ 2026-11-28 | `PREREG_v2.md` §LIVE-1 |
| D3 | §V16-1 H-STALE: how old is the weather a scheduled build forecasts on (`issued_date − wx_fcst_start`, every gauge). Prediction: median ≥ 7 d | first three complete fleet-archive days (`analysis/score_fleet_live.py age`) | `PREREG_v2.md` §V16 |
| D4 | §V16 fleet scoreboard: served blend vs persistence, every gauge, per lead, basin-bootstrap CI | issue dates ≥ 2026-11-18 only; nightly `verify.yml` → `fleet-archive:live_skill.json` | `PREREG_v2.md` §V16-0 |

## E. Product (the system the science runs in)

9,851 gauges; 14-day forecasts rebuilt every 6 h on GitHub Actions (64 shards); flow-class map from
USGS daily statistics; NWS flood stages, rating, gage height and official forecast per gauge
(client-side, NWPS); search, deep links, favorites, PWA; push alerts on public ntfy.sh topics, posted only on upward transitions diffed against the previous deploy (`scripts/alerts.py`). Not a warning service.
