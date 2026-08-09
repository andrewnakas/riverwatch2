# Overnight status — 2026-08-02

## Headline

**New deployable with-q record: day-1 median NSE 0.9137** on CAMELS-531
(previous 0.9058; Nearing 2022 AR = 0.879). CI95 [0.9052, 0.9215].
Plain mean of 3 forcings × 5 seeds, nothing fitted on test observations.

Per-member gain from the 5th seed: daymet +0.0116, maurer +0.0167, nldas +0.0114.
All three moved the same way by a similar amount — real seed depth, not one lucky
seed. Artifact: `benchmarks/withq5_deployable_final.json`.

The no-q track is unchanged at **0.8357** pending the AORC member.

## What was decided and closed

**KGE variance inflation — CLOSED, does not work.** Our KGE (0.8768) trails
Nearing's 0.896 entirely through alpha = 0.940; r = 0.964 and beta = 1.002 are
near-perfect, so the gap is ensemble under-dispersion. Fitting the per-basin
inflation coefficient on 1989-94 and applying it to 1994-99 made KGE *worse*
(0.8754 → 0.8552) and NSE worse (0.9225 → 0.9035), because
corr(k_early, k_late) = **0.084** — the coefficient does not transfer across
time. The oracle 0.9406 was hindsight. Do not retry with a different window;
the transfer correlation is the blocker, not the window.

**AORC's missing basin (13235000) — CLOSED, staying at 530.** Chased this
properly: found their published method (GAGES-II polygons + `exactextract`
coverage-weighted mean, `jmframe/CIROH_DL_NextGen/forcing_prep`), implemented it,
and validated against basins they *did* publish. Three hypotheses tested and all
eliminated:

| hypothesis | test | result |
|---|---|---|
| wrong polygon source | downloaded the real GAGES-II shapefile (205 MB); 13235000 present, area 1163 km² = exact CAMELS match | ratio 0.858 → 0.860, **unchanged** |
| time convention | published = water-year, zarr = calendar-year; lag scan −6…+6 h | peaks at **lag 0** |
| aggregation convention | coverage-weighted 100.76, centre-in 100.59, median 103.77, field mean 103.0 | published = **124.6**, the 78th percentile of the local field |

Temperature (0.999) and pressure (1.033) reproduce almost exactly, so the
aggregation is right; the residual is precipitation and it is **basin-varying**
(0.911/corr 0.990 on one basin, 0.809/corr 0.753 on another). Their series draws
on cells no public boundary contains — Wood's 2024 re-delineation is unpublished.
A basin-varying, partly decorrelated precip series is the profile that quietly
degrades an ensemble, so it is not shippable.

Cost of the decision, measured not assumed: the 530-basin baseline scores
**0.9134** vs 0.9137 on 531 — a 0.0003 shift, against margins of interest of
0.004-0.012.

## What was found

**CONUS404 has the same point-sampling bias — and the aggregate hides it.**
Overall ratio against the catchment-averaged products is 1.009, which passes a
naive check. Stratified by terrain it does not:

| | ratio |
|---|---|
| corr(slope, ratio) | **−0.424** |
| corr(elevation, ratio) | −0.414 |
| flattest quartile | 1.037 |
| steepest quartile | **0.849** |

A WRF model bias would not track terrain. Impact concentrates where it matters:
among pivotal basins (steep AND snowy — the population that controls our median),
median ratio **0.932** and **46% read >10% dry**, vs 1.027 / 14% elsewhere.
89 of the 405 benchmark basins read >10% dry, median slope 86.4 vs 25.0.

**RESOLVED — re-extracting with polygons.** The decision hinged on scale vs
shape: LSTMs normalize per basin, so a multiplicative offset is largely absorbed,
but distorted storm sequencing is irreducible information loss. Measured daily
precip correlation, point vs polygon, one steep basin per HUC region:

| basin | region | cells in basin | ratio | corr |
|---|---|---|---|---|
| 01333000 | 01 | 6 | 0.915 | 0.8332 |
| 02011400 | 02 | 26 | 0.902 | 0.7975 |
| 03050000 | 03 | 29 | 0.872 | 0.9011 |
| 06188000 | 06 | **101** | 0.700 | 0.8645 |
| 07083000 | 07 | 4 | 0.714 | 0.9108 |
| 08267500 | 08 | 6 | 0.615 | 0.7792 |

**6 of 6 below the 0.95 bar**, mean 0.848, worst 0.779, mean ratio 0.786 — only
~71% of daily precipitation variance survives point sampling. Basin 06188000 has
101 grid cells inside its boundary, so the point sample stood for 1% of the
catchment. This is the same range that disqualified AORC point sampling.

Actions: stopped the extraction at 498/671 and quarantined the output as
`camels_corpus_conus404_POINTSAMPLED_DO_NOT_USE`; patched `build_conus404.py` to
build a per-basin cell mask from GAGES-II polygons and average inside the tile
block. Cost is ~zero — the block is already in memory, and the expense was always
chunk I/O (measured: 44 s CPU over 14.5 min wall, i.e. network-bound). Smoke test
confirms 3/3 basins polygon-averaged at a median of 26 cells each, vs 1 before.

Also killed `retry_failed_tiles.sh`, which was respawning workers against the old
path and would have silently mixed both methods into one corpus.

## Caught before it caused a bad result

The AORC seeds trained tonight used `scripts/train_mblstm.py` (quantile LSTM),
but the no-q 0.8357 ensemble is built entirely from **neuralhydrology CudaLSTM**
members. Dropping one into the other would have confounded "new forcing" with
"new architecture" and the forcing claim could not have been isolated.

Fix: built `gpu1080/nh_data/aorc` (530 basins, 10957 rows, diurnal range 9.3 °C,
q_mm 100% finite; the q_cfs→q_mm conversion verified numerically at ratio
1.0000000) and generated configs differing from `cfg_daymet_s111.yml` in exactly
five lines — experiment name, seed, and three data paths. `supervisor_aorc_nh.sh`
is chained behind the with-q queue.

The train_mblstm seeds are not wasted: the with-q track *is* train_mblstm, so the
with-q AORC seeds join it natively.

## Running now

| track | state |
|---|---|
| AORC no-q (train_mblstm) | 3/3 done — val 0.809, 0.813, 0.805 |
| AORC with-q (train_mblstm) | 4/4 done — val 0.818, 0.821, 0.822, 0.821 |
| AORC with-q backtest | **running** — first physical-space number for AORC |
| AORC no-q (neuralhydrology) | s111 training since 04:04, ~8 h each |
| CONUS404 polygon re-extraction | running; **~3-4 days**, not the ~17 h I first estimated |

All seven AORC seeds trained cleanly and clustered tightly — no weak optimum of
the kind that took out LSTMmulti s333.

**Important caveat on those val numbers:** they are `val_medNSE(norm-asinh)`,
i.e. normalised asinh space, and are NOT comparable to the 0.9137 record, which
is physical day-1 median NSE. The backtest now running is the first physical
number for AORC and the real gate.

**A wrong turn worth recording:** the first dump attempt used
`train_mblstm.py --epochs 0 --dump-day1` and wrote **0 rows** — that path gates
on a day-1 mask that never fired, and even on success emits `ymed` only, one
checkpoint at a time. The published members are **seed-averaged at inference**:
`app/mblstm.py` splits the checkpoint path on colons and averages forecasts
across them. So a member is ONE dump from FOUR checkpoints, not four dumps.
`backtest_aorc_withq.sh` now copies the baseline protocol verbatim.

## Ready to run the moment members land

- `score_withq_aorc.py` — scores baseline and +AORC on the **same** 530-basin
  intersection, with a bootstrap CI on the delta. Baseline already measured at
  0.9134, so the comparison is apples-to-apples rather than against a 531 number.
- `gate_eval.py` with `GATE_WITH_AORC=1` — train-side leaderboard and residual
  decorrelation; `build_merged` inner-joins, so enabling AORC drops 13235000 from
  every stream at once and the basin sets stay matched automatically.

## Honest expectation on 0.84

Seed depth saturates: the fitted curve NSE(k) = 0.8395 − 0.0156/k gives ~0.8378
at k=5, short of 0.840. Recombining existing members covers ~10.6% of the
remaining gap. **AORC is the lever that can close it**, because it adds new
information rather than re-averaging what we already have — but that is exactly
why it has to be architecture-matched and scored on a consistent basin set, which
is what tonight's work was about.

## Notes for later

- All three `mblstm_backtest_*_withq5_full531.json` are corrupt: the backtest
  printed its report to the same descriptor while the JSON dump was buffered, so
  ~30 lines of text landed inside the serialized object. Per-station data is
  recoverable by brace-matching (`repair2.py`); the CSV dumps are unaffected, so
  no headline metric ever depended on them. Read metrics from the CSVs.
- A supervisor gated on `pgrep -f "backtest_mblstm.py"` deadlocked for an hour
  because a monitoring shell's own command line matched the pattern. Never put a
  gate's pattern in another command line; inspect with
  `ps -eo args | grep "[s]cripts/train_mblstm"`.


## Correction: CONUS404 cost, and a misdiagnosis

I twice called the extraction hung, on the evidence that `/proc/PID/io` rchar sat
at 206.1 MB across a 25-30 s window with threads in `ep_poll`. A direct read
probe shows that was wrong — every read succeeds:

| slice | bytes | time | rate |
|---|---|---|---|
| 36 days | 17.6 MB | 48.2 s | 0.37 MB/s |
| 72 days | 35.3 MB | 23.7 s | 1.49 MB/s |
| 180 days | 88.2 MB | 97.7 s | 0.90 MB/s |
| 360 days | 176.4 MB | 142.0 s | 1.24 MB/s |

At ~1 MB/s a single (tile, 360-day) block of 1058 MB takes ~15 minutes to land, so
a 25-second sample cannot see progress and flat rchar is the *expected*
appearance mid-block.

**The real correction is the cost.** Per tile: 30.4 blocks × 1058 MB = 32 GB;
across 11 tiles ~**354 GB**, i.e. **69-98 hours**. My earlier ~17 h came from
extrapolating per-basin timings and was simply wrong. The original point-sampled
run moved the same bytes — polygon aggregation adds no I/O — it just never had
its total spelled out.

Not usefully reducible: the zarr chunk is (36,350,350), so narrowing the spatial
slice moves no fewer bytes. Dropping SNOW saves 17%; shortening the window saves
67% but only serves the with-q track, while CONUS404 is meant as a 4th **no-q**
forcing needing the full Li/Song span.

**Decision: let it run.** It is network-bound and competes with neither the GPU
(neuralhydrology AORC training) nor the CPU (the AORC with-q backtest), so the
elapsed time costs us nothing else.


## Correction #2: the CONUS404 stall was my own timeout

The 4-16 day range in the section above was wrong, and so was blaming the
endpoint. Measured on an idle box:

| slice | rate |
|---|---|
| 72 days | 2.04 MB/s |
| 180 days | 2.11 MB/s |
| 360 days | 2.53 MB/s |

That is **faster** than the 1.43 MB/s the original point-sampled run achieved.
The endpoint is healthy; the earlier low figures were cold-start TLS setup and a
box saturated by the AORC backtest.

**The real cause was the guard I added.** One tile is 30.4 blocks × 1058 MB =
32.2 GB, which at ~2 MB/s takes **4.5 hours**. I set `timeout 5400` — 90 minutes.
Every tile was killed about a third of the way through, and because
`build_conus404.py` only skips basins whose output *files* exist, and files are
written at the *end* of a tile, each kill threw away all the work. That is
precisely why 5.5 hours produced zero files.

Raised to `timeout 21600` (6 h). Revised estimate: **39-49 h, about 2 days.**

Lesson worth keeping: derive a watchdog threshold from the measured duration of
the work it guards. Ninety minutes felt generous against a stall; a unit of work
here takes four and a half hours.
