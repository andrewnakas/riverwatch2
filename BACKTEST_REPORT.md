# NWM & NWM-Residual Backtest Report

**For:** the next model (Fable 5)
**From:** audit pass, 2026-06-09
**Question being handed off:** *Can we honestly backtest `nwm` and `nwm_residual`, and does `nwm_residual` actually beat `nwm` / the other members — or is the "scores way better" just an artifact?*

**Short answer:** Yes, a real backtest is feasible with data already in the repo. And it is **needed**, because today `nwm_residual` is never evaluated against held-out observations anywhere. Its headline MAE is computed by formula, not measured. This report tells you exactly what's wrong, what data exists, and how to build the honest backtest.

---

## ✅ STATUS UPDATE (2026-06-09, same day): backtest built, verdict in, fixes shipped

The backtest proposed below now exists (`scripts/backtest_nwm_residual.py`) and was run
on the full archive (66 issue-days, 2026-03-09 → 2026-06-09, 1,735 stations; note: the
archive has a gap 04-05 → 05-01, and `bias_scale_used` turned out to be **empty in every
row**, so the corrected baseline is reconstructed from the trailing h=1 forecast-vs-obs
window — same formula/clamp/min-overlap as `hindcast_skill`). Strict temporal split:
train issued ≤ 2026-05-16 (targets also ≤ 05-16), test issued 05-17 → 06-09. Truth =
USGS dv API, batch-fetched for all archive stations.

### Verdict on the report's pass/fail criteria (per-station **median** MAE, cfs)

| h | persistence | nwm_raw | nwm_corrected | residual (prod pickles*) | residual (clean avg) |
|---|---:|---:|---:|---:|---:|
| 1 | **137** | 304 | 263 | 245 | 159 |
| 4 | 341 | 471 | 428 | 374 | **330** |
| 7 | 450 | 542 | 493 | 417 | **396** |
| 10 | 487 | 564 | 500 | 458 | **387** |
| 14 | 435 | 571 | 589 | (no model → 589) | **366** |

\* prod pickles were trained through 2026-06-01, i.e. **leak-tainted in their favor** on this test window — and they still lose to persistence at h1–7.

- ✅ `nwm_corrected` beats `nwm_raw` at h1–13 — bias correction earns its keep.
- ❌ **The shipped `nwm_residual` was decoration**: even with leakage in its favor it never beat persistence before h8. The manifest's 0.21–0.38 ratios were pure artifact, as §1 predicted.
- ✅ **A clean retrain is real**: 3 feature variants (v1 = shipped set on honest inputs; v2 = +anchor-gap/obs-trend/local-NWM-skill; v3 = +per-(station,horizon) trailing signed residual) averaged in log space beat `nwm_corrected` at **every** horizon. True ratios: **0.60 (h1) … 0.83 (h5) … 0.62 (h14)** — modest and believable, exactly the range §3 said to expect. Win-rate vs corrected 48–80% of stations; the gain survives the flow-tercile split (see `benchmarks/nwm_backtest_v4.json`). Per-horizon variant choice was made by forward-chaining CV inside the train window only.

### What was fixed in production (v15.9)

1. **Trainer rewritten** (`scripts/train_nwm_residual.py`): `station_id` read as string (the old int-parse silently dropped every leading-zero gauge from training — that's why only 120k/1.19M rows were labeled), real `q_obs_t0` reconstructed for backfilled rows (no more `fillna(0)` skew), baseline = bias-corrected NWM (no more strawman), split by `issued_date` with target-leak guard, and the validation block restricted to issue-days with full h=14 truth. Manifest val ratios are now honest: 0.63 (h1) → ~0.9–1.0 (h11–13), median-station 0.53–0.77.
2. **Inference rewritten** (`app/nwm_residual.py`): now the **single source of truth for features** — the trainer imports its column definitions, so train/serve skew can't silently recur. Serves the v1/v2/v3 log-space ensemble; per-station trailing-skill stats ship in `sidecar.json`; declines (returns None) when the observation at issuance is missing instead of feeding the model a fake 0.
3. **`_load_resid_scale` clamp tightened** to [0.50, 1.05] (`app/forecast.py`) — a manifest claiming >50% gain is now treated as evidence of leakage, not signal. Fallback table replaced with the *measured* backtest ratios.
4. **Per-station measured NWM MAE now feeds blend weights** (v15.9 follow-up, same day): the trainer ships `holdout_stats.json` — trailing per-station per-horizon MAE of the *as-served* (bias-corrected + anchored) archived forecasts against truth — and `forecast._nwm_per_h_estimates` prefers it over the old `hindcast × 1.04h` formula. This closes most of §4.1: the `nwm` member's rolling MAE is now measured per station from its actual issued forecasts; `nwm_residual`'s is that measurement × the manifest's *served* holdout ratio (consistent anchoring on both sides).
5. **Anchoring measured, residual decay corrected**: the harness now scores members as-served (`anchor_member`). Anchor-to-observed roughly halves corrected-NWM median MAE at h1 (263→137) and helps through h7 — so `nwm` keeps `decay_h=7`. But the residual ensemble self-anchors through its features: full anchoring hurt it at h3–6, while `decay_h=2` matches persistence at h1 (159→137), improves h2, and leaves h3+ untouched (`benchmarks/nwm_backtest_v6.json`). Production now serves `nwm_residual` with `decay_h=2`.

### Still open (highest-value next steps)

- **Direct measured holdout for `nwm_residual` itself**: its per-station MAE is still measured-nwm × global served ratio. A per-station residual measurement needs the residual predictions scored over the archive window with purged (pre-window-trained) models — doable in the weekly train job.
- `benchmark_40.py` still doesn't cover NWM members; `scripts/backtest_nwm_residual.py` is the honest harness for them (a fair NWM holdout needs *archived issued* forecasts, which `benchmark_40`'s fetch-today design can't produce).
- The archive should start populating `bias_scale_used` (it never has), and a build-time job should keep `q_cfs_obs_today` flowing (live rows have it; backfilled rows don't).

Everything below this line is the original handoff analysis, kept for the record.

---

## 1. Why the current numbers can't be trusted

### 1a. `nwm_residual`'s displayed MAE is self-reported, not backtested
Every other member's "Rolling MAE" in the UI comes from a real on-station holdout (`_score_holdouts(...)` in `app/forecast.py:1326–1369`): predict past windows, compare to observed truth. **`nwm` and `nwm_residual` have no holdout predictions at all** — they're "live-only" members. So `nwm_residual`'s MAE is fabricated by formula (`app/forecast.py:1511–1517`):

```python
rolling_mae["nwm_residual"] = rolling_mae["nwm"] × resid_scale[h]
```

where `resid_scale[h]` is read straight from the **training run's own validation manifest** (`_load_resid_scale`, `app/forecast.py:117–143`). Current `data/nwm_residual_models/manifest.json` ratios:

| h | base_cfs | learn_cfs | ratio (learn/base) |
|---|---------:|----------:|-------------------:|
| 1 | 1864 | 664 | 0.36 |
| 5 | 4901 | 1862 | 0.38 |
| 7 | 9172 | 2564 | 0.28 |
| 10 | 11562 | 2456 | 0.21 |

The UI table literally multiplies NWM's error by 0.21–0.38 and prints the result next to genuinely-backtested members. **`nwm_residual` is structurally guaranteed to show ~3–5× lower MAE than `nwm` regardless of real performance**, because the "improvement" is an input constant, not a measurement. *This is the entire "scores better than all models a lot" effect.*

### 1b. The manifest ratios are themselves inflated
Three compounding problems in `scripts/train_nwm_residual.py`:

- **Train/serve feature skew.** Feature `log1p_q_obs_t0` (observed flow at issuance) is trained on backfilled rows whose `q_cfs_obs_today` is **empty → filled with 0** (`train_nwm_residual.py:129`, `.fillna(0)`). I confirmed this in the actual archive data — every backfilled row has an empty `q_cfs_obs_today`. So the model learns with that feature pinned to 0, then at inference is fed the *real* observation (`app/nwm_residual.py:89`). It was never validated under the inputs it actually runs on.
- **Strawman baseline.** Backfilled rows also have empty `bias_scale_used` → clipped to `1.0` (`train_nwm_residual.py:124`), so the manifest's `val_mae_baseline_cfs` is **raw, uncorrected NWM**. But the live `nwm` member it's compared against in the UI is *already bias-corrected*. The residual model beats a weaker baseline than the one it's displayed against.
- **Pooled, giant-river-dominated, not-truly-chronological val.** The 10% holdout (`train_nwm_residual.py:160–163`) pools all ~1,893 stations; cfs baseline MAEs of 1,800–11,500 mean a few huge rivers dominate the metric, and the split is only "roughly" chronological (`_load_archive` concatenates `sorted(rglob)` by filename, lines 72–81) so recent target dates can leak into train.

### 1c. The one honest harness doesn't test the residual
`scripts/benchmark_40.py` is truth-based (holds out the last N days, scores against observed flow) but only evaluates `persistence_lag1`, `runoff_ridge`, `chronos_bolt`, and the blend (`benchmark_40.py:154`). **Neither `nwm` nor `nwm_residual` is in it.** Nothing in the repo measures `nwm_residual` against observed flow.

**Net:** the residual correction probably helps NWM *somewhat* (correcting a process model usually does), but the magnitude shown is unearned. We don't currently know if it works.

---

## 2. The good news: the data for an honest backtest already exists

| Asset | Location | What it gives you |
|---|---|---|
| Real issued NWM forecasts | `nwm-archive` branch, `archive/YYYY/MM/*.csv.gz` | **~86 days**, 2026-03-09 → 2026-06-03, one t00z `medium_range_blend` cycle/day, h=1..10, ~1,893 stations. These are *actual operationally-issued* forecasts (forecast error, not perfect-forcing) — exactly what a backtest needs. |
| Observed truth | `data/cache/usgs_records/{id}.json` (`{date: cfs}`) | Ground-truth target flow to join on `(station, target_date)`. `_attach_targets` in `train_nwm_residual.py:93–119` already does this join — reuse it. |
| Bias-correction logic | `app/nwm.py:207` `hindcast_skill()` / `hindcast_mae()` | Computes the same multiplicative `bias_scale` the live `nwm` member uses (from analysis_assimilation overlap). Use it to make the baseline *fair*. |
| Trained residual models | `data/nwm_residual_models/h{N}.pkl` + `app/nwm_residual.py:apply_residual` | The thing under test. Apply per-horizon exactly as production does. |

This is enough for a genuine held-out backtest **on the station set you actually care about** (the ~238 whitewater gauges, not the 1,893-station pool).

---

## 3. The backtest to build

**Goal:** for each member ∈ {`nwm_raw`, `nwm_corrected` (= the live `nwm` member), `nwm_residual`, `persistence`}, produce per-horizon MAE on held-out observed flow, on the real station set, with an honest train/test time split.

**Design — strict temporal holdout (no leakage):**
1. **Split by `issued_date`.** Pick a cutoff (e.g. train on issued ≤ 2026-05-15, test on issued 2026-05-16 → 06-03). Retrain the residual models on the train side only; never let any test-period `target_date` appear in training.
2. **Build the test panel** from the archive test slice: `(issued_date, station, target_date, horizon, q_nwm_raw)`.
3. **Reconstruct each member honestly on the test panel:**
   - `nwm_raw` = `q_cfs_raw` as-is.
   - `nwm_corrected` = `q_cfs_raw × bias_scale`, where `bias_scale` is derived **only from data available at issuance** (analysis_assimilation overlap before `issued_date`, via `hindcast_skill`). Do *not* use the empty `bias_scale_used` column.
   - `nwm_residual` = `apply_residual(...)` with the **real** `q_obs_today` (observed flow on `issued_date`) — not 0. This requires fixing the `q_obs_t0` plumbing (see §4); if you backtest with the current pickles, at least feed the real obs so train and test match.
   - `persistence` = observed flow on `issued_date`, held flat (sanity-floor: the residual model must beat this or it's worthless).
4. **Join observed truth** on `(station, target_date)` from `usgs_records`.
5. **Score per horizon, per member:** report **median** MAE (the README itself notes mean is dominated by a few snowmelt giants — `README.md:107`), plus mean for completeness, plus % of station-horizons where `nwm_residual` < `nwm_corrected`. Stratify by flow tercile so a model that only helps on huge rivers can't masquerade as a global win.

**Pass/fail criteria — state them up front:**
- ✅ `nwm_corrected` beats `nwm_raw` (confirms bias correction earns its keep).
- ✅ `nwm_residual` beats `nwm_corrected` on **median** MAE on the real station set, at most horizons, by a margin that survives the tercile split. Expect something far more modest than the 0.21–0.38 ratios — if you see ratios that low on a true holdout, suspect leakage and re-check the split.
- ❌ If `nwm_residual` ≈ `nwm_corrected`, the member is decoration; either fix it or drop it from the blend.

**Suggested deliverable:** a `scripts/backtest_nwm_residual.py` that emits `benchmarks/nwm_backtest_<label>.json` with per-horizon, per-member, per-tercile MAE — mirroring `benchmark_40.py`'s output shape so it slots into the existing results trail.

---

## 4. Fixes that make the production member honest (separate from the backtest)

Even with a clean backtest, these need fixing or the *live* member stays misleading:

1. **Give `nwm_residual` (and `nwm`) a real on-station holdout** in `forecast_station`, like every other member, instead of the manifest-scaled estimate at `app/forecast.py:1511–1517`. Until then its blend weight and UI number are unearned. This is the single highest-value fix.
2. **Fix the `q_obs_t0` train/serve skew:** either backfill the real observation-at-issuance into the archive (the snapshot path `scripts/snapshot_nwm_archive.py` *could* populate `q_cfs_obs_today` going forward), or drop the feature from the model. Don't train on 0 and serve on real.
3. **Fix the manifest baseline** to compare against bias-corrected NWM, so `resid_scale` isn't structurally favorable.
4. **Add `nwm` + `nwm_residual` to `benchmark_40.py`** so the truth-based harness covers them.
5. **Report cfs MAE on the ~238-station product set**, not the 1,893-station pool.

---

## 5. Pointers (file:line)

- Fabricated residual MAE: `app/forecast.py:1502–1517`
- Manifest → scale loader: `app/forecast.py:117–153`
- Inference (real obs vs trained-on-0): `app/nwm_residual.py:68–116`
- Trainer leakage points: `scripts/train_nwm_residual.py:124` (bias=1.0), `:129` (`obs_today.fillna(0)`), `:160–163` (pooled rough split)
- Backfill leaves obs/bias empty: `scripts/backfill_nwm_archive.py:202–203`
- Honest truth-based harness (extend this): `scripts/benchmark_40.py:61–154`
- Bias-correction + hindcast to reuse: `app/nwm.py:201–249`
- Archive data: `nwm-archive` branch, `archive/2026/{03..06}/*.csv.gz` (2026-03-09 → 06-03)
- Observed truth: `data/cache/usgs_records/{id}.json`

---

*Everything above was verified against the code and the actual archive data, not inferred. The prior `AUDIT.md` covered validation rigor for the other 8 members but never caught that `nwm`/`nwm_residual` aren't backtested on-station at all — that gap is what this report exists to close.*
