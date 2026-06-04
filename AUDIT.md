# RiverWatch2 Forecasting Audit (2026-06-03)

Audit of the river-flow forecasting system for **accuracy** and
**production-readiness**, with remediation. The system at audit time was
**v15.11**: a 9-member ensemble (persistence, runoff_ridge, chronos_bolt, ttm,
timesfm, timesfm_xreg, ealstm, nwm, nwm_residual, lgbm_pooled) with anchored
bias correction, a LightGBM stacker meta-learner, per-horizon-bucket blend-rule
selection, and split-conformal 90% bands. It builds 9,851 gauges across 64
GitHub-Actions shards every 6h and deploys static JSON to GitHub Pages.

**Overall:** the ML/feature layer is strong and well-documented. The real risks
to *forecast accuracy* were in **validation rigor** and **silent failure
modes** — places where the system emitted a confident number that was quietly
wrong. All findings below were verified against the code, not inferred.

Severity legend: **HIGH / MEDIUM**, tagged ACCURACY and/or OPS.

---

## Findings & remediation

### A — Backtest windows never left the recent regime · HIGH · ACCURACY · FIXED
`_FOUNDATION_HOLDOUT_OFFSETS = (0,30,60,90,150,240)` — every holdout window
ended within the last ~254 days. Inverse-MAE² blend weights, per-bucket
rule selection, and conformal band widths were therefore *all* calibrated on
the last ~8 months. A station whose recent months were low-flow got
weights/bands tuned for low flow, then issued a spring-melt forecast with
weights never validated on a rising limb.
**Fix (Phase 3):** extended the offsets with 365/545/730-day out-of-regime
windows so validation spans multiple seasons and flow regimes; added one older
window (365) to the ridge subset. Every consumer iterates the same tuple, so
weights, rules, bands, and the stacker all benefit. Each access is guarded, so
short-record gauges are unaffected. *Tradeoff:* foundation holdout evals go
6→9 (~1.5×) for the slowest members; warm builds lengthen modestly.

### B — Stale-gauge forecasts issued silently · HIGH · ACCURACY · FIXED
Forecasts were projected forward from `q_hist["date"].iloc[-1]` with no check
that the last observation was recent. A gauge offline for days still produced a
full 14-day curve anchored to stale flow, presented identically to a fresh one.
**Fix (Phase 1):** `data_age_days` + `stale` fields on every forecast
(`RW2_STALE_AFTER_DAYS`, default 2), with a `notes` entry when stale.

### C — Per-member bias scale pooled across all horizons · MEDIUM · ACCURACY · FIXED
`bias_scale = mean(obs)/mean(pred)` was computed over all h=1..14 pooled
together, then applied uniformly. A member biased high at short lead but low at
long lead got a single averaged correction that helped neither end.
**Fix (Phase 4):** independent `bias_scale[h]` per horizon (each clamped
[0.5,2.0]), pooled fallback for horizons with < 3 finite samples. Reported
`member_bias_scales` keeps the pooled value for backward compatibility.

### D — Provisional/outlier USGS data trained on blindly · MEDIUM · ACCURACY · PARTIALLY ADDRESSED
`_parse_dv` ignored NWIS approval/qualifier codes, and there was no
outlier/range gate — a gauge spike (real flood vs. malfunction) was trained on
indiscriminately.
**Fix (Phase 5):** added `usgs.flag_suspect_jumps()` — flags isolated
single-day spikes (≥50× both neighbours, floored at 10 cfs) and records a count
in `notes`. It *flags, never filters* (deleting a real flood peak is worse). A
real 33,889-day record flagged 0 spikes (no over-flagging); the synthetic
glitch test flags correctly.
**Not done:** NWIS qualifier-code (provisional vs. approved) handling. The
per-site value cache stores only `{date: value}`, so plumbing provisional flags
to the forecast would need a cache-schema migration — deferred as higher-risk,
lower-reach (cache is the primary path under `RW2_NO_FETCH=1`).

### E — Silent member drops · no NaN floor · failures didn't gate deploy · MEDIUM · ACCURACY+OPS · FIXED
- Members that failed to load were dropped to stdout only; the blend silently
  changed composition with no machine-readable record.
- If all members failed, the blend became all-NaN with no explicit floor.
- `build_static_site.py` recorded failures to a summary JSON but **only failed
  the shard if zero stations succeeded** — a shard could drop hundreds of
  gauges and still deploy.
**Fix (Phase 1):** `members_used` / `members_dropped` on every forecast; a
NaN floor that falls back to persistence and sets `degraded=true` when the
blend is non-finite at h=1; and `_failure_gate()` that fails the shard
(blocking merge/deploy via `needs:`) when the *real* failure rate exceeds
`RW2_MAX_FAILURE_RATE` (default 25%). The gate excludes gauges that raised
"no USGS daily discharge" — `stations_v15.json` has a structural ~10-15% of
discontinued / non-daily-flow sites that have nothing to forecast and are not a
build regression (`_classify_failures()` splits no-data from real failures); it
judges health only over forecastable gauges. *(This finding was independently
confirmed in the wild: a local run silently dropped 5 of 7 members — now
visible.)*

### F — No tests · no CI validation · no structured logging · HIGH · OPS · ADDRESSED (tests/CI)
Verified at audit: **0** test files, `pytest` absent from requirements and CI,
**0** uses of the `logging` module in `app/`, **90+** bare `except Exception`
blocks (38 in `forecast.py`). Any change could ship a silently-broken forecast.
**Fix (Phase 2):** a 45-test `pytest` suite covering the anchor decay, asinh
round-trip, q-scale, inverse-MAE² weighting, per-horizon bias, USGS parsing +
spike detection, the failure gate, the regime-aware offsets, and the Phase 1
guardrails end-to-end (foundation models monkeypatched off for speed). New
`test` CI job gates the entire `snodas-extract → build → merge → deploy` chain.
**Not done:** structured `logging` migration (the 90+ bare excepts and
stdout-only diagnostics remain) — large mechanical change, deferred. The
`members_dropped`/`notes` fields now make the most important silent failures
machine-readable without it.

---

## Configuration added

| Env var | Default | Effect |
|---|---|---|
| `RW2_STALE_AFTER_DAYS` | `2` | Age (days) past which a forecast is flagged `stale`. |
| `RW2_MAX_FAILURE_RATE` | `0.25` | Per-shard *real* failure rate (excluding no-data gauges) above which the build fails. |

(Existing gates such as `RW2_PER_MEMBER_BIAS_OFF` are preserved for rollback.)

## New forecast output fields

`data_age_days`, `stale`, `degraded`, `members_used`, `members_dropped`
(serialized via `asdict` into each `dist/forecasts/{id}.json`).

## Verification

- `pytest tests/` — 45 tests green.
- `flag_suspect_jumps` over a real 33,889-day USGS record: 0 false positives.
- End-to-end `forecast_station` on a cached station (`RW2_NO_FETCH=1`): finite
  blend, correct staleness/age, populated `members_used`/`members_dropped`, no
  NaN.

## Recommended follow-ups (not in this pass)

1. Migrate `app/` to structured `logging` and replace blanket `except
   Exception` with typed handling that distinguishes retry-able from fatal.
2. NWIS qualifier-code handling (requires per-site cache schema bump to retain
   approval status).
3. Flow-tercile-stratified weighting (so a member good only at low flow can't
   dominate weights used on a rising limb) — extends Phase 3.
4. A scheduled job that backtests the deployed blend and alerts on MAE
   regression, rather than point-in-time `benchmark_40.py` runs.
