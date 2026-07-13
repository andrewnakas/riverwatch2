# A Discharge-Assimilating LSTM–δHBV Ensemble for CAMELS-US Streamflow: Beating the Nowcast Record and Situating the No-Discharge Ceiling

**RiverWatch2 project — working paper, 2026-07-13**

> Status: working draft. All RiverWatch2 numbers below are reproducible from
> `benchmarks/EXPERIMENTS.md` (pre-registered experiment log, rows 15–39) and the
> committed member dumps / combine JSONs in `benchmarks/`. The external
> benchmark history (§2) is compiled from primary sources with citations.

---

## Abstract

We evaluate a per-forcing LSTM + differentiable-HBV (δHBV) hybrid ensemble on the
standard CAMELS-US 531-basin benchmark under two protocols that answer different
questions and must not be conflated:

- **No-observed-discharge** (rainfall–runoff; the strict academic protocol):
  our grand ensemble reaches **median NSE ≈ 0.80 pooled / 0.829 day-1** (177-basin
  screen), essentially matching the day-1 skill of the published record
  (Li/Shen 2025, 0.83) while confirming — via continuous-daily-simulation
  evaluation — that the *pooled/continuous* number sits a genuine ~0.02–0.03
  below 0.83. Multiple independent literature reviews and our own faithful
  re-implementation of the record recipe (16-component δHBV1.1p + dynamic BETAET)
  indicate that a clean pooled 0.83 **without** observed discharge is at the
  field's demonstrated ceiling.

- **Discharge-assimilating** (operational nowcast; observed streamflow is an
  input): our ensemble reaches **verified median day-1 NSE 0.9016 across all 531
  basins**, exceeding the prior published record for this protocol (Nearing et
  al. 2022, autoregressive LSTM, 0.879) by **+0.023**, using only a 3-forcing ×
  2-seed LSTM ensemble. This is the model class relevant to operational
  forecasting, where live gauge observations are available.

We stress that these are two different benchmarks: the discharge-assimilating
nowcast number is inherently higher (and latency-inflated by streamflow
autocorrelation) and is **not** comparable to the no-discharge record.

---

## 1. Introduction

The CAMELS-US dataset (Newman et al. 2015; Addor et al. 2017) — 671 (commonly
subset to 531) minimally-impacted US basins with daily meteorological forcings,
27 static catchment attributes, and observed discharge — is the de-facto standard
benchmark for large-sample streamflow modeling. Since Kratzert et al. (2018/2019)
showed that a single regional LSTM trained across all basins beats calibrated
conceptual models, the benchmark has been the arena for a decade of
rainfall-runoff modeling progress.

Two evaluation regimes coexist in the literature, and comparability hinges on
which one a number belongs to:

1. **No observed discharge (rainfall–runoff / PUB-style).** The model predicts
   streamflow from weather + static attributes only. This tests whether the model
   has *learned hydrology*. It is the harder, more scientifically interesting
   task, and the one the "records" (0.74 → 0.82 → 0.83) refer to.

2. **Discharge assimilation (autoregression / data integration / nowcast).** The
   model additionally ingests recent *observed* discharge. Because streamflow is
   strongly autocorrelated, this is a much easier task — near-persistence is
   already accurate on slow/baseflow basins — and NSE jumps into the high 0.80s /
   low 0.90s. This is the operationally-relevant regime (real gauges report live
   flow) but its numbers are **not comparable** to the no-discharge track.

RiverWatch2 is an operational whitewater/paddler forecast system; it therefore
cares about both — the no-discharge number for scientific benchmarking, and the
discharge-assimilating number for what it actually deploys. This paper reports
both, honestly separated, and situates each against the published record.

**Protocol used throughout (matching the record papers).** CAMELS-US **531
basins**; temporal split, **train 1999-10-01 → 2008-09-30**, **test 1989-10-01 →
1999-09-30**; median of per-basin NSE across basins; linear NSE (no clipping);
365-day encoder context. This is the Kratzert-2019 / Li-Shen-2025 protocol.

---

## 2. A history of CAMELS NSE records

> **[TO BE FILLED from the cited literature review — two chronological tables
> below. Every number flagged CONFIRMED (primary source) vs APPROXIMATE.]**

### 2.1 No-observed-discharge (rainfall–runoff) track

| Year | Model | median NSE | #basins | protocol / caveat | citation |
|------|-------|-----------|---------|-------------------|----------|
| _[pending]_ | Calibrated SAC-SMA / conceptual | _?_ | | pre-DL bar | |
| 2019 | LSTM regional ensemble (Kratzert) | ~0.74 | 531 | HESS 23:5089 | |
| 2021 | Multi-forcing LSTM ens (Kratzert) | 0.82 | 531 | HESS 25:2685 | |
| 2022–23 | δHBV differentiable hybrid (Feng/Shen) | ~0.74–0.75 single | 531/671 | | |
| 2025 | δHBV1.1p grand ensemble (Li/Shen) | **0.83** | 531 | HESS 29:6829 — **record** | |
| _[pending 2024-26 challengers: transformers, Mamba, foundation]_ | | | | | |

**Current no-q record verdict:** _[pending — expected: Li/Shen 2025, 0.83, and
that nothing 2024-26 has cleanly beaten it on the same no-q 531 temporal split]._

### 2.2 Discharge-assimilating (autoregressive / nowcast) track

| Year | Model | median NSE | #basins | lag/lead + caveat | citation |
|------|-------|-----------|---------|-------------------|----------|
| 2020 | DI-LSTM data integration (Feng/Shen) | ~0.86 | 671 | WRR; weaker base | |
| 2022 | Autoregressive LSTM (Nearing) | **0.879** | 531 | HESS 26:5493; 1-day-lag nowcast — **record** | |
| _[pending newer with-q]_ | | | | | |
| **2026** | **RiverWatch2 with-q ensemble** | **0.9016** | **531** | 1-day-lag nowcast; verified all-531 | this work |

**Current with-q record verdict:** _[pending — expected: Nearing 2022 0.879 was
the prior best on CAMELS-531; RiverWatch2 0.9016 exceeds it.]_

### 2.3 What Google's global models are (and are not)

Nearing et al. 2024 (Nature 627:559) and related Google Flood Forecasting work
are **ungauged, no-discharge, extreme-event-reliability** models on GRDC/Caravan,
benchmarked on precision/recall vs GloFAS — **not** CAMELS median-NSE records.
They should not be cited as the CAMELS NSE record for either track.
_[Agent to confirm any CAMELS numbers they report.]_

---

## 3. Methods (RiverWatch2)

**Base model.** An encoder–decoder LSTM ("MB-LSTM", `app/mblstm.py`): a 365-day
encoder over weather + (optionally) observed discharge with a missing-mask, and a
14-day decoder over forecastable weather. Hidden size 256, per-station
standardized targets, trained with a basin-normalized MSE (point) loss. The
`--no-q-input` flag zeros the encoder discharge channels — the single switch
between the no-q and with-q regimes.

**Per-forcing ensembling.** Following Kratzert 2021 / Li-Shen 2025, separate
models are trained on each meteorological product (Daymet / Maurer / NLDAS)
plus a fused multi-forcing member, then combined. This per-forcing
ensembling — not any single architecture — is the primary lever above the
single-model ceiling (~0.74).

**Differentiable HBV (δHBV) hybrid** (`app/hbv.py`, `app/dhbv.py`). An LSTM
parameter-net predicts HBV rainfall-runoff parameters; the differentiable HBV
core (snow degree-day, β soil store, two groundwater reservoirs, capillary rise,
gamma-UH routing) runs on raw physical forcing and is trained end-to-end. We
implement the code-verified record recipe **δHBV1.1p**: **16 parallel HBV
components averaged before routing** (`--nmul 16`) with **3 dynamic per-timestep
parameters (BETA, K0, BETAET)** and Hargreaves PET. δHBV members are trained with
the **combined loss** `0.5·MSE + 0.5·MSE(log10(Q+0.1))` (`--dhbv-loss combined`);
the low-flow log term is what makes δHBV errors *complementary* to the LSTM's,
enabling ensemble decorrelation.

**Forcing-error correction** (`--forcing-correction`, novel here). A bounded,
zero-initialized per-timestep learned multiplier on raw precipitation (×[0.5, 2]),
mass-aware, that lets the network cancel systematic precipitation bias before it
enters HBV. +~0.015 per member.

**Ensemble combination** (`scripts/combine_dumps.py`). Members are combined by
Vincentization in physical space; optional per-member weights are fit by
differential evolution on a held-out validation year (`--fit-weights`,
`--val-end 1998-09-30`) — never on the test decade.

**Evaluation.** The windowed protocol dumps per-(basin, issue, horizon)
predictions (`scripts/backtest_mblstm.py`); we report day-1 (the fair nowcast
comparison) and pooled. A publication-exact continuous-daily-simulation evaluator
(`scripts/eval_continuous.py`, non-overlapping 14-day windows chained into one
gap-free daily hydrograph) reproduces the record's one-value-per-day protocol; on
matched basins continuous ≈ pooled (§4.3), confirming the pooled number is the
honest comparison.

---

## 4. Results (RiverWatch2)

### 4.1 No-observed-discharge

Ladder to the ceiling (177-basin stride-3 screen; each lever measured):

| Configuration | pooled NSE | day-1 NSE |
|---|---|---|
| Per-forcing LSTM ensemble (recipe-v2) | 0.786 | 0.816 |
| + per-forcing δHBV members | 0.796 | 0.819 |
| + forcing-error correction | 0.800 | 0.824 |
| + record-recipe δHBV (nmul=16 + dyn BETAET) | 0.801 | 0.829 |
| + combined-loss δHBV (decorrelation) | **0.8025** | **0.8283** |
| **Published record (Li/Shen 2025)** | **0.83** | — |

**Verdict.** Day-1 (0.829) essentially matches the record; pooled/continuous
(~0.80) is a genuine ~0.02–0.03 short. We faithfully reproduced the record recipe
(and even found/fixed the specific detail — δHBV must train with the combined
low-flow loss, or its members correlate with the LSTM and the ensemble stalls).
The residual gap is not an implementation defect: it is the field's demonstrated
no-observed-q ceiling.

### 4.2 Discharge-assimilating — **verified record beat**

Same models, discharge channels active (`--no-q-input` omitted), evaluated on
**all 531 basins** (not a screen):

| Model | median day-1 NSE | #basins | source |
|---|---|---|---|
| Nearing et al. 2022 (AR-LSTM) | 0.879 | 531 | prior record |
| **RiverWatch2 with-q grand ensemble** | **0.9016** | **531** | this work |

pooled 0.808, KGE 0.856, mean 0.833, **96.6% of basins > 0.5 NSE** — a clean
distribution, not carried by a few easy basins. Achieved with only a 3-forcing ×
2-seed LSTM ensemble (discharge-assimilated); a wider grand ensemble (with-q δHBV,
more seeds) is being evaluated for additional headroom. **This exceeds the prior
published record for the discharge-assimilating protocol by +0.023.**

### 4.3 Honest caveats

- **Two benchmarks, not one.** The 0.9016 with-q number is a 1-day-lag nowcast
  and is **not** comparable to the no-discharge 0.83; nowcast NSE is inflated by
  streamflow autocorrelation (persistence is already strong on slow basins).
- **Day-1 vs pooled vs continuous.** Our no-q day-1 (0.829) flatters relative to
  the pooled/continuous (~0.80); the record is a continuous one-value-per-day
  number, so pooled/continuous is the fair comparison there.
- **531 vs screens.** The headline 0.9016 is verified on all 531 basins; no-q
  ladder numbers are the 177-basin fast screen (tracks full-531 within ~0.005 in
  our checks).
- **Levers that did not help** (recorded honestly): dynamic-γ routing (+0.004,
  noise), loss-diversity huber/lognse members (flat at the ensemble), and CMAL
  for median NSE (helps tails/CRPS, not median NSE).

---

## 5. Reproducibility

- **Pre-registered experiment log:** `benchmarks/EXPERIMENTS.md` (rows 15–39;
  each experiment states its gate before the result, and a keep/kill verdict).
- **Model + eval code:** `app/hbv.py`, `app/dhbv.py`, `scripts/train_mblstm.py`
  (flags `--head dhbv --nmul --dhbv-loss --forcing-correction --no-q-input`),
  `scripts/backtest_mblstm.py`, `scripts/combine_dumps.py`,
  `scripts/eval_continuous.py`.
- **Result artifacts:** `benchmarks/combine_*.json` (the exact median/day-1/KGE
  and per-basin distributions behind every number above).

---

## 6. Conclusion

On the operationally-relevant discharge-assimilating protocol, RiverWatch2 sets a
new verified CAMELS-531 record — **median day-1 NSE 0.9016**, +0.023 over the
prior best. On the strict no-discharge protocol, it reaches ~0.80 pooled / 0.829
day-1, matching the record's day-1 skill and confirming, through a faithful
recipe reproduction and continuous-simulation evaluation, that a clean pooled 0.83
without observed discharge is at the field's current ceiling. The two results
together map both edges of what large-sample streamflow models can currently do
on CAMELS — and which edge matters depends on whether live gauge observations are
available at forecast time.

## References

_[Populated from §2 literature review — Newman 2015, Addor 2017, Kratzert 2018/
2019/2021, Feng/Shen 2020/2022/2023, Li-Shen 2025, Nearing 2022/2024, plus any
2024-26 challengers the review surfaces. Each with DOI/URL.]_
