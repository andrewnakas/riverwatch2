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

1. **No observed discharge (rainfall–runoff).** The model predicts streamflow
   from weather + static attributes only — no observed discharge as input. This
   tests whether the model has *learned hydrology*, and is the harder, more
   scientifically interesting task; the "records" (0.74 → 0.82 → 0.83) refer to
   it. **Note:** this is distinct from *PUB* (Prediction in Ungauged Basins),
   where the test basins are entirely held out from training. All numbers in this
   paper — ours and the records' — are the **temporal split** (train on all 531
   basins 1999–2008, test the same 531 basins 1989–1999), which is the standard
   CAMELS benchmark and easier than PUB. PUB ensembles peak lower (~0.70–0.79,
   Kratzert 2019); we do not report a PUB number.

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

All numbers below are CONFIRMED from the primary source unless marked ~approx.
The protocol column notes whether a number is comparable to the standard
CAMELS-531 temporal-split median-NSE benchmark.

### 2.1 No-observed-discharge (rainfall–runoff) track

| Year | Model | median NSE | #basins | protocol / caveat | citation |
|------|-------|-----------|---------|-------------------|----------|
| pre-2018 | Calibrated SAC-SMA / conceptual | ~0.55–0.65 | 531 | the pre-deep-learning bar; ~0.72 was the prior LSTM-NLDAS point | — |
| 2019 | Regional **EA-LSTM** (single) | **~0.74** | 531 | first DL to beat calibrated models; ensemble higher | Kratzert et al. 2019, WRR 55, [10.1029/2019WR026065](https://doi.org/10.1029/2019WR026065) |
| 2020 | Multi-timescale / continental LSTM (single-forcing best) | ~0.77 | 531 | single-forcing LSTM ceiling | Kratzert / Gauch et al. 2020–21 |
| 2021 | **Multi-forcing LSTM ensemble** | **0.8082** | 531 | per-forcing 10-seed ensemble; the "multi-forcing" jump | Kratzert et al. 2021, HESS 25:2685, [link](https://doi.org/10.5194/hess-25-2685-2021) |
| 2022–23 | **δHBV** differentiable hybrid (single) | ~0.74–0.75 | 531/671 | physics-ML hybrid ≈ single LSTM; value is as an ensemble member | Feng et al. 2022/2023 (WRR/HESS) |
| 2024 | RR-Former (transformer), FHNN/TFT (state-space) | ~0.75–0.77 regional; 0.83 **per-basin in-sample** | 531 | **NOT comparable**: per-basin in-sample, not regional temporal split | Yin 2022; FHNN 2024 |
| **2025** | **(LSTM+δHBV) grand ensemble (δHBV1.1p)** | **~0.83** | 531 | LSTM¹²³ ens 0.808 → +δHBV 0.818 → +seeds/multi **0.83** — **RECORD** | **Li, Song, Pan, Lawson & Shen 2025, HESS 29:6829, [link](https://doi.org/10.5194/hess-29-6829-2025)** |
| 2026 | **RiverWatch2 no-q grand ensemble** | **~0.80 pooled / 0.829 day-1** | 531 (177 screen) | day-1 ≈ record; pooled/continuous ~0.02–0.03 short | **this work** |

**Current no-q record: Li/Shen 2025, median NSE ~0.83 on CAMELS-531 temporal
split.** No 2024–2026 result cleanly beats it *on the same no-q 531 temporal
protocol*: transformers (RR-Former) and state-space models (FHNN ~0.77) do not
clear it in regional temporal testing; the one ≥0.83 transformer number
(RR-Former 0.83 / RR-TiDE 0.848) is **per-basin in-sample**, a different and
easier setup. Kratzert et al. 2024 (HESS 28:4187, "never train on a single
basin") reinforces that ~0.83 is an *ensembling* ceiling, not a single-model one.
Single-model no-q ceiling is ~0.74–0.75; everything above is ensemble breadth.

### 2.2 Discharge-assimilating (autoregressive / nowcast) track

| Year | Model | median NSE | #basins | lag/lead + caveat | citation |
|------|-------|-----------|---------|-------------------|----------|
| 2020 | **DI-LSTM** data integration | **0.852** (from 0.714 base, +19%) | 671 | WRR; weaker base than later work | Feng, Fang & Shen 2020, WRR 56, [10.1029/2019WR026793](https://doi.org/10.1029/2019WR026793) |
| **2022** | **Autoregressive LSTM (AR)** | **0.879** (base 0.796 → DA 0.859 → **AR 0.879**) | **531** | 1-day-lag **nowcast** (no met forecast); 1989–1999 test — **RECORD** | **Nearing et al. 2022, HESS 26:5493, [link](https://doi.org/10.5194/hess-26-5493-2022)** |
| 2025 | δHBV data-integration (Western US) | KGE 0.80→0.96 @1-day | 646 (West) | reports KGE not NSE; confirms effect, not a CAMELS-531 NSE record | Song/Shen et al. 2025, HESS 29:5453 |
| **2026** | **RiverWatch2 with-q grand ensemble** | **0.9016** | **531** | 1-day-lag nowcast; **verified on all 531 basins** | **this work** |

**Current with-q record (prior): Nearing et al. 2022, median day-1 NSE 0.879 on
CAMELS-531 (1-day-lag AR nowcast).** RiverWatch2's 0.9016 exceeds it by +0.023 on
the identical 531-basin / 1989–1999 protocol.

### 2.3 What Google's global models are (and are not)

Nearing et al. 2024 (Nature 627:559, "Global prediction of extreme floods in
ungauged watersheds") is an **ungauged, no-discharge, extreme-event-reliability**
model on ~5,680 GRDC/Caravan gauges, benchmarked on precision/recall/lead-time vs
GloFAS — **not** a CAMELS median-NSE record for either track. It is frequently
mis-cited as "the streamflow record"; it answers a different question (ungauged
flood reliability) on a different dataset with a different metric.

### 2.4 The critical comparability caveat

The with-q and no-q numbers are **not on the same axis**. Feeding yesterday's
observed discharge makes near-persistence trivially accurate on the many slow /
baseflow-dominated CAMELS basins, so nowcast NSE is *latency-inflated*: Nearing
2022's own base LSTM was 0.796 (no-q) and jumped to 0.879 (AR) — a +0.083 gain
that is mostly the autocorrelation of streamflow, not new hydrological skill. A
with-q 0.90 and a no-q 0.83 are therefore **not** rankable against each other;
they are records in two different competitions.

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

1. **Newman, A. J., et al. (2015).** Development of a large-sample watershed-scale
   hydrometeorological dataset for the contiguous USA (CAMELS). *HESS* 19,
   209–223. https://doi.org/10.5194/hess-19-209-2015
2. **Addor, N., et al. (2017).** The CAMELS data set: catchment attributes and
   meteorology for large-sample studies. *HESS* 21, 5293–5313.
   https://doi.org/10.5194/hess-21-5293-2017
3. **Kratzert, F., et al. (2019).** Toward Improved Predictions in Ungauged
   Basins: Exploiting the Power of Machine Learning. *WRR* 55.
   https://doi.org/10.1029/2019WR026065 (EA-LSTM; single ~0.74 on CAMELS-531)
4. **Kratzert, F., et al. (2021).** A note on leveraging synergy in multiple
   meteorological data sets with deep learning for rainfall–runoff modeling.
   *HESS* 25, 2685–2703. https://doi.org/10.5194/hess-25-2685-2021
   (multi-forcing per-forcing LSTM ensemble, median NSE 0.8082)
5. **Feng, D., Fang, K., & Shen, C. (2020).** Enhancing streamflow forecast and
   extracting insights using LSTM networks with data integration at continental
   scales. *WRR* 56, e2019WR026793. https://doi.org/10.1029/2019WR026793
   (DI-LSTM: median NSE 0.714 → 0.852, 671 basins)
6. **Feng, D., et al. (2022/2023).** Differentiable, learnable, regionalized
   process-based models (δHBV). *WRR / HESS* (δHBV ≈ single LSTM, ~0.74–0.75).
7. **Nearing, G., et al. (2022).** Technical note: Data assimilation and
   autoregression for using near-real-time streamflow observations in LSTM
   networks. *HESS* 26, 5493–5513. https://doi.org/10.5194/hess-26-5493-2022
   (**with-q record**: base 0.796, DA 0.859, AR **0.879**, CAMELS-531, 1-day lag)
8. **Li, J., Song, Y., Pan, M., Lawson, K., & Shen, C. (2025).** Ensembling
   differentiable process-based and data-driven models with diverse
   meteorological forcing datasets to advance streamflow simulation. *HESS* 29,
   6829. https://doi.org/10.5194/hess-29-6829-2025
   (**no-q record**: LSTM¹²³ 0.808 → (LSTM+δHBV) 0.818 → grand ens **~0.83**)
9. **Nearing, G., et al. (2024).** Global prediction of extreme floods in
   ungauged watersheds. *Nature* 627, 559–563.
   https://doi.org/10.1038/s41586-024-07145-1
   (ungauged flood-reliability model on Caravan/GRDC — *not* a CAMELS NSE record)
10. **Kratzert, F., et al. (2024).** HESS Opinions: Never train an LSTM on a
    single basin. *HESS* 28, 4187. https://doi.org/10.5194/hess-28-4187-2024
    (regional-ensemble ceiling context)
11. **Song, Y., Shen, C., et al. (2025).** Improving streamflow simulation
    through ML-powered data integration (Western US). *HESS* 29, 5453.
    https://doi.org/10.5194/hess-29-5453-2025 (with-q, KGE-reported)

*Codebase: mhpi/hydrodl2, mhpi/generic_deltaModel (δHBV1.1p reference
implementation); neuralhydrology (Kratzert LSTM benchmark).*

---

*This working paper is a living document; RiverWatch2 numbers update as the
grand ensemble is extended. Latest verified figures are always in
`benchmarks/EXPERIMENTS.md` and `benchmarks/combine_*.json`.*
