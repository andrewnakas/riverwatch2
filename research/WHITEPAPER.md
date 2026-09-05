# How Good Can a CAMELS Streamflow Model Get? A Gauge-Error Ceiling, and Two Records Measured Against It

**RiverWatch2 project — working paper, 2026-08-09**

> Status: working draft. Every RiverWatch2 number below is reproducible from the
> committed dumps and the scripts in `analysis/`. The external benchmark history
> (§2) is compiled from primary sources with citations. **Numbers changed
> substantially in the 2026-08 revision** — see §7 for what was corrected and why.

---

## Abstract

Large-sample streamflow benchmarking on CAMELS-US has become a contest of median
NSE, but no published work states how high that number can go before it is
measuring gauge noise rather than hydrology. We do three things.

**First, we derive an observational ceiling.** For multiplicative discharge
observation error with relative scale σ, the maximum achievable NSE against a
noisy gauge is

```
NSE_ceiling = 1 − (M/V)·(e^{σ²} − 1)
```

where `M` and `V` are the mean-square and variance of the true series. The
expression is verified analytically against Monte Carlo simulation to ~0.005
across σ = 0.13–0.70. We could not find any published paper that converts
discharge uncertainty into a maximum achievable NSE; the nearest work (Aerts et
al. 2024) tests whether *model differences* exceed observation uncertainty
without deriving a bound.

**Second, we measure σ itself on the CAMELS gauges**, rather than selecting a
scenario. Comparing **144,914 USGS field gaugings against the published daily
values they underpin**, across **523 of the 531 basins**, gives the total error in
the series NSE is computed against — measurement, rating-curve and shift error
together. σ **rises steeply with flow**, from 0.045 at typical flows to **0.121**
in the 90th–99th percentile and **0.251** above it, which is the *opposite* of the
gradient assumed by scenarios derived from the literature. Because the ceiling is
flow-weighted, that direction matters: the measured no-q ceiling is
**0.73–0.95**, a band the previously reported scenarios were too narrow to
contain. We also find the observations cannot be
repaired to escape it — gauge error carries almost no learnable per-station bias
(0.9% of variance) and almost no persistence (ρ ≈ 0.07) — but its **magnitude**
is a stable, predictable gauge property (split-half r = 0.50; log basin area
r = −0.58), which supports reporting skill relative to each basin's own ceiling.

**Third, we place two records against the ceiling.** On the discharge-assimilating
protocol we reach **median day-1 NSE 0.888355** on all 531 basins ⚠️ (was 0.9203 — see the
correction notice at §5.2; that figure was leaked and frame-inflated) (prior published
record: Nearing et al. 2022, 0.879). On the strict no-discharge protocol we reach
a held-out **0.8363** (prior published record: Li et al. 2025, 0.8294). Against
the **measured** bound the no-q record sits inside the band (0.73–0.95), and
against each basin's *own* measured σ it captures **84–89% of what its gauges can
reward** (robust–SD range, both sides computed on the held-out frame); under the
earlier scenario-based bounds it retained roughly +0.082. The with-q record sits against a
central-scenario bound of **0.9453** with **50%** of basins already saturated.
The direction of travel across both revisions is the same: **each time the
ceiling has been measured more carefully, it has moved closer to the records.**

**Fourth, and least expected, we find that the binding constraint is the metric.**
Median NSE is a rank statistic, so only basins near the median rank can move it —
improving the worst 50 basins by any amount changes it by exactly zero. Combining
that with the ceiling, just **44 of 531 basins** are simultaneously below their
gauge-error ceiling and near enough the median to matter; raising only those to
their ceilings would carry the no-q median past its target. Meanwhile **421 of
531 basins have real headroom the metric cannot see**, and the 44 cannot be
identified in advance because rank does not transfer between windows (27%
overlap). The observations have room almost everywhere; the *statistic* does not.

The scientific content is therefore not "we got a higher number." It is that the
benchmark now has a **stated upper bound derived from the observations rather than
from model performance**, that the bound is **steeply sensitive to the assumed
gauge error** (σ = 0.13 → 0.41 moves it from 0.977 to 0.756), and that both facts
are dominated by a **reporting choice** that discards most of the available
signal. We support this with a series of negative results sharing one diagnosed
mechanism: the residual error is **scatter in peak magnitude**, not bias, not
timing, and not an architectural output limit.

⚠️ **Correction notice.** An earlier draft reported the with-q record as sitting
0.0035 below a central-scenario ceiling of 0.9241 and concluded the protocol was
"essentially finished." That was an artifact of an error in our own closed form
under flow-dependent σ (§4.1); the corrected bound is 0.9453 and the conclusion
does not hold. We describe the error and its diagnosis rather than quietly
restating the numbers.

---

## 1. Introduction

The CAMELS-US dataset (Newman et al. 2015; Addor et al. 2017) — 671 basins,
commonly subset to 531 — is the standard large-sample streamflow benchmark. Since
Kratzert et al. (2018/2019) showed a single regional LSTM beats calibrated
conceptual models, progress has been reported as a rising median NSE: ~0.74
(single LSTM) → 0.808 (multi-forcing ensemble) → ~0.83 (LSTM + differentiable
HBV).

Two evaluation regimes coexist, and conflating them is the most common error in
citing this literature:

1. **No observed discharge (rainfall–runoff).** Weather and static attributes
   only. The harder, more scientifically interesting task; the "records" refer to
   it. Distinct from PUB (prediction in ungauged basins), where test *basins* are
   held out.
2. **Discharge assimilation (autoregressive nowcast).** Recent *observed*
   discharge is also an input. Because streamflow is strongly autocorrelated this
   is much easier, and NSE jumps into the high 0.80s / low 0.90s. It is the
   operationally relevant regime, and its numbers are **not comparable** to the
   no-discharge track.

We report both, separately, on their own published protocols — and we use
different train/test windows for each because the two record papers do
(§3). A rising number in either track eventually runs into a question nobody in
this literature has answered: **at what point does a higher NSE stop meaning a
better model?** NSE is computed against gauge observations, and gauge discharge
at high flows is itself uncertain — commonly 10–20%, and far worse where the
rating curve is extrapolated (Kiang et al. 2018 report 41–200% at an extrapolated
site). Once model error approaches gauge error, additional skill is unrewardable
in principle.

That is the question this paper is organized around.

---

## 2. A history of CAMELS NSE records

All numbers are confirmed from the primary source. The protocol column notes
comparability to the standard CAMELS-531 temporal-split median-NSE benchmark.

### 2.1 No-observed-discharge track

| Year | Model | median NSE | #basins | protocol / caveat | citation |
|------|-------|-----------|---------|-------------------|----------|
| pre-2018 | Calibrated SAC-SMA / conceptual | ~0.55–0.65 | 531 | the pre-deep-learning bar | — |
| 2019 | Regional **EA-LSTM** (single) | **~0.74** | 531 | first DL to beat calibrated models | Kratzert et al. 2019, WRR 55, [10.1029/2019WR026065](https://doi.org/10.1029/2019WR026065) |
| 2021 | **Multi-forcing LSTM ensemble** | **0.8082** | 531 | per-forcing 10-seed ensemble | Kratzert et al. 2021, HESS 25:2685 |
| 2022–23 | **δHBV** differentiable hybrid (single) | ~0.74–0.75 | 531/671 | physics-ML hybrid ≈ single LSTM | Feng et al. 2022/2023 |
| **2025** | **(LSTM+δHBV) grand ensemble** | **0.8294** | 531 | LSTM¹²³ 0.808 → +δHBV 0.818 → **0.8294** | **Li, Song, Pan, Lawson & Shen 2025, HESS 29:6829** |
| 2025 | 11 transformer variants + TimeGPT/Lag-Llama/TTM | LSTM wins (KGE 0.80 vs 0.73) | 531 | transformers do **not** beat LSTM | Liu et al. 2025, HESS 29:6811 |
| 2026 | Time-series foundation models (MOIRAI, Chronos-Bolt, TTM, Sundial) | best fine-tuned 0.753 ≈ single LSTM | 531 | zero-shot far below LSTM | Sun & Sun 2026, ML:Earth 2, 010501 |
| **2026** | **RiverWatch2 no-q grand ensemble** | **0.8363** | **531** | held-out, one-shot pre-registered query | **this work** |

### 2.2 Discharge-assimilating track

| Year | Model | median NSE | #basins | caveat | citation |
|------|-------|-----------|---------|--------|----------|
| 2020 | **DI-LSTM** data integration | **0.852** (from 0.714) | 671 | weaker base than later work | Feng, Fang & Shen 2020, WRR 56 |
| **2022** | **Autoregressive LSTM** | **0.879** | **531** | 1-day-lag nowcast — prior **RECORD** | **Nearing et al. 2022, HESS 26:5493** |
| 2026 | MLP orchestrator DA | ~0.81–0.84 @1-day | 531 | below prior record | Saint-Fleur et al. 2026, HESS 30:3497 |
| **2026** | **RiverWatch2 with-q grand ensemble** | **0.888355** | **531** | 1-day-lag nowcast, 5 members × 5 seeds, guarded split, all-days frame | **this work** |

### 2.3 What Google's global models are (and are not)

Nearing et al. 2024 (*Nature* 627:559) is an **ungauged, no-discharge,
extreme-event-reliability** model on ~5,680 GRDC/Caravan gauges, scored on
precision/recall/lead-time against GloFAS — **not** a CAMELS median-NSE record. It
is frequently mis-cited as "the streamflow record." Its operational successor
(Flood Hub, 2024–2026) uses a CMAL head and reanalysis-grade global forcings,
targeting ungauged prediction; that design choice is consistent with our own
finding (§6.3) that members without gauge-calibrated precipitation lose far more
skill than their diversity returns.

### 2.4 The comparability caveat

With-q and no-q numbers are **not on the same axis**. Feeding yesterday's observed
discharge makes near-persistence accurate on slow, baseflow-dominated basins, so
nowcast NSE is latency-inflated: Nearing 2022's own base LSTM was 0.796 and rose
to 0.879 with autoregression — a +0.083 gain that is mostly streamflow
autocorrelation. **A with-q 0.92 and a no-q 0.84 are records in two different
competitions.** They also use different windows and different sampling (§3), so
no arithmetic relating them is meaningful.

---

## 3. Methods

### 3.1 Two protocols, held strictly separate

| | **no-q track** | **with-q track** |
|---|---|---|
| protocol source | Li/Song 2025 | Kratzert 2019 / Nearing 2022 |
| train | 1980-10-01 → 1995-09-30 | 1999-10-01 → 2008-09-30 |
| test | 1995-10-01 → 2010-09-30 | 1989-10-01 → 1999-09-30 |
| sampling | daily | stride-14 issue dates, 14-day horizon |
| scored | day-1, median per-basin NSE | day-1, median per-basin NSE |

Two consequences we state rather than hide. **(a)** The Maurer forcing product
ends in 2008, so the no-q scored window is effectively **1995-10-01 → 2008-12-07**
— about 13.2 years, not 15. **(b)** With-q day-1 NSE rests on **261 issue dates
per basin**; a single missed flood can drive one basin's NSE to −0.9, so per-basin
with-q NSE is statistically fragile and we prefer pooled error for cohort claims.

A `check_split.py` guard refuses to produce a dump from any run whose config does
not match the intended split. This exists because two separate runs (~20 GPU-hours)
reached the dump stage on the wrong split, each looking perfect by every other
signal — correct epoch count, clean loss curve, 531 basins, intact corpus.

### 3.2 Models and members

**no-q track (neuralhydrology CudaLSTM).** Nine streams: three single-forcing
LSTMs (Daymet, Maurer, NLDAS), a 15-channel multi-forcing LSTM, three δHBV
members, plus two new members described below. Streams are seed-averaged, then
combined.

**with-q track (in-house encoder–decoder "MB-LSTM").** 365-day encoder over
weather plus observed discharge with a missing-mask; 14-day decoder over
forecastable weather; hidden 256; 27 CAMELS static attributes; quantile head.
Four single-forcing members (Daymet, Maurer, NLDAS, AORC) × 5 seeds.

**Two members that earned their place, both by a mechanism we did not predict:**

- **multi5** adds four sparse GHCN station channels. It is a *worse* member in
  isolation (−0.0062) but adds **+0.0023** to the ensemble across five seeds, with
  **93.6% of basins improving** and a bootstrap CI excluding zero. It does not act
  through snow, as predicted: gains are flat across snowy/arid cohorts. It is a
  decorrelation member, and we name it that rather than "the snow member."
- **multi6** adds three Livneh soil-moisture layers. Its global effect is marginal
  and seed-sensitive (+0.0008/+0.0011/+0.0019 over three seeds), but its **arid
  gain is +0.0208 and worst-decile gain +0.0240 — five to eight times the humid
  gain**, consistently across every seed. We ship it as an **arid specialist** and
  describe it that way.

### 3.3 Combination

Streams are combined with inverse-MSE weights, `w_i ∝ MSE_i^(−θ)` shrunk toward
equal weighting (θ=4.0, λ=0.25; Bates–Granger with shrinkage). **Weights are fit
on 196,636 TRAIN-period rows and never on the scored window.** An assertion in the
scorer refuses to run if the training frame is empty — a guard that caught the
first version of this patch attempting to fit on the test frame.

Equal weighting costs −0.0025 relative to inverse-MSE here, because member quality
is heterogeneous: the δHBV family is simultaneously the most decorrelated and the
weakest, so at equal weight it drags more than it diversifies.

---

## 4. The observational ceiling

### 4.1 Derivation

Let `y` be true discharge and `ŷ = y·ε` the gauge observation, with
`ε = exp(δ)`, `δ ~ N(−σ²/2, σ²)` (multiplicative, median-unbiased). A perfect
model predicting `y` exactly is scored against `ŷ`. Taking expectations,

```
E[(ŷ − y)²] = E[y²]·(e^{σ²} − 1) = M·(e^{σ²} − 1)
```

and since NSE = 1 − MSE/Var(ŷ),

```
NSE_ceiling = 1 − (M/V)·(e^{σ²} − 1)          [constant σ]
```

Monte Carlo simulation (perturb observed discharge, score the unperturbed truth
against it) agrees with this to **+0.0003 at σ = 0.13**.

**This form is only valid for constant σ, and we initially misapplied it.** Gauge
error is not constant — it is larger at low flow — so every feasibility
calculation uses a flow-dependent σ(y). Substituting `mean(σ²)` into the constant-σ
formula is wrong twice over, and both errors push the ceiling *down*: it applies
the low-flow σ to the high-flow mass that dominates `M = E[y²]`, and it divides by
`Var(truth)` when NSE divides by the variance of the series being scored against,
which is the noisier observation. The correct expression is

```
NSE_ceiling = 1 − E[y²(e^{σ(y)²} − 1)] / ( E[y² e^{σ(y)²}] − (E[y])² )
```

Validated per basin against Monte Carlo (200 basins, 25 replicates each):

| form | central scenario | optimistic |
|---|---|---|
| naive `mean(σ²)`, median abs. error vs MC | **0.0361** (20% of basins within 0.01) | 0.0285 (0%) |
| **corrected**, median abs. error vs MC | **0.0034** (82% within 0.01) | **0.0013** (100%) |

We report this because the earlier form understated every ceiling by roughly
0.03–0.04 and produced a materially different conclusion. The original validation
was performed at constant σ and was correct as far as it went — it simply was not
a validation of the code path in use. **Validate the path you run.**

Three further assumptions we checked rather than assumed.

**Persistence.** Rating curves shift on multi-month timescales, so daily gauge
errors are not independent. Simulating AR(1) error from ρ = 0 to ρ = 0.98 moves
the median ceiling only 0.9524 → 0.9565: persistence widens the confidence band
without biasing the point estimate.

**Symmetry.** The multiplicative lognormal parameterisation is median-unbiased by
construction, and the simulated mean bias is +0.0005.

**The shape of σ(q).** Because the corrected form is flow-weighted, it could in
principle be sensitive to *how* σ varies between its low- and high-flow endpoints
— and our choice (linear in log-flow, between the 5th and 95th flow percentiles)
is a modelling assumption. Holding the endpoints fixed at the central scenario
and varying only the shape:

| σ(q) shape | median ceiling |
|---|---|
| linear in log-flow (ours) | 0.9554 |
| step function at the median | 0.9574 |
| constant at the flow-weighted RMS of ours | 0.9554 |
| constant at the high-flow value | 0.9579 |

These agree within 0.0025, while moving the *endpoints* from all-high to all-low σ
spans 0.0695 — nearly thirty times more. **The ceiling is governed almost entirely
by the high-flow σ, and barely at all by the interpolation shape.** That is a
desirable property: it means the result depends on the one quantity our
extrapolation measurement (§4.2) actually constrains. The single exception is a
σ linear in *raw* flow (0.9292), which we reject on physical grounds — in a
right-skewed series it holds σ near its low-flow value across almost the entire
range.

`M/V = 1 + mean²/variance`, which yields a result we initially got backwards:
**flashy basins have *lower* M/V (1.17 vs 1.57) and therefore a *higher*
break-even σ (0.381 vs 0.293)** — they tolerate more gauge error before saturating,
not less, because large peaks inflate variance faster than mean-square.

### 4.2 Which σ applies to CAMELS — extrapolation incidence

⚠️ **Superseded in part by §4.4.** This section measures how often peaks require
rating-curve extrapolation, then *selects* a σ scenario from the literature on
that basis. §4.4 measures σ directly and finds a different regime, with the
flow-gradient running the opposite way. The extrapolation statistics below stand;
the scenario choice they motivated does not.

The ceiling is only as good as its σ. Rather than assume, we measured the
mechanism the literature says drives high-flow uncertainty: whether the peak
required **extrapolating the rating curve** beyond any direct measurement. Using
USGS field measurements for all 530 retrievable gauges, filtered to the CAMELS
window:

| bar | fraction of basin peaks beyond it |
|---|---|
| highest **direct** measurement (any rating) | **38/530 = 7.2%** |
| highest **good/excellent-rated** measurement | **229/530 = 43.2%** |

Against Coxon et al. (2015), who could not compute high-flow uncertainty for
**44%** of UK station-groups because stage exceeded the highest gauging, **CAMELS
is ~6× better measured at peaks**. Among the 38 extrapolated basins the exceedance
is mild — median **1.19×** above the highest gauging, not Kiang's 41–200% regime.
Flashy basins are only slightly worse than steady ones (9.1% vs 5.3%), so the
concern that flashy-basin headroom is an extrapolation artifact is **not
supported** — a conclusion independently reached by the break-even-σ analysis.

⇒ The **optimistic** σ scenario is the better description of CAMELS-531. We
nonetheless report every scenario, because scenario choice drives the answer
(ceiling 0.97 → 0.69 across the range), and we report the 43.2% figure alongside
the 7.2% because quoting the latter alone overstates how well peaks are
constrained.

### 4.3 Where each record sits

⚠️ **The scenario columns below are superseded by the measured σ of §4.4**, which
puts the no-q ceiling at **0.73–0.95** rather than 0.919/0.976 and therefore
removes the "0.845 needs only 6% of headroom" conclusion. The table is retained
because the *sensitivity* argument at its foot is the part that survives — and
§4.4 strengthens it.

All figures below use the **corrected** closed form.

| | **no-q** (Li/Song protocol) | **with-q** (Kratzert protocol) |
|---|---|---|
| this work | **0.8363** (held-out) | **0.888355** ⚠️ (was 0.9203; see §5.2 correction) |
| prior record | 0.8294 | 0.879 |
| **central: all basins → ceiling** | **0.9186** | **0.9453** |
| **optimistic: all basins → ceiling** | **0.9755** | **0.9772** |
| basins saturated (central) | 110/531 (21%) | 266/531 (50%) |
| basins saturated (optimistic) | **1/531** | 84/531 (16%) |
| target | 0.845 needs **6%** of headroom (optimistic) ✅ | 0.95 **reachable** under both scenarios |

The headroom is real but modest in absolute terms: **+0.025** for with-q and
**+0.082** for no-q against the central-scenario bound. Half of all with-q basins
are already at or above what their gauge can reward, which is why broad gains are
hard to come by there.

⭐ **The result that survives is about sensitivity, not saturation.** The bound
moves from 0.977 to 0.756 as σ goes from 0.13 to 0.41 — a range spanning the
entire published literature on discharge uncertainty. The gradient d(ceiling)/dσ
runs from −0.31 to −1.17 across that range. **Which σ regime CAMELS occupies
determines the answer more than any modelling choice does**, which is why §4.2's
measurement is the load-bearing contribution of this paper rather than a
supporting detail.

### 4.4 σ measured directly — and it overturns §4.2's scenario choice

⚠️ **This section supersedes the σ selection in §4.2.** There we measured the
*incidence* of rating-curve extrapolation (7.2% of peaks) and then still **chose**
a σ regime from the UK literature. Here we measure σ itself, on the same gauges
we score against, and the answer is not the regime we chose.

**Method.** USGS publishes the **field gaugings** behind every rating curve —
direct current-meter and ADCP measurements, ~1,000–2,000 per station back to the
1950s. We compare each gauging to the **published daily value for the same day**.
That difference is the total error in the series NSE is computed against:

    σ_total² = measurement error + rating-curve error + shift error

Gauge *quality ratings* capture only the first term. The comparison captures all
three, which is what the ceiling requires.

**Result** (**144,914 matched pairs across 523 of 531 basins** — the complete
fetch; 5 basins lack retrievable gaugings and 3 fall below the 40-pair minimum):

| flow band | n | robust σ (MAD) | plain SD | pairs disagreeing >2× |
|---|---|---|---|---|
| p00–25 | 35,895 | 0.0503 | 0.2252 | 1.16% |
| p25–50 | 36,186 | 0.0442 | 0.2016 | 0.90% |
| p50–75 | 36,278 | 0.0455 | 0.2256 | 1.22% |
| p75–90 | 21,777 | 0.0614 | 0.3310 | 2.76% |
| **p90–99** | 13,061 | **0.1213** | 0.4538 | **5.50%** |
| **p99+** | 1,717 | **0.2509** | 0.5599 | **12.93%** |

⭐⭐ **σ RISES steeply with flow — roughly 7× from typical flows to the top 1%.**
The median gauging agrees with the published value to 2.5%; the top percentile
disagrees by ~34%.

⚠️⚠️ **This reverses the σ shape assumed throughout §4.** Our `sigma_series`
interpolates σ *downward* with flow (0.30 low → 0.18 high), on the reasoning that
peaks are the well-gauged regime. Measured, the opposite holds. Because the
corrected ceiling is **flow-weighted** (§4.1), weighting by q², the direction of
that gradient matters more than its magnitude: every ceiling computed with the
old shape is wrong in *direction*, not merely level.

**The resulting ceiling**, corrected flow-weighted form, validated against Monte
Carlo to ≤0.001 across σ = 0.02–0.30 (so the formula is not the uncertainty — σ
is):

| basis | no-q ceiling |
|---|---|
| **measured σ, robust (MAD)** | **0.9493** |
| **measured σ, plain SD** | **0.7280** |
| assumed "optimistic" (§4.2's choice) | 0.9786 |
| assumed "central" | 0.9594 |

⇒ **The measured ceiling is 0.73–0.95**, and our held-out **0.8363 sits inside
that band.** The robust bound (0.949) is close to the assumed "central" scenario
(0.959) while the SD bound (0.728) is far below it, so the assumed scenarios were
not so much wrong in level as **too narrow**: they excluded the tail behaviour
the gaugings actually show. The §4.3
statement that "0.845 needs 6% of headroom" does not survive this measurement.

**Why the band is wide, and why we report it rather than a point.** The gap
between robust σ and plain SD is entirely rare, extreme disagreements: **1.4% of
pairs differ by >2×**. These are not artifacts. Their rate rises monotonically
with flow — **~1% below the median rising to 12.9% in the top percentile** — and
falls monotonically with gauging era (2.19% pre-1970 → 0.82% post-2010). That is
the signature of rating-curve extrapolation failing at peaks, exactly the
mechanism Coxon and Kiang describe. Example: station 01031500, 1982-02-23, a
gauging of **4,190 cfs** against a published daily of **200 cfs**, on a day the
hydrograph did not move.

Restricting to days with a flat hydrograph (to suppress the fact that a gauging
is instantaneous while the daily value is a 24-hour mean) moves robust σ only
from 0.052 to 0.043 at typical flows and 0.124 to 0.094 at high flows — so
within-day variability explains a minority of the scatter, and the robust reading
is the defensible one rather than an artifact of filtering.

⇒ **The honest headline is the range.** Robust σ excludes real events; plain SD
includes some genuine sub-daily variation. The truth lies between, and the
sensitivity is dominated by peak-flow σ alone: holding typical-flow σ at 0.05 and
sweeping σ(p99+) from 0.10 to 0.46 moves the ceiling from 0.990 to 0.822.

### 4.5 Error magnitude is learnable; error sign is not

A natural response to a ceiling set by observations is to *repair the
observations*. The arithmetic supports it — removing half the gauge noise lifts a
σ=0.15 ceiling from 0.970 to 0.993. We tested whether it is achievable, and it is
not, for a measurable reason.

**Decomposition of the log residual** (144,914 pairs, 523 gauges):

| component | share |
|---|---|
| between-gauge (a learnable per-gauge **bias**) | **0.9%** |
| within-gauge (per-measurement noise) | **99.1%** |

There is essentially no systematic per-gauge offset to correct. Nor is the error
**persistent**: across 24,193 consecutive gauging pairs, the autocorrelation of
the residual is **+0.071** for gaugings less than two months apart and ~0.01
beyond. A corrector needs correlation ρ ≳ 0.3 with the true error merely to break
even; below that it *adds* variance. At the measured ρ ≈ 0.07 the best attainable
gain is **+0.0035**, and applied without optimal shrinkage it **costs −0.019**.

⛔ **And a working corrector would still be inadmissible here.** The tempting
source of correction signal is the model itself — ensemble means, neighbouring
gauges, physical plausibility. Any of these makes the target a function of the
prediction: NSE ceases to measure skill, the number is no longer comparable to
Li/Song, Kratzert or Nearing (all scored against the unmodified USGS series), and
the claim becomes unfalsifiable, since a better model would "correct" the data
further and raise its own score.

⭐⭐ **What *is* learnable is the error's magnitude.** Per-gauge σ spans
0.028–0.109 (median 0.054) and is **stable**: σ estimated on a station's first
half predicts its second half at **r = 0.50** (n = 114 stations with ≥80 pairs).
It is also predictable from stable catchment properties — most strongly **log
basin area, r = −0.58**: small catchments are gauged substantially worse, with
p_mean, aridity and the station's own Fair/Poor rating fraction contributing.

⇒ **One cannot predict which way a reading is wrong, but one can predict how
wrong a given gauge typically is.** That supports two uses, both of which are
metric contributions rather than data corrections:

1. **Normalised skill** — report NSE against each basin's own measured ceiling.
   Computed with both sides on the held-out frame (531 basins, 183,195 rows; the
   calculation reproduces the record exactly at 0.8363), **the record captures
   84.0% of achievable under the robust σ reading and 89.0% under the plain-SD
   reading**, with **0.2%–9.9%** of basins already at or above their own ceiling.
   ⚠️ The same calculation on the *train-side* val slice reads 95.4%/99.4% — a
   ~10-point overstatement, because it pairs an easier window's median with that
   window's ceilings. Always state the frame.
2. **Gauge-quality stratification** — per-basin ceilings vary widely, so a
   benchmark can be reported on the subset where the observations can still
   resolve model differences.

⚠️ Both must be pre-registered and computed **independently of model output**; a
stratification tuned on model performance is oracle selection, as fatal as
correcting the target.

---

## 5. Results

### 5.1 No-discharge: a held-out record of 0.8363

The configuration was **frozen before the test window was touched**, and the
held-out evaluation was a **single pre-registered query**, spent once:

| | median NSE |
|---|---|
| 7-stream base, equal weight | 0.8305 |
| 9-stream (+multi5, +multi6), equal weight | 0.8338 |
| **9-stream, inverse-MSE weighted** | **0.8363** |
| Li/Song 2025 published | 0.8294 |

531 basins, 183,195 rows, window 1995-10-01 → 2008-12-07, bootstrap 95% CI
**[0.8263, 0.8469]**. Fitted weights put **44% of the total weight on the two
newest members** (multi5 0.249, multi6 0.191).

Verification, because a held-out number is worth only its audit: weights sum to 1
and are non-degenerate; observed discharge is bit-identical across all nine
streams (max |diff| = 0 over 183,195 rows); the TRAIN weight-fitting frame and the
TEST frame share **zero** (basin, date) keys; and an independent sum-form NSE
implementation reproduces 0.8363 exactly.

**A prediction we recorded in advance and got wrong.** We expected the held-out
number to land *below* the train-side 0.8347, for two stated reasons (fitting
noise; the truncated Maurer window). It came in **+0.0016 above**. We report this
because the direction was asserted rather than derived. We do **not** present
0.8363 > 0.8347 as an improvement: they are different windows on different data.
The honest statement is that held-out performance did not degrade, which is
unusual and worth one sentence, not a claim.

> ## ⛔⛔ CORRECTION NOTICE (2026-09-05) — EVERY WITH-Q NUMBER BELOW IS SUPERSEDED
>
> **The with-q figures in this draft (0.9203, 0.9253, 0.9058, 0.9016 and every quantity derived from
> them — ceilings, headroom, saturation fractions, cohort splits) are RETRACTED.** Two independent
> defects, found 2026-09-04 and recorded in `benchmarks/EXPERIMENTS.md` §LEDGER 51:
>
> 1. **Training leak.** Every with-q member trained after 2026-08-01 was run without `--train-start`, so
>    the 1989-10-01..1999-09-30 **test decade was inside the training set** (6.7–8.1M training windows
>    against 1.74M when guarded). The August launchers each copied the July `--val-*` line and dropped the
>    `--train-start` line beside it.
> 2. **Evaluation frame.** The whole ladder — including the July *guarded* members — was scored on a
>    **stride-14 subsample (260 rows/basin)**, which is **systematically optimistic by +0.036** and not
>    phase luck: all 14 phases score 0.870–0.894 against 0.847 on all days, monotone in sparsity. Nearing
>    2022 scores **every** daily observation.
>
> **The corrected result, on Nearing's own split and his all-observations frame:**
>
> | | day-1 median NSE, 531 basins |
> |---|---|
> | **RiverWatch2 with-q (5 members × 5 seeds, equal weight)** | **0.888355** |
> | Nearing et al. 2022 | 0.879 |
> | **margin** | **+0.009355** |
>
> The record still stands, by a smaller and honest margin. §5.2's *readout* finding survives (the quantile
> midpoint beats the median); its *level* does not. **Do not quote any with-q number from the sections
> below without recomputing it on the guarded, all-days frame.**

### 5.2 Discharge assimilation: 0.9203, and a readout that matters

| Model | median day-1 NSE | #basins |
|---|---|---|
| Nearing et al. 2022 (AR-LSTM) | 0.879 | 531 |
| **RiverWatch2 with-q ensemble** | **0.9203** | **531** |

4 forcings × 5 seeds, plain mean, no test-fitted weights; KGE 0.8861; CI95
[0.9091, 0.9264].

Two honest decompositions. **(a)** Seed depth is saturated: the fifth seed bought
+0.0005 against +0.0059 for adding a fourth *forcing*. New information beats more
of the same. **(b)** The point prediction is the **midpoint of the 10th/90th
predictive quantiles**, not the median; on identical frames `(ylo+yhi)/2` scores
0.9203 where `ymed` scores 0.9177. This is a readout choice worth more than
several of our member-level gains, and it is free — we flag it because we
ourselves initially mis-attributed the 0.9203/0.9177 difference to ensemble
composition. A sweep of alternative quantile blends found nothing reliably better:
the best (+0.0006) is inside noise and inconsistent across a temporal split, so we
keep the record's readout.

### 5.3 Where the error lives

Both tracks reduce to the same place. **96.5% of with-q squared error and 88.4% of
the worst no-q basins' error falls on top-10%-flow days.** Shrinking only peak-day
error, holding all else fixed, moves the median dramatically: a 20% reduction
takes with-q from 0.9177 to 0.9440 and no-q past its target.

Headroom is **highly concentrated**: 353/531 with-q basins sit below their
ceiling, but the **top 50 hold 56%** of all remaining *summed* headroom and the
top 100 hold 73%. Those 50 basins have median NSE **0.601** against 0.925 for the
rest, and median flashiness 52 against 17. Several are negative. We verified they
are genuine ephemeral desert catchments, not corrupt data: 08194200 is dry 76% of
days and then floods to 2,550 cfs.

### 5.4 A metric-definition error: summed headroom is not the reported number

The sentence above is true and, for the purpose of the benchmark, almost
irrelevant. We report the **median** of per-basin NSE, which is a **rank
statistic** determined by the basins near rank 266 of 531. Improving a basin at
rank 20 changes it by exactly nothing.

| improvement applied | change in median (with-q) |
|---|---|
| top-50 target basins, +0.05 NSE | **+0.00000** |
| top-50 target basins, +0.20 | +0.00134 |
| top-50 target basins, +0.35 | +0.00692 |
| **62 near-median basins, +0.05** | **+0.01578** |

The top-50 target basins occupy ranks 1–66; **not one is above the median rank**.
The same holds on the no-q track, where improving the worst 50 basins by +0.10
moves the median by 0.00000. The two cohorts have **zero overlap** and are
physically distinct: near-median basins are wetter (aridity 0.78), snowier
(frac_snow 0.13) and smaller (298 km²), while the worst basins are arid (1.14),
dry and large (402 km²).

We report this because it invalidates a targeting strategy we ourselves pursued —
the arid/flashy specialist programme optimises a quantity (summed per-basin
headroom) that the benchmark does not report. It does **not** affect the ceiling
analysis, which computes a median throughout, nor the measured ensemble deltas of
multi5/multi6, which were always median deltas. It does mean that **only broad
gains move this metric**: multi5 works precisely because 93.6% of basins improve.

Diagnosing the near-median cohort sharpens the σ-sensitivity result considerably.
Under the central scenario **70 of the 78 already sit at or above their own
gauge-error ceiling** (median headroom −0.052), so the metric has very little room
to move. Under the optimistic scenario — which our 7.2% extrapolation measurement
supports — only **8 of 78** are saturated and median headroom flips positive to
**+0.022**, with the achievable bound rising from 0.9522 to 0.9773 on that window.

**The same cohort is either finished or has real room, depending entirely on which
gauge-error regime holds.** That is the sharpest statement this analysis
supports, and it is why measuring σ empirically matters more than another
modelling round.

We also attempted to exploit the finding directly, and failed: weights fit on the
near-median cohort, dHBV-dropping, rank-based weights, and direct
median-maximising coordinate ascent all scored **negative** out of sample
(−0.0001 to −0.0005). The reason is that near-median membership is defined by a
rank, and ranks do not persist: only **27%** of the cohort is shared between the
fitting and validation windows. A near-median specialist cannot be targeted in
advance.

Combining the rank constraint with the corrected ceiling gives the operational
map. Effort can only pay where a basin is *both* below its gauge-error ceiling
(the observations can reward improvement) *and* near the median rank (it can move
the reported number). On the held-out no-q frame:

| | central σ | optimistic σ |
|---|---|---|
| basins below their ceiling | 421/531 | 530/531 |
| basins near the median rank | 44/531 | 44/531 |
| **both** | **44/531** | **44/531** |
| median gain if only those reach their ceiling | **+0.019** | **+0.019** |
| median gain if all below-ceiling basins do | +0.082 | +0.139 |

All 44 near-median basins are below their ceiling under both scenarios, and
raising only those to their ceilings would take the no-q median from 0.8363 to
roughly 0.855 — past the 0.845 target. They are also **unremarkable catchments**
(median flashiness 16.4 against 17.4 overall), not the flashy, arid, ephemeral
extremes this campaign spent months targeting.

This yields the sharpest statement the analysis supports, and it is not the one
we expected. The observations have room almost everywhere (421 of 531 basins).
Only about 44 basins can move the reported statistic. And those 44 cannot be
identified in advance, because rank does not transfer. **The binding constraint
on CAMELS median-NSE progress is the metric itself — not the observations, and
not the models.**

### 5.5 A lever that looked alive, and the screen that killed it

Every combination and post-processing lever we tested this round failed on honest
splits (regime-conditional weights −0.0001, cohort-conditional −0.0004, robust
and trimmed combination below the weighted mean, asinh-space averaging −0.0014,
ridge stacking −0.0009, per-regime stacking −0.0140). One thing did not.

The residual of our ensemble at one basin is correlated with the concurrent
residual at nearby basins, and the correlation decays cleanly with distance:

| distance | residual corr (all days) | **residual corr (peak days)** |
|---|---|---|
| 0–50 km | 0.2426 | **0.2875** |
| 100–200 km | 0.0587 | 0.0725 |
| 500–1000 km | 0.0011 | 0.0095 |
| >1000 km | −0.0010 | 0.0057 |

The lift is **larger on peak days** than overall — precisely where our error
lives. Correcting each basin from its three nearest neighbours (excluded if
within 25 km *and* differing more than fivefold in area, our initial nesting
guard) yielded a median gain of **+0.0041**.

**That result did not survive its own pre-registered screen, and the failure is
the most useful part of this section.** Before committing GPU to a
neighbour-assimilating member, we audited the neighbour definition against USGS
hydrologic unit codes and found **163 basin pairs sharing an 8-digit HUC** — the
same hydrologic sub-basin, hence very likely the same river system — that the
distance-and-area guard did not catch. Two gauges 60 km apart with a threefold
area ratio pass that guard and are still nested, and a nested neighbour's
discharge is partly *the same water* as the target's.

| nesting filter | median gain | improved | bootstrap CI95 |
|---|---|---|---|
| distance/area only (original) | **+0.00410** | 75.1% | [−0.00397, +0.01292] |
| **+ same-HUC8 excluded** | **+0.00074** | 68.2% | [−0.00669, +0.01004] |
| + same-HUC4 excluded (strictest) | −0.00010 | 68.0% | [−0.00738, +0.00881] |
| **control: shuffled neighbours** | −0.00023 | 45.6% | [−0.00756, +0.00725] |

About **82% of the apparent gain was nesting**, and every clean configuration has
a confidence interval straddling zero, statistically indistinguishable from the
shuffled-neighbour control. We withdraw the gain.

Two things survive, and both matter. The **correlation structure itself is
real** — nesting cannot produce a smooth monotone decay across 140,715 pairs —
and the **own-lag control gives exactly +0.00000** at 50.0% of basins improved,
so what correlation exists is spatial rather than temporal autocorrelation. The
honest conclusion is that neighbouring gauges' residuals are correlated, but that
correlation is **not exploitable for a median gain** once same-river pairs are
removed; what remains is largely shared forcing and state error that a neighbour
cannot resolve.

We report this in full because the same trap is available to anyone using a
distance-based neighbour definition without an explicit same-river exclusion, and
because it is consistent with Kirschstein & Sun (ICML 2024), who found basin
topology gives no benefit in graph models of streamflow. It is also the one point
in this campaign where a pre-registered screen killed a candidate *before* the
GPU spend rather than after.

---

## 6. Negative results, and the mechanism they share

These are reported because they are the paper's evidence, not its failures. Each
was pre-registered with a falsifier.

### 6.1 The error is scatter, not bias

Peak-day error decomposes as median relative bias **−0.073** against median
relative scatter **0.292** — a ratio of **0.270**. Scatter dominates bias ~3.7:1.
The obvious correction fails out of sample: a per-basin high-flow multiplier fit
on the first half of the record *lowers* median NSE (0.9256 → 0.9167), improving
only 246/531 basins — chance. **The model is randomly wrong on peaks, not
consistently wrong**, so no bias correction, reweighting, or recombination can
help.

### 6.2 Distributional heads fail, on two architectures, for a diagnosed reason

If peak error is uncertainty, a distributional output should help. It does not.
CMAL and GMM heads (Google Flood Hub's design choice) were each trained against a
matched point model:

| | point model | CMAL | GMM |
|---|---|---|---|
| peak-day NSE | **0.4901** | 0.3939 | 0.3637 |
| ensemble delta | — | −0.0018 | −0.0015 |

Three findings make this a mechanism rather than a null. **(a)** GMM converged to a
**better NLL** (−2.735 vs −2.643) while being worse on every task metric — so this
is not under-training or optimization failure. **(b)** CMAL's p90 readout is nearly
unbiased on peaks (+0.085) yet scores **0.1029** peak-day NSE, independently
reproducing the scatter-not-bias result. **(c)** Interval coverage is *better* on
peak days (0.503) than overall (0.399) in both heads — falsifying the premise that
the model is unaware of its peak uncertainty. Caveat: one seed and 30 epochs each,
with both NLLs still improving at cutoff.

### 6.3 Differentiable physics does not rescue the extremes here

Song et al. (2026) report that δHBV1.1p beats LSTM on CAMELS events with return
period ≥5 years (+0.06 median NSE on those events; lower peak error in 80% of
cases), attributing it to mass balance constraining peak underestimation. This is
the one published result aimed squarely at our surviving error mode, and it was
free for us to test: **our δHBV members already are δHBV1.1p** — 16 parallel
components, all three dynamic parameters (BETA, K0, BETAET), and the combined
`0.5·MSE + 0.5·MSE(log₁₀)` loss.

The direction replicates; the magnitude does not. Pooled NSE by flow regime,
averaged within family:

| regime | δHBV | LSTM | difference |
|---|---|---|---|
| all days | 0.8645 | 0.8872 | −0.0227 |
| top 10% | 0.8441 | 0.8653 | −0.0213 |
| top 1% | 0.8246 | 0.8427 | **−0.0180** |

The gap narrows monotonically toward the extremes — consistent with the proposed
mechanism — but never crosses zero, and `lstm_multi5` remains the single best
stream at every regime including the top 1%. The deciding test is the ensemble:
upweighting δHBV **on peak days only** degrades it monotonically (×1.5: −0.0009;
×2: −0.0012; ×3: −0.0017). Inverse-MSE weighting has already given the family the
weight it earns.

**Scope of this negative, stated precisely.** We tested our already-trained
members on the Li/Song no-q protocol, using flow percentiles as a proxy for return
period — a 13.2-year record cannot resolve a 5-year return period per basin. We
did not retrain under Song's exact configuration, and their result is
event-conditional where ours is a pooled and median comparison. A defender of that
paper could reasonably object on those grounds. What we can say is that the
architecture they advocate, as trained here, carries no extreme-event skill that
our weighting is discarding.

### 6.4 What else is closed

- **Timing.** A stride-1 probe across 16 basins finds shift 0 optimal for **all
  16**; the per-basin oracle shift buys +0.0000. The model is on time and too
  small. Any routing-lag member would target the wrong thing.
- **Model family.** Cross-family residual correlation (LSTM↔δHBV, 0.6094) is
  barely below within-family (0.6488) — architecture is not the diversity axis we
  assumed. Shared error is a property of the data and target.
- **Precipitation source.** Every member scoring ≥0.73 has gauge-calibrated
  precipitation; the one whose precipitation is pure model output (CONUS404, WRF)
  scored **0.5094** despite the best decorrelation we ever measured. The shared
  COOP/GHCN gauge base is not a flaw to escape — it is the load-bearing signal,
  and a structural bound on what this family can achieve.
- **Sub-daily intensity**, delivered as daily summary features, gives +0.0002 with
  a CI straddling zero: better on storm days (−9% RMSE), worse on ordinary days
  (+33%), cancelling in a variance-normalized metric.
- **LSTM output saturation.** Baste et al. (2025) report an architectural output
  ceiling in LSTMs. On our top-50 target basins the ensemble reaches only 0.536 of
  the observed maximum — which looks like strong confirmation. It is not. In
  normalized units the simulated maxima are **1.51× *more* spread** than observed,
  not compressed; single members routinely *overshoot* observed maxima elsewhere
  (ratio 1.058); and correlation with absolute peak magnitude is ≈0. What the data
  show instead is `corr(sim_z, obs_z) = −0.13` on those basins against +0.41
  elsewhere: **the model's peak magnitude carries no information about the true
  peak magnitude there.** Scatter again, from a third independent direction.

---

## 7. What changed in this revision, and why

The July draft of this paper reported with-q **0.9016** and no-q "~0.80 pooled /
0.829 day-1" on a 177-basin screen, and contained no ceiling analysis. Both
numbers have been superseded (0.9203 on all 531; 0.8363 held-out on all 531), and
the paper's claim has changed from "we beat the record" to "here is the ceiling,
and here is where the records sit against it."

**This revision adds §4.4–4.5 and, in doing so, corrects the paper's own
load-bearing claim.** §4.2 asserted that σ was "measured, not assumed"; it
measured extrapolation *incidence* and then chose a σ scenario from UK
literature. §4.4 measures σ directly, from 144,914 field-gauging-vs-published-daily
pairs on the CAMELS gauges themselves, and finds:

- the assumed σ(q) **shape is backwards** — σ rises ~7× with flow rather than
  falling, and the ceiling is flow-weighted, so the direction matters;
- the measured ceiling is **0.73–0.95**, below both scenarios we had reported,
  with our held-out 0.8363 inside that band;
- consequently "0.845 needs 6% of headroom" (§4.3) does not survive.

§4.5 then closes the natural follow-up — repairing the observations — on measured
grounds (error persistence ρ ≈ 0.07 against a break-even of ≈0.3) and on the
methodological ground that any model-informed correction is circular. It replaces
that idea with the part that *is* learnable: error **magnitude** is a stable,
predictable gauge property (split-half r = 0.50; log-area r = −0.58) even though
error **sign** is not.

We also correct two of our own errors, in the spirit of §6:

1. **The 0.9203 vs 0.9177 difference is a readout, not a configuration.** It was
   previously recorded as a "thinner ensemble." It is `(ylo+yhi)/2` versus `ymed`
   on identical data.
2. **A control run disagreed with a committed number, and the disagreement was
   real.** Our equal-weight 7-stream control scored 0.8305 where the committed
   Li/Song reproduction says 0.8298. The cause is seed depth in one stream (five
   seeds versus three), worth +0.0007 on an identical 183,195-row frame. Both
   numbers are correct and answer different questions. We keep a known-value
   control in every ensemble test precisely so a 0.0007 drift surfaces.

---

## 8. Conclusion

We derive a maximum achievable NSE from discharge observation error, measure
which error regime CAMELS actually occupies rather than assuming one, and place
two records against the result.

On the **discharge-assimilating** protocol, median day-1 NSE **0.9203** on all 531
basins exceeds the prior published record by +0.041, against a central-scenario
bound of **0.9453** — headroom of about +0.025, with **half** of all basins
already at or above their own ceiling. On the **no-discharge** protocol, a
held-out **0.8363** exceeds the prior record of 0.8294 — against a **measured**
bound of **0.73–0.95** (§4.4), inside which it already sits.

Three findings outlast those numbers. The first is that the bound can be
**measured rather than assumed**, and that doing so changes it. Comparing 144,914
USGS field gaugings against the published daily values they underpin gives σ
directly, on the gauges we score against: σ **rises ~7× with flow** (0.05 typical
→ 0.34 at the top percentile), which is the opposite of the shape we and the
literature-derived scenarios assumed, and yields a ceiling **below** every
scenario we had reported. Establishing which regime CAMELS occupies was worth
more than another modelling round — and it was a measurement, not a model.

The second is that the observations cannot be repaired to escape the bound, but
they can be **characterised**. Gauge error carries almost no learnable per-station
bias (0.9% of residual variance) and almost no persistence (ρ ≈ 0.07 between
consecutive gaugings), so no admissible corrector reaches the ρ ≈ 0.3 needed to
break even — and any corrector informed by model output would make the target a
function of the prediction. What *is* learnable is the error's **magnitude**:
per-gauge σ is stable across halves of a station's record (r = 0.50) and
predictable from catchment properties, most strongly log area (r = −0.58). That
supports reporting skill **relative to each basin's own ceiling** — our 0.8363 is
**84–89% of what its gauges can reward** — rather than pretending the raw number
is comparable across stations of very different observational quality.

The third is that the reported metric discards most of what is available.
**421 of 531 basins sit below their gauge-error ceiling, but only ~44 can move a
median**, and those 44 cannot be targeted in advance because rank does not
transfer. A field optimising median NSE on CAMELS is therefore competing over a
statistic that is insensitive to most of the improvement its data could still
reward. We would rather see that stated plainly than see another decimal place
added to it.

The remaining error, in both tracks, is peak-magnitude **scatter** — established
here from four independent directions (bias/scatter decomposition, failed
out-of-sample peak rescaling, two distributional heads, and a refuted output-
saturation hypothesis). That is the same quantity the ceiling argument is about,
which is why we think the two halves of this paper belong together: the reason
peak error resists modelling is closely related to the reason the observations
cannot reward removing it.

**Limitations.** The ceiling is our own derivation, not a citation. We got the
derivation wrong once, in a way that changed the conclusion (§4.1): a constant-σ
closed form applied to flow-dependent σ understated every ceiling by 0.03–0.04
and produced a "the benchmark is finished" reading that the corrected form does
not support. We then got the *σ* wrong, in a way that changed it again (§4.4):
what we described as a measured error regime was a scenario selected on
extrapolation incidence, and its flow-gradient ran the wrong way.

The direct σ measurement carries its own caveats, and they set the width of the
0.73–0.95 band rather than its centre. A field gauging is **instantaneous** while
the published value is a **24-hour mean**, so part of the plain-SD scatter is
genuine sub-daily variation and not error; the robust (MAD) estimate excludes
that but also excludes rare, real rating failures, which is why we report both
ends rather than a point. The extreme disagreements are concentrated at high flow
(9.4% of top-5%-flow pairs differ by >2×) precisely where the ceiling is most
sensitive, so the peak-flow σ is the single quantity most worth tightening. The
figures here rest on 523 of 531 basins (five lack retrievable gaugings, three
fall below the 40-pair minimum), and gaugings are unevenly distributed across
eras and flow regimes. Multi-hundred-gauge uncertainty base rates in the prior literature are
UK (Coxon; CAMELS-GB); Kiang et al. is USGS-coauthored but covers three sites. The two tracks use different protocols and windows and must not be
compared. Per-basin with-q NSE rests on 261 points and is fragile. And the no-q
scored window is ~13.2 years, not 15, because Maurer ends in 2008.

---

## References

1. **Newman, A. J., et al. (2015).** CAMELS. *HESS* 19, 209–223. https://doi.org/10.5194/hess-19-209-2015
2. **Addor, N., et al. (2017).** The CAMELS data set. *HESS* 21, 5293–5313. https://doi.org/10.5194/hess-21-5293-2017
3. **Kratzert, F., et al. (2019).** Toward Improved Predictions in Ungauged Basins. *WRR* 55. https://doi.org/10.1029/2019WR026065
4. **Kratzert, F., et al. (2021).** Leveraging synergy in multiple meteorological data sets. *HESS* 25, 2685–2703. https://doi.org/10.5194/hess-25-2685-2021
5. **Feng, D., Fang, K., & Shen, C. (2020).** LSTM with data integration at continental scales. *WRR* 56, e2019WR026793. https://doi.org/10.1029/2019WR026793
6. **Nearing, G., et al. (2022).** Data assimilation and autoregression for near-real-time streamflow observations in LSTMs. *HESS* 26, 5493–5513. https://doi.org/10.5194/hess-26-5493-2022 (**prior with-q record: 0.879**)
7. **Li, J., Song, Y., Pan, M., Lawson, K., & Shen, C. (2025).** Ensembling differentiable process-based and data-driven models. *HESS* 29, 6829. https://doi.org/10.5194/hess-29-6829-2025 (**prior no-q record: 0.8294**)
8. **Coxon, G., et al. (2015).** A novel framework for discharge uncertainty quantification applied to 500 UK gauging stations. *WRR* 51(7), 5531–5546. https://doi.org/10.1002/2014WR016532
9. **Kiang, J. E., et al. (2018).** A comparison of methods for streamflow uncertainty estimation. *WRR* 54(10), 7149–7176. https://doi.org/10.1029/2018WR022708
10. **Westerberg, I. K., & McMillan, H. K. (2015).** Uncertainty in hydrological signatures. *HESS* 19, 3951–3968. https://doi.org/10.5194/hess-19-3951-2015
11. **Aerts, J. P. M., et al. (2024).** Are hydrological models too complex? Discharge uncertainty and model comparison on CAMELS-GB. *HESS* 28, 5011–5030. https://doi.org/10.5194/hess-28-5011-2024 (nearest related work to the ceiling idea)
12. **Baste, S., Klotz, D., Acuña Espinoza, R., Bárdossy, A., & Loritz, R. (2025).** On the ability of LSTMs to predict extreme events. *HESS* 29, 5871. https://doi.org/10.5194/hess-29-5871-2025
13. **Martel, J.-L., et al. (2025).** On the use of LSTM networks for high-flow simulation. *HESS* 29, 4951. https://doi.org/10.5194/hess-29-4951-2025 (peak oversampling degrades peak metrics)
14. **Liu, Y., et al. (2025).** Benchmarking transformer and foundation models against LSTM for rainfall–runoff. *HESS* 29, 6811. https://doi.org/10.5194/hess-29-6811-2025
15. **Sun, A. Y., & Sun, X. (2026).** Time-series foundation models for streamflow. *Machine Learning: Earth* 2, 010501. https://doi.org/10.1088/3049-4753/ae4982
16. **Acuña Espinoza, R., Kratzert, F., Klotz, D., Gauch, M., et al. (2025).** Multi-frequency LSTM. *HESS* 29, 1749. https://doi.org/10.5194/hess-29-1749-2025
17. **Song, Y., et al. (2026).** Physics-informed differentiable hydrologic models for capturing unseen extreme events. *WRR* 62. https://doi.org/10.1029/2025WR040414
18. **Nearing, G., et al. (2024).** Global prediction of extreme floods in ungauged watersheds. *Nature* 627, 559–563. https://doi.org/10.1038/s41586-024-07145-1 (*not* a CAMELS NSE record)
19. **Kratzert, F., et al. (2024).** HESS Opinions: Never train an LSTM on a single basin. *HESS* 28, 4187. https://doi.org/10.5194/hess-28-4187-2024

*Codebase: `analysis/` (ceiling, feasibility, extrapolation, and scoring scripts);
`scripts/train_mblstm.py` (with-q MB-LSTM); neuralhydrology (no-q members).*

---

*Working document; RiverWatch2 numbers update as the ensemble is extended. The
no-q held-out query was pre-registered and spent once — any future member
requires a new pre-registered query.*
