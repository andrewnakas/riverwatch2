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

**Second, we measure which error scenario applies to CAMELS**, rather than
assuming one. Using USGS field-measurement records for all 530 gauges with
retrievable metadata, **7.2% of basin peak flows exceed the highest direct
discharge measurement** ever made at that gauge — against **44%** reported for UK
gauges by Coxon et al. (2015). CAMELS is roughly six times better measured at
peaks than the benchmark literature implies. The honest counterweight, which must
travel with that number: **43.2% of peaks exceed the highest *well-rated*
gauging**, so peaks are rarely unmeasured but frequently measured badly.

**Third, we place two records against the ceiling.** On the discharge-assimilating
protocol we reach **median day-1 NSE 0.9203** on all 531 basins (prior published
record: Nearing et al. 2022, 0.879). Against the central-scenario ceiling of
**0.9241**, that leaves a gap of **0.0035**, and **63% of basins already score at
or above their own gauge-error ceiling**. On the strict no-discharge protocol we
reach a held-out **0.8363** (prior published record: Li et al. 2025, 0.8294) with
substantially more room: 0.845 remains reachable.

The scientific content is therefore not "we got a higher number." It is that on
the discharge-assimilating protocol **CAMELS is essentially finished** — further
median-NSE gains there are largely fitting gauge error — while the no-discharge
protocol retains real headroom. We support this with a series of negative results
that share one diagnosed mechanism: the residual error is **scatter in peak
magnitude**, not bias, not timing, and not an architectural output limit.

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
| **2026** | **RiverWatch2 with-q grand ensemble** | **0.9203** | **531** | 1-day-lag nowcast, 4 forcings × 5 seeds | **this work** |

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

and since NSE = 1 − MSE/Var(ŷ) ≈ 1 − MSE/V for small σ,

```
NSE_ceiling = 1 − (M/V)·(e^{σ²} − 1).
```

Monte Carlo simulation (`analysis/gauge_ceiling_mc.py`; perturb observed
discharge, score the unperturbed truth against it) agrees with the closed form to
~0.005 over σ = 0.13–0.70.

`M/V = 1 + mean²/variance`, which yields a result we initially got backwards:
**flashy basins have *lower* M/V (1.17 vs 1.57) and therefore a *higher*
break-even σ (0.381 vs 0.293)** — they tolerate more gauge error before saturating,
not less, because large peaks inflate variance faster than mean-square.

### 4.2 Which σ applies to CAMELS — measured, not assumed

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

| | **no-q** (Li/Song protocol) | **with-q** (Kratzert protocol) |
|---|---|---|
| this work | **0.8363** (held-out) | **0.9203** (0.9206 on the 530-basin ceiling frame) |
| prior record | 0.8294 | 0.879 |
| **central-scenario ceiling** | 0.8890 | **0.9241** |
| optimistic-scenario ceiling | higher | 0.9541 |
| **basins already at/above ceiling** | 29% | **63%** |
| target | 0.845 needs **12–23%** of headroom ✅ | 0.95 **unreachable** centrally; needs **88%** optimistically |

⭐ **The central result: on the discharge-assimilating protocol the deployable
ensemble sits 0.0035 below the central-scenario ceiling, with 63% of basins
already saturated.** Under that error model, CAMELS with-q is effectively
finished — remaining median-NSE gains are largely fitting gauge noise. The
no-discharge track is the one with real room left.

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

Diagnosing the near-median cohort sharpens the ceiling result considerably. Under
the central σ scenario **all 78 of them already sit at or above their own
gauge-error ceiling**, and the maximum achievable median equals the current value
to four decimals — the metric cannot move at all. Under the optimistic scenario
(which our 7.2% extrapolation measurement supports) 58/78 are saturated and the
bound rises to 0.9581 on that window. **The σ scenario, not the modelling, now
decides whether the benchmark is finished.**

We also attempted to exploit the finding directly, and failed: weights fit on the
near-median cohort, dHBV-dropping, rank-based weights, and direct
median-maximising coordinate ascent all scored **negative** out of sample
(−0.0001 to −0.0005). The reason is that near-median membership is defined by a
rank, and ranks do not persist: only **27%** of the cohort is shared between the
fitting and validation windows. A near-median specialist cannot be targeted in
advance.

### 5.5 One lever survives: neighbouring gauges

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
lives. Correcting each basin from its three nearest non-nested neighbours, under
fully nested selection (fit 1981-87, shrinkage selected on 1987-90, scored
1990-95), yields a median gain of **+0.00142** with a bootstrap CI of
**[+0.00071, +0.00227]** that excludes zero, improving 60.9% of basins.

The control is what makes this credible: substituting the basin's **own lag-1
residual** — testing whether this is merely temporal autocorrelation — gives
**exactly +0.00000**, improving 50.0% of basins. The information is spatial.

⚠️ **This is not a legal no-q result.** It uses neighbours' *observed* discharge,
which the no-discharge protocol forbids; applied as post-processing to our no-q
ensemble it would be leakage, and we do not include it in the 0.8363. Its value
is as a **diagnostic**, and there it is important: it demonstrates that
information about our residual exists *outside the basin*. A residual that is
partly spatially structured is not purely irreducible scatter — which is the one
principled way past a conditional-mean bound, since that bound is beaten only by
new information. Building a neighbour-assimilating model is a well-defined next
study under a different protocol.

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
basins exceeds the prior published record by +0.041 — and sits **0.0035** below the
central-scenario observational ceiling, with **63% of basins already at or above
their own ceiling**. On this protocol the benchmark is, to a good approximation,
finished; we would treat further median-NSE improvements there as evidence about
gauge error rather than about hydrology.

On the **no-discharge** protocol, a held-out **0.8363** exceeds the prior record of
0.8294 with genuine headroom remaining: 0.845 requires only 12–23% of what the
gauge-error model says is recoverable.

The remaining error, in both tracks, is peak-magnitude **scatter** — established
here from four independent directions (bias/scatter decomposition, failed
out-of-sample peak rescaling, two distributional heads, and a refuted output-
saturation hypothesis). That is the same quantity the ceiling argument is about,
which is why we think the two halves of this paper belong together: the reason
peak error resists modelling is closely related to the reason the observations
cannot reward removing it.

**Limitations.** The ceiling is our own derivation, not a citation, and scenario
choice drives it — we report the full table. Multi-hundred-gauge uncertainty base
rates are UK (Coxon; CAMELS-GB); Kiang et al. is USGS-coauthored but covers three
sites. The two tracks use different protocols and windows and must not be
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
