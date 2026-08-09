# Literature notes: what actually moves the needle on CAMELS streamflow records

Working notes for the RiverWatch2 record campaigns (no-q Li/Song track, with-q
Nearing track). Every external claim is cited; every internal claim is tagged
**[measured]** with the number and the artifact that holds it.

Records currently held:
- **with-q (discharge-assimilating):** day-1 median NSE **0.9058** vs Nearing et
  al. 2022 **0.879**. Deployable — zero fitted parameters. 95% CI [0.8940,
  0.9126], P(>record)=1.000. `benchmarks/withq_deployable_final.json`
- **no-q (CAMELS protocol):** day-1 median NSE **0.8351** vs Li/Song 2025
  **0.8294** (our reproduction 0.8298). Margin not yet significant.

---

## 1. Architecture is NOT the lever (settled, twice)

**External.** *From RNNs to Transformers: benchmarking deep learning
architectures for hydrologic prediction*, HESS 29:6811 (2025).
<https://hess.copernicus.org/articles/29/6811/2025/>
Benchmarked **11 attention-based architectures** against LSTM on daily CAMELS.
LSTM won at median NSE **0.80**; best transformer (Crossformer) ~**0.73**. On
global streamflow, LSTM cut median ubRMSE by 12.0% vs the best attention model.
The only exception was snow water equivalent (a different task), where a
Non-stationary Transformer edged LSTM 0.88 vs 0.87 KGE.

**Internal [measured].** A reference seq-to-one CudaLSTM tied our existing
members (daymet 0.750, nldas 0.723) and contributed **+0.0002 / 0.000** fit
weight to the ensemble. Recorded as "the no-q ceiling is architecture-
independent; the gap is seed depth, not a better LSTM."

**Consequence.** No transformer / Mamba / state-space / foundation-model members
were trained. GPU went to forcings, seeds, and loss diversity instead. This is a
deliberate negative result, not an omission.

---

## 2. Multi-forcing ensembling — the single biggest lever

**External.** Li, Song, Pan, Lawson, Shen (2025), *Ensembling differentiable
process-based and data-driven models with diverse meteorological forcing
datasets to advance streamflow simulation*, HESS 29:6829.
<https://hess.copernicus.org/articles/29/6829/2025/>
The 0.83 CAMELS-531 no-discharge record. Key structural findings:
- a **per-forcing ensemble** of single-forcing models beats one fused model;
- **plain averaging beat every weighting strategy they tried**;
- 3→10 seeds is worth only ~+0.003 ("gaps did not change much");
- their headline weighting (GA, ~0.844) is fit **on test-period observations**
  and is explicitly "solely for qualitatively analysing relative contributions."

Also relevant: Kratzert et al. (2021) and *A note on leveraging synergy in
multiple meteorological data sets*, HESS 25:2685 (2021) — the origin of the
multi-forcing idea.

**Internal [measured].** LSTMmulti (one LSTM fed all three forcings, seed
averaged) scored **0.826** standalone vs 0.75-0.78 for single-forcing members —
the paper's **+0.027**, reproduced. It carries NNLS weight **0.49**, the largest
of any stream.

---

## 3. Differentiable physics (δHBV) as a decorrelating member

**External.** Same Li/Song paper; δHBV lineage from Feng, Liu, Lawson, Shen —
differentiable parameter learning wrapping the HBV conceptual model.

**Internal [measured].** δHBV members score 0.743-0.762 standalone — clearly
below the LSTMs — yet `dhbv_daymet` is **the most valuable member to keep**
(ensemble drops **−0.0041** without it). Their value is error decorrelation, not
skill. See §7.

---

## 4. Plain averaging beats fitted weighting (the recurring result)

**External.** Bates & Granger (1969) on combining forecasts; Timmermann (2006),
*Forecast combinations* (Handbook of Economic Forecasting) — the "forecast
combination puzzle": estimated weights routinely lose to equal weights because
weight estimation error swamps the theoretical gain. Prescribed fixes are
shrinkage toward equal weights, or trimming.

**Internal [measured].** This held on every track we tested:

| track | plain mean | fitted weights |
|---|---|---|
| with-q, 2-seed | **0.9019** | 0.9016 (`--fit-weights`, test-fit) |
| with-q, 2-seed | **0.9019** | 0.8996 (NNLS, test-fit) |
| no-q, 7-stream | 0.8298 | 0.8321 (NNLS oracle, +0.0023 only) |

**The headline consequence:** the published with-q claim of 0.9016 used weights
optimised on test observations — an oracle a reviewer would reject. Replacing
them with a plain arithmetic mean *raised* the number to 0.9019, and with 4-seed
members to **0.9058**. The record was made publishable by **removing** the
sophisticated component.

---

## 5. Loss diversity — the one genuinely novel technique we adopted

**External.** *Ensemble streamflow forecasting with diverse loss functions*,
Applied Soft Computing (2026).
<https://www.sciencedirect.com/science/article/abs/pii/S1568494626007246>
Trains LSTMs under different objectives (MSE, NSE, KGE, Huber, quantile,
expectile) and stacks them; reports significant NSE gains over same-loss
ensembles and 93.4% uncertainty coverage at 1-day lead.

**Internal [measured].** Trained an LSTMmulti with **MSE** loss (s777) against
the NSE-loss baseline — identical corpus, architecture, and forcings.

| quantity | value |
|---|---|
| own day-1 medNSE | 0.8040 (**worse** than lstm_multi 0.8262) |
| residual r vs its NSE-loss twin | **0.807** |
| residual r between lstm_multi and lstm_daymet | 0.86 |
| marginal on shipped 4-stream subset | 0.8339 → **0.8351** (+0.0012) |
| marginal on 7-stream plain mean | 0.8298 → 0.8320 (+0.0022) |

**Finding worth publishing:** *changing the loss function buys more error
independence than changing the forcing product does* (0.807 < 0.86). Same data,
same architecture — only the objective differs. This was adopted as the
substitute lever when ERA5-Land looked infeasible, and it worked.

---

## 6. Seed depth saturates — and predictably

**External.** Li/Song report 3→10 seeds as worth ~+0.003 and describe the gains
as "rapidly becoming marginal."

**Internal [measured].** Measured the curve rather than assuming it:

| seeds k | day-1 medNSE | increment |
|---|---|---|
| 1 | 0.8238 | — |
| 2 | 0.8321 | +0.0083 |
| 3 | 0.8340 | +0.0020 |

Fitting `NSE(k) = 0.8395 − 0.0156/k` (residuals ≤0.0004) gives an **asymptote of
0.8395** — so *even infinite seeds* fall short of the 0.84 target. Confirmed
out-of-sample: a real 4th seed moved 0.8340 → 0.8323 (**−0.0018**).

Two consequences: (a) ~20 GPU-hours were cancelled on evidence rather than run
to completion; (b) the stream with the **largest ensemble weight** (lstm_multi,
0.49) turned out to be the **least seed-responsive** (+0.0009 for 1→3 vs +0.0037
for lstm_maurer) — so the intuitive allocation was the wrong one.

**Training instability [measured].** 2 of 5 LSTMmulti seeds trained badly (s333
val NSE 0.601, s444 0.765, vs a healthy band of 0.911-0.917). Both are
detectable from **training-period validation NSE alone**, giving a deployable
gate: reject seeds below ~0.90. This partly explains why seed depth pays so
little — each new seed carries real risk of being a drag.

---

## 7. Own skill does not predict ensemble contribution

**Internal [measured].** Residual-correlation analysis over the 7 no-q streams:

| member | own NSE | mean residual r | Δ if removed |
|---|---|---|---|
| dhbv_maurer | 0.7429 | 0.676 | **+0.0033** (remove it) |
| dhbv_nldas | 0.7395 | **0.601** (most decorrelated) | +0.0014 |
| dhbv_daymet | 0.7615 | 0.679 | **−0.0041** (most valuable) |
| lstm_multi | 0.8262 | 0.747 | −0.0017 |

`dhbv_daymet` and `dhbv_maurer` have near-identical standalone skill and
**opposite** ensemble value. Selecting members by individual performance is
actively misleading. This also corrected a prior internal note that had
identified the wrong member as the one to drop.

---

## 8. Transfer asymmetry — the campaign's main methodological result

**Internal [measured].** How well does train-period information predict
test-period behaviour?

| what is ranked | train→test rank correlation |
|---|---|
| the 99 candidate member **subsets** (global) | **0.781** |
| **members within a basin** (per-basin) | **0.286** |

Per-basin member choice is barely better than chance: the train-best member is
also test-best in **25.4%** of basins vs 14.3% at random.

**This explains Li/Song's own method.** Their per-basin GA reaches ~0.844 only
by fitting on test observations — because that per-basin signal is largely *not
recoverable* from the training window. Anyone attempting deployable per-basin
gating on this benchmark should expect noise.

Consistently, per-basin *trimming* rules went **negative** on held-out basins
(topk4: mean −0.0001, worst −0.0067), while global rules held up.

---

## 9. In-sample inflation is large and uneven

**Internal [measured].** Every member inflates in-sample, by 0.11-0.17, spread
**0.059** across members:

| member | train NSE | test NSE | inflation |
|---|---|---|---|
| lstm_multi | 0.9388 | **0.8262** | **0.1125** (least) |
| lstm_nldas | 0.9240 | 0.7525 | 0.1715 (most) |

The **best** test member inflates the **least**, so ranking members on raw train
skill systematically *understates* the strongest one. Any cross-family
comparison must z-normalise within member first.

---

## 10. Peak-validation selection ships regressions

**Internal [measured].** Unconstrained ridge stacking topped **every** train-CV
leaderboard (+0.0056 mean over rolling-origin splits, stable across
leave-basin-out folds) — then scored **below the plain mean on test**
(0.8295 vs 0.8298, −0.0003).

Cause, found by auditing coefficients before spending the test query: a
persistent **−0.19** weight on `lstm_daymet`, cancelling against its
0.87-correlated sibling `dhbv_daymet`. Real in-sample gain; did not transfer.

A **non-negative** variant — adopted *because* of that audit, at a nominal
train-CV cost (+0.0049 vs +0.0056) — returned **+0.0025** on test.

**Had the rule been chosen by peak validation score, the campaign would have
shipped a regression while appearing to succeed.**

---

## 11. Things that did not work (recorded so they aren't retried)

| lever | outcome |
|---|---|
| per-basin trimming (topk) | negative on held-out basins (−0.0067 worst) |
| per-basin static-attribute gate | ties plain mean; per-basin signal doesn't transfer |
| unconstrained ridge stacker | −0.0003 on test despite winning every val split |
| NNLS / `--fit-weights` | oracle, and *worse* than plain mean on with-q |
| geometric mean | ≈ arithmetic (0.9038 vs 0.9037) |
| clipping negatives to zero | no change |
| blending persistence | monotonically worse (w=0.05 → −0.0009; w=0.20 → −0.0164) |
| pooling 2-seed **and** 4-seed dumps | 0.9037 vs 0.9058 — they share seeds, so it double-weights the weaker early ones |
| ERA5 via Google ARCO (GCS Zarr) | probe timed out at 900 s; also ERA5 (~31 km) not ERA5-**Land** (~9 km) |
| δHBV loss diversity (earlier work) | net-negative or neutral — note this is the **δHBV** side; LSTM-side loss diversity **did** work (§5) |

---

## 12. Operational lessons (cost real hours)

- **CDS limits queued requests PER DATASET, not per account.**
  `derived-era5-land-daily-statistics` has its own budget separate from
  `reanalysis-era5-land`. In one year, 24/24 monthly reanalysis requests
  succeeded while 3/3 daily-statistics requests were rejected — same account,
  same moment, same concurrency. Three wrong theories (slow service → account
  concurrency cap → request too large) preceded simply reading the HTTP 400
  body, which said it outright. **Read the error body before theorising from
  symptoms.**
- **Editing a running bash script does not change its queue** — bash has already
  parsed the loop. Kill the PID.
- **`pkill -f <pattern>` matches your own SSH command** containing that string.
  Kill explicit PIDs.
- **0% GPU with no output is not necessarily a hang** — `nh_to_dump.py` converts
  a 70 MB results file at 100% CPU for minutes. Check the process table before
  restarting; a restart there would have discarded ~10 h of training.
- **A superseded dump silently rejoined a stream.** `..._nldas_nhlstm_s111_OLD642`
  (a discarded 0.642 seed) matched a bare glob and inflated the ensemble to
  0.8309 vs the true 0.8298. Loaders now quarantine `_OLD/_test_/_bak/_broken`
  and print a member manifest every run.

---

## 13. Where a paper's contribution actually lies

Not the architecture — it's an LSTM, as everyone else uses. The contribution is
**methodological**, and it is unusual in this literature:

1. **The transfer asymmetry** (§8) — global combination structure transfers
   (0.781), per-basin does not (0.286), which explains why the published
   per-basin method needs test observations.
2. **A quantified demonstration that peak-validation selection ships a
   regression** (§10) — usually only warned about in the abstract.
3. **Loss diversity beats forcing diversity for decorrelation** (§5) — 0.807 vs
   0.86 residual correlation.
4. **Seed saturation is predictable from three cheap points** (§6) — an
   asymptote fit that redirected ~20 GPU-hours away from a dead end.
5. **A record made publishable by removing sophistication** (§4) — plain mean
   beat the fitted weights it replaced.

The framing that fits the evidence: *standard components, rigorous validation,
and a documented account of why the sophisticated methods failed.*

---

## 14. Is there a CEILING? What the field says, and what our data says

### The literature debate is explicit

- **[Probing the limit of hydrologic predictability with the Transformer
  network](https://www.sciencedirect.com/science/article/abs/pii/S0022169424007844)**
  (J. Hydrology, 2024) — built a Transformer specifically to test whether LSTM
  is at a limit. Conclusion: LSTMs "may be nearing their performance ceiling,
  with additional improvements potentially limited by **data uncertainties or
  intrinsic constraints of current hydrologic datasets**."
- **[HESS 29:6811 (2025)](https://hess.copernicus.org/articles/29/6811/2025/)** —
  11 attention architectures, none beat LSTM (0.80 vs ~0.73 best). Independent
  corroboration that architecture is not the binding constraint.
- **[Unveiling the limits of deep learning models in hydrological extrapolation](https://hess.copernicus.org/articles/29/5871/2025/)**
  (HESS 2025) — a *different* ceiling: a stand-alone LSTM cannot predict
  discharge above a theoretical **73 mm/day**, below the **183 mm/day** maximum
  in its own training data. LSTMs show a concave runoff response to extreme
  precipitation that hybrid physics-ML models do not. This is an architectural
  limit on **extremes**, not on median NSE.

### Two hard bounds on any achievable score

- **Gauge uncertainty.** Discharge measurement error runs to **20%** of observed
  value ([Hydrol. Process. rating-curve review](https://onlinelibrary.wiley.com/doi/10.1002/hyp.9567)).
  No model can score above the noise in its own target.
- **Forcing error.** Precipitation products disagree materially — which is
  *why* multi-forcing ensembling is worth +0.027 (§2). The gain exists precisely
  because no single product is right.
- Li/Song's own worst basins (NSE 0.3-0.4) are Great Plains / West with "high
  dam density or heterogeneous hydrogeological conditions" — anthropogenic
  modification and karst, physically unpredictable from weather alone.

### [measured] Our own ceiling probe — we are NOT data-limited yet

`ceiling_probe.py`, no-q test set, 531 basins, 7 streams. The discriminating
test: **in bad basins, do members disagree (model-limited) or agree-and-fail
(data-limited)?**

| quantity | value | reading |
|---|---|---|
| spearman(inter-member spread, basin NSE) | **−0.371** | where members disagree, we do badly |
| mean spread, poor basins (NSE<0.5, n=48) | 0.263 | |
| mean spread, good basins (NSE≥0.8) | 0.125 | |
| **spread ratio poor/good** | **2.10** | poor basins are much more *contested* |

**Verdict: still MODEL-limited, not data-limited.** If the residual error were
irreducible observation noise, members would agree and still be wrong. They
don't — disagreement means information remains recoverable by better or more
decorrelated members.

### [measured] But the median is a badly-behaved target

| quantity | value |
|---|---|
| plain-mean ensemble | 0.8298 |
| per-basin best member (oracle over current members) | 0.8478 |
| headroom from **recombination alone** | +0.0180 = **10.6%** of the remaining gap |
| headroom to a perfect NSE 1.0 | +0.1522 = the other **89.4%** |
| median vs mean | 0.8298 vs **0.6289** (gap 0.2009 — a severe hard tail) |
| basins with NSE < 0 | **2.8%** (worse than predicting mean flow) |
| **basins within 0.01 of the median** | **50** |
| basins within 0.05 of the median | 186 |

**The structural insight:** a median moves by improving the basins *adjacent to
it* — around the 265th-ranked basin — not the catastrophic failures. Only ~50
basins actually control the metric.

That explains why every global lever in this campaign returned +0.001 to +0.004:
**global methods spread effort over all 531 basins when ~50 determine the
score.** It also explains why per-basin targeting is so attractive *and* why it
fails (§8) — the per-basin signal needed to exploit this does not transfer from
the training window.

**Consequence for 0.84:** +0.006 from 0.8339 requires moving ~50 specific
mid-ranked basins. Seed depth (capped at 0.8395, §6) and recombination (10.6% of
the gap) cannot do it. Only a genuinely new information source — a decorrelated
4th forcing, or discharge assimilation — plausibly can. This is the strongest
argument for the ERA5-Land member.

## 15. WHY the shared error is shared: one gauge base, measured three ways

**2026-08-04.** §14 left the 70% shared component unexplained. It now has a
mechanism, a literature basis, and three independent measurements.

### 15.1 Provenance — the four forcings are not four independent views

| product | precipitation source | lineage |
|---|---|---|
| Daymet | **GHCN-Daily** direct (Thornton 1997 interpolation) | GHCN |
| NLDAS-2 | CPC 1/8° gauge-only daily, PRISM-adjusted, radar-disaggregated (daily totals unchanged) | COOP/GTS via CPC (Chen 2008, JGR 113 D04110) |
| Maurer | NCDC **COOP**, SYMAP/Shepard + PRISM scaling | COOP |
| AORC | Livneh daily (COOP-derived) + NLDAS-2 method + Stage II/CMORPH → Stage IV | COOP + radar |

They are **four interpolations of one heavily overlapping COOP/GHCN gauge base**,
differing mainly in interpolation scheme, orographic adjustment, and (AORC) radar
sub-daily structure. A gauge that missed a storm is invisible to all four — which
is exactly the signature of error that survives ensembling.

*Verified:* Daymet↢GHCN (ORNL DAAC V4 guide), NLDAS-2↢CPC (NASA LDAS FAQ).
*Partial:* Maurer 2002 (J.Climate 15:3237) via secondary sources. *Unverified:*
AORC's gauge list first-hand (Fall 2023 JAWRA paywalled; NOAA PDFs unreadable).

⚠️ **ASOS is INSIDE GHCN-Daily**, so it cannot serve as an independent check —
GHCN's US component explicitly compiles ASOS alongside COOP and RAWS. Separately,
ASOS precipitation is documented as biased: heated tipping buckets undercatch
**2-10% vs COOP**, capture almost nothing below 15 °F, and the rising ASOS share
imparted a measurable drying trend to gridded products (Diem 2026, HESS
30:1999-2011; National Academies 2012 App. E). We also measured `p01m` as only
**4.3% populated in 1995**, with precip-reporting stations within 25 km of just
7 of 531 basins. ASOS is not a usable precipitation reference for this window.

### 15.2 Three measurements, one direction

1. **Products disagree more with each other on failure days.** Magnitude-matched:
   at the same rainfall, spread on top-1% error days is **1.272×** ordinary days
   (consistent across bins 0.1-40 mm). ⚠️ Raw statistics mislead here — pairwise
   *r* says "worse" (0.704 vs 0.771) while *CV* says "better" (0.528 vs 0.844),
   because event days are **8.1× wetter**; only the matched control is defensible.
2. **Products track gauges worse on failure days.** Against a 529-basin GHCN
   station corpus, matched on *observed* rainfall: mean Δcorr **−0.074**
   (daymet −0.097, maurer −0.093, nldas −0.068, aorc −0.039), with |bias| growing
   7-13 mm. Survives a within-observed-rainfall control (**−0.066**), so it is
   specific to failure days, not a wet-day artifact.
3. **All four are biased WET on those days** (+0.85 to +4.53 mm, mean +2.45). The
   ensemble under-predicts flow while the forcings over-report rain.

### 15.3 What this does NOT support

- **Input error is a contributor, not the explanation.** Product spread predicts
  error at only **r = 0.09-0.19**. Far too weak to carry the 70%. Do not overclaim.
- **The orographic hypothesis failed at the event-day level.** Event-weighted
  `elev_mean` 442 vs 439 unweighted, `frac_snow` 0.100 vs 0.105 — flat. The one
  real signal is `p_mean` (4.23 vs 3.63): **wet basins own the failures**.
- **Circularity.** Daymet agreeing best with GHCN stations (0.85 COOP-only, 0.93
  SNOTEL) is partly definitional — it ingests them. Always state this.
- **Point-vs-areal.** A station is a point, a basin mean areal (53% effect, §
  AORC orography). Absolute product-vs-station gaps are *not* product error; only
  the ordinary→failure *change* is load-bearing, since the penalty cancels.

### 15.4 The gap in the literature

**Kratzert et al. 2021 (HESS 25:2685)** established the canonical multi-forcing
result — single-forcing 10-member LSTM ensembles ≈ 0.77/0.77/0.74 (Daymet/Maurer/
NLDAS), three-forcing **0.82**, ΔNSE 0.074, gains scaling with inter-product
disagreement. **They never ask whether the three products share a station network,
and never separate forcing error from structural error.** That omission is the
opening. **Renard et al. 2010 (WRR 46:W05521)** states input and structural error
are weakly identifiable *without independent information on rainfall error*; an
ambiguity decomposition plus a provenance-tagged corpus is leverage on exactly
that. Novelty checked: **Willard et al. 2025** (JGR-MLC 10.1029/2025JH000732), the
nearest competitor, is stream *temperature* in unmonitored basins — different
target, does not occupy this framing.

### 15.5 The true ceiling is BELOW our computed 0.930

Discharge itself is uncertain, worst exactly where our error lives. **Coxon et al.
2015 (WRR 51:5531)**, 500 UK stations: relative uncertainty 9-397%, low flows
20-397%, and **44% of station-groups had missing high-flow values because ratings
could not extrapolate**. **Kiang et al. 2018 (WRR 54)**: 95% widths 3-17% at median
flow but **41-200% at high flows in extrapolated rating sections**. Our 1% of days
carrying 93% of squared error are precisely the extrapolated-rating days — so
0.9301 is an upper bound on *ensembling*, and the attainable ceiling is lower.
This strengthens the paper rather than weakening it.

## 16. What is now CLOSED, and what that implies (2026-08-04)

Six results this cycle, all measured rather than argued. Together they say the
same thing: **reprocessing the same information is finished; only genuinely new
information can move the number.**

### 16.1 With-q record 0.9203 — but seed depth is saturated

| config | medNSE | medKGE | alpha | basins |
|---|---|---|---|---|
| baseline (3 forcings × 5 seeds) | 0.9137 | 0.8768 | 0.9401 | 531 |
| + AORC 4 seeds (prior record) | 0.9196 | 0.8850 | 0.9512 | 530 |
| **+ AORC 5 seeds** | **0.9203** | **0.8861** | 0.9507 | **531** |

First result on the **complete** benchmark (the 0.9196 record was 530 basins —
AORC lacked 13235000, since reconstructed). CI95 [0.9091, 0.9264]; Nearing 2022 =
0.879.

**Decomposed on the same 530 basins:** the 5th seed contributes **+0.0005** and
the coverage fix **+0.0002**. The reconstructed basin scores NSE 0.9934, so it
lifts the median by re-ranking — a **coverage fix, not a skill gain**. Against
+0.0059 for adding AORC as a 4th *forcing*, **seed depth is exhausted.**

**A pre-registered mechanism check came back negative.** The 0.9196 gain came with
alpha 0.9401→0.9512, attributed to reduced under-dispersion. The 5th seed moves
alpha 0.9512→**0.9506 (flat)** while still adding skill, so that mechanism
explains the *forcing* addition, not the *seed* addition. Do not extend it.

### 16.2 Combination rules are exhausted too

Train-side gate over 22 rules (no test observations touched): plain mean
**0.9420**, best rule (`gate k=6 lam=0.7`) **0.9429** — a **+0.0009** spread
across every shrunk-inverse-MSE, top-k and static-gate variant.

This is the forecast-combination puzzle (Bates-Granger 1969; Timmermann 2006) in
our own data, and it is now established **train-side**, where it is actually
decidable — the earlier oracle audit (global NNLS 0.8321 vs plain 0.8298) could
only hint at it from the test side. **Stop tuning weighting schemes.**

### 16.3 AORC no-q fails on OWN SKILL — the discriminator, restated

| stream | day-1 median NSE | seeds |
|---|---|---|
| daymet | 0.7759 | 3 |
| maurer | 0.7749 | 3 |
| nldas | 0.7532 | 3 |
| **aorc** | **0.7292** | 2 |

**s222 trained 31% better than s111 (0.00833 vs 0.01089) and scored 0.004
WORSE** — two independent seeds, one healthy and one salvaged from divergence,
landing within 0.004. This is the forcing, not a bad seed. Verified not an
artifact: 189,241-row inner join, 531 basins, `max|Δtruth| = 0.000000`, prediction
correlation 0.9079 with daymet.

**Same forcing, opposite verdicts across tracks.** AORC *won* on with-q (+0.0062)
where its members hit val 0.818-0.822, in band with peers; it fails on no-q where
they do not. Decorrelation is 0.0398 in both cases. **Own skill is the
discriminator, not decorrelation** — the cleanest evidence this campaign has for
that rule, and the reason a member must clear the peer band before its diversity
is worth anything (Krogh-Vedelsby: `MSE(ens) = mean member MSE − diversity`).

*Method note:* the original pre-registration rested on AORC being in band on
**training loss** at matched epoch. It was — and training loss did not rank test
skill. A matched-epoch loss comparison is not a substitute for measuring skill.

### 16.4 Station-corrected daymet — built, measured, rejected without training

Two corpora built from the GHCN station corpus (§15): `daymetSC` (per-basin
scalar) and `daymetSB` (daily weighted blend), both terrain-gated to elev<400 m,
frac_snow<0.15, station ≤15 km, with train-period-only fits.

The gate threshold was **tuned, not guessed**: at 1000 m the correction still
tracked terrain (corr −0.337), because the station/daymet ratio falls monotonically
with elevation (0.964 → 0.936 → 0.883 across 0-300/300-600/600-1000 m). At 400 m
it is essentially terrain-free (−0.087).

**Both rejected on measurement, before spending any GPU:**

| variant | corr with daymet | decorrelation vs other forcings |
|---|---|---|
| daymetSC (scalar) | **1.0000** | 0 (volume only) |
| daymetSB (blend) | **0.9943** | **−0.0048** (wrong direction) |

A member 0.994-correlated with one already in the ensemble cannot add ensemble
value, and the blend is *more* like maurer/nldas/AORC than plain daymet is.

**Why it washed out is itself the lesson.** The probe that justified building it
measured a 50/50 blend moving heavy days 4.67 mm; the shipped version realised
mean weight 0.226 across only 24% of basins, because confidence weighting and the
terrain gate — each correct in isolation — compounded. **This approach is squeezed
between two constraints: where stations are trustworthy (low, flat, close) daymet
already agrees with them; where they would change things materially (mountains)
they are the biased estimator.** The 0.3% bias² share said as much in advance.

### 16.5 What remains

Every cheap lever is now closed by measurement: phase/timing, sub-daily intensity,
KGE variance inflation, model-family diversity, seed depth, combination rules,
station-corrected forcings, and AORC on the no-q track. What is left is a source
that is **not another interpolation of the COOP/GHCN gauge base** — which is the
entire case for CONUS404 (WRF reanalysis), currently extracting.

Its criteria are pre-registered: **own skill first** (no-q stream ≥0.750, i.e.
inside the peer band, not merely above AORC's failed 0.7292), then decorrelation
≥0.08 (2× the same-family axis), then train-CV delta ≥+0.001. Honest prior: ~50/50
— decorrelation should pass since a reanalysis genuinely is new information, but
reanalysis precipitation is not gauge-corrected and own skill is the real risk.

If it fails, the defensible conclusion is that **with-q 0.9203 / no-q ~0.836 is
where this architecture tops out on this data**, with the remaining error being
input and observational rather than anything ensembling can reach.

## Open threads

- ⚠️ **STALE ENTRIES BELOW** — several of these were written before §15-16 and are
  superseded. Current status, 2026-08-04:
  - **CONUS404** is the live lever, extracting **serially** (~43 h, per-tile cost
    ~3.9 h × 11 tiles). PAR=4 failed twice from truncated chunk reads
    (`ContentLengthError: received 6974636 of 14822897 bytes`) — bandwidth does
    not parallelize away. Polygon averaging verified (0 point-fallbacks).
  - **AORC no-q** resolved: stream 0.7292, below every peer — see §16.3. The
    train-side gate decides inclusion against a baseline of **0.9420** locked
    before the stream existed; prediction on record is that it adds ~nothing.
  - **Seed depth and combination rules** are both closed (§16.1, §16.2).
  - **Station corpus** published (529/531 basins, GHCN COOP+ASOS+SNOTEL,
    provenance-tagged) and used diagnostically in §15; it is NOT yet tested as a
    member input, and the derived `daymetSC`/`daymetSB` variants were rejected
    on measurement (§16.4).

- **ERA5-Land as a 4th decorrelating forcing** — *superseded*: AORC took this slot
  (with-q +0.0062), and the CDS fetch was measured at ~6 months for 1980-2010.
  Dead for this campaign.
- **Fused with-q member** (`camels3fv2`) — corpus built (671 basins, 18 cols,
  1980-2008; Maurer's end date truncates it, so **with-q track only**). Fusion
  was +0.027 on no-q and has never been tried with discharge active.
- **RMSE loss-diversity member** — training; gate on val NSE ≥ ~0.90 before
  inclusion.
- **Significance for the no-q margin** — +0.0053 over baseline is not yet
  bootstrap-significant; with-q already is.
- **Station corpus is built and unused as a MEMBER** —
  `data/local_corpora/camels_corpus_station_v1/` (529/531 basins, GHCN COOP+ASOS+
  SNOTEL, provenance-tagged). §15 uses it diagnostically only. Open question:
  does a station-observed **DTR** channel help maurer/nldas, which have
  `tmax==tmin` on **100%** of days? Gate before shipping: in band on own skill
  AND train-CV ensemble delta ≥ +0.001 — decorrelation alone is insufficient
  (AORC 0.0398 and δHBV 0.039 both bought ~nothing).
- **SNOTEL as the clean independent control** — `app/snotel.py` now takes an
  `elements` argument (AWDB REST, no credentials; default `WTEQ,SNWD` preserved).
  NRCS sits outside the COOP/GHCN base: 913 sites, 470 pre-1980, 68 basins with a
  pre-1990 site ≤25 km. Measured daymet 0.833 vs maurer 0.641 / aorc 0.623 /
  nldas 0.581 on WY1996-2000 precip — the provenance ordering, independently.
- **Publish the corpus** — HF push blocked on a token (`huggingface_hub` not
  installed). Reuse `scripts/sync_openmeteo_corpus.py`; cite Menne et al. 2012
  (GHCN-Daily, doi:10.7289/V5D21VHZ) and state the Daymet-circularity caveat.
