# What dataset/member would actually get us to 0.84?

Analysis to decide what to build next, instead of assuming a 4th forcing helps.
Current deployable no-q: **0.8351**; target 0.84; gap **+0.0049**.

## 1. The median is decided by 120 basins — and they are NOT the hard tail

`median_basins.py`, 531 basins, 7-stream plain mean.

| attribute | **PIVOTAL** (120, NSE 0.802-0.859) | HARD tail (20, NSE<0.3) | EASY (13, >0.95) |
|---|---|---|---|
| aridity | **0.797** humid | 1.580 arid | 0.816 |
| p_mean | **3.389** wet | 2.007 dry | 3.125 |
| frac_snow | **0.123** (snowiest group) | 0.025 | 0.049 |
| slope_mean | **27.4** steep | 9.7 flat | 62.1 |
| frac_forest | **0.904** | 0.034 | 0.909 |
| area_gages2 | **252** km² small | 490 | 689 |
| **inter-member spread** | **0.132** | 0.378 | 0.090 |

**The pivotal and hard-tail groups are opposites.** Everything the literature
says about hard basins — arid southwest, karsted Edwards aquifer, Prairie
Pothole, dam-regulated — describes the **20-basin tail**, which sits so far
below the median that fixing it entirely would not move the median at all.

The basins that *do* control the score are humid, forested, steep, small, and
the **snowiest** group in the dataset.

## 2. That reframes what a new member must do

Two consequences:

- **Precipitation totals are not the bottleneck.** Humid forested basins are
  where gridded precipitation products are *most* accurate and agree most.
  Pivotal spread is 0.132 — only 1.47× the easiest basins, vs 2.9× for the tail.
  Our members already largely agree there.
- **What limits steep, small, snowy catchments is TIMING**: sub-daily rainfall
  intensity and snowmelt timing, not monthly water balance.

So the question is not "which precipitation product is best" but **"what
information is missing at daily resolution in snowy, steep, small basins."**

## 3. Ranked candidates

### ① AORC — hourly, 1 km, CONUS ★ best fit to the diagnosis
[Ghimire et al. 2025, WRR](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024WR038256):
"Peak flow predictability is enhanced significantly with AORC relative to
others, **particularly at the small basin scales**"; hourly AORC and Stage-IV
"lead to improved annual peak flow performance over Daymet-driven streamflow,
particularly in smaller basins, highlighting the value of **high temporal
resolution** forcings."

Matches the pivotal profile exactly — small, steep basins where timing dominates.
Directly addresses the mechanism the literature blames for daily-product failure
(sub-daily intensity). Caveat: the same paper ranks AORC only 3rd on NCRMSD in
most basins, so it is a *peak-flow* specialist, not a uniform upgrade.

### ② Snow state (SWE) as an input — cheapest, best-targeted
The pivotal set is the snowiest group (frac_snow 0.123 vs 0.025/0.049).
Literature: "incorporating **time-lagged snow water equivalent (SWE)**
substantially improved model performance in snowmelt-dominated basins."
This is a *variable*, not a new forcing product — likely obtainable from
existing reanalyses (ERA5-Land carries snow depth / SWE) without a new corpus.

### ③ CONUS404 — 4 km dynamical downscaling of ERA5 over CONUS
[NCAR/USGS, BAMS 2023](https://opensky.ucar.edu/system/files/2024-08/articles_26601.pdf).
40+ years, 4 km. Genuinely decorrelated from Daymet/NLDAS/Maurer (dynamical, not
interpolated from gauges), and *purpose-built for CONUS* — which is exactly what
plain ERA5 is not. Better bet than ERA5-Land on resolution grounds.

### ④ ERA5-Land — currently queued, now the WEAKEST of these
Warning from [Frontiers in Water 2023](https://www.frontiersin.org/journals/water/articles/10.3389/frwa.2023.1166124/full):
ERA5 forcing dropped median NSE **0.71 → 0.54** vs NLDAS-2, with ~80% of US
catchments worse, attributed to coarse resolution. ERA5-Land (~9 km) should beat
ERA5 (~31 km) but is still a global product competing against US-tuned ones, at
coarser resolution than AORC (1 km) or CONUS404 (4 km).
**No published CAMELS-531 ERA5-Land benchmark was found.**

## 4. What our own data says about "just add a decorrelated member"

Value of each existing member when added to the rest:

| own NSE | mean residual corr | Δ when added |
|---|---|---|
| 0.7395 | **0.601** (most decorrelated) | **−0.0014** |
| 0.7525 | 0.645 | −0.0005 |
| 0.7429 | 0.676 | −0.0033 |
| **0.7615** | 0.679 | **+0.0041** (best) |
| 0.7749 | 0.700 | +0.0016 |
| 0.7759 | 0.706 | +0.0019 |
| 0.8262 | 0.747 | +0.0017 |

**The two most decorrelated members both HURT.** The best contributor is
mid-pack on both axes. A regression of Δ on (skill, correlation) gives R²=0.28
with the *wrong sign* on correlation and 0.865 collinearity between predictors —
it cannot be trusted to extrapolate. So "ERA5-Land will be decorrelated,
therefore it helps" is **not supported by our own evidence**.

## 5. Recommendation

1. **Do not count on ERA5-Land for 0.84.** It is the weakest of the four
   candidates on resolution, has a published negative result for its coarser
   sibling, and our member-value data does not support decorrelation alone.
   The fetch is cheap to let finish (it is running, network-bound, no GPU), so
   let it complete and *measure* it — but stop treating it as the plan.
2. **Investigate AORC** as the targeted lever: hourly/1 km, documented peak-flow
   gains in small basins, which is precisely the pivotal profile.
3. **Try SWE as an added input variable** before building any new corpus — the
   pivotal set is the snowiest group and this is the cheapest experiment.
4. Keep the diversity levers that are *measured* to work: the MSE member gave
   +0.0022 (§5 of literature notes); RMSE is training.

## 6. Honest odds on 0.84

Gap is +0.0049 from 0.8351. Measured levers so far deliver +0.001..+0.004 each,
and recombination can only close 10.6% of the remaining distance to NSE 1.0.
Reaching 0.84 likely needs **two or three** successful additions, not one.
It is achievable but not by a single new forcing member, and specifically not by
the one currently queued.
