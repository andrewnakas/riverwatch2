# PRE-REGISTRATION v2 — the second (and final) no-q held-out query

**Written 2026-08-09, BEFORE any candidate was scored against the val slice.**
Nothing in this document may be revised after seeing a gate result. If a rule
here turns out to be wrong, the correct action is to record the deviation
explicitly, not to edit the rule.

The v1 query is SPENT (held-out **0.8363**, config published-frozen). This
document governs the ONE additional query, if it is earned.

---

## 1. The decision surface

All development is judged **train-side**, on a temporal sub-split of the TRAIN
window only:

| | |
|---|---|
| weight fitting | 1980-10-01 … 1990-09-30 (127,606 rows) |
| **scoring (val slice)** | **1990-10-01 … 1995-09-30 (69,030 rows, 531 basins)** |
| combination rule | ONE GLOBAL inverse-MSE weight vector, θ=4.0, λ=0.25 |
| harness | `noq_harness.py` (baseline), `score_swap.py` (candidate gates) |

**FRESH TRAIN-SIDE BASELINE = 0.950458** (frozen 9-stream, production rule).
Equal-weight on the same frame = 0.946105.

Harness validity check (passed): the fitted weights reproduce the frozen
held-out ordering — multi5 0.2410 vs 0.2491, multi6 0.1958 vs 0.1910,
lstm_multi 0.1823 vs 0.1819.

⚠️ **The TEST window is not to be touched by any development step.** Scores
computed on the test period (e.g. `ft_pilot.sh`) are mechanics checks and may
NOT be used to select a configuration.

---

## 2. Gates every candidate must pass (pre-registered)

A candidate enters the shipped ensemble only if **ALL** hold, on the val slice:

1. **ENSEMBLE DELTA ≥ +0.001** vs the baseline recomputed on the *same* merged
   frame (paired, not against the standalone 0.950458).
2. **BREADTH ≥ 85%** of basins improved. *This is the binding gate.* The metric
   is a median over 531 basins — a rank statistic — so only broad gains move it
   (multi5 ships at 93.6% breadth; ≤70% cannot move a median regardless of
   cohort performance).
3. **Paired bootstrap 95% CI on the per-basin delta EXCLUDES ZERO** (10,000
   resamples of the per-basin delta — the paired test, not a difference of
   independent medians).
4. **≥2 SEEDS**, and the sign must not flip between them.
5. **A KNOWN-VALUE CONTROL** in the same run behaves as established (a
   redundant/duplicate member must come out ≈0 or negative). A control that
   drifts means the baseline is wrong, not that the candidate works.
6. **Both temporal halves** of the val slice agree in sign.

Member-level NSE is **not** a ship criterion in either direction: multi5 is a
*worse* member (−0.0062) and the single most load-bearing stream.

**Swap vs add.** A candidate that is a modified version of an existing stream
(SWA, per-basin finetuning) is scored as a **SWAP** for that stream, so the
result is never confounded with "one more member".

---

## 3. The go/no-go rule for spending the query

The query is spent **only if both** hold:

- **(A)** the assembled stack scores **≥ 0.954458** on the val slice
  (= baseline 0.950458 + 0.004), and
- **(B)** that gain is **confirmed on a ROTATED sub-slice** (fit on
  1980-88 + 1993-95, validate on 1988-93). This guards against the fact that
  every intermediate gate reused the 1990-95 slice, which makes it a de facto
  second test set.

⚠️ **+0.004 train-side is required for a +0.0037 test-side target** because
train→test shrinkage is expected and unquantified. It is not a safety margin to
be negotiated down after seeing a number.

**And in all cases: ASK THE USER before running the query.** (User decision,
2026-08-09: freeze the config in writing, present the train-side evidence, then
wait for an explicit go.)

If (A) is not met, the campaign ends by publishing **0.8363** together with the
metric-constraint and corrected-ceiling findings. That is a legitimate outcome,
not a failure.

---

## 4. The gate-shopping ledger

Every candidate screened against the 1990-95 val slice is recorded here, pass or
fail, so the final number can be discounted for multiplicity. **A candidate that
is not in this list was not screened; a candidate that was screened cannot be
removed from it.**

| # | candidate | mode | verdict | delta | breadth |
|---|---|---|---|---|---|
| 1 | Tier 0a NaN-matched weight refit | weights | **KILLED — premise false** (0.000% NaN in all 9 streams on the fit slice; the inner join removes dHBV NaN before fitting) | n/a | n/a |
| 2 | Tier 0b SWA | swap | **FAILED — redundant with seed averaging** | **-0.00051** | **60.1%** (gate 85%) |
|   | ↳ per-SEED SWA (wrong unit: nothing ships as 1 seed) | — | *would have shown* +0.0035/+0.0040 | | |
| 3 | Tier 0b epoch-snapshot pseudo-seeds | add | **NOT RUN — same mechanism as #2, and literally seed depth (measured-negative)** | | |
| 4 | Tier 1 per-basin finetuning | swap | **FAILED — median -0.0001, and its winners have ZERO median leverage (ranks 69/91 of 531)** | mean +0.0014, **median -0.0001** | **50.0%** (gate 85%) |
|   | ↳ same, scored TEST-side (mechanics only, NOT a selector) | — | *would have shown* +0.0005 | | corr(test,honest) = **-0.013** |
| 5 | Tier 2 hidden-512 member | add | *pending* | | |
| 6 | Tier 2 **hidden-128** member (1 seed) | add | **FAILED** — lowest breadth of any member tested | **-0.00087** | **27.5%** |
|   | ↳ solo skill **0.7964** vs the shipped h256 **0.8065** (−0.0101): less capacity, same corpus, **no new information** — unlike multi5, which was also weaker but added diversity | | | | |
| 7 | Tier 2 multi5b COOP-only stations | add | *not started* | | |
| 9 | Loss-variant member **RMSE s888** | add | **FAILED** (CI excludes 0 NEGATIVELY) | **-0.00214** | **39.9%** |
| 10 | Loss-variant member **MSE s777** | add | **FAILED** (CI excludes 0 NEGATIVELY) | **-0.00238** | **33.1%** |
|   | ↳ member skill: RMSE **0.8071** BEATS the NSE-loss baseline 0.8065; MSE 0.8021. Residual corr with existing streams ~0.82 (multi5 is 0.81) | | | | |
| 8 | Tier 2 channel-dropout member | add | *not started* | | |
| 22 | **multi5b s222** (multi5 minus its two snow channels) | add | **FAILED — 1 of 3 gates** | paired **+0.00022** (diff-of-medians −0.00044) | **56.7%** |
|    | ↳ CI [+0.00008,+0.00039] **excludes zero**, so the effect is REAL but ~5× too small and far too narrow. ⚠️ Solo skill **0.7985 vs multi5's 0.7927** — dropping the snow channels makes a BETTER standalone member and a WORSE ensemble contributor, so multi5's snow channels earn their place by decorrelation, not by own skill. | | | | |
| 21 | **Multi-lead temporal ensembling** (average h=1..k predictions sharing a target date) | recombine | **IMPOSSIBLE BY CONSTRUCTION** — not a failed lever | n/a | n/a |
|    | ↳ the dumps use a **stride-14 t0 grid with h=1..14**, so the windows TILE the timeline exactly: every target date has **exactly one** prediction (h=1 targets also appearing at h=2: **0 of 385**). There is nothing to average. Capturing this would require re-dumping at stride<14. | | | | |
| 20 | **Rank/robust combinations** (equal-weight, median, trimmed x3, winsorised x3) | recombine | **ALL FAILED** | equal **-0.0044**, median **-0.0037**, best winsorised -0.0004 | 25-50% |
|    | ↳ sanity check PASSED: rank-remapped mean reproduces equal-weight to 6 dp (0.946105) | | | | |
| 18 | **lam swept past equal-weighting** (−0.5 … +1.5 at frozen theta) | reweight | **FAILED — and confirms the baseline is a true optimum** | best +0.00033 (lam=−0.5); lam=+0.5/+0.75 give **−0.0012/−0.0031** | 27–64% |
| 19 | **Sample-level diagnostics** (warm-up trim, zero-flow exclusion) — DIAGNOSTIC ONLY, would change the protocol | n/a | zero-flow rows are **NOT** dragging the metric (excluding them costs −0.0005); the apparent +0.0025 from trimming 90 rows/basin is a **sampling artifact** (69,030 → 21,240 rows) | n/a | n/a |
| 17 | **Per-SEED stream weighting** (31 individual dumps instead of 9 seed-averaged streams) | reweight | **FAILED** — weights come out near-uniform (top six 0.054–0.058 vs 0.032 uniform) ⇒ **seed identity carries no exploitable signal** | **−0.00021** | 62.0% |
| 14 | **Nonlinear combination space** (geometric / asinh / log1p mean) | recombine | **ALL FAILED** | −0.00131 each | 39.7% |
| 15 | **Per-basin affine correction of the ENSEMBLE** (a·ens+b, 5 shrinkages) | post-hoc | **ALL FAILED** | −0.0109 … −0.0005 | 36.7–53.1% |
| 16 | **NNLS stacking fit on FIT** (deployable, unlike the historical TEST-fit oracle) | reweight | **FAILED** — zeroes 6 of 9 streams and still loses | **−0.00170** | 60.5% |
| 12 | **Per-basin weighting rules** (7 variants: best-member, top-k, per-basin inv-MSE) | reweight | **ALL FAILED** — monotonically worse the more they commit per-basin | best-member **-0.0087** … mildest **-0.0001** | 38-53% |
|    | ↳ ORACLE bound for the whole family: **+0.0070** at **83.6% breadth** (hindsight only — member-within-basin rank does not transfer) | | | | |
| 13 | **theta/lam hyperparameter sweep** (35 cells) | reweight | **FAILED — confirmed noise on the ROTATED slice** | +0.00058 → **+0.00030** rotated | n/a |
|    | ↳ on rotation the RANKING RESHUFFLED (grid winner theta=8 fell to +0.00030; 3rd-place theta=12 rose to +0.00040) ⇒ selection artifact. **Keep theta=4.0/lam=0.25.** | | | | |
| 11 | **AORC as a 10th stream** (3 seeds, previously rejected under EQUAL weighting; never tested under inverse-MSE) | add | **FAILED — but harmless, not harmful** | **+0.00045** | **51.4%**, CI [-0.00005,+0.00010] straddles 0 |
|    | ↳ inverse-MSE assigned it **0.0411** (rank 8/10) — the rule down-weights an inhomogeneous stream to irrelevance rather than being damaged by it | | | | |

| 23 | **SNOTEL station channels ("multi8")** — the last data-source candidate on the honest list | add | **KILLED BEFORE GPU — premise false.** GHCN-Daily *ingests* SNOTEL, so multi5 station channels ALREADY carry it in **50 basins** (`src_networks` = COOP+SNOTEL). A direct AWDB pull adds **20 new basins = 3.8% of 531**, median frac_snow **0.63** (vs 0.096 overall) => a snow specialist, the arithmetically dead category. Best-case breadth 12.8% vs an 85% gate. | n/a | n/a |
|    | > WARNING, a near-miss false mechanism: multi5 `obs_mask_stn` is non-zero in **497/531 = 93.60%**, exactly its shipped breadth (93.6%). Tested against `gate_multi5b_s222.json` per-basin deltas: **uncovered basins improve MORE often (64.7%) than covered (56.1%)** => coincidence, not causation. A retrained member perturbs every basin, not only where a channel is non-zero. | | | | |

| 24 | **THE PHASE LEVER** — `h==1` is a 1-in-14 DATE SUBSAMPLE, not a lead time | evaluation sample | **NEW SURFACE — not a candidate but a MEASUREMENT-PRECISION finding that bears on every row above.** `nh_to_dump.py` reshapes a CONTINUOUS daily simulation onto a stride-14 grid, so `h` is position-within-window; `phase_c.py:31` filters to `h==1`, which lands on ONE fixed weekday (all 69,561 val targets, dayofweek=2). **13/14 of every LSTM prediction is discarded and is already on disk.** | phase spread **0.017333 = 4.68x the +0.0037 gap**; h=1 ranks **3rd luckiest of 14**; all-dates **0.940182** vs h=1 **0.943833** | see below |
|    | > **WHERE IT BITES: BREADTH, the binding gate.** The paired DELTA is stable (multi5 +0.00204..+0.00273 across phases, 14/14 clear +0.001) — a level shift cancels in a paired comparison. **Breadth does not:** multi5 measures **76.5-81.2% on every single phase** but **94.7% on ALL DATES**. Falsified properly — random 1/14 subsamples of the all-dates frame reproduce 78.5/78.9/79.8%, so **breadth is SAMPLE-SIZE limited**, not a property of the member. A sign test on ~130 rows/basin is the weakest statistic in the harness and it is the gate. | | | | |
|    | > **Consequences.** (a) Members rejected at 27-57% breadth were measured the same noisy way — ordering may survive, absolute values do not; **re-scoring the ledger on all leads costs ZERO GPU.** (b) All-dates is LOWER (0.9402 vs 0.9438), so this is a tighter estimate, **NOT a route to 0.840 by redefinition** — any such reframing is a protocol change needing its own pre-registration. (c) **dHBV dumps are h=1 ONLY**, so the full 9-stream join collapses to h=1; extending this to dHBV needs a re-dump (CPU eval, no training). | | | | |
|    | > WARNING, the trap: the first run printed spread **0.000000** / "rank 1 of 14" because the 9-stream inner join silently collapsed to h=1 at exactly 69,030 rows. Per ledger entry 21, **an exactly-0.000000 result is a pipeline hypothesis, not a null** — that rule has now paid for itself twice. | | | | |
|    | > **CONFIRMED ON THE FULL 9-STREAM PRODUCTION ENSEMBLE.** All 9 dHBV members re-dumped with a new opt-in `--dump-all-leads` (2,752,904 rows each, 14 leads, 531 basins, gzip-valid 9/9, ~2h15m CPU-only). **Regression check: the new h=1 reproduces the old h=1-only dumps BIT-IDENTICALLY, max|diff| = 0.000000000.** With the **frozen production weights held fixed**, h==1 = **0.950452** vs PREREG **0.950458** (diff **-0.000006**) => the frame is provably right. Phase spread **0.014655 = 3.96x the gap**; h=1 ranks **4th of 14**; all-dates **0.949978**. | | | | |
|    | > ⚠️⚠️ **A CONFOUND THAT ALMOST CORRUPTED THIS:** refitting the inverse-MSE weights on the all-leads frame FAILED the validity check (h=1 = 0.946695, **-0.003763** vs PREREG — bigger than the gap). Cause: **dHBV h=1 is dHBV's WORST phase** (MSE 44,650 at h=1 vs 28,452 at h=7), so fitting on all leads **tripled** dHBV weight (dhbv_nldas 0.0435 -> 0.1398) and lowered the ensemble. **To measure a phase effect the weights MUST be held fixed** — refitting changes two things at once. (Live follow-up question: production's low dHBV weights are partly an artifact of fitting on dHBV's unluckiest phase.) | | | | |
|    | > **Ledger re-score on all leads** (vs a 4-LSTM baseline, NOT the 9-stream one): **multi5 77.4% -> 92.5% flips FAIL->PASS**; multi5swa **84.7%** and multi6 **81.2%** land just UNDER the 85% bar; bad members get decisively worse (multi4 27.7% -> **6.0%**, multih128 32.2% -> **8.9%**). **It separates rather than inflates**, which is the reason to believe it. ⚠️ An earlier version of this row used `mse**(-THETA/2)`; the production rule is `mse**(-THETA)` — corrected numbers shown. | | | | |
|    | > ⛔⛔ **FINAL: AGAINST THE FROZEN 9-STREAM BASELINE, EVERY CANDIDATE IS NEGATIVE ON ALL LEADS.** `rescore9.py`, validity anchor **EXACT** (9-stream h==1 = **0.950458** vs PREREG **0.950458**, diff −0.000000; and the h=1 breadth column reproduces this ledger to the decimal — aorc 51.4%, multi5b 56.7%, multih128 27.5%, rmse 39.9%, mse 33.1%). Deltas on ALL leads: aorc **−0.000193**, multi5b **−0.000214**, multi5swa **−0.000372**, eps05 −0.000575, multi4 −0.000728, multih128 −0.000775, mse −0.001444, rmse −0.001483. Breadth collapses too (multi4 27.5% → **3.0%**, multih128 27.5% → **4.1%**). | | | | |
|    | > ⇒ **The all-leads sample does NOT rehabilitate any rejected member — it CONFIRMS the rejections and sharpens them.** The 9-stream baseline itself rises to **0.953930** on all leads (vs 0.950458 at h=1), i.e. a stronger baseline the candidates fail to beat. ⚠️ The apparent **multi5swa revival** (+0.00207 / 84.7% against a 4-LSTM baseline) is an ARTIFACT of that thin baseline: against the frozen 9 it is **−0.00037 / 30.7%**. Adding SWA to an ensemble already holding multi5×5 + multi6×3 is redundant — the original SWA closure STANDS. | | | | |
|    | > ⭐ **METHOD LESSON: the BASELINE you score against matters more than the SAMPLE you score on.** A candidate can look like a clear pass against 4 streams and be worthless against 9. Always gate against the frozen production set. ⇒ The phase lever is a **measurement-precision** finding (the reported number sits on a 1-in-14 phase, spread ≈4x the gap), **not** a source of new members. | | | | |



| 25 | **Fit the ensemble weights on ALL 14 LEADS** (14x the weight-fitting data, free) | reweight | **FAILED — and the apparent gain was a SCORING-SAMPLE change, not better weights.** Refit-and-rescore shows 0.953930 vs 0.950458 (+0.003472, almost exactly the gap) — but that changes BOTH the fitting and the scoring sample. Holding the SCORING ROWS FIXED: | **−0.003763** | **31.3%** |
|    | > The 2x2 (weights never see VAL in any arm): **A** fit@h1→score@h1 = **0.950458** (= PREREG exactly) · **B** fit@all→score@h1 = **0.946695** (**−0.003763**) · C fit@h1→score@all = 0.949991 · D fit@all→score@all = 0.953930 (+0.003939). **A→B is the decision-relevant arm** — same scoring rows, 14x fitting data — paired median **−0.001565**, breadth **31.3%**, bootstrap CI **[−0.002103, −0.001151] EXCLUDES ZERO NEGATIVELY**. | | | | |
|    | > **Mechanism: dHBV h=1 is dHBV's WORST phase** (MSE 44,650 at h=1 vs 28,452 at h=7), so fitting on all leads flatters it and **triples** dHBV weight (dhbv_nldas 0.0435→0.1398, dhbv_daymet 0.0651→0.1382, dhbv_maurer 0.0399→0.1071) at the expense of **multi5 0.2410→0.1734** and **multi6 0.1958→0.1326** — the two members that carry the ensemble. ⇒ **Production's low dHBV weights are CORRECT FOR THE METRIC REPORTED**; weights must be fit on the same sample the metric is computed on. | | | | |
|    | > ⇒ **NO RECORD CHANGES.** Held-out **0.8363** stands (the phase work touched no test-window data; the dHBV re-dump wrote NEW `*_ALLH_TRAIN_*` files and left every original byte-identical). Train-side **0.950458** stands. ⚠️ **Restating the number as 0.9539 would be a protocol change made after seeing that the new sample scores higher** — exactly what the pre-registration exists to prevent. | | | | |


| 26 `[v3]` | **W1/W2 — PHASE-ROBUST WEIGHTING** (the last untested combination idea: every rule in entries 1-21 was fit AND scored on one phase) | reweight | **BOTH FAIL — and this is the cleanest demonstration in the ledger of why the SAMPLE rule had to be pre-registered.** W1 = fit inverse-MSE on each of the 14 phases separately, average the 14 weight VECTORS. W2 = fit on a phase-balanced subsample of the SAME size as the h=1 fit (127,596 vs 127,606 rows), so fitting-set SIZE is held constant and only phase COMPOSITION varies. Scoring rows IDENTICAL across arms; only the weight vector differs (the A->B comparison of entry 25). **VALIDITY ANCHOR EXACT: production weights @ h==1 = 0.950458 vs PREREG 0.950458, diff −0.000000.** | **W1 −0.001921 / W2 −0.001246** | **W1 29.9% / W2 31.6%** |
|    | > ⚠️⚠️ **THE TRAP, MEASURED: the SAME candidate reads PASS or FAIL depending only on the scoring sample.** W1 on ALL LEADS = **+0.004418 paired, 90.4% breadth, CI [+0.003959,+0.004915]** — that clears every gate and is larger than the entire remaining gap to 0.840. W1 on **h==1 (the REPORTED sample, PREREG v3.2 decision rule for a weighting candidate)** = **−0.001921 paired, 29.9% breadth, CI [−0.002583,−0.001371] EXCLUDES ZERO NEGATIVELY.** Same weights, same basins, opposite verdict. W2 behaves identically (all-leads +0.003860 / 91.5%; h=1 −0.001246 / 31.6%). **v3.2's per-class sample rule was pre-registered 2026-08-10 19:57, BEFORE these numbers existed** — which is the only reason the +0.0044 is reported here as an artifact rather than banked as a result. | | | | |
|    | > **MECHANISM — the pre-registered dHBV signature fires exactly as entry 25 predicted.** Total dHBV weight mass shifts **+0.2618** (dhbv_nldas 0.0435 -> **0.1565**, ~3.6x; dhbv_daymet 0.0651 -> 0.1382; dhbv_maurer 0.0399 -> 0.1156) at the expense of every LSTM stream (multi5 0.2410 -> 0.1625, multi6 0.1958 -> 0.1266, multi 0.1823 -> 0.1232). Because **dHBV's h=1 is dHBV's WORST phase**, any fitting scheme that gives the other 13 phases a vote flatters dHBV and reweights toward it — then the metric is read back on h=1, where dHBV is weakest. ⇒ The per-phase-average and the phase-balanced fit are **two different routes into the same failure**, which is what makes it a mechanism rather than a coincidence. | | | | |
|    | > Both temporal halves agree in sign and negatively (W1 early −0.001884 / late −0.001846; W2 −0.001332 / −0.001458), so this is not a window artifact. **No rotated sub-slice was run for either arm** — PREREG section 3(B) confirmation is earned only by an arm that PASSES; running it on a failed arm would be searching for a slice that rescues it. W2's pre-registered conditional-skip (v3.3) was moot: it ran in the same pass and failed independently, so the mechanism is **confirmed on two arms rather than assumed from one**. | | | | |
|    | > ⇒ ⭐⭐⭐ **COMBINATION IS NOW CLOSED ON THE PHASE SURFACE TOO.** The one combination idea the closed set did not cover — that every prior test lived on a single phase — has been tested and fails, by the same mechanism that killed entry 25. The global inverse-MSE vector fit at h==1 remains the attractor. ⇒ **Production's low dHBV weights are not a fitting artifact to be corrected; they are correct FOR THE METRIC REPORTED.** ⇒ **NO RECORD CHANGES: 0.8363 held-out and 0.950458 train-side both stand.** | | | | |
|    | > ✅✅ **INDEPENDENTLY VERIFIED (`verify_w1.py`), and the verification changes the INTERPRETATION without changing the verdict.** Scoring production vs W1 weights on **each phase separately** (anchor again exact, 0.950458, diff −0.000000): **W1 beats production on 12 of 14 phases** — h=7 **+0.008289**, h=11 +0.007477, h=10 +0.007299, h=9 +0.006757 — and loses on only two: h=2 (−0.000694) and **h=1, the REPORTED phase, by −0.004178**. ⇒ The all-leads "+0.0044" is not noise and is not a coding error: **W1 really is a better weighting on 12/14 phases.** It fails because the benchmark reports the ONE phase where it is worst. | | | | |
|    | > **MECHANISM MEASURED, not assumed — the stream families have INVERTED phase profiles.** `dhbv_nldas` MSE: **53,040 at h=1** (its WORST) falling to **24,487 at h=9** (2.2x swing). `lstm_multi5` MSE: **25,737 at h=1** (its BEST) rising to **41,213 at h=10**. ⇒ **h=1 is simultaneously dHBV's worst phase and the LSTMs' best.** Any fitting scheme that lets the other 13 phases vote therefore moves weight toward dHBV — and that weight is then graded on the single phase where dHBV is weakest and the LSTMs strongest. This is why W1 and W2 fail by the same signature, and why the failure is structural rather than incidental. | | | | |
|    | > ⚠️⚠️ **THE HONEST SUMMARY, which is NOT "phase-robust weighting is worse":** it is better on 12/14 phases and worse on the reported one. **The reported metric sits on an unrepresentative phase.** That is a stronger version of entry 24's precision finding — the h=1 convention does not merely add sampling noise, it systematically favours one stream family over another. ⭐ For the paper this is a **benchmark-design** result: a reported number computed on one fixed weekday phase can invert the ranking of two model families. ⛔ It is **NOT** a licence to restate the record on all leads — that remains a protocol change made after seeing the new sample score higher, and the pre-registered decision sample is h==1. | | | | |

| 27 `[v3]` | **CHANNEL DROPOUT (multidrop)** — the last "final four" member family, and the only concrete GPU experiment the 2026-08-10 handoff left standing | add | **s222 FAILS on BOTH samples. v3 decision (ALL LEADS): paired −0.000609, BREADTH 4.9%.** h=1: paired −0.000383, breadth 26.4% (140/531), CI [−0.000482,−0.000291] excluding zero negatively. Anchor EXACT (0.950458, diff −0.000000) and the h=1 column reproduces independently across **two separate code paths** (`score_swap.py` via gate_watcher, and `perseed_allleads.py`) ⇒ harness validated. ⭐ Breadth **collapses 26.4% → 4.9%** on all leads, the diagnostic signature of a genuinely bad member (multi4 27.5%→3.0%, multih128 27.5%→4.1%), not of a marginal one. ⭐ Inverse-MSE independently discounted it: **weight 0.0308, the LOWEST of all 10 streams** — fail-safe behaviour again ([[aorc-under-invmse-harmless-not-helpful]]). ⚠️ Dumped from the NESTED `continue_training_from_epoch008/model_epoch016.pt`; the queue's run-dir glob had silently written no dump for 6 h. | **−0.000609** | **4.9%** |
|    | > ⚠️⚠️ **TWO CONFOUNDS, recorded BEFORE the verdict is read.** (1) **No `clip_gradient_norm`** — multi5 carries 1.0 precisely because sparse channels produce rare huge gradients, and a channel-DROPOUT member has the same exposure but its config never inherited the key. (2) **The seeds are not comparable to each other** — s222 is epoch 16 of a RESUMED run, s111 a fresh uninterrupted 30-epoch run. ⇒ A FAIL from s222 is **weak evidence about channel dropout as an idea**; only a PASS would have been remarkable. | | | | |
|    | > ⭐⭐ **NEW: the s222 divergence was the RESUME, not only the missing clipping.** From the resumed-segment log: the LR drop to 5e-4 was applied at **14:43**, but the blow-up is at **epoch 17 (14:23, 0.01248→0.02305)** and **epoch 18 (14:33, 0.05863)** — i.e. **BEFORE the drop, still at the initial 1e-3.** `continue_training` **restarts the LR schedule**, so the resumed segment re-ran a high-LR phase the original 30-epoch schedule had already finished with. ⇒ s222 destabilised in an LR regime it should never have re-entered, and its instability is **not** evidence that channel dropout is inherently unstable. | | | | |
|    | > ⛔ **D1 (the clipped retest) STANDS DOWN on the pre-registered rule.** PREREG v3.3 fixed the condition in advance as **all-leads breadth ≥ 15%**; s222 measures **4.9%**, in the multi4/multih128 dead regime. The rule fires on its own terms — no argument after the fact. ⚠️ Two caveats kept on the record: it is **one seed**, and it is the **confounded** one. **multidrop s111 (fresh, full schedule, best 0.01241 at ep16 — already better than s222's 0.01248) is the cleaner read and was still training when this was written.** If s111's all-leads breadth also lands in single digits the family is closed on two independent reads; if it clears 15%, D1 is re-armed under this same pre-registered rule. Configs are staged and verified as single-variable changes (`gpu1080/cfgls_multidropc_s{111,222}.yml`). | | | | |
|    | > ✅✅ **CONFIRMED BY THE FRESH RUN: the resume, not channel dropout, caused the divergence.** multidrop **s111 passed straight through epoch 17** — the exact epoch where s222 blew up — **descending monotonically**, on the SAME unclipped config (the two YAMLs differ only in `seed`/`experiment_name`): ep14 0.01288 → ep15 0.01262 → ep16 0.01241 → **ep17 0.01230**, versus s222's ep16 **0.01248** → ep17 **0.02305** (1.9×) → ep18 **0.05863**. ⇒ The instability belonged to the **resumed segment's re-run of the 1e-3 phase** (`continue_training` restarts the LR schedule), not to making inputs sparser. **The earlier root-cause note that blamed the missing `clip_gradient_norm` alone is superseded**: clipping is still a real omission, but it is NOT what produced this divergence. ⭐ **Ops rule: judge stability from a FRESH run; a resumed run is re-exposed to an LR phase it has already survived once.** | | | | |

| 28 `[v3]` | **THE PHASE REORDERS THE MODEL LEADERBOARD** — not a candidate, a **benchmark-design finding** that generalises entry 26 from "our weighting is phase-sensitive" to "the reported convention can rank a whole model FAMILY last" | evaluation sample | **MEASURED, zero GPU, 531 basins / 967,482 rows, standalone seed-averaged streams, NO weights and NOTHING fitted.** At **h=1 (the reported phase) all SIX LSTM streams beat all THREE δHBV streams** — δHBV holds ranks 7/8/9. **Pooled over all 14 phases `dhbv_daymet` is RANK 1** (0.9500 vs lstm_multi5 0.9466), and δHBV takes 3 of the top 4. `dhbv_daymet` moves **+6 rank places**. **21 of 36 stream pairs (58%) invert**, including at the top (dhbv_daymet beats lstm_multi5 on **10/14** phases and lstm_multi6 on **11/14**, but loses to both at h=1). Kendall tau vs the h=1 ranking decays **monotonically +1.000 → −0.111 (h=12)**, mean over the other 13 phases **+0.107** ⇒ effectively uncorrelated. | n/a | n/a |
|    | > **MECHANISM: spin-up of explicit storage states.** Every δHBV stream is **worst at h=1** and best at h=12 (daymet 0.9179→**0.9607**, spread 0.0428; nldas 0.0463; maurer 0.0460), while the LSTMs peak at h=1-2 and **decline** mid-window (lstm_multi 0.9464 → trough **0.9294** at h=6). h=1 is the first day after each 14-day window's `t0` re-initialisation: a **conceptual model with explicit storage must re-equilibrate**, an LSTM entering with a 365-day input sequence need not. ⇒ **The reported convention samples precisely the day that penalises conceptual models most.** | | | | |
|    | > ✅✅ **FALSIFIED PROPERLY — three checks, `check_dhbv_alignment.py`.** (1) **Truth consistency**: δHBV `truth` vs LSTM `truth` on identical (basin,date) gives max abs diff **0.0078 at shift_days=1 (production)** versus **65,200 cfs at shifts 0 and 2** ⇒ the loader is right by ~8 orders of magnitude; misalignment ruled out. (2) **Opposite-direction control (decisive)**: a misalignment would displace δHBV **uniformly at every phase** (flat penalty, not a monotone ramp) and a scoring artifact would move **both** series the same way — instead **δHBV climbs while the LSTM falls**, which only a real model-class difference produces. (3) ⚠️ **Reported though it is ambiguous**: δHBV's own skill is nominally best at shift 0 (0.950060 vs 0.950034 at shift 1) — a **0.000026** gap, 3 orders of magnitude under the 0.043 phase spread, because median NSE is dominated by the seasonal cycle and is nearly shift-invariant. The discriminating evidence is the truth match, not the skill tie. | | | | |
|    | > ⚠️ **MY PRE-REGISTERED PREDICTION WAS WRONG.** `NOVEL_PROBES_prereg.md` (timestamped **22:10**, before any output existed) predicted *"inversions among the middle/bottom ranks, not a change of winner."* **The winner changed.** Recorded as a miss rather than revised — the point of writing predictions down first. | | | | |
|    | > ⛔ **DOES NOT RAISE OUR NUMBER AND MUST NOT BE USED TO.** The record stays the pre-registered h=1 **0.8363**. The honest corollary belongs in the paper: **our own reported number sits on the phase that most flatters our LSTM-heavy ensemble.** | | | | |
|    | > ⚠️⚠️⚠️ **PARTIALLY RETRACTED THE SAME NIGHT — the cross-family comparison is CONFOUNDED.** A source read (`scripts/train_mblstm.py`, `scripts/nh_to_dump.py`) showed **`h` MEANS DIFFERENT THINGS to the two families.** **LSTM**: `h` is a RESHAPING INDEX over a *continuous daily simulation* that re-anchors every day (`seq_length: 365`, `predict_last_n: 1`) ⇒ **every LSTM prediction is 1-day-ahead regardless of h**. **δHBV**: `h` is a TRUE HORIZON — one forward pass emits all 14 (`y = q_n[t0+1 : t0+1+HORIZON]`, *"the model computes all HORIZON of them either way"*) and it drives on **raw physical forcing over the FULL context+horizon span**, `sl = slice(a, t0+1+HORIZON)`. ⇒ At "h=7" this compared **the LSTM's 1-day-ahead simulation against δHBV's 7-days-after-initialisation hindcast.** Both see observed forcing (neither is a real forecast), but δHBV **accumulates more forcing per initialisation as h grows**, which is why its skill RISES with h rather than decaying. **NOT like-for-like.** | | | | |
|    | > ⭐ **This single mechanism explains BOTH puzzles at once**: (a) why δHBV is worst at h=1 — that is its *least-informed* lead, one day past initialisation, **not** a storage "cold start"; (b) why the deficit is **uniform across catchment attributes** (frac_snow **−0.100**, aridity +0.006, area +0.041, n=531) — an *information-accumulation* effect has no reason to track catchment storage, which is exactly what the spin-up test measured and why my storage mechanism was falsified. | | | | |
|    | > ✅ **WHAT SURVIVES (unchanged):** (1) the **within-LSTM** phase effect — all six LSTM streams share identical `h` semantics, so entry 24's precision finding and entry 26's weighting inversion **stand in full**; (2) ⭐ **the production ensemble MIXES streams whose `h` means different things** — a real, previously undocumented pipeline asymmetry, and a legitimate methods caveat for the paper. ❌ **RETRACTED:** *"pooled over phases δHBV is rank 1"* as a claim about **model skill**, and the headline *"a CAMELS leaderboard can rank a whole model family last."* The tabulated numbers are correct as computed; their **cross-family interpretation is not**. | | | | |

| 29 `[v3]` | **NO-Q TIMING PROBE** — is the no-q ensemble mistimed? (the with-q "no lag" verdict was 16 basins / 1 water year and **anchored to observed Q by construction**) | diagnostic | **NO LAG — decisively, and on a 33× larger sample.** Frozen 9-stream ensemble, production weights, **967,482 CONTINUOUS daily rows** (day-gap==1 fraction **0.999**), 531 basins. Global shift scan: **shift 0 optimal** (0.949991; ±1 day collapses to ~0.52). **PER-BASIN ORACLE shift gain = +0.000000**, with **526/531 basins choosing shift 0 unprompted** (+1: 4, −1: 1). ⇒ Even with full hindsight, re-timing buys **exactly nothing**; every routing/lag/per-basin-timing lever is dead on this track too. | +0.000000 | n/a |
|    | > ⭐⭐ **THE STRUCTURAL FINDING — BOTH limbs under-predict on event days.** Top-1% squared-error days (n=10,089 = **1.04%** of rows): rising **−435.32 cfs**, falling **−148.46 cfs**. Ordinary days (n=957,393): rising −0.41, falling **+5.57**. Rising-limb under-prediction is **1054× larger** on event days. ⭐ **A genuine lag produces OPPOSITE signs** (late ⇒ under-predict the rise, over-predict the recession); these are the **same sign**, while ordinary days show the normal opposite pattern. ⇒ **Two independent lines of evidence — a zero oracle gain and same-sign limb errors — converge: the model is ON TIME and TOO SMALL on the 1% of days carrying the error. Magnitude, not phase.** | | | | |
|    | > ⭐ **A FALSE PREMISE IN OUR OWN NOTES made this look impossible.** `shared-error-is-phase` states *"a sub-daily phase error cannot be diagnosed from these dumps at all — a continuous or stride-1 dump is required."* True when written; **expired with the ALLH re-dump**, because `h` is position-within-window so h=1..14 are **consecutive days** (verified: within-window date gaps are all exactly 1 day). The continuous series had been on disk for hours. ⇒ **Re-read old "impossible" conclusions after any change to the dump format.** | | | | |

| 30 `[v3]` | **THE `h`-SEMANTICS ASYMMETRY** — our own pipeline mixes two different meanings of the window index | pipeline / methods | **ESTABLISHED FROM SOURCE, then quantified.** `scripts/nh_to_dump.py`: the LSTM dump reshapes a **continuous seq-to-one daily simulation** onto the stride-14 grid (`seq_length: 365`, `predict_last_n: 1`) ⇒ **`h` is a RESHAPING INDEX and every LSTM row is effectively 1-DAY-AHEAD**. `scripts/train_mblstm.py`: δHBV emits **all 14 horizons from ONE forward pass** per `t0` (`y = q_n[t0+1 : t0+1+HORIZON]`, *"the model computes all HORIZON of them either way"*) and drives on **raw forcing over the full 365+14 span** (`sl = slice(a, t0+1+HORIZON)`) ⇒ **`h` is a TRUE HORIZON and δHBV accumulates more forcing per initialisation as h grows.** ⇒ **Comparing the families at matched `h` compares an LSTM 1-day-ahead simulation against a δHBV h-days-past-init hindcast — NOT like-for-like.** | n/a | n/a |
|    | > **QUANTIFIED** (`h_semantics_impact.py`, anchor exact 0.950458 / diff −0.000000). Family sub-ensembles **SWAP PLACES**: at **h=1 (reported)** LSTM-only **0.948166** vs δHBV-only **0.928088** (LSTM +0.020); at **h=8** (δHBV's best lead, chosen on the FIT window) LSTM-only **0.940326** vs δHBV-only **0.958248** (δHBV +0.018). The inverse-MSE rule responds on its own: δHBV weight mass **0.1486 at h=1 → 0.4890 at h=8**. ⇒ ⭐ **Production's low δHBV weights are NOT undervaluation — they are correct FOR THE LEAD WE SCORE AT.** Third independent confirmation that weights must be fit on the sample the metric is computed on (entries 25, 26, 30). | | | | |
|    | > ⚠️⚠️ **THE `C − A = +0.001855` FIGURE IS NOT HEADROOM — half the remaining gap, and the third instance of the SAME trap.** (a) A and C **score on different days** (h=1 rows vs h=8 rows) — the C→D scoring-sample confusion of entry 25 and entry 26; (b) **not deployable even in principle**: at inference the reporting date fixes the lead, so δHBV cannot be read at its 8-day lead for a date that must be reported at 1-day lead; (c) the honest **deployable** version was already measured **negative** (entry 26: re-weighting toward δHBV on the reported sample = **−0.0019**, CI excluding zero). | | | | |
|    | > ⇒ **VALUE: a methods caveat, not a lever.** Our production ensemble mixes streams whose window index means different things and **nothing in the pipeline flags it**; the reported number is measured where one family is at its most-informed and the other at its least. **The clean fix if the cross-family question ever matters: re-dump δHBV as a CONTINUOUS DAILY SIMULATION matching the LSTM construction (~2 h, CPU-only).** Until then **no honest δHBV-vs-LSTM ranking can be drawn from these dumps** — which is why the entry-28 cross-family claim was retracted. ⛔ Record unchanged: **0.8363**. | | | | |

| 31 `[v3]` | **POOLED-RESIDUAL CLUSTER WEIGHTING** — the one combination arm the closure never actually tested | reweight | **FAILS, and its SHAPE confirms the mechanism.** Anchor EXACT (K=1 reproduces **0.950458**, diff −0.000000). 28 cells (K = 1,2,3,4,6,8,12 × lam = 0,0.25,0.5,0.75), clusters from **static attributes only** (no skill information, so membership cannot leak val performance), weights fit on POOLED fit-window residuals per cluster. **Best cell K=3, lam=0: paired +0.000025 at 52.0% breadth** — **40× below the +0.001 delta gate and 33 points below the 85% breadth gate.** | **+0.000025** | **52.0%** |
|    | > ⚠️ **WHY IT WAS NOT ALREADY CLOSED:** the combination closure enumerates 12 per-basin + 5 affine + 3 nonlinear + 1 NNLS + 35 θ/λ rules — **cluster/two-regime is absent from that list**, and `optimizer_pass8.py` (written 2026-08-08 to test exactly this) has **no result file and no log**: designed, never run, surface then declared closed on the other 56. ⚠️ Also distinct from `gate_eval.w_cluster`, which sets *cluster weight = MEAN of its basins' per-basin inv-MSE weights* — that **averages already-noisy estimates and keeps their noise**; this arm **pools the RESIDUALS first**, the only version that actually reduces variance. | | | | |
|    | > **THE HYPOTHESIS (reasonable, and now falsified).** Per-basin weighting fails *monotonically with commitment* (best-member −0.0087 → top-2 −0.0033 → top-4 −0.0006 → per-basin inv-MSE −0.0001), a **variance signature** rather than absent signal; and the per-basin ORACLE has **+0.0070 at 83.6% breadth — the only measured quantity in the campaign LARGER than the +0.0040 gap to the query bar.** Pooling K basins cuts weight-estimation variance ~√K: the untested middle of the bias–variance curve. | | | | |
|    | > ⭐⭐ **THREE STRUCTURES MAKE THIS A MECHANISM, NOT A NULL.** (1) It **peaks at small K and decays monotonically** as K grows (+0.000025 at K=3 → **−0.000120** at K=12) — the per-basin variance problem **reappearing exactly as clusters shrink**. (2) **Shrinkage walks every K≥4 back toward zero from below** (K=12: −0.000120 → −0.000021 as lam 0→0.75), reproducing the campaign's signature ordering. (3) **Breadth never exceeds 53%** — a coin flip at every setting. ⇒ **Pooling does NOT rescue per-basin weighting: the +0.0070 oracle is pure hindsight, and any rule exploiting it captures ≤0.4% of it.** The per-basin skill ordering is not merely noisily estimated — **it is not there between windows.** | | | | |
|    | > ✅ **Pre-registered and called correctly.** `CLUSTER_PREREG.md` (timestamped **23:54:45**, before any output) predicted *"small positive deltas at K=2-4 with breadth well under 85% — a FAIL under the gate"*, plus four traps including *"a win at large K is overfitting wearing a cluster costume"*. **Correct on all counts.** ⇒ **COMBINATION IS NOW CLOSED WITH NO UNTESTED ARM: 84 rules across six independent families, none positive.** | | | | |

| 32 `[v3]` | **HIDDEN-512 (multih512)** — the width lever; fills the ledger's long-open slot #5 | add | **FAILS. Paired +0.000456 at 62.1% breadth (330/531)** vs gates of +0.001 and 85%. Anchor exact (baseline 0.950458). Trained cleanly to epoch 30 (best loss **0.00610**, monotone descent, no divergence despite carrying no `clip_gradient_norm`). ⭐ It is the **best member delta ever measured in this campaign** (just above aorc's +0.000448) and its bootstrap CI **excludes zero POSITIVELY** [+0.00028, +0.00063], and the inverse-MSE rule gave it a substantial **0.139 weight** — so the effect is REAL, merely **~2× too small on delta and 23 points short on breadth**. ⚠️ Temporal halves decay sharply (**+0.00173 early → +0.00014 late**), so extrapolation to the later test window is worse still. | **+0.000456** | **62.1%** |
|    | > ⇒ **Width is closed.** Combined with hidden-128 (−0.000432 / 27.5%), the architecture-size axis is now measured in both directions from the shipped 256: **narrower is clearly worse, wider is real but far too small and too narrow to ship.** This was the last of the "final four" member families to report. | | | | |
---

## 5. The query itself

1. Copy `score_noq_test.py` → `score_noq_test_v2.py`. **The v1 script is never
   re-run.** Update only the stream list/seeds for candidates that passed §2.
2. Run `--mode equal` FIRST as the reproduction control; it must reproduce the
   known equal-weight value on the same frame. A drift here halts the query.
3. Present the evidence to the user; **wait for an explicit go**.
4. Run `--mode frozen` exactly once.
5. Run `verify_noq_v2.py` — all six checks: weights sum to 1 and are
   non-degenerate; truth bit-identical across streams; **zero (basin,date) key
   overlap between the TRAIN weight-fitting frame and the TEST frame**;
   independent sum-form NSE recomputation; bootstrap CI over basins; count of
   basins beating 0.8294.

## 6. Reporting rules

- Report the held-out number with its CI, the scored window (**Maurer ends 2008
  ⇒ ~13.2 years, not 15**), and the basin count.
- State the number of candidates screened (§4) alongside the result.
- Never present a train-side figure and a held-out figure as comparable.
- If the result lands below 0.840, report it as measured. The pre-registration
  exists precisely so that outcome is publishable rather than reroll-able.

---

### 33 `[v4]` — T1: THE EVENT-CONDITIONAL GLOBAL TRANSFORM — **FAILS, and the event arm is rejected BY FIT ITSELF**

| | |
|---|---|
| mode | post-hoc output transform, 1–3 GLOBAL parameters (shared by all basins) |
| rule | `p' = mean_p + k·(p − mean_p)`, then `×m` where `p > Q_τ(fit-window p)` |
| grid (fixed in v4.3 before running) | k ∈ 1.00…1.06 step 0.01, τ ∈ {q90,q95,q99}, m ∈ 1.00…1.30 step 0.05 |
| selection | on FIT only, applied frozen to VAL |
| decision statistic | h==1 (a rule FIT to a sample ⇒ v3.2 rule branch) |
| anchor | **0.950458 reproduced exactly** (diff −0.000000) |
| **VERDICT** | **FAIL — reported, not banked** |

**⭐ THE INFORMATIVE PART IS NOT THE VERDICT — IT IS *HOW* THE EVENT ARM DIED.**

Every one of the top-8 FIT cells selected **m = 1.00**, at every τ:

```
k=1.01 q99 m=1.00   FIT 0.941896      <-- best; m=1.00
k=1.01 q95 m=1.00   FIT 0.941896          identical: the multiplier is inert
k=1.01 q90 m=1.00   FIT 0.941896
k=1.03 q90 m=1.00   FIT 0.941710
```

SELECTION GUARD: the event arm beat k-only on FIT by **+0.000000** (needed
≥ +0.0005) ⇒ **k-ONLY SELECTED**.

⇒ The event-conditional idea did **not** fail a transfer test. **FIT rejected it
outright**: on the fitting window itself, the best available event multiplier is
"no multiplier". Any m > 1.00 applied to a prediction-side tail makes FIT worse
at every threshold tested. This is a *stronger* negative than a failed transfer,
and it is the third independent line of evidence converging on the same place:

1. per-basin peak multiplier, fit on first half → **−0.0089** out of sample;
2. per-basin timing/shift oracle → **+0.000000** (526/531 pick shift 0);
3. **now: a GLOBAL event multiplier is rejected by its own fitting window.**

⭐ **Reading**: the event-day deficit is real (both limbs under-predict on event
days) but it is **NOT a scalable magnitude offset**. Multiplying the tail up
adds more error on the event days the model got right than it recovers on the
ones it got wrong — i.e. the deficit is **scatter, not bias**, exactly as the
76%-correlation decomposition says. **No monotone function of the prediction
can fix a correlation deficit.** The lever class is closed at the global scale
as well as the per-basin scale.

### ⚠️ CONTROL (a) FAILED — and it exposed a look-ahead in `variance_shrinkage.py`

The k-only reproduction control did not reproduce the overnight **+0.000641**;
this script measures **+0.001050** at the same k=1.02. Per v4.3 the result is
**not banked** on a failed control, and it is not. The cause is a genuine
methodological difference, diagnosed rather than assumed:

| | `variance_shrinkage.py` (overnight) | `event_transform.py` (this run) |
|---|---|---|
| centering constant | `p.mean()` of **the VAL window being scored** | **FIT-window** mean |
| basin set | every basin with ≥20 VAL rows (**531**) | ≥20 rows in **both** FIT and VAL (**527**) |

Centering on the window being scored is a **mild look-ahead**: the constant is
fit to the data it is applied to. The FIT-window mean is the deployable choice
and is what this script uses. ⚠️ **Neither number is banked** — but the honest
deployable k-only figure is the one measured with FIT-window centering.

⚠️⚠️ **The k lever's own reported value is therefore NOT +0.000641 as recorded
in the 08-11 handoff.** It must be restated from a deployable computation before
being quoted anywhere.

**Also recorded**: even at its best, k-only reads paired median **+0.000596**,
breadth **68.5%**, CI [+0.000435, +0.000698], both halves positive
(early +0.000963, late +0.000937) — **positive and real, but ~1/7 of the +0.004
gap and far below the 85% breadth gate.** A dispersion correction is a genuine
small effect, not a lever that closes the gap.


---

#### 33b `[v4]` — CORRECTION to entry 33's control-(a) attribution (measured, `diag_k.py`)

Entry 33 attributed the k-only discrepancy to the centering constant alone.
**That was half right.** The full 2x2 was then measured (anchor 0.950458 exact):

| basin set | centering | k=1.02 | k=1.03 | k=1.05 |
|---|---|---|---|---|
| all 531 | VAL-window (as run overnight) | **+0.000641** | +0.000993 | +0.000820 |
| 527 (fit ∩ val) | VAL-window | +0.000932 | +0.001170 | +0.000754 |
| 527 (fit ∩ val) | **FIT-window (deployable)** | **+0.001050** | +0.001150 | +0.001135 |

**BOTH factors contribute**, and they are separable:
- **basin set** moves k=1.02 by **+0.000291** (531 → 527). The 4 basins that
  have VAL rows but no usable FIT window are dropping the measured gain.
- **centering** moves it a further **+0.000118** (VAL-mean → FIT-mean).

⭐⭐ **THE COUNTERINTUITIVE PART, AND THE REASON THIS IS RECORDED:**
**the DEPLOYABLE computation scores HIGHER than the look-ahead one**
(+0.001050 vs +0.000932 on the same 527 basins). Centering on the FIT-window
mean beats centering on the window being scored.

That is not a paradox once stated correctly: this transform is **not fitting**
the centering constant to minimise VAL error — it is *dispersing around* it.
Around the VAL-window mean, the residual is already mean-centred, so inflating
spread adds variance symmetrically. Around the FIT-window mean, the offset
between the two means carries information about the basin's systematic level,
and inflating around it partially corrects that offset as well. **The
look-ahead was not buying an advantage; it was destroying one.**

⇒ **The overnight +0.000641 UNDERSTATED the lever.** The honest deployable
figure is **+0.001050 at k=1.02** and **+0.001150 at k=1.03** on 527 basins.

⚠️ **STILL NOT BANKED, and the reason matters.** k=1.03 is the better VAL cell
but **FIT selects k=1.01** (FIT 0.941896 vs 0.941710), which delivers only
**+0.000380** on VAL. The k-vs-VAL curve is nearly flat from 1.02 to 1.05
(+0.00105…+0.00114) while the FIT curve peaks at 1.01 — **so FIT is a poor
selector of k on this surface, and honest selection loses most of the available
gain.** Reporting +0.00105 as "the lever" would be selecting k on the scored
window: exactly the scoring-sample trap, in its subtlest form yet.

**Recorded honest value of the dispersion lever, selected on FIT:
paired median +0.000596, breadth 68.5%, CI [+0.000435, +0.000698], both
temporal halves positive.** Real, deployable, ~1/7 of the +0.004 gap, and far
below the 85% breadth gate.

⚠️ **Any future quotation of the dispersion result must use the FIT-selected
number, not the best VAL cell** — and `variance_shrinkage.py`'s VAL-window
centering should be treated as superseded by `diag_k.py`.


---

### 34 `[v4]` — **multidrop s111 (the CLEAN fresh run): member FAILS, but D1 RE-ARMS on the pre-registered rule**

Anchor **0.950458 reproduced exactly** (diff −0.000000) on both scoring passes.
Dumps: 531 basins each, 2,570,572 test rows / 2,861,029 train rows, `gzip -t` OK.
Training: 30/30 epochs, **monotone**, final train loss 0.00704, **val NSE 0.90463**.

#### (a) MEMBER GATE — h=1, add-mode — **FAIL on every criterion**

| criterion | bar | s111 |
|---|---|---|
| paired median delta | ≥ +0.001 | **+0.000097** |
| breadth | ≥ 85% | **53.9%** (286/531) |
| bootstrap 95% CI | excludes 0 | **[−0.000008, +0.000195]** — straddles |
| temporal halves | same sign | **both negative** (early −0.000234, late −0.000899) |
| diff of medians | — | **−0.000981** (the REPORTED metric FALLS) |

⚠️ **Paired delta and diff-of-medians disagree in SIGN again** (+0.000097 vs
−0.000981) — the 2nd recorded instance ([[paired-median-vs-difference-of-medians]]).
Gate on the paired value, but note the reported metric moves DOWN.

#### (b) ALL LEADS — the v3.2 MEMBER decision statistic, 2 seeds

```
seed s111    ALL: -0.000159  breadth  35.6%   |  h=1: +0.000097  breadth  53.9%
seed s222    ALL: -0.000609  breadth   4.9%   |  h=1: -0.000383  breadth  26.4%
GATE 4 (>=2 seeds, no sign flip) on ALL LEADS: consistent sign (NEGATIVE)
```

⇒ **As a MEMBER, multidrop is dead**: negative on the decision statistic for
BOTH seeds, consistently signed, and nowhere near the breadth gate.

#### (c) ⭐ THE MECHANISM CLAIM IS SETTLED — and it was recorded BACKWARDS

s111 ran the **same unclipped config** as s222 (the YAMLs differ only in
`seed`/`experiment_name`; neither carries `clip_gradient_norm`) and passed
**straight through epoch 17**, where s222 blew up, descending monotonically to
epoch 30. ⇒ **The s222 divergence was the RESUME restarting the LR schedule,
NOT channel dropout, and NOT the missing gradient clipping.**
[[multidrop-diverged-no-grad-clipping]] attributes it to missing clipping —
that attribution is now **FALSIFIED**. Add to the causal-claims tally.

s111 also beats s222 on every axis (+0.000097/53.9% vs −0.000383/26.4% at h=1;
−0.000159/35.6% vs −0.000609/4.9% all-leads) — the fresh run is genuinely
better, just not good enough.

#### (d) ⚠️ D1 RE-ARMS — the rule fires on its own terms

PREREG v3.3 fixed D1's condition **in advance** as a **GPU-SPENDING** gate, not
a member gate, verbatim:

> ⚠️ **Pre-registered CONDITION for spending the GPU:** D1 runs ONLY if the
> confounded multidrop gates … do NOT land in the **decisively-dead regime** —
> operationalised as **all-leads breadth >= 15%** (the multi4/multih128
> rejections sit at 3.0-4.1%; below 15% the family is dead and no clipped
> retest is warranted).

**s111 measures 35.6%, vs the 15% bar and the 3.0–4.1% dead regime. The
condition is MET. D1 MAY RUN.** No argument after the fact — the same
discipline that made s222's 4.9% stand down.

⚠️⚠️ **BUT THE PREMISE D1 WAS BUILT ON IS NOW FALSE.** D1 exists to retest
channel dropout **with `clip_gradient_norm: 1.0`**, on the stated reasoning
that "multidrop s222 diverged at epoch 17 **because of it** [the missing
clipping], so the family has never had a fair test." §(c) shows the family
**HAS now had a fair test** — s111 is exactly that fair test, clean and
unclipped — and it FAILED as a member on both samples and both seeds.

⇒ **Adding clipping would not address anything the measurement identified.**
The condition is met; the *motivation* has evaporated. Recorded as such:
**D1 is ELIGIBLE but NOT RECOMMENDED**, and the decision to spend ~10 GPU-h on
it is deferred to the user rather than taken silently in either direction —
firing it automatically would honour the letter of a rule whose stated purpose
no longer applies, and cancelling it silently would be gate-shopping.


---

### 35 `[v4]` — **M1 `multi671`: TRAINING ON 671 BASINS INSTEAD OF 531 — PROBE GATE FAILS. The last untested lever is closed.**

The campaign's only never-tested surface: **more TRAINING BASINS**. 140 extra
CAMELS basins (+26%), identical recipe, identical window, identical
architecture; scoring frame byte-identical (same 531 basins, 69,030 val rows).
Anchor **0.950458 reproduced exactly** on every pass. Join clean: 531 → 531
basins, 196,636 → 196,636 rows (no basin loss).

#### The numbers

| arm | paired delta | breadth | CI | diff of medians |
|---|---|---|---|---|
| **swap** for `lstm_multi` | **+0.000062** | 51.2% (272/531) | [−0.000168, +0.000239] straddles | **−0.000912** |
| **add** as 10th stream | **+0.000079** | 52.0% (276/531) | [−0.000051, +0.000194] straddles | **−0.000400** |
| **ALL LEADS (the v3.2 MEMBER decision statistic)** | **−0.000414** | **22.4%** | — | — |

Candidate weight when swapped in: 0.1110 (vs `lstm_multi`'s 0.1823 — the
671-trained stream is trusted *less* by the inverse-MSE rule than the stream it
replaced).

#### Verdict against the PRE-REGISTERED probe bars (v4.3)

Probe bars were **≥ +0.0003 all-leads AND ≥ 45% breadth AND no sign flip**:
- all-leads delta **−0.000414** ⇒ **fails** (wrong sign entirely)
- all-leads breadth **22.4%** ⇒ **fails** (bar 45%)
- **SIGN FLIPS** between h=1 (+0.000079) and all-leads (−0.000414) ⇒ **fails**

⇒ **M1-PROBE FAILS on all three. The cascade (M2, M3) does NOT run.**
~60–100 GPU-h correctly not spent. M1 also fails the full §2 member gate by a
wide margin (needs +0.001 and 85%).

#### ⭐ Against the control — this is a SEED EFFECT, not a basin-count effect

Established marginal-seed value: **−0.00010 median / 56.3% breadth**. multi671's
h=1 deltas (+0.000062, +0.000079) sit **within noise of adding one more seed of
the same information**, and its breadth (51–52%) is *below* the marginal seed's
56.3%. ⇒ **140 additional basins bought nothing distinguishable from a seed.**

#### ⚠️ THE IN-RUN VALIDATION NUMBER WAS MISLEADING — 5th instance of the trap

multi671's in-run **val NSE 0.91887** vs multidrop's **0.90463** on the same 531
validation basins looked like a real gain. It is not one the ensemble can use:
the member is better *solo* and worth *less* in combination (weight 0.111 <
0.182). This is the **5th** demonstration that member skill and ensemble value
are close to unrelated ([[what-predicts-member-skill]]; RMSE-loss remains the
extreme case — best solo member ever, most ensemble harm). ⭐ **Never quote an
in-run validation metric as evidence for a member.**

#### ⭐⭐⭐ WHAT THIS CLOSES, AND WHY IT IS THE IMPORTANT RESULT

The 08-11 audit established that **exactly two levers had never been tested**:
more training BASINS and more training YEARS. Years is blocked by data (CAMELS
discharge starts 1980; the test window is untouchable). **Basins is now
measured, and it is negative.**

⇒ **The no-q surface is exhausted at the 531-basin scale.** Every axis — members
(14 now), combination (84 rules), per-basin anything, seed depth, width,
diversity training, sub-daily, timing, post-hoc correction, observation
correction, and now training-set SIZE — has been measured and closed.

⭐ **The deeper reading, consistent with the 76%-correlation decomposition:**
adding 26% more basins is *more of the same kind of information*. The residual
is not a shortage of rainfall-runoff examples — it is per-event, per-basin
scatter that more examples of other basins cannot inform. That is why breadth
(22.4%) is *worse* than the member's own solo skill would suggest: the extra
basins teach the shared weights slightly different compromises, which helps the
member's average and hurts its complementarity with the other eight streams.

**0.8363 stands.** The honest close of the campaign.


---

# PRE-REGISTRATION v3 ADDENDUM — the all-leads gating protocol

**Written 2026-08-10 ~20:00 MDT, BEFORE any candidate was scored under it.**
This is the firewall condition: entry 24 established that the binding breadth
gate is measured at 1/14 of available precision, and a protocol change made
AFTER seeing which candidates it would revive is gate-shopping across 25 ledger
entries. Everything in this section is fixed before the first v3 number exists.

## v3.1 What does NOT change

- **The reported metric remains h==1** (`phase_c.py:31`), for Li/Song
  comparability. The held-out record **0.8363** and the train-side baseline
  **0.950458** are unchanged and are not restated on any other sample.
- **Production weights remain fit at h==1**, theta=4.0 lambda=0.25, ONE global
  inverse-MSE vector. Ledger 25 measured refitting on all leads at
  **-0.003763** with scoring rows held fixed; that arm is CLOSED.
- **All six gates of section 2 remain in force**, unchanged, along with the
  swap-vs-add rule.
- **The section 3 query rule is untouched**: >= 0.954458 on val AND a rotated
  sub-slice confirmation AND an explicit user go.

## v3.2 What changes: the SAMPLE the gate statistics are computed on

Entry 24 proved breadth is sample-size limited (multi5: 76.5-81.2% on any
single phase, 94.7% on all dates; random 1/14 subsamples of the all-dates frame
reproduce ~79%). The gate therefore rejects on sampling noise. v3 computes the
gate statistics on all 14 leads (~1,820 rows/basin instead of ~130).

⚠️ **The sample rule is PER CANDIDATE CLASS.** Conflating these is exactly the
error entry 25 documents (the C->D confusion):

| candidate class | DECISION statistic | also REPORTED |
|---|---|---|
| **MEMBER** (a new/modified stream) | paired delta + breadth on **ALL 14 LEADS** | h=1 |
| **WEIGHTING RULE** (a change to how streams are combined) | **h==1** — the reported sample | all-leads |

Rationale for the split: a member candidate is a fixed artifact, and scoring it
on 14x the rows is a pure precision gain on the same question. A weighting rule
is FIT to a sample, so scoring it on a sample other than the reported one
measures the wrong quantity — that is precisely how "+0.0035" appeared in
entry 25 and evaporated when scoring rows were held fixed.

**The baseline is always the FROZEN 9-STREAM production ensemble** with frozen
weights, never a subset (entry 24 method lesson: the baseline you score against
matters more than the sample you score on — multi5swa read +0.00207/84.7%
against 4 streams and -0.00037/30.7% against the frozen 9).

**Validity anchor, mandatory before any v3 number counts:** the harness must
reproduce 9-stream h==1 = **0.950458** and the recorded h=1 breadth column to
the decimal (aorc 51.4%, multi5b 56.7%, multih128 27.5%, rmse 39.9%,
mse 33.1%). A drift halts the run.

## v3.3 The pre-registered candidate list (fixed; no additions after scoring)

**W1 — per-phase weight averaging.** Fit the production inverse-MSE rule
separately on each of the 14 phases, average the 14 weight VECTORS, score at
h==1. Motivation: every combination test in the closed set (entries 1-21) was
run on a single phase, so a phase-robust weighting is the one combination idea
not yet tested. Prior stated in advance: **weakly negative**, because weights
fit on the scored sample are self-consistent for it.
DECISION: all six section-2 gates, decision statistic at h==1.

**W2 — phase-balanced subsample fit.** Fit on a subsample drawn evenly across
phases (same row count as the h=1 fit, so fitting-set SIZE is held constant and
only its phase COMPOSITION varies), score at h==1.
⚠️ **Conditional-skip rule, pre-registered:** if W1 fails AND its failure
carries the dHBV-weight-mass signature of entry 25 (weight shifting toward the
dHBV streams because their h=1 is their worst phase), W2 is recorded as
**closed-by-mechanism and NOT run** — it is nearly the same experiment and the
val slice is a reused surface.

**D1 — channel dropout WITH clip_gradient_norm: 1.0 (conditional GPU arm).**
The existing multidrop configs lack the clipping that multi5 carries, and
multidrop s222 diverged at epoch 17 because of it, so the family has never had
a fair test. Add-mode, 2 seeds.
⚠️ **Pre-registered CONDITION for spending the GPU:** D1 runs ONLY if the
confounded multidrop gates (s111 fresh run, s222 epoch-16) do NOT land in the
decisively-dead regime — operationalised as **all-leads breadth >= 15%** (the
multi4/multih128 rejections sit at 3.0-4.1%; below 15% the family is dead and
no clipped retest is warranted). User-approved 2026-08-10.

**No other candidates may be added to v3 after any v3 number is observed.**

## v3.4 Ledger entries under v3

Every v3 result is appended to the section-4 ledger with its sample AND its
baseline stated explicitly in the row, pass or fail. Rows scored under v3 are
marked `[v3]` so no future reader can confuse a v3 breadth number with a v2 one.

---

# PRE-REGISTRATION v4 ADDENDUM — the training-corpus arm (531 → 671 basins)

**Written 2026-08-11 ~01:40 MDT, BEFORE any v4 candidate was built, trained or
scored.** Same firewall discipline as v3: everything here is fixed before the
first v4 number exists, and no rule below may be revised after seeing a result.
A rule that turns out to be wrong is recorded as a deviation, not edited.

## v4.0 Why a v4 exists at all

The v2/v3 ledger closed every axis it opened: 13 members, 84 combination rules
across six families, per-basin anything (oracle +0.0070, deployable ≤0.4% of
it), seed depth, diversity training, sub-daily intensity, timing (per-basin
oracle exactly 0.000000), post-hoc bias/scale, and observation correction. The
no-q error decomposition (bias² 2.18% / variability 21.35% / **correlation
76.46%**) says the residual is event-magnitude scatter — on time, too small —
and that no reweighting or post-hoc rescale can touch the correlation term.

A full audit of the campaign record finds **exactly two levers never tested on
this pipeline**:

1. **more TRAINING BASINS than 531**, and
2. **more TRAINING YEARS than 1980-10-01..1995-09-30**.

(2) is blocked by data: CAMELS discharge on disk begins 1980 and the test window
1995-2010 is untouchable. (1) is not blocked — `gpu1080/corpora671/` already
holds **671-basin corpora in six forcing variants**, built and unused by the
LSTM streams.

**The in-ensemble precedent is decisive**: the three shipped δHBV streams were
already trained on 671 basins while the six LSTM streams train on 531. The
shipped ensemble is therefore *already* a mixed-training-corpus artifact. v4
removes that asymmetry in the direction of more data.

⚠️ **This is a TRAINING-side change only.** It is the one class of change that
adds genuinely new information (140 additional basins' rainfall-runoff
mappings) rather than re-fitting existing information, which is what the
correlation term requires.

## v4.1 What does NOT change — the firewall against the scoring-sample trap

The scoring-sample trap fired **four times in one night** (+0.0035, +0.0044,
+0.0019, and "score only good gauges"). v4's whole risk profile is that it
changes training while leaving evaluation identical, so this is stated
explicitly and is checkable:

- **The scored basins remain the SAME 531.** The 140 extra basins are training
  data only. They are never scored, never in the val frame, never in the
  median.
- **The scored rows remain byte-identical**: val slice 1990-10-01..1995-09-30,
  h==1 for the reported metric, 69,030 rows, 531 basins.
- **The weighting rule is untouched**: ONE global inverse-MSE vector, θ=4.0,
  λ=0.25, fit on the 1980-90 fit slice.
- **The reported metric remains h==1** for Li/Song comparability. The held-out
  record **0.8363** and the train-side baseline **0.950458** are unchanged.
- **All six gates of §2 remain in force**, plus the v3.2 sample rule.
- **The §3 query rule is untouched**: ≥ 0.954458 on val, AND rotated sub-slice
  confirmation (fit 1980-88 + 1993-95, validate 1988-93), AND an explicit user
  go before the query is spent.
- **Validity anchor, mandatory**: any v4 scoring run must reproduce 9-stream
  h==1 = **0.950458**. A drift halts the run and nothing else is reported.

⇒ Because a 671-trained member changes only the weights inside one stream, it
enters the ledger through the ordinary member path. **It is not a protocol
change and it does not alter what is measured.**

## v4.2 The disclosure obligation (fixed in advance)

If a 671-trained stream ships, **every reported number must disclose that the
LSTM streams were trained on CAMELS-671 while Li/Song's reference LSTM trains
on 531.** The comparison to Li/Song 0.8294 is then explicitly a
different-training-corpus comparison, disclosed in the same sentence as the
number. User decision 2026-08-11: the as-practiced protocol permits this (the
δHBV precedent), and disclosure — not suppression — is the remedy.

Pre-registered wording obligation: no headline may read "same protocol as
Li/Song" once a 671-trained stream is in the stack.

## v4.3 The pre-registered candidate list (fixed; no additions after any v4 number is observed)

**M1 — `multi671`: the lstm_multi recipe trained on 671 basins.**
Identical 15-channel multi corpus recipe, identical architecture, identical
train window (1980-10-01..1995-09-30), identical seed protocol — the *only*
change is that the training basin list is 671 instead of 531. Scored on the
same 531.
- **Primary mode: SWAP** for `lstm_multi` (weight 0.1823). Swap, not add, so
  the result is never confounded with "one more member" (§2 swap-vs-add rule).
- **Secondary, reported but not decisive: ADD** as a 10th stream.
- **DECISION STATISTIC (v3.2 member rule): paired delta + breadth on ALL 14
  LEADS.** h==1 also reported. The LSTM dumps carry all 14 leads natively, so
  this costs nothing extra.
- Prior stated in advance: **weakly positive but below the member gate.** The
  only basin-count datum ever measured is the bagging bound — dropping 20% of
  basins costs ~0.0008 — so +26% basins plausibly buys ~+0.001 on a single
  stream, at or just under the +0.001 bar. **The honest expectation is that M1
  passes the probe gate below and fails the full §2 member gate**, and that the
  cascade (M2/M3) is where a shippable total would come from, if anywhere.

**M1-PROBE — the go/no-go for spending cascade GPU (this is a resource gate, NOT a ship gate).**
One seed (s111) decides whether the 671 arm continues. Pre-registered bars, all
on the ALL-LEADS decision statistic, swap mode, vs the frozen 9:
- **paired delta ≥ +0.0003**, and
- **breadth ≥ 45%**, and
- **sign does not flip between h==1 and all-leads.**

Rationale for bars this low: this gate authorises *further measurement*, not
shipping. The dead regime is well characterised — multi4 3.0%, multih128 4.1%
all-leads breadth — and the best member ever measured (multih512) sits at
42.6%. A candidate below 45% breadth with a delta under +0.0003 is
indistinguishable from the marginal-seed control (−0.0001) and the cascade is
not worth ~60-100 GPU-h.
⚠️ **A passed probe gate is NOT a ship.** M1 ships only on all six §2 gates.

**M1-CONTROL — mandatory, pre-registered, run in the SAME scoring pass.**
The known-value control for M1 is the **marginal-seed arm**: adding/swapping a
further 531-trained `lstm_multi` seed, whose established value is **−0.0001
median, 56.3% breadth** (seed depth is saturated). M1's delta is interpretable
only as the amount by which it exceeds this control. If the control drifts from
its established value, the baseline is wrong and **nothing else in the run is
reported.**

**M2 — the 671 single-forcing streams** (`daymet671`, `nldas671`,
`maurer671`), each a SWAP for its 531 counterpart. Combined stream weight
≈0.232. **Conditional: runs only if M1-PROBE passes.** Same decision statistic,
same §2 gates, same control.

**M3 — the 671 corpus-extension members** (`multi5_671`, `multi6_671`),
requiring the station corpus and the Livneh soil-moisture extraction to be
extended from 531 to 671 basins. **Conditional: runs only if M2 produces at
least one stream passing its own §2 gate.**
⚠️ Pre-registered data-integrity condition: the station corpus and Livneh
extractions for the extra 140 basins must pass the same Gate-0 checks the
original builds passed (0% NaN in dynamic inputs, correct date range, areal
extraction via GAGES-II polygons). A basin failing Gate 0 is DROPPED from the
training list, not zero-filled — a zero reads as "no rain".

**T1 — the event-conditional global transform (post-hoc, CPU-only).**
Motivated by the one post-hoc result that ever transferred honestly: the global
dispersion factor k (one parameter, 531 basins) read **+0.000641 on VAL** with
k=1.02 chosen on FIT, against a measured under-dispersion (ensemble
sd_pred/sd_obs median 0.9445, 82.5% of basins < 1; single-member multi5
0.9562/75.1%, so averaging is not the sole cause). T1 asks whether making the
factor *event-conditional* — the regime where the documented deficit lives
(both limbs under-predict on event days) — does better than the linear factor.

    p' = mean_p + k·(p − mean_p),  then  ×m  where  p > Q_τ(fit-window p)

The τ-quantile is computed **per basin on FIT-WINDOW PREDICTIONS ONLY** — a
prediction-side threshold, so the rule is deployable and uses no observation
at decision time. Grid, fixed in advance: k ∈ {1.00…1.06 step 0.01},
τ ∈ {q90, q95, q99}, m ∈ {1.00…1.30 step 0.05}. **Selected on FIT, applied
frozen to VAL.**
- **DECISION STATISTIC: h==1** — T1 is a rule FIT to a sample, so by the v3.2
  split it is decided on the reported sample. All-leads also reported.
- **Selection guard**: the event arm is chosen over the k-only arm only if it
  beats the best k-only arm **on FIT** by ≥ +0.0005. Otherwise k-only is
  reported as the result and the event arm is recorded as not selected.
- **Two mandatory controls**, both of which must behave or the result is
  declared an artifact and discarded:
  (a) **k-only reproduction** — the k-only arm must reproduce **+0.000641** at
      k=1.02 on VAL;
  (b) **permutation control** — apply m to a RANDOM day-mask of the same
      per-basin cardinality; must come out ≈0 or negative.
- **Bank criteria (all required)**: VAL paired delta ≥ +0.001, paired bootstrap
  95% CI excludes zero, both temporal halves agree in sign, anchor reproduces.
- Prior stated in advance: **+0.0005 to +0.002, i.e. probably short of the
  +0.001 bank bar on its own.** Reported either way.

⚠️ **T1 and any accepted member are NOT independent.** If a member is accepted,
the ensemble composition changes, so **T1 must be refit on FIT and re-reported
after every accepted swap.** A T1 gain measured on one composition may not be
added to a member gain measured on another.

**No other candidates may be added to v4 after any v4 number is observed.**

## v4.4 Closed by re-reading the existing record (no new measurement)

Recorded here so the ledger stays complete and the item is not re-opened:

**eps=0.5 NSE-loss member under the production inverse-MSE rule.** The audit
flagged this as a gap on the grounds that eps05 was closed under EQUAL
weighting, before the inverse-MSE rule shipped. **It is not a gap**:
`rescore9.py` already carries `eps05` in its CANDIDATES map and scored it
against the FULL frozen 9 under the production rule, with the anchor
reproducing 0.950458 exactly. Result: **h=1 −0.000052 (breadth 47.6%),
ALL-LEADS −0.000575 (breadth 17.9%)** — negative on both samples.
**CLOSED. No GPU, no further test.**

## v4.5 Ledger entries under v4

Every v4 result is appended to the §4 ledger, pass or fail, with its sample AND
its baseline stated in the row, marked `[v4]`. A candidate not in §v4.3 was not
screened; a candidate that was screened cannot be unscreened.

**GPU-spend authorisation on file (user, 2026-08-11):** the full gated cascade
M1→M2→M3 is pre-authorised, each step behind its own gate. **The held-out query
is NOT pre-authorised and never becomes automatic** — §3's explicit-go rule
stands unchanged.

## v4.6 RECORDED DEVIATION — M1-CONTROL cannot be re-measured, and why

**Logged 2026-08-11 ~02:20, BEFORE any multi671 number exists.** Per the §v4.0
firewall rule ("a rule that turns out to be wrong is recorded as a deviation,
not edited"), this is a deviation notice, not a revision.

**What v4.3 specified:** M1-CONTROL = the marginal-seed arm, run in the SAME
scoring pass, i.e. add/swap a further **531-trained `lstm_multi` seed** and
confirm it reproduces its established value (−0.0001 median, 56.3% breadth).

**Why it cannot be run as written.** The production `lstm_multi` stream already
consumes **s111, s222, s3334, s555, s666**. The only `lstm_multi` seeds left on
disk are **s333 and s444**, and `gate_eval.py:83` excludes both by name:

```python
#   s444  val NSE 0.7646  (weak)
# Decided on train-side evidence only -- excluding a seed for a poor TEST score
# would be test-set selection.
_MULTI_WEAK_SEEDS = ("s333", "s444")
```

⇒ Re-measuring the control with either spare seed would measure **"add a
KNOWN-WEAK seed"**, not "add an average marginal seed". That is a biased
control **in the direction that flatters the candidate**: a weak-seed arm would
read more negative than −0.0001, making any multi671 delta look larger by
comparison. Running it would be worse than not running it.

**The deviation.** M1-CONTROL is evaluated against the **established published
value** rather than a fresh measurement:

> marginal seed under the production weighted rule = **median −0.00010,
> breadth 56.3%**, negative for **7/9 streams** ([[seed-depth-SATURATED-both-tracks]])

**The decision rule is unchanged in substance**: multi671's delta counts as a
*basin-count* effect only if it **clearly exceeds** that marginal-seed value. A
multi671 result in the −0.0001 neighbourhood is a seed effect, not 140 basins.

⚠️ **Consequence for interpretation, fixed in advance:** because the control is
a citation rather than a same-pass measurement, the usual "if the control
drifts, the baseline is wrong" check is **not available for M1**. The validity
anchor (frozen 9 @ h==1 = 0.950458) therefore carries the whole burden of
detecting a broken baseline in this run, and it is mandatory.

**A clean control is available later at a cost**: training a 6th 531-trained
`lstm_multi` seed (~15-25 GPU-h) would give a same-pass marginal-seed arm. It
is NOT run now — the probe exists to decide whether the 671 arm deserves that
GPU at all. If M1-PROBE passes, the 6th-seed control SHOULD be run alongside
the M1 seed-2/seed-3 confirmations, because at that point the delta is being
taken seriously enough to deserve an unbiased control.


# PRE-REGISTRATION v5 ADDENDUM — the sequence-length arm (seq 365 → 730)

Written 2026-08-11 (evening session), BEFORE any v5 config was generated or any
v5 number observed. Anchor reproduced this session before registration:
9-stream inv-MSE 0.950458 (exact).

## v5.0 Decisions taken by the user this session (recorded verbatim)

1. **Protocol: matched-protocol only.** The years-extension arm (train
   1950–1995 via Livneh metvars + pre-1980 NWIS discharge) was verified
   feasible (data exists; Livneh altprecip/unsplit tree on PSL THREDDS;
   531-basin NWIS census run 2026-08-11: 247 basins have daily q by 1950,
   441 by 1970, 45 add zero) but can only be published as a disclosed
   protocol-extension row, never as the benchmark number, because the Li/Song
   lineage retrains all baselines on identical data. The user declined the
   two-row framing ⇒ the years arm is DEAD as a benchmark lever and is NOT run.
2. **D1 IS CANCELLED** (user decision). D1 was ELIGIBLE by the letter of
   v3.3 (s111 all-leads breadth 35.6% ≥ 15%) but its stated premise — "s222
   diverged from missing clipping, so the channel-dropout family never had a
   fair test" — is FALSIFIED: s111 was the fair unclipped test (30 clean
   epochs, full schedule) and the member failed both decision stats
   (−0.000159 / 35.6% and −0.000609 / 4.9%). Cancelling a premise-dead arm on
   user authority, recorded here rather than silently, is the honest exit.
   This is a DEVIATION in the v4.6 style: the letter of the rule is not being
   honoured, and the reason is recorded before any further GPU is spent.
3. **GPU budget for the v5 round: approved up to ~60 GPU-h; plan uses ≤ ~25.**

## v5.1 Candidates screened WITHOUT training this round (gate-shopping ledger)

Screened by an 8-agent verification panel (workflow wf_3a1d3ec9-a36, reports
archived session-side), each with the decisive reason:

| candidate | verdict | decisive evidence |
|---|---|---|
| years-extension (1950–95 Livneh/NWIS) | DEAD (user protocol decision) | benchmark train window is de facto part of the Li/Song protocol; adversary panel estimated P(+0.001 gate) 10–20%, P(+0.004 query bar) ≈ 0 |
| multi-task aux-state targets (UA-SWE / Livneh VIC states) | EVIDENCE-CLOSED, not trained | Lees et al. 2022 HESS (LSTM cell states already encode SWE/SM unsupervised, R²>0.8); Ouyang et al. 2025 WRR (591 CAMELS basins, flow+ET MTL: no streamflow gain); this ledger's objective-change class is 0-for-6 (RMSE, MSE, eps=0.5, CMAL, GMM, Tier-3) |
| Livneh-as-member (1980–2010 met forcing) | remains SCREENED (2026-08-04) | same COOP base ≈ maurer; nothing new rebuts criterion (d) |
| ESA CCI passive soil moisture (aux or input) | SCREENED | SMMR→SSM/I sensor cascade inside the window; the exact NCA-LDAS/SMERGE failure class |

## v5.2 THE ARM: multi730 — sequence length 730, the last never-varied axis

Audit basis: every config on disk trains at seq_length 365; of the three
"never varied" axes named in the 0.84-push handoff (hidden size, sequence
length, input dropout), hidden size was closed both directions (h128/h512)
and input dropout was closed (multidrop); sequence length alone was never
run at any other value. It is matched-protocol-legal: a pure hyperparameter,
no new data, no new inputs.

**Honest prior, stated before running:** adjacent to the closed
"more-of-the-same-information" class; the expected pattern is multih512's
(real-but-too-small, possible h=1/all-leads sign flip). P(probe pass) is
estimated 10–20%. The arm is worth running because it is the LAST untested
matched-protocol axis: either it ships, or the matched-protocol surface is
fully measured — both outcomes are wanted.

### v5.2.1 The member (fixed before generation)

- `cfgls_multi730_s111.yml` = byte-copy of `cfgls_multi_s111.yml` changing
  ONLY: `experiment_name: rw2ls_multi730_lstm_mm_s111`, `seq_length: 730`.
  Same corpus (nh_data_multi), same (absent) clipping, same LR schedule, same
  30 epochs, same everything else. Seed 111 first.
- Known, accepted, conservative confound: a 730-day lookback forfeits train
  targets before ~1982-01 (~8% of samples; corpus starts 1980-01-01). If 730
  wins despite fewer targets the signal is real; this is disclosed, not
  corrected for.
- Box: the 1080 (idle; the 4050 is in personal use). ~35–40 min/epoch
  expected ⇒ ~18–20 h.

### v5.2.2 The PROBE gate (s111 only; bars fixed now, mirroring ledger-35)

Score BOTH arms under the production rule (ONE global inverse-MSE vector,
θ=4.0, λ=0.25, weights fit 1980-10-01..1990-09-30, scored on the 1990-95 val
slice), with the anchor reproduced in the same pass:

- P-ADD: multi730 s111 as 10th stream.
- P-SWAP: multi730 s111 replacing lstm_multi.

PASS requires, on the ALL-LEADS decision sample: paired delta ≥ +0.0003 AND
all-dates breadth ≥ 45% AND no sign flip between h=1 and all-leads, AND the
delta must exceed the marginal-seed control band (−0.00010 / 56.3% breadth,
measured 2026-08-09). In-run validation NSE is NOT evidence (it has lied 5×).

FAIL ⇒ STOP. No s222, no cascade. The closure note is written
("sequence length CLOSED — the matched-protocol no-q surface is now fully
measured") and the arm is filed with the other 35 ledger entries.

### v5.2.3 The CASCADE (only if the probe passes)

- s222, identical config but seed. Ship gate on the 2-seed average as a
  stream: all-leads delta ≥ +0.001 AND breadth ≥ 60% AND both seeds
  individually positive AND rotated-subslice confirmation (fit 1980-88 +
  1993-95, validate 1988-93) before any query discussion.
- Even a shipped +0.001 does NOT clear the +0.0040 query bar. Any v5 result
  banks train-side. THE HELD-OUT QUERY IS NOT SPENT WITHOUT EXPLICIT USER
  APPROVAL, and nothing in v5 licenses it.

### v5.2.4 Mechanics guards (from the ops ledger, binding)

assert_no_nan on the corpus before training; flock on the launcher; no
`pkill -f`/`pgrep -f` (resolve PIDs via /proc exe); dump the BEST epoch, not
blindly the final; `check_split.py` before the dump; TRAIN dump with
--dump-all-leads; assert 531 basins IN the dump; `gzip -t` before use; the
gate scorer reads only gpu1080/dumps.


## v5.3 PRE-SCORING CONTROL — measured 2026-08-11 18:22, BEFORE any multi730 number exists

Logged in the v4.6 style: a control measured and interpreted in advance, so it
cannot be recruited afterwards to explain whichever way the arm lands.

**`perseed_allleads.py multi5` — re-adding an ALREADY-INCLUDED known-good
member to the frozen 9:**

| seed | ALL-leads delta | breadth |
|---|---|---|
| s111 | −0.000458 | 23.4% |
| s222 | −0.000402 | 29.8% |
| s333 | −0.000207 | 37.1% |
| s444 | −0.000340 | 29.9% |
| s555 | −0.000431 | 29.0% |
| **seed-avg** | **−0.000385** | **33.0%** |

multi5 is the campaign's best-evidenced member (+0.0023 ensemble, 93.6%
breadth, CI excludes zero). Re-added on top of a frozen ensemble that ALREADY
CONTAINS IT, it reads **−0.000385 / 33.0%** — because the inverse-MSE rule
correctly declines a duplicate stream. Anchor exact (0.950458, diff −0.000000).

### What this fixes in advance about reading multi730

1. **The bars DO NOT MOVE.** ≥ +0.0003 all-leads AND ≥ 45% breadth AND no sign
   flip, exactly as registered in v5.2.2. This entry exists to constrain
   INTERPRETATION, not to loosen a threshold after seeing a control.
2. **The SWAP arm is the primary read for this arm specifically.** multi730 is a
   near-duplicate of `lstm_multi` (same corpus, same 15 channels, same recipe;
   only the lookback differs). The ADD arm therefore inherits the duplicate
   handicap this control quantifies (~−0.0004 / ~33% for a genuinely good
   duplicate). Registered NOW: if ADD fails while SWAP passes all three bars,
   the arm counts as PASS and the cascade is authorised. If SWAP fails, the arm
   FAILS regardless of ADD.
3. **A ~−0.0004 / ~33% multi730 reading is the DUPLICATE signature, not
   evidence about sequence length.** It is what a good-but-redundant stream
   looks like here. Only a reading that clears the marginal-seed control
   (−0.00010 / 56.3%) distinguishes a sequence-length effect from redundancy.
4. This is the 6th instance of member-skill ≠ ensemble-value and the cleanest:
   the SAME member is worth +0.0023 when added once and −0.000385 when added
   twice. Cite it wherever the trap is discussed.


---

### 36 `[v5]` — **multi730: SEQUENCE LENGTH 730 vs 365 — PROBE GATE FAILS. The last matched-protocol axis is closed.**

The only never-varied matched-protocol hyperparameter (width and input dropout
from the same "never varied" note were tested and closed in v3/v4; sequence
length was not). Config = 2-line diff from `cfgls_multi_s111.yml`
(`experiment_name`, `seq_length: 730`). Trained 2026-08-11 17:37 → 2026-08-12
07:58 on the 1080, 30 epochs, **monotone throughout** (ep1 0.02846 → ep30
0.00677, no divergence), 28:41/epoch, 10,139 it/epoch.

#### The numbers

| arm | paired delta | breadth | CI | diff of medians |
|---|---|---|---|---|
| **swap** for `lstm_multi` | +0.000056 | 51.4% (273/531) | [−0.000104,+0.000211] straddles | **−0.001365** |
| **add** as 10th stream | +0.000054 | 52.9% (281/531) | [−0.000029,+0.000162] straddles | −0.000535 |
| **ALL LEADS (v3.2 decision statistic)** | **−0.000357** | **25.0%** | — | — |

h=1: +0.000054 / 52.9%. Candidate weight 0.1187 (swap) / 0.0993 (add) — the
inverse-MSE rule trusts it **less** than the 0.1823 `lstm_multi` it replaces.

#### Verdict against the PRE-REGISTERED v5.2.2 probe bars

Bars were **≥ +0.0003 all-leads AND ≥ 45% breadth AND no sign flip**:
- all-leads delta **−0.000357** ⇒ **FAILS** (wrong sign)
- all-leads breadth **25.0%** ⇒ **FAILS** (bar 45%)
- **SIGN FLIPS** h=1 (+0.000054) → all-leads (−0.000357) ⇒ **FAILS**

⇒ **V5-PROBE FAILS on all three. s222 and the cascade do NOT run.** Also fails
the §2 ship gate by a wide margin. Per v5.3 the SWAP arm was pre-registered as
primary for this member; **SWAP fails too**, so the v5.3 ADD-handicap clause
never engages.

#### Against the controls — indistinguishable from redundancy

| comparison | delta / breadth |
|---|---|
| **multi730 (this arm)** | **−0.000357 / 25.0%** |
| duplicate signature (v5.3: multi5 re-added to an ensemble containing it) | −0.000385 / 33.0% |
| marginal seed (established) | −0.00010 / 56.3% |

multi730 sits **on the duplicate signature** and **below the marginal seed on
both axes**. ⇒ Doubling the lookback bought nothing distinguishable from adding
a redundant stream — while costing a year of training targets.

#### ⚠️ The join guard fired, and it was RIGHT

First pass aborted: `join lost >5% of rows` (196,636 → 184,281, 6.3%).
**Diagnosed as the pre-registered confound, not a bug**: the 730-day lookback
needs an extra year of warmup, so the TRAIN dump starts **1981-12-23 vs multi's
1980-12-24** (360 vs 386 stride-14 windows = 6.7%, matching the row loss).
Re-run under `ALLOW_SHORT_JOIN=1`, the escape built for AORC's genuinely
shorter record. ⭐ **The loss falls entirely in the FIT window — val rows stay
at the full 69,030 and basins at 531** — so the scoring sample is untouched and
the comparison is fair. Anchor 0.950458 reproduced exactly on every pass
(`perseed_allleads` diff −0.000000).

⭐ **Note the direction of the confound: it is CONSERVATIVE.** multi730 fits its
weights on 115,251 rows vs the baseline's 127,606 and still had to clear the
same bars. A win would have been real; a loss is not attributable to it.

#### ⇒ What this closes

**Sequence length was the last untested matched-protocol axis. The
matched-protocol no-q surface is now FULLY measured**: members (15),
combination (84 rules), per-basin anything, seed depth, width, input dropout,
diversity training, sub-daily, timing, post-hoc correction, observation
correction, training-set SIZE, and now **sequence length**. Under the user's
matched-protocol-only decision (v5.0), no untested axis remains.

**0.8363 stands.**


---

# PRE-REGISTRATION v6 ADDENDUM — the TENDENCY-CHANNEL arm (multi8)

Written 2026-08-12 BEFORE any corpus was built or any v6 number observed.
Anchor reproduced 0.950458 exactly on every screening pass this session.

## v6.0 How this candidate was found — a NEW screening method

Every previous member was proposed from physics and then trained. v6's candidate
was **priced first**, with zero GPU, by a screen introduced this session:

    median over basins of |partial corr( channel , |ensemble residual| | precip )|
    reported on the NEAR-MEDIAN cohort (the ~78 basins that set the metric),
    against TWO anchors measured on the identical frame:
        A_sm_l3 = 0.2839  (shipped as multi6, worth +0.000997 median)
        A_noise = 0.0531  (worth 0 by construction)

13 channels were screened. **IVT scored 0.0944 (1.39x noise) and was rejected
without extraction.** Forcing disagreement (1.09x), station snow depth (1.12x),
srad (1.34x) and trange (1.49x) also died at the floor.

## v6.1 THE CANDIDATE: `multi8` = multi6 + soil-moisture TENDENCY channels

**Channels added** (all derived from the EXISTING nh_data_multi6 corpus — zero
new data acquisition): `dsm_l1`, `dsm_l2`, `dsm_l3`, each = the one-day
difference of the corresponding `sm_l*`, **LAGGED ONE DAY** so only information
available at prediction time is used.

**Why it is not redundant with multi6** (measured, [[dsm3-tendency-channel-SURVIVES-screen]]):
- lagged screen value **0.2632 (3.86x noise)**, at the sm_l3 anchor's level;
- lagging costs only 16% of the raw association ⇒ not a concurrent artifact;
- ⭐ **multi6's OWN residual is still 3.87x noise correlated with the tendency of
  its own input**, and removing multi6 from the ensemble moves the signal by
  +0.0041 (nothing) ⇒ the LSTM is not deriving the rate of change from the level.

**Honest prior, fixed in advance:** the screen predicts INFORMATION CONTENT, not
ensemble value, via a monotone assumption calibrated on ONE channel. sm_l3 sits
at parity and bought only +0.000997. ⇒ **Expect ~+0.001, i.e. BORDERLINE at the
gate**, with a tail-weighted landing profile
([[where-channel-gains-LAND-the-aim-problem]]). Both decorrelation and
information content have failed to predict ensemble value before. P(pass) ~25-35%.

## v6.2 The PROBE gate (s111 only) — bars fixed now

Scored under the production rule (one global inverse-MSE vector, theta=4.0,
lambda=0.25, weights fit 1980-10-01..1990-09-30, scored on the 1990-95 val
slice), anchor reproduced in the same pass. Two arms:
  P-ADD  : multi8 as a 10th stream.
  P-SWAP : multi8 replacing multi6 (the member it extends — the primary read,
           since multi8 strictly contains multi6's channels).

**PASS requires, on the ALL-LEADS decision sample:**
  paired delta >= +0.0003  AND  all-dates breadth >= 45%  AND  no sign flip
  between h=1 and all-leads, AND the delta must exceed the marginal-seed
  control band (-0.00010 / 56.3%).

**Also reported (not gating, diagnostic — this is the first arm to have it):**
the NEAR-MEDIAN delta and breadth. Reference values: multi5 +0.001896 @ 70.3%
(shipped, moved the median +0.002); multi6 +0.000794 @ 57.7% (shipped,
+0.000997); multi730 +0.000214 and multi671 +0.000293 (both FAILED with
positive near-median deltas). ⇒ A near-median delta below ~+0.0008 predicts a
failed member even if the all-leads bars pass.

FAIL any bar ⇒ STOP. No s222. Write the closure note.

## v6.3 The CASCADE (only if the probe passes)

s222, identical config but seed. Ship gate on the 2-seed average as a stream:
all-leads delta >= +0.001 AND breadth >= 60% AND both seeds individually
positive AND rotated-subslice confirmation (fit 1980-88 + 1993-95, validate
1988-93). Even if it ships, +0.001 does NOT reach the +0.0040 query bar.
**THE HELD-OUT QUERY IS NOT SPENT WITHOUT EXPLICIT USER APPROVAL.**

## v6.4 Mechanics guards (binding)

assert_no_nan on the new corpus BEFORE training (a 0.3% NaN input once killed
95/531 basins); the first differenced row per basin is NaN by construction and
MUST be handled explicitly, not zero-filled; preflight_member.py against
cfgls_multi6 as the reference; flock the launcher; dump the BEST epoch;
check_split before the dump; --dump-all-leads; assert 531 basins IN the dump;
gzip -t; no pkill -f / pgrep -f (resolve via /proc exe).

## v6.5 RECORDED IN ADVANCE — the C_p7 / C_api30 question

`C_p7` (0.2724) and `C_api30` (0.2264) also cleared the screen but are
PRECIPITATION-DERIVED, i.e. plausibly already available to a model that ingests
365 days of precip. They are **deliberately NOT in multi8**: including them
would confound "the LSTM cannot difference soil moisture" with "the LSTM cannot
integrate precipitation", and the seq-730 arm already showed more precip context
buys nothing. If multi8 passes, they become a separate follow-up arm; if multi8
fails, they are closed with it.


---

# PRE-REGISTRATION v7 ADDENDUM — the SOIL-COLUMN STRUCTURE arm (multi9)

Written 2026-08-12 BEFORE the multi9 corpus was built and BEFORE any multi8
result was observed. Anchor 0.950458 reproduced on every screening pass.

## v7.0 Round-2 screen — what was priced, and what died

Screen: median over basins of |partial corr(channel, |ens residual| | precip)|
on the NEAR-MEDIAN cohort. Anchors on the identical frame: A_sm_l3 = 0.2839
(shipped as multi6, worth +0.000997), A_noise = 0.0644.

| channel | near \|pcorr\| | ×noise | verdict |
|---|---|---|---|
| A_sm_l3 (ANCHOR) | 0.2839 | 5.35 | — |
| **S_sm1_minus_sm3** (vertical wetting gradient) | **0.2639** | **4.97** | **BUILD** |
| R1_dsm3_lag1 (round-1 winner, in multi8) | 0.2632 | 4.96 | training |
| S_dsm1_lag1 | 0.1241 | 2.34 | marginal, excluded |
| O_stn_grid_absdiff | 0.0882 | 1.66 | dead |
| O_stn_minus_grid | 0.0834 | 1.57 | dead |
| X_prcp_max | 0.0778 | 1.46 | dead |
| X_prcp_range | 0.0710 | 1.34 | dead |
| S_dtmax_lag1 | 0.0686 | 1.29 | dead |
| A_noise (ANCHOR) | 0.0644 | 1.21 | — |
| **O_n_stations** | **0.0378** | **0.71** | **dead — BELOW noise** |
| **O_obs_quality** | **0.0314** | **0.59** | **dead — BELOW noise** |
| **O_min_dist** | **0.0307** | **0.58** | **dead — BELOW noise** |

⭐⭐ **THE OBSERVABILITY HYPOTHESIS IS FALSIFIED.** The idea was strong: all four
forcings interpolate ONE COOP/GHCN gauge base, so on a day with 1 gauge 40 km
away every product is guessing the same way, and per-day `n_stations` /
`min_dist_km` would tell the model when to distrust its own precipitation. All
three observability channels scored **BELOW the noise anchor**. Gauge density
carries no information about where the ensemble errs. This independently
vindicates the older "+2.7% on product spread" dismissal, and for a better
reason: that measured spread BETWEEN PRODUCTS; this measures the model's ERROR.
⇒ **Do not revisit observability metadata as a channel.**

Also dead: cross-product envelope/range (the "wettest product" and
disagreement-magnitude ideas), and temperature tendency.

## v7.1 THE CANDIDATE: `multi9` = multi6 + soil-column STRUCTURE

**Channels** (all derived from the EXISTING nh_data_multi6 corpus; zero new data):
- `grad_13` = `sm_l1 − sm_l3` — the vertical wetting gradient
- `grad_12` = `sm_l1 − sm_l2`, `grad_23` = `sm_l2 − sm_l3` — the same structure
  resolved per adjacent pair, so the model is not forced through one contrast

**Why it is not multi8 in disguise — MEASURED**: |corr(sm1−sm3, dsm3_lag1)| has
median **0.0617** (q75 0.0712) over 200 basins. **Near-orthogonal.** dsm3 is
*how fast* storage changes; grad is *where the water sits*. Both score ~4.96×
noise independently.

**Why the LSTM does not already have it**: multi6 ingests `sm_l1..3` as separate
z-scored levels, and their DIFFERENCE is exactly what the network is empirically
failing to form — the same failure multi8 tests for the time-derivative, here
for the depth-derivative. Physically it is runoff generation: wet-over-dry =
infiltration that never reaches the channel; saturated column = the next storm
runs off. That is the mechanism the campaign identified as the residual's home
([[where-the-remaining-error-lives]]) and the axis
[[livneh-soil-moisture-route]] measured as NOT reconstructible from precip
(sm_l3 vs API(30d) r = −0.001).

**These channels are NOT lagged** and require no lag: they are contemporaneous
STATE (like sm_l1..3 themselves, which multi6 already uses unlagged), not a
difference across time. No leakage question arises — but see v7.4.

## v7.2 The PROBE gate (s111 only) — bars fixed now, identical to v6.2

Production rule, anchor reproduced in the same pass. Arms:
  P-SWAP : multi9 replacing multi6 (**primary** — multi9 strictly contains it)
  P-ADD  : multi9 as a 10th stream

PASS = all-leads paired delta **≥ +0.0003** AND all-dates breadth **≥ 45%** AND
no h=1/all-leads sign flip AND clears the marginal-seed band (−0.00010/56.3%).
FAIL any ⇒ STOP, no s222.

Non-gating diagnostic: near-median delta/breadth. References: multi5 +0.001896 @
70.3% (shipped, +0.002 median); multi6 +0.000794 @ 57.7% (+0.000997); multi730
+0.000214 and multi671 +0.000293 (**both FAILED with POSITIVE near-median
deltas**). Below ~+0.0008 predicts failure even if the all-leads bars pass.

## v7.3 Relationship to multi8 — fixed in advance to prevent post-hoc selection

multi8 (tendency) and multi9 (structure) are **independent arms of one
hypothesis**: *the LSTM does not form derivatives of its own state inputs* —
multi8 tests d/dt, multi9 tests d/dz. Registered NOW:
- Each is judged on its OWN pre-registered bars. Neither result may be used to
  reinterpret the other's bars.
- **If BOTH pass**, a combined member (multi6 + tendency + gradient) is a THIRD
  arm requiring its own pre-registration and its own seeds — it is NOT assumed
  additive, because [[error-structure-and-what-to-build]] measured that
  decorrelation is necessary but not sufficient and that members substitute
  rather than add.
- **If BOTH fail**, the "LSTM cannot form state derivatives" hypothesis is
  CLOSED in both directions and that is the finding.

## v7.4 ⚠️ The one honest caveat, recorded before the result

`sm_l1..3` are **VIC model output** driven by the same gauge precipitation the
LSTM already ingests. A gradient built from them is a function of a
land-surface model's internal state, not an observation. If multi9 gains, the
mechanism claim must be "the LSTM benefits from an explicit VIC-derived
structural feature", NOT "we added new physical information". This is the same
distillation caveat that applies to multi6 itself and it must not be overstated
in the paper.

## v7.5 Mechanics guards (binding)

assert_no_nan BEFORE training; the gradient is defined on every row (no leading
NaN, unlike v6's tendency) but this MUST be verified rather than assumed;
preflight_member.py against cfgls_multi6 as reference; flock; dump the BEST
epoch; check_split; --dump-all-leads; assert 531 basins IN the dump; gzip -t;
no pkill -f / pgrep -f. **multi9 QUEUES BEHIND multi8 — one GPU, no racing**
([[duplicate-supervisor-races-gpu]]).


---

### 37 `[v6+v7]` — **multi8 (d/dt) and multi9 (d/dz) BOTH FAIL. The "LSTM cannot form derivatives of its own state inputs" hypothesis is CLOSED in both directions.**

Two arms of ONE hypothesis, both pre-registered before their corpora existed
(v6.1, v7.1), trained in parallel (multi8 on the 1080, multi9 on the 4050),
anchor 0.950458 reproduced exactly on every pass, joins clean 531→531 basins
and 196,636→196,636 rows (no ALLOW_SHORT_JOIN needed for either).

#### The numbers

| arm | swap-vs-multi6 delta | add delta | **ALL-LEADS (decision)** | breadth |
|---|---|---|---|---|
| **multi8** (tendency, d/dt) | −0.000250 (46.9%) | +0.000031 (50.7%) | **−0.000323** | **26.4%** |
| **multi9** (structure, d/dz) | −0.000401 (41.6%) | −0.000092 (47.6%) | **−0.000399** | **24.9%** |

Bars were ≥ +0.0003 all-leads AND ≥ 45% breadth AND no sign flip.
- multi8: all-leads **−0.000323** (wrong sign), breadth **26.4%**, and it
  **SIGN-FLIPS** (h=1 +0.000031 → all-leads −0.000323) ⇒ **fails all three**.
- multi9: all-leads **−0.000399** (wrong sign), breadth **24.9%**, consistent
  negative sign, and its swap arm's CI **[−0.000600, −0.000213] EXCLUDES ZERO
  ON THE WRONG SIDE** ⇒ **fails all three, and is confidently negative.**

⇒ **Neither cascades. No s222 for either.** Candidate weights 0.0993 (multi8)
and 0.1189 (multi9) — the production rule trusts both LESS than the multi6
(0.1910) each was built to extend.

#### ⭐⭐⭐ What this closes — the hypothesis, in both directions

v7.3 registered in advance: multi8 tests whether the LSTM fails to form the
TIME derivative of its state inputs, multi9 the DEPTH derivative, and **if BOTH
fail the hypothesis is closed in both directions and that is the finding.**
Both failed. ⇒ **CLOSED.**

#### ⭐⭐⭐ THE METHODOLOGICAL RESULT — the screen does NOT predict ensemble value

Both channels were selected by the information-content screen at ~5× the noise
floor, at parity with the shipped `sm_l3` anchor (0.2839):

| channel | screen (×noise) | ensemble result |
|---|---|---|
| A_sm_l3 (shipped as multi6) | 5.35 | **+0.000997** |
| dsm3 → multi8 | 4.96 | **−0.000323** |
| grad_13 → multi9 | 4.97 | **−0.000399** |

⇒ **Three channels at statistically indistinguishable information content;
one is worth +0.001 and two are worth ≈−0.0004.** The screen is a valid
NECESSARY filter (it correctly killed IVT at 1.39×, observability BELOW noise,
forcing-disagreement at 1.09×) but it is **NOT SUFFICIENT** — it does not
predict ensemble value. This is the **third** statistic to fail that way, after
decorrelation ([[what-predicts-member-skill]]) and error-correlation
([[tier3-diversity-loss-BOUNDED-DEAD]], where correlation ANTI-predicted).

⭐ Record the honest version: **the screen's value is in cheap REJECTION, not
selection.** It saved the IVT extraction (~30 y × 3 variables) and four other
candidates for ~30 min of CPU each; it cost ~24 GPU-h to learn that clearing it
is not enough.

#### ⭐ The near-median diagnostic called it — first predictive use

Both members' near-median deltas were **+0.000023 (multi8) and +0.000360
(multi9)**, far below the ≳+0.0008 threshold v6.2 registered as predicting
failure, and nowhere near multi5's +0.001896 @ 70.3%. **The diagnostic
correctly forecast both failures from the same numbers that produced the
verdicts** — and it now has 6 members behind it (multi5 +0.001896 ✅, multi6
+0.000794 ✅, multi9 +0.000360 ❌, multi671 +0.000293 ❌, multi730 +0.000214 ❌,
multi8 +0.000023 ❌). ⇒ **A near-median delta below ~+0.0008 has never shipped.**

#### ⚠️ Note on multi8's temporal split

multi8's swap arm reads **+0.000367 early / −0.001515 late** — a large
reversal. Do NOT build a story on it: this is one seed, and
[[my-causal-claims-keep-failing]] records a prior instance where a
sub-window mechanism story was built on one seed and reversed by the second.

#### v6.5 / v7.3 consequences, applied as registered

- `C_p7` and `C_api30` (precipitation-derived, cleared the screen) were
  **closed with multi8** per v6.5 — they do not get their own arm.
- The combined tendency+gradient member (v7.3's third arm) is **not built**:
  it was conditional on BOTH passing. Both failed.

---

### 37b `[v6]` — **multi8 SECOND SEED confirms the closure: −0.000358 / 24.3%, consistent sign, seed-average WORSE than either seed.**

Ledger 37 closed multi8 on one seed. s222 was run to convert that into a
properly-evidenced closure (the pre-registered GATE 4 needs ≥2 seeds and no sign
flip). Trained 2026-08-12 23:53 → 2026-08-13 09:00 on the 1080, 30 epochs
monotone to 0.00737, anchor 0.950458 exact.

| seed | ALL-leads delta | breadth | h=1 |
|---|---|---|---|
| s111 | −0.000323 | 26.4% | +0.000031 |
| **s222** | **−0.000358** | **24.3%** | −0.000033 |
| **SEED-AVG (shippable unit)** | **−0.000456** | **21.7%** | −0.000011 |

⇒ **GATE 4 satisfied: 2 seeds, consistent NEGATIVE sign.** The member is closed
on the unit it would actually ship as.

⭐ **The seed-average is WORSE than either individual seed** (−0.000456 vs
−0.000323 / −0.000358). That is the signature of a member whose errors ADD
rather than diversify — the opposite of multi5, whose seed-average (+0.0023) beat
every individual seed. Worth contrasting the two directly in the paper.

⭐ **The s111 epoch-15 loss spike (0.01212 → 0.01769 → 0.01238) did NOT recur in
s222.** Seed-specific noise, not a property of the tendency channels — exactly
the question a second seed exists to answer. No mechanism story was built on it
(cf. [[my-causal-claims-keep-failing]], where a sub-window story from one seed
was reversed by the second).

---

### 37c `[v7]` — **multi9 SECOND SEED confirms: −0.000372 / 24.1%. BOTH arms of the state-derivative hypothesis are now closed on TWO CLEAN SEEDS each.**

| multi9 seed | ALL-leads | breadth | h=1 |
|---|---|---|---|
| s111 | −0.000399 | 24.9% | −0.000092 |
| **s222** | **−0.000372** | **24.1%** | +0.000011 |
| **SEED-AVG (shippable)** | **−0.000507** | **22.0%** | +0.000032 |

GATE 4 satisfied: 2 seeds, consistent NEGATIVE sign. Anchor 0.950458 exact.

#### ⭐ The seed-average-is-worse signature REPLICATES across both members

| member | s111 | s222 | **seed-avg** |
|---|---|---|---|
| multi8 | −0.000323 | −0.000358 | **−0.000456** |
| multi9 | −0.000399 | −0.000372 | **−0.000507** |

Both seed-averages are **worse than either constituent seed** — errors ADD
rather than diversify. The exact inverse of multi5, whose seed-average (+0.0023)
beat every individual seed. ⇒ **This is now a two-member replicated signature
distinguishing a useful perturbation from a harmful one**, and it is visible
BEFORE the ensemble gate.

#### ⚠️ s222's route to this number was not clean — disclosed

s222 showed a 2-epoch loss excursion after the LR drop (0.00976 → 0.01461 →
0.01714) and RECOVERED to finish best-at-epoch-30 (0.00714)
([[multi9-s222-loss-excursion-NOT-divergence]]). It was called a "divergence" at
epoch 22 on two ascending points — **wrong, and corrected by epoch 23**.
Its dump then had to be produced by hand because a launcher patch I applied
mid-run broke the eval block
([[patching-a-running-script-broke-the-dump]]). The DUMP ITSELF is clean:
epoch 030 (genuinely the best epoch), 531 basins, 2,861,029 / 2,570,572 rows —
byte-comparable to s111's, gzip-verified on both boxes.

#### ⇒ THE HYPOTHESIS IS CLOSED, PROPERLY EVIDENCED

v7.3 registered in advance: *if BOTH fail, "the LSTM cannot form derivatives of
its own state inputs" is closed in both directions and that is the finding.*
Both failed, each on two clean seeds, on every arm (swap, add, all-leads).
**CLOSED.** The combined tendency+gradient member is NOT built (it was
conditional on both passing), and v6.5 closes C_p7/C_api30 with them.

---

# ADDENDUM v8 — LEDGER 38: THE ARCHITECTURE AXIS (multi10, EA-LSTM)

**Registered 2026-08-13, BEFORE the config file or corpus reference exists.**
Both GPUs idle at time of writing (1080 free; 4050 GPU occupied by an unrelated
non-project process and treated as unavailable).

## v8.1 — WHY THIS ARM EXISTS

Measured, this session:

```
$ grep "^model:" gpu1080/cfgls_*.yml | sort | uniq -c
     49 model: cudalstm
```

**49 of 49 configs, 19 run families, one architecture.** Every member ever
trained in this campaign is `cudalstm`. The campaign has closed the input axis
three ways (more channels, better channels, derived channels), the combination
axis in both directions, and the training-lever axis (width, depth, dropout,
loss variants, diversity regularisation). It has never varied the model.

This was already the recommendation of our own measured error-structure analysis
(`optimizer_pass1.py`), which found every channel-add member sits at 0.90+ error
correlation "because it is the same architecture on nearly the same inputs" and
concluded: **"The multi7 template is a DIFFERENT MODEL FAMILY, not a different
dataset."** That recommendation was never executed; 14 further `cudalstm`
channel-add members were built instead, and all failed.

## v8.2 — THE CANDIDATE

`multi10` = the multi6 recipe with `model: ealstm`, `hidden_size: 312`.
**Identical corpus** (`nh_data_multi6`), identical 18 dynamic inputs, identical
27 static attributes, identical split/dates/loss/schedule/clip/seed. Zero new
data. multi6 is therefore an exact same-information control — any delta is
attributable to architecture alone, which is a cleaner control than any
channel-add arm ever had.

**hidden_size 312, not 256 — registered in advance with its reason.** Measured
on the box: EA-LSTM at h=256 has 218,625 params vs cudalstm 310,529 (a 30%
deficit; its input gate is driven by the 27 statics only). Ledger entry 6 (the
hidden-128 member) already measured what a capacity-reduced member on the same
corpus does: solo 0.7964 and 27.5% breadth, the worst in campaign history.
Running at 256 would confound architecture with capacity. h=312 gives 318,865
params (+2.7% parity); measured cost is identical (184 vs 180 ms/batch).
No `statics_embedding` (it would insert a nonlinear net between the statics and
the input gate, diluting the mechanism under test).

## v8.3 — MECHANISM (verified in source, not assumed)

`ealstm.py` is the only model in the zoo that does NOT concatenate statics onto
every timestep: `self.embedding_net(data, concatenate_output=False)`, then
`i = torch.sigmoid(self.input_gate(x_s))` — the input gate is computed ONCE per
basin from statics alone, and the dynamic gates never see statics.
`cudalstm.py` feeds dynamics+statics (45) into one fused `nn.LSTM`.
⇒ EA-LSTM imposes a hard factorization: basin identity multiplicative via one
static gate, weather additive. cudalstm CAN learn this and demonstrably does not
(49/49 members at 0.90+ error-corr).

## v8.4 — THE GATE (kill-fast, 1 seed s111). STOP unless ALL hold:

1. all-leads paired delta >= +0.0003   (DECISION statistic; h=1 never decides)
2. all-dates breadth >= 45%
3. no h=1 / all-leads sign flip
4. clears the marginal-seed control band (-0.00010 / 56.3%)
5. own competence: solo train-side median NSE >= 0.790
   (below = an h128 repeat; inverse-MSE correctly discards a weak member)
6. error-corr vs ensemble mean <= 0.88

**Bar 6 is READ FIRST.** Every channel-add lands at 0.90+. If EA-LSTM also lands
at 0.90+, the architecture-diversity hypothesis is FALSIFIED on one seed
regardless of the ensemble delta, and s222 is NOT funded.
⚠️ Bar 6 is necessary, NOT sufficient: decorrelation has failed as a POSITIVE
predictor three times (ledger: CONUS404, multi5, Tier-3). Low error-corr does
not mean it ships; high error-corr does mean it does not.

Primary read is SWAP vs multi6 (isolates architecture); ADD is secondary.
Also report the greedy curve with multi10 available: the curve peaks at 4
members (0.951293) and declines to 0.950458 at nine, so entering the top-4 is a
real result even if the 9-stream add is flat.

**NON-GATING DIAGNOSTIC — the campaign best predictor, 6-for-6 separation:**
near-median delta (n~78, +/-0.01 NSE of median). Shipped: multi5 +0.001896@70.3%,
multi6 +0.000794@57.7%. Failed: multi9 +0.000360, multi671 +0.000293,
multi730 +0.000214, multi8 +0.000023. **Below ~+0.0008 has never shipped.**

**IF kill-fast passes -> s222**, then the full ship gate (Section 2, verbatim):
>= +0.001, breadth >= 85%, paired bootstrap 95% CI excludes zero, >=2 seeds no
sign flip, control behaves, both temporal halves agree in sign.
PLUS: **seed-avg >= max(constituent seeds)** — multi8/multi9 both had seed-avg
WORSE than either seed (errors adding); multi5 had seed-avg BEATING every seed.
Violation = automatic FAIL.
PLUS: invmse candidate weight >= 0.15 (failures got 0.099-0.119; multi6 0.191).

## v8.5 — HONEST PRIOR, recorded before the result

**P(pass kill-fast) ~35%. P(ship) ~15%.**
Higher than the 0-for-14 base rate because this is the first arm to change the
factor our own analysis named as binding, with an exact same-corpus control.
Not higher because the greedy curve says the ensemble is saturated in
combination, so a 10th stream fights a headwind by construction.

Most likely failure (~40%): competent but insufficiently decorrelated —
error-corr 0.89-0.92, near-median delta +0.0003-0.0006, i.e. the "not-quite"
band that killed 6 arms. Second (~20%): optimization deficit under a schedule
tuned for cudalstm, solo NSE 0.78-0.79 (h=312 rules out the parameter half of
this, not the optimization half). Third (~5%): decorrelated but too weak, the
dHBV pattern, guarded by bar 5.

⛔ **No test query.** The no-q query is spent. Everything train-side.
A negative result closes the last open axis and converts "we only ever tried one
architecture" — an obvious reviewer objection to the ceiling paper — into a
measured statement. That is worth ~20 GPU-h.

| 32 `[l43]` | **THE STATIC ATTRIBUTE SET** — CAMELS-27 → CAMELS-27 + 8 screened GAGES-II attrs (`multi14`, 1 seed, swap vs `lstm_multi6`) | swap | **FAILED — 3 of 6 bars.** Anchor EXACT (0.950458, diff −0.000000). all-leads **+0.000213 @ 62.2%** (bar +0.0003) · h==1 **−0.000138 @ 47.1%** ⇒ **sign flip** · near-median **−0.000081** (h==1, n=78) / +0.000223 (all-leads, n=160) vs a +0.0008 bar ⇒ **fails on both frames** · solo 0.9355 (PASS) · beats the marginal-seed band (PASS). ⭐ Best all-leads swap of any failed candidate measured, and still a fail. Branch A fired: axis closed on one seed. ⚠️ Two leaks excluded pre-GPU (`RUNAVE7100`, `BFI_AVE` — both q-derived); `HGD_PCT` rejected at R²=0.9068. | **+0.000213** (all-leads) | **62.2%** |

---

## LEDGER 46 — WITHIN-BASIN SPATIAL HETEROGENEITY OF PRECIPITATION

**Pre-registered 2026-08-30, BEFORE the screen was read.** The chunk fetch was
still running when this was written; no channel number existed yet.

### The claim under test

Every forcing this campaign has ever used is a basin **areal mean**. The spatial
*distribution* of rain inside the basin has never been a channel. A storm
covering 20% of a catchment and uniform rain of the same areal mean produce
different peak runoff, so this is event-magnitude information the areal mean
cannot express — the shape of the measured residual (symmetric magnitude scatter
at high flow; not bias, not timing).

### Scope audit — done before any GPU, recorded because it is the whole premise

- `C_prcp_cv` (screened dead at 0.65x noise) is `np.nanstd` across the **three
  forcing products** (`channel_screen.py:62-64`, labelled "FORCING
  DISAGREEMENT"). It is *not* within-basin spatial CV. **Different quantity.**
- `better-inputs-not-more-inputs-CLOSED` scopes itself explicitly to
  re-combining the forcings on hand and states it "does not bound a genuinely
  NEW observation."
- The AORC/CONUS404 closures concern **point-vs-areal sampling** — how to
  estimate the mean — not within-basin variability as a signal.
⇒ The axis is untried.

### Channels

`spcv` (weighted spatial CV) · `wetf` (basin-area fraction over 1 mm) ·
`mxmn` (max cell / areal mean) · `cdsp` (rain-centroid displacement over basin
radius). All **dimensionless by construction**: the V4 areal mean is carried as
a diagnostic for the registration lag scan only and never enters a member
(`daymet-v4-upgrade-CLOSED`: V4 carries −0.0244; the one-day offset is +0.229
NSE and invisible to mean/bias checks).

### Controls, registered in advance

- `C_amean` — POSITIVE control. It is essentially precipitation, so
  `r2_extended` must come out near 1.0. If it does not, the screen is miswired
  and no number on the page is trustworthy.
- `C_wetday` — NEGATIVE control, the bare wet/dry indicator. On dry days the
  dispersion statistics are sentinels rather than measurements, so the all-rows
  channels are partly an indicator of wetness, and wetness is confounded with
  precip in a way a *linear* partial correlation does not remove.
- `C_*W` — wet-only twins (NaN on dry days). Their matched null is drawn on
  their own, fewer rows, which is exactly what the T2e fix provides.

### THE DECISION RULE (registered before the read)

1. **Screen gate.** A channel is alive only if it clears **>=3x its own matched
   null** AND **R^2_existing < 0.9**, AND its **wet-only twin agrees in
   direction**, AND it **clearly beats `C_wetday`**. A channel that passes only
   in its all-rows form is re-encoding wetness ⇒ DEAD.
2. **If `C_amean` does not read as highly redundant on `r2_extended`** ⇒ the
   screen is miswired; fix it before reading anything else.
3. **If no channel clears** ⇒ STOP. Zero GPU. Write the closure. **Explicitly
   do NOT fall back to "build it anyway as a perturbation member":** perturbation
   depth is already bounded at **+0.0016…+0.0023**
   (`compute-route-bounded-infinite-seeds-buy-0.0016`), below the +0.003 gate, so
   a member with no information content cannot clear the bar *by construction*.
   This is registered now precisely because it is the tempting move later.
4. **If a channel clears** ⇒ full 1980-2008 extraction, rebuild, 1 seed
   kill-fast, then 3v3 with fresh same-session controls.

### Ship gates (unchanged, from the ledger-46 design brief)

Window 1980-2008 homogeneous · no discharge · screen conjunction · **breadth
reported on every read (<0.5 is rank manipulation, not skill)** · **>= +0.003
held-out on h==1 AND all-leads with signs agreeing** (floor ±0.00103) · **h==1
must carry it** (all-leads-only gains do not transfer).

### Prior, stated honestly before the read

The screen predicts **information content, not ensemble value**: `multi8` passed
at the `sm_l3` anchor's level and shipped **−0.000456 at 21.7% breadth**. A
historical screen PASS bought **~+0.001** against a **+0.003** gate. And
`multi5`, the best member, works by **perturbation, not information**, which is
how 14 information-driven members failed. ⇒ **P(screen pass) ~30%, P(ship)
~8%.** The screen is a cheap KILL, not a promise.

### Verification already banked before the read

- Record md5 `c4d7639df8647308e48dedc9f45f824e`, median **0.8362893021622821**.
- Train-side anchor reproduced **0.950458, diff −0.000000**.
- No prior spatial member exists in `gpu1080/dumps/` or this file (checked the
  **artifacts**, not the notes).
- Polygon gates: log-area corr **1.000000**, 0 gross mismatches;
  log(ncell) vs log(area) **0.999029**; 531/531 basins, median **338** cells,
  only 1 basin under 10 cells.
- The dispersion accumulation was checked against an independent brute-force
  recomputation on 75 basin-days: **max |diff| 0.000e+00** on every statistic.

### ADDENDUM, same day, still BEFORE the screen was read

**The strongest prior threat to gate 1, named now so it cannot be rationalised
away later.** Daymet is an *interpolated* product. Its effective smoothing is a
function of station density, and station density rises over 1980-2008. Spatial
dispersion of an interpolated field is therefore partly a measure of **how many
stations the interpolator had**, not of how the storm was actually distributed.
More stations ⇒ more resolved structure ⇒ higher spatial CV, with no change in
the weather. This is the same failure shape as the CCI soil-moisture retraction,
where availability rather than signal drove the statistic.

Registered consequences:
1. A channel that clears the conjunction screen is **still not cleared for GPU**
   until the date-trend test has been run **over the full 1980-2008 window**
   against a matched null. The 1990-1995 screen window is too short to see a
   multi-decadal density trend, so a clean homogeneity read there is **not
   evidence of homogeneity** — it is merely the absence of a short-window trend.
2. That full-window test comes free with the Phase C extraction, so the ordering
   is: screen → (pass) → full extraction → **full-window homogeneity** → GPU.
   The homogeneity gate is not satisfied by anything available today.
3. If the trend is real but modest, the deseasonalised/detrended channel is the
   thing to screen, not the raw one — and it must re-clear the 3x bar on its own
   matched null, not inherit the raw channel's pass.

### ⛔ RESULT — CLOSED NEGATIVE, ZERO GPU (2026-08-30, same day)

Artifact `benchmarks/ledger46_spatial_screen.json` md5 `ecd809c7f9e9e2e01929d0df8b36a707`.
Full corpus: **531 basins, 1990-1995, 2,190 days**, anchor **0.950458** exact.
Cohort: near-band **78** basins (bar ≥10); straddle **265/265** above/below median.

| channel | × its own MATCHED NULL | R²_existing | R²_extended | n_near |
|---|---|---|---|---|
| `C_spcv` spatial CV | **0.95** | 0.029 | 0.039 | 78 |
| `C_wetf` wet-area fraction | **1.32** | 0.546 | 0.630 | 78 |
| `C_mxmn` max-cell/mean | **0.78** | 0.013 | 0.021 | 78 |
| `C_cdsp` centroid displacement | **0.93** | 0.049 | 0.060 | 78 |
| wet-only twins `C_*W` | **0.73 – 0.94** | 0.07–0.20 | 0.09–0.23 | 12 |
| `C_amean` *(positive control)* | 1.96 | 0.365 | **0.916** | 78 |
| `C_wetday` *(negative control)* | **1.70** | 0.501 | 0.556 | 78 |

**Bar is 3×. Nothing is close. The dispersion channels sit AT OR BELOW their own
matched null** — `C_spcv`'s near-|pcorr| is **0.0545 against the `A_noise`
anchor's 0.0674**, i.e. the spatial CV of the rain field is *less* associated
with the ensemble residual than a random normal series drawn on the same rows.

**Both controls did their job, which is why the read is trustworthy:**
- **Positive**: `C_amean` — essentially precipitation — comes out **R²_ext
  0.916**, cleanly reconstructible from precip + rolling sums, exactly as
  pre-registered. The instrument is wired correctly.
- **Negative**: `C_wetday`, the bare wet/dry indicator, reads **1.70×** —
  *higher than every dispersion channel*. Whatever faint association exists is
  **wetness, not heterogeneity**. And the wet-only twins (which remove the
  dry-day sentinels entirely) agree at 0.73–0.94×, so this is not a confound
  masking a real signal. There is no signal.

**Homogeneity** (not the binding constraint, reported anyway): date-trend
1.16–1.45× its matched null over the screen window. The §addendum's
station-density concern is **moot** — the channels died on information first.

### ⇒ WHAT THIS CLOSES, AND WHAT IT DOES NOT

**CLOSED: within-basin spatial heterogeneity of precipitation as an input
channel.** Eight forms of "the shape of the rain field" (4 statistics × all-rows
and wet-only) are all at the null. Per the pre-registered rule this does **NOT**
fall back to a perturbation member: perturbation depth is bounded
**+0.0016…+0.0023**, below the +0.003 gate, so a channel with no information
cannot clear the bar by construction.

**NOT closed — stated honestly**: precipitation structure organised along a
**physically meaningful axis**, specifically **elevation-band** decomposition
(orography → snow → melt timing). The four statistics tested are
orientation-agnostic; `cdsp` is directional but not elevation-aware. Needs a DEM
and a rebuild. ⚠️ Prior is **low** — eight orientation-free forms at the null is
strong evidence the rain field's *shape* is not the missing variable — but it is
a genuinely different hypothesis and was not measured here.

### ⭐ THE FINDING WORTH CARRYING

The heterogeneity is **real and large** (spatial CV median **0.158**, p90
**1.31** on wet days; centroid displacement to 0.6 basin radii) and it is
**uncorrelated with where the ensemble errs**. That is the *same wall* as
[[better-inputs-not-more-inputs-CLOSED]], now from a **fourth** independent
angle — forcing disagreement, per-basin member ordering, event-day residual
direction, and now within-basin spatial structure. Real physical variation that
carries **no per-event direction** about model error.

---

## LEDGER 47 — SUB-BASIN PRECIPITATION PHASE (the freezing line inside the basin)

**Pre-registered 2026-09-01, BEFORE any temperature data was extracted.**
Follows the ledger-46 closure, which explicitly left open "precipitation
structure along a **physically meaningful axis**" while killing the
orientation-agnostic amount-dispersion forms.

### The claim, and why it is not what ledger 46 killed

Ledger 46 killed the spatial distribution of precipitation **amount** (CV,
max/mean, wet fraction, centroid displacement — all orientation-agnostic, all at
their matched null). This is a different quantity: the spatial distribution of
precipitation **PHASE**.

A basin spanning 1,000–3,000 m has part of its area above the rain/snow
transition and part below on any given day. **The area fraction below freezing
is not derivable from basin-mean temperature** — it requires the basin's
hypsometry, and the member's static set contains **`elev_mean` and `slope_mean`
only** (verified: 27 attributes, no elevation range, std, or hypsometry). So the
LSTM cannot form this quantity from what it is given, however much capacity it
has.

Why it is the best remaining candidate rather than just the next one:
- **Snow is the only axis that has ever produced a member gain here.** `multi5`
  (snow perturbation) is the strongest member the campaign has, held-out
  **+0.0022** ([[why-multi5-works-PERTURBATION-not-information]]).
- Phase determines whether a storm runs off **today** or is stored for weeks.
  That is **event-magnitude**, which is the measured shape of the residual —
  not bias, not timing.
- Rain-on-snow and partial-area melt are exactly the events where a
  basin-mean-temperature model should mis-size a peak.

### Channels (from Daymet 1 km tmax/tmin, same verified pipeline)

- `C_frzf`  — basin-area fraction with tmean < 0 °C
- `C_snowf` — fraction of the day's **precipitation** falling on sub-freezing
  cells (phase-weighted, the physically active quantity)
- `C_tsd`   — within-basin spatial SD of tmean (a hypsometry/lapse-rate proxy;
  the campaign has no DEM, and this is the same information)
- `C_straddle` — `4·f·(1−f)` where `f = C_frzf`: peaks when the freezing line
  sits **inside** the basin, which is precisely the unrepresentable state

### Controls, registered in advance (ledger 46 showed these carry the read)

- **POSITIVE**: `C_tmean` — the basin-mean temperature. Must come out highly
  reconstructible (`R²_ext` near 1.0) since it is an existing input. If it does
  not, the instrument is miswired and nothing else on the page counts.
- **NEGATIVE**: `C_cold` — the bare indicator `tmean < 0`. It contains no
  sub-basin information. **Any phase channel that does not clearly beat
  `C_cold` is merely re-encoding "it is cold today"** and is DEAD. This is the
  control that did the real work in ledger 46, where `C_wetday` (1.70×) beat
  every dispersion channel and exposed the whole family.

### Decision rule (registered before the read)

1. Alive only if **≥3× its own matched null** AND **R²_existing < 0.9** AND it
   **clearly beats `C_cold`**.
2. `R²_extended` is read as a diagnostic: a channel the LSTM could rebuild from
   temperature + rolling sums is dead on arrival even if it clears the null.
3. **If nothing clears ⇒ STOP at zero GPU.** As in ledger 46, this does **not**
   fall back to "build it as a perturbation member": perturbation depth is
   bounded **+0.0016…+0.0023**, below the +0.003 gate.
4. Restrict the honest read to basins where the quantity can vary — a flat basin
   has no freezing line inside it. **Report the cold/high-relief cohort
   separately**, and treat a whole-corpus null with a live sub-cohort as a
   *cohort* result, stating the breadth cost explicitly.

### ⚠️ The prior, stated honestly before the read

**P(screen pass) ~25%, P(ship) ~7%.** Lower than it feels, because:
- The screen measures **information**, and information has repeatedly failed to
  predict ensemble value here (`multi8` cleared every screen, shipped
  **−0.000456** at 21.7% breadth).
- **`multi5` already perturbs the snow pathway** and is in the frozen nine, so
  part of this signal may already be captured
  ([[snotel-member-CLOSED-already-in-multi5]] is the cautionary precedent).
- The gate is **+0.003** and the best member ever built contributes **+0.0022**.

### Cost and the honest note on ordering

Daymet 1 km `tmax`+`tmin`, 1990–1995 = **672 chunks ≈ 2–3 h**, CPU/network,
**zero GPU**, abortable. ⚠️ I am starting the fetch **before** the cheap
prior-updating analysis finishes, because no cheap test of the *real* channel
exists without the data — the download **is** the screen. Recorded so the
ordering is not mistaken for having skipped a gate.

### ⛔ LEDGER 47 RESULT — CLOSED NEGATIVE, ZERO GPU, killed after 20 of 672 chunks

Record **UNCHANGED** at 0.8362893021622821. Anchor 0.950458 exact on every pass.

**It died on ERROR LEVERAGE, upstream of the conjunction screen.** A channel can
only help where the error is; every phase-relevant population carries squared
error in proportion to its size or less (leverage = SE share / row share, per
basin, median across basins, within **wet** days):

| population | leverage |
|---|---|
| `tmean < 0` (snow-dominated) | **0.37** |
| `tmax < 5` (cold season) | **0.52** |
| `tmin < 0 < tmax` (diurnal straddle) | **1.08** |
| `tmean ∈ [-3,3]` (phase genuinely open) | **1.12** |
| top-10% flow **[calibration]** | **7.03** |

Snow-dominated wet days are **23.3% of wet rows and carry 8.6% of squared
error** — the ensemble is *better* there, not worse. The sub-basin refinement
cannot matter when the whole cold-season population holds no disproportionate
error. Fetch cancelled at 20/672 chunks; ~2.5 h of network saved.

**⚠️ Two traps, both in my own analysis, both caught before they became a
result:**
1. *Pooled vs paired* — pooled, near-freezing error looked elevated (1.07);
   paired within basins it is **−0.1438 at breadth 0.436**.
2. *Unmatched control* — wet straddle days vs a control containing every **dry**
   day gave **+0.74 at breadth 0.908**, spectacular and meaningless. Matched to
   wet-only and then within precip decile: +0.38/+0.40, and the **relief
   gradient runs the wrong way** (slope corr **−0.357**), refuting the mechanism.

### ⭐ WHAT LEDGER 47 LEAVES BEHIND: GATE 0, the error-leverage screen

Zero cost, needs only the residual surface + forcings on disk, and sits
**upstream** of the conjunction screen (which requires the channel to exist).
Within top-30% flow days:

| population | leverage |
|---|---|
| big storm today (p90 precip) | **2.63** |
| high forcing disagreement | 1.96 → **1.07 matched on storm size** |
| wet antecedent · rising limb · rain-on-snow | 1.41 · 1.35 · 1.32 |
| cold · falling limb · low forcing disagreement | 0.37 · 0.24 · 0.22 |

**It validated itself by rediscovering a known closure unprompted:** forcing
disagreement surfaced at 1.96, fell to **1.07** once matched on storm size, and
the direction check reproduced `better-inputs-not-more-inputs-CLOSED` exactly —
corr(spread, **|error|**) **+0.2406** breadth 0.863, corr(spread, **signed
error**) **+0.0055** breadth **0.505**.

⚠️ Leverage is **necessary, not sufficient**: big storms read 2.63 and are
exactly where the residual is symmetric magnitude scatter with no predictable
direction. It says where the error *is*, never whether it is *reducible*.

### ⇒ STATE AFTER LEDGERS 46-47

A **systematic 12-population scan** finds no untapped high-leverage target. The
only high-leverage populations are big storms (2.63 — definitional) and
things that reduce to storm size. Combined with the ledger-45 result that every
combination class is bounded, the no-q **member** program is, on present
evidence, exhausted: the residual is symmetric magnitude scatter on large
storms, and no available input predicts its direction.

---

## LEDGER 48 — MRMS RADAR QPE (dynamical.org): NO ERROR DIRECTION. ~1 h CPU, zero GPU.

**The hypothesis was the best-motivated one in a while.** All three frozen
forcings interpolate ONE shared gauge base (~70%), which is the standing
explanation for the campaign's most repeated wall: forcing disagreement predicts
error **magnitude** but never **direction** (ledger 47: |error| r +0.2406 breadth
0.863 vs **signed** r +0.0055 breadth 0.505). MRMS is gauge-corrected **radar** —
a different instrument. If shared provenance were the cause, MRMS would show
direction the gauge products cannot.

**It does not.** 43 basins, 2018, 15,561 basin-days, wet days, per-basin r:

| difference channel | median r | CI95 | breadth |
|---|---|---|---|
| **MRMS − Daymet** (radar vs gauge) | **+0.0360** | **[−0.077, +0.106] spans zero** | 0.605 |
| NLDAS − Daymet **[CONTROL, gauge]** | −0.0448 | [−0.125, −0.002] | 0.349 |

27/43 positive, Wilcoxon **p=0.37**. **+0.036 is the same level as the already-
closed gauge products (+0.044/+0.013/+0.042).** On *magnitude* the control
actually beats it (0.184/0.884 vs 0.052/0.698).

⇒ **The wall is not shared provenance. Daily areal rainfall — from any
instrument — does not determine which way the model is wrong.**

**Radar-hostile failure, measured because the sample was built to test it:**

| cohort | corr(MRMS, Daymet) | annual ratio |
|---|---|---|
| radar-FRIENDLY (elev<500, snow<0.15) n=28 | **0.929** | **0.921** |
| radar-HOSTILE (elev>1500, snow>0.3) n=13 | **0.410** | **0.392** |

MRMS misses **~61% of precipitation** in mountainous/snowy basins — the pivotal
population. Even its good half is a **cohort**, and cohort-targeted members are
already closed.

**Mechanics banked** (a rerun costs an hour, not a day): Icechunk store
`s3://dynamical-noaa-mrms/...v0.3.0.icechunk`, us-west-2, anonymous, CC-BY-4.0;
0.01° grid, lat DESCENDING, chunks (648,100,100), cos-lat weights.
⚠️⚠️ **UNITS: `precipitation_*` is a RATE in kg m⁻² s⁻¹ (mm/s), not an hourly
accumulation** — summing raw values is 3600× too small and prints as "0 mm",
which looks like a failed extraction rather than a units error.
**Day boundary 12 h UTC, lag 0** (r 0.9248), resolved by lag scan.

**Scope**: nothing in the dynamical.org catalog reaches the no-q window (earliest
IMERG 1998, GEFS 2000, MRMS/HRRR 2014); this ran on the modern benchmark corpus.

## LEDGER 50 — PUSH THE NO-Q RECORD: A FUSED δHBV MEMBER, SEED DEPTH, AND THE NEVER-VARIED LSTM AXIS (opened 2026-09-01)

**Written BEFORE any ledger-50 number was computed.** Record 0.8362893021622821 (h==1, 531, 183,195 rows).
User decisions, 2026-09-01: (1) **any real gain ships** — the +0.003 "defensible" bar is replaced by the
ship bar below; (2) **matched Li/Song protocol only** (no pre-1980 years, no transductive use of test
forcings); (3) **`nakas-1080` only**.

### Facts established at zero GPU before this entry
- The δHBV h==1 transient is **in-sample only**. Held-out (ALLH test dumps, 1995-10-01→2008-12-07),
  dhbv_daymet 3-seed: h1 0.7615 … h14 0.7675 with no ramp; lstm_multi5 h1 0.8229 … h14 0.8287. δHBV−LSTM
  is flat at ≈ −0.06 on every lead. ⇒ reading δHBV at a later lead is dead; the val-slice ramp
  (0.918→0.961) was a training-window artifact.
- δHBV single-seed solo h==1 held-out: s111 0.7314, s222 0.7389, s333 0.7409 (sd ≈ 0.005; 3-seed avg 0.7615).
- δHBV training on the 1080: 5,642 s/epoch at the shipped recipe ⇒ ≈ 78 h per 50-epoch member.
- No fcorr and no fused-encoder δHBV has ever been trained on the Li/Song split.

### Ship bar (fixed now; applies to every arm; read on the BASE frame that reproduces the record)
A configuration ships iff ALL of: (a) paired h==1 median Δ > 0 with basin-bootstrap 95 % CI excluding zero
and |Δ| > 0.00103; (b) all-leads paired Δ same sign; (c) h==1 breadth ≥ 0.50; (d) weights = frozen rule
(inverse-MSE θ=4 λ=0.25, fit on TRAIN rows only); nothing tuned on test. Reads in fixed order: swap, then
add. FAIL ANY → STOP for that arm. One read per arm; no re-reads; anchor line logged with every read.

### G0 — δHBV DILUTION BOUND (zero GPU, existing dumps) — decides whether Stage 1 runs
Synthetic `dhbv' = dhbv + α(truth − dhbv)` on TRAIN and TEST, α set so the member's solo h==1 TEST median
rises by +0.03 and by +0.06; refit the frozen rule; read paired h==1 Δ + CI. Single member (dhbv_daymet)
and family (all three). Rule: **family bound at +0.06 below the ±0.00103 floor ⇒ Stage 1 SKIPPED**
(no δHBV upgrade can ship); single-member bound at +0.03 clears the ship bar ⇒ Stage 1 runs as written;
in between ⇒ run the probe, family extension only if the swap read clears the bar on its own.
Caveat recorded: a uniform pull preserves error correlation, so this bounds magnitude, not "differently good".

### Stage 1 — `dhbvM` probe: `--enc-vars camels3fv2 --forcing-correction` on corpora671/camels_corpus_fused_v2,
otherwise the shipped recipe (`--head dhbv --dhbv-loss mse --nmul 16 --hidden 256 --batch 256 --lr 1e-3
--epochs 50 --windows-per-station 1000`, train 1980-10-01→1995-09-30), seed 111.
Budget variant: if epoch-1 time × 50 > 96 h, rerun at `--windows-per-station 500 --epochs 40` and mark the
probe BUDGET-REDUCED (a fail there is NOT a closure of the axis; a pass is a pass).
- **B1**: `dhbvM s111` solo h==1 held-out median ≥ **0.7521** (= 0.7371 mean of the three shipped single
  seeds + 0.015, i.e. 3× the seed sd). Pass → seeds 222/333 → swap-for-dhbv_daymet then add-as-10th, ship bar.
  Fail → Stage 1 CLOSED, GPU to Stage 3 then Stage 2.
- Prior stated: the July fcorr gain was on pre-nmul16 δHBV; Li/Shen found per-forcing > fused for δHBV.

### Stage 2 — seed depth (filler; a law, no probe): multi6 s444,s555,s666 → multi5 s666,s777 →
lstm_multi s999,s1010 → lstm_nldas s444. TEST and TRAIN dumps from the SAME best epoch (assert).
Scored only as part of the final configuration under the ship bar.

### Stage 3 — `multiL` probe (if Stage 1 is skipped or fails): cfgls_multi_s111 + `clip_gradient_norm: 1.0`,
`epochs: 45`, LR {0:1e-3, 30:5e-4, 40:1e-4}. B1 analogue: solo h==1 ≥ mean(lstm_multi single-seed solos)
+ 3× their sd. **Filled in 2026-09-01 from the Stage-0 solo read (before any Stage-3 run):** lstm_multi
single seeds s111 0.8067, s222 0.8046, s3334 0.7992, s555 0.8037, s666 0.8058 (mean 0.8040, sd 0.0029) ⇒
**B1(multiL) = 0.8127**. δHBV bar confirmed with the same tool: mean 0.7371, sd 0.0050 ⇒ **B1(dhbvM) = 0.7522**.

### Final number: `score_noq_test50.py` (copy of the frozen scorer with the new stream list) →
`analysis/noq_test_result_l50.json`; the record artifact is never overwritten. Reported with its CI.

### ⭐⭐⭐ G0b RESULT + AMENDMENT 1 (2026-09-01 16:06, zero GPU) — `benchmarks/ledger50_G0b_perstream.json`
**Written before any ledger-50 arm produced an outcome.** G0 priced a lift on δHBV only. G0b prices the
SAME +0.03 solo lift on every stream family, so the arms can be compared instead of assumed:

| stream | ensemble weight | h==1 paired per +0.03 solo | CI95 | per unit weight |
|---|---|---|---|---|
| **lstm_multi5** | 0.2491 | **+0.010899** | [+0.00988, +0.01186] | 0.0438 |
| **lstm_multi6** | 0.1910 | **+0.008637** | [+0.00773, +0.00938] | 0.0452 |
| **lstm_multi** | 0.1819 | **+0.008287** | [+0.00760, +0.00907] | 0.0455 |
| lstm_daymet | 0.0737 | +0.002288 | [+0.00209, +0.00258] | 0.0311 |
| **dhbv_daymet** | 0.0615 | **+0.001855** | [+0.00171, +0.00202] | 0.0302 |

⇒ **The value of improving a member is ≈ proportional to its ensemble weight** (and the three
multi-forcing LSTMs convert ~1.45× better per unit weight than the single-forcing and δHBV streams).

⛔⛔ **THIS REFUTES G0's HEADLINE READING, WHICH WAS MINE AND WAS WRONG.** G0 measured only the δHBV family
and I concluded "the δHBV family is the one place a modest member gain is not diluted." The comparison shows
the opposite: δHBV is the **lowest-value** place to put a member gain, because it holds the least weight.
Another entry for `my-causal-claims-keep-failing` — a one-family measurement cannot license a claim about
where value is highest. **Only the comparison can.**

**Consequence for Stage 1, computed before acting:** B1 asks dhbvM for +0.015 solo. Through this map that is
worth ≈ **+0.0009 — below the ±0.00103 resolution floor.** The arm would need ≳ +0.017 solo *merely to be
resolvable*, at **78 h/seed × 3 seeds = 234 GPU-h**. The July fcorr evidence (+0.008…+0.023 solo, on the
weaker pre-nmul16 δHBV) tops out at ≈ +0.0014 ensemble.

**AMENDMENT 1 — Stage 1 is CANCELLED and replaced by Stage 3 as the main swing.** `dhbvM s111` was killed by
PID at 16:14 after 2 h 11 m (partial checkpoint deleted; nothing read from it, no outcome exists).
The GPU goes to the `lstm_multi`-family axis, which converts ~4.5× better per unit of solo gain and costs
~12–30 h/seed instead of 78 h. Stage 2 (seed depth) is unchanged and is now *quantitatively* justified
rather than filler — see AMENDMENT 2.

⚠️ Integrity note: G0/G0b fit and score on held-out rows and are **bounds**, in the same sense as ledger 45's
oracle rows — they price *where to spend GPU before a candidate exists*. They are NOT a selection among
trained candidates on test performance. The ship bar is unchanged, and every arm still gets exactly one read.
Caveat: a uniform pull toward truth is the most ensemble-friendly possible improvement, so the absolute
numbers are upper bounds; the *ordering across streams* is what this read is used for and is far more robust.

### AMENDMENT 2 — Stage 2 repriced (same map, so it is now a prediction, not a hope)
Seed depth raises a stream's solo along `solo(k) = a − b/k`. From the measured single-seed and multi-seed
solos (lstm_multi: single-seed mean 0.8040, 5-seed 0.8306 ⇒ b = 0.0333): one more seed on lstm_multi is
+0.0013 solo ⇒ **≈ +0.00036 ensemble**. multi6 is the shallowest high-weight stream (3 seeds, w 0.191), so
its 3→6 step is the largest single depth purchase available. **Pre-registered prediction, before the dumps
exist:** multi6 3→6 seeds is worth **+0.0010 … +0.0018** h==1 paired, and the full chain
(multi6 ×3, multi5 ×2, multi ×1) **+0.0015 … +0.0025** — consistent with ledger 44's independent
infinite-seed headroom of +0.0016…+0.0023. If the measured chain lands outside that band, the map is wrong
and every number above it must be re-examined.

### ⛔⛔ AMENDMENT 2 IS REFUTED BY DIRECT MEASUREMENT — THE DEPTH LANE IS CANCELLED (2026-09-01 17:2x)
`ledger50_depth.py` → `benchmarks/ledger50_depth.json`. Anchor exact. Per stream, the ensemble median is
recomputed with that stream rebuilt from k of its seeds (production weights refit on TRAIN each time),
then `ensemble(k) = a − b/k` is fitted:

| stream | seeds now | measured k=1 → k=max | **+3 seeds** | to infinity |
|---|---|---|---|---|
| lstm_multi6 | 3 | 0.835034 → 0.836289 | **+0.000268** | +0.000536 |
| lstm_multi5 | 5 | 0.835961 → 0.836289 | +0.000021 | +0.000055 |
| lstm_multi | 5 | 0.835484 → 0.836289 | +0.000081 | +0.000217 |

**AMENDMENT 2 predicted multi6 3→6 = +0.0010…+0.0018 and the chain +0.0015…+0.0025. Measured: +0.000268
and ≈ +0.00037 for the whole chain — 4–6× below the pre-registered band.** By the rule written with that
prediction, the map is wrong for this purpose and everything resting on it must be re-examined. Doing that:

⭐⭐ **THE ERROR, AND IT IS A REAL DISTINCTION: G0b's map prices a member becoming GENUINELY BETTER
(closer to truth). Seed averaging does not do that — it removes SEED NOISE, and the 31-net ensemble is
already a variance-reduction machine, so a stream's seed noise is largely redundant with the averaging the
ensemble performs anyway.** Two different currencies; I converted between them and was wrong. The G0b
ordering across streams (value ∝ weight) is *unaffected* — it compares like with like — but G0b may
**not** be used to price variance reduction. Consistent with ledger 44's independent infinite-seed
headroom (+0.0016 for the whole ensemble): the three streams here sum to +0.0008 of it.

⇒ **The depth lane is cancelled.** 6 seeds × ~13 h = 78 GPU-h would buy ≈ **+0.00032**, below the ±0.00103
floor — it cannot ship under the ship bar, whatever it measures. `multi6 s444` was killed at 7 of 30 epochs
and its run dir removed (no dump was produced, so no stream changed and the frozen frames are untouched).
The freed lane runs **multiL s222**; the arm now completes in ~2 days instead of ~3.

⚠️ Caveat recorded: the depth curves are noisy (subset spread 0.002–0.004 at k<k_max, and multi5 is
non-monotone in k), so the *fitted* 5→8 extrapolations are indicative. The decision does not rest on them:
`b` (the whole 1→∞ range) is ~0.0016 for multi6 and ~0.0011 for multi, so no seed purchase on these streams
can reach the floor even if the fit is off by 2×.

⇒ **What survives as a live lever: exactly one — a genuinely better member on a high-weight stream.**
That is `multiL` (3 seeds, matched-depth read against a 3-seed `lstm_multi`), and nothing else is queued.

### G0c — THE MAP IS CONVEX, SO THE multiL BAR IS SET NOW, BEFORE ITS DUMPS EXIST
`benchmarks/ledger50_G0c_linearity.json` (9 rows; the first attempt lost 3 rows to a tag collision at
2 dp and is kept as `*_INCOMPLETE_key_collision.json` — an artifact that silently drops rows still exits 0).

Ensemble h==1 paired Δ per unit of solo lift, measured at four lift sizes:

| solo lift | lstm_multi5 (w .249) | **lstm_multi (w .182)** | dhbv_daymet (w .062) |
|---|---|---|---|
| +0.005 | +0.001272 (0.254/unit) | **+0.000902 (0.180)** | +0.000235 (0.047) |
| +0.010 | +0.002727 (0.273) | **+0.001967 (0.197)** | +0.000496 (0.050) |
| +0.020 | +0.006274 (0.314) | **+0.004676 (0.234)** | +0.001103 (0.055) |
| +0.030 | +0.010899 (0.363) | **+0.008287 (0.276)** | +0.001855 (0.062) |

⭐ **The map is CONVEX: value per unit of member gain RISES with the size of the gain** (lstm_multi 0.180 →
0.276 per unit from +0.005 to +0.030). ⇒ **Small member improvements are disproportionately worthless to
this ensemble** — a structural fact that fits the campaign's history of 14 information members failing.
It also means the earlier linear extrapolation from the +0.03 anchor **overstated** small gains, so quoting
G0b linearly at small lifts is wrong in the optimistic direction. Use this table, not a proportion.

**⇒ PRE-REGISTERED BAR FOR multiL, fixed before any multiL dump exists:**
- multiL must beat `lstm_multi` by **≥ +0.0057 solo** (h==1 held-out, 3-seed stream vs a 3-seed subset of
  lstm_multi — matched depth) merely to reach the ±0.00103 ensemble floor, and **≈ +0.010 solo** to clear it
  with room. The 3-seed stream mean has sd ≈ 0.0029/√3 ≈ **0.0017**, so +0.010 is ≈ 6 sd and +0.0057 ≈ 3.3 sd:
  the arm is adequately powered at 3 seeds.
- The decisive read remains the **ensemble swap/add under the ship bar**, not the solo number. Solo is the
  early-kill screen; if solo < +0.0057 the arm STOPS and no ensemble read is taken.

### ⭐⭐⭐ G0d — DECORRELATION AT FIXED SKILL, PRICED IN THE SAME CURRENCY
`ledger50_decorr.py` → `benchmarks/ledger50_decorr_multi.json`. Anchor exact. Per basin, `lstm_multi`'s error
vector is ROTATED toward the subspace orthogonal to its own and the other eight members' errors, preserving
its norm exactly — so per-basin MSE, and hence solo NSE, is unchanged to floating point (measured shift
+0.000000 at every angle), while its mean error-correlation with the other streams falls.

| θ | mean err-corr | solo | **ensemble h==1 paired** | CI95 | breadth |
|---|---|---|---|---|---|
| 0.00 | 0.8175 | 0.8306 | +0.000000 | [0, 0] | — (identity check ✅) |
| 0.25 | 0.7921 | 0.8306 | **+0.001484** | [+0.00139, +0.00158] | **1.000** |
| 0.50 | 0.7174 | 0.8306 | **+0.005842** | [+0.00548, +0.00621] | **1.000** |
| 0.75 | 0.5981 | 0.8306 | **+0.012804** | [+0.01202, +0.01361] | **1.000** |

⭐ **Exactly linear: 0.0584 of ensemble median per unit of error-correlation removed** (0.0584/0.0584/0.0584
across the three angles), and **breadth 1.000** — every basin gains, which is precisely what a median rewards.
⭐ The stream's weight does **not** move (0.1819 → 0.1819): inverse-MSE sees only MSE, which is unchanged.
So the entire gain is **error cancellation**, not reweighting — the production rule cannot even see it.

⭐⭐ **THE EXCHANGE RATE, and it is the design criterion this campaign never had:**
combining with G0c (lstm_multi pays ≈ 0.20 per unit of solo gain at small gains),
> **removing 0.10 of a member's error correlation is worth about +0.029 of solo NSE.**
> A new member is worth building iff it loses **less than ~0.029 solo per 0.10 of correlation it removes.**

This *reconciles* the campaign's five refutations of decorrelation rather than overturning them: those
measured that decorrelation does not predict value **among members actually built**, because every one of
them bought decorrelation by fitting worse. At **fixed** skill it is worth more per unit than skill is.
Applied to `multi11` (Mamba): it gave up ≈ 0.074 solo, which at this rate needed ≈ 0.25 of correlation
reduction to break even, and it delivered ≈ 0.165 — so it should have failed, and it did.

⚠️ **UPPER BOUND, three reasons, all load-bearing:** (1) the rotation decorrelates from all eight other
members *simultaneously and orthogonally*, which no model constrained by the data can do; (2) independent
noise is drawn for TRAIN and TEST, so the weight fit sees a consistently-decorrelated member — the
optimistic case; (3) "mean error correlation" here is pooled over all rows against each other stream, which
is not the campaign's historical pairwise definition — do not compare the numbers directly without recomputing.

⇒ **Implication for the running arm, stated before its result:** `multiL` is a longer-schedule `lstm_multi`,
so it will be *highly* correlated with the stream it replaces and its value must come from **skill alone** —
the bar of +0.0057 solo stands unchanged. This read does not rescue it; it says where to look **next** if it
fails, which is a member that is *differently* wrong at equal skill (the `multi5` perturbation mechanism,
now with a quantitative acceptance criterion instead of an intuition).

### OPS — TOOLING VERIFIED END TO END, AND TWO FAULTS CAUGHT BEFORE THEY MATTERED (2026-09-01 18:1x)
1. **`score_noq_test50.py` v1 was wrong by +0.000232 and looked fine.** It reimplemented the dump globs and
   so missed `gate_eval._lstm_seeds`'s rule that the retrained nldas member (`_s111_NEW` / `_TRAIN_s1111`)
   **displaces** the original `s111` on both sides: it averaged **4** nldas seeds where the record averages
   3, reading **0.8365215**. A self-test against the record caught it. v2 builds the frozen 9 through the
   **same** `build_streams`→`merge` path as every ledger-44/45 read and now reproduces
   **0.8362893021622820, diff −1.11e-16**. ⇒ **Never reimplement a stream definition; import the one the
   record was computed with.** The error was larger than several effects this ledger is chasing.
2. **A queued lane collision, killed before it fired.** Lane `multiL` still had `s222` and `s333` queued
   while lane `multiLb` was already training `s222`; when `s111` finished it would have started a second
   `s222` and clobbered its run dir. The lane's queue was killed by PID (its running trainer 3997579 and its
   dump-runner 3997555 verified still alive afterwards) and a dedicated `queue_l50_s333.sh` was armed to
   start `s333` only when the `s111` trainer exits, keeping exactly two NH jobs on the card.
3. Verified alongside: `ledger50_gate.py --anchor` exact; `--rebuild` matched-depth path exact
   (3-seed `lstm_multi` baseline 0.8361782476726014, −0.000111 vs the record, as expected for one fewer
   pair of seeds); the frozen base frames are read-only with MD5SUMS and the gate now **refuses to rebuild
   them** rather than silently folding in new seeds.

**State at 18:15:** `multiL s111` (epoch 3/50) and `s222` (epoch 1/50) training, ~27 h each; `s333` armed.
Record unchanged at 0.8362893021622821. Nothing shipped. No test query spent.

### ⚠️⚠️ G0e — KNOWN-ANSWER TEST OF THE MAP: IT IS ~10× TOO LARGE OUTSIDE ITS MEASUREMENT REGIME
`ledger50_validate.py` → `benchmarks/ledger50_validate.json`. Ledger 44 measured the held-out h==1 swap
delta for seven already-rejected candidates, all swapped for `lstm_multi6` (solo 0.8270, mean err-corr
0.8100). The map predicts `Δ ≈ 0.188·(solo_C − solo_I) + 0.0584·(corr_I − corr_C)`:

| candidate | Δsolo | Δcorr removed | **predicted** | **measured** |
|---|---|---|---|---|
| multi8 | −0.0122 | +0.0338 | −0.000325 | −0.000108 |
| multi9 | −0.0088 | +0.0130 | −0.000899 | −0.000243 |
| multi10 | −0.0298 | +0.0631 | −0.001915 | +0.000118 |
| multi11 | −0.0707 | +0.0940 | **−0.007802** | **−0.000367** |
| multi5b | −0.0266 | +0.0679 | −0.001027 | −0.000039 |
| multidrop | −0.0331 | +0.0584 | −0.002822 | −0.000657 |
| multi14 | −0.0187 | +0.0472 | −0.000753 | +0.000057 |

**pearson +0.473, spearman +0.429 (n=7); mean |predicted| 0.00222 vs mean |measured| 0.00023 — the scale is
~10× too large.**

⭐ **THE REASON, and it is the map's boundary condition: the production weight rule is `MSE^−4`, so it
DOWN-WEIGHTS a weak candidate hard and absorbs most of the difference.** The map's coefficients were
measured by perturbing a member within ±0.03 while its weight barely moved (G0b/G0c) or provably did not
move at all (G0d, weight 0.1819 → 0.1819). These seven candidates sit 0.012–0.071 *below* the incumbent,
far outside that neighbourhood, where the weight response dominates.

⇒ **SCOPE, now stated properly: the map prices SMALL CHANGES TO AN EXISTING MEMBER. It is not a swap
predictor for a substantially different member, and must not be quoted as one.** Directional signal at
n=7 is weak-positive, no more.

⭐⭐ **Both of today's kill decisions are REINFORCED, not undermined** — each used the map in the direction
where it *overstates*: the δHBV arm was predicted at ≈ +0.0009 (already below the floor, so the true value
is smaller still), and the depth lane at +0.0010…+0.0018 against a measured +0.00027.

⚠️ **And the `multiL` bar tightens: +0.0057 solo is a LOWER bound on what is needed.** multiL is the same
recipe as `lstm_multi`, so it sits inside the small-change regime where the map is calibrated — but any
error is in the optimistic direction. The pre-registered bar is unchanged (moving a bar after seeing a
calibration would be the forking path); it is simply now understood as necessary, not sufficient.

### G0f — THE CORRELATION STRUCTURE, AND A CORRECTION TO HOW I FRAMED G0d
`benchmarks/ledger50_corr_structure.json`. Per-stream weight, solo h==1, and MEAN error correlation
against the other eight (held-out, h==1):

| stream | weight | solo | **mean err-corr** | max pair | with |
|---|---|---|---|---|---|
| lstm_multi | 0.1819 | 0.8306 | 0.8175 | 0.9464 | lstm_multi6 |
| **lstm_multi5** | **0.2491** | 0.8229 | **0.8078** | **0.9450** | lstm_multi |
| lstm_multi6 | 0.1910 | 0.8270 | 0.8100 | 0.9464 | lstm_multi |
| lstm_nldas | 0.0913 | 0.7525 | 0.6818 | 0.8773 | dhbv_nldas |
| lstm_daymet | 0.0737 | 0.7759 | 0.7303 | 0.8812 | dhbv_daymet |
| lstm_maurer | 0.0680 | 0.7749 | 0.7046 | 0.8506 | dhbv_maurer |
| dhbv_daymet | 0.0615 | 0.7615 | 0.6923 | 0.8812 | lstm_daymet |
| dhbv_nldas | 0.0426 | 0.7395 | **0.6436** | 0.8773 | lstm_nldas |
| dhbv_maurer | 0.0408 | 0.7429 | 0.6983 | 0.8506 | lstm_maurer |

Block structure: LSTM block mean **0.7889**, δHBV block mean **0.5872**, cross-block **0.7084**. Each
single-forcing LSTM's nearest neighbour is its **own-forcing δHBV** (same inputs), not another LSTM.

⚠️⚠️ **THE CORRECTION: `multi5`'s mean error correlation is 0.8078 — indistinguishable from `multi`
(0.8175) and `multi6` (0.8100) — and its pairwise correlation with `multi` is 0.9450, the highest in the
ensemble.** So **G0d does NOT explain why multi5 ships.** Anyone reading G0d ("decorrelation at fixed skill
is worth 0.0584/unit") together with [[why-multi5-works-PERTURBATION-not-information]] would naturally
conclude that multi5 works by decorrelation. **It does not.** multi5 carries the largest weight (0.2491)
despite *not* having the best solo median NSE, because inverse-MSE ranks on **pooled** MSE, not median NSE.

⇒ **G0d identifies an opportunity that no member this campaign ever built has actually taken, rather than a
mechanism that explains past success.** State it that way; the two claims are easy to conflate and I nearly
did. The three multi-forcing LSTMs are ~0.945 mutually correlated and hold **62 % of the weight** — that
redundancy is where G0d says the value is, and it is exactly what nothing has managed to break.

⛔ **And the weighting door stays shut:** correlation-aware and median-targeted weight rules were already
measured in ledger 45 (`benchmarks/ledger45_medianfit.json`) — the legal train-fit median-targeted rule is
**−0.001325 paired h==1, CI [−0.00211, −0.00040] excluding zero**, and even the illegal test-fit ORACLE is
**−0.001659 paired at breadth 0.422** while its *median* reads 0.8413 (a textbook instance of
[[a-median-gain-is-not-a-skill-gain]]). Nothing in G0d reopens combination.

### THE multiL READ IS NOW EXECUTABLE — `gpu1080/read_multiL.sh`, written 2026-09-01 21:4x BEFORE ANY multiL DUMP EXISTS
The decision rule is encoded in a script rather than applied by hand, so it cannot be adjusted after the
numbers are visible. It runs three stages and **refuses to reach stage 3 unless stage 2 passes**:

1. **Artifacts** — `gzip -t`, 531 basins, 14 leads, both TRAIN and TEST present for every seed.
2. **Solo screen, matched depth** — multiL's k-seed stream vs the FIRST k of `lstm_multi`'s canonical seed
   order (`s111,s222,s3334`), h==1 held-out.
3. **Ensemble swap + add** under the ship bar, with `lstm_multi` rebuilt at matched depth.

**AMENDMENT 3 — a k=2 interim early-stop, registered now because seed s333 cannot start until s111 frees
the lane (~26 h later), so a hopeless arm would otherwise burn a further day.**
- At k=2 the ONLY permitted action is to **STOP**: if `solo(multiL,2) − solo(lstm_multi,2) ≤ −0.005`, the arm
  is abandoned and `s333` is not trained. The 2-seed stream mean has sd ≈ 0.0029/√2 ≈ 0.0021, so −0.005 is
  ≈ 2.4 sd below zero — it can only fire on a clearly-worse arm.
- **No ensemble read is taken at k=2**, and a k=2 result may never be used to *pass* the arm. The script
  exits 0 with "not hopeless" and nothing else.
- The registered read remains **k=3, solo ≥ +0.0057**, then the ensemble read. Below that the arm STOPS with
  no ensemble read — this is what prevents a failed solo screen from being followed by a hopeful peek at the
  ensemble number, which is how a forking path would open here.

Headline is produced only on a SHIP, via `score_noq_test50.py --selftest` (must print the record) followed
by the `--add`/`--swap` invocation the script prints.

## ⛔⛔ LEDGER 50 — multiL FAILS THE INTERIM STOP. THE ARM IS CLOSED. (2026-09-02 19:0x)

`gpu1080/read_multiL.sh 2` → `benchmarks/ledger50_multiL_solo_k2.json`, `ledger50_multi_solo_k2.json`.
Artifacts verified first: all four dumps `gzip -t` clean, 531 basins, 14 leads, 2,570,572 TEST /
2,861,029 TRAIN rows each.

**Solo h==1 held-out, matched 2-seed depth:**

| | s111 | s222 | single-seed mean (sd) | **2-seed stream** |
|---|---|---|---|---|
| **multiL** (50 ep, LR 25/35, clip 1.0) | 0.800099 | 0.805558 | 0.802828 (0.00386) | **0.815918** |
| `lstm_multi` (30 ep, LR 20/25, no clip) | 0.806711 | 0.804624 | 0.805667 (0.00148) | **0.823176** |

**DELTA = −0.007258**, past the pre-registered stop threshold of −0.005 (≈ 2.4 sd of the 2-seed mean).
⇒ **AMENDMENT 3 fires: the arm is abandoned, `s333` was killed by PID mid-training and its run dir
removed, and NO ENSEMBLE READ IS TAKEN.** The read script enforced this — it exited at stage 2 and never
reached stage 3, which is exactly the forking path the design was meant to close.

### ⭐⭐ WHY IT FAILED — the 7th time in-run training loss has misled this campaign
multiL reached a **lower training loss than any standard run** (0.00564 / 0.00544 at epoch 50, monotone to
the end, "best epoch 50", versus ~0.0067 for the 30-epoch members) and was **worse on held-out data**.
Training 20 epochs longer with a later LR decay **fit the training window better and generalised worse** —
textbook overfitting, on a corpus of 531 basins × 15 years that the field treats as data-rich.
⭐ Note also **multiL's seed spread more than doubled** (sd 0.00386 vs 0.00148): the longer schedule made
the member both worse and less stable, so a 3rd seed could not have rescued the mean.

### ⇒ WHAT THIS CLOSES
The **training-schedule axis** — epochs, LR-decay placement, and gradient clipping — was the last
never-varied knob on the no-q LSTM members. It is now measured and **negative**. Together with the
already-closed width, sequence length, input dropout, loss variants, distributional heads, aux targets,
pretrained init, training-set size, statics and architecture, **the LSTM training surface is exhausted.**

⚠️ **Declared confound, as registered:** multiL changed the schedule AND added `clip_gradient_norm: 1.0`
together. A pass would not have attributed; **the fail closes both jointly.** Clipping alone is not
implicated — multi5/multi6 use it and are the two best members — so the honest reading is that the
**longer schedule** is what cost the generalisation.

**Record unchanged: 0.8362893021622821. Nothing shipped. No test query spent. GPU now idle.**

## ⭐⭐⭐⭐⭐ G0g — THE ACHIEVABLE FRONTIER IS 1.47× TOO EXPENSIVE (2026-09-02, closing result)

`ledger50_frontier.py` → `benchmarks/ledger50_frontier.json`. Every member this campaign ever built that has
usable TEST dumps (n=19) placed on the (correlation removed, solo lost) plane against the break-even line
implied by G0c/G0d, referenced to `lstm_multi` (solo 0.8306, mean err-corr 0.8175).

⚠️ **A BUG CAUGHT BY IMPLAUSIBILITY, RECORDED:** the first run wrote the slope as `0.188/0.0584 = 3.22` —
the **reciprocal** — and reported **"19 of 19 members profitable"** against 19 known failures. An
implausible result is a bug until proven otherwise. Corrected to `0.0584/0.188 = 0.311`; the wrong artifact
is kept as `ledger50_frontier_WRONG_slope_inverted.json`.

| | slope of `d_solo` vs `d_corr_removed` |
|---|---|
| **what the ensemble PAYS** (break-even, from G0c+G0d) | **−0.311** |
| **what the models ACHIEVE** (fit through origin, n=19, **pearson −0.805**) | **−0.458** |

⇒ ⭐⭐ **Members pay 1.47× more skill per unit of decorrelation than the ensemble rewards, and the
relationship is tight (r = −0.805 across 19 members spanning architectures, widths, losses, corpora,
dropout, bagging, sequence length and now training schedule).**

**18 of 19 lie below the break-even line** (margins −0.0433 … +0.0044, median −0.0111). Per-member ratios
run from **−0.284 (gmm)** — the only one beating −0.311 — to −0.885 (multi5swa). Even gmm's implied
contribution is ≈ +0.0008, **below the ±0.00103 floor**, and it is the best of 19 selected on held-out
numbers, so it is **not** a candidate; the distributional arm is already closed and stays closed.

### ⇒ THE CLOSING STATEMENT OF LEDGER 50
> The campaign's members all sit on a single, tight skill-versus-decorrelation frontier, and that frontier
> is **1.47× steeper than the rate this ensemble pays**. Every axis tried moves a member *along* the
> frontier; none moves it *off*. Moving the record needs a member off the frontier — decorrelation cheaper
> than 0.311 solo per unit — and 19 attempts across every axis this campaign could think of have not
> produced one.

⚠️ **Scope, three limits:** (1) the coefficients come from perturbing `lstm_multi`, and G0e showed
magnitudes are unreliable outside that neighbourhood — this analysis uses the **ratio**, which the sign
test supported (5/7), but individual margins are within the map's uncertainty; (2) the frontier is
"achievable by **this** family of recipes on **this** corpus", not achievable in principle; (3) the fit is
through the origin by construction, since the reference is a member of the same family.

## ⭐⭐⭐⭐ G0h — DOES A BETTER WITH-Q MODEL HELP NO-Q? THE ADVANTAGE IS OBSERVATIONAL, NOT FUNCTIONAL (2026-09-03)

**The question:** with-q reaches 0.9253 while no-q sits at 0.8363. Could a very good with-q model be
distilled into a no-q one? The with-q edge is *persistence from observed q* **plus** possibly *better
catchment-state inference*. Only the second could ever transfer, because the first needs the missing input.
**Persistence decays with lead; state inference should not.** So measure the gap as a function of lead.

Matched by construction: same forcing (daymet), same window, same stride-14 × 14-lead grid, same 177 basins,
`camels531_daymet_withq5_full531` vs `camels531_daymet_nhlstm_s14`.
⚠️ **A date-convention trap, diagnosed not assumed:** with-q dumps use `target_date = t0+h`, NH no-q dumps
use `t0+(h−1)`. Joining on `(t0,h)` pairs rows one day apart — truth agreed on only **13.4 %** of rows.
Joined on `(station_id, date)` the truth agreement is **1.0000**.

| lead | with-q | no-q | **gap** | naive persistence |
|---|---|---|---|---|
| 1 | 0.8889 | 0.7731 | **+0.1158** | 0.5121 |
| 3 | 0.8423 | 0.7892 | +0.0532 | −0.1871 |
| 7 | 0.8202 | 0.7590 | +0.0612 | −0.4631 |
| 14 | 0.8169 | 0.7557 | **+0.0612** | **−0.6056** |

⭐⭐ **Only 47 % of the lead-1 advantage decays away. A gap of ≈ +0.058 PERSISTS across leads 8–14, while
naive persistence has collapsed to −0.61 — i.e. worthless.** Both models see identical forcings over the
horizon; the with-q model additionally saw discharge up to 14 days earlier. So the durable gap is **not**
persistence: observing discharge carries **catchment-state information that is still worth ~+0.06 NSE two
weeks later**.

⇒ ⭐ **This is consistent with, and sharpens, the DIRECTION WALL** ([[LEDGER-49-PLANNING-BRIEF]]): daily
areal *rainfall* — an observation of the **input** — cannot tell you which way the model is wrong;
observed *discharge* — an observation of the **state** — can, and durably.

### ⛔ BUT IT DOES NOT TRANSFER, FOR TWO INDEPENDENT REASONS
1. **The advantage is observational, not functional.** Distillation transfers a *function*; the teacher's
   function has an input the student cannot evaluate. The student already has the exact labels for the
   forcings→flow mapping (that is its training target), so a teacher's output is a *worse* copy of what it
   already has. What the teacher knows extra is the realised catchment state, which is observed, not inferred.
2. **Privileged-information training improves sample efficiency, and this problem is not sample-limited.**
   Gauch et al. (same 531 basins) find skill grows strongly only to ~1.6e5 samples; we train on ~2.9e6 —
   **~18× past saturation**. And [[ledger41-CLOSED-competence-is-not-contribution]] measured, in this exact
   data-rich regime, that a competent prior finetunes **worse** than random init.

⚠️ **Scope:** measured on the OLDER split (1989–99 backtest, 177 basins, models trained 1999–2008), so
+0.058 is the *mechanism's* size, **not** a number to quote against 0.8362893.

⇒ **Answer: no — making the with-q model better does not move the no-q record.** What the measurement does
give is a **new, independent statement of the no-q ceiling**: ≈ +0.06 of member-level NSE is available only
to a model that can observe discharge, and is unavailable in principle without it.

---

# LEDGER 51 — THE WITH-Q TRACK, REBUILT ON THE NEARING PROTOCOL (opened 2026-09-04)

## ⛔⛔ WHY THIS LEDGER EXISTS: THE WITH-Q RECORD WAS TRAINED ON ITS OWN TEST DECADE

Audited 2026-09-03/04, read-only, from the persisted checkpoint `cfg` of all 28 with-q checkpoints on
nakas-1080 plus the training logs. `scripts/train_mblstm.py` applies **no lower bound on training windows
unless `--train-start` is passed** (its own help text names this exact failure), and:

| checkpoints | `train_start` | `train_end` | basins | verdict |
|---|---|---|---|---|
| `{daymet,maurer,nldas}_withq_s981–s984` (2026-07-13, rented GPU) | set | **2008-09-30** | 531 | **GUARDED** |
| `{daymet,maurer,nldas}_withq_s985/s986` (08-01) | — | 2024-12-31 | 671 | **UNGUARDED** |
| `aorc_withq_s981–s985` (08-02/03) | — | 2024-12-31 | 530/531 | **UNGUARDED** |
| `fused_withq_s981/s982` (08-01) | — | 2024-12-31 | 671 | **UNGUARDED** |
| `withqmulti_s973–s975` (08-09/10) | — | 2024-12-31 | 671 | **UNGUARDED** |

Every August launcher (`gpu1080/supervisor_withq.sh`, `supervisor_withq_fused.sh`, `train_aorc_s985.sh`,
`train_withqmulti.sh`) copied the July `--val-*` line and **dropped the `--train-start 1999-10-01
--train-end 2008-09-30` line sitting next to it**. `train_withqmulti.sh` says "protocol copied VERBATIM";
it was — including the omission. Measured confirmation: unguarded runs report `windows: train=` of
**6,699,323–8,103,907**, identical to the deliberately-unguarded `dhbv_allh_*` pool; the first guarded
LEDGER-51 run reports **1,739,025**. The 1989-10-01..1999-09-30 **test decade was 100 % inside training**.

⇒ **0.9203 (4-member) and 0.9253 (5-member) are in-sample and are RETRACTED.** They are not comparable to
Nearing 2022's held-out 0.879. The honest surviving high-water mark is **0.9058** (3 forcings × 4 guarded
seeds, plain mean, `benchmarks/withq_push_sweep.json`) and **0.9016** (3 × 2 guarded seeds,
`benchmarks/combine_withq_full531_2seed.json`, EXPERIMENTS row 38).

⭐ **The lesson, and it is a new one for this campaign**: *a protocol guard that lives in the CALLER is
lost the moment a script is copied.* Six months of gates, controls and pre-registration sat downstream of a
flag nobody re-checked, and no log recorded the invocation. Fixed structurally in this ledger: the trainer
now echoes `ARGV:` and persists `train_start` in the checkpoint, and `queue_l51_withq.sh` **reads the guard
back out of the artifacts** (checkpoint `cfg` + the log's window count) and aborts rather than trusting
its own command line. Add to [[my-causal-claims-keep-failing]].

⚠️ Separately: the Aug-11 3-seed `withqmulti` backtest overwrote the dump behind the published 0.9253, so
that artifact no longer exists. It is moot — both are unguarded.

## THE PROTOCOL, FROZEN BEFORE ANY GPU

Nearing et al. 2022, HESS 26:5493 (fetched 2026-09-03), the number to beat = **0.879**:
train **1999-10-01..2008-09-30**, hyperparameter validation **1980–1989**, test **1989-10-01..1999-09-30**,
**531 basins**, metric = NSE per basin over **all daily observations** in the test period, median across
basins; AR input = lagged observed discharge + an observed/simulated flag; forcings Daymet+NLDAS+Maurer.

Ours, matching it:
- `--train-start 1999-10-01 --train-end 2008-09-30`, `--val-start 1980-10-01 --val-end 1989-09-30`,
  test 1989-10-01..1999-09-30. The val window is **strictly outside test**, unlike every prior with-q run
  (which validated on 1998-10..1999-09, i.e. inside the test decade), and doubles as the honest window for
  readout and ensemble-weight selection.
- ⚠️ **Declared and accepted**: the 365-day encoder reaches ~1 year before train-start, so encoder inputs
  span 1998-10..1999-09. That is the standard convention (neuralhydrology loads `start_date − seq_length`
  for exactly this reason); **no test-decade day is ever a training TARGET**.
- Corpora `camels_corpus_*_v2_531` (symlink trees of the 531 benchmark basins). `usable stations: 531`
  is asserted.
- Recipe unchanged from the record members so the comparison is like-for-like: `--epochs 30 --head quantile
  --q-transform linear --static-set camels --hidden 256`, `camels1f` / `camels3fv2`, wps 300 (default),
  AR-masking augmentation `ar_mask_p=0.3` as before. Seeds s501–s505.
- Readout `(ylo+yhi)/2` (the record's; a 2026-08-09 sweep of 8 blends found nothing better out of sample).
  Seed averaging **downstream**: mean of the physical quantile slots, then sort, then clip at 0 — identical
  to `app/mblstm.py` under the linear transform, where denorm is affine.
- **Headline frame = day-1 on EVERY day of the test window** (stride 1). The old stride-14 h==1 frame is
  reported alongside for continuity with 0.9058/0.9203.

## GATES (fixed in advance)

1. **MEMBER** — a clean single-forcing member ≥ 0.85 day-1 median NSE; fused ≥ 0.87. Below 0.80 ⇒ broken,
   stop and diagnose rather than adding seeds.
2. **ANCHOR** — the 12 guarded July checkpoints re-dumped on the new frames must reproduce their published
   member scores (daymet 0.8868 / maurer 0.8797 / nldas 0.8777 at 4 seeds, stride-14) within ±0.005.
   This is the control that separates "the guard cost us skill" from "the rebuild is broken".
3. **SHIP BAR** for any change to the ensemble (the ledger-50 rule): paired per-basin Δ > 0, basin-bootstrap
   95 % CI excludes zero, breadth ≥ 0.5, on the all-days frame; the stride-14 sign must agree.
4. **SEEDS** — 5 per member, never more (5th seed measured +0.0005; multi6 3→6 measured +0.00027).
5. **CONTROL** — duplicate-member control (adding a copy of an existing member) must be ≤ 0, as in
   `verify_withqmulti.py` V3, before any new member is credited.

## PREDICTIONS ON RECORD (so they can be wrong)

- Clean 5-member ensemble, stride-14 frame: **0.905–0.915**; all-days frame within ±0.01 of it.
  Nearing's 0.879 still beaten. **Falsifier**: < 0.879 ⇒ reported as a failure to beat the record, and the
  campaign's with-q claim is withdrawn entirely.
- The guarded-vs-unguarded gap on the same recipe: **0.010–0.020** of median NSE (i.e. most of the
  0.9058 → 0.9253 climb was leak, not skill).
- **P1 (NSE-aligned loss, `--point-loss mse` on fused3)**: solo **+0.003…+0.010** vs pinball fused3 at
  matched seeds. Rationale: every with-q member ever trained used pinball; `--point-loss mse` is the
  basin-NSE loss that lifted the no-q recipe, and NSE is the scored metric. Never tried on this track.
- **P2 (fused4 = +AORC, `camels4fv2`)**: solo **+0.002…+0.006** vs fused3. Counter-evidence on record:
  AORC is the weakest solo forcing measured (0.7047 no-q) — but it was worth +0.0059 as a with-q *member*.
- **Weighting on the val window**: **+0.001…+0.003**, or zero. With-q members were previously found
  exchangeable (own-NSE spread 0.008 ⇒ equal weight optimal); a 5th, differently-built member may change that.
- **P3 (stale-gauge augmentation removed, `--ar-mask-p 0.0` on fused3)**: solo **+0.002…+0.015** vs
  fused3 at matched seeds. ⭐ The mechanism, and it is a **train/eval mismatch on the scored quantity**:
  `Corpus.sample` masks the trailing 1–14 days of encoder discharge on **30 %** of training samples
  (hard-coded `ar_mask_p=0.3`, never a flag until now), simulating a stale gauge — but the benchmark
  scores a day-1 nowcast with a **complete** observation (Nearing's headline 0.879 is the
  "no holdout" row). Every with-q member ever trained here paid that tax. **Counter-hypothesis,
  stated in advance**: the masking is a regulariser that stops the model leaning on persistence, so
  removing it could overfit to the lagged observation and *hurt*. A negative result is informative
  either way, and 0.0 is the extreme — if the mechanism is real it shows most strongly there.
- **P4 (lead-1 loss weighting, `--h1-weight 0.5` on fused3)**: solo **+0.005…+0.025** vs fused3 at
  matched seeds. ⭐ **The largest train/eval mismatch found so far, and it is on the LEAD axis**: the
  loss is a masked mean over all 14 forecast leads, so lead 1 carries **1/14 = 0.071 of the
  gradient** — while the benchmark scores **lead 1 and nothing else**. Nearing's AR-LSTM is a 1-day
  model; ours spends 13/14 of its capacity on leads that are never scored. `--h1-weight 0.5` gives
  lead 1 half the loss (a 7× upweight) and splits the rest over leads 2–14, renormalised so the loss
  keeps the same scale. Best-epoch selection uses the same weighted loss, so the checkpoint chosen is
  the best *day-1* model. **Counter-hypothesis, stated in advance**: the multi-lead task is a
  regulariser and a shared encoder may lose more from narrowing than lead 1 gains.
- A probe below its band at 3 seeds is dropped **with no ensemble read**.

## NOT DOING (measured dead; do not re-open)

More seeds past 5 · hidden 512 · sequence length · readout blends · per-basin weights or per-basin
finetuning · extra training years (protocol) · neighbour-gauge inputs (a third protocol, and the clean
effect is +0.0007 CI-straddling) · a top-50 specialist (the metric is a rank statistic; +0.05 on the worst
50 moves the median by 0.00000).

## 🏆 LEDGER 51 — CLOSING BLOCK (2026-09-05)

**Result: 0.888355** day-1 median NSE, 531 basins, all daily observations, guarded Nearing split.
**Nearing 2022 = 0.879 ⇒ +0.009355.** 5 members × 5 seeds, equal weight, readout `(ylo+yhi)/2`, zero
fitted combination parameters. Artifact `benchmarks/l51_FINAL_5member_test1.json`.

### Predictions vs outcomes (scored against the bands fixed before any GPU)

| prediction | band | outcome |
|---|---|---|
| clean 5-member ensemble, stride-14 frame | 0.905–0.915 | **0.9045** (4-member 1-seed, uniform) — ✅ in band |
| guarded-vs-unguarded gap on the same recipe | 0.010–0.020 | **≈0.038 at member level** — ❌ **far larger than predicted** |
| **P4** lead-1 weighting, solo | +0.005…+0.025 | **+0.0143** ✅ in band, and it **shipped** |
| **P1** `--point-loss mse`, solo | +0.003…+0.010 | **−0.0069 / −0.0108** ❌ falsified, both windows |
| **P3** `--ar-mask-p 0`, solo | +0.002…+0.015 | +0.0097 test / **−0.0029 val** ⇒ **rejected on the honest window** |
| **P2** 4-forcing (+AORC), solo | +0.002…+0.006 | +0.0094 test; val paired CI touches zero — **open** |
| weighting on val | +0.001…+0.003, or zero | **zero** — equal weight survived ✅ |
| falsifier: clean ensemble < 0.879 ⇒ report the failure | — | not triggered (0.888355) |

⚠️ **The one prediction that was badly wrong was mine about the leak's size**: I predicted the guarded
recipe would lose 0.010–0.020; the member-level loss was ~0.038, and the honest starting point
(0.874270) was **below** the published record. The ledger only cleared 0.879 because of P4.

### What the ledger changed structurally
1. `train_mblstm.py` echoes `ARGV:` and persists `train_start`; `--dump-day1` now works for the quantile
   head (it silently wrote header-only files before, for every non-δHBV member);
   `--ar-mask-p` and `--h1-weight` exposed; `camels4fv2` added.
2. `queue_l51_withq.sh` **reads the protocol back out of the artifacts** (checkpoint `cfg` + the log's
   window count) and aborts — it never trusts its own argv.
3. `l51_withq_score.py` **refuses any dump without a sidecar** recording the guarded window; that check
   also caught a half-written dump during the run.
⇒ **An experiment's protocol must be verifiable from the artifact it produced, not from the script that
produced it.**

### Open items (not blocking the result)
- **P2** (4-forcing `fused4h1`): 3-seed matched-row val adjudication in flight. If it ships it is additive
  to 0.888355, not a correction of it.
- The 12 **guarded July checkpoints** were never re-dumped on the all-days frame, so the exact honest
  value of the historical 0.9058 ladder is unmeasured (its frame inflation is bounded at ≈+0.036).
- `research/WHITEPAPER.md` carries a correction notice at §5.2; the numeric claims in §5.3–§8 (ceilings,
  headroom, saturation) are all computed from retracted with-q dumps and **need recomputation**.

---

# LEDGER 52 — CAN THE WITH-Q RECORD REACH 0.89? (opened 2026-09-05, before any result)

Record after LEDGER 51: **0.888355** (Nearing split, all-days frame, 531 basins). **0.89 needs +0.0016.**
⚠️ Stated in advance: **0.90 needs +0.0116, which exceeds the entire lead-weighting gain (+0.0111 at
ensemble level). It is not reachable by tuning** and is not a target of this ledger.

## The premise, and why it is not a fishing expedition

Every gain in LEDGER 51 came from removing a **train/eval mismatch**, never from new data. Two of the three
mismatches found are now fixed (the evaluation frame, the lead weighting). This ledger tests the remaining
one plus a proper tuning of the fix that worked.

## S1 — CHECKPOINT SELECTION ON THE SCORED QUANTITY (`--select-by nse`)

`train_mblstm.py` keeps the epoch with the lowest **val pinball loss**; the benchmark scores **NSE at lead
1**. Measured over **all 28 LEDGER-51 runs** before building anything:

| | |
|---|---|
| runs where best-pinball and best-NSE are the **same** epoch | **5 / 28 (18 %)** |
| mean val-NSE gap (best-NSE minus best-pinball epoch) | **+0.00146** |
| median / max | +0.00100 / +0.00400 |

⇒ In 82 % of runs the saved checkpoint is **not** the best day-1 model the run produced.
Implementation adds `--select-by {loss,nse}`; `nse` keeps the highest **lead-1** median NSE (the existing
`val_medNSE` pools all 14 leads, so a new `val_h1NSE` is computed and logged). NSE is invariant under a
common per-basin affine transform, so the z-space value equals the physical one.

**Prediction: +0.000…+0.004 solo** at 3 matched seeds. ⚠️ The +0.00146 above is an **upper bound** — it is
the val-side gap, measured on the window the selection itself uses, so it will not transfer 1:1.

## S2/S3 — IS 0.5 ACTUALLY THE OPTIMAL LEAD WEIGHT? (`--h1-weight` 0.3 and 0.7)

`--h1-weight 0.5` shipped after comparing **only 0.5 against 0.9, at one seed each**, where the two windows
disagreed in sign. That is a coarse tuning of the campaign's largest lever. P2 demonstrated the hazard the
same day: its 1-seed test signal (+0.001888, CI excluding zero) **evaporated at 3 seeds** (+0.000290, CI
straddling zero). 0.3 and 0.7 bracket the shipped value at 3 seeds each.

**Prediction: flat, |Δ| < 0.002 vs 0.5.** The 0.5-vs-0.9 gap was ~0.5 sd, which suggests a broad plateau.

## DECISION RULE (fixed now)

Selection on **val1** (stride-1, held out, the honest frame — never val7, whose grids are offset across
corpora and which is inflated ≈+0.023). A change ships only if, at **3 matched seeds**: paired Δ > 0,
basin-bootstrap 95 % CI **excludes zero**, breadth ≥ 0.5. Test is read **after** the val decision and never
selects. A winner is then applied to all 5 members (25 seeds) before any new record is claimed.

**Falsifier, stated in advance:** if all three screens are within noise on val1, **the recipe is final at
0.888355 and tuning is exhausted** — that is the reportable result, and the next move would have to be a
new member class (NeuralHydrology's `arlstm`, installed and never run here; or a δHBV with-q member at
78 GPU-h), not another sweep.

## Already closed this ledger, at zero GPU

**READOUT — no change.** Ten point-readouts swept on val. At **member** level two beat the record readout on
**both** frames (`.40lo+.20med+.40hi` +0.0011 on stride-1, `(ylo+ymed+yhi)/3` +0.0010 on stride-7), but at
**ensemble** level the record `(ylo+yhi)/2` wins (the best alternative is −0.0011). ⭐ The disagreement is
**level, not frame**: averaging across members already widens the point estimate, so an individually
under-dispersed member gains from widening and the ensemble does not. **The record is an ensemble number, so
the decision is made at ensemble level.** `(ylo+yhi)/2` stands.

### LEDGER 52 — INTERIM (3 seeds, val stride-7; formal decision pending on val1)

| variant | val7, 3 seeds | vs shipped 0.5 | **1-seed read** |
|---|---|---|---|
| `fused3h1` (`--h1-weight 0.5`, shipped) | 0.894470 | — | — |
| **`fused3h1w07` (0.7)** | **0.900560** | **+0.006090** | +0.0031 |
| `fused3h1sel` (`--select-by nse`) | 0.895776 | +0.001306 | **−0.0032** |
| `fused3h1w03` (0.3) | 0.893480 | −0.000990 | −0.0054 |

⚠️ **MY PREDICTION WAS WRONG.** I predicted the weight curve would be **flat, |Δ| < 0.002**, on the
grounds that 0.5-vs-0.9 was ~0.5 sd. Measured: **0.7 is +0.006 over 0.5** — a third of the size of the
original lead-weighting discovery, left on the table because I tuned the campaign's largest lever with
**two points at one seed each**. The lesson is not "0.7 is better"; it is that **the coarse sweep that
established a lever is not a substitute for tuning it**, and 0.5 was simply the first value I tried.

⚠️ **Both single-seed reads misled**, one in sign (`sel` −0.0032 → +0.0013; `w07` +0.0031 → +0.0061).
That is the third time in two days that a 1-seed read did not survive to 3 seeds (after P2). **Treat any
1-seed screen on this stack as directional only.**

⇒ The sweep is extended to **0.8 and 0.9 at 3 seeds** (the optimum is clearly above 0.5 and 0.9 was only
ever read at 1 seed). S1 (`--select-by nse`) is positive but small and still inside its predicted band;
its formal verdict waits for val1 with a paired CI.

⚠️ These are **val7** numbers, used for direction only — the pre-registered decision rule selects on
**val1**, whose dumps are running.

### ⚠️ LEDGER 52 — THE INTERIM READING ABOVE IS WITHDRAWN (formal decision on val1, paired)

The interim table said `--h1-weight 0.7` was **+0.006090** over 0.5 and that my "flat curve" prediction had
failed. **Both statements are wrong.** That reading used **val7 medians**; the pre-registered rule is
**val1 + the paired statistic**:

| variant | val1 median Δ | **PAIRED Δ** | CI | breadth | ship |
|---|---|---|---|---|---|
| `fused3h1w07` (0.7) | +0.001302 | **−0.000262** | [−0.000659, +0.000073] | **0.467** | **no** |
| `fused3h1sel` (`--select-by nse`) | −0.000323 | −0.000093 | [−0.000255, +0.000042] | 0.465 | no |
| `fused3h1w03` (0.3) | −0.003727 | −0.001663 | [−0.002323, −0.001231] | 0.383 | no (worse) |

**The +0.0061 was inflated twice**: the frame (val7 median +0.0061 → val1 median +0.0013) and then the
statistic (val1 median +0.0013 → **paired −0.000262**, breadth 0.467). Difference-of-medians vs paired have
now disagreed **six times** across ledgers 51–52, and the paired statistic has been right every time.

⇒ **My original prediction (flat, |Δ| < 0.002) was CORRECT**, and the self-criticism recorded in the interim
block ("I under-tuned the campaign's largest lever") was itself an artifact of reading a rank statistic on
an inflated frame. **`--h1-weight 0.5` stands.** ⭐ The transferable point is sharper than the one I wrote
before: *a lever can look under-tuned purely because the diagnostic is a median on a sparse frame.* Check
the statistic before rewriting the conclusion.

⚠️ `w08`/`w09` val1 dumps still pending; the verdict is provisional until they land, but three of five
weights now sit inside noise on the honest reading.

## ⛔ LEDGER 52 — CLOSED NEGATIVE. THE FALSIFIER TRIGGERED; THE RECORD IS FINAL AT 0.888355

Final decision, **val1 (stride-1, held out), paired statistic, 3 matched seeds**, as pre-registered:

| variant | val1 median Δ | **PAIRED Δ** | CI | breadth | ship |
|---|---|---|---|---|---|
| `w03` (0.3) | −0.003727 | −0.001663 | [−0.002323, −0.001231] | 0.383 | no — **worse** |
| `w07` (0.7) | +0.001406 | +0.000039 | [−0.000202, +0.000491] | 0.510 | no |
| **`w08` (0.8)** | +0.001512 | **+0.000311** | **[−0.000017, +0.000638]** | 0.539 | **no** — CI lower bound misses zero by **1.7e-5** |
| `w09` (0.9) | −0.000290 | −0.000226 | [−0.000668, +0.000240] | 0.469 | no |
| `sel` (`--select-by nse`) | −0.000323 | −0.000093 | [−0.000255, +0.000042] | 0.465 | no |

**Every screen fails the bar.** The lead-weight curve is a **very shallow plateau**: 0.3 is clearly worse
(−0.0017), and 0.5 / 0.7 / 0.8 / 0.9 are mutually indistinguishable at ~±0.0003 — an order of magnitude
below the campaign's ±0.00103 significance floor. `w08` is the nearest miss and even if 5 seeds tightened
its CI to exclude zero, **+0.0003 would not carry 0.888355 to 0.89** (which needs +0.0016).

⇒ **`--h1-weight 0.5` stands. `--select-by loss` stands. The with-q recipe is FINAL at 0.888355.**

### Predictions scored
| prediction | outcome |
|---|---|
| S1 `--select-by nse`: **+0.000…+0.004** | **−0.000093** — ❌ falsified (just outside the band) |
| S2/S3 weight sweep: **flat, \|Δ\| < 0.002** | ✅ **CORRECT** (max \|paired Δ\| = 0.0017, and that is the *worse* end) |
| "0.89 is likely reachable" (told to the user) | ❌ **WRONG** — no tuning lever reaches it |
| "0.90 is not reachable by tuning" | ✅ correct, and the same now applies to 0.89 |

⭐ **S1's failure is instructive and I should have foreseen it.** The +0.00146 "gap" I measured from the
logs was the **argmax of a noisy validation curve**; taking that argmax as the checkpoint is selecting on
val noise, not on signal. **A measured gap that is itself an argmax over a noisy series is not available
headroom** — it is the optimism of maximisation. Same family as the in-sample/oracle bounds this campaign
already knows to distrust.

⚠️ And the interim block above stands as a recorded error of mine: reading val7 **medians** made a flat
lever look like a +0.006 lever, and I wrote a confident self-criticism on that basis before running the
paired test on the honest frame.

### ⇒ What would actually be needed to pass 0.89
Not a sweep. A new member class: **NeuralHydrology's `arlstm`** (installed in the box venv, never run here,
and it is the reference implementation of Nearing's own AR setup), or a **δHBV with-q member** (no
process-model member exists on this track; ~78 GPU-h each on the 1080). Both are ledger-sized undertakings,
not tuning.
