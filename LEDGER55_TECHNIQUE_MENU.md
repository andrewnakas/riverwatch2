# LEDGER 55 — THE TRAINING-TECHNIQUE MENU FOR THE 0.90 PUSH

**Written 2026-09-21, zero GPU.** Companion to `PREREG_v2.md` §LEDGER 54 and the L55 brief.
Record: **0.898364** test1 (8 members, equal weight, 5-seed ship bar). **0.90 needs +0.001636.**
Nothing here is pre-registered yet — pre-register the arms you pick, with falsifiers, before any run.

---

## 0. OPS — THE BOX IS OFFLINE (2026-09-21)

Tailscale: `100.101.6.73 nakas-1080 ... active; relay "den"; offline`, ssh times out.
The `nhar0h512` capacity arm (the last pre-registered L54 arm, resumed from epoch 20 on 09-20)
**cannot be read.** Per [[a-watcher-must-check-the-resource-not-the-file]] this is exactly the
shape that cost four days in L54 — the watchers will sit happily on a dead box.
**First action next session: reach the box, read `nhar0h512`, score its falsifier.**

---

## 1. ⭐⭐ THE ARITHMETIC THAT PICKS THE TECHNIQUE — THE 1/8 RULE

The record is an **equal-weight mean of 8 members**. An improvement to one member is divided by 8
before it reaches the score. Two independent routes, and they agree:

| route | one member must gain (solo) to buy +0.001636 |
|---|---|
| **A — 1/N leverage** (ensemble mean of 8, best case, fully common-mode) | **+0.013** |
| **B — ledger-50 sensitivity map**, convex, rescaled to w = 0.125 | **+0.012 … +0.020** |

| all 8 members lifted by δ | ensemble gain (sublinear … common-mode upper bound) |
|---|---|
| +0.003 | +0.0015 … +0.0030 — marginal |
| **+0.005** | **+0.0025 … +0.0050 — CLEARS IT** |
| +0.008 | +0.0043 … +0.0080 |

> ⭐⭐ **A ninth member must be +0.013 solo better than the incumbents. A training technique that
> lifts every member by +0.005 clears the gap with room to spare.**
> The best member ever built here contributes **+0.000759**. Asking for 2.2× that from one more
> member is asking for something this campaign has never produced. Asking for +0.005 solo from a
> *recipe* change is asking for one third of what `--h1-weight 0.5` already delivered (+0.0143).

⇒ **Prioritise recipe-level techniques that apply to every member, over a new member or a new
architecture.** This is the single most important reframe in this document.

⚠️ Caveats, both stated: (a) the map was calibrated on the **inverse-MSE** no-q ensemble, where part
of the gain is weight reallocation — equal weight has no such channel, so route B is optimistic;
(b) sublinearity is measured only to 3 members (ledger-50 G0: 3 × +0.03 = 1.91× one member), the
0.5 factor at 8 is an extrapolation. Route A needs neither and gives the same answer.

---

## 2. TIER A — CHEAP, MECHANISM-GROUNDED, DO THESE FIRST

### A1. ⭐⭐ DENSE SUPERVISION — `predict_last_n` 1 → 30
**The mismatch.** The frozen AR recipe is `seq_length: 365`, `predict_last_n: 1`:

| | per epoch |
|---|---|
| samples | 1,745,397 |
| timestep evaluations (forward compute) | 637,069,905 |
| **supervised targets** | **1,745,397 = 0.27 % of the forward compute** |

**99.73 % of the recurrent compute produces no gradient signal.** At `predict_last_n: 30` it is
8.2 % — a **30× increase in supervised targets per GPU-second**, at the same architecture, same
data, same epochs.

⭐ **This is structurally the same discovery as [[lead1-loss-weighting-the-biggest-member-gain]]**
("check what fraction of the loss lands on the quantity you are scored on") applied one axis over:
*what fraction of the compute produces a gradient at all.* That one was worth +0.0143 solo.

**Why it is still the same task**: the AR input is `q_mm_shift1` at **every** timestep, so a target
at sequence position 336 is the identical one-day-ahead-with-yesterday's-flow problem as position
365, with 336 days of warm-up instead of 365. n = 30 keeps every warm-up ≥ 335 days.

**Two ways to cash it**, and they are different experiments — run them in this order:
1. *as skill* — same 30 epochs, 30× the gradient signal, lower gradient variance. Predicted: better
   convergence within the fixed budget.
2. *as compute* — if (1) is flat, it is still a 30× supervision multiplier, which is what makes
   h512 / xLSTM / SAM (2× cost) affordable on one 1080.

**Cost**: one config line. **Verify on the box first**: that `arlstm` + NSE loss accept
`predict_last_n > 1` (the AR substitution loop is per-timestep; with holdout 0 it never fires, so it
should be inert — assert `max|Δy| = 0` against a `predict_last_n: 1` control at step 365, exactly
the check that validated [[the-no-nan-fused-path-63x]]).

**Falsifier**: 1-seed val1 vs the matched `nhar0h256` 1-seed (**0.879828**). ≥ +0.002 ⇒ a skill
lever, roll it across the AR family. Within ±0.002 ⇒ a **compute** lever only; keep it and spend the
savings on capacity. ≤ −0.002 ⇒ the shorter warm-ups hurt; drop to n = 10 once, then abandon.

### A2. ⭐ THE MISSING CELL OF THE 2×2 — `nharh256` (hidden 256 **+ holdout 0.5**)
The AR family has been explored on two axes and **one of the four cells was never run**:

| val1 asymptote `a` | holdout 0.0 | holdout 0.5 |
|---|---|---|
| **h128** | 0.888715 (`nhar0`) | 0.890130 (`nhar`) |
| **h256** | **0.895585** (`nhar0h256`) | ⬜ **NEVER RUN** |

Holdout is worth **+0.0014** on `a` at h128; capacity is worth **+0.0069** at holdout 0. If they are
even half additive the missing cell sits near **a ≈ 0.896 … 0.897** — above every member built.
The holdout is a **stochastic augmentation** ([[the-seed-depth-law-per-member]]: it buys seed
diversity and sells 1-seed level), so it is *expected* to look worse at 1 seed and better at 3+.

**Cost**: one flag on an existing, fully-tooled variant (`--holdout 0.5`), 3 seeds ≈ 7 h on the fused
path. **Cheapest shot in the document.**
**Falsifier**: fit its own `a` from 3 seeds ([[the-seed-depth-law-per-member]] — **its own** b, never
a borrowed one, the error made twice in L54); if `a ≤ nhar0h256`'s 0.895585, the axes are not
additive and the 2×2 is closed.

### A3. READ `nhar0h512` AND SCORE ITS FALSIFIER
Already pre-registered, already running (if the box lives). It is the last open L54 arm and it gates
whether "capacity" survives as an axis at all. Read it before spending anything else.

---

## 3. TIER B — THE ACTUAL NEXT-GENERATION SWINGS

### B1. ⭐⭐⭐ DIVERSITY-REGULARIZED TRAINING (Negative Correlation Learning)
**Why this one is different from everything already closed.** [[the-ensemble-sensitivity-map]] §4
measured the exchange rate: **removing 0.10 of a member's error correlation ≈ +0.029 of solo NSE**,
at breadth 1.000. Removing just **0.03** would buy **+0.00175 — the whole gap.** §6 then measured
that **18 of 19 members ever built here sit below break-even, and every axis tried moves a member
ALONG the skill-vs-decorrelation frontier, never OFF it.**

That is not a statement that diversity is worthless. It is a statement that **diversity has never
once been optimised directly** — every attempt (EA-LSTM, Mamba, IVT, MRMS, `nhar`) bought
decorrelation *by accident*, usually by fitting worse. NCL is the one family of methods that puts
the correlation term **in the loss** and optimises skill and diversity jointly:

```
L_k = NSE_k  +  λ · Σ_{j≠k} (ŷ_k − y)·(ŷ_j − y)        # j frozen
```

**The cheap sequential form** (no joint retraining): the other 8 members' predictions already exist
as dumps. Train member 9 with a penalty against their **frozen** errors. Cost = one member.

⚠️⚠️ **The failure mode is known and must be designed against.**
[*Joint Training of Deep Ensembles Fails Due to Learner Collusion* (arXiv 2301.11323)] shows joint
diversity objectives are routinely gamed: members manufacture *in-sample* disagreement that does not
generalise. And the frozen members' **training-window** errors are partly memorisation, so
decorrelating against them decorrelates against the wrong thing.
⇒ **Mandatory design**: hold out a slice of the training window (e.g. train the reference members on
1999–2006, compute their errors on 2006–2008, penalise correlation **on that slice only**).
⇒ **Mandatory gate, zero GPU, before the run**: confirm that the member's train-slice error
correlation and its val1 error correlation actually track each other. If they do not, the penalty
cannot transfer and the arm dies for free.
**Falsifier**: val1 error-correlation vs the other members must fall by **≥ 0.03** while solo val1
falls by **< 0.009** (the break-even rate, 0.311 solo per unit of correlation). Off the frontier or
it is arm #20 on it.

### B2. ⭐⭐ FORCING DROPOUT — DIVERSITY BY FORCING, WITHOUT THE SKILL COLLAPSE
[[nhar-won-on-skill-not-decorrelation]] §2 concluded: *"the route to 0.90 is AR members that are
better AND differ by FORCING."* L54 then tried the literal version — single-forcing AR members —
and it **died**: `nhar0` loses **−0.026** going 15 inputs → 5, where the MB family loses ~0.008.
The AR edge is multi-forcing-specific.

**Forcing dropout is the middle path the family skipped**: keep all 15 inputs, but randomly mask one
of the three forcing products (5 channels) per sequence, at rate p. The member stays multi-forcing
(keeps the edge) but cannot lean on any one product, and different seeds/rates lean differently.
⭐ It is also a genuine regulariser on a 9-year training window.

**Implementation**: NH's `random_holdout_from_dynamic_features` is built for AR inputs (NaN →
substitution). For ordinary dynamic inputs NaN is poison ([[nan-inputs-poison-nh-members]]) **and**
`_validate_samples` would drop the sample — so mask **to zero after normalisation** (= the channel
mean), ~10 lines in the dataset/collate path. **Assert the patch applied** — count masked channels
per batch and log it ([[assert-the-patch-applied]]).
**Falsifier**: p ∈ {0.15, 0.30} on 3 seeds; solo val1 must be ≥ `nhar0h256`'s within −0.002 **and**
the val1 error correlation against the other members must fall by ≥ 0.02. Both, or drop it.

### B3. ⭐ SHARPNESS-AWARE MINIMISATION (SAM / ASAM)
Flat-minimum optimisation, the most reliably transferable generalisation technique of the last few
years, and this is its best case: **9 years of training data, 531 basins, a 0.4-dropout network that
is squarely in the small-data regime.** It raises `a` (the asymptote) rather than chasing `b/k`,
which [[the-seed-depth-law-per-member]] says is the only thing that matters.
**Cost**: 2× per step (two forward/backward per update) — which is exactly what A1 pays for.
**Falsifier**: 1-seed val1 ≥ +0.002 over matched control at equal wall-clock (not equal epochs —
charge it its 2×). ρ ∈ {0.02, 0.05} only, chosen on val1, no sweep.
Pair with **schedule-free AdamW** (drop-in, removes the 1e-3/5e-4/1e-4 step schedule) only if SAM
passes — one variable at a time.

### B4. ⭐ MEASURED-σ NOISE ON THE ASSIMILATED DISCHARGE
The with-q track feeds the model **an observation with a known error distribution** — and this
campaign is the rare one that has actually measured it: σ from **144,914 USGS gaugings**
([[measured-gauge-sigma-and-the-incoherent-ceiling]], and `k = 1.04` variance inflation already
deployed). Today the network is trained on clean `q_mm_shift1` and therefore learns to trust it more
than it deserves; at inference the input carries that error.
**Technique**: perturb the lagged-discharge input during training with its *own measured*
per-basin/per-magnitude error. This is not generic noise augmentation — it is a **calibration** of
how much the model should lean on yesterday's gauge, and the holdout result already proves this
input's augmentation schedule moves `a` (+0.0014).
**Falsifier**: 3-seed `a` must exceed the matched no-noise control. If it only raises `b`, it is a
diversity knob, not a skill knob — report it as such and do not ship it as a member upgrade.

---

## 4. TIER C — ARCHITECTURE (the xLSTM question), NOW UNBLOCKED BUT DEMOTED

The L55 brief's unlock stands: with complete discharge the AR substitution never fires, so
`q_mm_shift1` as a **plain `dynamic_input`** opens `x_lstm`, `mamba`, `transformer`, `mclstm`… to
the with-q track. Two reasons it is **Tier C and not Tier A**:

1. **The 1/8 rule.** A new architecture is one member ⇒ it must be **+0.013 solo** better than
   `nhar0h256`. Nothing in the literature offers that: the strongest CAMELS transformer result
   ("Probing the limit of hydrologic predictability with the Transformer network") *matches* LSTM
   rather than beating it, and the architecture axis already closed negative here twice
   ([[multi11-mamba-FAILED-architecture-axis-closes]]).
2. **Two hard blockers, both measured/known**: the `xlstm` pip package is **not installed**, and its
   fast kernels generally want ≥ Ampere + Triton — the 1080 is **sm_61**. Expect a fallback path or
   a build failure. **Time-box the install + a speed check to one session.**

⚠️ And the val-window trap from the L55 brief applies to every plain-input member:
`_validate_samples` drops samples with NaN dynamic inputs, val1 discharge is ~5.5 % missing (mean
per-basin coverage 0.9448, 10.4 % of basins < 99 %), train/test are complete. **Print join sizes and
exclusive counts before believing any val1 delta** ([[a-stride-subsample-offsets-grids-across-corpora]]).
Run a `cudalstm` plain-input control against `nhar0` first; only then an exotic architecture.

---

## 5. SCREENED AND DEPRIORITISED (state them so they are not re-proposed)

| idea | why not |
|---|---|
| Loss reweighting toward the near-median cohort | **Closed.** [[near-median-cohort-is-UNAIMABLE-trackC-closed]] — train→test cohort lift 1.70× vs a 3.0× bar; the cohort cannot be identified in advance. Any per-basin reweighting is aiming. |
| Ensemble distillation into one student | A student fitted to the ensemble is maximally correlated with it ⇒ LOO ≈ 0 by [[the-ensemble-sensitivity-map]] §4. |
| More seeds / more weight on the AR family | **Closed twice.** Ensemble-level depth dead; paired 2×-weight = zero; and [[seed-depth-helps-the-member-and-can-hurt-the-ensemble]]. |
| Per-basin or online test-time finetuning | **Closed.** [[per-basin-finetuning-test-vs-honest]] — +0.002 on test, **−0.0045** honest, corr −0.013. |
| Cross-corpus pretraining, transfer | **Closed negative.** L40, L41. |
| Weight averaging / SWA / EMA | [[swa-redundant-with-seed-averaging]]; and seed-averaging a member now has a *measured* ensemble cost. |
| More lags of discharge | Non-lever: `q_mm_shift1` is present at **every** timestep of the 365-day sequence, so the model already ingests the full shifted hydrograph. |
| Target transform (√q, log q) | Loss variants closed on the MB family; low prior, no mechanism. |

---

## 6. RECOMMENDED SEQUENCING

1. **Reach the box. Read `nhar0h512`. Score the pre-registered falsifier.** (blocked, zero GPU once up)
2. **A2 `nharh256`** — the missing 2×2 cell. One flag, 3 seeds, biggest prior-per-GPU-hour here.
3. **A1 `predict_last_n`** — verify inertness against a control, then screen at 1 seed. Whichever way
   it lands (skill lever or compute lever) it is kept.
4. **B1 zero-GPU gate first** (does train-slice error correlation predict val1 error correlation?),
   then the NCL member if it passes.
5. **B2 forcing dropout** in parallel with 4 — it is the only arm that answers the "differ by
   FORCING" requirement without the single-forcing collapse.
6. **B3 SAM** once A1 has paid for its 2×.
7. **Tier C** only if 2–6 leave the gap open, and time-boxed.

**Unchanged rules**: pre-register predictions + falsifiers in `PREREG_v2.md` before any run; select
on **val1**; paired per-basin delta + basin-bootstrap 95 % CI + breadth; **3 seeds to screen, 5 to
ship**; equal weight, zero fitted combination parameters; test1 read once after val1 decides; and
quote the **paired** statistic, which has been right **7/7** against difference-of-medians.
