# Ledger 41 — G3 DECISION TABLE (pre-registered)

**Written 2026-08-20, while both arms were still training. No G3 number existed.**
Agreed between two independent sessions; each holds a copy.

## The bar

| item | value |
|---|---|
| frame | TRAIN-side val slice, **from 1990-10-01** (`gate_eval` `CV_VAL_START`) |
| control | `camels531ls_daymet_nhlstm_TRAIN_s{111,222,333}` = 0.902412 / 0.917969 / **0.919630** |
| **PRIMARY BAR** | **complete seed separation** — `min(treatment) > max(control)` ⇒ beat **0.919630** |
| power precondition | **≥3 seeds per arm**; fewer ⇒ `INCONCLUSIVE (underpowered)`, never PASS |
| tripwire | Δ > **+0.02** ⇒ re-audit leakage, do not celebrate |

⛔ **Never** substitute paired median or breadth for the bar: on pure seed noise
they read **+0.0110** (CI excluding 0) and **69.9%**. A basin bootstrap cannot
see seed variance.

⛔ **Three frames, never crossed**: G2b full-train-period (0.5739/0.6063) ·
G3 val slice (from 1990-10-01) · seed-**averaged** control median 0.926178,
which sits *above* all three per-seed medians. Descriptive stats are
seed-averaged; **the bar is on per-seed medians**.

## The four outcomes

### 1. DOUBLE PASS — both priors separate
The **only** outcome that licenses Stage 5. **Stage 5 then runs
unconditionally**, and the hard stop moves to **Stage 5 → Stage 6**.

⭐⭐ **Reachability does not merely cap Stage 5's value — it changes what Stage 5
IS.** Stage 5 was specified as *"retrain the 4-member subset from the new init"*
(multi5 ×5, multi6 ×3, lstm_daymet, lstm_nldas). But a **5-input daymet-shaped**
prior cannot initialise multi5 or multi6 **at all** (15+ inputs), and lstm_nldas
would need its own nldas-shaped finetune plus the `tmax==tmin` semantics problem.
**Nothing is left to retrain.** The only stream this prior reaches is
`lstm_daymet`, and **G3 has already produced it — three seeds of it.**

⇒ For this prior, Stage 5 is **not a GPU stage**. It is: substitute the
finetuned `lstm_daymet` dumps into the frozen 9-stream inverse-MSE combination,
**refit weights on train rows**, score on the **train frame**. CPU-only, minutes,
on dumps that already exist.

⭐ **So it must not be gated on the near-median predictor** — the predictor
exists to avoid spending GPU on a member that will not move the median, but here
the predicted quantity can simply be **MEASURED, for free**. *Measuring beats
predicting whenever measuring is cheap.* ⚠️ And the predictor's 9-for-9 record
was earned on members being **ADDED** to the ensemble; this is a member being
**REPLACED** by a better version of itself. The mechanism plausibly transfers,
but leaning on a validated record **outside the operation it was validated on**
is unnecessary when the direct measurement costs nothing.

⇒ On a null we can say *"the member improved and the ensemble did not move, by
this much"* — a considerably better sentence than *"we predicted it would not."*

**THE HARD STOP — Stage 5 → Stage 6.** Both conditions, no substitutions:

| condition | value |
|---|---|
| recombined train-frame ensemble ≥ | **0.954458** (anchor **0.950458** + 0.004) |
| near-median Δ on the recombined ensemble ≥ | **+0.0008** |

**Fail either and the single test query is NOT spent.**

The bar is a **pre-registered number, not a phrase** — "beats the anchor by more
than the seed-noise floor" would have been an unspecified bar of exactly the
shape this ledger keeps catching. Frame, unchanged from the anchor's definition:
weights fit **1980-10-01 … 1990-09-30** (127,606 rows), scored on the val slice
**1990-10-01 … 1995-09-30** (69,030 rows, 531 basins) — the *same* val slice G3
uses, under ONE GLOBAL inverse-MSE vector, θ=4.0, λ=0.25.

⚠️ **Must not drift**: recombination is on the **TRAIN frame**, weights refit on
**train rows only**. Nothing in Stage 5 touches 1995-2010. The embargo is
unchanged and the query remains exactly one, at Stage 6.

### 2. SPLIT — one prior passes, one fails
**INCONCLUSIVE FOR THE RECIPE CLAIM.** The pair is confounded three ways —
**depth (ep15 vs ep7), seed, and torch build** — so a split licenses only the
ledger-40-level statement *"this particular prior beats random init"*, **not**
*"domain-aligned pretraining on gauge-calibrated forcings transfers."* The
second claim is the entire reason a second prior exists.

**Does NOT license Stage 5.**

**Optional attribution arm — ONE named artefact, fixed now: `p222/ep7`.**
Fires **only if** the passing margin is worth attributing (a passer clearing
0.9196 by 0.001 gets **no** ablation). Finetune-only; no pretrain.

⭐ Both split cases collapse to the same checkpoint, which is why it can be
*named* rather than *derived after the result*:

| case | reasoning | artefact |
|---|---|---|
| **A** p111/ep7 passes, p222/ep15 fails | ablate the **failer** down to the passer's depth | **p222/ep7** |
| **B** p222/ep15 passes, p111/ep7 fails | `p111/ep15` **does not exist** (s111 holds 1..13), so ablate the **passer** down to the failer's depth | **p222/ep7** |

`p222` is the only run spanning both depths (1..22 vs 1..13). Reading: **p222/ep7
passes ⇒ depth carried it; fails ⇒ seed or build carried it.** Holds seed, torch
build, data order and recipe fixed; varies depth alone.

⚠️ 22 and 13 checkpoints are a **menu**, and choosing one after seeing G3 is a
garden of forking paths. Naming `p222/ep7` in advance removes the menu: there is
nothing to choose and no result that makes another checkpoint look attractive.

### 3. DOUBLE FAIL — neither prior separates
**A clean, publishable result, written with the same confidence as a pass.**
Both priors clear own-domain (G2a) and zero-shot (G2b) yet neither improves the
endpoint over random init ⇒ **the prior is COMPETENT but CONTRIBUTES NOTHING**
at 531 basins × 15 years. That is a **data-rich regime** where "pretraining buys
nothing" is the *expected* finding, not a disappointing one. It closes the last
open axis in the strategy brief with a real answer rather than ledger 40's
confounded one.

### 4. UNDERPOWERED — fewer than 3 seeds in either arm
`INCONCLUSIVE`, never PASS. Escalate seeds; do not score the survivors.
⚠️ If a seed dies overnight, scoring the two that worked would silently be a
**2 v 3** comparison reported as PASS. The gate now refuses.

## Escalation — decided ONCE, BOTH ARMS OR NEITHER

Fires **iff** separation fails **and** seed-averaged paired median
Δ > **+0.0157** ⇒ **+2 seeds per arm**, re-read at **5 v 5**.

⚠️ **The 5v5 read SUPERSEDES the 3v3 read FOR BOTH ARMS.** An already-passing
arm is re-read too and **can lose its pass**. Escalation **replaces** the
verdict; it does **not** top up the loser.

⛔ **Never escalate one arm.** In a split the failing arm would qualify, and
escalating only it converts a split into a double pass by choosing — after the
fact — the analysis that rescues the arm that needed rescuing. That is the worst
forking path available here **because it looks like following the protocol**.

Expensive by design (four more finetunes) so it fires on the pre-registered
statistic and nothing else.

## Scope, declared in advance and declined

G3 compares **endpoints at a matched 30-epoch budget**, so it cannot see data-
or compute-**EFFICIENCY** (same endpoint, fewer epochs or basins), which is often
the real benefit in the transfer literature. If G3 nulls, an efficiency benefit
**could still be real** — and it is **explicitly declined**: this campaign needs
*skill on a fixed benchmark* and it *has* GPU. Recorded before the result so a
null cannot be walked back into *"but maybe it trains faster."*

## Wording that must survive into any writeup

> **The zero-shot number measures the prior's COMPETENCE. It does not measure
> the prior's CONTRIBUTION.**

G2b measures competence (0.5739 / 0.6063 — a model that never saw CAMELS
predicts it at ~0.60; a real result). G3 measures contribution (endpoint margin
over random init; **may be ~0**). On the G3 frame the zero-shot arms read
**0.601427 / 0.618649** against a control at ~0.91 — the prior starts **~0.30
below** the control it must beat.

⛔ The trap sentence: *"the prior already reaches 0.60 zero-shot, and finetuning
takes it to 0.9x."* The control reaches ~0.91 **from random init, starting from
nothing** — the head start does **not compound** and is **not a floor**. Two
paths converge on the same plateau; G3 asks only whether one lands slightly
higher. The 0.30 gap is **not a deficit to make up**; it is evidence that most of
the level is learned **from CAMELS itself in either arm**.

## Test-window embargo

The l41 **test dumps exist from 15:52 / 16:24 on 2026-08-20 onward and are
WRITE-ONLY until Stage 6.** The protocol allows **one** test-window read, at
Stage 6, on the recombined ensemble. If a test number is ever computed before
then it is recorded as a **SPENT QUERY**, not quietly discarded — *"I looked but
didn't use it"* does not survive contact with a writeup. As of this writing
neither session has computed anything from them.

---

## REPORTING RULE — added 2026-08-21, CHANGES NO VERDICT

Added after `p222/s333` destabilised mid-finetune (min 0.02609 @ep12 → 0.03646
@ep14, plateau ~0.0307, ending ~60% above its two siblings' 0.01916 / 0.01922).
All three **controls were clean** — every one monotone to its minimum at epoch
30, last/min = 1.000.

⛔ **The seed is NOT dropped, NOT re-run, NOT re-rolled with clipping.** The bar
is unchanged. This rule governs only what gets **recorded**.

When p222's three per-seed medians exist, state explicitly whether the **other
two** clear 0.919630:

| observed | verdict | what the record must say |
|---|---|---|
| other two clear, only s333 does not | **arm FAILS** (unchanged) | the **binding seed destabilised mid-finetune**, so this arm is **not a clean test of the prior** |
| none of the three clear | **arm FAILS** (unchanged) | the destabilisation is **irrelevant** to the verdict — a **clean** failure, and the **stronger** result. Say so plainly. |

⇒ This is the difference between *"the prior failed"* and *"we cannot tell
whether the prior failed."*

## ⛔ AN EXPECTATION I HAD TO STRIKE

I wrote that s333 *"will very likely set that minimum and make the arm harder to
pass."* **Struck.** That runs from **TRAIN LOSS** to **SCORED MEDIAN**, and this
campaign's standing rule is that an in-run training loss is **near-uninformative
about scored value** — it has misled here repeatedly, and ledger 40's two arms
agreed to **five significant figures at epoch 5** before diverging **0.168 NSE**
([[ledger40-RESULT-transfer-collapsed-not-a-clean-test]], [[my-causal-claims-keep-failing]]).

Correct wording: **s333 MAY set the minimum; its scored median is not yet known.**
A destabilised run can end flatter and *better* regularised on held-out days as
easily as worse.

⚠️ **Why this matters practically**: carrying the expectation into scoring means a
p222 failure feels *explained the moment it appears* — with an explanation
written **before** the evidence. And if s333's scored median lands in line with
its siblings, the train-loss event is interesting and **irrelevant to the gate**,
an outcome the struck framing made hard to see.

## ⚠️ A LIMITATION OF THE ESCALATION RULE — recorded, NOT fixed

Escalation is the pre-registered remedy for a contaminated arm and is
outcome-independent (it triggers on the delta statistic, not on anyone's opinion
of s333). **But a dragging seed LOWERS the seed-averaged paired median delta**,
so the **+0.0157 trigger is LESS likely to fire precisely when the contamination
is worst.**

⛔ **Deliberately NOT changed.** The pre-registration is closed, and changing a
trigger *after seeing which way it cuts* is the worst available edit.

⇒ **"Escalation did not fire" must NOT be read as "the arm was cleanly tested."**

## ⛔ CROSS-ARM TRAIN-LOSS COMPARISON IS INADMISSIBLE

Control lr `{0: 1e-3, 20: 5e-4, 25: 1e-4}`; treatment lr
`{0: 1e-4, 20: 5e-5, 25: 1e-5}` — **10× lower throughout** (the pre-registered
ledger-40 choice). Treatment seeds ending ~2× the control's train loss is
substantially **that**, not evidence about the prior. G3 reads **scored median
NSE**, so the LR difference does not touch the gate at all — and **cross-arm
train-loss comparison must not appear in the writeup**. Only the **within-arm**
comparison (s333 against its own siblings) is clean.
