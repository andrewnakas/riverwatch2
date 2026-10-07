# LEDGER 43 — BRANCH TABLE

**Written 2026-08-26, BEFORE the multi14 run was launched and therefore before
any result exists.** Every bar is stated in ENGLISH first, then as an
inequality, then checked against the sentence — the ledger-42 rule, because all
six of that ledger's specification errors had the shape *"the code did exactly
what I wrote, and what I wrote was not the question I meant."*

**Every anchor, bar and null below names the RECORDED ARTIFACT and the
COMPUTATION that produced it. A number in a `print()` is an expectation, never
an anchor.**

---

## 0. THE QUESTION

Does replacing the canonical **CAMELS-27** static attribute set with
**CAMELS-27 + 8 screened GAGES-II attributes** improve the no-q CAMELS-531
ensemble, under the matched Li/Song protocol?

This axis has never been a candidate in this campaign. `PREREG_v2.md` contains
no entry for the static attribute SET; `multi10` (EA-LSTM) changed how statics
are *consumed* and used the identical 27; cluster weighting used statics to
cluster, not as model inputs.

⭐ It is proposed because it is the only untested within-protocol axis whose
mechanism is **broad by construction** — statics condition every basin's
regionalisation — which is the one shape
`why-the-camels-noq-median-does-not-move` says can move a median. It is **not**
channel-adding, the family measured **anti-targeted** (3–7× tail-weighted).

**Honest prior, recorded in advance: ~15–20% to clear the train-side gate; ~10%
that this ledger sets a new record.**

---

## 1. ANCHORS — both reproduced EXACTLY before anything else was run

| anchor | value | artifact | computation | diff vs prereg |
|---|---|---|---|---|
| **TRAIN-side** | **0.9504577864064554** | `benchmarks/ledger43_anchor_train.json` | `noq_harness.py --mode baseline` | **−0.000000214** |
| **TEST-frame (the record)** | **0.8362893021622821** | `noq_test_result.json` (md5 pinned in `benchmarks/ledger43_s0_anchors.json`) | median over `per_basin_nse`, the **already-spent** 2026-08-09 query | **+0.000000302** |

⇒ Recorded in `benchmarks/ledger43_s0_anchors.json`. **No new query was spent** —
anchor 2 is arithmetic on per-basin NSEs recorded 17 days ago.

## 2. THE CONTROL COSTS ZERO GPU — and that is a deliberate finding, not an omission

Training is **bit-deterministic** (`training-is-bit-deterministic-check-before-budgeting-controls`,
n=2, md5-identical 23 days apart). Ledger 42 spent **~25 GPU-h — half its
budget — on controls that were byte-identical reproductions.**

`multi14` differs from `multi6` in **exactly one field** (`static_attributes`);
`dynamic_inputs` are identical (18 channels, asserted in
`scripts/preflight_multi14.py`) and the `time_series/` directory is **per-file
symlinked** to multi6's, so not one byte of forcing data differs.

⇒ **multi6's existing dumps ARE the control.** Pinned by md5:

```
a26b64f03f63c8abb4a586cb2b30746a  camels531ls_multi6_nhlstm_TRAIN_s111.csv.gz
ff6d8468a732f6aa526130e1df62a39d  camels531ls_multi6_nhlstm_TRAIN_s222.csv.gz
a0a33ec71d6edd1ef7f6c9c52fb0dbbb  camels531ls_multi6_nhlstm_TRAIN_s333.csv.gz
22c4d66fd86eae55bcedf3f307428c5a  cfgls_multi6_s111.yml
```

**GPU budgeted for controls: 0 h.**

## 3. STAGE 1 — PASSED (zero GPU). The screen, and what it excluded.

`analysis/ledger43_statics_screen.py` → `benchmarks/ledger43_s1_statics_screen.json`.

Validated against **three known answers before any candidate was read**: a copy
of a CAMELS-27 column → R²=1.00000; an exact linear combination of two → 1.00000;
pure noise → 0.04826.

**FILTER 1 — leakage.** Two attributes excluded as inadmissible under no-q:

- ⛔ `RUNAVE7100` — mean annual runoff 1971–2000: derived from **observed
  discharge**, and its period **overlaps the 1995–2008 test window**.
- ⛔ `BFI_AVE` — baseflow index from hydrograph separation on the gauge's **own
  observed record**; a summary statistic of the target series. Kratzert's
  CAMELS-27 contains no q-derived signature by deliberate design.

Seven further attributes excluded as duplicates of CAMELS-27.

**FILTER 2 — redundancy**, R² vs CAMELS-27 < 0.90, then greedily vs CAMELS-27 +
already-accepted (so a compositional family cannot smuggle in its last member).
⭐ The guard fired as predicted: **`HGD_PCT` rejected at R²=0.9068.**

**8 survivors:** `HGC_PCT` `HGB_PCT` `HGA_PCT` `DEVNLCD06` `EMERGWETNLCD06`
`AWCAVE` `WOODYWETNLCD06` `TOPWET` (R² vs CAMELS-27 = 0.362 … 0.874).

Falsifier (<3 survivors ⇒ stop, no GPU) **did not fire**. Stage 2 licensed.

## 4. STAGE 2 GATE — kill-fast, 1 seed (s111). STOP unless ALL SIX hold.

Primary mode is **swap vs `lstm_multi6`** (`score_swap.py --mode swap
--replaces lstm_multi6`, already run 7 times in this campaign), so the stream
count stays at **9** and the result is not confounded by "one more member" —
which matters because the greedy curve **peaks at 4** and the marginal member is
measured worth ≤ 0.

| # | the sentence | the inequality |
|---|---|---|
| 1 | *Scored on every lead, not one weekday in fourteen, the typical basin must gain at least +0.0003.* | `all_leads_paired_median_delta >= +0.0003` |
| 2 | *It must help more basins than it hurts, by a clear margin.* | `all_dates_breadth >= 0.45` |
| 3 | *The lucky-phase reading and the all-days reading must agree about the direction.* | `sign(h1_delta) == sign(all_leads_delta)` |
| 4 | *It must be worth more than simply adding one more seed to an existing stream.* | clears the marginal-seed band (**−0.00010 / 56.3%**, `seed_depth_value.py`) |
| 5 | *It must be a competent model in its own right, not a broken run.* | `solo_val_slice_median_nse >= 0.934653` — see the amendment below |
| 6 | *It must land where the median is actually set.* | `near_median_delta >= +0.0008` |

### ⚠️ AMENDMENT 1 — recorded BEFORE the run, and it is a FRAME correction

The inherited competence bar from the `multi10` prereg reads *"solo train-side
median NSE ≥ 0.790"*. **That number does not live on the train-side frame.**
0.790 is a **day-1 median NSE on the TEST window** (`solo_multi12.py` reads
`camels531ls_multi12_nhlstm_s111.csv.gz`, the 1995–2010 dump); the train-side
val-slice solos are ~0.92–0.95 (`multi5` 0.9486, `multi` 0.9463, **`multi6`
0.9447**, `noq_trainside_baseline.json`).

Applying 0.790 to a val-slice number would be **exactly ledger-42 error #5** —
predicting on one frame and measuring on another. Two frames, never mixed.

⇒ **Bar 5 is restated on the frame it is measured on:** multi14's solo val-slice
median NSE must be within 0.010 of its own control, i.e. **≥ 0.934653**
(multi6 solo **0.9446529029247024**, from `benchmarks/ledger43_anchor_train.json` key `solo_val.lstm_multi6`, minus 0.010). The bar's job is to catch a broken run, not to
require a strong solo member — `multi5` is a measurably *weaker* member that is
the campaign's only successful one.

⛔ This also means **no test-window dump is produced at Stage 2.** Nothing in
this stage touches 1995–2008.

### ⚠️ Two standing rules that bind this gate

- **h==1 never decides.** Ledger 42 measured h==1 as a 1-in-14 fixed-weekday
  subsample and the *lucky* phase (record 0.836289 h==1 vs **0.831422**
  all-leads). All-leads is the decision statistic; h==1 is reported for sign
  agreement only.
- **Gate on the effect size, never on distinguishable-from-zero.** A p=0.0385
  would have passed an arm whose lift was 1.70× against a 3.0× bar.

## 5. STAGE 3 — replication, only if Stage 2 clears (~10 GPU-h)

Seeds s222, s333; the **seed-average is the shippable unit**.

⭐ **Pre-registered discriminator:** `multi8` and `multi9` both produced a
seed-average **below either constituent seed** (errors adding rather than
diversifying) — the exact inverse of `multi5` (+0.0023, above every seed). If
that signature appears it predicts failure *before* the ensemble gate, and I
will say so at the time rather than after.

## 6. STAGE 4 — rotated-slice confirmation (zero GPU)

The 1990–95 val slice is a **reused surface**. Re-fit and re-score on the
rotated split (**fit 1980-88 + 1993-95, validate 1988-93**); sign and rough
magnitude must hold. The θ/λ sweep looked like a winner on the grid and
**reshuffled on rotation** — a selection artifact. Log the candidate in
`PREREG_v2.md`'s gate-shopping ledger; the count of screened candidates travels
with the result.

## 7. STAGE 5 — the query. NOT SPENT WITHOUT EXPLICIT USER APPROVAL.

Licensed only if Stages 2–4 all clear. Both frames reported — h==1 **and**
all-leads. The record's measured asterisk travels with any new number.

## 8. THE BRANCHES — written now, so none can be chosen after the fact

| branch | condition | what is written |
|---|---|---|
| **A** | Stage 2 fails any bar | *The static-attribute axis is closed on one seed.* Record the deltas, add the axis to the closed list, **stop**. The 8 survivors were legal and non-redundant, so this is a real closure of a real axis, not a screening artifact. |
| **B** | Stage 2 clears, Stage 3 seed-average below its constituents | *The multi8/multi9 signature.* Report the discriminator fired; do not proceed to Stage 4. |
| **C** | Stages 2–3 clear, Stage 4 sign flips on rotation | *Selection artifact on the reused val surface.* No query. |
| **D** | Stages 2–4 all clear | Ask the user to spend the query. Report both frames whatever it returns. |

⚠️ **A validity check gates INTERPRETATION, not EXECUTION.** Nothing here aborts
before computing the thing the budget was spent on.
