# LEDGER 44 — BRANCH TABLE

**Written 2026-08-27, BEFORE any ledger-44 number was computed.**
Plan: `~/.claude/plans/cheerful-orbiting-milner.md`. Supersedes nothing; ledger 43
closed Branch A on 2026-08-27.

## THE BAR, IN ENGLISH FIRST

> Ledger 44 ships a number only if the improvement, measured on the held-out
> Li/Song window at h==1 AND at all-leads, exceeds three times the measured
> resolution floor of a member-swap delta on that window.

Inequality: **`Δ_heldout ≥ +0.003` on both frames, signs agreeing.**
Anything in `[0, +0.003)` is REPORTED AS MEASURED and NOT claimed as skill.

Check of the inequality against the sentence: the resolution floor measured
2026-08-27 is a bootstrap half-width of **±0.00103** for a paired-median
member-swap delta on the full held-out window (4-stream ensemble, one-member seed
swap, a true null, h==1). 3 x 0.00103 = 0.0031 ≈ +0.003. ✅ consistent.

## WHY THE BAR IS NOT THE OLD +0.001 SHIP BAR

The old ship bar (+0.001) and the ledger-43 gate bar (+0.0003) were both written
on the TRAIN-side surface. That surface is **inside the training window**
(`analysis/check_split.py:31`: train `01/10/1980 -> 30/09/1995`; the harness scores
`1990-10-01 -> 1995-09-30`). Measured in-sample inflation, daymet s111, h==1:
**0.9024 in-sample vs 0.7634 held-out = 0.139 NSE.** Bars written on that surface
do not transfer to the surface the record is claimed on, and S2 measures by how much.

## S1 BRANCHES — the held-out ensembling law. Read `a - current` ONCE.

`a` = asymptote of the fit `NSE(n) = a - b/n` on the HELD-OUT window;
`current` = the shipped ensemble's held-out score at its present net count.

| outcome | condition | action |
|---|---|---|
| **A** | `a - current >= +0.003` | pure compute clears the bar. Fire S3. This is the ledger's answer. |
| **B** | `+0.001 <= a - current < +0.003` | real but sub-bar. Run S3 anyway (GPU unbounded, sign is a law not a hypothesis), report as measured, **do NOT claim skill**. |
| **C** | `a - current < +0.001` | seed/ensemble depth is genuinely saturated held-out. **Do not spend GPU on seeds.** Go to S2, then close. |
| **D** | held-out greedy still rising at stream 9 while the in-sample greedy peaks at 4 | "ensemble size peaks at 4" is REFUTED. Reopen member-count; S2 decides which rejected candidates deserve a re-look. |

D is not exclusive with A/B/C; it is read from the same curve and recorded separately.

**Pre-registered prior (recorded before the number exists):** most likely **B**,
possibly **C**. Evidence: the 4-forcing equal-weight family fits
`NSE(n) = 0.834004 - 0.040461/n`, whose asymptote **0.8340 is BELOW the shipped
record 0.8362893**, which says the shipped ensemble is already past that family's
ceiling. Outcome **A** would be a genuine surprise and must not be assumed.

## S2 — NO-SHIP CLAUSE. Recorded before any candidate is scored.

S2 reads the held-out window for candidates that have **already been rejected**
(`multi8`, `multi9`, `multi14`, `multi5b`, `multidrop`, `multi10`, `multi11`).

1. **No candidate may be shipped, reopened, or re-seeded on the basis of this read.**
2. Its sole purpose is to estimate how a TRAIN-side gate delta maps to a HELD-OUT delta.
3. If a rejected candidate reads positive held-out, that is **selection on test**.
   It is recorded as such and NOT acted on.
4. The deliverable is the **residual scatter** of the regression of held-out delta
   on train-side delta — the train-side gate's resolution expressed in held-out NSE.

Cost correction found at S0.1: TEST dumps **already exist** for 6 of the 7
(`multi8` x2 seeds, `multi9` x2, `multi10`, `multi11`, `multi5b`, `multidrop` x2).
Only `multi14` is TRAIN-only. S2 therefore costs **one ~17-min inference**, not the
~2 GPU-h budgeted.

## S3 — fired only by S1 branch A or B

- N seeds per stream chosen from the S1 fit so predicted residual gap <= 20% of headroom.
- `clip_gradient_norm: 1` on every run.
- Divergence criterion: *"rose above its running minimum and never returned"* —
  NOT a per-step ratio, and never a mid-run loss comparison between arms.
- Every new seed screened for the `multi_s333` failure mode (solo held-out NSE >= 0.75)
  before entering any average. `multi_s333` reads **0.5604** vs ~0.805 for its siblings
  and is already excluded by `_MULTI_WEAK_SEEDS` at `analysis/gate_eval.py:83`.

## STOP RULES

- Both anchors must reproduce from recorded artifacts before any other number is
  reported: train-side **0.950458**, record **0.8362893021622821**
  (`analysis/noq_test_result.json`, md5 `c4d7639df8647308e48dedc9f45f824e`).
- The scorer must reproduce a KNOWN answer (`multi8`, `gate_multi8_swap_2seed.json`,
  12 decimals, 5 quantities) before being pointed at anything unknown — and the
  anchor passed must match the CONFIGURATION being globbed (1-seed and 2-seed
  anchors differ by 30%).
- Weight fitting stays on TRAIN rows in every arm. It is the fourth frame.
- Every number reported on BOTH frames (h==1 and all-leads).

---

# AMENDMENT 1 — S1b, THE BREADTH CURVE. Recorded 2026-08-27 BEFORE it was run.

## Why an amendment

S1 measured **depth** (more seeds of the same 9 recipes) and bounded it at
**+0.0016..+0.0023** held-out — Branch B, below the +0.003 bar. That asymptote is
a property of THIS ensemble family. It says nothing about **breadth**: more
DISTINCT recipes move the asymptote itself.

Branch **D** of the main table ("held-out greedy still rising while the in-sample
greedy peaks at 4") was never computed, and S0.1 found the reason it now can be:
**20 extra streams exist with matched seeds on BOTH surfaces**, every one of them
already trained, every one rejected on the in-sample surface. Cost to measure: **zero GPU.**

This is the one remaining thing that could clear +0.003, and it is exactly the
axis the in-sample surface is measured to understate (2.32x on depth).

## The construction, fixed before any number is read

1. **Pool** = streams with matched TRAIN/TEST seed sets, minus the shipped 9:
   `aorc, cmal, conus404, gmm, l41ft, l41zeroshot, multi4, multi5b, multi8,
   multi9, multi10, multi11, multi12, multi671, multi730, multi_eps05,
   multibagb, multidrop, multih128, multih512`.
   Seed-mismatched tags (`multi5swa`) are excluded — a train and test stream that
   average different members makes every train->test comparison silently invalid.
2. **Competence screen, TRAIN-SIDE ONLY**: solo median NSE on the val slice
   **>= 0.90**. This is the campaign's existing band — the one that excluded
   `_MULTI_WEAK_SEEDS` ("final training-period validation NSE must be >= ~0.90").
   It exists to keep a diverged run out of an average (`multi_s333` reads 0.5604
   held-out), and it uses NO test information.
3. **Ordering**: descending TRAIN-side solo median NSE. A train-side criterion,
   so the ordering carries no test information.
4. Production rule at every step: one global inverse-MSE vector, theta=4.0,
   lam=0.25, **weights refit on TRAIN rows only**.

## The readout, and what may be claimed

- **PRIMARY: the take-ALL configuration.** All 9 + every pool member that passes
  the train-side screen. This involves **no selection whatsoever**, so it is a
  single pre-specifiable configuration and it is shippable.
- **SECONDARY: the full curve** k = 9 .. 9+N. Reported in full. ⛔ **No point on
  the interior of the curve may be shipped** — choosing where to stop, having
  seen held-out scores, is selection on test.

| outcome | condition (take-all, vs record 0.836289) | verdict |
|---|---|---|
| **D-PASS** | `>= +0.003` on **both** h==1 and all-leads, signs agreeing | a defensible claim. Breadth is NOT exhausted; the in-sample greedy was wrong. |
| **D-PARTIAL** | `> 0` but `< +0.003` on either frame | reported as measured, **not claimed as skill**. Records that breadth is real but sub-bar. |
| **D-FAIL** | `<= 0` | breadth is exhausted held-out too. Combined with S1, the ensemble route closes entirely. |

**Pre-registered prior:** D-PARTIAL. The in-sample greedy found adds 5-9 to sum
negative, and the held-out understatement factor measured on depth is 2.32x,
which applied to an in-sample marginal of roughly -0.0003 per add does not by
itself produce +0.003. But the pool members are DISTINCT RECIPES, not extra
seeds, and their decorrelation is larger than a seed's — so the depth
understatement factor is a lower bound here, not an estimate. **D-PASS is
possible and is not being assumed.**

⚠️ Note what a D-PASS would NOT be: it would not be a new *member*, and it would
not overturn any individual member's rejection. It would say that a pool of
individually-rejected members is collectively worth more than the in-sample
surface could see.

---

# AMENDMENT 2 — the FULL-FRAME re-read. Recorded 2026-08-27 BEFORE it was run.

## The problem it fixes

The take-all configuration scores on the pool's common frame: **166,663 rows /
531 basins**, vs the record's **183,195 rows / 531 basins**. Basin coverage is
identical, but ~9% of dates are lost, so the take-all number is **not directly
comparable to 0.8362893021622821** and the honest comparator has been the
shipped-9 re-scored on the same reduced frame (0.836748).

## The change, and its justification, fixed before the number

Drop **`aorc`** — and only `aorc` — from the take-all pool. The reason is
**frame coverage, not score**: AORC's record starts later than the Li/Song test
window and it is the only kept stream that truncates the frame (the other ten are
`multi`-family and share the grid). This is stated before running, and it is the
same reason `ALLOW_SHORT_JOIN` exists in `noq_harness.py`.

⚠️ To be explicit about what would make this illegitimate: dropping a member
because it scored badly would be selection on test. `aorc` is dropped because of
its **date coverage**, a property of the corpus knowable without any score — and
its measured contribution is **+0.000043**, so this is not a scoring decision by
any reading.

## What is reported

**BOTH** configurations, always, on **both** frames:
- take-all (20 streams, reduced frame) — the AMENDMENT 1 pre-registered config;
- take-all-minus-aorc (19 streams) — reported on whatever frame it yields, with
  the row count stated, and compared to the shipped 9 **on that same frame**.

The bar is unchanged: **paired Δ ≥ +0.003 on both h==1 and all-leads.** If the
19-stream config lands on the full 183,195-row frame, its shipped-9 comparator is
the record itself and the comparison becomes direct.
