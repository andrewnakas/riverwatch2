# Ledger 42 — DECISION TABLES (pre-registered)

**Written 2026-08-21, before any ledger-42 number existed.** Companion to the
LEDGER 42 registration in `EXPERIMENTS.md`. Bars are numbers, never phrases.

## THE THREE FRAMES — enumerated once, never crossed

| frame | definition | weights | anchors (must reproduce) |
|---|---|---|---|
| **F1** | h==1, merge (station_id, target_date), test window 1995-10-01…2008-12-21, 531 basins | frozen production inverse-MSE vector, θ=4.0 λ=0.25, fit on TRAIN rows (the 0.8363 vector) | equal 0.8298 · frozen **0.8363** · train-val 0.950458 |
| **F2** | ALL-LEADS: per-basin NSE over the union of (target_date, h) rows, h=1..14, seed-averaged then combined | **the SAME frozen F1 weight vector** — F2 is a *sample* change, never a *configuration* change. A per-frame refit may be reported as a secondary descriptive line, clearly labelled, never as the F2 statistic | train-side 9-stream all-leads 0.953930 (from rescore9) |
| **F3** | continuous-daily. Exists **iff** S0.2 confirms stride-14 tiling with no lead overlap on the TEST dumps — in which case **F2 ≡ F3 by construction** (each target_date appears exactly once) and the stride-14 question is resolved without a separate score | n/a | n/a |

⛔ **dHBV composition rule (registered at S0.2 time, before any test number):**
F2 requires **all 9 streams at every row**. If any test dump lacks leads 2–14 it
is re-dumped from existing checkpoints (B3, inference only, write-only). If a
stream **cannot** be re-dumped, F2 is recorded as **UNDEFINED** and Stage Q
reads q1 only. Per-row weight renormalisation over available streams is
**prohibited** — that is a new combination rule, and the combination axis is
closed (84 rules, none positive).

## NOT-A-QUERY — registered definitions

1. Metadata-only reads of test dumps: columns (station_id, t0/date, h) —
   **no truth, no predictions**. Logged in the ledger with the column list.
2. Reprocessing of `analysis/noq_test_result.json` (per-basin NSEs from the
   already-spent 2026-08-09 query). B2 lives entirely here.
3. `gzip -t` / row-count / basin-count integrity checks on dumps.
4. **Observed discharge (`truth`) alone from the test window, with NO model
   predictions of any kind in the same computation.** The observation ceiling is
   a property of the GAUGE, not of any model — it is identical whatever we
   predict, and it cannot reveal how well we did. Precedent:
   `analysis/noq_ceiling_corrected.py` already computed it, and σ itself was
   measured from 144,914 USGS field measurements, independent of every dump.
   ⛔ The moment a prediction column enters the same script, class 4 no longer
   applies and it is a SPENT QUERY. B2 combines ceilings with the **already-spent**
   per-basin NSEs from `noq_test_result.json` (class 2) — never with a fresh
   prediction read.

Anything else that touches 1995-2010 truth or test predictions is a **SPENT
QUERY** and gets recorded as one — "I looked but didn't use it" does not
survive contact with a writeup (ledger-41 wording, kept).

## TRACK A SCREENS — pass bars

### A0 — multi-forcing settling test (licenses ONLY the A3 corpus build)

| bar | value |
|---|---|
| sample | shared HYSETS∩CAMELS gauges, **n ≥ 500** (else INCONCLUSIVE — the 10-gauge smoke test sign-flipped) |
| A0-1 event-day disagreement inflation, pretrain triple {SCDNA, Livneh, QC-stations} | same sign as CAMELS triple's **+27%** AND within **[0.5×, 2×]** of it (i.e. median inflation ∈ [+13.5%, +54%]) |
| A0-2 cross-triple disagreement coupling | median per-gauge corr(pretrain-triple disagreement series, CAMELS-triple disagreement series) ≥ **0.3** |
| verdict | PASS = both bars; FAIL ANY → **A3 closed at zero GPU** |

#### ⬜→⛔ A0 STATUS (2026-08-22): BLOCKED, and independently LOW-VALUE — stated before it is skipped

**Blocked:** A0 needs **three separate** pretrain-side precip series
{SCDNA, Livneh, QC-stations}. The surviving ledger-41 artifact
`gpu1080/nh_data_l41` holds a **single FUSED** 5-variable set
(`prcp, tmax, tmin, vp, srad`, 1979-01-01…1995-09-30) — precip from SCDNA and
temp from Livneh **already merged**, with the rejected Livneh precip discarded.
The raw HYSETS `.nc` files are on neither box. Running A0 therefore requires a
**fresh ~2–4 h HYSETS download**, not a CPU job on existing data.

**And it is low-value independent of the blocker** — recorded so the skip is a
reasoned decision, not a blocker used as cover:

1. A0's PASS licenses **only the A3 corpus build** — not GPU, not an arm. It is
   two full steps from any number that could move the record.
2. **A3 sits in the regime ledger 41 already answered.** Ledger 41's licensed
   claim is *"a competent prior does not improve a **data-rich** target"*
   (531 basins × 15 yr). A3's target is **that same target**. The *mechanism*
   differs (inter-slot disagreement vs a 5-input prior), but the *contribution*
   question is the one that just returned a measured **cost** of 0.015–0.078
   across 6/6 seeds and 2 independent priors.
3. Even a full success reaches streams carrying 62% of weight, and AMENDMENT 3's
   arithmetic shows what weight buys: at 0.131 a stream needs **~+0.03** to move
   the ensemble +0.004.

⇒ **A0/A3 are DEFERRED, not refuted.** ⛔ Nothing here licenses the sentence
*"multi-forcing pretraining fails"* — it has **not been tested**. The honest
statement is: *it is blocked on a data re-download, and its most likely outcome
is the one ledger 41 already measured, so it is not what the remaining effort
should buy.* If it is ever run, A0's bars above stand unchanged.

### A1 — magnitude referee + conjunction screen (licenses ONLY a GPU arm for that candidate)

Events: per basin, top-1% observed-flow days, 1981-01-01…1995-09-30 (train side
only). Event depth = candidate precip summed t−3…t. Event response = observed
quickflow volume (ΣQ over event window minus pre-event baseflow). Referee is
**streamflow, never another forcing** (Maurer realignment: forcing-to-forcing
agreement had a 63% false-positive rate).

#### ⚠️ AMENDMENT 1 (2026-08-21, BEFORE any A1 number existed)

As first written, A1-1 was `Δ corr(event depth, event volume)`. **A correlation
is scale-invariant** — it is invariant to any global rescaling and, if ranked,
to any monotone transform at all. That is the *exact* defect that makes the
ledger-41 timing referee blind to magnitude, so the original A1-1 would have
rebuilt the blindness this screen exists to remove. Amended before any
candidate was screened; the original wording is preserved here so the change is
visible rather than silent.

**A1-1 is now decided by a CALIBRATION RUN against known truth**, not assumed:

- Candidate statistics, both magnitude-aware and both invariant to a *global*
  precip scale factor (which a model can learn) while sensitive to *per-event*
  magnitude error:
  - **S1 = CV of the event runoff ratio** (event quickflow volume ÷ event precip
    depth), across events within a basin. **Lower is better.**
  - **S2 = R² of event volume ~ event depth through the origin.** Higher is
    better.
- **Calibration**: run both on the five products whose member skill is already
  measured — daymet 0.7542, maurer 0.7453, nldas 0.7336, aorc 0.7047,
  **conus404 0.5094** (pure model-output precip, the known catastrophic
  failure).
- **Pre-committed selection rule, written before the calibration ran:** A1-1 is
  whichever of S1/S2 (i) ranks **conus404 last** and (ii) ranks **daymet
  first**. If both qualify, S1 is A1-1 and S2 is reported alongside. **If
  NEITHER qualifies, A1 is declared an INVALID SCREEN, no candidate is screened
  with it, and that is recorded as the result** — a screen that cannot recover a
  known ordering may not be used to license GPU.
- ⚠️ The calibration is a **necessary, not sufficient** check: reproducing a
  known ranking shows the statistic is not blind, not that it predicts skill.

| bar | value |
|---|---|
| sample | **n ≥ 500** basins with ≥ 10 events (else INCONCLUSIVE) |
| A1-1 magnitude gain | paired per-basin Δ on the CALIBRATION-SELECTED statistic vs camels_daymet: median favours the candidate AND win share ≥ **55%** |
| A1-2 information vs noise | split-half (odd/even events) Δ reproducibility: signal ≥ **3×** its split-half noise |
| A1-3 non-redundancy | event-day R² < **0.9** against EVERY existing precip input (daymet, nldas, maurer) — defends against re-deriving daymet from its own GHCN base |
| verdict | PASS = all three (conjunction); FAIL ANY → candidate closed at zero GPU |

Timing referee (`ledger41_forcing_referee.py`) runs as a **secondary check
only**: a candidate that degrades timing (paired < **−0.01** at n≥500) fails
regardless of A1-1..3.

#### ⚠️ AMENDMENT 2 (2026-08-21, after the A1 calibration, before any candidate was built)

The A1 calibration returned **REFUSED** (n = 465–482 < 500) and, visibly, would
have returned **INVALID SCREEN** at full n: the statistic separates the known
catastrophic product enormously but **cannot order the four gauge-calibrated
products by skill**, which is precisely the near-equal discrimination that
screening a candidate against daymet requires.

**Decision, recorded before any candidate exists:**

1. **A1-1 and A1-2 are DEAD as a gate.** No candidate is screened on them and no
   result of theirs licenses or blocks anything. The calibration outcome is a
   *result*, and it is recorded as one.
2. **A1-3 (redundancy) SURVIVES and remains mandatory.** It needs only a
   threshold, never discriminative power: **event-day R² < 0.9 vs daymet**. A
   construction that merely re-derives its own base is pointless, and the
   station corpus shares GHCN provenance with daymet, so this guard is the one
   that actually earns its place. A candidate failing A1-3 is closed at zero GPU.
3. **Track A proceeds directly to the CONTRIBUTION gate.** ⛔ Not a relaxation:
   the GPU-gate bars are **unchanged** (3v3 complete separation · recombined
   ≥ 0.954458 · near-median Δ ≥ +0.0008). The pre-GPU screen existed to
   **save GPU**, and GPU is explicitly **not** the constraint (user: no cap, run
   until answered). The campaign's own governing rule — *screen on CONTRIBUTION,
   the endpoint margin, never on component quality* — says the endpoint gate is
   the valid criterion regardless.
4. ⚠️ **What this costs, stated plainly:** without a working pre-GPU screen,
   a failed candidate costs GPU-hours instead of CPU-minutes. That is the
   accepted trade, it is the user's stated preference, and every hour lands in
   the all-in cost table.

#### ⚠️⚠️ AMENDMENT 3 (2026-08-21, before any GPU) — THE HOST STREAM CHANGES, and the REACHABILITY CEILING is stated in advance

The gate as first written put the treatment in the **multi5** recipe. Inspecting
`gpu1080/cfgls_multi5_s111.yml` shows **multi5 already carries `prcp_stn`**
(plus `snowf_stn`, `snowd_stn`, `obs_mask_stn`) — 19 inputs, not 15. A
station-derived precip construction substituted there would be a **re-derivation
of a channel the model already has**, i.e. the closed "better fusion of existing
inputs" class. Worse, multi5's gain was measured **not to track the station data
at all** (r = −0.034; zero-coverage basins gain equally) — it works by
PERTURBATION, not information. multi5 is the wrong host.

**Host is now `lstm_multi`** — the pure 15-input multi-forcing stream, verified
to carry **no station channels** — with the **daymet precip slot** replaced by
the candidate construction. Weight in the frozen vector: **0.1309** (vs
lstm_daymet 0.0612). Everything else — seeds, controls, bars — is unchanged.

⭐⭐ **THE REACHABILITY ARITHMETIC, recorded BEFORE the result** (the ledger-41
lesson: a treatment that reaches one stream of nine cannot move the record even
if the mechanism is real):

> `lstm_multi` carries **0.1309** of the ensemble weight. In the naive linear
> limit, moving the recombined ensemble by the required **+0.004** needs that
> single stream to improve by roughly **+0.03 NSE** — an enormous move for a
> stream already at ~0.80, and ~13× the largest member gain this campaign has
> ever banked (multi5, +0.0023 on the ensemble).

⇒ **A Track-A pass is therefore unlikely on the CONTRIBUTION bar even if the
construction genuinely improves the stream.** This is written down now so that a
stream-level improvement followed by an ensemble-level null is read as **the
predicted outcome of the weight arithmetic**, not as a surprise or a
disappointment — and so that a *stream* gain is never quietly reported as if it
were a *record* gain. The bars do not move to accommodate this.

## TRACK A GPU GATE — any arm that passes A1-3

Treatment: multi5 recipe, daymet-precip slot substituted with the screened
construction, **3 seeds** (s111/s222/s333). Controls: **3 fresh random-init
multi5 seeds trained in the same session** — the existing multi5 dumps are the
**null source** (S0.3), not the control arm (torch-build/session confounds).

| item | value |
|---|---|
| frame | TRAIN-side val slice from 1990-10-01 (`gate_eval` CV_VAL_START), per-seed medians |
| **PRIMARY BAR** | **complete seed separation**: min(treatment) > max(control), **≥3 seeds per arm**; fewer ⇒ INCONCLUSIVE (underpowered), never PASS |
| tripwire | Δ > **+0.02** ⇒ leakage audit before anything else |
| destabilised seed | NOT dropped, NOT re-run; ledger-41 reporting rule applies verbatim (state whether the other two clear the bar) |
| escalation | fires iff separation fails AND seed-averaged paired median Δ > **+0.0157** ⇒ +2 seeds **both arms**, 5v5 **supersedes** 3v3 for both, a passing arm can lose its pass |
| train loss | never evidence; cross-arm train-loss comparison inadmissible |

**CONTRIBUTION STOP — separate and binding (competence ≠ contribution, ledger 41):**
recombine into the frozen 9-stream (treatment 3-seed average replacing the
multi5 5-seed average), refit ONE GLOBAL inverse-MSE vector θ=4.0 λ=0.25 on
train rows (1980-10-01…1990-09-30), score the val slice (1990-10-01…1995-09-30):

| condition | value |
|---|---|
| recombined train-frame median ≥ | **0.954458** (anchor 0.950458 + 0.004) |
| near-median Δ (±0.01 window) ≥ | **+0.0008** |

**Fail either ⇒ the candidate does NOT ship and does NOT appear in Stage Q.**
Both bars are measured, not predicted — measuring beats predicting when
measuring is free.

## STAGE Q — the single test query

One execution of `analysis/score_noq_test42.py`, **user-approved immediately
before running**, after every track is terminal and this table is frozen.
Emits ONE JSON containing:

- **q0** control: `--mode equal` F1 — must reproduce **0.8298** or the query is
  VOID (bug, not verdict) and the JSON is quarantined as a spent-void query.
- **q1** frozen 9-stream, F1 — must reproduce **0.8363** (same VOID rule).
- **q2** frozen 9-stream, F2 (iff F2 defined).
- **q3** iff a Track-A candidate shipped: the recombined configuration, F1 and F2.

### Branch readings — written before any number exists

| branch | pre-written reading |
|---|---|
| **q3(F1) ≥ 0.84** | **SKILL CLAIM — the only branch that moves the record.** New record = q3(F1); q3(F2) reported as secondary. |
| q3 exists, q3(F1) ∈ (0.8363, 0.84) | Real improvement, target not crossed. Record updates to q3(F1); no 0.84 claim. |
| q3 exists, q3(F1) ≤ 0.8363 | The shipped member does not transfer held-out. Record stands. Contribution gate's train-side pass recorded as a failed transfer — stated plainly. |
| **q2 ≥ 0.84 AND q1 < 0.84** | **MEASUREMENT FINDING, never a skill gain**: the fixed-weekday 1-in-14 subsample understates the same model's daily-frame skill; 0.84 is crossed under F2 only. Headline stays q1 (F1); the F2 sentence is: "scored on every day rather than one weekday in fourteen, the same frozen model reads X." |
| q2 ≥ 0.84 AND q1 ≥ 0.84 | F1 record updates (q1 > 0.8363 would itself be surprising — frozen config, VOID rule applies if it moves beyond reproduction tolerance ±0.0005). |
| q2 ∈ [q1, 0.84) | Phase lever real but insufficient; **closed**, spread reported. |
| **q2 < q1** | The 4-LSTM direction generalizes held-out: all-leads scores LOWER; phase lever **closed negative**; h==1 was the lucky phase and the record keeps its asterisk. |
| F2 undefined (B3 impossible) | q1 only; frame question recorded OPEN with the blocker named. |

**In every branch:** q1 (F1, h==1) is reported; per-basin bootstrap CIs are
reported as context, **never as pass bars** (a basin bootstrap passed on pure
seed noise — registered-invalid design). σ-scenario sentence from B2 attaches
to whichever branch fires: under CENTRAL σ the achievable-median statement is
made alongside the number, not instead of it.

## STOP RULE — evidential, replaces a GPU cap (user decision 2026-08-21)

Ledger 42 ends when ALL of:
1. Track B verdicts exist: frame/stride resolution (B1 + S0.2) and test-frame
   σ-headroom (B2).
2. Every Track-A candidate is **terminal**: failed A0/A1 at zero GPU, failed
   the GPU gate or contribution stop, or shipped.
3. Stage Q is spent and its branch read.

No arm may be added after `score_noq_test42.py` freezes. A failed construction
re-enters only as a NEW construction through A1 with a fresh registration
appended. All-in GPU-h table (productive + discarded, split) maintained from
the first launch.
