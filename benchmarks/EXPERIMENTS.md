# Experiment Log — Maximum-Accuracy Campaign (2026-07)

Append-only. Every experiment pre-registers its gate BEFORE the run completes.
North star: median NSE on the honest harness (1,758 stations, 2025 window,
real archived forecast forcings), plus KGE / log-NSE / FHV / approx-CRPS /
PICP90. Frozen reference: `mblstm_backtest_baseline_sota_frozen.json`
(NSE 0.506, FHV −50.9, CRPS 125.2, PICP90 ~0.79, hybrid HRRR+GFS forcing,
unanchored, median point).

Protocol: stride-3 (~586 stations) for screening, full 1,758 for confirmation.
Never compare across stride/window bases (compare_backtests.py enforces).
Anchored (as-served) and unanchored both reported for headline runs.

| # | date | experiment | config | gate (pre-registered) | result | verdict |
|---|------|-----------|--------|----------------------|--------|---------|
| 1 | 07-01 | P0 perfect-forcing re-baseline | ens4ft, perfect forcing, full | (measurement, not gated) — sizes the forcing gap: ≥0.15 keeps forcing work top priority | **NSE 0.577, FHV −48.2, CRPS 112.2** → real gap = 0.577−0.506 = **0.071** | **gap < 0.08 → re-weight to model/data levers.** FHV barely moves under perfect forcing (−48 vs −51): peaks are a MODEL problem → CMAL is the lever. The 0.742 "oracle" was pilot-ckpt fiction. |
| 2 | 07-01 | P0 stride-3 refs | ens4ft, gfs / gfs+hrrr?, str3 | (references, not gated) | gfs 0.5098, hybrid 0.5043 (586 st) | HRRR overlay confirmed neutral-to-negative at stride-3 |
| 3 | 07-01 | P1 anchor×point sweep | offline sweep over hybrid str3 dump; decay∈{0,1,2,3,4,7} × {median,mean3,blend.1/.2/.3} | adopt best decay if full-scale anchored NSE ≥ +0.01 vs unanchored; point ships only if FHV ≥ +5 pts at NSE cost ≤ 0.005 | **anchoring strictly hurts** (median: 0.504 unanchored vs 0.494–0.499 anchored; unanchored h1 ratio already 0.80). blend0.1: FHV +4.7 @ NSE −0.005, KGE +0.032 — borderline fail | **No anchoring for MB-LSTM** — encoder assimilation beats the mechanical anchor; production decay_h=2 on this member is a candidate REMOVAL (verify on 2026 panel). Point stays median; FHV fix rides on CMAL (exp 6). |
| 4 | 07-02 | P2 forcing A/Bs | ens4ft str3: gefs / ecmwf / ecmwf+hrrr? / gefs+hrrr? vs gfs ref | +0.015 NSE str3 → full confirm +0.01, no tercile regression; else kill product swaps | pending | — |
| 5 | 07-0x | P3 forcing-mixture ft | 4 seeds ft from base s101-104, --forcing-mix perfect/gfs/gefs | ens NSE ≥ 0.53 full (vs 0.510); kill < +0.01 | pending | — |
| 6 | 07-0x | P3 CMAL v2 | warm-start from mixture seeds, same mix, 12-16 ep, 4 seeds | CRPS −5% AND FHV +8 pts, NSE within −0.01 of mixture-quantile → ship; kill if ≥0.05 NSE cost at equal CRPS | pending | — |
| 7 | 07-0x | P3 asinh-free ablation | 1 seed, per-station standardization w/o asinh | FHV +8 pts at NSE ≥ −0.005 else discard | pending | — |
| 8 | 07-01 | CMAL v2 PILOT (early de-risk of exp 6) | s101 warm-start from gfsft, mix perfect:0.4,gfs:0.6, noise 0.15, 14 ep; eval str3 GFS vs s101ft quantile ref | CRPS −5% AND FHV +8 pts, NSE within −0.01 (same as exp 6, single-seed read) | pending | — |
| 9 | 07-02 | P4 blend panel 2026 (build) | mblstm (no anchor, median) + nwm_corrected + nwm_residual + persistence on NWM-archive window; cohort = corpus_openmeteo ∩ panel | blend beats best single member's median MAE at ≥10/14 horizons (cohort caveat until trickle fills) | **blend_mean2 (mblstm+nwm_residual) best/near-best at every horizon** (h1 86 vs resid 100; h5 164 vs 200; h14 208 vs 284). mblstm alone dominates h≥12 (h14 197 vs 284, −30%). 107 stations, 29 issue dates; nwm_residual leak-advantaged → wins conservative | **PASS — production ship of MB-LSTM member + blend justified** (scripts/backtest_blend_2026.py, benchmarks/blend_2026_panel.json). Cohort grows with corpus trickle; re-run to firm up |
