# RiverWatch2 vs NOAA NWM v3 medium-range — per-lead head-to-head

**The first public per-lead skill table for operationally-issued NWM v3
medium-range forecasts at USGS gauges.** No such reference exists in the
literature (NWM medium-range verification is either paywalled, retrospective
open-loop, or aggregate-only) — so we measured it, against our own system,
on real issued forecasts.

- **Window:** 2026-03-09 → 2026-07-06 issuances (29 strided issue dates,
  stride 3; known archive gap 2026-04-05→05-01), one t00z
  `medium_range_blend` cycle per day.
- **Cohort:** 229 USGS gauges = NWM archive ∩ corpus_openmeteo encoder
  history (74,158 scored forecast-obs pairs; grows as the corpus trickle
  fills — re-run to refresh).
- **Truth:** USGS daily values; only finite observations scored.
- **NWM baseline (`NWM corrected`):** archived `q_cfs_raw` × bias scale
  reconstructed from trailing h=1 forecast-vs-obs error (`bias_scale_used`
  is empty in the archive) — i.e. NWM *with* a fair post-hoc bias correction,
  not raw NWM.
- **RW2 mblstm:** shipped 4-seed CMAL ensemble (cmalv2p, Vincentized), mean
  point, no anchoring, decoder forcing = archived GFS-2026 + HRRR overlay
  leads 1–2. Checkpoints never saw 2026 data — zero leakage.
- **RW2 blend:** plain mean of mblstm + the production NWM-residual member.
  CAVEAT: the residual member's pickles were trained through 2026-06-10,
  inside most of this window — leak-advantaged in its favor. The clean
  member is mblstm.
- **Metrics:** per-station station-median MAE (cfs), win rate vs
  NWM-corrected, per-station median NSE/KGE across issue dates per lead
  (cells under 20 scored dates → unscorable, reported honestly as —).

| lead | NWM corrected MAE | RW2 mblstm MAE (win%) | RW2 blend MAE (win%) | NWM NSE | mblstm NSE | blend NSE | mblstm KGE | scorable |
|------|------------------:|----------------------:|----------------------:|--------:|-----------:|----------:|-----------:|---------:|
| 1 | 183 | 72 (95%) | 52 (100%) | 0.435 | 0.817 | 0.906 | 0.831 | 217 |
| 2 | 208 | 108 (91%) | 94 (96%) | 0.271 | 0.701 | 0.795 | 0.749 | 216 |
| 3 | 217 | 125 (92%) | 110 (96%) | 0.136 | 0.596 | 0.669 | 0.700 | 216 |
| 4 | 225 | 116 (92%) | 114 (96%) | 0.042 | 0.568 | 0.626 | 0.670 | 216 |
| 5 | 239 | 129 (91%) | 126 (95%) | -0.094 | 0.495 | 0.523 | 0.626 | 216 |
| 6 | 235 | 148 (90%) | 142 (93%) | -0.111 | 0.465 | 0.446 | 0.598 | 216 |
| 7 | 219 | 147 (91%) | 142 (95%) | -0.184 | 0.407 | 0.417 | 0.578 | 216 |
| 8 | 260 | 150 (89%) | 150 (94%) | -0.223 | 0.324 | 0.337 | 0.535 | 215 |
| 9 | 284 | 174 (87%) | 174 (93%) | -0.259 | 0.302 | 0.340 | 0.469 | 214 |
| 10 | 267 | 171 (86%) | 181 (92%) | -0.331 | 0.228 | 0.273 | 0.429 | 214 |
| 11 | 254 | 133 (87%) | 132 (93%) | — | — | — | — | 0 |
| 12 | 259 | 137 (84%) | 129 (91%) | — | — | — | — | 0 |
| 13 | 232 | 133 (83%) | 120 (91%) | — | — | — | — | 0 |
| 14 | 241 | 136 (82%) | 123 (90%) | — | — | — | — | 0 |

Headlines:

- **RW2 beats bias-corrected NWM at every one of 14 leads** on station-median
  MAE, with win rates 82–100% of stations.
- **Bias-corrected NWM's median NSE goes negative from lead 5** — beyond
  ~4 days it is worse than predicting each gauge's mean. RW2 still carries
  NSE 0.50 at lead 5, 0.41 at lead 7, 0.23 at lead 10.
- Discharge assimilation (the encoder sees the gauge's recent observations)
  is the structural advantage: at lead 1 the blend's NSE is 0.906 vs NWM's
  0.435.
- A parallel run with the pre-CMAL frozen ensemble (gfsft, median point,
  `blend_2026_panel.json`) lands within noise of the shipped config on this
  GFS-forced panel — the CMAL ensemble's measured edge (EXPERIMENTS row 13)
  is ECMWF-forcing-specific.

Additional caveats for any external use: single forecast cycle per day
(t00z); NSE/KGE at leads 11–14 need a longer archive (cells fall under the
20-date floor once USGS provisional-lag losses hit); cohort is
corpus-availability-selected, not a stratified sample (it grows daily);
NWM archive covers h1–14 of the blend product only.

Reproduce: `RW2_ENABLE_MBLSTM=1 .venv/bin/python scripts/backtest_blend_2026.py
--archive-dir /tmp/rw2-backtest/nwm-archive/archive --stride-days 3
--ckpt <cmalv2p ckpts> --point mean --label panel_cmalv2p`
(source JSONs: `blend_2026_panel.json`, `blend_2026_panel_cmalv2p.json`;
experiment registration: EXPERIMENTS.md row 18).
