# External Streamflow-Forecasting Benchmark Landscape (2026-07)

Compiled 2026-07-04 via multi-agent web research; 13/14 load-bearing claims
verified against primary PDFs. Labels: [SIM] = simulation w/ observed
forcings; [FCST] = true forecast w/ NWP forcings; [PSEUDO-FCST] = mixed.

## Reference numbers to position against

| Benchmark | Type | Number | Source |
|---|---|---|---|
| CAMELS-531, LSTM ens n=8 (2019) | SIM | median NSE 0.76 | Kratzert HESS 23:5089 |
| CAMELS-531, LSTM ens n=10 3-forcing (2021) | SIM | median NSE 0.82 | Kratzert HESS 25:2685 |
| CAMELS-531, LSTM+dHBV grand ens (2025) — record | SIM | ~0.83 | Li/Shen HESS 29:6829 |
| Klotz CMAL (test 1995-2005, non-std) | SIM | 0.784 | HESS 26:1673 |
| NWM v2.0 reanalysis on CAMELS | SIM | 0.62 | Frame 2021 JAWRA |
| NWM v2.1 retrospective, 5,390 gauges | SIM | median KGE 0.53 | Towler HESS 27:1809 |
| AIFL (ECMWF 2026), 2,003 gauges global, IFS-forced, 2021-24 | FCST | median NSE 0.53, KGE' 0.66 | arXiv:2602.16579 |
| Google vs AIFL, 1,218 shared stations | FCST | KGE' 0.678 vs 0.636 | same |
| LSTM vs NWM medium-range, 42 basins | FCST | +0.14 NSE over 7-day horizon | White AIES/NASA 2025-26 |
| HEFS probabilistic day 14 | FCST | CRPSS ~0.05-0.10 | Demargne BAMS 95 |
| GloFAS reforecast day 10 (proxy-verified!) | FCST* | CRPSS 0.55 vs persistence | Harrigan HESS 27:1 |
| CAMELS-GB LSTM | SIM | median NSE 0.88 | Lees HESS 25:5517 |

2023-2026 challengers (transformers, S4D, TS foundation models) all fail to
beat the LSTM ensemble on CAMELS; the community converged on "predictive
limit" language (Liu 2024 J.Hydrol 637:131389).

## Where RiverWatch2 stands today (2026-07-04, shipped config)

- Real-forcing (ECMWF ens-mean), 1,758 US gauges, pooled 1-14d, 2025:
  median NSE 0.558, KGE 0.636 — ABOVE AIFL's global forecast-mode 0.53
  (different gauge set; directionally strong).
- Perfect-forcing: 0.577 (0.637 on the 59-basin CAMELS-531 overlap) vs
  CAMELS SIM record 0.83 — NOT protocol-comparable (pooled leads, years,
  basins); a strict CAMELS run is required for a leaderboard number.
- vs NWM: no public per-lead NSE verification of NWM medium-range exists
  (Cosgrove 2024 paywalled). Our 2026 panel already beats bias-corrected
  NWM at every horizon on 107 gauges.

## The gap (verified)

No established benchmark covers gauged, 1-14-day, US, daily streamflow
forecasts verified against gauge observations with real archived NWP
forcings at issue time. Closest infrastructure: Caravan MultiMet (Nov 2024,
archived IFS-HRES/GraphCast leads 1-10d, no leaderboard, no 11-14d leads).
The RiverWatch2 honest harness (1,758 gauges, 2025 issue dates, archived
ECMWF/GFS/GEFS forcings, pre-registered gates) fills it.

## Ranked plays (effort -> credibility)

1. CAMELS-531 strict protocol run [SIM]: train Oct1999-Sep2008, test
   Oct1989-Sep1999, basin-mean forcings, n>=8 ensemble, median NSE.
   Direct line to 0.76/0.82/0.83.
2. NWM v3 medium-range head-to-head at our 1,758 gauges on 2025 issue
   dates [FCST]: we would PRODUCE the missing public reference.
3. AIFL/Google comparison on Caravan-MultiMet US subset, 2021-24 [FCST].
4. Klotz CMAL protocol (natural fit for our head) [SIM].
5. RFC forecasts via IEM HML archives [FCST] (messy; stage-flow ratings).
6. Publish our benchmark: protocol + forcing archives (HF dataset) +
   NWM/persistence/climatology baselines + leaderboard.

Full agent report with all URLs in the session transcript; key sources:
hess.copernicus.org/articles/23/5089/2019, 25/2685/2021, 26/1673/2022,
29/6829/2025; nature.com/articles/s41586-024-07145-1; arXiv:2602.16579;
arXiv:2411.09459 (MultiMet); sciencedirect.com/science/article/pii/S2589915526000027.
