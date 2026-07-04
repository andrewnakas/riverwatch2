# CAMELS-US subset report

Gauge-id lists: `data/camels_gauge_ids.json` (671 full CAMELS-US, 531 Kratzert et al. subset).

| config | subset | n overlap | median NSE | median KGE |
|---|---|---:|---:|---:|
| baseline_sota_frozen | all | 1758 | 0.506 | 0.607 |
| baseline_sota_frozen | 671 | 92 | 0.473 | 0.504 |
| baseline_sota_frozen | 531 | 59 | 0.515 | 0.538 |
| ens4ft_ecmwf_full | all | 1758 | 0.540 | 0.615 |
| ens4ft_ecmwf_full | 671 | 92 | 0.506 | 0.500 |
| ens4ft_ecmwf_full | 531 | 59 | 0.575 | 0.556 |
| cmalv2p_ens4v_ecmwf_full | all | 1758 | 0.558 | 0.636 |
| cmalv2p_ens4v_ecmwf_full | 671 | 92 | 0.515 | 0.541 |
| cmalv2p_ens4v_ecmwf_full | 531 | 59 | 0.580 | 0.584 |
| ens4ft_perfect | all | 1758 | 0.577 | 0.619 |
| ens4ft_perfect | 671 | 92 | 0.603 | 0.562 |
| ens4ft_perfect | 531 | 59 | 0.637 | 0.593 |

Overlap: 59/531 and 92/671 CAMELS basins are in this corpus of 1758 stations.

Caveats — read before comparing to published CAMELS numbers:

- Our metrics are pooled-horizon 1-14-day *forecast* NSE/KGE on a 2025
  backtest window. Published CAMELS benchmarks (e.g. Kratzert et al.
  median NSE ~0.74-0.76 on the 531 subset) are *simulation* NSE with
  observed/analysis forcing over the full test period — a much easier task.
- Our train/test protocol is not the CAMELS protocol: different training
  years, different test years (2025), and our corpus is 1,758 CONUS gauges
  chosen for operational relevance, not the CAMELS basin set.
- The perfect-forcing run (ens4ft_perfect) is the only quasi-comparable
  config: it removes weather-forecast error, so it is the closest analog to
  published CAMELS *simulation* benchmarks (still pooled over 14-day
  horizons and still a different train/test split).
- Overlap is partial: only the intersection counts below are scored, not
  the full 531/671 lists, so medians are not computed over the same basin
  population as the published numbers.

