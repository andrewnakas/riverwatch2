# nwm-archive

NWM medium_range_blend operational forecasts in the v14.2 schema:
`(issued_date, station_id, target_date, horizon_day, q_cfs_raw,
q_cfs_obs_today, bias_scale_used, schema_version)`.

Two writers feed this branch:
1. `pages.yml` snapshot job — one row per build, accumulates live
2. `nwm-backfill.yml` — historical fill from `s3://noaa-nwm-pds/`

Consumed by `scripts/train_nwm_residual.py` to fit the v15.1
residual learner.
