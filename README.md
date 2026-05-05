# snodas-archive

Per-station NOHRSC SNODAS daily SWE/melt extracts produced by
`.github/workflows/snodas-backfill.yml`. Each `data/snodas_extracts/{station_id}.json`
maps ISO date → `{swe_in, melt_24h_mm}`.

Schema and source documented in `app/snodas.py` on `main`.
