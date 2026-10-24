# m1-forcing-archive

Per-basin daily forcing for the MODERN-1 served member, maintained by
the `m1-forcing` job in `.github/workflows/pages.yml` and consumed by
`app/modern_forcing.load_table()`.

`data/m1_forcing/{gauge_id}.csv.gz`:
`date` + temperature_2m_mean/_max/_min (degC), precipitation_sum (mm),
shortwave_radiation_sum (MJ/m2/day), vapor_pressure (Pa), `src`.

`src` records the provenance of each row. `om_point` is Open-Meteo at
the gauge coordinate, which GATE 1 measured costs -0.059534 paired
median NSE against the NLDAS-2 areal forcing the member trained on.

`data/m1_forcing/_built.json` records WHEN each basin was last
refreshed. It exists because freshness cannot be read off a row date:
the table carries ~16 forecast rows, one of them always dated today,
so a `newest row <= today` test is satisfied by a table nobody has
touched. That is not hypothetical -- it froze this table for five
days in October 2026 while every published number looked current.

`data/m1_forcing/_served_skill.json` is the public label, rebuilt
here from the forcing actually published so it always names the
configuration it describes.

Single commit, force-pushed each run: this is current state, not a
history. The forecast record lives elsewhere.
