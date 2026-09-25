#!/usr/bin/env python3
"""Extract the GAGES-II (Falcone 2011) workbook sheets we need to cached CSVs.

The published archive ships one 49 MB xlsx; openpyxl re-parses it in ~40 s per
run, so every consumer would pay that cost. Cache the handful of sheets the
ledger-41 pool and static-imputation work touch as plain CSV instead.

Sheets and why:
  BasinID      gauge coords + drainage area  -> leakage dedup, area screen
  Bas_Classif  Ref/Non-ref, ecoregion, HYDRO_DISTURB_INDX -> pool ranking
  FlowRec      per-water-year record flags   -> the Q-record census (Probe A),
               which removes the need for a bulk NWIS inventory query entirely
  Soils/Topo/Climate/Geology/LC06_Basin      -> features for imputing the 27
               production statics onto pretrain basins (Probe C)
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data" / "cache" / "gages2" / "basinchar_and_report_sept_2011.zip"
MEMBER = "gagesII_sept30_2011_conterm.xlsx"
OUT = ROOT / "data" / "cache" / "gages2" / "sheets"
SHEETS = ("BasinID", "Bas_Classif", "FlowRec", "Soils", "Topo", "Climate",
          "Geology", "LC06_Basin", "Hydro", "Bas_Morph")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [s for s in SHEETS if not (OUT / f"{s}.csv").exists()]
    if not missing:
        print(f"all {len(SHEETS)} sheets already cached in {OUT}")
        return
    if not ZIP.exists():
        sys.exit(f"missing {ZIP} (fetch water.usgs.gov/GIS/dsdl/basinchar_and_report_sept_2011.zip)")

    wb = openpyxl.load_workbook(
        io.BytesIO(zipfile.ZipFile(ZIP).read(MEMBER)), read_only=True, data_only=True)
    for name in missing:
        ws = wb[name]
        rows = ws.iter_rows(values_only=True)
        hdr = [c for c in next(rows)]
        # trailing empty header cells appear in several sheets (Soils, FlowRec)
        while hdr and hdr[-1] is None:
            hdr.pop()
        n = len(hdr)
        path = OUT / f"{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr)
            count = 0
            for r in rows:
                if r[0] is None:
                    continue
                w.writerow(["" if v is None else v for v in r[:n]])
                count += 1
        print(f"{name}: {count} rows x {n} cols -> {path}")


if __name__ == "__main__":
    main()
