#!/usr/bin/env bash
# Extract AORC basin 13235000 for WY1981-2010, 4 years in parallel.
#
# Parallel by YEAR is safe here, unlike the CONUS404 tile parallelism that
# produced Errno 5 storms. AORC publishes one zarr store PER YEAR
# (1981.zarr, 1982.zarr, ...), so workers touch entirely disjoint objects.
# The CONUS404 failures came from 6 workers contending on chunks of a single
# store.
#
# Runs on the laptop and against a different bucket (noaa-nws-aorc) than the
# box's CONUS404 extraction (hytest), so the two do not interfere.
#
# Resume is automatic: aorc_exact_extract.py skips a year whose CSV exists.
set -u
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=/tmp/aorcvenv/bin/python
OUT=data/aorc_extra
mkdir -p "$OUT" logs

echo "=== 13235000 EXTRACTION START $(date +%F_%T) ==="
for grp in "1981,1985,1989,1993,1997,2001,2005,2009" \
           "1982,1986,1990,1994,1998,2002,2006,2010" \
           "1983,1987,1991,1995,1999,2003,2007" \
           "1984,1988,1992,1996,2000,2004,2008"; do
  tag=$(echo "$grp" | cut -d, -f1)
  nohup $PY aorc_exact_extract.py --basin 13235000 --years "$grp" --out "$OUT" \
    > "logs/ex13235000_${tag}.log" 2>&1 &
done
wait
echo "=== DONE $(date +%F_%T): $(ls $OUT/13235000_WY*.csv 2>/dev/null | wc -l)/30 years ==="
