#!/bin/zsh
# Phase 0 forcing-archive fetch queue (sequential; every stage resumable —
# fetchers skip inits whose csv.gz already exists). Ordered so the data that
# unblocks Phase 2/3 soonest lands first. All outputs are SD-card symlinks.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/fetch_ens_queue.log

echo "=== stage 1: ECMWF ENS mean+members (2024-04..2025-12, weekly) $(date)" >> $LOG
caffeinate -i $PY scripts/fetch_ens_forcings.py --model ecmwf \
  --start 2024-04-01 --end 2025-12-29 --stride-days 7 \
  --members 10 --members-start 2025-01-01 --members-end 2025-12-31 >> $LOG 2>&1

echo "=== stage 2: GEFS mean+members 2025 (weekly) $(date)" >> $LOG
caffeinate -i $PY scripts/fetch_ens_forcings.py --model gefs \
  --start 2025-01-06 --end 2025-12-29 --stride-days 7 \
  --members 10 --members-start 2025-01-01 --members-end 2025-12-31 >> $LOG 2>&1

echo "=== stage 3: GEFS mean backfill 2021-2024 (weekly) $(date)" >> $LOG
caffeinate -i $PY scripts/fetch_ens_forcings.py --model gefs \
  --start 2021-05-03 --end 2024-12-30 --stride-days 7 >> $LOG 2>&1

echo "=== stage 4: GFS 2026 daily (blend panel) $(date)" >> $LOG
caffeinate -i $PY scripts/fetch_gfs_forcings.py \
  --start 2026-03-10 --end 2026-06-30 --stride-days 1 \
  --out-dir data/mblstm/gfs_fcst_2026 >> $LOG 2>&1

echo "=== stage 5: HRRR 2026 daily (blend panel) $(date)" >> $LOG
caffeinate -i $PY scripts/fetch_hrrr_forcings.py \
  --start 2026-03-10 --end 2026-06-30 --stride-days 1 \
  --out-dir data/mblstm/hrrr_fcst_2026 >> $LOG 2>&1

echo "=== fetch queue complete $(date)" >> $LOG
