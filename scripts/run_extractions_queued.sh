#!/bin/zsh
# Serialized SD extractions, queued behind ALL heavy jobs (2026-07-06).
# First attempt ran two unzips beside the B1 trainer: their filecache
# pressure pushed swap 11→14.8 GB, internal disk to 1.9 GB, and the kernel
# memory-killer silently reaped both unzips (zero files extracted, empty
# logs). Rule: no bulk SD extraction while a torch job is resident.
# Fires after the flood-F1 follow-up (last link in the armed chain:
# B1 ens8 → aifl527 backtest → flood-F1), then runs the two unzips one at
# a time.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
LOG=logs/extractions_queue.log
echo "=== extractions queued $(date)" >> $LOG
until [ -f benchmarks/flood_f1_aifl527_chained.json ]; do
  sleep 900
done
while pgrep -f "train_mblstm.p[y]" > /dev/null \
   || pgrep -f "backtest_mblstm.p[y]" > /dev/null; do
  sleep 600
done

echo "extracting maurer+nldas $(date)" >> $LOG
cd /Volumes/STORAGE_SD/riverwatch2_data/camels_raw
unzip -n -q basin_timeseries_v1p2_metForcing_obsFlow.zip \
  "basin_dataset_public/basin_mean_forcing/maurer/*" \
  "basin_dataset_public/basin_mean_forcing/nldas/*" >> $LOG 2>&1
N=$(find basin_dataset_public/basin_mean_forcing/maurer \
      basin_dataset_public/basin_mean_forcing/nldas \
      -name "*.txt" 2>/dev/null | wc -l | tr -d ' ')
echo "camels extraction done: $N files $(date)" >> $LOG

echo "extracting HRES zarr $(date)" >> $LOG
cd /Volumes/STORAGE_SD/riverwatch2_data/external/caravan_multimet
unzip -n -q HRES.zip >> $LOG 2>&1
echo "HRES extraction done $(date)" >> $LOG
echo "=== extractions complete $(date)" >> $LOG
