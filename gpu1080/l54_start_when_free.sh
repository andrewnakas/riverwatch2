#!/usr/bin/env bash
# LEDGER 54 — start a lane only when the GPU has capacity.
# Counts REAL CUDA processes via nvidia-smi (pgrep -f counts DataLoader workers
# too and reads 3x high — [[pgrep-f-self-matches]] family).
# usage: l54_start_when_free.sh <max_gpu_procs> <lane> <variant:seed> ...
set -u
cd ~/riverwatch2
MAX="${1:?max_gpu_procs}"; LANE="${2:?lane}"; shift 2
LOG="logs/l54_lane_${LANE}.log"
echo "=== LANE $LANE WAITING for < $MAX gpu procs $(date +%F_%T) : $* ===" >> "$LOG"
while :; do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [ "$N" -lt "$MAX" ] && break
  sleep 300
done
echo "=== LANE $LANE GO (gpu procs=$N) $(date +%F_%T) ===" >> "$LOG"
exec bash gpu1080/run_l54_lane.sh "$LANE" "$@"
