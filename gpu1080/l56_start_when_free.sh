#!/usr/bin/env bash
# LEDGER 56 — start a lane only when the GPU has capacity. Counts REAL CUDA
# processes via nvidia-smi ([[the-gate-i-bypassed-and-the-hardware-fault-that-wasnt]]:
# never call queue_l56_withq.sh directly, even for a smoke).
# usage: l56_start_when_free.sh <max_gpu_procs> <lane> <member:seed> ...
set -u
cd ~/riverwatch2
MAX="${1:?max_gpu_procs}"; LANE="${2:?lane}"; shift 2
LOG="logs/l56_lane_${LANE}.log"
echo "=== LANE $LANE WAITING for < $MAX gpu procs $(date +%F_%T) : $* ===" >> "$LOG"
while :; do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [ "$N" -lt "$MAX" ] && break
  sleep 300
done
echo "=== LANE $LANE GO (gpu procs=$N) $(date +%F_%T) ===" >> "$LOG"
exec bash gpu1080/run_l56_lane.sh "$LANE" "$@"
