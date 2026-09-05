#!/usr/bin/env bash
# LEDGER 51 supervisor: run a lane of <member>:<seed> jobs serially.
# usage: run_l51_lane.sh <lanename> <member:seed> [<member:seed> ...]
# Each job is the guarded queue script, which aborts on its own if the training
# window is not the Nearing split. Kill by PID only.
set -u
cd ~/riverwatch2
LANE="${1:?lane}"; shift
LOG="logs/l51_lane_${LANE}.log"
echo "=== LANE $LANE START $(date +%F_%T) : $* ===" >> "$LOG"
for JOB in "$@"; do
  M="${JOB%%:*}"; S="${JOB##*:}"
  echo "[$(date +%F_%T)] -> $M s$S" >> "$LOG"
  bash gpu1080/queue_l51_withq.sh "$M" "$S" >> "$LOG" 2>&1
  echo "[$(date +%F_%T)] <- $M s$S rc=$?" >> "$LOG"
done
echo "=== LANE $LANE DONE $(date +%F_%T) ===" >> "$LOG"
