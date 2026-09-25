#!/usr/bin/env bash
set -u; cd ~/riverwatch2
LANE="${1:?lane}"; shift
LOG="logs/l56_lane_${LANE}.log"
echo "=== LANE $LANE START $(date +%F_%T) : $* ===" >> "$LOG"
for JOB in "$@"; do M="${JOB%%:*}"; S="${JOB##*:}"
  echo "[$(date +%F_%T)] -> $M s$S" >> "$LOG"
  bash gpu1080/queue_l56_withq.sh "$M" "$S" >> "$LOG" 2>&1
  echo "[$(date +%F_%T)] <- $M s$S rc=$?" >> "$LOG"; done
echo "=== LANE $LANE DONE $(date +%F_%T) ===" >> "$LOG"
