#!/usr/bin/env bash
set -u; cd ~/riverwatch2
LANE="${1:?lane}"; shift
LOG="logs/l54_lane_${LANE}.log"
echo "=== LANE $LANE START $(date +%F_%T) : $* ===" >> "$LOG"
for JOB in "$@"; do V="${JOB%%:*}"; S="${JOB##*:}"
  echo "[$(date +%F_%T)] -> $V s$S" >> "$LOG"
  bash gpu1080/queue_l54_nhar.sh "$V" "$S" >> "$LOG" 2>&1
  echo "[$(date +%F_%T)] <- $V s$S rc=$?" >> "$LOG"; done
echo "=== LANE $LANE DONE $(date +%F_%T) ===" >> "$LOG"
