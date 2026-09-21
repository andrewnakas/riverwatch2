#!/usr/bin/env bash
# LEDGER 53 lane supervisor for the NH AR-LSTM member: serial jobs, one file
# per lane so a running lane is never edited (never-edit-a-running-shell-script).
set -u
cd ~/riverwatch2
LANE="${1:?lane}"; shift
LOG="logs/l53_nhar_lane_${LANE}.log"
echo "=== LANE $LANE START $(date +%F_%T) : $* ===" >> "$LOG"
for JOB in "$@"; do
  V="${JOB%%:*}"; S="${JOB##*:}"
  echo "[$(date +%F_%T)] -> $V s$S" >> "$LOG"
  bash gpu1080/queue_l53_nhar.sh "$V" "$S" >> "$LOG" 2>&1
  echo "[$(date +%F_%T)] <- $V s$S rc=$?" >> "$LOG"
done
echo "=== LANE $LANE DONE $(date +%F_%T) ===" >> "$LOG"
