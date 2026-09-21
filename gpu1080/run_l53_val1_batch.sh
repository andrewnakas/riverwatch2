#!/usr/bin/env bash
# two serial lanes of val1 dumps for the members that never had the stride-1 val frame
set -u; cd ~/riverwatch2
LANE="${1:?lane}"; shift
LOG="logs/l53_val1_batch_${LANE}.log"
echo "=== VAL1 LANE $LANE START $(date +%F_%T): $* ===" >> "$LOG"
for JOB in "$@"; do M="${JOB%%:*}"; S="${JOB##*:}"
  bash gpu1080/dump_l53_val1.sh "$M" "$S" >> "$LOG" 2>&1; echo "[$(date +%F_%T)] <- $M s$S rc=$?" >> "$LOG"; done
echo "=== VAL1 LANE $LANE DONE $(date +%F_%T) ===" >> "$LOG"
