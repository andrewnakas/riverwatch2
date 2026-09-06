#!/usr/bin/env bash
# LEDGER 51 lane C — uses queue_l52_withq.sh (the copy carrying fused3h1).
# A separate file because lanes A/B are mid-flight in queue_l51_withq.sh and
# bash reads a running script by byte offset: editing it corrupts the run.
set -u
cd ~/riverwatch2
LANE="${1:?lane}"; shift
LOG="logs/l51_lane_${LANE}.log"
echo "=== LANE $LANE START $(date +%F_%T) : $* ===" >> "$LOG"
for JOB in "$@"; do
  M="${JOB%%:*}"; S="${JOB##*:}"
  echo "[$(date +%F_%T)] -> $M s$S" >> "$LOG"
  bash gpu1080/queue_l52_withq.sh "$M" "$S" >> "$LOG" 2>&1
  echo "[$(date +%F_%T)] <- $M s$S rc=$?" >> "$LOG"
done
echo "=== LANE $LANE DONE $(date +%F_%T) ===" >> "$LOG"
