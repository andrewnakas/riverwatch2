#!/usr/bin/env bash
# LEDGER 50 — one lane of sequential NH jobs. Two lanes run concurrently on the 1080
# (≈1.6 GB each of 8 GB). Within a lane, jobs are serial by construction.
# Usage: bash gpu1080/queue_l50_lanes.sh <lane-name> <name:seed> [name:seed ...]
set -u
cd "$HOME/riverwatch2" || exit 1
LANE=${1:?lane}; shift
LOG=logs/l50_lane_${LANE}.log
echo "=== lane $LANE start $(date +%F_%T) pid=$$ queue: $* ===" >> "$LOG"
for item in "$@"; do
  NAME=${item%%:*}; S=${item##*:}
  echo "[$(date +%T)] launching $NAME s$S" >> "$LOG"
  bash gpu1080/queue_l50_seeds.sh "$NAME" "$S" >> "$LOG" 2>&1
  echo "[$(date +%T)] $NAME s$S returned $?" >> "$LOG"
done
echo "=== lane $LANE DONE $(date +%F_%T) ===" >> "$LOG"
