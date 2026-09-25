#!/usr/bin/env bash
# LEDGER 50 — the DEPTH lane. Waits for the currently-running multi6 s444 (launched
# outside any lane) to finish, then runs the seed queue serially. Exactly one job per
# lane at a time; the multiL lane runs concurrently, so the card holds two NH trainers.
set -u
cd "$HOME/riverwatch2" || exit 1
LOG=logs/l50_lane_depth.log
WAIT_PID=${WAIT_PID:-}
echo "=== lane depth start $(date +%F_%T) pid=$$ waiting on pid ${WAIT_PID:-none} ===" >> "$LOG"
if [ -n "$WAIT_PID" ]; then
  for i in $(seq 1 2880); do ps -p "$WAIT_PID" >/dev/null 2>&1 || break; sleep 60; done
  ps -p "$WAIT_PID" >/dev/null 2>&1 && { echo "[$(date +%T)] TIMEOUT waiting on $WAIT_PID" >> "$LOG"; exit 1; }
  echo "[$(date +%T)] predecessor $WAIT_PID finished" >> "$LOG"
  sleep 120   # let its evaluate+dump step run before adding load
fi
for item in multi6:555 multi6:666 multi5:666 multi5:777 multi:999; do
  NAME=${item%%:*}; S=${item##*:}
  echo "[$(date +%T)] launching $NAME s$S" >> "$LOG"
  bash gpu1080/queue_l50_seeds.sh "$NAME" "$S" >> "$LOG" 2>&1
  echo "[$(date +%T)] $NAME s$S returned $?" >> "$LOG"
done
echo "=== lane depth DONE $(date +%F_%T) ===" >> "$LOG"
