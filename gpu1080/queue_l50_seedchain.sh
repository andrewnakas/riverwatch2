#!/usr/bin/env bash
# LEDGER 50 Stage 2 — run the seed-depth queue SEQUENTIALLY, one NH job at a time.
# Deliberately does NOT wait for the δHBV probe (they co-schedule); it only ensures
# a single neuralhydrology trainer is resident, so the card holds δHBV + one NH job.
# Order = marginal ensemble value per net (w^2/(k(k+1)) at the shipped weights).
# Kill by PID only. Idempotent: queue_l50_seeds.sh skips a member whose dumps exist.
set -u
cd "$HOME/riverwatch2" || exit 1
LOG=logs/l50_seedchain.log
QUEUE="multi6:555 multi6:666 multi5:666 multi5:777 multi:999 nldas:444"
echo "=== L50 seed chain start $(date +%F_%T) pid=$$ queue: $QUEUE ===" >> "$LOG"
for item in $QUEUE; do
  NAME=${item%%:*}; S=${item##*:}
  # wait for any running NH trainer (including the current multi6 s444) to exit
  for i in $(seq 1 2880); do
    ps -eo args | grep -q "[n]euralhydrology.nh_run train" || break
    sleep 60
  done
  if ps -eo args | grep -q "[n]euralhydrology.nh_run train"; then
    echo "[$(date +%T)] TIMEOUT waiting for the GPU; stopping chain" >> "$LOG"; exit 1
  fi
  echo "[$(date +%T)] launching $NAME s$S" >> "$LOG"
  bash gpu1080/queue_l50_seeds.sh "$NAME" "$S" >> "$LOG" 2>&1
  echo "[$(date +%T)] $NAME s$S returned $?" >> "$LOG"
done
echo "=== L50 seed chain DONE $(date +%F_%T) ===" >> "$LOG"
