#!/usr/bin/env bash
# MODERN-1 batch controller. usage: run_m1_batch.sh <member:seed> ...
#
# Copied from run_l56b_batch2.sh, which counts its OWN children rather than
# polling nvidia-smi once and exec'ing -- that race is how an 8 GB card ended up
# with 5 trainers. NOTE the drain check uses `wc -l`, NOT `grep -c . || echo 0`:
# grep -c EXITS 1 on a zero count, so the `|| echo 0` also fires, N becomes the
# two-line string "0\n0", the integer test dies, and the gate blocks FOREVER
# exactly when the GPU is free. That cost ~9 h of idle card in LEDGER 56.
set -u
cd ~/riverwatch2
LOG="logs/m1_batch.log"
MAX=${M1_MAX:-3}
echo "=== M1 BATCH WAITING for the GPU to drain $(date +%F_%T) : $* ===" >> "$LOG"
QUIET=0
while :; do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$N" -eq 0 ]; then QUIET=$((QUIET + 1)); else QUIET=0; fi
  [ "$QUIET" -ge 2 ] && break
  sleep 120
done
echo "=== M1 BATCH GO $(date +%F_%T) ===" >> "$LOG"
for JOB in "$@"; do
  M="${JOB%%:*}"; S="${JOB##*:}"
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do wait -n; done
  echo "[$(date +%F_%T)] -> $M s$S" >> "$LOG"
  ( bash gpu1080/queue_m1_withq.sh "$M" "$S"; echo "[$(date +%F_%T)] <- $M s$S rc=$?" >> "$LOG" ) &
  sleep 90    # let each job load its corpus before the next starts
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n; done
echo "=== M1 BATCH DONE $(date +%F_%T) ===" >> "$LOG"
