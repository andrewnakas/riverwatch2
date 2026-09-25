#!/usr/bin/env bash
# LEDGER 56b — run the hybrid members at EXACTLY 3-way concurrency.
#
# ⚠️ WHY NOT l5*_start_when_free.sh HERE. That gate polls nvidia-smi once per
# lane and then execs; with several lanes waiting on the same free slot, two can
# both observe "2 < 3" and both start, because a job needs minutes to load its
# corpus before it appears as a CUDA process. Three lanes racing one slot is how
# an 8 GB card gets 5 trainers ([[the-gate-i-bypassed-and-the-hardware-fault-that-wasnt]]).
# This controller counts its OWN children instead, which cannot race, and waits
# for the previous batch to fully drain before it starts.
#
# usage: run_l56b_batch.sh <member:seed> ...
set -u
cd ~/riverwatch2
LOG="logs/l56b_batch.log"
MAX=3
echo "=== L56b BATCH WAITING for the GPU to drain $(date +%F_%T) : $* ===" >> "$LOG"
QUIET=0
while :; do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0)
  if [ "$N" -eq 0 ]; then QUIET=$((QUIET + 1)); else QUIET=0; fi
  [ "$QUIET" -ge 2 ] && break          # two consecutive quiet polls = previous batch done
  sleep 120
done
echo "=== L56b BATCH GO $(date +%F_%T) ===" >> "$LOG"
for JOB in "$@"; do
  M="${JOB%%:*}"; S="${JOB##*:}"
  # ANY *smoke* job is a GATE: drain to it, run it synchronously, and abort the
  # whole batch if it fails, so a broken guard cannot burn hours of real runs.
  if [ "${M%smoke}" != "$M" ]; then
    while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n; done
    echo "[$(date +%F_%T)] GATE -> $M s$S" >> "$LOG"
    bash gpu1080/l56_dispatch.sh "$M" "$S" >> "logs/l56b_job_${M}_s${S}.log" 2>&1
    RC=$?
    echo "[$(date +%F_%T)] GATE <- $M s$S rc=$RC" >> "$LOG"
    [ "$RC" -eq 0 ] || { echo "=== L56b BATCH ABORTED: gate failed $(date +%F_%T) ===" >> "$LOG"; exit 1; }
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$MAX" ]; do wait -n; done
  echo "[$(date +%F_%T)] -> $M s$S" >> "$LOG"
  ( bash gpu1080/l56_dispatch.sh "$M" "$S" >> "logs/l56b_job_${M}_s${S}.log" 2>&1
    echo "[$(date +%F_%T)] <- $M s$S rc=$?" >> "$LOG" ) &
  sleep 90                              # let the new job's CUDA context settle
done
wait
echo "=== L56b BATCH DONE $(date +%F_%T) ===" >> "$LOG"
