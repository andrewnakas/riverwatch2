#!/usr/bin/env bash
# LEDGER 54 — a watcher that notices the RESOURCE dying, not just the file missing.
# The 2026-09-15 outage went unnoticed for 4 days because every watcher was a
# `while [ ! -f "$dump" ]; do sleep 300; done` loop: a dead GPU looks exactly like
# a slow job. This polls the GPU and records a loud, timestamped failure.
set -u
cd ~/riverwatch2
LOG=logs/l54_gpu_health.log
while :; do
  if nvidia-smi -L > /dev/null 2>&1; then
    echo "[$(date +%F_%T)] GPU OK  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" >> "$LOG"
  else
    echo "[$(date +%F_%T)] ⚠️ GPU DOWN — driver not responding (kernel $(uname -r)); jobs are dead, not slow." >> "$LOG"
    echo "    fix: sudo apt-get install -y linux-modules-nvidia-580-\$(uname -r) && sudo modprobe nvidia" >> "$LOG"
    echo "    then: bash gpu1080/l54_recover_after_gpu.sh" >> "$LOG"
  fi
  sleep 1800
done
