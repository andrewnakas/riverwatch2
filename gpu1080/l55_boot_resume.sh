#!/usr/bin/env bash
# LEDGER 55 — @reboot hook for the Track 2 forcing-dropout lanes.
# Replaces the L54 h512 hook, which is disarmed by gpu1080/L55_STOP_H512
# (capacity closed at one seed 2026-09-21; delete that sentinel to re-arm h512).
# Lanes are ordinary background shells, so a boot loses them; queue_l55_fdrop.sh
# resumes each run from its last checkpoint, so re-launching is idempotent.
set -u
cd ~/riverwatch2 || exit 0
LOG=logs/l55_boot_resume.log
echo "=== @reboot $(date +%F_%T) kernel=$(uname -r) ===" >> "$LOG"
[ -f gpu1080/L55_STOP_FDROP ] && { echo "[$(date +%F_%T)] DISARMED by L55_STOP_FDROP" >> "$LOG"; exit 0; }
for S in 501 502 503; do
  echo "[$(date +%F_%T)] resuming nhar0h256fd30 s$S" >> "$LOG"
  nohup setsid bash gpu1080/l55_fdrop_start_when_free.sh 3 "F1_${S}" "nhar0h256fd30:${S}" \
    </dev/null >/dev/null 2>&1 &
  sleep 4
done
echo "=== @reboot handoff done $(date +%F_%T) ===" >> "$LOG"
