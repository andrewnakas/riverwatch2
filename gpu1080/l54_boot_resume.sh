#!/usr/bin/env bash
# LEDGER 54 — @reboot hook. Watchers and lanes are ordinary background shells, so a
# reboot silently kills all of them (this cost 4 days on 09-15 and another eval on
# 09-21). Everything below is idempotent: the queue scripts hold flock locks, skip
# training when a final checkpoint exists, and resume from the last epoch otherwise.
set -u
cd ~/riverwatch2 || exit 0
LOG=logs/l54_boot_resume.log
echo "=== @reboot $(date +%F_%T) kernel=$(uname -r) ===" >> "$LOG"

# The driver module can lag the kernel (09-15: a dpkg hold froze the driver while the
# kernel advanced). Wait briefly, then report loudly rather than silently doing nothing.
for i in $(seq 1 30); do nvidia-smi -L > /dev/null 2>&1 && break; sleep 10; done
if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "[$(date +%F_%T)] GPU ABSENT after boot. Check: apt-mark showhold; find /lib/modules/\$(uname -r) -name 'nvidia.ko*'" >> "$LOG"
  echo "  fix: sudo apt-mark unhold nvidia-driver-580 nvidia-utils-580 && sudo apt-get install -y nvidia-driver-580 linux-modules-nvidia-580-\$(uname -r) && sudo modprobe nvidia" >> "$LOG"
fi
nohup bash gpu1080/l54_gpu_health_watch.sh > /dev/null 2>&1 &

# finish any member whose dumps are incomplete (training is skipped when done)
for VS in nhar0h512:501 nhar0h512:502 nhar0h512:503; do
  V="${VS%%:*}"; S="${VS##*:}"
  [ -f "data/mblstm/l51_dumps/camels531_l51_${V}_s${S}_test1.csv.gz" ] && continue
  echo "[$(date +%F_%T)] resuming $V s$S" >> "$LOG"
  nohup bash gpu1080/l54_start_when_free.sh 3 B_${V}_${S} "${V}:${S}" > /dev/null 2>&1 &
  sleep 2
done
echo "=== @reboot handoff done $(date +%F_%T) ===" >> "$LOG"
