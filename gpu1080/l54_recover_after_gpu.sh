#!/usr/bin/env bash
# LEDGER 54 — recover after the 2026-09-15 GPU outage.
#
# CAUSE: unattended-upgrades kept upgrading the kernel (7.0.0-28 -> -31) while
# `linux-modules-nvidia-580-*` stayed pinned at -28 ("fails to be marked for
# upgrade"). The box rebooted 2026-09-15 04:44 into 7.0.0-31, which has no
# nvidia.ko, so the driver never loaded and every job died.
# FIX (needs root, run BEFORE this script):
#   sudo apt-get install -y linux-modules-nvidia-580-$(uname -r) && sudo modprobe nvidia
#
# NOTHING TRAINED IS LOST: h256 s501-504 all have complete 30-epoch checkpoints
# (only their eval/dumps are missing, which this re-runs), and h512 s501 resumes
# from epoch 20 via the launcher's built-in self-healing resume.
set -u
cd ~/riverwatch2
if ! nvidia-smi -L > /dev/null 2>&1; then
  echo "ABORT: the NVIDIA driver is still not loaded. Run, as root:"
  echo "  sudo apt-get install -y linux-modules-nvidia-580-\$(uname -r) && sudo modprobe nvidia"
  exit 1
fi
echo "GPU OK: $(nvidia-smi -L | head -1)"
# 1. finish the members that already trained: the launcher skips training when the
#    final checkpoint exists, so these are eval+dump only.
nohup bash gpu1080/l54_start_when_free.sh 4 R1 nhar0h256:503 nhar0h256:504 > /dev/null 2>&1 &
sleep 2
# 2. resume h512 s501 from epoch 20, then its remaining seeds.
nohup bash gpu1080/l54_start_when_free.sh 4 R2 nhar0h512:501 nhar0h512:502 nhar0h512:503 > /dev/null 2>&1 &
sleep 2
# 3. the last h256 seed for the 5-seed ship bar.
nohup bash gpu1080/l54_start_when_free.sh 4 R3 nhar0h256:505 > /dev/null 2>&1 &
sleep 2
# 4. re-arm the prediction scorer (its log was lost with the outage).
nohup bash gpu1080/l54_predictions_watch.sh > /dev/null 2>&1 &
sleep 2
echo "relaunched: lanes R1/R2/R3 + predictions watcher"
pgrep -fa "l54_start_when_free|l54_predictions_watch" | sed -E 's/.*l54_/  l54_/' | cut -c1-100
