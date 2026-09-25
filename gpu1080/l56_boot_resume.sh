#!/usr/bin/env bash
# LEDGER 56 — @reboot self-heal. Every launcher is idempotent (skips a member/seed
# whose final checkpoint and dumps exist, resumes a partial run), so re-issuing
# the full queue after a reboot is safe and re-does nothing that finished.
# Disarm by creating gpu1080/L56_STOP.
set -u
cd ~/riverwatch2 || exit 0
[ -f gpu1080/L56_STOP ] && { echo "$(date +%F_%T) L56_STOP present, standing down" >> logs/l56_boot.log; exit 0; }
sleep 60                                   # let the driver and filesystems settle
echo "=== L56 BOOT RESUME $(date +%F_%T) ===" >> logs/l56_boot.log
setsid nohup bash gpu1080/run_l56b_batch.sh \
  fused3h1d3:504 fused3h1d3:505 fused4h1d4:501 fused4h1d4:502 fused4h1d4:503 \
  fused3h1d3qsmoke:999 fused3h1d3q:501 fused3h1d3q:502 fused3h1d3q:503 \
  fused3h1decmsmoke:999 fused3h1decm:501 fused3h1decn:501 fused3h1decm:502 \
  fused3h1decn:502 fused3h1decm:503 fused3h1decn:503 \
  daymeth1d3:501 maurerh1d3:501 nldash1d3:501 \
  daymeth1d3:502 maurerh1d3:502 nldash1d3:502 \
  daymeth1d3:503 maurerh1d3:503 nldash1d3:503 \
  >> logs/l56_boot.log 2>&1 < /dev/null &
