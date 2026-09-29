#!/usr/bin/env bash
# MODERN-1 @reboot entry point. Delegates to m1_rearm.sh.
#
# It is deliberately a thin shim: the crontab already contains
#   @reboot bash $HOME/riverwatch2/gpu1080/m1_boot_resume.sh
# and the previous version of this file was a NO-OP because the crontab entry pointed
# at a script that had never been rsync'd to the box -- ~20 h of idle GPU on the one
# real reboot. Keeping the crontab target fixed and the logic in one delegated file
# means there is a single file to get onto the box, and the composition can be tested
# by simply running this shim.
#
# disarm: touch ~/riverwatch2/gpu1080/M1_STOP
set -u
cd ~/riverwatch2 || exit 1
exec >> logs/m1_boot_resume.log 2>&1
echo "=== M1 BOOT RESUME $(date +%F_%T) uptime:$(uptime -p) ==="
sleep 60                      # let the network and the nvidia stack settle
exec bash gpu1080/m1_rearm.sh
