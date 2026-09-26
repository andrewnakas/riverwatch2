#!/usr/bin/env bash
# MODERN-1 @reboot resume. Idempotent: everything it restarts skips completed work.
#
# WHY THIS EXISTS. On 2026-09-26 00:15 the box stopped answering ssh while
# Tailscale still reported it active. There were @reboot resumes for the L54 and
# L56 queues but NONE for MODERN-1, so a reboot would have silently left the
# Daymet build and the Gate B batch stopped, and the first sign would have been a
# stall notification hours later.
#
# install:  crontab -l | { cat; echo "@reboot bash \$HOME/riverwatch2/gpu1080/m1_boot_resume.sh"; } | crontab -
# disarm:   touch ~/riverwatch2/gpu1080/M1_STOP
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_boot_resume.log
exec >>"$LOG" 2>&1
echo "=== M1 BOOT RESUME $(date +%F_%T) ==="
[ -f gpu1080/M1_STOP ] && { echo "M1_STOP present - disarmed, doing nothing"; exit 0; }
sleep 60                         # let the network and nvidia stack settle
PY=gpu1080/.venv/bin/python

# ---- 1. the Daymet back-extension (resumes from its var-year cache) ----------
n=$(ls data/modern_corpus/_daymet_1980_2013/*.csv.gz 2>/dev/null | wc -l)
if [ "$n" -lt 671 ] && ! pgrep -af "build_daymet_areal_modern.py --start" >/dev/null 2>&1; then
  echo "[$(date +%T)] daymet at $n/671 and not running - resuming (6 workers)"
  nohup setsid $PY scripts_modern/build_daymet_areal_modern.py \
    --start-year 1980 --end-year 2013 --workers 6 \
    --out data/modern_corpus/_daymet_1980_2013 >> logs/m1_daymet_back.log 2>&1 < /dev/null &
else
  echo "[$(date +%T)] daymet at $n/671 - no action"
fi

# ---- 2. any Gate B / roster run whose CHECKPOINT is missing ------------------
# A finished member has a checkpoint and four dumps; queue_m1_withq.sh skips
# training when the checkpoint exists, so re-issuing is free.
TODO=""
for spec in "${M1_RESUME_JOBS:-gridmetm:501 gridmetm:502 gridmetm:503 gridmetm9:501 gridmetm9:502 gridmetm9:503}"; do
  for job in $spec; do
    M="${job%%:*}"; S="${job##*:}"
    if [ ! -f "data/mblstm/l51_dumps/m1_${M}_s${S}_test14.csv.gz" ]; then
      TODO="$TODO $job"
    fi
  done
done
if [ -n "$TODO" ]; then
  echo "[$(date +%T)] re-issuing incomplete runs:$TODO"
  nohup setsid bash gpu1080/run_m1_batch.sh $TODO > /dev/null 2>&1 < /dev/null &
else
  echo "[$(date +%T)] all listed runs complete - no GPU work to resume"
fi
echo "=== M1 BOOT RESUME ISSUED $(date +%F_%T) ==="
