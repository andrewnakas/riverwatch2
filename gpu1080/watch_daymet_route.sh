#!/usr/bin/env bash
# MODERN-1: resume the Daymet build only when its DATA route is genuinely usable.
#
# ⚠️ The previous watcher polled the collection's .dmr metadata. That returned 200
# while data requests still failed with curl rc=35 (SSL connect error), so it
# relaunched into a flapping service and the next basin failed at once.
# This one runs the builder's own fetch() against a real basin bbox, and demands
# THREE consecutive successes 10 min apart before it will restart anything.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
LOG=logs/m1_daymet_route_watch.log
NEED=3
DEADLINE=$(( $(date +%s) + 172800 ))   # 48 h, then report instead of waiting silently
ok=0
echo "=== DAYMET ROUTE WATCH START $(date +%F_%T) (need $NEED consecutive) ===" >> "$LOG"
while :; do
  if nice -n 10 $PY scripts_modern/probe_daymet_route.py >> "$LOG" 2>&1; then
    ok=$((ok + 1))
    echo "[$(date +%F_%T)] probe OK ($ok/$NEED)" >> "$LOG"
  else
    [ "$ok" -gt 0 ] && echo "[$(date +%F_%T)] probe failed - streak reset (was $ok)" >> "$LOG"
    ok=0
  fi
  [ "$ok" -ge "$NEED" ] && break
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[$(date +%F_%T)] DEADLINE HIT after 48 h - Daymet NOT resumed" >> "$LOG"
    exit 2
  fi
  sleep 600
done
echo "[$(date +%F_%T)] route stable across $NEED probes - relaunching Daymet" >> "$LOG"
nohup setsid $PY scripts_modern/build_daymet_areal_modern.py \
  --start-year 1980 --end-year 2013 --workers 6 \
  --out data/modern_corpus/_daymet_1980_2013 >> logs/m1_daymet_back.log 2>&1 < /dev/null &
echo "[$(date +%F_%T)] Daymet relaunched (pid $!)" >> "$LOG"
