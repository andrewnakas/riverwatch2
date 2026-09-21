#!/usr/bin/env bash
# LEDGER 54 — freeze the record the moment the PRE-REGISTERED ship bar (5 seeds) is met.
# Deliberately does NOT ship at 3 seeds, even though 3 currently scores higher.
set -u; cd ~/riverwatch2
PY=gpu1080/.venv/bin/python; LOG=logs/l54_ship.log; O=benchmarks/l54
B=fused3h1,daymeth1,nldash1,maurerh1,aorch1
mkdir -p "$O"
exec 9>/tmp/rw2_l54_ship.lock; flock -n 9 || exit 0
need=""
for s in 504 505; do for f in val1 test1; do need="$need data/mblstm/l51_dumps/camels531_l51_nhar0h256_s${s}_${f}.csv.gz"; done; done
echo "=== ship watcher: waiting for h256 s504/s505 $(date +%F_%T) ===" >> "$LOG"
while :; do
  if ! nvidia-smi -L > /dev/null 2>&1; then
    echo "[$(date +%F_%T)] GPU DOWN — see l54_gpu_health.log; not a slow job." >> "$LOG"
  fi
  ok=1; for f in $need; do [ -f "$f" ] || ok=0; done
  [ "$ok" = 1 ] && break
  sleep 300
done
echo "=== 5 seeds present $(date +%F_%T); freezing at the ship bar ===" >> "$LOG"
$PY analysis/l54_freeze_record.py --members "$B,nhar,nhar0,nhar0h256" --new nhar0h256 \
    --seeds 501,502,503,504,505 --out benchmarks/l54_FINAL_8member_test1.json >> "$LOG" 2>&1
echo "=== done rc=$? $(date +%F_%T) ===" >> "$LOG"
