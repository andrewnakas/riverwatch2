#!/usr/bin/env bash
# wait for the deciding dumps, then adjudicate targets (2) and (3) automatically
set -u; cd ~/riverwatch2
LOG=logs/l54_targets.log; PY=gpu1080/.venv/bin/python
exec 9>/tmp/rw2_l54_targets.lock; flock -n 9 || exit 0
need="data/mblstm/l51_dumps/camels531_l51_nhar0_s503_test1.csv.gz
data/mblstm/l51_dumps/camels531_l51_nhar0h256_s501_val1.csv.gz
data/mblstm/l51_dumps/camels531_l51_nhar0h256_s501_test1.csv.gz"
echo "=== waiting $(date +%F_%T) ===" >> "$LOG"
while :; do ok=1; for f in $need; do [ -f "$f" ] || ok=0; done; [ "$ok" = 1 ] && break; sleep 300; done
echo "=== deciding dumps present $(date +%F_%T) ===" >> "$LOG"
$PY analysis/l54_targets_verdict.py --candidates nhar0,nhar0h256 \
    --out benchmarks/l54/TARGETS_VERDICT.json >> "$LOG" 2>&1
echo "=== done $(date +%F_%T) ===" >> "$LOG"
