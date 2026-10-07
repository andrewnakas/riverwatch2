#!/usr/bin/env bash
# MODERN-1 — the --enc-lead screen. Registered in PREREG_v2.md before any number exists:
# gridmetm9fix3L1 beats gridmetm9fix3 by +0.008..+0.016 on val1 and the misregistered gridmetm9
# by +0.002..+0.012.
#
# Why: the misregistered gridMET member beat the corrected one by ~0.010, and per-row attribution
# put 67% of that on high-target-day-rain rows while showing it is WORSE where future rain would
# help. So the gain is the encoder's final step carrying rain(T) with its discharge still ending at
# T-1 -- reproducible deliberately, for any product, and entitled under perfect forcing.
# If it holds it generalizes to all four products, which by the 1/N rule is the real prize.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_enclead_chain.log
exec >>"$LOG" 2>&1
echo "=== ENCLEAD CHAIN START $(date +%F_%T) ==="

# Gate on the PRODUCER being gone, not on a quiet card: a job is invisible to nvidia-smi for
# minutes while it loads, and a long CPU phase makes a busy box look idle.
QUIET=0
for i in $(seq 1 2880); do
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  T=$(ps -eo cmd --no-headers | awk 'index($0,"train_mblstm.py")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$B" -eq 0 ] && [ "$T" -eq 0 ] && [ "$N" -eq 0 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear on 3 consecutive polls"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: batch=$B train=$T gpu=$N quiet=$QUIET/3"
  sleep 120
done
B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
[ "$B" -ne 0 ] && { echo "ABORT: a batch is still running after the wait window"; exit 5; }

# Do not run the screen if its own baseline never landed: without fix3 the probe has nothing to
# be compared against and would burn 4.5 GPU-hours for an uninterpretable number.
NF=$(ls data/mblstm/l51_dumps/m1_gridmetm9fix3_s50?_val1.csv.gz 2>/dev/null | wc -l)
if [ "$NF" -lt 3 ]; then
  echo "ABORT: gridmetm9fix3 has only $NF/3 val1 dumps - the screen has no baseline"; exit 6
fi
echo "[$(date +%T)] baseline present ($NF/3 fix3 val1 dumps)"

echo "[$(date +%T)] training gridmetm9fix3L1, seeds 501-503 (--enc-lead 1)"
M1_QUEUE=queue_m1f_withq.sh M1_LOG=m1_batch_enclead M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh gridmetm9fix3L1:501 gridmetm9fix3L1:502 gridmetm9fix3L1:503
echo "[$(date +%T)] training returned rc=$?"

PY=gpu1080/.venv/bin/python
echo "[$(date +%T)] ===== REGISTERED READ 1: enc-lead vs the correctly-registered baseline ====="
$PY analysis/m1_composition.py --frame val1 --members gridmetm9fix3L1 --against gridmetm9fix3 \
    --protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_enclead_vs_fix3.json 2>&1 | tail -16

echo "[$(date +%T)] ===== REGISTERED READ 2: enc-lead vs the MISREGISTERED original ====="
$PY analysis/m1_composition.py --frame val1 --members gridmetm9fix3L1 --against gridmetm9 \
    --protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_enclead_vs_orig.json 2>&1 | tail -16

echo "[$(date +%T)] ===== where the enc-lead advantage lives (same partition as the audit) ====="
$PY analysis/m1_leak_future_rain_attribution.py --a gridmetm9fix3L1 --b gridmetm9fix3 \
    --out benchmarks/m1_enclead_attribution.json 2>&1 | tail -16

echo "=== ENCLEAD CHAIN DONE $(date +%F_%T) ==="
