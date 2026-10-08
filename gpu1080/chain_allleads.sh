#!/usr/bin/env bash
# MODERN-1 — the last outstanding deliverable: the ALL-LEADS 1-14 table, part of decision (a)
# ("day-1 median NSE as the headline, PLUS an all-leads 1-14 table").
#
# The corrected members were trained with M1_ALLLEADS=0, so their 14-lead dumps do not exist. The
# queue re-dumps without retraining when the checkpoint is present.
#
# ⚠️ The table is computed ONLY after every dump is finished, never beside one. The all-leads frame
# is 14x the day-1 frame (~17M rows per dump) and reading it next to three trainings exhausted this
# 15 GiB box earlier today and took sshd down with it. analysis/m1_allleads.py is now memory-bounded
# (compact dtypes, no string-key round trip) but the sequencing is belt and braces.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_allleads_chain.log
exec >>"$LOG" 2>&1
echo "=== ALLLEADS CHAIN START $(date +%F_%T) ==="

QUIET=0
for i in $(seq 1 2880); do
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  T=$(ps -eo cmd --no-headers | awk 'index($0,"train_mblstm.py")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$B" -eq 0 ] && [ "$T" -eq 0 ] && [ "$N" -eq 0 ]; then QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear on 3 consecutive polls"; break; }
  sleep 60
done

echo "[$(date +%T)] all-leads dumps for the WITH-Q roster"
M1_QUEUE=queue_m1h_withq.sh M1_LOG=m1_batch_al_withq M1_ALLLEADS=1 \
  bash gpu1080/run_m1_batch.sh \
    fused4md49fix3:501 fused4md49fix3:502 fused4md49fix3:503 fused4md49fix3:504 fused4md49fix3:505 \
    fused4m9fix3:501 fused4m9fix3:502 fused4m9fix3:503 fused4m9fix3:504 fused4m9fix3:505 \
    nldasm9:504 nldasm9:505
echo "[$(date +%T)] with-q dumps rc=$?"

echo "[$(date +%T)] all-leads dumps for the NO-Q member"
M1_QUEUE=queue_m1g_noq.sh M1_LOG=m1_batch_al_noq M1_ALLLEADS=1 \
  bash gpu1080/run_m1_batch.sh \
    lstm_fused4md4fix3:501 lstm_fused4md4fix3:502 lstm_fused4md4fix3:503 \
    lstm_fused4md4fix3:504 lstm_fused4md4fix3:505
echo "[$(date +%T)] no-q dumps rc=$?"

for m in fused4md49fix3 fused4m9fix3 nldasm9 lstm_fused4md4fix3; do
  echo "[$(date +%T)] $m: test14=$(ls data/mblstm/l51_dumps/m1_${m}_s50?_test14.csv.gz 2>/dev/null | wc -l)/5"
done

PY=gpu1080/.venv/bin/python
echo "[$(date +%T)] ===== WITH-Q all-leads 1-14, test frame, 5 seeds ====="
$PY analysis/m1_allleads.py --frame test14 \
    --members fused4md49fix3,fused4m9fix3,nldasm9 --seeds 501,502,503,504,505 \
    --protocol modern-v3-9yr --crosscheck \
    --out benchmarks/m1_withq_allleads_test14_FINAL.json 2>&1 | tail -30

echo "[$(date +%T)] ===== NO-Q all-leads 1-14, test frame, 5 seeds ====="
$PY analysis/m1_allleads.py --frame test14 \
    --members lstm_fused4md4fix3 --seeds 501,502,503,504,505 \
    --protocol modern-v3-9yr --crosscheck \
    --out benchmarks/m1_noq_allleads_test14_FINAL.json 2>&1 | tail -30
echo "=== ALLLEADS CHAIN DONE $(date +%F_%T) ==="
