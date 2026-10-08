#!/usr/bin/env bash
# MODERN-1 — the NO-Q replacement member. The no-q record (test1 0.843090) is RETRACTED because its
# single member lstm_fused4md4 sits on camels_corpus_fused4m_m3, which carries the misregistered
# *_gridmet columns, so its decoder reads precipitation for T+1 while predicting T.
#
# lstm_fused4md4fix3 is the same recipe on the corrected camels_corpus_fused4mfix3_m3. 5 seeds
# directly: the no-q arm is a single member, so there is no composition to screen and the ship bar
# is the only bar that matters.
#
# ⚠️ queue_m1g_noq.sh had EXTRA uninitialized and unused, which would have silently dropped any
# flag; that is fixed and GUARD 1 now asserts enc_lead/dec_lead both ways. This member passes no
# lead flags, so both must read 0.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_noqfix3_chain.log
exec >>"$LOG" 2>&1
echo "=== NOQ-FIX3 CHAIN START $(date +%F_%T) ==="

QUIET=0
for i in $(seq 1 2880); do
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  T=$(ps -eo cmd --no-headers | awk 'index($0,"train_mblstm.py")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  C=$(ps -eo cmd --no-headers | awk 'index($0,"chain_decay.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$B" -eq 0 ] && [ "$T" -eq 0 ] && [ "$C" -eq 0 ] && [ "$N" -eq 0 ]; then
    QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear on 3 consecutive polls"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: batch=$B train=$T decay=$C gpu=$N quiet=$QUIET/3"
  sleep 120
done
B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
[ "$B" -ne 0 ] && { echo "ABORT: a batch is still running"; exit 5; }

echo "[$(date +%T)] training lstm_fused4md4fix3, seeds 501-505"
M1_QUEUE=queue_m1g_noq.sh M1_LOG=m1_batch_noqfix3 M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh \
    lstm_fused4md4fix3:501 lstm_fused4md4fix3:502 lstm_fused4md4fix3:503 \
    lstm_fused4md4fix3:504 lstm_fused4md4fix3:505
echo "[$(date +%T)] training returned rc=$?"

n=$(ls data/mblstm/l51_dumps/m1_lstm_fused4md4fix3_s50?_val1.csv.gz 2>/dev/null | wc -l)
echo "[$(date +%T)] lstm_fused4md4fix3: $n/5 val1 dumps"
[ "$n" -lt 5 ] && { echo "ABORT: only $n/5 dumps - below the ship bar"; exit 6; }

PY=gpu1080/.venv/bin/python
echo "[$(date +%T)] ===== the corrected NO-Q member at the 5-seed ship bar (val1) ====="
for C in mean trimmed; do
  $PY analysis/m1_composition.py --frame val1 --members lstm_fused4md4fix3 \
      --protocol modern-v3-9yr --seeds 501,502,503,504,505 --combiner $C \
      --out benchmarks/m1_noq_corrected_5seed_$C.json 2>&1 | grep -E "selected combiner"
done
echo "[$(date +%T)] ===== against the RETRACTED member, same seeds (the cost of correctness) ====="
$PY analysis/m1_composition.py --frame val1 --members lstm_fused4md4fix3 \
    --against lstm_fused4md4 --protocol modern-v3-9yr --seeds 501,502,503,504,505 \
    --out benchmarks/m1_noq_corrected_vs_retracted.json 2>&1 | tail -14
echo "=== NOQ-FIX3 CHAIN DONE $(date +%F_%T) ==="
