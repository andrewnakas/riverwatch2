#!/usr/bin/env bash
# MODERN-1 — THE DECOMPOSITION TEST. Registered in PREREG_v2.md before this member existed:
# gridmetm9fix3L1D1 (--enc-lead 1 --dec-lead 1 on the CORRECTED corpus) reproduces the
# misregistered gridmetm9 on val1 to within +/-0.003, because both shift every column one day.
#
# If it reproduces, the original decomposes into (led encoder) + (decoder reading rain(T+1)). The
# second term is +0.021541 over the entitled counterpart (enc-lead 1 alone, 0.842792) and is
# forcing BEYOND entitlement -> the anti-leak attribution verdict was WRONG and the frozen
# gridmetm9-bearing records are RETRACTED.
#
# ⛔ dec_lead!=0 is unservable; check_sidecar refuses it unless M1_ALLOW_DIAGNOSTIC=1.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_decomp_chain.log
exec >>"$LOG" 2>&1
echo "=== DECOMP CHAIN START $(date +%F_%T) ==="

QUIET=0
for i in $(seq 1 2880); do
  B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  T=$(ps -eo cmd --no-headers | awk 'index($0,"train_mblstm.py")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  C=$(ps -eo cmd --no-headers | awk 'index($0,"chain_shipbar_fix3.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$B" -eq 0 ] && [ "$T" -eq 0 ] && [ "$C" -eq 0 ] && [ "$N" -eq 0 ]; then
    QUIET=$((QUIET+1)); else QUIET=0; fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] clear on 3 consecutive polls"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: batch=$B train=$T sbchain=$C gpu=$N quiet=$QUIET/3"
  sleep 120
done
B=$(ps -eo cmd --no-headers | awk 'index($0,"run_m1_batch.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
[ "$B" -ne 0 ] && { echo "ABORT: a batch is still running"; exit 5; }

echo "[$(date +%T)] training gridmetm9fix3L1D1, seeds 501-503 (--enc-lead 1 --dec-lead 1)"
M1_QUEUE=queue_m1h_withq.sh M1_LOG=m1_batch_decomp M1_ALLLEADS=0 \
  bash gpu1080/run_m1_batch.sh gridmetm9fix3L1D1:501 gridmetm9fix3L1D1:502 gridmetm9fix3L1D1:503
echo "[$(date +%T)] training returned rc=$?"

n=$(ls data/mblstm/l51_dumps/m1_gridmetm9fix3L1D1_s50?_val1.csv.gz 2>/dev/null | wc -l)
echo "[$(date +%T)] gridmetm9fix3L1D1: $n/3 val1 dumps"
[ "$n" -lt 3 ] && { echo "ABORT: only $n/3 dumps"; exit 6; }

PY=gpu1080/.venv/bin/python
echo "[$(date +%T)] ===== THE DECOMPOSITION TEST: does it reproduce the misregistered member? ====="
M1_ALLOW_DIAGNOSTIC=1 $PY analysis/m1_composition.py --frame val1 \
    --members gridmetm9fix3L1D1 --against gridmetm9 \
    --protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_decomp_vs_orig.json 2>&1 | tail -16

echo "[$(date +%T)] ===== the entitled counterpart: how much does the decoder's rain(T+1) add? ====="
M1_ALLOW_DIAGNOSTIC=1 $PY analysis/m1_composition.py --frame val1 \
    --members gridmetm9fix3L1D1 --against gridmetm9fix3L1 \
    --protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_decomp_vs_enclead.json 2>&1 | tail -16

echo "[$(date +%T)] ===== where that increment lives (the test that may have been wrong) ====="
M1_ALLOW_DIAGNOSTIC=1 $PY analysis/m1_leak_future_rain_attribution.py \
    --a gridmetm9fix3L1 --b gridmetm9fix3L1D1 \
    --out benchmarks/m1_decomp_attribution.json 2>&1 | tail -18
echo "=== DECOMP CHAIN DONE $(date +%F_%T) ==="
