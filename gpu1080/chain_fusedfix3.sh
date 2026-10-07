#!/usr/bin/env bash
# MODERN-1 — score the FULLY-CORRECTED roster. Registered in PREREG_v2.md before these members
# existed: the corrected roster scores BELOW the frozen 0.913216 val1 by -0.001..-0.008, and it
# becomes the headline anyway, because the misregistered data's advantage is UNEXPLAINED after two
# falsified mechanisms (de-sync -0.000298 null; --enc-lead +0.000098 null) and an unexplained gain
# from provably misdated inputs is the shape of the 0.9253 retraction.
#
# Scoring is sequential after training on purpose: an all-leads read launched BESIDE training
# exhausted this 15 GiB box earlier today and took sshd down with it.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_fusedfix3_chain.log
exec >>"$LOG" 2>&1
echo "=== FUSEDFIX3 CHAIN START $(date +%F_%T) ==="

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

# Refuse to score a roster whose members did not all land -- a partial roster would silently
# score as a different, smaller ensemble.
for m in fused4md49fix3 fused4m9fix3 gridmetm9fix3; do
  n=$(ls data/mblstm/l51_dumps/m1_${m}_s50?_val1.csv.gz 2>/dev/null | wc -l)
  echo "[$(date +%T)] $m: $n/3 val1 dumps"
  [ "$n" -lt 3 ] && { echo "ABORT: $m has only $n/3 dumps"; exit 6; }
done

PY=gpu1080/.venv/bin/python
CORR=fused4md49fix3,fused4m9fix3,nldasm9
FROZEN=gridmetm9,fused4md49,fused4m9,nldasm9

echo "[$(date +%T)] ===== REGISTERED READ: fully-corrected roster vs the frozen record ====="
$PY analysis/m1_composition.py --frame val1 --members "$CORR" --against "$FROZEN" \
    --protocol modern-v3-9yr --seeds 501,502,503 --combiner trimmed \
    --out benchmarks/m1_corrected_roster_vs_frozen.json 2>&1 | tail -20

echo "[$(date +%T)] ===== does the corrected gridMET member belong in it? ====="
$PY analysis/m1_composition.py --frame val1 --members "gridmetm9fix3,$CORR" --against "$CORR" \
    --protocol modern-v3-9yr --seeds 501,502,503 --combiner trimmed \
    --out benchmarks/m1_corrected_roster_plus_gridmet.json 2>&1 | tail -16

echo "[$(date +%T)] ===== backward elimination on the corrected pool ====="
$PY analysis/m1_backward_eliminate.py --frame val1 \
    --members "gridmetm9fix3,fused4md49fix3,fused4m9fix3,nldasm9" \
    --protocol modern-v3-9yr --seeds 501,502,503 \
    --out benchmarks/m1_corrected_backward.json 2>&1 | tail -24

echo "[$(date +%T)] ===== combiner choice (1 bit, on val) ====="
for C in mean trimmed; do
  $PY analysis/m1_composition.py --frame val1 --members "$CORR" --protocol modern-v3-9yr \
      --seeds 501,502,503 --combiner $C --out /tmp/corr_$C.json 2>&1 | grep -E "selected combiner"
done
echo "=== FUSEDFIX3 CHAIN DONE $(date +%F_%T) ==="
