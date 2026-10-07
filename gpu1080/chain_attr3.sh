#!/usr/bin/env bash
# Final step of the MODERN-1 leakage audit: re-run the future-rain attribution on the
# CORRECTLY-BUILT corrected corpus.
#
# The attribution already ran on gridmetm9fix (precip-only, mis-specified) and returned an
# ANTI-LEAK verdict: the original is worse (-0.116) where rain(T+1) is high and rain(T) low, and 67%
# of its advantage sits on high-rain(T) rows. Contrast -0.117677, CI [-0.184177, -0.051213].
# The attribution asks WHERE the advantage lives, which fix1's internal de-sync should not confound,
# but confirming on fix3 is cheap and belongs on the right artifact.
#
# ⚠️ val1 only. This is an integrity question adjudicated on the selection frame; test1 is read once.
set -u
cd ~/riverwatch2 || exit 1
LOG=logs/m1_attr3_chain.log
exec >>"$LOG" 2>&1
echo "=== ATTR3 CHAIN START $(date +%F_%T) ==="

# Wait for the artifact this step CONSUMES -- the 3 fix3 val1 dumps -- not for the whole chain.
# A verdict must not wait on artifacts it does not consume.
for i in $(seq 1 2880); do
  N=$(ls data/mblstm/l51_dumps/m1_gridmetm9fix3_s50?_val1.csv.gz 2>/dev/null | wc -l)
  [ "$N" -ge 3 ] && { echo "[$(date +%T)] 3 fix3 val1 dumps present"; break; }
  C=$(ps -eo cmd --no-headers | awk 'index($0,"chain_gridfix3.sh")>0 && $0 !~ /grep/ {c++} END{print c+0}')
  if [ "$C" -eq 0 ] && [ "$N" -lt 3 ]; then
    echo "[$(date +%T)] ABORT: the gridfix3 chain is GONE with only $N/3 dumps"; exit 5
  fi
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: dumps=$N/3 chain=$C"
  sleep 120
done
N=$(ls data/mblstm/l51_dumps/m1_gridmetm9fix3_s50?_val1.csv.gz 2>/dev/null | wc -l)
[ "$N" -ge 3 ] || { echo "ABORT: only $N/3 dumps after the wait window"; exit 5; }

PY=gpu1080/.venv/bin/python
echo "[$(date +%T)] ===== ATTRIBUTION: fix3 (correct) vs gridmetm9 (misregistered) ====="
$PY analysis/m1_leak_future_rain_attribution.py --a gridmetm9fix3 --b gridmetm9 \
    --out benchmarks/m1_leakaudit_attr_fix3.json 2>&1 | tail -24
echo "[$(date +%T)] attribution rc=$?"
echo "=== ATTR3 CHAIN DONE $(date +%F_%T) ==="
