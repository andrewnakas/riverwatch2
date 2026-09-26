#!/usr/bin/env bash
# Train the modern NH AR-LSTM, verifying the ARTIFACT rather than the exit code.
#
# ⚠️ The first attempt logged "NH train rc=0" while the run had died instantly with
# FileNotFoundError: gpu1080/basins.txt -- I had passed --basin-file basins.txt when
# the renderer joins it to the base. So rc is not trustworthy here; a run is only
# real if NH wrote a run directory containing a model checkpoint.
# Keyed on the GPU emptying (the resource), two consecutive polls, with a deadline.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
CFG=gpu1080/cfg_nhar0m_s501.yml
LOG=logs/m1_nh_chain2.log
exec >>"$LOG" 2>&1
echo "=== NH CHAIN2 START $(date +%F_%T) ==="

# pre-flight: every path in the config must exist, or we fail here not in torch
$PY /tmp/nh_pathcheck.py "$CFG" || { echo "PATHCHECK FAILED - not launching"; exit 1; }

QUIET=0
for i in $(seq 1 360); do
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$N" -eq 0 ]; then QUIET=$((QUIET + 1)); else QUIET=0; fi
  [ "$QUIET" -ge 2 ] && { echo "[$(date +%T)] GPU free on two consecutive polls"; break; }
  sleep 120
done
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)" -ne 0 ] && {
  echo "[$(date +%T)] GPU never freed - NOT launching"; exit 2; }
A=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
[ "${A:-0}" -lt 3000 ] && { echo "[$(date +%T)] only ${A}MiB RAM - NOT launching"; exit 3; }

echo "[$(date +%T)] training m1_nhar0m s501 (${A}MiB RAM available)"
nice -n 5 $PY scripts/nh_arlstm_train.py --config-file "$CFG" > logs/m1_nhtrain_nhar0m_s501.log 2>&1
RC=$?
echo "[$(date +%T)] reported rc=$RC -- now VERIFYING THE ARTIFACT, which is what counts"

RUN=$(ls -dt gpu1080/runs/m1_nhar0m_s501_* 2>/dev/null | head -1)
[ -z "$RUN" ] && RUN=$(ls -dt runs/m1_nhar0m_s501_* 2>/dev/null | head -1)
if [ -z "$RUN" ]; then
  echo "ARTIFACT FAIL: no run directory for m1_nhar0m_s501. Last log lines:"
  tail -6 logs/m1_nhtrain_nhar0m_s501.log
  exit 4
fi
CK=$(ls "$RUN"/model_epoch*.pt 2>/dev/null | tail -1)
if [ -z "$CK" ]; then
  echo "ARTIFACT FAIL: run dir $RUN has no model_epoch*.pt. Last log lines:"
  tail -6 logs/m1_nhtrain_nhar0m_s501.log
  exit 5
fi
echo "ARTIFACT OK: $RUN"
echo "  final checkpoint: $(basename "$CK") ($(stat -c %s "$CK") bytes)"
grep -cE "^# Epoch 30" logs/m1_nhtrain_nhar0m_s501.log | sed 's/^/  epoch-30 lines: /'
echo "=== NH CHAIN2 DONE $(date +%F_%T) ==="
