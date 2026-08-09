#!/usr/bin/env bash
# Wait for the withqmulti backtest to finish, then score it against the
# PRE-REGISTERED gate and write the verdict to disk. Detached, so it survives
# session clears and disconnects. Result lands in withqmulti_VERDICT.txt.
set -u
cd ~/riverwatch2
exec 9>/tmp/rw2_autoscore.lock
flock -n 9 || { echo "autoscore already running"; exit 0; }
PY=gpu1080/.venv/bin/python
OUT=withqmulti_VERDICT.txt
D=data/mblstm/gpu_dumps_s14/camels531_withqmulti_full531.csv.gz

# wait for the DONE marker (bounded: 3h)
for i in $(seq 1 360); do
  grep -q "BACKTEST DONE" logs/withqmulti_backtest.log 2>/dev/null && break
  pgrep -f "backtest_mblstm.*withqmulti" >/dev/null 2>&1 || break
  sleep 30
done

{
  echo "=== WITHQMULTI VERDICT  $(date +%F_%T) ==="
  if [ ! -f "$D" ]; then
    echo "NO DUMP PRODUCED — backtest did not complete successfully."
    tail -20 logs/backtest_withqmulti.log 2>/dev/null
  elif ! gzip -t "$D" 2>/dev/null; then
    echo "DUMP FAILED gzip -t — corrupt or truncated."
  else
    ls -la "$D"
    echo
    $PY score_withqmulti.py 2>&1
  fi
} > "$OUT" 2>&1
echo "wrote $OUT"
