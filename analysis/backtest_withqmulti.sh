#!/usr/bin/env bash
# Backtest the ALREADY-TRAINED fused with-q checkpoints (s981,s982) as the
# withqmulti candidate. Zero training cost — these were trained 2026-08-01 and
# never evaluated (the Aug-1 dump log has only a START line).
# Verified protocol-identical to the record members: 18 enc / 6 dec vars,
# hidden 256, quantile head, linear q-transform, 27 statics, ctx 365, hor 14,
# val 1998-10-01..1999-09-30, no_q_input=False.
# Gate: memory note `withq-multi-prereg-gate`.
set -u
cd ~/riverwatch2
exec 9>/tmp/rw2_withqmulti_bt.lock
flock -n 9 || { echo "another backtest holds the lock; exiting"; exit 0; }

PY=gpu1080/.venv/bin/python
CK=data/mblstm/gpu_ckpts
CORP=gpu1080/corpora671/camels_corpus_fused_v2
LOG=logs/withqmulti_backtest.log
mkdir -p logs benchmarks

echo "=== WITHQMULTI BACKTEST START $(date +%F_%T) ===" >> $LOG
PAT_NH="nh_run"" train"
while pgrep -f "$PAT_NH" >/dev/null 2>&1; do sleep 300; done

CKPTS="$CK/camels531_fused_withq_s981.pt:$CK/camels531_fused_withq_s982.pt"
for f in $(echo "$CKPTS" | tr ":" " "); do
  [ -f "$f" ] || { echo "ABORT: missing $f" >> $LOG; exit 1; }
done
echo "[$(date +%T)] ckpts=$CKPTS" >> $LOG

# Protocol copied verbatim from the record members' backtest.
nice -n 5 $PY scripts/backtest_mblstm.py \
  --ckpt "$CKPTS" \
  --label "withqmulti_full531" \
  --corpus-dir "$CORP" \
  --start 1989-10-01 --end 1999-09-30 \
  --stride 14 --stride-stations 1 \
  --dump-windows "data/mblstm/gpu_dumps_s14/camels531_withqmulti_full531.csv.gz" \
  > benchmarks/mblstm_backtest_withqmulti_full531.json 2>> logs/backtest_withqmulti.log
echo "=== backtest rc=$? $(date +%T) ===" >> $LOG

# Verify the ARTIFACT, not the exit code.
D=data/mblstm/gpu_dumps_s14/camels531_withqmulti_full531.csv.gz
if gzip -t "$D" 2>>$LOG; then
  echo "[$(date +%T)] dump gzip OK $(ls -la $D | awk '{print $5}') bytes" >> $LOG
else
  echo "[$(date +%T)] WARNING dump failed gzip -t" >> $LOG
fi
echo "=== WITHQMULTI BACKTEST DONE $(date +%F_%T) ===" >> $LOG
