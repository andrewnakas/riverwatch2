#!/bin/zsh
# CAMELS lever 2 — LEAN variant (2026-07-08) after 3 watchdog-timeout kernel
# panics traced to swap-thrash on the 8 GB box. Same recipe as
# run_b1_lever2_multiforcing.sh but tuned to keep peak RSS under the panic
# threshold:
#   - --batch 128 (was 256): halves activation memory, the main lever
#   - pre-seed swap guard: refuse to START a seed until swap < SWAP_OK_MB, so
#     we never begin training into an already-pressured kernel
#   - purge + settle between seeds (sync; let the kernel reclaim before the
#     next allocation spike)
#   - NOTHING else may run: the guard also waits out any train/backtest/fetch
# .partial + promote so a panic costs only the in-progress seed.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/b1_lever2.log
CORPUS=/Volumes/STORAGE_SD/riverwatch2_data/camels_corpus_3f
SWAP_OK_MB=6000   # don't start a seed while swap is above this (drain first)
echo "=== lever2 LEAN armed $(date)" >> $LOG
[ -d "$CORPUS" ] || { echo "3f corpus missing" >> $LOG; exit 1; }

swap_mb() { sysctl -n vm.swapusage | awk '{print int($6)}'; }

for SEED in 931 932 933 934 935 936 937 938; do
  OUT=data/mblstm/camels531_3f_mse_s${SEED}.pt
  [ -f "$OUT" ] && continue
  # No other heavy job.
  while pgrep -f "train_mblstm.p[y]" > /dev/null \
     || pgrep -f "backtest_mblstm.p[y]" > /dev/null \
     || pgrep -f "build_multimet_corpus.p[y]" > /dev/null; do sleep 300; done
  # Wait for swap to drain below the guard before starting (avoid panic).
  tries=0
  while [ "$(swap_mb)" -gt "$SWAP_OK_MB" ] && [ "$tries" -lt 60 ]; do
    sync; sleep 60; tries=$((tries+1))
  done
  echo "training 3f seed ${SEED} (swap $(swap_mb)MB) $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py \
    --corpus-dir "$CORPUS" --enc-vars camels3f --no-q-input --point-loss mse \
    --train-start 1999-10-01 --train-end 2008-09-30 \
    --val-start 1998-10-01 --val-end 1999-09-30 \
    --hidden 256 --epochs 16 --windows-per-station 1000 --batch 128 \
    --val-stride 20 --lr 2e-4 --seed ${SEED} --device mps \
    --out "$OUT.partial" > logs/camels531_3f_s${SEED}.log 2>&1 \
    && mv "$OUT.partial" "$OUT" \
    && echo "seed ${SEED} done (swap $(swap_mb)MB) $(date)" >> $LOG \
    || echo "seed ${SEED} FAILED/killed $(date)" >> $LOG
  rm -f "$OUT.partial"
  sync; sleep 30   # let the kernel reclaim before the next seed
done

CK=$(ls data/mblstm/camels531_3f_mse_s93*.pt 2>/dev/null | paste -sd: -)
NDONE=$(ls data/mblstm/camels531_3f_mse_s93*.pt 2>/dev/null | grep -vc partial)
if [ -n "$CK" ] && [ "$NDONE" -ge 4 ] && [ ! -f benchmarks/mblstm_backtest_camels531_3f_ens8.json ]; then
  while pgrep -f "backtest_mblstm.p[y]" > /dev/null; do sleep 300; done
  while [ "$(swap_mb)" -gt "$SWAP_OK_MB" ]; do sync; sleep 60; done
  echo "eval 3f ens (${NDONE} seeds) $(date)" >> $LOG
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CK" --corpus-dir "$CORPUS" \
    --start 1989-10-01 --end 1999-09-30 --stride 14 \
    --camels-subset 531 --label camels531_3f_ens8 \
    --dump-windows data/mblstm/dumps/camels531_3f_ens8.csv.gz \
    > logs/bt_camels531_3f_ens8.log 2>&1 \
    && echo "3f eval done $(date)" >> $LOG || echo "3f eval FAILED $(date)" >> $LOG
fi
echo "=== lever2 LEAN complete $(date)" >> $LOG
