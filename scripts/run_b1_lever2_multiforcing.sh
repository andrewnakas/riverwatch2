#!/bin/zsh
# CAMELS lever 2 — Daymet+Maurer+NLDAS multi-forcing (the published 0.74→0.82
# jump). Built + CPU-validated 2026-07-06; runs AFTER lever 1 (ens8) and the
# serialized extractions finish (needs maurer/nldas fully extracted). Verify
# BOTH corpora exist before launching.
#
# Step 1 (CPU, no GPU): build the merged 3-forcing corpus once.
#   .venv/bin/python scripts/build_camels_corpus.py \
#     --forcing-set daymet,maurer,nldas --merge \
#     --out-dir /Volumes/STORAGE_SD/riverwatch2_data/camels_corpus_3f
# Step 2 (MPS, one at a time): 8 seeds, strict protocol, 15-var encoder.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/b1_lever2.log
CORPUS=/Volumes/STORAGE_SD/riverwatch2_data/camels_corpus_3f
echo "=== lever2 armed $(date)" >> $LOG
[ -d "$CORPUS" ] || { echo "3f corpus missing — run step 1 first" >> $LOG; exit 1; }
for SEED in 931 932 933 934 935 936 937 938; do
  OUT=data/mblstm/camels531_3f_mse_s${SEED}.pt
  [ -f "$OUT" ] && continue
  while pgrep -f "train_mblstm.p[y]" > /dev/null || pgrep -f "backtest_mblstm.p[y]" > /dev/null; do sleep 600; done
  echo "training 3f seed ${SEED} $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py \
    --corpus-dir "$CORPUS" --enc-vars camels3f --no-q-input --point-loss mse \
    --train-start 1999-10-01 --train-end 2008-09-30 \
    --val-start 1998-10-01 --val-end 1999-09-30 \
    --hidden 256 --epochs 16 --windows-per-station 1000 --batch 256 \
    --val-stride 20 --lr 2e-4 --seed ${SEED} --device mps \
    --out "$OUT.partial" > logs/camels531_3f_s${SEED}.log 2>&1 \
    && mv "$OUT.partial" "$OUT"
  rm -f "$OUT.partial"
done
CK=$(ls data/mblstm/camels531_3f_mse_s93*.pt 2>/dev/null | paste -sd: -)
if [ -n "$CK" ] && [ ! -f benchmarks/mblstm_backtest_camels531_3f_ens8.json ]; then
  while pgrep -f "backtest_mblstm.p[y]" > /dev/null; do sleep 600; done
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt "$CK" --corpus-dir "$CORPUS" \
    --start 1989-10-01 --end 1999-09-30 --stride 14 \
    --camels-subset 531 --label camels531_3f_ens8 \
    --dump-windows data/mblstm/dumps/camels531_3f_ens8.csv.gz \
    > logs/bt_camels531_3f_ens8.log 2>&1
fi
echo "=== lever2 complete $(date)" >> $LOG
