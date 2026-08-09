#!/bin/zsh
# Sequential GFS fine-tunes of seeds 102-104 (same recipe as s101's ft).
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
for SEED in 102 103 104; do
  [ -f data/mblstm/model_h256_s${SEED}_gfsft.pt ] && continue
  caffeinate -i $PY scripts/train_mblstm.py --gfs-finetune \
    --init-ckpt data/mblstm/model_h256_s${SEED}.pt \
    --epochs 6 --windows-per-station 300 --batch 256 --val-stride 5 \
    --lr 1e-4 --seed ${SEED} --device mps \
    --out data/mblstm/model_h256_s${SEED}_gfsft.pt \
    >> logs/mblstm_ft_queue.log 2>&1
done
echo "ft queue complete $(date)" >> logs/mblstm_ft_queue.log
