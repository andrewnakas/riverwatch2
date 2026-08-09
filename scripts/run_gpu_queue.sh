#!/bin/zsh
# Sequential GPU queue: wait for the running seed-101 trainer, then GFS
# fine-tune s101, then train seeds 102-104 (taking over the Windows worker's
# seeds 103/104). Skips a seed if its checkpoint already exists, so a
# checkpoint pulled from the worker branch is never re-trained.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
WAIT_PID=$1

if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
fi

if [ ! -f data/mblstm/model_h256_s101_gfsft.pt ]; then
  caffeinate -i $PY scripts/train_mblstm.py --gfs-finetune \
    --init-ckpt data/mblstm/model_h256_s101.pt \
    --epochs 6 --windows-per-station 300 --batch 256 --val-stride 5 \
    --lr 1e-4 --seed 101 --device mps \
    --out data/mblstm/model_h256_s101_gfsft.pt \
    > logs/mblstm_ft_gfs_s101.log 2>&1
fi

for SEED in 102 103 104; do
  if [ -f data/mblstm/model_h256_s${SEED}.pt ]; then
    echo "seed ${SEED}: checkpoint exists, skipping" >> logs/gpu_queue.log
    continue
  fi
  caffeinate -i $PY scripts/train_mblstm.py --compat-vars --epochs 12 \
    --windows-per-station 300 --hidden 256 --batch 256 --val-stride 20 \
    --lr 2e-4 --seed ${SEED} --device mps \
    --out data/mblstm/model_h256_s${SEED}.pt \
    > logs/mblstm_train_h256_s${SEED}.log 2>&1
done
echo "gpu queue complete $(date)" >> logs/gpu_queue.log
