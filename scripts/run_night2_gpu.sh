#!/bin/zsh
# Day-2 GPU chain: CMAL v2 seeds 102-104 with the PILOT-PROVEN recipe
# (perfect:0.4,gfs:0.6 mix — pilot s101 beat its quantile baseline:
# NSE +0.022, KGE +0.024, CRPS -5.4%). Waits for the current trainer (s106)
# to finish. The P3 queue (mixture-ft, needs GEFS) runs after these — its
# gate also waits for a free trainer.
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
LOG=logs/night2_gpu.log
echo "=== night2 gpu armed $(date)" >> $LOG
while pgrep -f "train_mblstm.py" > /dev/null; do sleep 600; done

for SEED in 102 103 104; do
  OUT=data/mblstm/model_h256_s${SEED}_cmalv2p.pt
  [ -f "$OUT" ] && continue
  echo "starting cmalv2p s${SEED} $(date)" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py \
    --init-ckpt data/mblstm/model_h256_s${SEED}_gfsft.pt --head cmal --cmal-k 3 \
    --forcing-mix "perfect:0.4,gfs:0.6" --forcing-noise 0.15 \
    --epochs 14 --windows-per-station 300 --batch 256 --val-stride 5 \
    --lr 2e-4 --seed ${SEED} --device mps \
    --out "$OUT" > logs/mblstm_cmalv2p_s${SEED}.log 2>&1
  [ -f "$OUT" ] || echo "cmalv2p s${SEED} FAILED" >> $LOG
done
echo "=== night2 gpu complete $(date)" >> $LOG
