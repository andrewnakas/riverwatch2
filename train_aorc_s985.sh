#!/usr/bin/env bash
# Train a 5th AORC with-q seed (s985) and re-backtest the member on 531 basins.
#
# Two gaps close together here:
#   1. AORC contributes 4 with-q seeds against the other forcings' 5, so it is
#      the member that is BEHIND. Measured 4->5 gains on the others were +0.011
#      to +0.017 per member, which diluted across 4 members projects to ~+0.003
#      on the ensemble: 0.9196 -> ~0.9226.
#   2. The current AORC dump covers 530 basins. The backtest started 04:10 but
#      the reconstructed 531st basin only landed 15:37, so the dump predates it.
#      Retraining and re-dumping now picks up the full corpus.
#
# Seed averaging happens AT INFERENCE: app/mblstm.py splits the checkpoint path
# on ":" and averages forecasts across them, and backtest_mblstm.py --ckpt passes
# that through. So the member is ONE dump from FIVE checkpoints, not five dumps.
#
# Protocol copied verbatim from backtest_withq_5seed.sh, which produced the
# baseline dumps: window 1989-10-01..1999-09-30, --stride 14, --stride-stations 1.
# Any deviation would compare against the wrong baseline.
#
# GATE: waits for the neuralhydrology retrain to finish. Gate patterns are built
# from string fragments so this script's own command line cannot match them --
# a self-matching pgrep deadlocked this box for an hour earlier in the campaign.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
CK=data/mblstm/gpu_ckpts
CORP=gpu1080/corpora671/camels_corpus_aorc_v2
LOG=logs/aorc_s985.log
mkdir -p logs benchmarks

echo "=== AORC s985 START $(date +%F_%T) ===" >> $LOG

PAT_NH="nh_run"" train"
while pgrep -f "$PAT_NH" >/dev/null 2>&1; do sleep 300; done
echo "[$(date +%T)] GPU free" >> $LOG

# sanity: the corpus must actually hold 531 benchmark basins now
N=$($PY -c "
import json,glob,os
ids=set(str(x).zfill(8) for x in json.load(open('data/camels_gauge_ids.json'))['531'])
have=set(os.path.basename(f)[:8] for f in glob.glob('$CORP/*.csv.gz'))
print(len(ids & have))
")
echo "[$(date +%T)] corpus covers $N/531 benchmark basins" >> $LOG
[ "$N" -eq 531 ] || { echo "[$(date +%T)] ABORT: expected 531" >> $LOG; exit 1; }

CKPT="$CK/camels531_aorc_withq_s985.pt"
if [ ! -f "$CKPT" ]; then
  echo "=== training s985 $(date +%F_%T) ===" >> $LOG
  nice -n 5 $PY scripts/train_mblstm.py \
    --epochs 30 --seed 985 --out "$CKPT" \
    --head quantile --enc-vars camels1f --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start 1998-10-01 --val-end 1999-09-30 \
    > logs/aorc_withq_s985.log 2>&1
  echo "=== s985 trained rc=$? $(date +%T) ===" >> $LOG
fi
[ -f "$CKPT" ] || { echo "[$(date +%T)] ABORT: no checkpoint" >> $LOG; exit 1; }

# re-backtest the member with ALL FIVE seeds over the now-531-basin corpus
CKPTS="$CK/camels531_aorc_withq_s981.pt:$CK/camels531_aorc_withq_s982.pt:$CK/camels531_aorc_withq_s983.pt:$CK/camels531_aorc_withq_s984.pt:$CKPT"
echo "=== backtest 5-seed on 531 $(date +%F_%T) ===" >> $LOG
nice -n 10 $PY scripts/backtest_mblstm.py \
  --ckpt "$CKPTS" \
  --label "aorc_withq5_full531" \
  --corpus-dir "$CORP" \
  --start 1989-10-01 --end 1999-09-30 \
  --stride 14 --stride-stations 1 \
  --dump-windows "data/mblstm/gpu_dumps_s14/camels531_aorc_withq5_full531.csv.gz" \
  > benchmarks/mblstm_backtest_aorc_withq5_full531.json 2>> logs/backtest_aorc_withq5.log
echo "=== backtest rc=$? $(date +%T) ===" >> $LOG
ls -la data/mblstm/gpu_dumps_s14/camels531_aorc_withq5_full531.csv.gz >> $LOG 2>&1
echo "=== AORC s985 ALL DONE $(date +%F_%T) ===" >> $LOG
