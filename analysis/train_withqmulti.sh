#!/usr/bin/env bash
# withqmulti: a MULTI-FORCING with-q member (camels3fv2, 18 fused vars).
# Pre-registered gate: memory note `withq-multi-prereg-gate`.
#
# Protocol copied VERBATIM from train_aorc_s985.sh (which produced the record
# members). The ONLY deviations are --enc-vars and --corpus-dir. Any other
# deviation would compare against the wrong baseline.
#
# flock guard: two different queue scripts once raced the same seed for 7h and
# shipped a mid-training dump as an official seed.
set -u
cd ~/riverwatch2
exec 9>/tmp/rw2_withqmulti.lock
flock -n 9 || { echo "another withqmulti launcher holds the lock; exiting"; exit 0; }

PY=gpu1080/.venv/bin/python
CK=data/mblstm/gpu_ckpts
CORP=gpu1080/corpora671/camels_corpus_fused_v2
LOG=logs/withqmulti.log
mkdir -p logs benchmarks "$CK"

echo "=== WITHQMULTI START $(date +%F_%T) ===" >> $LOG

# Wait for any GPU work to clear. Pattern built from fragments so this script's
# own command line cannot self-match (a self-matching pgrep deadlocked the box).
PAT_NH="nh_run"" train"
while pgrep -f "$PAT_NH" >/dev/null 2>&1; do sleep 300; done
echo "[$(date +%T)] GPU free" >> $LOG

# GUARD 1: the corpus must cover all 531 benchmark basins.
N=$($PY -c "
import json,glob,os
ids=set(str(x).zfill(8) for x in json.load(open('data/camels_gauge_ids.json'))['531'])
have=set(os.path.basename(f)[:8] for f in glob.glob('$CORP/*.csv.gz'))
print(len(ids & have))
")
echo "[$(date +%T)] corpus covers $N/531 benchmark basins" >> $LOG
[ "$N" -eq 531 ] || { echo "[$(date +%T)] ABORT: expected 531" >> $LOG; exit 1; }

# GUARD 2: the corpus must carry every camels3fv2 variable, non-NaN.
$PY -c "
import pandas as pd, glob, sys
V=[f'{v}_{p}' for p in ('daymet','maurer','nldas')
   for v in ['precipitation_sum','temperature_2m_max','temperature_2m_min',
             'temperature_2m_mean','shortwave_radiation_sum','vapor_pressure']]
f=sorted(glob.glob('$CORP/*.csv.gz'))[0]
d=pd.read_csv(f)
miss=[v for v in V if v not in d.columns]
if miss: print('MISSING', miss); sys.exit(1)
bad=[v for v in V if d[v].isna().mean() > 0.05]
if bad: print('HIGH-NaN', bad); sys.exit(1)
print(f'18/18 vars present, max NaN {max(d[v].isna().mean() for v in V):.4f}')
" >> $LOG 2>&1 || { echo "[$(date +%T)] ABORT: corpus var check failed" >> $LOG; exit 1; }

# ---- train seeds -------------------------------------------------------
SEEDS="971 972 973"
for S in $SEEDS; do
  CKPT="$CK/camels531_withqmulti_s${S}.pt"
  if [ -f "$CKPT" ]; then
    echo "[$(date +%T)] s$S already trained, skipping" >> $LOG
    continue
  fi
  echo "=== training s$S $(date +%F_%T) ===" >> $LOG
  nice -n 5 $PY scripts/train_mblstm.py \
    --epochs 30 --seed "$S" --out "$CKPT" \
    --head quantile --enc-vars camels3fv2 --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start 1998-10-01 --val-end 1999-09-30 \
    > "logs/withqmulti_s${S}.log" 2>&1
  RC=$?
  echo "=== s$S trained rc=$RC $(date +%T) ===" >> $LOG
  [ -f "$CKPT" ] || { echo "[$(date +%T)] ABORT: no checkpoint for s$S" >> $LOG; exit 1; }
done

# ---- backtest the seed-averaged member (seed averaging is at INFERENCE) ----
CKPTS=""
for S in $SEEDS; do
  P="$CK/camels531_withqmulti_s${S}.pt"
  [ -f "$P" ] && CKPTS="${CKPTS:+$CKPTS:}$P"
done
echo "=== backtest ckpts=$CKPTS $(date +%F_%T) ===" >> $LOG
nice -n 10 $PY scripts/backtest_mblstm.py \
  --ckpt "$CKPTS" \
  --label "withqmulti_full531" \
  --corpus-dir "$CORP" \
  --start 1989-10-01 --end 1999-09-30 \
  --stride 14 --stride-stations 1 \
  --dump-windows "data/mblstm/gpu_dumps_s14/camels531_withqmulti_full531.csv.gz" \
  > benchmarks/mblstm_backtest_withqmulti_full531.json 2>> logs/backtest_withqmulti.log
echo "=== backtest rc=$? $(date +%T) ===" >> $LOG

# GUARD 3: verify the artifact, not the exit code.
gzip -t data/mblstm/gpu_dumps_s14/camels531_withqmulti_full531.csv.gz 2>>$LOG \
  && echo "[$(date +%T)] dump gzip OK" >> $LOG \
  || echo "[$(date +%T)] WARNING dump failed gzip -t" >> $LOG
ls -la data/mblstm/gpu_dumps_s14/camels531_withqmulti_full531.csv.gz >> $LOG 2>&1
echo "=== WITHQMULTI ALL DONE $(date +%F_%T) ===" >> $LOG
