#!/bin/zsh
# Auto-retrain orchestrator: build the full-13-var Open-Meteo model once the
# corpus is large enough, then backtest it against the frozen SOTA baseline.
#
# Idempotent + resumable: re-running syncs new stations, joins discharge for any
# that lack it, skips seeds whose checkpoint already exists, and only fires the
# backtest after all 4 seeds are present. The /loop babysitter calls this each
# wake; it's a no-op until the corpus crosses MIN_STATIONS.
#
# Usage: zsh scripts/run_openmeteo_retrain.sh [MIN_STATIONS]
set -u
cd /Users/nakas/Documents/RiverWatch2/riverwatch2
PY=.venv/bin/python
CORPUS=data/mblstm/corpus_openmeteo
MIN_STATIONS=${1:-1500}
LOG=logs/openmeteo_retrain.log
mkdir -p logs
echo "=== $(date) orchestrator tick (min=$MIN_STATIONS) ===" >> $LOG

# 1. Sync any newly-published stations + join discharge (both resumable, cheap
#    when nothing changed).
$PY scripts/sync_openmeteo_corpus.py >> $LOG 2>&1
$PY scripts/sync_openmeteo_corpus.py --join-discharge >> $LOG 2>&1

# 2. Gate on completeness: count stations that actually have a q_cfs column.
READY=$($PY - <<'EOF'
import glob, pandas as pd
n=0
for p in glob.glob("data/mblstm/corpus_openmeteo/*.csv.gz"):
    try:
        if "q_cfs" in pd.read_csv(p, nrows=1).columns: n+=1
    except Exception: pass
print(n)
EOF
)
echo "ready stations (with discharge): $READY / need $MIN_STATIONS" >> $LOG
if [ "$READY" -lt "$MIN_STATIONS" ]; then
  echo "corpus still filling — orchestrator no-op this tick" >> $LOG
  echo "WAIT $READY"   # parsed by the loop
  exit 0
fi

# 3. Train 4 seeds, enc-13/dec-5 split, sequential. The 13-var encoder keeps
#    the soil/snow/wind signal; the 5-var (compat) decoder stays drivable by
#    the GFS/GEFS/ECMWF forecast archives, so the om13 model can be
#    forcing-fine-tuned and honestly backtested under real forecast error
#    (an enc=dec=13 model could not). Skip any seed whose checkpoint already
#    exists (crash/resume safe). lr 2e-4 (the stable h256 setting from v16).
for SEED in 201 202 203 204; do
  OUT=data/mblstm/model_h256_om13_s${SEED}.pt
  if [ -f "$OUT" ]; then echo "seed $SEED exists, skip" >> $LOG; continue; fi
  echo "=== training seed $SEED ($(date)) ===" >> $LOG
  caffeinate -i $PY scripts/train_mblstm.py --corpus-dir $CORPUS \
    --enc-vars full --dec-vars compat \
    --epochs 12 --windows-per-station 300 --hidden 256 --batch 256 \
    --val-stride 20 --lr 2e-4 --seed $SEED --device mps \
    --out $OUT >> logs/train_om13_s${SEED}.log 2>&1
  if [ ! -f "$OUT" ]; then echo "seed $SEED FAILED (no ckpt)" >> $LOG; echo "TRAINFAIL $SEED"; exit 1; fi
done

# 4. Ensemble backtest vs frozen baseline (perfect-forcing here; the Open-Meteo
#    serve path uses live OM forecasts, so this measures the encoder/forcing
#    upgrade. GFS-forcing A/B comes after if this wins).
CK=data/mblstm/model_h256_om13_s201.pt:data/mblstm/model_h256_om13_s202.pt:data/mblstm/model_h256_om13_s203.pt:data/mblstm/model_h256_om13_s204.pt
if [ ! -f benchmarks/mblstm_backtest_om13_ens.json ]; then
  echo "=== backtesting om13 ensemble ($(date)) ===" >> $LOG
  RW2_ENABLE_MBLSTM=1 caffeinate -i $PY scripts/backtest_mblstm.py \
    --ckpt $CK --camels-subset 531 --label om13_ens >> $LOG 2>&1
fi
echo "DONE"   # parsed by the loop
exit 0
