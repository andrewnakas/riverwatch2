#!/bin/bash
# Run the CAMELS no-q combined-loss δHBV campaign on a Lightning AI Studio (or any
# CUDA GPU box). Trains the decorrelation-loss δHBV members that were NEVER trained
# into the shipped ckpts, then dumps them for the local grand-ensemble combine.
#
# Prereq: bash lightning_setup.sh  (repo + data + deps + GPU verified)
# Usage:  bash notebooks/run_campaign.sh [forcings] [seeds] [epochs]
#   bash notebooks/run_campaign.sh "daymet maurer nldas" "971 972" 50
set -u
cd "$(dirname "$0")/.."
export RW2_ENABLE_MBLSTM=1 PYTHONUNBUFFERED=1
FORCINGS=${1:-"daymet maurer nldas"}
SEEDS=${2:-"971"}
EPOCHS=${3:-50}
DATA=${DATA:-data/gpu_corpora}
mkdir -p data/mblstm/gpu_ckpts data/mblstm/gpu_dumps_s14 logs

# recipe-v2 δHBV + the fidelity fix (--dhbv-loss combined). windows-per-station 150
# was P100-sized; on a faster T4/A10 you can raise to 300-500 for better members.
WPS=${WPS:-300}

echo "=== CAMELS no-q δHBV campaign: forcings=[$FORCINGS] seeds=[$SEEDS] epochs=$EPOCHS wps=$WPS ==="
for F in $FORCINGS; do
  CORPUS="$DATA/camels_corpus_${F}_v2"
  [ -d "$CORPUS" ] || { echo "SKIP $F — no corpus at $CORPUS"; continue; }
  for S in $SEEDS; do
    OUT="data/mblstm/gpu_ckpts/camels531_${F}_dhbv_combined${EPOCHS}_s${S}.pt"
    if [ -f "$OUT" ]; then echo "skip (exists) $OUT"; continue; fi
    echo ">>> train δHBV $F seed $S $(date +%T)"
    python scripts/train_mblstm.py --corpus-dir "$CORPUS" \
      --no-q-input --head dhbv --nmul 16 --dhbv-loss combined --forcing-correction \
      --enc-vars camels1f --static-set camels --q-transform linear \
      --hidden 256 --windows-per-station "$WPS" --batch 256 --val-stride 10 \
      --train-start 1999-10-01 --train-end 2008-09-30 \
      --val-start 1998-10-01 --val-end 1999-09-30 \
      --epochs "$EPOCHS" --seed "$S" --device cuda --out "$OUT" \
      2>&1 | tee "logs/train_${F}_c${EPOCHS}_s${S}.log"
    echo ">>> done $F s$S rc=$? $(date +%T)"
  done
done

echo "=== dumps (stride-14/ss3, 177-basin screen — asserts CAMELS static overlay) ==="
for F in $FORCINGS; do
  CK=$(ls data/mblstm/gpu_ckpts/camels531_${F}_dhbv_combined${EPOCHS}_s*.pt 2>/dev/null | paste -sd: -)
  [ -z "$CK" ] && continue
  DUMP="data/mblstm/gpu_dumps_s14/camels531_${F}_combined${EPOCHS}_ens_s14.csv.gz"
  echo ">>> dump $F $(date +%T)"
  python scripts/backtest_mblstm.py --ckpt "$CK" --corpus-dir "$DATA/camels_corpus_${F}_v2" \
    --start 1989-10-01 --end 1999-09-30 --stride 14 --stride-stations 3 --camels-subset 531 \
    --label "${F}_combined${EPOCHS}" --dump-windows "$DUMP" \
    2>&1 | tee "logs/dump_${F}_c${EPOCHS}.log" | grep -iE "CAMELS static overlay|median NSE"
done

echo "=== DONE. Pull these back to local for the grand-ensemble combine: ==="
ls -la data/mblstm/gpu_ckpts/camels531_*_combined${EPOCHS}_s*.pt 2>/dev/null
ls -la data/mblstm/gpu_dumps_s14/camels531_*_combined${EPOCHS}_ens_s14.csv.gz 2>/dev/null
