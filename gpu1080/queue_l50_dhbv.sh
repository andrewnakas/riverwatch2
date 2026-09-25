#!/usr/bin/env bash
# LEDGER 50 Stage 1 — the fused δHBV probe (dhbvM = camels3fv2 encoder + forcing correction).
# Pre-registered in PREREG_v2.md (LEDGER 50) BEFORE launch. Matched recipe except the two flags.
# Usage: bash gpu1080/queue_l50_dhbv.sh <SEED> [WPS] [EPOCHS]   (defaults 1000 / 50 = shipped recipe)
# Idempotent on the CHECKPOINT (.partial -> mv promote). Kill by PID only (never pkill -f).
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
export PYTHONUNBUFFERED=1
S=${1:?seed}; WPS=${2:-1000}; EPOCHS=${3:-50}
CORP=gpu1080/corpora671/camels_corpus_fused_v2
CK=gpu1080/dhbv_ckpts/camels531ls_daymet_dhbvM_s${S}.pt
LOG=logs/l50_dhbvM_s${S}.log
[ -d "$CORP" ] || { echo "MISSING corpus $CORP"; exit 2; }
if [ -f "$CK" ]; then echo "[skip] $CK exists"; else
  echo "=== L50 dhbvM TRAIN s$S wps=$WPS epochs=$EPOCHS $(date +%F_%T) pid=$$ ==="
  $PY scripts/train_mblstm.py \
    --corpus-dir "$CORP" --enc-vars camels3fv2 --forcing-correction --no-q-input \
    --q-transform linear --static-set camels --head dhbv --dhbv-loss mse --nmul 16 \
    --train-start 1980-10-01 --train-end 1995-09-30 \
    --val-start 1994-10-01 --val-end 1995-09-30 \
    --hidden 256 --epochs "$EPOCHS" --windows-per-station "$WPS" --batch 256 \
    --val-stride 10 --lr 1e-3 --seed "$S" --device cuda \
    --out "${CK}.partial" > "$LOG" 2>&1 \
    && mv "${CK}.partial" "$CK" && echo "=== L50 dhbvM DONE s$S $(date +%F_%T) ===" \
    || { echo "=== L50 dhbvM FAIL s$S (see $LOG) ==="; rm -f "${CK}.partial"; exit 1; }
fi
# ---- dumps (inference only): TEST and TRAIN on the LSTM-aligned stride-14 grids, all 14 leads.
# ⚠️ with --init-ckpt enc_vars come from the ckpt, so the corpus MUST be the fused one.
dump() {  # $1 = TEST|TRAIN
  local ALLOW DUMP VS VE
  if [ "$1" = TEST ]; then ALLOW=dhbv_dumps/_dhbv_t0_allowlist.json; DUMP=dhbv_dumps/camels531ls_daymet_dhbvM_ALLH_s${S}.csv.gz; VS=1995-10-01; VE=2010-09-30
  else ALLOW=dhbv_dumps/_dhbv_t0_allowlist_train.json; DUMP=dhbv_dumps/camels531ls_daymet_dhbvM_ALLH_TRAIN_s${S}.csv.gz; VS=1980-10-01; VE=1995-09-30; fi
  if [ -f "$DUMP" ] && gzip -t "$DUMP" 2>/dev/null && [ "$(zcat "$DUMP" | tail -n +2 | cut -d, -f3 | sort -un | wc -l)" -eq 14 ]; then echo "[skip] $DUMP"; return 0; fi
  echo "=== L50 dhbvM DUMP $1 s$S $(date +%T) ==="
  nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CK" \
    --val-start "$VS" --val-end "$VE" --val-stride 1 --dump-allowlist "$ALLOW" \
    --head dhbv --enc-vars camels3fv2 --no-q-input --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --dump-all-leads --dump-day1 "$DUMP" > "logs/l50_dhbvM_dump_$1_s${S}.log" 2>&1
  gzip -t "$DUMP" 2>/dev/null && echo "=== DUMP $1 OK rows=$(zcat "$DUMP" | wc -l) basins=$(zcat "$DUMP" | tail -n +2 | cut -d, -f1 | sort -u | wc -l) ===" || echo "=== DUMP $1 FAIL ==="
}
dump TEST; dump TRAIN
echo "=== L50 dhbvM s$S ALL DONE $(date +%F_%T) ==="
