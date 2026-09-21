#!/usr/bin/env bash
# LEDGER 54 — score the outstanding PRE-REGISTERED predictions as their dumps land.
# Each block waits for exactly the dumps it needs, then writes its verdict. Nothing
# here selects anything: val1 decides, test1 is read after, per PREREG_v2.md §L54.
set -u; cd ~/riverwatch2
PY="nice -n 19 gpu1080/.venv/bin/python"; LOG=logs/l54_predictions.log; O=benchmarks/l54
B=fused3h1,daymeth1,nldash1,maurerh1,aorch1
mkdir -p "$O"
exec 9>/tmp/rw2_l54_pred.lock; flock -n 9 || exit 0
wait_for () { while :; do ok=1; for f in "$@"; do [ -f "$f" ] || ok=0; done; [ "$ok" = 1 ] && return; sleep 300; done; }
D=data/mblstm/l51_dumps/camels531_l51

# ---- BLOCK 1: TARGET (2), a single member at 0.89. Prediction: 3-seed val1 0.8905..0.8925
wait_for ${D}_nhar0h256_s503_val1.csv.gz
{ echo "=== BLOCK 1 (target 2: single member 0.89) $(date +%F_%T) ==="
  $PY analysis/l54_seed_depth.py --member nhar0h256 --frame val1 --out $O/SEEDDEPTH_h256_val1_3seed.json
  echo "--- PREDICTION was 3-seed val1 solo in 0.8905..0.8925; >=0.89 means TARGET 2 ACHIEVED"
} >> "$LOG" 2>&1

# ---- BLOCK 2: TARGET (3) firmed up at 3 seeds
wait_for ${D}_nhar0h256_s503_test1.csv.gz
{ echo "=== BLOCK 2 (target 3: single network) $(date +%F_%T) ==="
  $PY analysis/l54_targets_verdict.py --candidates nhar0h256 --out $O/TARGETS_VERDICT_3seed.json
} >> "$LOG" 2>&1

# ---- BLOCK 3: h512 falsifier — 3-seed val1 solo must beat h256's, else capacity saturated
wait_for ${D}_nhar0h512_s501_val1.csv.gz ${D}_nhar0h512_s502_val1.csv.gz ${D}_nhar0h512_s503_val1.csv.gz
{ echo "=== BLOCK 3 (h512 capacity falsifier) $(date +%F_%T) ==="
  for m in nhar0h256 nhar0h512; do $PY analysis/l51_withq_score.py --frame val1 --members $m --seeds 501,502,503 2>&1 | grep -E "member $m"; done
  echo "--- FALSIFIER: if h512 3-seed val1 <= h256 3-seed val1, capacity is saturated -> stop"
  $PY analysis/l54_seed_depth.py --member nhar0h512 --frame val1 --out $O/SEEDDEPTH_h512_val1.json
} >> "$LOG" 2>&1

# ---- BLOCK 4: the 0.90 attempt at the ship bar (h256 x5), decided on val1 then read on test1
wait_for ${D}_nhar0h256_s505_val1.csv.gz ${D}_nhar0h256_s505_test1.csv.gz
{ echo "=== BLOCK 4 (0.90 attempt, h256 at 5 seeds) $(date +%F_%T) ==="
  for FR in val1 test1; do
    echo "--- $FR  8-member"
    $PY analysis/l51_withq_score.py --frame $FR --members $B,nhar,nhar0,nhar0h256 --loo --bootstrap \
        --out $O/ENS8_${FR}_h256x5.json 2>&1 | grep -E "^ENSEMBLE|drop nhar0h256"
  done
  echo "--- PREDICTION was 8-member test1 at 5 h256 seeds = 0.8985..0.9005"
} >> "$LOG" 2>&1
echo "=== all blocks done $(date +%F_%T) ===" >> "$LOG"
