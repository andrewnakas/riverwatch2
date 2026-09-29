#!/usr/bin/env bash
# Evaluate + dump NH seeds 502/503, then score the 3-seed NH member properly.
# Runs now because the GPU is free (NH training finished) and Daymet is ~46 basins
# from triggering the fused3m batch, which will want the card.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
LOG=logs/m1_nh_seeds.log
exec >>"$LOG" 2>&1
echo "=== NH SEEDS CHAIN START $(date +%F_%T) ==="
for S in 502 503; do
  RUN=$(ls -dt gpu1080/nh_runs/m1_nhar0m_s${S}_* 2>/dev/null | head -1)
  [ -z "$RUN" ] && { echo "no run dir for s$S"; continue; }
  NCK=$(ls "$RUN"/model_epoch*.pt 2>/dev/null | wc -l)
  [ "$NCK" -lt 30 ] && { echo "s$S only $NCK/30 epochs - skipping"; continue; }
  echo "[$(date +%T)] s$S run=$RUN ($NCK epochs)"
  for PER in validation test; do
    RES="$RUN/$PER/model_epoch030/${PER}_results.p"
    if [ ! -f "$RES" ]; then
      echo "[$(date +%T)] eval s$S $PER (holdout OFF)"
      nice -n 10 $PY scripts/nh_arlstm_eval.py --run-dir "$RUN" --period "$PER" --epoch 30 \
        > "logs/m1_nheval_s${S}_${PER}.log" 2>&1 || { echo "eval s$S $PER FAILED"; continue; }
    fi
    FR=val1; [ "$PER" = test ] && FR=test1
    OUT="data/mblstm/l51_dumps/m1_nhar0m_s${S}_${FR}.csv.gz"
    [ -f "$OUT" ] && { echo "[$(date +%T)] s$S $FR exists"; continue; }
    $PY scripts/nh_to_l51dump.py --results "$RES" --run-dir "$RUN" \
        --member nhar0m --seed "$S" --frame "$FR" --epoch 30 \
        --protocol modern-v3-9yr --basin-list data/m1_cohort.json --out "$OUT" || continue
    gzip -t "$OUT" && echo "[$(date +%T)] s$S $FR OK ($(zcat "$OUT"|tail -n +2|wc -l) rows)"
  done
done
echo "=== 3-SEED NH SCREEN: does the ensemble contribution hold at depth? ==="
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members gridmetm9,nclimgridm9,nhar0m --against gridmetm9,nclimgridm9 \
    --seeds 501,502,503 --corr --out benchmarks/m1_nh_ensemble_3seed.json
echo "=== NH SEEDS CHAIN DONE $(date +%F_%T) ==="
