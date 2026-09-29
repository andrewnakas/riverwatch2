#!/usr/bin/env bash
# MODERN-1 — the 3-PRODUCT NH AR-LSTM member (nhar0m3).
#
# Why: nhar0m has the BEST solo of the roster (val1 0.873176, 3 seeds) and it trained
# on only TWO products -- 10 dynamic inputs, gridMET + nClimGrid (verified from its
# own config.yml, not assumed). "+1 forcing product" is the largest lever measured on
# this corpus (~+0.0165), so adding Daymet to the strongest single member is the
# cheapest remaining upside. 15 dynamic inputs instead of 10.
#
# ⚠️⚠️ GPU SAFETY. This chain must never race run_m1_batch.sh for the card: two
# independent waiters that each poll "is the GPU free" will both claim the same slot,
# and a job is invisible to nvidia-smi for MINUTES while it loads its corpus. That is
# how this 8 GB card hit a CUDA OOM once misdiagnosed as a hardware fault. So the CPU
# work happens immediately, and the GPU wait additionally requires that NO batch
# controller or member launcher is alive -- checked with character-class patterns so
# the check cannot match its own command line.
#
# ⚠️ The wait condition is REACHABLE IN BOTH BRANCHES: it does not wait for chain2 to
# SUCCEED (if Gate A fails, chain2 halts and daymetm never runs). It waits for the box
# to be quiet, which happens either way. An unreachable wait condition once spun for
# three weeks here.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
V3=data/modern_corpus/v3
ND=gpu1080/nh_data_multim3
LOG=logs/m1_nh3_chain.log
exec >>"$LOG" 2>&1
echo "=== NH3 CHAIN START $(date +%F_%T) ==="

# ---- 1. wait for daymet_m3 (chain2 assembles it), stall-keyed ---------------
LAST=-1; STALL=0
for i in $(seq 1 2880); do
  n=$(ls "$V3/camels_corpus_daymet_m3"/*.csv.gz 2>/dev/null | wc -l)
  [ "$n" -ge 671 ] && { echo "[$(date +%T)] daymet_m3 ready: $n/671"; break; }
  if [ "$n" -gt "$LAST" ]; then LAST=$n; STALL=0; else STALL=$((STALL + 1)); fi
  [ "$STALL" -ge 120 ] && { echo "daymet_m3 absent/stalled at $n/671 for 4 h - stopping"; exit 2; }
  sleep 120
done
[ "$(ls "$V3/camels_corpus_daymet_m3"/*.csv.gz 2>/dev/null | wc -l)" -lt 671 ] && exit 2

# ---- 2. CPU WORK: build the 3-product NH dataset (safe beside GPU jobs) -----
if [ ! -f "$ND/basins.txt" ]; then
  echo "[$(date +%T)] building $ND from 3 products"
  nice -n 15 $PY scripts/corpus_to_nh_multi.py \
    --products "daymet=$V3/camels_corpus_daymet_m3,gridmet=$V3/camels_corpus_gridmet_m3,nclimgrid=$V3/camels_corpus_nclimgrid_m3" \
    --basin-list data/m1_cohort.json --out "$ND" 2>&1 | tail -8
fi
NB=$(wc -l < "$ND/basins.txt" 2>/dev/null || echo 0)
NEED=$($PY -c 'import json;print(len(json.load(open("data/m1_cohort.json"))["cohort"]))')
echo "[$(date +%T)] $ND: $NB basins (cohort $NEED)"
[ "$NB" != "$NEED" ] && { echo "ARTIFACT FAIL: $NB != $NEED basins"; exit 3; }

# ---- 3. render + pathcheck the configs (CPU) -------------------------------
for S in 501 502 503; do
  CFG="gpu1080/cfg_nhar0m3_s${S}.yml"
  if [ ! -f "$CFG" ]; then
    $PY scripts/render_nh_arlstm_cfg.py --variant nhar0m3 --base gpu1080 \
      --data-dir nh_data_multim3 --forcings daymet,gridmet,nclimgrid \
      --basin-file nh_data_multim3/basins.txt --out "$CFG" \
      --seed "$S" --hidden 256 --holdout 0.0 --epochs 30 \
      --protocol modern-v3-9yr --exp-prefix m1 || exit 4
  fi
  $PY scripts_modern/nh_pathcheck.py "$CFG" || { echo "PATHCHECK FAILED $CFG"; exit 4; }
  # assert the arm actually applied: 3 products x 5 vars = 15 dynamic inputs
  $PY - "$CFG" <<'PYC'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
d = c["dynamic_inputs"]
assert len(d) == 15, f"{sys.argv[1]}: {len(d)} dynamic inputs, expected 15 (3 products x 5 vars): {d}"
prods = sorted({v.rsplit("_", 1)[1] for v in d})
assert prods == ["daymet", "gridmet", "nclimgrid"], prods
assert c["train_start_date"] == "01/10/2005", c["train_start_date"]
assert c["random_holdout_from_dynamic_features"]["q_mm_shift1"]["missing_fraction"] == 0.0
print(f"  cfg OK: {sys.argv[1]} 15 inputs, products {prods}")
PYC
  [ $? -eq 0 ] || exit 4
done

# ---- 4. GPU WAIT: idle AND no batch controller / member launcher alive ------
echo "[$(date +%T)] CPU prep done. Waiting for an exclusive GPU slot."
QUIET=0
for i in $(seq 1 1440); do            # up to 48 h
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  B=$(ps -eo cmd --no-headers | grep -c "run_m1_batc[h]\.sh")
  Q=$(ps -eo cmd --no-headers | grep -c "queue_m1_with[q]\.sh")
  if [ "$N" -eq 0 ] && [ "$B" -eq 0 ] && [ "$Q" -eq 0 ]; then
    QUIET=$((QUIET + 1)); else QUIET=0
  fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] exclusive slot: gpu=0 batch=0 queue=0 on 3 polls"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: gpu=$N batch=$B queue=$Q quiet=$QUIET/3"
  sleep 120
done
N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
B=$(ps -eo cmd --no-headers | grep -c "run_m1_batc[h]\.sh")
{ [ "$N" -ne 0 ] || [ "$B" -ne 0 ]; } && { echo "never got an exclusive slot (gpu=$N batch=$B)"; exit 5; }

# ---- 5. train one seed at a time, verifying the ARTIFACT not the rc ---------
for S in 501 502 503; do
  RUN=$(ls -dt gpu1080/nh_runs/m1_nhar0m3_s${S}_* 2>/dev/null | head -1)
  if [ -n "$RUN" ] && [ "$(ls "$RUN"/model_epoch*.pt 2>/dev/null | wc -l)" -ge 30 ]; then
    echo "[$(date +%T)] s$S already trained ($RUN)"; continue
  fi
  A=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  echo "[$(date +%T)] training nhar0m3 s$S (${A}MiB RAM available)"
  # --no-graph: CUDA graph capture fails on this Pascal card. The 63x speedup comes
  # from the FUSED cuDNN path (holdout 0.0 => no missing lagged inputs => ARLSTM's
  # step loop degenerates to one cuDNN call), not from graphs.
  nice -n 5 $PY scripts/nh_arlstm_train.py --config-file "gpu1080/cfg_nhar0m3_s${S}.yml" \
    --no-graph > "logs/m1_nhtrain_nhar0m3_s${S}.log" 2>&1
  echo "[$(date +%T)] s$S reported rc=$? -- verifying the artifact"
  RUN=$(ls -dt gpu1080/nh_runs/m1_nhar0m3_s${S}_* 2>/dev/null | head -1)
  if [ -z "$RUN" ]; then
    echo "ARTIFACT FAIL s$S: no run dir. Last log lines:"; tail -6 "logs/m1_nhtrain_nhar0m3_s${S}.log"; continue
  fi
  NCK=$(ls "$RUN"/model_epoch*.pt 2>/dev/null | wc -l)
  echo "[$(date +%T)] s$S run=$RUN checkpoints=$NCK/30"
  [ "$NCK" -lt 30 ] && { echo "ARTIFACT FAIL s$S: only $NCK/30"; continue; }

  # ---- eval + dump both frames -------------------------------------------
  for PER in validation test; do
    RES="$RUN/$PER/model_epoch030/${PER}_results.p"
    [ -f "$RES" ] || nice -n 10 $PY scripts/nh_arlstm_eval.py --run-dir "$RUN" \
        --period "$PER" --epoch 30 > "logs/m1_nheval_nhar0m3_s${S}_${PER}.log" 2>&1
    [ -f "$RES" ] || { echo "EVAL FAIL s$S $PER"; continue; }
    FR=val1; [ "$PER" = test ] && FR=test1
    OUT="data/mblstm/l51_dumps/m1_nhar0m3_s${S}_${FR}.csv.gz"
    [ -f "$OUT" ] && continue
    $PY scripts/nh_to_l51dump.py --results "$RES" --run-dir "$RUN" \
        --member nhar0m3 --seed "$S" --frame "$FR" --epoch 30 \
        --protocol modern-v3-9yr --basin-list data/m1_cohort.json --out "$OUT" \
      && gzip -t "$OUT" && echo "[$(date +%T)] s$S $FR dumped OK"
  done
done

# ---- 6. score: does the 3rd product lift the strongest member? -------------
echo "[$(date +%T)] scoring nhar0m3 against nhar0m (same seeds, one join)"
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members gridmetm9,nclimgridm9,nhar0m3 --against gridmetm9,nclimgridm9,nhar0m \
    --seeds 501,502,503 --corr --out benchmarks/m1_nhar0m3_verdict.json
echo "=== NH3 CHAIN DONE $(date +%F_%T) ==="
