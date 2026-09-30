#!/usr/bin/env bash
# MODERN-1 — the 3-product NH member (nhar0m3). Corrected replacement for
# chain_nh3.sh, whose GPU gate was insufficient.
#
# ⚠️ WHAT WENT WRONG, and it is a general lesson about "wait for an exclusive slot":
# chain_nh3's gate required `gpu=0 AND no run_m1_batch.sh AND no queue_m1_withq.sh`.
# All three were TRUE at 05:52 -- because chain_fused3m2 was in its long CPU phase
# (assembling daymet_m3, fusing 3 products, verifying columns, building cohort trees).
# It launched Gate A three minutes later. So NH claimed the card immediately before
# the registered STOP gate wanted it, and would have held it ~18 h for 3 sequential
# seeds. No oversubscription -- the batch controller's own drain gate made it wait --
# but the critical path would have been delayed by most of a day.
#
# ⭐ A quiet box is not an idle box: absence of a RUNNING consumer says nothing about
# an IMMINENT one. So the gate must also require that the PRODUCER of GPU work is
# finished, not merely that no worker is currently up:
#     gpu=0 AND no run_m1_batch AND no queue_m1_withq AND no queue_m1_noq
#           AND **no chain_fused3m2** (the thing that launches the batches)
# ⭐ Still reachable in both branches: chain_fused3m2 exits on success (after the
# daymetm batch) and on halt (a failing Gate A), so this cannot wait forever.
#
# CPU prep is already done by the killed run and is verified, not repeated.
set -u
cd ~/riverwatch2 || exit 1
PY=gpu1080/.venv/bin/python
ND=gpu1080/nh_data_multim3
LOG=logs/m1_nh3b_chain.log
exec >>"$LOG" 2>&1
echo "=== NH3B CHAIN START $(date +%F_%T) ==="

# ---- 0. verify the prep rather than rebuild it -----------------------------
NB=$(wc -l < "$ND/basins.txt" 2>/dev/null || echo 0)
NEED=$($PY -c 'import json;print(len(json.load(open("data/m1_cohort.json"))["cohort"]))')
[ "$NB" != "$NEED" ] && { echo "PREP MISSING: $ND has $NB basins, need $NEED"; exit 3; }
for S in 501 502 503; do
  CFG="gpu1080/cfg_nhar0m3_s${S}.yml"
  [ -f "$CFG" ] || { echo "PREP MISSING: $CFG"; exit 4; }
  $PY - "$CFG" <<'PYC'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
d = c["dynamic_inputs"]
assert len(d) == 15, f"{sys.argv[1]}: {len(d)} dynamic inputs, expected 15: {d}"
assert sorted({v.rsplit('_', 1)[1] for v in d}) == ["daymet", "gridmet", "nclimgrid"]
assert c["train_start_date"] == "01/10/2005", c["train_start_date"]
PYC
  [ $? -eq 0 ] || { echo "CFG CHECK FAILED $CFG"; exit 4; }
done
echo "[$(date +%T)] prep verified: $NB basins, 3 configs, 15 inputs each"

# ---- 1. wait for a genuinely exclusive slot --------------------------------
QUIET=0
for i in $(seq 1 2160); do            # up to 72 h
  N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  B=$(ps -eo cmd --no-headers | grep -c "run_m1_batc[h]\.sh")
  Q=$(ps -eo cmd --no-headers | grep -c "queue_m1_with[q]\.sh")
  QN=$(ps -eo cmd --no-headers | grep -c "queue_m1_no[q]\.sh")
  C2=$(ps -eo cmd --no-headers | grep -c "chain_fused3m[2]\.sh")
  if [ "$N" -eq 0 ] && [ "$B" -eq 0 ] && [ "$Q" -eq 0 ] && [ "$QN" -eq 0 ] && [ "$C2" -eq 0 ]; then
    QUIET=$((QUIET + 1)); else QUIET=0
  fi
  [ "$QUIET" -ge 3 ] && { echo "[$(date +%T)] exclusive slot on 3 polls (gpu=0 batch=0 queue=0 chain2=0)"; break; }
  [ $((i % 30)) -eq 0 ] && echo "[$(date +%T)] waiting: gpu=$N batch=$B queue=$Q noq=$QN chain2=$C2 quiet=$QUIET/3"
  sleep 120
done
N=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
C2=$(ps -eo cmd --no-headers | grep -c "chain_fused3m[2]\.sh")
{ [ "$N" -ne 0 ] || [ "$C2" -ne 0 ]; } && { echo "no exclusive slot after 72 h (gpu=$N chain2=$C2)"; exit 5; }

# ---- 2. train, up to MAX concurrent, ARTIFACT-verified ---------------------
# ⚠️ MEASURED 2026-09-30: nhar0m3 runs ~24 min/epoch => ~12 h/seed. Training the three
# seeds SEQUENTIALLY (the original loop) is 36 h of wall-clock for work the card can
# overlap: one NH run holds ~1986 MiB of 8192, and the earlier 2-product member was
# proven at 2 concurrent on this same card. So train 2 at a time -- ~24 h -> ~12 h for
# the two remaining seeds.
# ⛔ NOT 3 concurrent: 2 is what has actually been observed to work here, and this box
# has a recorded CUDA OOM from putting one more job on the card than was verified.
NHMAX=${NH_MAX:-2}
PIDS=""
for S in 501 502 503; do
  RUN=$(ls -dt gpu1080/nh_runs/m1_nhar0m3_s${S}_* 2>/dev/null | head -1)
  if [ -n "$RUN" ] && [ "$(ls "$RUN"/model_epoch*.pt 2>/dev/null | wc -l)" -ge 30 ]; then
    echo "[$(date +%T)] s$S already trained"; continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$NHMAX" ]; do wait -n; done
  A=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  echo "[$(date +%T)] training nhar0m3 s$S (${A}MiB RAM, max $NHMAX concurrent)"
  ( nice -n 5 $PY scripts/nh_arlstm_train.py --config-file "gpu1080/cfg_nhar0m3_s${S}.yml" \
      --no-graph > "logs/m1_nhtrain_nhar0m3_s${S}.log" 2>&1
    echo "[$(date +%F_%T)] s$S train exited rc=$?" >> "$LOG" ) &
  sleep 120   # let it load its dataset before the next starts (a job is invisible to
              # nvidia-smi for minutes while loading -- see the oversubscription note)
done
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n; done
echo "[$(date +%T)] all training finished -- now evaluating and dumping"

# ---- 2b. eval + dump each seed whose checkpoints are complete --------------
for S in 501 502 503; do
  RUN=$(ls -dt gpu1080/nh_runs/m1_nhar0m3_s${S}_* 2>/dev/null | head -1)
  [ -z "$RUN" ] && { echo "ARTIFACT FAIL s$S: no run dir"; tail -6 "logs/m1_nhtrain_nhar0m3_s${S}.log" 2>/dev/null; continue; }
  NCK=$(ls "$RUN"/model_epoch*.pt 2>/dev/null | wc -l)
  echo "[$(date +%T)] s$S run=$RUN ckpts=$NCK/30"
  [ "$NCK" -lt 30 ] && { echo "ARTIFACT FAIL s$S: $NCK/30"; continue; }
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
      && gzip -t "$OUT" && echo "[$(date +%T)] s$S $FR dumped"
  done
done

# ---- 3. two questions, both against the CURRENT roster ---------------------
# ⚠️ The original comparison here was written when the roster was the
# gridmetm9+nclimgridm9+nhar0m trio. That roster is obsolete: the fix factorial landed
# (fused3md3 solo 0.893339) and nclimgridm9 was DROPPED by the standing removal rule
# (LOO -0.000874, CI excluding zero). Comparing against a stale baseline would answer a
# question nobody is asking.
#
# Q1. THE SWAP, at matched everything-else: does a 3-product NH beat the 2-product one
#     inside the 5-member roster? This is the first PRODUCT-MATCHED test of the
#     model-family axis -- nhar0m's earlier failure to add was measured at TWO products
#     against three-product members, which confounded family with forcing count.
echo "[$(date +%T)] Q1: swap nhar0m -> nhar0m3 inside the 5-member roster"
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused3m,fused3md3,fused3md3q,nhar0m3,gridmetm9 \
    --against fused3m,fused3md3,fused3md3q,nhar0m,gridmetm9 \
    --seeds 501,502,503 --corr --out benchmarks/m1_nhar0m3_swap.json

# Q2. THE ADD: is the 3-product NH worth keeping ALONGSIDE the 2-product one, or does it
#     replace it? Both are reported; the registered rule admits on the larger paired
#     delta whose CI excludes zero.
echo "[$(date +%T)] Q2: add nhar0m3 to the 5-member roster (keep nhar0m too)"
$PY analysis/m1_composition.py --frame val1 --protocol modern-v3-9yr \
    --members fused3m,fused3md3,fused3md3q,nhar0m,nhar0m3,gridmetm9 \
    --against fused3m,fused3md3,fused3md3q,nhar0m,gridmetm9 \
    --seeds 501,502,503 --corr --out benchmarks/m1_nhar0m3_add.json
echo "=== NH3B CHAIN DONE $(date +%F_%T) ==="
