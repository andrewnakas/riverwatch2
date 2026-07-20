#!/bin/bash
# Orchestrate 9 one-seed Kaggle kernels (3 forcings × 3 seeds), each a distinct
# slug so all fit the 12hr wall independently. Respects the 2-concurrent-GPU cap:
# launches, waits for a slot, pulls dumps on completion. Idempotent — skips a
# (forcing,seed) whose dump already landed.
set -u
cd "$(dirname "$0")/.."
export KAGGLE_CONFIG_DIR=~/.kaggle
DRIVER=notebooks/kaggle_driver.py
DUMPS=data/mblstm/gpu_dumps_s14
EPOCHS=${EPOCHS:-30}
USER=andrewnakas
mkdir -p "$DUMPS"

JOBS=""
for F in daymet nldas maurer; do for S in 111 222 333; do JOBS="$JOBS $F:$S"; done; done

kstatus() { kaggle kernels status $USER/$1 2>/dev/null | grep -oE "COMPLETE|RUNNING|ERROR|CANCEL" | head -1; }
running_count() {
  local n=0
  for j in $JOBS; do local f=${j%:*} s=${j#*:}
    local st=$(kstatus "rw2-nhs-$f-$s"); [ "$st" = "RUNNING" ] && n=$((n+1)); done
  echo $n
}

for round in $(seq 1 40); do
  # first, harvest dumps from the ORIGINAL multi-seed kernels (nldas/daymet seed 111)
  for F in nldas daymet maurer; do
    st=$(kstatus "rw2-nh-train-$F")
    if [ "$st" = "COMPLETE" ] || [ "$st" = "ERROR" ]; then
      kaggle kernels output $USER/rw2-nh-train-$F -p "$DUMPS/" >/dev/null 2>&1
    fi
  done
  done_n=0
  for j in $JOBS; do
    F=${j%:*}; S=${j#*:}
    slug="rw2-nhs-$F-$S"
    dump="$DUMPS/camels531_${F}_nhlstm_s${S}.csv.gz"
    [ -f "$dump" ] && { done_n=$((done_n+1)); continue; }
    st=$(kstatus "$slug")
    if [ "$st" = "COMPLETE" ] || [ "$st" = "ERROR" ]; then
      kaggle kernels output $USER/$slug -p "$DUMPS/" >/dev/null 2>&1
      [ -f "$dump" ] && { echo "GOT $F s$S"; done_n=$((done_n+1)); continue; }
    fi
    if [ -z "$st" ]; then   # not launched yet
      if [ "$(running_count)" -lt 2 ]; then
        echo "launch $F s$S (slug $slug)"
        python3 "$DRIVER" nh-train --forcing "$F" --seeds "$S" --epochs "$EPOCHS" \
          --slug "$USER/$slug" 2>&1 | grep -iE "pushed|error" | tail -1
        sleep 15
      fi
    fi
  done
  echo "round $round: $done_n/9 dumps done"
  [ "$done_n" -ge 9 ] && { echo "ALL 9 SEEDS DONE"; break; }
  sleep 300
done
