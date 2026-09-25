#!/usr/bin/env bash
# LEDGER 42 — divergence watch for the Track-A arm. QUIET BY DESIGN.
#
# Emits a line ONLY when a seed's training loss has ROSE ABOVE ITS RUNNING
# MINIMUM AND NOT RETURNED -- the criterion this campaign settled on after a
# per-step 1.5x threshold MISSED a two-epoch slide (worst single step 1.219).
# A per-epoch ratio-to-minimum catches the slide; a step-to-step ratio does not.
#
# Also emits once per finished seed, so silence means "running, nothing wrong"
# rather than "the watcher died".
#
# ⛔ Divergence is REPORTED, never acted on: the branch table's reporting rule
# says a destabilised seed is NOT dropped, NOT re-run, NOT re-rolled. The dump
# step already selects the BEST epoch, so a late spike cannot silently ship.
set -u
cd ~/riverwatch2
RATIO="${RATIO:-1.30}"      # current / running-min that counts as a slide
MIN_EPOCH="${MIN_EPOCH:-5}" # ignore the noisy first epochs
declare -A FLAGGED DONE
while true; do
  for f in logs/l42_evsub_*.log; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .log)
    # ⚠️ STRICT pattern. The loose `avg_loss: [0-9.]+` also matches the
    # end-of-run "average VALIDATION loss" line, which made SEED COMPLETE report
    # the validation loss as "last" (0.01410 vs the true final train 0.01246) and
    # would let a late val line trip a spurious divergence flag. Same class as
    # the loose `grep -c ' True'` that matched a torch warning. The queue script
    # already used the strict form, which is why its epoch pick stayed correct.
    losses=$(grep -oE "Epoch [0-9]+ average loss: avg_loss: [0-9.]+" "$f" | awk '{print $6}')
    [ -z "$losses" ] && continue
    ne=$(echo "$losses" | wc -l)
    if [ "$ne" -ge "$MIN_EPOCH" ] && [ -z "${FLAGGED[$n]:-}" ]; then
      read -r cur mn <<<"$(echo "$losses" | awk '{c=$1; if(NR==1||$1<m) m=$1} END{print c, m}')"
      bad=$(awk -v c="$cur" -v m="$mn" -v r="$RATIO" 'BEGIN{print (m>0 && c/m>r)?1:0}')
      if [ "$bad" = "1" ]; then
        echo "DIVERGENCE $n: epoch $ne loss $cur is $(awk -v c=$cur -v m=$mn 'BEGIN{printf "%.2f", c/m}')x its running minimum $mn — REPORT ONLY, do not re-run (branch-table rule)"
        FLAGGED[$n]=1
      fi
    fi
    if [ "$ne" -ge 30 ] && [ -z "${DONE[$n]:-}" ]; then
      best=$(echo "$losses" | sort -g | head -1)
      last=$(echo "$losses" | tail -1)
      bestep=$(echo "$losses" | awk '{if(NR==1||$1<m){m=$1;e=NR}} END{print e}')
      # A latched DIVERGENCE flag would otherwise be the only thing a reader
      # sees, implying a ruined seed. State recovery explicitly: s111 peaked at
      # 3.11x its running minimum and still ended BELOW it.
      note=""
      [ -n "${FLAGGED[$n]:-}" ] && note=" — RECOVERED from the earlier excursion (best epoch $bestep)"
      [ -n "${FLAGGED[$n]:-}" ] && awk -v l="$last" -v b="$best" 'BEGIN{exit !(l/b>1.10)}' \
        && note=" — did NOT recover; ended $(awk -v l=$last -v b=$best 'BEGIN{printf "%.0f%%", 100*(l/b-1)}') above its best (epoch $bestep)"
      echo "SEED COMPLETE $n: 30 epochs, best $best (ep$bestep), last $last, last/best $(awk -v l=$last -v b=$best 'BEGIN{printf "%.3f", l/b}')$note"
      DONE[$n]=1
    fi
  done
  sleep 600
done
