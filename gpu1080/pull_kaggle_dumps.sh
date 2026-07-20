#!/bin/bash
# Pull whatever NH member dumps Kaggle already produced into ./dumps/ so run9.sh
# trains ONLY the missing (forcing,seed) combos overnight on the 1080.
# Needs ~/.kaggle/kaggle.json. Harmless to re-run.
set -u
cd "$(dirname "$0")"
export KAGGLE_CONFIG_DIR=~/.kaggle
USER=andrewnakas
mkdir -p dumps
# one-seed kernels (rw2-nhs-<f>-<s>) + the original multi-seed kernels (rw2-nh-train-<f>)
for F in daymet nldas maurer; do
  kaggle kernels output $USER/rw2-nh-train-$F -p dumps/ 2>/dev/null || true
  for S in 111 222 333; do
    kaggle kernels output $USER/rw2-nhs-$F-$S -p dumps/ 2>/dev/null || true
  done
done
echo "NH dumps present in ./dumps/:"
ls dumps/camels531_*_nhlstm_s*.csv.gz 2>/dev/null | sed 's#.*/##' || echo "  (none yet)"
echo ""
echo "Now run:  bash run9.sh   # trains only the missing seeds"
