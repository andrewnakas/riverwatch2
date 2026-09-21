#!/usr/bin/env bash
# LEDGER 53 ship bar: re-read the record at 5 seeds of `nhar` once s504/s505 land,
# val1 first then test1, with the paired statistic. Composition is NOT re-chosen
# here — A (equal weight) was decided on val1 at the screen; this confirms it at
# the ship depth.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python; OUTD=benchmarks/l53; LOG=logs/l54_rescore5.log
mkdir -p "$OUTD"
exec 9>/tmp/rw2_l54_rescore5.lock; flock -n 9 || exit 0
need=""
for s in 504 505; do for f in val1 test1; do need="$need data/mblstm/l51_dumps/camels531_l51_nhar_s${s}_${f}.csv.gz"; done; done
echo "=== waiting for the 5-seed dumps $(date +%F_%T) ===" >> "$LOG"
while :; do ok=1; for f in $need; do [ -f "$f" ] || ok=0; done; [ "$ok" = 1 ] && break; sleep 600; done
echo "=== all 5 seeds present $(date +%F_%T); re-scoring ===" >> "$LOG"
for FR in val1 test1; do
  $PY analysis/l53_composition.py --frame "$FR" --out "$OUTD/COMPOSITION_${FR}_5seed.json" \
    > "$OUTD/COMPOSITION_${FR}_5seed.txt" 2>&1
  echo "[$(date +%T)] scored $FR" >> "$LOG"
done
$PY - "$OUTD" >> "$LOG" 2>&1 <<'PYV'
import json, sys
o = sys.argv[1]
v3 = json.load(open(f"{o}/COMPOSITION_test1_3seed.json"))
v5 = json.load(open(f"{o}/COMPOSITION_test1_5seed.json"))
w5 = json.load(open(f"{o}/COMPOSITION_val1_5seed.json"))
out = {"test1_3seed_A": v3["median"]["A"], "test1_5seed_A": v5["median"]["A"],
       "seed_depth_gain": v5["median"]["A"] - v3["median"]["A"],
       "test1_5seed_base5": v5["median"]["base5"], "solo_nhar_5seed_test1": v5["solo"]["nhar"],
       "val1_5seed_A_vs_base5": w5["paired"]["A_vs_base5"],
       "test1_5seed_A_vs_base5": v5["paired"]["A_vs_base5"],
       "ship_bar_pass": bool(w5["paired"]["A_vs_base5"]["significant"]
                             and w5["paired"]["A_vs_base5"]["breadth"] >= 0.5),
       "reaches_0.90": v5["median"]["A"] >= 0.90}
json.dump(out, open(f"{o}/VERDICT_5seed.json", "w"), indent=1)
print(json.dumps(out, indent=1))
PYV
echo "=== done $(date +%F_%T) ===" >> "$LOG"
