#!/usr/bin/env bash
# LEDGER 53 — unattended: wait for the nhar 3-seed dumps, score the
# pre-registered compositions (PREREG_v2.md §LEDGER 53) on val1 THEN test1,
# check the pre-registered falsifiers, and launch the pre-registered
# continuation lanes if — and only if — they pass. Never edits a running script.
set -u
cd ~/riverwatch2
PY=gpu1080/.venv/bin/python
DD=data/mblstm/l51_dumps
LOG=logs/l53_autoscore.log
OUTD=benchmarks/l53
mkdir -p "$OUTD" logs
exec 9>/tmp/rw2_l53_autoscore.lock; flock -n 9 || { echo "autoscore already running"; exit 0; }

need=""
for s in 501 502 503; do for f in val1 test1; do need="$need $DD/camels531_l51_nhar_s${s}_${f}.csv.gz"; done; done
echo "=== autoscore start $(date +%F_%T) ===" >> "$LOG"
while :; do
  ok=1; for f in $need; do [ -f "$f" ] || ok=0; done
  [ "$ok" = 1 ] && break
  sleep 600
done
echo "=== all 6 nhar dumps present $(date +%F_%T) ===" >> "$LOG"

FIVE=fused3h1,daymeth1,nldash1,maurerh1,aorch1
FOUR=fused3h1,daymeth1,nldash1,maurerh1
echo '{"fused3h1":1,"daymeth1":1,"nldash1":1,"maurerh1":1,"aorch1":1,"nhar":2}' > "$OUTD/weights_C.json"
score () {  # <tag> <frame> <args...>
  local TAG="$1" FR="$2"; shift 2
  [ -f "$OUTD/${TAG}_${FR}.json" ] && return 0
  echo "[$(date +%T)] score $TAG $FR" >> "$LOG"
  # shellcheck disable=SC2068
  $PY analysis/l51_withq_score.py --frame "$FR" $@ --out "$OUTD/${TAG}_${FR}.json" > "$OUTD/${TAG}_${FR}.txt" 2>&1 \
    || echo "  FAILED $TAG $FR (see $OUTD/${TAG}_${FR}.txt)" >> "$LOG"
}
# val1 first (selection), test1 after (read-only)
for FR in val1 test1; do
  score base5 "$FR" --members "$FIVE" --loo --bootstrap
  score A     "$FR" --members "$FIVE,nhar" --loo --bootstrap --dup-control
  score B     "$FR" --members "$FOUR,nhar" --loo --bootstrap --dup-control
  score C     "$FR" --members "$FIVE,nhar" --weights "$OUTD/weights_C.json" --loo --bootstrap
  for s in 501 502 503; do score "solo_nhar_s$s" "$FR" --members nhar --seeds "$s"; done
  score solo_nhar3 "$FR" --members nhar
done

$PY - "$OUTD" <<'PYV' >> "$LOG" 2>&1
import json, sys, statistics as st
o = sys.argv[1]
J = lambda n: json.load(open(f"{o}/{n}.json"))
solo = [J(f"solo_nhar_s{s}_test1")["member_median_nse"]["nhar"] for s in (501, 502, 503)]
solo_val = [J(f"solo_nhar_s{s}_val1")["member_median_nse"]["nhar"] for s in (501, 502, 503)]
A = J("A_val1"); At = J("A_test1"); B = J("B_val1"); C = J("C_val1"); b5 = J("base5_val1"); b5t = J("base5_test1")
loo = A["loo"]["nhar"]
fi = st.mean(solo) >= 0.860                      # falsifier (i): the reproduction worked
fii = loo["paired_delta"] > 0 and loo["ci"][0] > 0 and loo["breadth"] >= 0.5   # val1 screen passes
v = {"solo_1seed_test1": solo, "solo_1seed_val1": solo_val,
     "solo_3seed_test1": J("solo_nhar3_test1")["member_median_nse"]["nhar"],
     "solo_3seed_val1": J("solo_nhar3_val1")["member_median_nse"]["nhar"],
     "val1": {"base5": b5["ensemble_median_nse"], "A": A["ensemble_median_nse"], "B": B["ensemble_median_nse"],
              "C": C["ensemble_median_nse"], "nhar_loo_A": loo, "nhar_loo_B": B["loo"]["nhar"], "nhar_loo_C": C["loo"]["nhar"],
              "dup_control_A": A.get("dup_control")},
     "test1": {"base5": b5t["ensemble_median_nse"], "A": At["ensemble_median_nse"],
               "B": J("B_test1")["ensemble_median_nse"], "C": J("C_test1")["ensemble_median_nse"],
               "nhar_loo_A": At["loo"]["nhar"]},
     "falsifier_i_reproduction_ok": fi, "val1_screen_pass": fii,
     "continue_seeds_504_505": fi and fii, "continue_nhar256": fi}
json.dump(v, open(f"{o}/VERDICT_3seed.json", "w"), indent=1)
print(json.dumps(v, indent=1))
PYV

CONT_SEEDS=$($PY -c "import json;print(int(json.load(open('$OUTD/VERDICT_3seed.json'))['continue_seeds_504_505']))")
CONT_256=$($PY -c "import json;print(int(json.load(open('$OUTD/VERDICT_3seed.json'))['continue_nhar256']))")
echo "[$(date +%T)] continue_seeds=$CONT_SEEDS continue_nhar256=$CONT_256" >> "$LOG"
# 2026-09-09 addendum (PREREG_v2.md §LEDGER 53): the paper's 0.879 is the NO-holdout
# model, so nhar0 is the exact reproduction and takes the variant slot before nhar256.
if [ "$CONT_SEEDS" = 1 ] && [ "$CONT_256" = 1 ]; then
  nohup bash gpu1080/run_l53_nhar_lane.sh D nhar:504 nhar0:502 > /dev/null 2>&1 &
  nohup bash gpu1080/run_l53_nhar_lane.sh E nhar:505 nhar0:503 > /dev/null 2>&1 &
  nohup bash gpu1080/run_l53_nhar_lane.sh F nhar0:501 > /dev/null 2>&1 &
  echo "[$(date +%T)] launched lanes D/E/F: nhar 504/505 + nhar0 501-503" >> "$LOG"
elif [ "$CONT_256" = 1 ]; then
  nohup bash gpu1080/run_l53_nhar_lane.sh F nhar0:501 nhar0:502 nhar0:503 > /dev/null 2>&1 &
  echo "[$(date +%T)] val1 screen FAILED for nhar; launched lane F: nhar0 501-503 only" >> "$LOG"
else
  echo "[$(date +%T)] falsifier (i) TRIGGERED: reproduction failed — nothing launched, diagnose first" >> "$LOG"
fi
echo "=== autoscore done $(date +%F_%T) ===" >> "$LOG"
