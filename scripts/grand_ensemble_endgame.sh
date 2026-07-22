#!/bin/bash
# Grand-ensemble endgame for the 0.83 no-q attempt. Runs the moment all 9 NH
# LSTM members are present. Three rungs, each vs the paper's Table D1:
#   rung 1: 9 NH LSTM only            -> target 0.808 (LSTM^123 ensemble)
#   rung 2: 9 NH LSTM + 3 δHBV        -> target 0.818 (+δHBV)
#   rung 3: same, --fit-weights       -> the beat lever (optimal blend)
# All on the CAMELS-531 no-q temporal split. Reports pooled + exact-Kratzert day-1.
set -u
cd "$(dirname "$0")/.."
D=data/mblstm/gpu_dumps_s14
OUT=data/mblstm/grand_ensemble
mkdir -p "$OUT"

NH=$(ls $D/camels531_daymet_nhlstm_s{111,222,333}.csv.gz \
        $D/camels531_nldas_nhlstm_s{111,222,333}.csv.gz \
        $D/camels531_maurer_nhlstm_s{111,222,333}.csv.gz 2>/dev/null)
NNH=$(echo "$NH" | grep -c csv)
echo "NH LSTM members found: $NNH/9"
[ "$NNH" -lt 9 ] && { echo "NOT all 9 present — aborting"; exit 1; }

# δHBV members = the shipped combined50 per-forcing ensembles
DHBV=$(ls $D/camels531_daymet_combined50_ens_s14.csv.gz \
          $D/camels531_nldas_combined50_ens_s14.csv.gz \
          $D/camels531_maurer_combined50_ens_s14.csv.gz 2>/dev/null)
echo "δHBV members found: $(echo "$DHBV" | grep -c csv)/3"

echo ""
echo "======================================================================"
echo "RUNG 1: 9 NH LSTM only  (target 0.808 = Table D1 LSTM^123)"
echo "======================================================================"
python3 scripts/combine_dumps.py --dumps $NH \
  --label "grand_9nhlstm" --out "$OUT/rung1_9nhlstm.json" 2>&1 | \
  grep -iE "pooled|day.?1|median|scorable|NSE|members" | head -20

echo ""
echo "======================================================================"
echo "RUNG 2: 9 NH LSTM + 3 δHBV  (target 0.818 = +δHBV)"
echo "======================================================================"
python3 scripts/combine_dumps.py --dumps $NH $DHBV \
  --label "grand_9nh_3dhbv" --out "$OUT/rung2_9nh_3dhbv.json" 2>&1 | \
  grep -iE "pooled|day.?1|median|scorable|NSE|members" | head -20

echo ""
echo "======================================================================"
echo "RUNG 3: 9 NH LSTM + 3 δHBV, --fit-weights  (the beat lever)"
echo "======================================================================"
python3 scripts/combine_dumps.py --dumps $NH $DHBV --fit-weights \
  --label "grand_fitweights" --out "$OUT/rung3_fitweights.json" 2>&1 | \
  grep -iE "pooled|day.?1|median|scorable|NSE|members|weight" | head -30

echo ""
echo "=== ladder summary (pooled medians from the JSONs) ==="
python3 - <<'PY'
import json, glob, os
OUT="data/mblstm/grand_ensemble"
for tag,f in [("rung1 9-NH-LSTM","rung1_9nhlstm.json"),
              ("rung2 +δHBV","rung2_9nh_3dhbv.json"),
              ("rung3 +fit-weights","rung3_fitweights.json")]:
    p=os.path.join(OUT,f)
    if not os.path.exists(p): print(f"  {tag:22s} (missing)"); continue
    d=json.load(open(p))
    # find pooled + day1 medians in the mblstm_backtest JSON shape
    def dig(o,*keys):
        for k in keys:
            if isinstance(o,dict) and k in o: o=o[k]
            else: return None
        return o
    pooled=dig(d,"pooled","median") or dig(d,"median") or dig(d,"all","median")
    day1=dig(d,"day1","median") or dig(d,"h1","median")
    print(f"  {tag:22s} pooled={pooled}  day1={day1}")
print("  ---")
print("  paper Table D1: LSTM^123 0.808 -> +δHBV 0.818 -> +seeds 0.830")
PY
