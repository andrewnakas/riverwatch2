#!/usr/bin/env bash
# MODERN-1 — train one with-q member on the modern corpus. usage: <member> <seed>
#
# Protocol P-MODERN, pre-registered in PREREG_v2.md MODERN-1 before any data existed:
#   train 1980-10-01..2014-09-30 | val 2014-10-01..2018-09-30 | test 2018-10-01..2025-09-30
# Cohort: data/m1_cohort.json (490 basins), frozen from discharge coverage BEFORE
# any score was seen. The 531 are NOT the modern cohort -- 24 of their gauges are
# entirely dead in the test window.
set -u
cd ~/riverwatch2 || exit 1
M="${1:?member}"; S="${2:?seed}"
PY=gpu1080/.venv/bin/python
DD=data/mblstm/l51_dumps
V3=data/modern_corpus/v3
LOG="logs/m1_${M}_s${S}.log"
CKPT="data/mblstm/gpu_ckpts/m1_${M}_s${S}.pt"
TRS=1980-10-01; TRE=2014-09-30
VS=2014-10-01;  VE=2018-09-30
TS=2018-10-01;  TE=2025-09-30
SMOKE=0
case "$M" in *smoke*) SMOKE=1 ;; esac
LIMIT=""; [ "$SMOKE" = "1" ] && LIMIT="--limit-stations 8"

EXTRA=""
case "$M" in
  daymetm|daymetmsmoke)      CORP="$V3/camels_corpus_daymet_m3_cohort";    ENC="camels1f" ;;
  nldasm|nldasmsmoke)        CORP="$V3/camels_corpus_nldas_m3_cohort";     ENC="camels1f" ;;
  gridmetm|gridmetmsmoke)    CORP="$V3/camels_corpus_gridmet_m3_cohort";   ENC="camels1f" ;;
  nclimgridm|nclimgridmsmoke)CORP="$V3/camels_corpus_nclimgrid_m3_cohort"; ENC="camels1f" ;;
  # ---- GATE B: identical recipe, 9-year training window instead of 34 ----------
  # P-MODERN offers 34 years; the record used 9 only for comparability with
  # Nearing 2022. Nothing here has ever tested whether the extra 25 buy skill, and
  # the answer is also a BUDGET decision: 34 years is ~6.0M train windows against
  # ~1.6M, i.e. ~3.5x the GPU cost per seed, across ~50 runs.
  gridmetm9)                 CORP="$V3/camels_corpus_gridmet_m3_cohort";   ENC="camels1f"; TRS=2005-10-01 ;;
  nclimgridm9)               CORP="$V3/camels_corpus_nclimgrid_m3_cohort"; ENC="camels1f"; TRS=2005-10-01 ;;
  # 24-var encoder, single-product (daymet) decoder -- the OPERATIONALLY servable shape
  fused4m|fused4msmoke)      CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm4v2" ;;
  # 24-var encoder AND decoder -- the LEDGER 56 fix
  fused4md4|fused4md4smoke)  CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm4v2d4" ;;
  # ⭐ + --dec-q: the assimilated discharge reaches the DECODER too. Measured on
  # the CAMELS analogue 2026-09-25: fused3h1d3q solo +0.006384 CI
  # [+0.004554,+0.008198] -- the best single member class this campaign has
  # produced, ahead of the all-product decoder alone (+0.005604).
  fused4md4q|fused4md4qsmoke)CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm4v2d4"; EXTRA="--dec-q" ;;
  # single-product ENCODER + all-product DECODER. Measured ADMITTING on the CAMELS
  # analogue (daymeth1d3 +0.000261, nldash1d3 +0.000209, both CI-positive), unlike
  # the mirror shape (all-product encoder / single-product decoder), which read NULL.
  daymetmd4)                 CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm1f_daymet_d4" ;;
  nldasmd4)                  CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm1f_nldas_d4" ;;
  gridmetmd4)                CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm1f_gridmet_d4" ;;
  nclimgridmd4)              CORP="$V3/camels_corpus_fused4m_m3_cohort";   ENC="camelsm1f_nclimgrid_d4" ;;
  *) echo "unknown member $M"; exit 2 ;;
esac

# Smoke members run against the SYNTHETIC proxy corpus (product suffixes renamed,
# dates shifted) so the whole launcher can be exercised before the real corpus
# exists. ⛔ Never a scientific result -- see scripts_modern/make_m1_smoke_corpus.py.
# ⚠️ And they run on the CPU by default: the 8 GB card is already at the L56
# batch's MAX=3, and adding an uncounted 4th job is exactly how this box got a
# CUDA OOM that was first misdiagnosed as a hardware fault.
DEV=""
if [ "$SMOKE" = "1" ]; then
  CORP="${M1_SMOKE_CORPUS:-data/modern_corpus/SMOKE_fused4m_cohort}"
  DEV="--device ${M1_SMOKE_DEVICE:-cpu}"
fi

exec >>"$LOG" 2>&1
echo "=== M1 $M s$S $(date +%F_%T) ==="

# ---- GUARD 0: the corpus must be exactly the frozen cohort -----------------
NEED=$($PY -c 'import json;print(len(json.load(open("data/m1_cohort.json"))["cohort"]))')
HAVE=$(ls "$CORP"/*.csv.gz 2>/dev/null | wc -l)
if [ "$SMOKE" = "0" ] && [ "$HAVE" != "$NEED" ]; then
  echo "ABORT GUARD 0: corpus $CORP has $HAVE basins, cohort is $NEED"; exit 1
fi
echo "GUARD 0 OK: $HAVE/$NEED cohort basins in $CORP (enc=$ENC smoke=$SMOKE)"

# ---- train -----------------------------------------------------------------
if [ ! -f "$CKPT" ]; then
  echo "=== TRAIN $M s$S $(date +%F_%T) ==="
  # shellcheck disable=SC2086
  nice -n 5 $PY scripts/train_mblstm.py \
    --epochs 30 --seed "$S" --out "$CKPT" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --train-start "$TRS" --train-end "$TRE" \
    --val-start "$VS" --val-end "$VE" \
    --h1-weight 0.5 $EXTRA $LIMIT $DEV
  echo "=== trained rc=$? $(date +%T) ==="
fi

# ---- GUARD 1: read the protocol back out of the CHECKPOINT, never argv -----
$PY - "$CKPT" "$LOG" "$ENC" "$TRS" "$TRE" "$VS" "$VE" <<'PYG'
import re, sys, torch
ck, log, enc, trs, tre, vs, ve = sys.argv[1:8]
sd = torch.load(ck, map_location="cpu")
cfg = sd["cfg"]
assert cfg["train_start"] == trs, f"train_start {cfg['train_start']} != {trs}"
assert cfg["train_end"] == tre, f"train_end {cfg['train_end']} != {tre}"
assert cfg["val_range"] == [vs, ve], f"val {cfg['val_range']} != {[vs, ve]}"
assert cfg["no_q_input"] is False, "this is the WITH-Q arm"
assert abs(cfg.get("h1_weight", -1) - 0.5) < 1e-9, cfg.get("h1_weight")
ndec = len(cfg["dec_vars"])
# the checkpoint stores weights under "state_dict", not "model" -- caught by
# the proxy-corpus smoke before the real corpus existed
w = sd["state_dict"]["decoder.weight_ih_l0"].shape[1]
exp = ndec + 2 + 1 + len(cfg["static_feats"])
assert w == exp, f"decoder width {w} != {exp} (dec={ndec})"
prods = {v.rsplit("_", 1)[1] for v in cfg["enc_vars"] if "_" in v}
txt = open(log, encoding="utf-8", errors="replace").read()
assert re.search(r"^epoch 30/30 ", txt, re.M), "no 'epoch 30/30' line - partial checkpoint"
m = re.search(r"windows: train=(\d+).*?val=(\d+)", txt, re.S)
print(f"GUARD 1 OK: stations={cfg['n_stations']} train={cfg['train_start']}..{cfg['train_end']} "
      f"enc={len(cfg['enc_vars'])} dec={ndec} products={sorted(prods)} dec_in={w}")
PYG
[ $? -eq 0 ] || { echo "ABORT: GUARD 1 failed"; exit 1; }

# ---- dumps -----------------------------------------------------------------
dump () {  # <tag> <start> <end> <stride> <allleads 0|1>
  local TAG="$1" A="$2" B="$3" ST="$4" AL="$5"
  local OUT="$DD/m1_${M}_s${S}_${TAG}.csv.gz"
  [ -f "$OUT" ] && { echo "[$(date +%T)] $TAG dump exists"; return 0; }
  local FLAG=""; [ "$AL" = "1" ] && FLAG="--dump-all-leads"
  echo "=== DUMP $TAG ($A..$B stride $ST allleads=$AL) $(date +%F_%T) ==="
  # shellcheck disable=SC2086
  nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
    --out "$CKPT.dumpsink_$TAG" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start "$A" --val-end "$B" --val-stride "$ST" \
    --dump-day1 "$OUT" $FLAG $EXTRA $LIMIT $DEV >> "logs/m1_dump_${M}_s${S}_${TAG}.log" 2>&1
  rm -f "$CKPT.dumpsink_$TAG"
  gzip -t "$OUT" || { echo "ABORT: $TAG dump failed gzip -t"; return 1; }
  $PY - "$OUT" "$CKPT" "$TAG" "$M" "$S" "$SMOKE" "$A" "$B" "$AL" "$TRS" "$TRE" <<'PYS'
import datetime as dt, gzip, hashlib, json, os, sys
out, ck, tag, member, seed, smoke, a, b, al, trs, tre = sys.argv[1:12]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
assert n > 1000, f"only {n} rows - the dump path emitted nothing"
# ⚠️ The row bar is DERIVED, not a magic constant. LEDGER 56 hard-coded
# `n > 1_500_000`, which was calibrated on the fused3 frame and then wrongly
# failed a perfectly good fused4 dump of 1,458,982 rows, losing its sidecar and
# making it unscoreable. Derive from cohort x window and allow for warmup/gaps.
if smoke == "0":
    ids = json.load(open("data/m1_cohort.json"))["cohort"]
    days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days + 1
    leads = 14 if al == "1" else 1
    floor = int(0.40 * len(ids) * days * leads)
    assert n >= floor, f"{n} rows < derived floor {floor} ({len(ids)} basins x {days} d x {leads})"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
json.dump({"member": member, "seed": int(seed), "frame": tag, "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": trs, "train_end": tre, "protocol": "modern-v3",
           "all_leads": al == "1", "bytes": os.path.getsize(out), "ledger": "MODERN-1"},
          open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"DUMP OK {tag}: {n} rows -> {out}")
PYS
}

dump val1      "$VS" "$VE" 1 0   # selection frame, every day, lead 1
dump test1     "$TS" "$TE" 1 0   # THE HEADLINE FRAME - read once, at the end
dump val14     "$VS" "$VE" 1 1   # all-leads table (selection side)
dump test14    "$TS" "$TE" 1 1   # all-leads table (reported with the headline)
echo "=== M1 $M s$S ALL DONE $(date +%F_%T) ==="
