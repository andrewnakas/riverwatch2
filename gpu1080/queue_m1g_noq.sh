#!/usr/bin/env bash
# MODERN-1 PHASE 5b — the 4-PRODUCT no-q members. usage: <member> <seed>
#
# A SEPARATE FILE, not an edit to queue_m1_noq.sh, because that script was RUNNING the
# no-q Gate B when these members were needed. bash reads a script by BYTE OFFSET, so a
# mid-run rewrite has already truncated a training log to 0 bytes in this campaign. The
# standing rule is: create queue_m1b_*.sh instead.
#
# Everything else -- the arm guard, the dump sidecars, the row-count floor -- is inherited
# verbatim from queue_m1_noq.sh, including GUARD 1's assertion that the checkpoint really
# declares no_q_input=True and dec_q=False.
# MODERN-1 PHASE 5 — train one NO-Q member on the modern corpus. usage: <member> <seed>
#
# The no-q arm is the second half of the request ("both with q and no q"). It is the
# CAMELS-protocol-comparable setting: pure rainfall-runoff, the encoder never sees
# discharge. --no-q-input ZEROES the q and mask channels rather than removing them
# (train_mblstm.py:389-396), so every tensor width is identical to the with-q arm and
# GUARD 1's decoder-width formula carries over unchanged. The distinguishing check is
# cfg["no_q_input"] is True, asserted below.
#
# ⛔ --dec-q is FORBIDDEN in this arm and asserted absent. It exists to put the
# assimilated discharge in the decoder; combining it with a "no-q" label would be a
# leak wearing the name of its own control. The guard refuses it rather than trusting
# the member table.
#
# Protocol P-MODERN, 9-year training window, matching the with-q members so the two
# arms sit on one frame. ⚠️ Gate B (34 vs 9 years, +0.001370 for ~3.75x compute) was
# measured on the WITH-Q arm only. A no-q model lives entirely on its forcings, so
# that result may not transfer -- `lstm_gridmetm34` exists to measure it rather than
# assume it, and is registered as the no-q Gate B.
set -u
cd ~/riverwatch2 || exit 1
M="${1:?member}"; S="${2:?seed}"
PY=gpu1080/.venv/bin/python
DD=data/mblstm/l51_dumps
V3=data/modern_corpus/v3
LOG="logs/m1_${M}_s${S}.log"
CKPT="data/mblstm/gpu_ckpts/m1_${M}_s${S}.pt"
TRS=2005-10-01; TRE=2014-09-30
VS=2014-10-01;  VE=2018-09-30
TS=2018-10-01;  TE=2025-09-30
EXTRA=""
SMOKE=0
case "$M" in *smoke*) SMOKE=1 ;; esac
LIMIT=""; [ "$SMOKE" = "1" ] && LIMIT="--limit-stations 8"

case "$M" in
  lstm_daymetm|lstm_daymetmsmoke)   CORP="$V3/camels_corpus_daymet_m3_cohort";    ENC="camels1f" ;;
  lstm_gridmetm)                    CORP="$V3/camels_corpus_gridmet_m3_cohort";   ENC="camels1f" ;;
  lstm_nclimgridm)                  CORP="$V3/camels_corpus_nclimgrid_m3_cohort"; ENC="camels1f" ;;
  lstm_nldasm)                      CORP="$V3/camels_corpus_nldas_m3_cohort";     ENC="camels1f" ;;
  # the no-q GATE B: identical recipe, 34-year window. Measured, not assumed.
  lstm_gridmetm34)                  CORP="$V3/camels_corpus_gridmet_m3_cohort";   ENC="camels1f"; TRS=1980-10-01 ;;
  # multi-forcing: the strongest no-q class in the frozen record (multi/multi5/multi6
  # took 62% of its fitted weight).
  lstm_fused3m|lstm_fused3msmoke)   CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm3v2" ;;
  # ⭐ the all-product DECODER. Worth testing in this arm specifically: the LEDGER 56
  # fix is about TARGET-DAY FORCING reaching the decoder, which has nothing to do with
  # discharge -- so it should transfer to no-q, where forcings are the only input.
  lstm_fused3md3)                   CORP="$V3/camels_corpus_fused3m_m3_cohort";   ENC="camelsm3v2d3" ;;
  # ---- the 4-PRODUCT no-q members -------------------------------------------
  # ⭐ lstm_fused4md4 is the no-q arm's expected champion: the decoder count effect is
  # LARGER without discharge (+0.020291 vs +0.015528 with-q), because forcing is then the
  # only source of target-day information. Raising the decoder's product count 3 -> 4
  # should therefore pay at least as well here as in the with-q arm.
  lstm_fused4m)                     CORP="$V3/camels_corpus_fused4m_m3_cohort"; ENC="camelsm4v2" ;;
  lstm_fused4md4fix3L1)             CORP="$V3/camels_corpus_fused4mfix3_m3_cohort"; ENC="camelsm4v2d4"; EXTRA="--enc-lead 1" ;;
  lstm_fused4md4fix3)               CORP="$V3/camels_corpus_fused4mfix3_m3_cohort"; ENC="camelsm4v2d4" ;;
  lstm_fused4md4)                   CORP="$V3/camels_corpus_fused4m_m3_cohort"; ENC="camelsm4v2d4" ;;
  *) echo "unknown no-q member $M"; exit 2 ;;
esac

DEV=""
if [ "$SMOKE" = "1" ]; then
  CORP="${M1_SMOKE_CORPUS:-data/modern_corpus/SMOKE_fused4m_cohort}"
  DEV="--device ${M1_SMOKE_DEVICE:-cpu}"
fi

exec >>"$LOG" 2>&1
echo "=== M1-NOQ $M s$S $(date +%F_%T) ==="

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
    --h1-weight 0.5 --no-q-input $EXTRA $LIMIT $DEV
  echo "=== trained rc=$? $(date +%T) ==="
fi

# ---- GUARD 1: read the arm back out of the CHECKPOINT, never argv ----------
M1_EXTRA="$EXTRA" $PY - "$CKPT" "$LOG" "$TRS" "$TRE" "$VS" "$VE" <<'PYG'
import os, re, sys, torch
ck, log, trs, tre, vs, ve = sys.argv[1:7]
sd = torch.load(ck, map_location="cpu")
cfg = sd["cfg"]
assert cfg["train_start"] == trs, f"train_start {cfg['train_start']} != {trs}"
assert cfg["train_end"] == tre, f"train_end {cfg['train_end']} != {tre}"
assert cfg["val_range"] == [vs, ve], f"val {cfg['val_range']} != {[vs, ve]}"
# ⭐ THE assertion this arm exists for. A with-q checkpoint dumped under a no-q
# member name would be a silent 0.06-NSE gift, and the member name is not evidence.
assert cfg["no_q_input"] is True, "NOT a no-q checkpoint -- cfg says no_q_input=False"
# ⛔ and the forbidden flag, refused rather than trusted
assert bool(cfg.get("dec_q", False)) is False, \
    "dec_q=True in a NO-Q member: the decoder would receive discharge"
assert abs(cfg.get("h1_weight", -1) - 0.5) < 1e-9, cfg.get("h1_weight")
ndec = len(cfg["dec_vars"])
w = sd["state_dict"]["decoder.weight_ih_l0"].shape[1]
exp = ndec + 2 + 1 + len(cfg["static_feats"])
assert w == exp, f"decoder width {w} != {exp} (dec={ndec})"
# --enc-lead changes no tensor SHAPE, so the width check above cannot see it. Assert it in BOTH
# directions, or a dropped flag trains a duplicate and its null gets read as a measurement.
_m = re.search(r"--enc-lead\s+(\d+)", os.environ.get("M1_EXTRA", ""))
_want = int(_m.group(1)) if _m else 0
assert int(cfg.get("enc_lead", 0)) == _want, \
    f"enc_lead {cfg.get('enc_lead')} != requested {_want}"
txt = open(log, encoding="utf-8", errors="replace").read()
assert re.search(r"^epoch 30/30 ", txt, re.M), "no 'epoch 30/30' line - partial checkpoint"
if _want:
    assert re.search(rf"^enc_lead={_want} ", txt, re.M), "no enc_lead line in the training log"
print(f"GUARD 1 OK: NO-Q confirmed, stations={cfg['n_stations']} "
      f"train={cfg['train_start']}..{cfg['train_end']} enc={len(cfg['enc_vars'])} dec={ndec}")
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
    --dump-day1 "$OUT" $FLAG --no-q-input $LIMIT $DEV \
    >> "logs/m1_dump_${M}_s${S}_${TAG}.log" 2>&1
  rm -f "$CKPT.dumpsink_$TAG"
  gzip -t "$OUT" || { echo "ABORT: $TAG dump failed gzip -t"; return 1; }
  $PY - "$OUT" "$CKPT" "$TAG" "$M" "$S" "$SMOKE" "$A" "$B" "$AL" "$TRS" "$TRE" <<'PYS'
import datetime as dt, gzip, hashlib, json, os, sys
out, ck, tag, member, seed, smoke, a, b, al, trs, tre = sys.argv[1:12]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
assert n > 1000, f"only {n} rows - the dump path emitted nothing"
if smoke == "0":
    ids = json.load(open("data/m1_cohort.json"))["cohort"]
    days = (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days + 1
    leads = 14 if al == "1" else 1
    floor = int(0.40 * len(ids) * days * leads)
    assert n >= floor, f"{n} rows < derived floor {floor}"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
# ⭐ "arm" is recorded so a no-q dump can never be silently joined to the with-q
# roster: both arms share a protocol and a frame, and only this field separates them.
json.dump({"member": member, "seed": int(seed), "frame": tag, "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": trs, "train_end": tre,
           "protocol": "modern-v3" if trs == "1980-10-01" else "modern-v3-9yr",
           "arm": "noq", "no_q_input": True,
           "all_leads": al == "1", "bytes": os.path.getsize(out), "ledger": "MODERN-1"},
          open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"DUMP OK {tag}: {n} rows -> {out}")
PYS
}

dump val1      "$VS" "$VE" 1 0
dump test1     "$TS" "$TE" 1 0
# all-leads dumps are OPT-IN -- see the measurement in queue_m1_withq.sh: a 14-lead
# dump is ~22M rows, >12 h at 5-6.5 GiB RSS and only ~34% GPU. A screen does not need
# one; only members reaching the final composition do, and they can be re-dumped on
# demand because the checkpoint is kept and `dump` skips existing frames.
if [ "${M1_ALLLEADS:-0}" = "1" ]; then
  dump val14   "$VS" "$VE" 1 1
  dump test14  "$TS" "$TE" 1 1
else
  echo "[$(date +%T)] all-leads dumps SKIPPED (M1_ALLLEADS!=1); re-dump with M1_ALLLEADS=1"
fi
echo "=== M1-NOQ $M s$S ALL DONE $(date +%F_%T) ==="
