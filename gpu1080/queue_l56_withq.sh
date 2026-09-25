#!/usr/bin/env bash
# LEDGER 56 — the fused MB-LSTM member with an ALL-PRODUCT decoder.
#
# The fused member (fused3h1) was trained with a _daymet-only decoder (6 vars)
# because forecast archives are single-source. The CAMELS benchmark is perfect-
# forcing, so that constraint does not apply: at lead 1 the member never saw
# NLDAS/Maurer for the day it is scored on. fused3h1d3 = same corpus, same
# recipe, decoder fed all 18 fused variables (scripts/train_mblstm.py
# --enc-vars camels3fv2d3). fused4h1d4 = the AORC-inclusive 24-var version.
#
# A NEW FILE, not an edit of queue_l53_withq.sh ([[never-edit-a-running-shell-script]]).
# Protocol hard-coded here and READ BACK from the checkpoint before any dump.
# Dumps: val1 (selection frame) and test1 (headline), both stride 1.
#
# usage: queue_l56_withq.sh <member> <seed>
#   member = fused3h1d3 | fused4h1d4 | fused3h1d3smoke
set -u
cd ~/riverwatch2
M="${1:?member}"; S="${2:?seed}"
exec 9>"/tmp/rw2_l56_${M}_${S}.lock"
flock -n 9 || { echo "another launcher holds ${M}/${S}; exiting"; exit 0; }

PY=gpu1080/.venv/bin/python
CK=data/mblstm/gpu_ckpts
DD=data/mblstm/l51_dumps
LOG=logs/l56_${M}_s${S}.log
mkdir -p logs "$CK" "$DD" benchmarks

EPOCHS=30; SMOKE=0; LIMIT=""
case "$M" in
  fused3h1d3) CORP="gpu1080/corpora671/camels_corpus_fused_v2_531";  ENC="camels3fv2d3"; NDEC=18; EXTRA="--h1-weight 0.5" ;;
  fused4h1d4) CORP="gpu1080/corpora671/camels_corpus_fused4_v2_531"; ENC="camels4fv2d4"; NDEC=24; EXTRA="--h1-weight 0.5" ;;
  fused3h1d3smoke) CORP="gpu1080/corpora671/camels_corpus_fused_v2_531"; ENC="camels3fv2d3"; NDEC=18; EXTRA="--h1-weight 0.5"
                   EPOCHS=1; SMOKE=1; LIMIT="--limit-stations 8" ;;
  *) echo "unknown member $M"; exit 1 ;;
esac
CKPT="$CK/l56_${M}_s${S}.pt"

# GUARD 0: the corpus must be exactly the 531 benchmark basins.
N=$($PY -c "
import json,glob,os
ids=set(str(x).zfill(8) for x in json.load(open('data/camels_gauge_ids.json'))['531'])
have=set(os.path.basename(f)[:8] for f in glob.glob('$CORP/*.csv.gz'))
print(f'{len(ids & have)}/{len(have)}')")
echo "[$(date +%F_%T)] $M s$S corpus $CORP -> $N benchmark/total (enc=$ENC ndec=$NDEC epochs=$EPOCHS smoke=$SMOKE)" | tee -a "$LOG"
[ "$N" = "531/531" ] || { echo "ABORT: corpus is not the 531 set" | tee -a "$LOG"; exit 1; }

# ---- train -----------------------------------------------------------------
if [ ! -f "$CKPT" ]; then
  echo "=== TRAIN $M s$S $(date +%F_%T) ===" | tee -a "$LOG"
  echo "ARGV: --epochs $EPOCHS --seed $S --head quantile --enc-vars $ENC --q-transform linear --static-set camels --hidden 256 --corpus-dir $CORP --train-start 1999-10-01 --train-end 2008-09-30 --val-start 1980-10-01 --val-end 1989-09-30 $EXTRA $LIMIT" | tee -a "$LOG"
  # shellcheck disable=SC2086
  nice -n 5 $PY scripts/train_mblstm.py \
    --epochs "$EPOCHS" --seed "$S" --out "$CKPT" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --train-start 1999-10-01 --train-end 2008-09-30 \
    --val-start 1980-10-01 --val-end 1989-09-30 \
    $EXTRA $LIMIT > "logs/l56_train_${M}_s${S}.log" 2>&1
  RC=$?
  echo "=== trained rc=$RC $(date +%T) ===" | tee -a "$LOG"
fi
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint" | tee -a "$LOG"; exit 1; }

# GUARD 1: protocol AND the decoder width, read back from the checkpoint.
$PY - "$CKPT" "logs/l56_train_${M}_s${S}.log" "$NDEC" "$SMOKE" "$EPOCHS" <<'PYG'
import re, sys, torch
ck, log, ndec, smoke, ep = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
cfg = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]
assert cfg["train_start"] == "1999-10-01", f"train_start={cfg['train_start']!r}"
assert cfg["train_end"] == "2008-09-30", f"train_end={cfg['train_end']!r}"
assert cfg["val_range"] == ["1980-10-01", "1989-09-30"], cfg["val_range"]
assert cfg["no_q_input"] is False, "with-q member must see discharge"
assert abs(float(cfg["h1_weight"]) - 0.5) < 1e-9, cfg["h1_weight"]
assert int(cfg["hidden"]) == 256 and cfg["q_transform"] == "linear", (cfg["hidden"], cfg["q_transform"])
# THE ARM: the decoder must see every product, i.e. decoder == encoder.
assert len(cfg["dec_vars"]) == ndec, (len(cfg["dec_vars"]), ndec)
assert cfg["dec_vars"] == cfg["enc_vars"], "decoder is not the full encoder set"
prods = {v.rsplit("_", 1)[1] for v in cfg["dec_vars"]}
assert len(prods) == ndec // 6, f"decoder products {prods}"
# the decoder LSTM's input width must reflect it (weights, not just cfg)
sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
w = sd["decoder.weight_ih_l0"].shape[1]
want = ndec + 2 + 1 + len(cfg["static_feats"])
assert w == want, f"decoder.weight_ih_l0 width {w} != {want}"
txt = open(log).read()
m = re.search(r"windows: train=(\d+) val=(\d+)", txt)
assert m, "no window line in log"
tr = int(m.group(1))
assert tr < 3_000_000, f"train windows {tr} — the guard did not bite"
m2 = re.search(r"enc_vars=(\d+) dec_vars=(\d+)", txt)
assert m2 and int(m2.group(2)) == ndec, f"log dec_vars {m2 and m2.group(2)} != {ndec}"
if not smoke:
    assert cfg["n_stations"] == 531, cfg["n_stations"]
    assert "usable stations: 531" in txt, "not 531 stations"
# the checkpoint is saved at every improving epoch, so a run interrupted mid-way
# leaves a partial ckpt that the "skip if exists" logic would happily dump.
# Refuse unless the log shows the final epoch completed.
assert re.search(rf"^epoch {ep}/{ep} ", txt, re.M), f"final epoch {ep} not in log — partial checkpoint, refusing"
print(f"GUARD OK: train_windows={tr} val_windows={m.group(2)} stations={cfg['n_stations']} "
      f"train={cfg['train_start']}..{cfg['train_end']} enc={len(cfg['enc_vars'])} dec={len(cfg['dec_vars'])} "
      f"products={sorted(prods)} dec_in={w}")
PYG
[ $? -eq 0 ] || { echo "ABORT: guard failed" | tee -a "$LOG"; exit 1; }

# ---- dumps -----------------------------------------------------------------
dump () {  # <tag> <vstart> <vend> <stride>
  local TAG="$1" VS="$2" VE="$3" ST="$4"
  local OUT="$DD/camels531_l51_${M}_s${S}_${TAG}.csv.gz"
  [ -f "$OUT" ] && { echo "[$(date +%T)] $TAG dump exists" | tee -a "$LOG"; return 0; }
  echo "=== DUMP $TAG ($VS..$VE stride $ST) $(date +%F_%T) ===" | tee -a "$LOG"
  # shellcheck disable=SC2086
  nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" \
    --out "$CKPT.dumpsink_$TAG" \
    --head quantile --enc-vars "$ENC" --q-transform linear \
    --static-set camels --hidden 256 --corpus-dir "$CORP" \
    --val-start "$VS" --val-end "$VE" --val-stride "$ST" \
    --dump-day1 "$OUT" $LIMIT >> "logs/l56_dump_${M}_s${S}_${TAG}.log" 2>&1
  rm -f "$CKPT.dumpsink_$TAG"
  gzip -t "$OUT" || { echo "ABORT: $TAG dump failed gzip -t" | tee -a "$LOG"; return 1; }
  $PY - "$OUT" "$CKPT" "$TAG" "$M" "$S" "$SMOKE" <<'PYS'
import gzip, hashlib, json, sys, os
out, ck, tag, member, seed, smoke = sys.argv[1:7]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
assert n > 1000, f"only {n} rows — the dump path emitted nothing"
if smoke == "0":
    assert n > 1_500_000, f"only {n} rows for a stride-1 531-basin frame"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
side = out.replace(".csv.gz", ".json")
json.dump({"member": member, "seed": int(seed), "frame": tag, "rows": n,
           "ckpt": ck, "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": "1999-10-01", "train_end": "2008-09-30",
           "protocol": "nearing2022", "bytes": os.path.getsize(out),
           "decoder": "all-product (LEDGER 56)", "ledger": 56},
          open(side, "w"), indent=1)
print(f"DUMP OK {tag}: {n} rows -> {out}")
PYS
}

dump val1  1980-10-01 1989-09-30 1    # the selection frame, every day
dump test1 1989-10-01 1999-09-30 1    # THE HEADLINE FRAME: every day, day-1
echo "=== L56 $M s$S ALL DONE $(date +%F_%T) ===" | tee -a "$LOG"
