#!/usr/bin/env bash
# LEDGER 53 — dump the stride-1 VAL window (1980-10-01..1989-09-30, the honest
# selection frame) for an existing LEDGER-51 MB-LSTM checkpoint. Mirrors the
# dump() step of queue_l53_withq.sh exactly (same flags, same sidecar), so the
# val1 dumps of every member are built the same way as fused3h1's.
# usage: dump_l53_val1.sh <member> <seed>      member = daymeth1|maurerh1|nldash1|aorch1|fused3h1
set -u
cd ~/riverwatch2
M="${1:?member}"; S="${2:?seed}"
exec 9>"/tmp/rw2_l53_val1_${M}_${S}.lock"; flock -n 9 || { echo "another dumper holds ${M}/${S}"; exit 0; }
PY=gpu1080/.venv/bin/python
CK=data/mblstm/gpu_ckpts; DD=data/mblstm/l51_dumps
LOG=logs/l53_val1_${M}_s${S}.log
case "$M" in
  daymeth1|maurerh1|nldash1|aorch1) CORP="gpu1080/corpora671/camels_corpus_${M%h1}_v2_531"; ENC="camels1f" ;;
  fused3h1) CORP="gpu1080/corpora671/camels_corpus_fused_v2_531"; ENC="camels3fv2" ;;
  *) echo "unknown member $M"; exit 1 ;;
esac
CKPT="$CK/l51_${M}_s${S}.pt"
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint $CKPT" | tee -a "$LOG"; exit 1; }
TAG=val1; VS=1980-10-01; VE=1989-09-30; ST=1
OUT="$DD/camels531_l51_${M}_s${S}_${TAG}.csv.gz"
[ -f "$OUT" ] && { echo "[$(date +%T)] $TAG dump exists for $M s$S" | tee -a "$LOG"; exit 0; }
# GUARD (same as the launcher): the checkpoint's own cfg must carry the Nearing split.
$PY - "$CKPT" <<'PYG'
import sys, torch
cfg = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["cfg"]
assert cfg["train_start"] == "1999-10-01" and cfg["train_end"] == "2008-09-30", (cfg["train_start"], cfg["train_end"])
assert cfg["val_range"] == ["1980-10-01", "1989-09-30"], cfg["val_range"]
assert cfg["no_q_input"] is False and cfg["n_stations"] == 531
print("GUARD OK", cfg["train_start"], cfg["train_end"])
PYG
[ $? -eq 0 ] || { echo "ABORT: guard failed for $CKPT" | tee -a "$LOG"; exit 1; }
echo "=== DUMP $TAG $M s$S ($VS..$VE stride $ST) $(date +%F_%T) ===" | tee -a "$LOG"
nice -n 10 $PY scripts/train_mblstm.py --epochs 0 --init-ckpt "$CKPT" --out "$CKPT.dumpsink_val1" \
  --head quantile --enc-vars "$ENC" --q-transform linear --static-set camels --hidden 256 --corpus-dir "$CORP" \
  --val-start "$VS" --val-end "$VE" --val-stride "$ST" --dump-day1 "$OUT" >> "logs/l51_dump_${M}_s${S}_${TAG}.log" 2>&1
rm -f "$CKPT.dumpsink_val1"
gzip -t "$OUT" || { echo "ABORT: $TAG dump failed gzip -t" | tee -a "$LOG"; exit 1; }
$PY - "$OUT" "$CKPT" "$TAG" "$M" "$S" <<'PYS'
import gzip, hashlib, json, sys, os
out, ck, tag, member, seed = sys.argv[1:6]
n = sum(1 for _ in gzip.open(out, "rt")) - 1
assert n > 1000, f"only {n} rows"
hdr = gzip.open(out, "rt").readline().strip()
assert hdr == "station_id,t0,h,truth,ylo,ymed,yhi", hdr
json.dump({"member": member, "seed": int(seed), "frame": tag, "rows": n, "ckpt": ck,
           "ckpt_md5": hashlib.md5(open(ck, "rb").read()).hexdigest(),
           "train_start": "1999-10-01", "train_end": "2008-09-30", "protocol": "nearing2022",
           "bytes": os.path.getsize(out)}, open(out.replace(".csv.gz", ".json"), "w"), indent=1)
print(f"DUMP OK {tag}: {n} rows -> {out}")
PYS
echo "=== $M s$S $TAG done $(date +%F_%T) ===" | tee -a "$LOG"
